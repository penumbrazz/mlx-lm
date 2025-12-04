# ==============================================================================
# MLX LM HTTP 服务器 - 面向小白的详细中文注释版本
# ==============================================================================
# 
# 本文件实现了一个基于 MLX 框架的大型语言模型 HTTP 服务器
# 提供与 OpenAI API 兼容的接口，支持文本生成和聊天对话功能
# 
# 主要功能：
# 1. 动态加载和管理多个语言模型
# 2. 支持 OpenAI 兼容的 API 接口（completions 和 chat/completions）
# 3. 实现高效的缓存机制（LRU Cache）
# 4. 支持批处理和流式响应
# 5. 提供健康检查和模型列表接口
# 
# 架构设计：
# - ModelProvider: 负责模型的加载和管理
# - ResponseGenerator: 处理文本生成的核心逻辑
# - APIHandler: 处理 HTTP 请求和响应
# - LRUPromptCache: 实现高效的提示缓存
# 
# Copyright © 2023-2024 Apple Inc.
# ==============================================================================

import argparse  # 命令行参数解析，用于处理启动时的各种配置选项
import copy      # 深拷贝对象，用于缓存管理时的数据复制
import json      # JSON 数据处理，用于 API 请求和响应的序列化
import logging   # 日志记录，用于调试和监控服务器运行状态
import platform  # 获取系统平台信息，用于生成系统指纹
import socket    # 网络套接字，用于 HTTP 服务器的网络通信
import time      # 时间处理，用于生成时间戳和超时控制
import uuid      # 生成唯一标识符，用于请求 ID 的生成
import warnings  # 警告信息输出，用于提示用户潜在问题

from collections import deque  # 双端队列，用于 LRU 缓存的实现
from dataclasses import dataclass, field  # 数据类装饰器，简化数据结构定义
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # HTTP 服务器基础类
from pathlib import Path  # 路径处理，用于文件路径操作
from queue import Empty as QueueEmpty  # 队列空异常，用于队列操作
from queue import Queue  # 队列，用于请求的异步处理
from threading import Condition, Lock, Thread  # 线程相关，用于并发处理

from typing import (  # 类型注解，提供更好的代码可读性和类型检查
    Any,       # 任意类型
    Callable,  # 可调用对象
    Dict,      # 字典类型
    List,      # 列表类型
    Literal,   # 字面量类型
    NamedTuple, # 命名元组
    Optional,  # 可选类型
    Sequence,  # 序列类型
    Tuple,     # 元组类型
    Union,     # 联合类型
)

import mlx.core as mx  # MLX 核心库，用于张量计算和模型推理
from huggingface_hub import scan_cache_dir  # 扫描 Hugging Face 缓存目录，获取已下载模型信息

from ._version import __version__  # 版本信息，用于系统指纹生成
from .generate import BatchGenerator, stream_generate  # 批量生成和流式生成函数
from .models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache  # 缓存相关函数
from .sample_utils import make_logits_processors, make_sampler  # 采样相关工具函数
from .utils import load  # 模型加载工具函数


# ==============================================================================
# 系统工具函数
# ==============================================================================

def get_system_fingerprint():
    """
    生成系统指纹，用于标识当前系统的特征
    
    这个函数会收集以下信息来生成唯一的系统标识：
    1. MLX LM 的版本号
    2. MLX 核心库的版本号  
    3. 操作系统平台信息
    4. GPU 架构信息（如果使用 Metal）
    
    Returns:
        str: 系统指纹字符串，格式为 "版本-MLX版本-平台-GPU架构"
    """
    # 获取 GPU 架构信息，如果 Metal 可用的话
    gpu_arch = mx.metal.device_info()["architecture"] if mx.metal.is_available() else ""
    
    # 组合所有信息生成指纹
    return f"{__version__}-{mx.__version__}-{platform.platform()}-{gpu_arch}"


# ==============================================================================
# 数据结构定义
# ==============================================================================

class StopCondition(NamedTuple):
    """
    停止条件的数据结构
    
    这个命名元组用于表示生成过程中是否应该停止的条件
    """
    stop_met: bool          # 是否满足停止条件
    trim_length: int        # 需要修剪的 token 数量
    trim_text_length: int   # 需要修剪的文本字符数量


def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]], 
    stop_words: List[str],
    eos_token_id: Union[int, None],
) -> StopCondition:
    """
    判断是否应该停止 token 生成的条件检查函数
    
    这个函数会检查多种停止条件：
    1. 是否生成了结束符（EOS token）
    2. 是否匹配了指定的停止词序列
    
    Args:
        tokens (List[int]): 当前已生成的 token 序列
        stop_id_sequences (List[List[int]]): 停止词的 token ID 序列列表
        stop_words (List[str]): 停止词的文本列表（与 stop_id_sequences 对应）
        eos_token_id (Union[int, None]): 结束符的 token ID
        
    Returns:
        StopCondition: 包含停止判断结果和需要修剪的长度信息
        
    Example:
        >>> tokens = [1, 2, 3, 4]
        >>> stop_id_sequences = [[3, 4]]
        >>> stop_words = ["stop"]
        >>> eos_token_id = 0
        >>> result = stopping_criteria(tokens, stop_id_sequences, stop_words, eos_token_id)
        >>> result.stop_met
        True
    """
    # 检查是否生成了结束符
    if tokens and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=0, trim_text_length=0)

    # 检查是否匹配了任何停止词序列
    for stop_ids, stop_word in zip(stop_id_sequences, stop_words):
        if len(tokens) >= len(stop_ids):
            # 检查当前 token 序列的末尾是否匹配停止词
            if tokens[-len(stop_ids):] == stop_ids:
                return StopCondition(
                    stop_met=True,
                    trim_length=len(stop_ids),          # 需要修剪的 token 数量
                    trim_text_length=len(stop_word),    # 需要修剪的文本长度
                )

    # 没有满足停止条件
    return StopCondition(stop_met=False, trim_length=0, trim_text_length=0)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    检查两个序列是否有重叠部分
    
    具体来说，检查序列 s1 的后缀是否与序列 s2 的前缀有重叠
    这个函数主要用于处理停止词的边界情况
    
    Args:
        s1 (Sequence): 第一个序列（通常是已生成的 tokens）
        s2 (Sequence): 第二个序列（通常是停止词 tokens）
        
    Returns:
        bool: 如果两个序列有重叠则返回 True，否则返回 False
        
    Example:
        >>> sequence_overlap([1, 2, 3], [3, 4, 5])
        True  # 3 是重叠部分
        >>> sequence_overlap([1, 2], [3, 4])
        False
    """
    # 计算最大可能的重叠长度
    max_overlap = min(len(s1), len(s2))
    
    # 检查从 1 到最大重叠长度的所有可能重叠
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    """
    将聊天消息列表转换为单一提示文本
    
    当分词器没有聊天模板时，使用这个函数将多轮对话转换为单一字符串
    每条消息都会添加相应的前缀和分隔符
    
    Args:
        messages (List[dict]): 聊天消息列表，每条消息包含 role 和 content
        role_mapping (Optional[dict]): 角色映射字典，定义不同角色的前缀和分隔符
        
    Returns:
        str: 转换后的提示文本
        
    Example:
        >>> messages = [
        ...     {"role": "user", "content": "你好"},
        ...     {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
        ... ]
        >>> prompt = convert_chat(messages)
        >>> print(prompt)
        USER: 你好
        ASSISTANT: 你好！有什么可以帮助你的吗？ASSISTANT: 
    """
    # 默认的角色映射配置
    default_role_mapping = {
        "system_prompt": (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant follows the given rules no matter what."
        ),
        "system": "ASSISTANT's RULE: ",    # 系统消息前缀
        "user": "USER: ",                 # 用户消息前缀
        "assistant": "ASSISTANT: ",       # 助手消息前缀
        "stop": "\n",                     # 消息分隔符
    }
    
    # 使用提供的映射或默认映射
    role_mapping = role_mapping if role_mapping is not None else default_role_mapping

    prompt = ""
    
    # 遍历所有消息，逐个添加到提示中
    for line in messages:
        role_prefix = role_mapping.get(line["role"], "")  # 获取角色前缀
        stop = role_mapping.get("stop", "")              # 获取分隔符
        content = line.get("content", "")                # 获取消息内容
        prompt += f"{role_prefix}{content}{stop}"

    # 添加助手回复的前缀，准备生成新的回复
    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()


def process_message_content(messages):
    """
    处理消息内容，转换为适合 apply_chat_template 的格式
    
    这个函数会处理消息中的复杂内容格式，特别是当 content 是列表时
    会提取其中的文本部分并合并为单一字符串
    
    Args:
        messages (list): 消息列表，每个消息可能包含复杂的 content 结构
        
    Raises:
        ValueError: 当遇到不支持的内容类型或缺少必要字段时抛出
        
    Example:
        >>> messages = [
        ...     {"role": "user", "content": [{"type": "text", "text": "你好"}]}
        ... ]
        >>> process_message_content(messages)
        >>> messages
        [{"role": "user", "content": "你好"}]
    """
    for message in messages:
        content = message["content"]
        
        # 如果 content 是列表格式，提取文本片段
        if isinstance(content, list):
            text_fragments = [
                fragment["text"] 
                for fragment in content 
                if fragment["type"] == "text"
            ]
            
            # 检查是否所有片段都是文本类型
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
                
            # 将文本片段合并为单一字符串
            message["content"] = "".join(text_fragments)
            
        # 如果 content 为 None，转换为空字符串
        elif content is None:
            message["content"] = ""


# ==============================================================================
# LRU 缓存系统
# ==============================================================================

class LRUPromptCache:
    """
    最近最少使用（LRU）提示缓存系统
    
    这个类实现了高效的缓存机制，用于存储和检索之前处理过的提示
    通过缓存可以显著提高重复或相似提示的处理速度
    
    主要特性：
    1. 基于树形结构的缓存索引，支持部分匹配
    2. LRU 淘汰策略，自动管理缓存大小
    3. 支持精确匹配、前缀匹配和后缀匹配
    4. 支持缓存的修剪和扩展
    """

    @dataclass
    class CacheEntry:
        """缓存条目数据结构"""
        prompt_cache: List[Any]  # 实际的缓存内容（通常是 KV 缓存）
        count: int              # 引用计数，用于 LRU 管理

    @dataclass
    class SearchResult:
        """缓存搜索结果"""
        model: Any              # 模型标识
        exact: List[int]        # 精确匹配的 token 序列
        shorter: List[int]      # 较短的匹配序列（前缀匹配）
        longer: List[int]       # 较长的匹配序列（可修剪）
        common_prefix: int      # 公共前缀长度

    def __init__(self, max_size: int = 10):
        """
        初始化 LRU 缓存
        
        Args:
            max_size (int): 缓存的最大条目数量，默认为 10
        """
        self.max_size = max_size
        self._cache = {}      # 主缓存存储，按模型组织
        self._lru = deque()   # LRU 队列，记录访问顺序

    def _search(self, model, tokens):
        """
        在缓存中搜索最匹配的条目
        
        这个方法会执行智能匹配，包括：
        1. 精确匹配：完全匹配的 token 序列
        2. 前缀匹配：当前请求是缓存条目的前缀
        3. 后缀匹配：缓存条目是当前请求的前缀
        4. 可修剪匹配：较长的缓存条目可以修剪为合适的长度
        
        Args:
            model: 模型标识
            tokens (List[int]): 要搜索的 token 序列
            
        Returns:
            SearchResult: 搜索结果，包含各种匹配情况
        """
        # 如果模型不在缓存中，返回空结果
        if model not in self._cache:
            return self.SearchResult(model, None, None, None, 0)

        current = self._cache[model]
        last_cache_index = -1  # 最后匹配的缓存位置
        index = 0              # 当前匹配位置

        # 沿着 token 序列在缓存树中遍历
        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "cache" in current:
                last_cache_index = index  # 记录找到缓存的位置
            index += 1

        # 如果找到精确匹配，直接返回
        if last_cache_index == len(tokens) - 1:
            return self.SearchResult(model, tokens, None, None, 0)

        # 寻找较短的缓存（前缀匹配）
        shorter = None
        if last_cache_index > 0:
            shorter = tokens[:last_cache_index + 1]

        # 寻找较长的缓存（可能需要修剪）
        longer = None
        common_prefix = index
        if index > 0 and last_cache_index <= 0:
            best = None
            stack = [(current, [])]  # 使用栈进行深度优先搜索
            
            # 在缓存树中搜索可以修剪的长缓存
            while stack:
                current, extra = stack.pop()
                if "cache" in current:
                    if best is None or len(extra) < len(best):
                        best = extra
                else:
                    for tok in current:
                        stack.append((current[tok], extra + [tok]))
            
            longer = tokens[:index] + best
            
        return self.SearchResult(model, None, shorter, longer, common_prefix)

    def _get(self, model, tokens):
        """
        从缓存中获取指定位置的缓存条目
        
        Args:
            model: 模型标识
            tokens (List[int]): token 序列路径
            
        Returns:
            CacheEntry: 缓存条目
        """
        current = self._cache[model]
        for tok in tokens:
            current = current[tok]
        return current["cache"]

    def _delete(self, model, tokens):
        """
        从缓存中删除指定的缓存条目
        
        这个方法会同时清理空的中间节点
        
        Args:
            model: 模型标识
            tokens (List[int]): 要删除的 token 序列
        """
        # 构建从根到目标节点的路径
        path = [self._cache[model]]
        for tok in tokens:
            path.append(path[-1][tok])
            
        # 删除缓存条目
        del path[-1]["cache"]
        
        # 清理空的中间节点
        for i in reversed(range(len(tokens))):
            d_prev, d, t = path[i], path[i + 1], tokens[i]
            if len(d) > 0:
                break
            del d_prev[t]

    def _extract(self, model, tokens):
        """
        从缓存中提取并管理缓存条目
        
        这个方法会处理引用计数，当计数为 0 时删除缓存
        当计数大于 1 时，创建缓存条目的深拷贝
        
        Args:
            model: 模型标识
            tokens (List[int]): token 序列
            
        Returns:
            CacheEntry: 提取的缓存条目
        """
        cache_entry = self._get(model, tokens)
        
        # 如果引用计数为 1，删除缓存并从 LRU 队列中移除
        if cache_entry.count == 1:
            self._delete(model, tokens)
            self._lru.remove((model, tokens))
            return cache_entry

        # 减少引用计数并创建深拷贝
        cache_entry.count -= 1
        return self.CacheEntry(
            copy.deepcopy(cache_entry.prompt_cache),
            1,
        )

    def fetch_nearest_cache(self, model, tokens):
        """
        获取最匹配的缓存条目
        
        这是缓存系统的主要入口点，会：
        1. 搜索最佳匹配
        2. 处理各种匹配情况
        3. 返回可用的缓存和剩余需要处理的 token
        
        Args:
            model: 模型标识
            tokens (List[int]): 请求的 token 序列
            
        Returns:
            tuple: (缓存条目, 剩余 token 序列)，如果没有匹配则为 (None, tokens)
        """
        result = self._search(model, tokens)
        
        # 精确匹配：直接返回缓存
        if result.exact is not None:
            cache_entry = self._extract(result.model, result.exact)
            return cache_entry.prompt_cache, []

        # 前缀匹配：返回较短缓存和剩余 tokens
        if result.shorter is not None:
            cache_entry = self._extract(result.model, result.shorter)
            prefix_len = len(result.shorter)
            return cache_entry.prompt_cache, tokens[prefix_len:]

        # 可修剪匹配：修剪较长缓存
        if result.longer is not None:
            cache_entry = self._get(result.model, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache_entry = self.CacheEntry(
                    copy.deepcopy(cache_entry.prompt_cache),
                    1,
                )
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache_entry.prompt_cache, num_to_trim)
                return cache_entry.prompt_cache, tokens[prefix:]

        # 没有匹配的缓存
        return None, tokens

    def insert_cache(self, model, tokens, prompt_cache):
        """
        将新的缓存条目插入缓存
        
        Args:
            model: 模型标识
            tokens (List[int]): token 序列键
            prompt_cache: 要缓存的内容
        """
        # 如果模型不存在，创建新的模型缓存
        if model not in self._cache:
            self._cache[model] = {}
            
        # 构建缓存树路径
        current = self._cache[model]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        # 处理已存在的缓存条目
        if "cache" in current:
            current["cache"].count += 1
            self._lru.remove((model, tokens))
        else:
            current["cache"] = self.CacheEntry(prompt_cache, 1)

        # 添加到 LRU 队列末尾
        self._lru.append((model, tokens))
        
        # 如果缓存超过最大大小，删除最旧的条目
        if len(self._lru) > self.max_size:
            model, tokens = self._lru.popleft()
            self._delete(model, tokens)


# ==============================================================================
# 数据类定义 - 用于 API 参数传递
# ==============================================================================

@dataclass
class ModelDescription:
    """
    模型描述信息
    
    用于指定要使用的模型、草稿模型和适配器
    """
    model: str     # 主模型路径或名称
    draft: str     # 草稿模型路径或名称（用于推测解码）
    adapter: str   # 适配器路径或名称（用于微调模型）


@dataclass
class SamplingArguments:
    """
    采样参数配置
    
    控制文本生成时的随机性和多样性
    """
    temperature: float        # 温度参数，控制随机性（0-2，越高越随机）
    top_p: float             # 核采样参数（0-1，控制候选token范围）
    top_k: int              # Top-K采样参数（选择概率最高的 K 个token）
    min_p: float            # 最小概率阈值（0-1）
    xtc_probability: float  # XTC采样概率（0-1）
    xtc_threshold: float    # XTC采样阈值（0-0.5）


@dataclass
class LogitsProcessorArguments:
    """
    Logits处理参数
    
    用于调整模型输出的概率分布
    """
    logit_bias: Optional[Dict[int, float]]  # token偏差调整
    repetition_penalty: float                # 重复惩罚系数
    repetition_context_size: int             # 重复惩罚的上下文大小


@dataclass
class GenerationArguments:
    """
    生成参数的完整配置
    
    包含所有生成过程中需要的参数
    """
    model: ModelDescription              # 模型配置
    sampling: SamplingArguments         # 采样配置
    logits: LogitsProcessorArguments    # Logits处理配置
    stop_words: List[str]               # 停止词列表
    max_tokens: int                     # 最大生成token数
    num_draft_tokens: int               # 草稿token数量（推测解码用）
    logprobs: int                       # 返回概率信息token数量
    seed: Optional[int]                 # 随机种子


@dataclass
class CompletionRequest:
    """
    完成请求数据结构
    
    封装API请求中的核心信息
    """
    request_type: Literal["chat", "text"]  # 请求类型：聊天或文本补全
    prompt: str                             # 原始提示文本
    messages: List[Any]                     # 聊天消息列表（聊天模式用）
    tools: Optional[List[Any]]             # 工具列表（功能调用用）
    role_mapping: Optional[Dict[str, Any]]  # 角色映射配置


@dataclass
class GenerationContext:
    """
    生成上下文信息
    
    包含生成过程中需要的状态信息
    """
    has_tool_calling: bool              # 是否支持工具调用
    tool_call_start: str               # 工具调用开始标记
    tool_call_end: str                 # 工具调用结束标记
    eos_token_id: int                  # 结束符token ID
    stop_token_sequences: List[List[int]]  # 停止词token序列
    prompt: List[int]                  # 提示token序列
    _should_stop: bool = False         # 内部停止标志

    def stop(self):
        """设置停止标志"""
        self._should_stop = True


@dataclass
class Response:
    """
    生成响应数据结构
    
    包含单个token的完整信息
    """
    text: str                          # 生成的文本
    token: int                         # token ID
    logprob: float                     # 对数概率
    finish_reason: Optional[str]       # 完成原因
    top_tokens: Optional[Tuple[int, float]]  # 高概率token列表


# ==============================================================================
# 模型提供者
# ==============================================================================

class ModelProvider:
    """
    模型提供者类
    
    负责模型的动态加载、缓存和管理
    支持在运行时切换不同的模型、适配器和草稿模型
    
    主要功能：
    1. 按需加载模型，避免启动时的内存占用
    2. 模型缓存，避免重复加载
    3. 支持主模型、适配器和草稿模型的组合
    4. 分词器配置和聊天模板管理
    """

    def __init__(self, cli_args: argparse.Namespace):
        """
        初始化模型提供者
        
        Args:
            cli_args (argparse.Namespace): 命令行参数，包含默认配置
        """
        self.cli_args = cli_args
        self.model_key = None          # 当前模型的唯一标识
        self.model = None             # 当前加载的主模型
        self.tokenizer = None         # 当前加载的分词器
        self.draft_model = None       # 当前加载的草稿模型

        # 默认模型映射，用于处理特殊的"default_model"标识
        self.default_model_map = {}
        
        # 如果启动时指定了默认模型，预加载它
        if self.cli_args.model is not None:
            self.default_model_map[self.cli_args.model] = "default_model"
            self.load(self.cli_args.model, draft_model_path="default_model")

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        """
        加载指定的模型和分词器
        
        这个方法支持动态加载，如果请求的模型已经加载，会直接返回缓存
        支持加载主模型、适配器和草稿模型的组合
        
        Args:
            model_path: 主模型路径
            adapter_path: 适配器路径（可选）
            draft_model_path: 草稿模型路径（可选）
            
        Returns:
            tuple: (模型, 分词器)
            
        Raises:
            ValueError: 当缺少必要的模型路径时抛出
        """
        # 处理默认模型标识
        model_path = self.default_model_map.get(model_path, model_path)
        
        # 如果请求的模型已经加载，直接返回
        if self.model_key == (model_path, adapter_path, draft_model_path):
            return self.model, self.tokenizer

        # 清理旧模型
        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.draft_model = None

        # 构建分词器配置
        tokenizer_config = {
            "trust_remote_code": True if self.cli_args.trust_remote_code else None
        }
        if self.cli_args.chat_template:
            tokenizer_config["chat_template"] = self.cli_args.chat_template

        # 加载主模型和分词器
        if model_path == "default_model":
            if self.cli_args.model is None:
                raise ValueError(
                    "A model path has to be given as a CLI "
                    "argument or in the HTTP request"
                )
            adapter_path = adapter_path or self.cli_args.adapter_path
            model, tokenizer = load(
                self.cli_args.model,
                adapter_path=adapter_path,
                tokenizer_config=tokenizer_config,
            )
        else:
            model, tokenizer = load(
                model_path, 
                adapter_path=adapter_path, 
                tokenizer_config=tokenizer_config
            )

        # 处理默认聊天模板
        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        # 更新当前模型信息
        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer

        # 验证草稿模型分词器兼容性
        def validate_draft_tokenizer(draft_tokenizer):
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logging.warning(
                    "Draft model tokenizer does not match model tokenizer. "
                    "Speculative decoding may not work as expected."
                )

        # 加载草稿模型（如果指定）
        if (
            draft_model_path == "default_model"
            and self.cli_args.draft_model is not None
        ):
            self.draft_model, draft_tokenizer = load(self.cli_args.draft_model)
            validate_draft_tokenizer(draft_tokenizer)

        elif draft_model_path is not None and draft_model_path != "default_model":
            self.draft_model, draft_tokenizer = load(draft_model_path)
            validate_draft_tokenizer(draft_tokenizer)
            
        return self.model, self.tokenizer


# ==============================================================================
# 响应生成器
# ==============================================================================

class ResponseGenerator:
    """
    响应生成器类
    
    这是服务器的核心组件，负责：
    1. 管理生成请求队列
    2. 实现批处理优化
    3. 协调模型加载和文本生成
    4. 管理缓存系统
    5. 处理流式和非流式响应
    
    采用生产者-消费者模式，主线程接收请求，后台线程处理生成
    """

    def __init__(self, model_provider: ModelProvider, prompt_cache: LRUPromptCache):
        """
        初始化响应生成器
        
        Args:
            model_provider: 模型提供者实例
            prompt_cache: 缓存系统实例
        """
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache
        self.requests = Queue()  # 请求队列

        # 启动后台生成线程
        self._stop = False
        self._generation_thread = Thread(target=self._generate)
        self._generation_thread.start()

    def stop_and_join(self):
        """停止生成线程并等待其结束"""
        self._stop = True
        self._generation_thread.join()

    def _tokenize(self, tokenizer, request):
        """
        将请求分词为token序列
        
        根据请求类型（聊天或文本）使用不同的分词策略
        
        Args:
            tokenizer: 分词器实例
            request: 完成请求对象
            
        Returns:
            List[int]: token序列
        """
        if request.request_type == "chat":
            messages = request.messages
            tools = request.tools
            role_mapping = request.role_mapping

            # 如果分词器有聊天模板，使用它
            if tokenizer.chat_template:
                process_message_content(messages)
                return tokenizer.apply_chat_template(
                    messages,
                    tools,
                    add_generation_prompt=True,
                    **self.model_provider.cli_args.chat_template_args,
                )
            else:
                # 否则使用简单的聊天转换
                return tokenizer.encode(convert_chat(messages, role_mapping))
        else:
            # 文本补全模式，直接编码提示
            return tokenizer.encode(request.prompt)

    def _is_batchable(self, args):
        """
        判断请求是否可以批处理
        
        批处理需要满足特定条件，确保多个请求可以一起高效处理
        
        Args:
            args: 生成参数
            
        Returns:
            bool: 是否可以批处理
        """
        # 草稿模型不支持批处理
        if (
            args.model.draft != "default_model"
            or self.model_provider.cli_args.draft_model is not None
        ):
            return False
            
        # logits偏置不支持批处理
        if args.logits.logit_bias is not None:
            return False
            
        # 重复惩罚不支持批处理
        if args.logits.repetition_penalty != 0:
            return False
            
        # 概率信息不支持批处理
        if args.logprobs > 0:
            return False
            
        # 指定随机种子不支持批处理
        if args.seed is not None:
            return False

        return True

    def _generate(self):
        """
        后台生成线程的主函数
        
        这个方法在单独的线程中运行，负责：
        1. 从队列中获取请求
        2. 管理批处理生成器
        3. 处理单个请求生成
        4. 管理缓存和响应
        """
        current_model = None
        current_sampling = None
        current_tokenizer = None
        current_model_key = None
        batch_generator = None          # 批处理生成器
        drain_batch = False             # 是否需要清空当前批次
        batch_results = {}              # 批次结果字典

        unprocessed_requests = []       # 未处理的请求列表

        def get_next_request():
            """获取下一个要处理的请求"""
            if unprocessed_requests:
                return unprocessed_requests.pop()
            else:
                try:
                    return self.requests.get_nowait()
                except QueueEmpty:
                    return None

        def progress_callback(info):
            """批处理进度回调函数"""
            for uid, processed, total in info:
                if uid in batch_results:
                    batch_results[uid]["rqueue"].put((min(processed, total), total))

        # 主生成循环
        while not self._stop:
            request = None
            if not drain_batch:
                request = get_next_request()

            # 处理新请求
            if request is not None:
                rqueue, request, args = request
                is_batchable = self._is_batchable(args)

                # 可以添加到当前批次
                if (
                    batch_generator is not None
                    and current_model == args.model
                    and current_sampling == args.sampling
                    and is_batchable
                ):
                    prompt = self._tokenize(current_tokenizer, request)
                    ctx = GenerationContext(
                        has_tool_calling=current_tokenizer.has_tool_calling,
                        tool_call_start=current_tokenizer.tool_call_start,
                        tool_call_end=current_tokenizer.tool_call_end,
                        eos_token_id=current_tokenizer.eos_token_id,
                        stop_token_sequences=[
                            current_tokenizer.encode(stop_word, add_special_tokens=False)
                            for stop_word in args.stop_words
                        ],
                        prompt=prompt,
                    )
                    rqueue.put(ctx)

                    # 获取缓存
                    cache, rest = self.prompt_cache.fetch_nearest_cache(
                        current_model_key, prompt
                    )
                    if cache is None:
                        cache = make_prompt_cache(self.model_provider.model)

                    # 插入到批处理生成器
                    (uid,) = batch_generator.insert(
                        [rest], args.max_tokens, caches=[cache]
                    )
                    batch_results[uid] = {
                        "ctx": ctx,
                        "cache_key": prompt[:],
                        "rqueue": rqueue,
                        "detokenizer": current_tokenizer.detokenizer,
                    }
                    continue

                # 无法批处理，需要单独处理
                elif batch_generator is None and not is_batchable:
                    self._serve_single((rqueue, request, args))
                    continue

                # 创建新的批处理生成器
                elif batch_generator is None:
                    try:
                        model, tokenizer = self.model_provider.load(
                            args.model.model, args.model.adapter, args.model.draft
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    current_model = args.model
                    current_sampling = args.sampling
                    current_tokenizer = tokenizer
                    current_model_key = self.model_provider.model_key
                    batch_results = {}
                    batch_generator = BatchGenerator(
                        model,
                        stop_tokens=tokenizer.eos_token_ids,
                        sampler=make_sampler(
                            args.sampling.temperature,
                            top_p=args.sampling.top_p,
                            top_k=args.sampling.top_k,
                            min_p=args.sampling.min_p,
                            xtc_probability=args.sampling.xtc_probability,
                            xtc_threshold=args.sampling.xtc_threshold,
                            xtc_special_tokens=[
                                tokenizer.eos_token_id,
                                tokenizer.encode("\n"),
                            ],
                        ),
                        prompt_progress_callback=progress_callback,
                    )
                    unprocessed_requests.append((rqueue, request, args))
                    continue

                # 需要清空当前批次
                else:
                    drain_batch = True
                    unprocessed_requests.append((rqueue, request, args))
                    continue

            # 从当前批次获取响应
            elif batch_generator is not None:
                if len(batch_results) == 0:
                    if drain_batch:
                        current_model = None
                        current_sampling = None
                        current_tokenizer = None
                        current_model_key = None
                        batch_generator = None
                        drain_batch = False
                    continue

                uids_to_remove = []
                time_budget = 0.5  # 时间预算，避免长时间阻塞
                start = time.time()
                
                while True:
                    if time.time() - start > time_budget:
                        break

                    responses = batch_generator.next()
                    if not responses:
                        break

                    for r in responses:
                        result = batch_results[r.uid]
                        result["cache_key"].append(r.token)
                        result["detokenizer"].add_token(r.token)

                        top_tokens = None
                        # 处理概率信息
                        if args.logprobs > 0:
                            sorted_indices = mx.argpartition(
                                -gen.logprobs, kth=args.logprobs - 1
                            )
                            top_indices = sorted_indices[: args.logprobs]
                            top_logprobs = gen.logprobs[top_indices]
                            top_token_info = zip(
                                top_indices.tolist(), top_logprobs.tolist()
                            )
                            top_tokens = tuple(top_token_info)
                            
                        # 发送响应到请求队列
                        result["rqueue"].put(
                            Response(
                                result["detokenizer"].last_segment,
                                r.token,
                                r.logprobs[r.token].item(),
                                r.finish_reason,
                                top_tokens,
                            )
                        )

                        # 处理完成情况
                        if r.finish_reason is not None:
                            result["rqueue"].put(None)
                            self.prompt_cache.insert_cache(
                                current_model_key, result["cache_key"], r.prompt_cache
                            )
                            del batch_results[r.uid]

                        # 检查是否需要停止
                        if result["ctx"]._should_stop:
                            uids_to_remove.append(r.uid)

                    # 移除需要停止的请求
                    if uids_to_remove:
                        batch_generator.remove(uids_to_remove)

    def _serve_single(self, request):
        """
        处理单个请求的生成
        
        对于无法批处理的请求，使用单独的生成流程
        
        Args:
            request: 包含队列、请求和参数的元组
        """
        rqueue, request, args = request

        # 进度回调函数
        def progress(tokens_processed, tokens_total):
            rqueue.put((tokens_processed, tokens_total))

        try:
            # 加载模型和分词器
            model, tokenizer = self.model_provider.load(
                args.model.model, args.model.adapter, args.model.draft
            )
            draft_model = self.model_provider.draft_model

            # 准备提示
            prompt = self._tokenize(tokenizer, request)

            # 创建生成上下文
            ctx = GenerationContext(
                has_tool_calling=tokenizer.has_tool_calling,
                tool_call_start=tokenizer.tool_call_start,
                tool_call_end=tokenizer.tool_call_end,
                eos_token_id=tokenizer.eos_token_id,
                stop_token_sequences=[
                    tokenizer.encode(stop_word, add_special_tokens=False)
                    for stop_word in args.stop_words
                ],
                prompt=prompt,
            )
            rqueue.put(ctx)

            # 设置随机种子
            if args.seed is not None:
                mx.random.seed(args.seed)

            # 创建采样器和logits处理器
            sampler = make_sampler(
                args.sampling.temperature,
                top_p=args.sampling.top_p,
                top_k=args.sampling.top_k,
                min_p=args.sampling.min_p,
                xtc_probability=args.sampling.xtc_probability,
                xtc_threshold=args.sampling.xtc_threshold,
                xtc_special_tokens=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n"),
                ],
            )
            logits_processors = make_logits_processors(
                args.logits.logit_bias,
                args.logits.repetition_penalty,
                args.logits.repetition_context_size,
            )

            # 获取KV缓存
            cache, rest = self.prompt_cache.fetch_nearest_cache(
                self.model_provider.model_key, prompt
            )
            cache_key = prompt[:]
            if cache is None:
                cache = make_prompt_cache(self.model_provider.model)
                if self.model_provider.draft_model is not None:
                    cache += make_prompt_cache(self.model_provider.draft_model)

            # 生成token序列
            for gen in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=rest,
                max_tokens=args.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=cache,
                draft_model=draft_model,
                num_draft_tokens=args.num_draft_tokens,
                prompt_progress_callback=progress,
            ):
                top_tokens = None
                if args.logprobs > 0:
                    sorted_indices = mx.argpartition(
                        -gen.logprobs, kth=args.logprobs - 1
                    )
                    top_indices = sorted_indices[: args.logprobs]
                    top_logprobs = gen.logprobs[top_indices]
                    top_token_info = zip(top_indices.tolist(), top_logprobs.tolist())
                    top_tokens = tuple(top_token_info)

                # 发送生成的响应
                rqueue.put(
                    Response(
                        gen.text,
                        gen.token,
                        gen.logprobs[gen.token].item(),
                        gen.finish_reason,
                        top_tokens,
                    )
                )
                cache_key.append(gen.token)

                # 检查是否需要停止
                if ctx._should_stop:
                    break

            rqueue.put(None)

            # 保存缓存
            self.prompt_cache.insert_cache(
                self.model_provider.model_key, cache_key, cache
            )

        except Exception as e:
            rqueue.put(e)

    def generate(
        self,
        request: CompletionRequest,
        generation_args: GenerationArguments,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        生成文本的主要入口点
        
        Args:
            request: 完成请求
            generation_args: 生成参数
            progress_callback: 进度回调函数
            
        Returns:
            tuple: (生成上下文, 响应生成器)
        """
        response_queue = Queue()
        self.requests.put((response_queue, request, generation_args))

        def _inner():
            """内部响应生成器"""
            while True:
                response = response_queue.get()
                if response is None:
                    break
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, tuple):
                    if progress_callback is not None:
                        progress_callback(*response)
                    continue
                yield response

        # 获取生成上下文
        ctx = response_queue.get()
        if isinstance(ctx, Exception):
            raise ctx

        return ctx, _inner()

    @property
    def cli_args(self):
        """获取命令行参数"""
        return self.model_provider.cli_args


# ==============================================================================
# API 请求处理器
# ==============================================================================

class APIHandler(BaseHTTPRequestHandler):
    """
    HTTP API 请求处理器
    
    这个类继承自 BaseHTTPRequestHandler，负责处理所有 HTTP 请求
    支持与 OpenAI API 兼容的接口格式
    
    主要端点：
    - POST /v1/completions: 文本补全
    - POST /v1/chat/completions: 聊天对话
    - GET /v1/models: 模型列表
    - GET /health: 健康检查
    """

    def __init__(
        self,
        response_generator: ResponseGenerator,
        *args,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化 API 处理器
        
        Args:
            response_generator: 响应生成器实例
            *args: 传递给父类的位置参数
            system_fingerprint: 系统指纹，用于响应头
            **kwargs: 传递给父类的关键字参数
        """
        # 创建请求特定的元数据
        self.created = int(time.time())
        self.response_generator = response_generator
        self.system_fingerprint = system_fingerprint or get_system_fingerprint()
        super().__init__(*args, **kwargs)

    def _set_cors_headers(self):
        """设置 CORS（跨域资源共享）头"""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _set_completion_headers(self, status_code: int = 200):
        """设置完成请求的响应头"""
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()

    def _set_stream_headers(self, status_code: int = 200):
        """设置流式响应的响应头"""
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()

    def do_OPTIONS(self):
        """处理 OPTIONS 请求（用于 CORS 预检）"""
        self._set_completion_headers(204)
        self.end_headers()

    def do_POST(self):
        """
        处理 POST 请求
        
        这是主要的请求处理方法，会根据路径调用相应的处理函数
        支持的端点：
        - /v1/completions: 文本补全
        - /v1/chat/completions: 聊天对话
        - /chat/completions: 聊天对话（兼容性路径）
        """
        # 路径到处理函数的映射
        request_factories = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
        }

        # 检查路径是否支持
        if self.path not in request_factories:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # 读取和解析请求体
        content_length = int(self.headers["Content-Length"])
        raw_body = self.rfile.read(content_length)
        try:
            self.body = json.loads(raw_body.decode())
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e} - Raw body: {raw_body.decode()}")
            self._set_completion_headers(400)
            self.wfile.write(
                json.dumps({"error": f"Invalid JSON in request body: {e}"}).encode()
            )
            return

        # 记录请求信息
        indent = "\t"
        logging.debug(f"Incoming Request Body: {json.dumps(self.body, indent=indent)}")
        assert isinstance(
            self.body, dict
        ), f"Request should be dict, but got {type(self.body)}"

        # 提取请求参数
        self.stream = self.body.get("stream", False)
        self.stream_options = self.body.get("stream_options", None)
        self.requested_model = self.body.get("model", "default_model")
        self.requested_draft_model = self.body.get("draft_model", "default_model")
        self.num_draft_tokens = self.body.get(
            "num_draft_tokens", self.response_generator.cli_args.num_draft_tokens
        )
        self.adapter = self.body.get("adapters", None)
        self.max_tokens = self.body.get("max_completion_tokens", None)
        if self.max_tokens is None:
            self.max_tokens = self.body.get(
                "max_tokens", self.response_generator.cli_args.max_tokens
            )
        self.temperature = self.body.get(
            "temperature", self.response_generator.cli_args.temp
        )
        self.top_p = self.body.get("top_p", self.response_generator.cli_args.top_p)
        self.top_k = self.body.get("top_k", self.response_generator.cli_args.top_k)
        self.min_p = self.body.get("min_p", self.response_generator.cli_args.min_p)
        self.repetition_penalty = self.body.get("repetition_penalty", 1.0)
        self.repetition_context_size = self.body.get("repetition_context_size", 20)
        self.xtc_probability = self.body.get("xtc_probability", 0.0)
        self.xtc_threshold = self.body.get("xtc_threshold", 0.0)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", -1)
        self.seed = self.body.get("seed", None)
        self.validate_model_parameters()

        # 获取停止词
        stop_words = self.body.get("stop")
        stop_words = stop_words or []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words

        # 创建完成请求并处理
        request = request_factories[self.path]()
        self.handle_completion(request, stop_words)

    def validate_model_parameters(self):
        """
        验证模型参数的类型和值
        
        确保所有传入的参数都在有效范围内
        
        Raises:
            ValueError: 当参数无效时抛出
        """
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")

        if not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")

        if not isinstance(self.temperature, (float, int)) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative float")

        if not isinstance(self.top_p, (float, int)) or self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be a float between 0 and 1")

        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")

        if not isinstance(self.min_p, (float, int)) or self.min_p < 0 or self.min_p > 1:
            raise ValueError("min_p must be a float between 0 and 1")

        if not isinstance(self.num_draft_tokens, int) or self.num_draft_tokens < 0:
            raise ValueError("num_draft_tokens must be a non-negative integer")

        if (
            not isinstance(self.repetition_penalty, (float, int))
            or self.repetition_penalty < 0
        ):
            raise ValueError("repetition_penalty must be a non-negative float")

        if self.logprobs != -1 and not (0 < self.logprobs <= 10):
            raise ValueError(
                f"logprobs must be between 1 and 10 but got {self.logprobs:,}"
            )

        if (
            not isinstance(self.repetition_context_size, int)
            or self.repetition_context_size < 0
        ):
            raise ValueError("repetition_context_size must be a non-negative integer")

        if self.logit_bias is not None:
            if not isinstance(self.logit_bias, dict):
                raise ValueError("logit_bias must be a dict of int to float")

            try:
                self.logit_bias = {int(k): v for k, v in self.logit_bias.items()}
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")
                
        if not (
            isinstance(self.xtc_probability, float)
            and 0.00 <= self.xtc_probability <= 1.00
        ):
            raise ValueError(f"xtc_probability must be a float between 0.00 and 1.00")
            
        if not (
            isinstance(self.xtc_threshold, float) and 0.00 <= self.xtc_threshold <= 0.50
        ):
            raise ValueError(f"xtc_threshold must be a float between 0.00 and 0.5")
            
        if not isinstance(self.requested_model, str):
            raise ValueError("model must be a string")
            
        if self.adapter is not None and not isinstance(self.adapter, str):
            raise ValueError("adapter must be a string")
            
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    def generate_response(
        self,
        text: str,
        finish_reason: Union[Literal["length", "stop"], None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Dict[int, float]]] = None,
        tokens: Optional[List[int]] = None,
        tool_calls: Optional[List[str]] = None,
    ) -> dict:
        """
        生成符合 OpenAI API 格式的响应
        
        根据响应类型（流式或非流式）和完成类型生成相应的响应结构
        
        Args:
            text (str): 模型生成的文本
            finish_reason: 完成原因（"length", "stop" 或 None）
            prompt_token_count: 提示 token 数量（非流式时使用）
            completion_token_count: 完成 token 数量（非流式时使用）
            token_logprobs: 每个 token 的对数概率
            top_tokens: 每个位置的 top token 列表
            tokens: token 列表
            tool_calls: 工具调用列表
            
        Returns:
            dict: 符合 OpenAI API 格式的响应字典
        """
        # 初始化默认值
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []
        tool_calls = tool_calls or []

        # 解析工具调用
        def parse_function(tool_text):
            tool_call = json.loads(tool_text.strip())
            return {
                "function": {
                    "name": tool_call.get("name", None),
                    "arguments": json.dumps(tool_call.get("arguments", "")),
                },
                "type": "function",
                "id": None,
            }

        # 静态响应部分
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                },
            ],
        }

        # 添加概率信息（如果有）
        if token_logprobs or top_logprobs or tokens:
            response["choices"][0]["logprobs"] = {
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "tokens": tokens,
            }

        # 添加使用情况（非流式时）
        if not self.stream:
            if not (
                isinstance(prompt_token_count, int)
                and isinstance(completion_token_count, int)
            ):
                raise ValueError(
                    "Response type is complete, but token counts not provided"
                )

            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }

        choice = response["choices"][0]

        # 添加动态响应内容
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {
                "role": "assistant",
                "content": text,
                "tool_calls": [parse_function(tool_text) for tool_text in tool_calls],
            }
        elif self.object_type == "text_completion":
            choice.update(text=text)
        else:
            raise ValueError(f"Unsupported response type: {self.object_type}")

        return response

    def handle_completion(self, request: CompletionRequest, stop_words: List[str]):
        """
        处理完成请求的主函数
        
        协调整个生成过程，包括参数设置、生成执行和响应发送
        
        Args:
            request: 完成请求对象
            stop_words: 停止词列表
        """
        # 构建生成参数
        args = GenerationArguments(
            model=ModelDescription(
                model=self.requested_model,
                draft=self.requested_draft_model,
                adapter=self.adapter,
            ),
            sampling=SamplingArguments(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                xtc_probability=self.xtc_probability,
                xtc_threshold=self.xtc_threshold,
            ),
            logits=LogitsProcessorArguments(
                logit_bias=self.logit_bias,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
            ),
            stop_words=stop_words,
            max_tokens=self.max_tokens,
            num_draft_tokens=self.num_draft_tokens,
            logprobs=self.logprobs,
            seed=self.seed,
        )

        # 保活回调函数，用于长提示处理期间保持连接
        def keepalive_callback(processed_tokens, total_tokens):
            logging.info(
                f"Prompt processing progress: {processed_tokens}/{total_tokens}"
            )
            if self.stream:
                try:
                    # 发送 SSE 注释作为保活信号
                    self.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # 客户端断开连接，忽略错误
                    pass

        # 开始生成
        try:
            ctx, response = self.response_generator.generate(
                request,
                args,
                progress_callback=keepalive_callback,
            )
        except Exception as e:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write((f"{e}").encode())
            return

        # 设置响应头
        if self.stream:
            self._set_stream_headers(200)
            self.end_headers()
            logging.debug("Starting stream:")
        else:
            self._set_completion_headers(200)
            logging.debug("Starting completion:")

        # 初始化状态变量
        in_tool_call = False     # 是否在工具调用中
        tool_calls = []          # 工具调用列表
        tool_text = ""           # 当前工具调用文本

        tokens = []              # 生成的 token 列表
        token_logprobs = []      # token 对数概率
        top_tokens = []          # top token 信息

        text = ""                # 完整生成文本
        segment = ""             # 当前文本片段

        finish_reason = "length" # 默认完成原因

        # 处理生成的 token 序列
        for gen in response:
            logging.debug(gen.text)

            # 处理工具调用或普通文本
            if ctx.has_tool_calling and gen.text == ctx.tool_call_start:
                in_tool_call = True
            elif in_tool_call:
                if gen.text == ctx.tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = ""
                    in_tool_call = False
                else:
                    tool_text += gen.text
            else:
                text += gen.text
                segment += gen.text

            # 保存 token 信息
            tokens.append(gen.token)
            token_logprobs.append(gen.logprob)

            # 保存 top token 信息
            if gen.top_tokens is not None:
                top_tokens.append(gen.top_tokens)

            # 检查停止条件
            stop_condition = stopping_criteria(
                tokens, ctx.stop_token_sequences, stop_words, ctx.eos_token_id
            )
            if stop_condition.stop_met:
                finish_reason = "stop"
                ctx.stop()
                tokens = tokens[: len(tokens) - stop_condition.trim_length]
                text = text[: len(text) - stop_condition.trim_text_length]
                segment = ""
                break

            # 流式响应处理
            if self.stream and not in_tool_call:
                # 检查是否有序列重叠
                if any(
                    (
                        sequence_overlap(tokens, sequence)
                        for sequence in ctx.stop_token_sequences
                    )
                ):
                    continue
                elif segment or tool_calls:
                    response = self.generate_response(
                        segment, None, tool_calls=tool_calls
                    )
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                    self.wfile.flush()
                    segment = ""
                    tool_calls = []

            # 检查生成器报告的完成原因
            if gen.finish_reason is not None:
                finish_reason = gen.finish_reason

        # 发送最终响应
        if self.stream:
            response = self.generate_response(
                segment, finish_reason, tool_calls=tool_calls
            )
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()
            
            # 发送使用情况信息（如果请求）
            if self.stream_options is not None and self.stream_options["include_usage"]:
                response = self.completion_usage_response(len(ctx.prompt), len(tokens))
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()
                
            # 发送流结束标志
            self.wfile.write("data: [DONE]\n\n".encode())
            self.wfile.flush()
        else:
            response = self.generate_response(
                text,
                finish_reason,
                len(ctx.prompt),
                len(tokens),
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
                tool_calls=tool_calls,
            )
            response_json = json.dumps(response).encode()
            indent = "\t"
            logging.debug(f"Outgoing Response: {json.dumps(response, indent=indent)}")

            # 发送 Content-Length 头
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json)
            self.wfile.flush()

    def completion_usage_response(
        self,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
    ):
        """
        生成使用情况响应
        
        Args:
            prompt_token_count: 提示 token 数量
            completion_token_count: 完成 token 数量
            
        Returns:
            dict: 使用情况响应字典
        """
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion",
            "model": self.requested_model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        }
        return response

    def handle_chat_completions(self) -> CompletionRequest:
        """
        处理聊天完成请求
        
        Returns:
            CompletionRequest: 构建的完成请求对象
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # 设置响应类型和ID
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"

        return CompletionRequest(
            "chat",
            "",
            body["messages"],
            body.get("tools") or None,
            body.get("role_mapping"),
        )

    def handle_text_completions(self) -> CompletionRequest:
        """
        处理文本完成请求
        
        Returns:
            CompletionRequest: 构建的完成请求对象
        """
        # 设置响应类型和ID
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"
        assert "prompt" in self.body, "Request did not contain a prompt"
        
        return CompletionRequest(
            "text",
            self.body["prompt"],
            [],
            None,
            None,
        )

    def do_GET(self):
        """
        处理 GET 请求
        
        支持的端点：
        - /v1/models: 获取模型列表
        - /health: 健康检查
        """
        if self.path.startswith("/v1/models"):
            self.handle_models_request()
        elif self.path == "/health":
            self.handle_health_check()
        else:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def handle_health_check(self):
        """
        处理健康检查请求
        
        返回服务器的健康状态
        """
        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write('{"status": "ok"}'.encode())
        self.wfile.flush()

    def handle_models_request(self):
        """
        处理模型列表请求
        
        扫描 Hugging Face 缓存目录，返回已下载的 MLX 模型列表
        """
        self._set_completion_headers(200)
        self.end_headers()

        # MLX 模型的特征文件
        files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

        # 解析路径参数
        parts = self.path.split("/")
        filter_repo_id = None
        if len(parts) > 3:
            filter_repo_id = "/".join(parts[3:])

        # 判断是否为 MLX 模型的函数
        def probably_mlx_lm(repo):
            if repo.repo_type != "model":
                return False
            if "main" not in repo.refs:
                return False
            if filter_repo_id is not None and repo.repo_id != filter_repo_id:
                return False
            file_names = {f.file_path.name for f in repo.refs["main"].files}
            return all(f in file_names for f in files)

        # 扫描缓存目录
        hf_cache_info = scan_cache_dir()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)
        ]

        # 构建模型列表
        models = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": self.created,
            }
            for repo in downloaded_models
        ]

        response = {"object": "list", "data": models}
        response_json = json.dumps(response).encode()
        self.wfile.write(response_json)
        self.wfile.flush()


# ==============================================================================
# 服务器启动和配置
# ==============================================================================

def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=ThreadingHTTPServer,
    handler_class=APIHandler,
):
    """
    启动 HTTP 服务器
    
    Args:
        host (str): 监听主机地址
        port (int): 监听端口
        model_provider: 模型提供者实例
        server_class: HTTP 服务器类（默认使用线程化服务器）
        handler_class: 请求处理器类
    """
    server_address = (host, port)
    
    # 创建响应生成器和缓存系统
    response_generator = ResponseGenerator(model_provider, LRUPromptCache())
    
    # 获取网络地址信息
    infos = socket.getaddrinfo(
        *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    server_class.address_family, _, _, _, server_address = next(iter(infos))
    
    # 创建 HTTP 服务器
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            response_generator,
            system_fingerprint=get_system_fingerprint(),
            *args,
            **kwargs,
        ),
    )
    
    # 发出生产环境警告
    warnings.warn(
        "mlx_lm.server is not recommended for production as "
        "it only implements basic security checks."
    )
    
    logging.info(f"Starting httpd at {host} on port {port}...")
    httpd.serve_forever()


def main():
    """
    主函数 - 解析命令行参数并启动服务器
    
    这个函数处理所有命令行参数的解析，并启动 HTTP 服务器
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="MLX Http Server.")
    
    # 模型相关参数
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    
    # 服务器配置参数
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    
    # 安全和信任参数
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    
    # 日志配置参数
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    
    # 聊天模板参数
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    
    # 生成参数默认值
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 配置日志系统
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # 启动服务器
    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    # 显示弃用警告
    print(
        "Calling `python -m mlx_lm.server...` directly is deprecated."
        " Use `mlx_lm.server...` or `python -m mlx_lm server ...` instead."
    )
    main()
