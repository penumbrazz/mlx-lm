# Copyright © 2023-2024 Apple Inc.

# =============================================================================
# MLX Language Model HTTP Server (中文注释版)
# =============================================================================
# 
# 这个文件实现了一个基于 MLX 框架的语言模型 HTTP 服务器，提供类似 OpenAI API 的接口。
# 主要功能包括：
# 1. 支持文本补全和聊天补全 API
# 2. 模型动态加载和管理
# 3. 提示缓存机制，提高推理效率
# 4. 支持流式和非流式响应
# 5. 支持多种采样参数配置
# 6. 支持工具调用（Tool Calling）
# 7. 支持推测解码（Speculative Decoding）
#
# 使用方法：
# python -m mlx_lm.server --model <模型路径> --host 127.0.0.1 --port 8080
#
# API 端点：
# - POST /v1/completions - 文本补全
# - POST /v1/chat/completions - 聊天补全  
# - GET /v1/models - 获取可用模型列表
# - GET /health - 健康检查
#
# 注意：此服务器主要用于开发和测试，不建议在生产环境中使用。
# =============================================================================

import argparse  # 命令行参数解析模块，用于处理启动时的各种参数
import json  # JSON 数据处理模块，用于 API 请求和响应的序列化
import logging  # 日志记录模块，用于输出运行时的调试信息
import platform  # 系统平台信息模块，用于获取操作系统等信息
import socket  # 网络套接字模块，用于创建 HTTP 服务器
import time  # 时间处理模块，用于生成时间戳和性能计时
import uuid  # UUID 生成模块，用于为每个请求生成唯一标识符
import warnings  # 警告信息模块，用于输出安全提醒
from dataclasses import dataclass, field  # 数据类装饰器，用于简化类的定义
from http.server import BaseHTTPRequestHandler, HTTPServer  # HTTP 服务器基础类
from pathlib import Path  # 路径处理模块，用于文件系统操作
from typing import (  # 类型注解模块，用于提供更好的类型检查
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx  # MLX 核心模块，苹果芯片的机器学习框架
from huggingface_hub import scan_cache_dir  # Hugging Face Hub 缓存扫描工具

from ._version import __version__  # 版本信息
from .generate import stream_generate  # 流式文本生成函数
from .models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache  # 缓存相关函数
from .sample_utils import make_logits_processors, make_sampler  # 采样工具函数
from .utils import common_prefix_len, load  # 通用工具函数


def get_system_fingerprint():
    """
    生成系统指纹，用于标识当前运行环境的唯一性
    
    这个函数会收集以下信息来生成一个唯一的指纹字符串：
    - 当前 MLX-LM 的版本号
    - MLX 框架的版本号
    - 操作系统平台信息
    - GPU 架构信息（如果使用 Metal）
    
    返回的指纹可以用于：
    - 缓存键的生成
    - 调试时的环境标识
    - API 响应中的系统信息
    
    Returns:
        str: 系统指纹字符串，格式为 "版本信息-MLX版本-平台信息-GPU架构"
    """
    # 检查是否支持 Metal（苹果 GPU），如果不支持则 GPU 架构为空字符串
    gpu_arch = mx.metal.device_info()["architecture"] if mx.metal.is_available() else ""
    # 组合所有信息生成指纹
    return f"{__version__}-{mx.__version__}-{platform.platform()}-{gpu_arch}"


class StopCondition(NamedTuple):
    """
    停止条件的命名元组，用于封装生成停止时的相关信息
    
    Attributes:
        stop_met (bool): 是否满足停止条件
        trim_length (int): 需要修剪的 token 数量，用于处理停止词重叠的情况
    """
    stop_met: bool
    trim_length: int


def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]],
    eos_token_id: Union[int, None],
) -> StopCondition:
    """
    判断是否应该停止 token 生成
    
    这个函数是生成过程的核心控制逻辑，它会检查多种停止条件：
    1. 是否生成了结束符（EOS token）
    2. 是否匹配到了用户指定的停止词序列
    
    Args:
        tokens (List[int]): 当前已生成的 token 序列
        stop_id_sequences (List[List[int]]): 停止词的 token 序列列表
            - 每个元素是一个 token 序列，代表一个停止词
            - 例如：["你好"] -> [123, 456]
        eos_token_id (Union[int, None]): 结束符的 token ID
            - 当生成这个 token 时应该停止
            - None 表示没有明确的结束符
    
    Returns:
        StopCondition: 停止条件对象
            - stop_met: 是否满足停止条件
            - trim_length: 如果因为停止词而停止，需要修剪的 token 数量
    
    Examples:
        >>> tokens = [1, 2, 3, 4]
        >>> stop_id_sequences = [[3, 4]]  # 停止词序列
        >>> eos_token_id = 5
        >>> result = stopping_criteria(tokens, stop_id_sequences, eos_token_id)
        >>> print(result.stop_met)  # True，因为匹配到停止词
        >>> print(result.trim_length)  # 2，需要修剪2个token
    """
    # 检查是否生成了结束符
    if tokens and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=0)

    # 检查是否匹配到任何停止词序列
    for stop_ids in stop_id_sequences:
        # 确保当前 token 序列长度足够进行比较
        if len(tokens) >= len(stop_ids):
            # 检查序列末尾是否与停止词匹配
            if tokens[-len(stop_ids):] == stop_ids:
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    # 没有满足任何停止条件
    return StopCondition(stop_met=False, trim_length=0)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    检查两个序列是否存在重叠
    
    具体来说，这个函数检查 s1 的后缀是否与 s2 的前缀有重叠。
    这个函数在处理停止词时非常重要，因为可能出现停止词被分割到两次生成中的情况。
    
    例如：
    - 停止词是 "hello"
    - 第一次生成 "hel"
    - 第二次生成 "lo world"
    - 需要检测到这种跨序列的重叠
    
    Args:
        s1 (Sequence): 第一个序列（通常是已生成的 token 序列）
        s2 (Sequence): 第二个序列（通常是停止词序列）
    
    Returns:
        bool: 是否存在重叠
        
    Examples:
        >>> sequence_overlap([1, 2, 3], [3, 4, 5])  # True，3 重叠
        >>> sequence_overlap([1, 2], [3, 4])       # False，无重叠
        >>> sequence_overlap([1, 2, 3], [1, 2, 3]) # True，完全重叠
    """
    # 最大可能的重叠长度是两个序列长度的较小值
    max_overlap = min(len(s1), len(s2))
    # 检查从1到最大重叠长度的所有可能重叠情况
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    """
    将聊天消息列表转换为文本提示
    
    这个函数将结构化的聊天消息（包含角色和内容）转换为模型可以直接处理的文本格式。
    如果没有提供角色映射，则使用默认的映射规则。
    
    默认的聊天格式是：
    ```
    ASSISTANT's RULE: 系统消息内容
    USER: 用户消息内容
    ASSISTANT: 助手回复内容
    USER: 用户消息内容
    ASSISTANT: 
    ```
    
    Args:
        messages (List[dict]): 聊天消息列表
            - 每个消息是字典，包含 'role' 和 'content' 键
            - role 可以是 'system', 'user', 'assistant' 等
            - content 是消息的文本内容
        role_mapping (Optional[dict]): 角色映射字典
            - 定义不同角色的前缀和分隔符
            - 如果为 None，使用默认映射
    
    Returns:
        str: 转换后的文本提示，可以直接输入给模型
        
    Examples:
        >>> messages = [
        ...     {"role": "user", "content": "你好"},
        ...     {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
        ... ]
        >>> prompt = convert_chat(messages)
        >>> print(prompt)
        USER: 你好
        ASSISTANT: 你好！有什么可以帮助你的吗？
        ASSISTANT: 
    """
    # 默认的角色映射配置
    default_role_mapping = {
        "system_prompt": (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant follows the given rules no matter what."
        ),
        "system": "ASSISTANT's RULE: ",  # 系统消息的前缀
        "user": "USER: ",               # 用户消息的前缀
        "assistant": "ASSISTANT: ",      # 助手消息的前缀
        "stop": "\n",                   # 消息之间的分隔符
    }
    # 使用提供的映射或默认映射
    role_mapping = role_mapping if role_mapping is not None else default_role_mapping

    prompt = ""
    # 遍历所有消息，逐个构建提示
    for line in messages:
        role_prefix = role_mapping.get(line["role"], "")  # 获取角色前缀
        stop = role_mapping.get("stop", "")              # 获取分隔符
        content = line.get("content", "")                # 获取消息内容
        prompt += f"{role_prefix}{content}{stop}"         # 拼接消息

    # 添加助手角色的最后前缀，准备让模型生成回复
    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()  # 移除末尾的多余空白字符


def process_message_content(messages):
    """
    处理消息内容，确保其格式适合 apply_chat_template 函数
    
    在某些 API 中，消息的 content 字段可能是一个复杂结构，包含多种类型的内容
   （如文本、图片等）。这个函数会提取其中的文本部分，将复杂结构简化为纯文本。
    
    主要处理情况：
    - content 是列表：包含多个内容片段，提取所有文本片段
    - content 是 None：转换为空字符串
    - content 是字符串：保持不变
    
    Args:
        messages (list): 消息列表，会就地修改
        
    Raises:
        ValueError: 当遇到不支持的内容类型时抛出
        
    Examples:
        >>> messages = [
        ...     {"role": "user", "content": [{"type": "text", "text": "你好"}]}
        ... ]
        >>> process_message_content(messages)
        >>> print(messages[0]["content"])  # "你好"
    """
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            # 提取所有文本片段
            text_fragments = [
                fragment["text"] for fragment in content if fragment["type"] == "text"
            ]
            # 检查是否所有片段都是文本类型
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            # 将文本片段合并为一个字符串
            message["content"] = "".join(text_fragments)
        elif content is None:
            # None 转换为空字符串
            message["content"] = ""


@dataclass
class PromptCache:
    """
    提示缓存类，用于缓存已处理的提示和相关状态
    
    这个类实现了智能缓存机制，可以避免重复处理相同的提示内容，
    从而显著提高生成性能。缓存包含了模型的 KV 缓存状态。
    
    Attributes:
        cache (List[Any]): 实际的缓存数据，包含各层的键值缓存
        model_key (Tuple[str, Optional[str]]): 模型标识键
            - 包含模型路径、适配器路径、草稿模型路径的元组
            - 用于检测模型是否发生变化
        tokens (List[int]): 缓存对应的 token 序列
            - 用于计算新提示与缓存的公共前缀
    """
    cache: List[Any] = field(default_factory=list)  # 模型缓存数据
    model_key: Tuple[str, Optional[str]] = ("", None, None)  # 模型标识
    tokens: List[int] = field(default_factory=list)  # token 序列


class ModelProvider:
    """
    模型提供者类，负责模型的动态加载和管理
    
    这个类是服务器的核心组件之一，实现了按需加载模型的机制。
    支持同时管理多个模型，包括主模型、适配器和草稿模型。
    
    主要功能：
    - 按需加载模型，避免启动时加载所有模型
    - 模型生命周期管理
    - 支持适配器（Adapter）动态加载
    - 支持草稿模型用于推测解码
    - 验证模型兼容性
    
    使用示例：
        provider = ModelProvider(args)
        model, tokenizer = provider.load("model_path")
    """
    
    def __init__(self, cli_args: argparse.Namespace):
        """
        初始化模型提供者
        
        Args:
            cli_args (argparse.Namespace): 命令行参数
                - 包含模型路径、适配器路径等配置信息
        """
        self.cli_args = cli_args  # 保存命令行参数
        self.model_key = None     # 当前加载的模型标识
        self.model = None         # 当前加载的主模型
        self.tokenizer = None     # 当前加载的分词器
        self.draft_model = None   # 当前加载的草稿模型

        # 预加载默认模型（如果提供了）
        self.default_model_map = {}
        if self.cli_args.model is not None:
            # 将命令行指定的模型标记为 "default_model"
            self.default_model_map[self.cli_args.model] = "default_model"
            # 预加载默认模型和草稿模型
            self.load(self.cli_args.model, adapter_path="default_model", draft_model_path="default_model")

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        """
        加载指定的模型、适配器和草稿模型
        
        这个方法实现了智能的模型加载逻辑：
        1. 检查是否已经加载了相同的模型组合，如果是则直接返回
        2. 清理当前加载的模型和缓存
        3. 加载新的模型和分词器
        4. 配置分词器参数（如聊天模板）
        5. 加载草稿模型（如果指定）
        
        Args:
            model_path (str): 模型路径或 "default_model"
            adapter_path (str, optional): 适配器路径或 "default_model"
            draft_model_path (str, optional): 草稿模型路径或 "default_model"
        
        Returns:
            Tuple[Any, Any]: 返回 (model, tokenizer) 元组
            
        Raises:
            ValueError: 当模型路径无效时抛出
        """
        # 处理默认模型的特殊标记
        model_path = self.default_model_map.get(model_path, model_path)
        
        # 检查是否已经加载了相同的模型组合
        if self.model_key == (model_path, adapter_path, draft_model_path):
            return self.model, self.tokenizer

        # 清理当前加载的模型
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
                model_path, adapter_path=adapter_path, tokenizer_config=tokenizer_config
            )

        # 配置默认聊天模板（如果需要）
        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        # 更新当前模型标识和对象
        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer

        # 验证草稿模型分词器兼容性的内部函数
        def validate_draft_tokenizer(draft_tokenizer):
            # 检查分词器是否兼容
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
            self._validate_model_path(draft_model_path)
            self.draft_model, draft_tokenizer = load(draft_model_path)
            validate_draft_tokenizer(draft_tokenizer)
            
        return self.model, self.tokenizer


class APIHandler(BaseHTTPRequestHandler):
    """
    API 请求处理器类，继承自 BaseHTTPRequestHandler
    
    这个类是服务器的核心，负责处理所有的 HTTP 请求，包括：
    - POST /v1/completions (文本补全)
    - POST /v1/chat/completions (聊天补全)
    - GET /v1/models (模型列表)
    - GET /health (健康检查)
    
    主要功能：
    - 解析和验证请求参数
    - 调用模型进行文本生成
    - 管理提示缓存
    - 格式化响应数据
    - 处理流式和非流式输出
    - 支持多种采样参数
    - 支持工具调用
    """
    
    def __init__(
        self,
        model_provider: ModelProvider,
        *args,
        prompt_cache: Optional[PromptCache] = None,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化 API 处理器
        
        Args:
            model_provider (ModelProvider): 模型提供者实例
            *args: 传递给父类的位置参数
            prompt_cache (Optional[PromptCache]): 提示缓存对象
            system_fingerprint (Optional[str]): 系统指纹
            **kwargs: 传递给父类的关键字参数
        """
        # 创建静态的请求特定元数据
        self.created = int(time.time())  # 请求创建时间戳
        self.model_provider = model_provider  # 模型提供者引用
        self.prompt_cache = prompt_cache or PromptCache()  # 提示缓存
        self.system_fingerprint = system_fingerprint or get_system_fingerprint()  # 系统指纹
        super().__init__(*args, **kwargs)  # 调用父类初始化

    def _set_cors_headers(self):
        """
        设置 CORS（跨域资源共享）头部
        
        CORS 允许来自不同域名的网页应用访问这个 API。
        这里使用了非常宽松的设置（允许所有来源），在生产环境中
        应该更加谨慎地配置。
        """
        self.send_header("Access-Control-Allow-Origin", "*")  # 允许所有来源
        self.send_header("Access-Control-Allow-Methods", "*")  # 允许所有 HTTP 方法
        self.send_header("Access-Control-Allow-Headers", "*")  # 允许所有头部

    def _set_completion_headers(self, status_code: int = 200):
        """
        设置常规响应（非流式）的 HTTP 头部
        
        Args:
            status_code (int): HTTP 状态码，默认 200
        """
        self.send_response(status_code)  # 设置状态码
        self.send_header("Content-type", "application/json")  # JSON 格式响应
        self._set_cors_headers()  # 添加 CORS 头部

    def _set_stream_headers(self, status_code: int = 200):
        """
        设置流式响应的 HTTP 头部
        
        流式响应使用 Server-Sent Events (SSE) 格式，
        允许服务器向客户端推送数据片段。
        
        Args:
            status_code (int): HTTP 状态码，默认 200
        """
        self.send_response(status_code)  # 设置状态码
        self.send_header("Content-type", "text/event-stream")  # SSE 格式
        self.send_header("Cache-Control", "no-cache")  # 禁用缓存
        self._set_cors_headers()  # 添加 CORS 头部

    def do_OPTIONS(self):
        """
        处理 OPTIONS 预检请求
        
        浏览器在发送跨域请求前会先发送 OPTIONS 请求来检查权限。
        这里简单地返回 204 No Content 状态码。
        """
        self._set_completion_headers(204)  # 204 No Content
        self.end_headers()  # 结束头部发送

    def do_POST(self):
        """
        处理 POST 请求的主入口函数
        
        这是 API 处理的核心函数，负责：
        1. 路由到不同的端点处理函数
        2. 解析和验证请求体
        3. 提取请求参数
        4. 加载模型（如果需要）
        5. 调用具体的生成逻辑
        """
        # 定义支持的端点映射
        endpoints = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
        }

        # 检查请求路径是否支持
        if self.path not in endpoints:
            self._set_completion_headers(404)  # 404 Not Found
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
            # JSON 解析错误处理
            if self.stream:
                self._set_stream_headers(400)
                self.wfile.write(
                    f"data: {json.dumps({'error': f'Invalid JSON in request body: {e}'})}\n\n".encode()
                )
            else:
                self._set_completion_headers(400)
                self.wfile.write(
                    json.dumps({"error": f"Invalid JSON in request body: {e}"}).encode()
                )
            return

        # 调试输出请求体
        indent = "\t"  # 反斜杠不能在 f-strings 中
        logging.debug(f"Incoming Request Body: {json.dumps(self.body, indent=indent)}")
        assert isinstance(
            self.body, dict
        ), f"Request should be dict, but got {type(self.body)}"

        # 从请求体中提取参数
        self.stream = self.body.get("stream", False)  # 是否流式输出
        self.stream_options = self.body.get("stream_options", None)  # 流式选项
        self.requested_model = self.body.get("model", "default_model")  # 请求的模型
        self.requested_draft_model = self.body.get("draft_model", "default_model")  # 草稿模型
        self.num_draft_tokens = self.body.get(
            "num_draft_tokens", self.model_provider.cli_args.num_draft_tokens
        )  # 草稿 token 数量
        self.adapter = self.body.get("adapters", None)  # 适配器路径
        self.max_tokens = self.body.get("max_completion_tokens", None)  # 最大生成 token 数
        if self.max_tokens is None:
            self.max_tokens = self.body.get(
                "max_tokens", self.model_provider.cli_args.max_tokens
            )
        # 采样参数
        self.temperature = self.body.get(
            "temperature", self.model_provider.cli_args.temp
        )
        self.top_p = self.body.get("top_p", self.model_provider.cli_args.top_p)
        self.top_k = self.body.get("top_k", self.model_provider.cli_args.top_k)
        self.min_p = self.body.get("min_p", self.model_provider.cli_args.min_p)
        self.repetition_penalty = self.body.get("repetition_penalty", 1.0)
        self.repetition_context_size = self.body.get("repetition_context_size", 20)
        self.xtc_probability = self.body.get("xtc_probability", 0.0)
        self.xtc_threshold = self.body.get("xtc_threshold", 0.0)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", -1)
        self.seed = self.body.get("seed", None)
        
        # 验证参数
        self.validate_model_parameters()
        if self.seed is not None:
            mx.random.seed(self.seed)
            
        # 加载模型（如果需要）
        try:
            self.model, self.tokenizer = self.model_provider.load(
                self.requested_model,
                self.adapter,
                self.requested_draft_model,
            )
        except Exception as e:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write((f"{e}").encode())
            return

        # 获取停止词的 token 序列
        stop_words = self.body.get("stop")
        stop_words = stop_words or []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words
        stop_id_sequences = [
            self.tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # 根据是否流式输出设置响应头
        (
            self._set_stream_headers(200)
            if self.stream
            else self._set_completion_headers(200)
        )

        # 调用具体的端点处理函数
        prompt = endpoints[self.path]()
        self.handle_completion(prompt, stop_id_sequences)

    def validate_model_parameters(self):
        """
        验证模型参数的类型和值是否正确
        
        这个函数确保所有传入的参数都符合预期的格式和范围，
        防止无效参数导致模型运行错误。
        
        Raises:
            ValueError: 当参数不符合要求时抛出
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
        生成单个响应数据包
        
        根据响应类型（流式或非流式）和完成类型生成符合 OpenAI API 格式的响应。
        
        Args:
            text (str): 模型生成的文本
            finish_reason (Union[Literal["length", "stop"], None]): 结束原因
                - "length": 达到最大 token 数
                - "stop": 遇到停止词
                - None: 继续生成（流式）
            prompt_token_count (Optional[int]): 提示的 token 数量
            completion_token_count (Optional[int]): 完成文本的 token 数量
            token_logprobs (Optional[List[float]]): 每个 token 的对数概率
            top_tokens (Optional[List[Dict[int, float]]]): 每个位置的前 N 个 token
            tokens (Optional[List[int]]): token 序列
            tool_calls (Optional[List[str]]): 工具调用列表
        
        Returns:
            dict: 符合 OpenAI API 格式的响应字典
        """
        # 设置默认值
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []
        tool_calls = tool_calls or []

        # 解析工具调用的内部函数
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

        # 构建基础响应结构
        response = {
            "id": self.request_id,  # 请求唯一标识
            "system_fingerprint": self.system_fingerprint,  # 系统指纹
            "object": self.object_type,  # 响应类型
            "model": self.requested_model,  # 使用的模型
            "created": self.created,  # 创建时间戳
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,  # 结束原因
                },
            ],
        }

        # 添加概率信息（如果需要）
        if token_logprobs or top_logprobs or tokens:
            response["choices"][0]["logprobs"] = {
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "tokens": tokens,
            }

        # 非流式响应需要添加使用统计
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

        # 根据响应类型添加内容
        if self.object_type.startswith("chat.completion"):
            # 聊天补全格式
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {
                "role": "assistant",
                "content": text,
                "tool_calls": [parse_function(tool_text) for tool_text in tool_calls],
            }
        elif self.object_type == "text_completion":
            # 文本补全格式
            choice.update(text=text)
        else:
            raise ValueError(f"Unsupported response type: {self.object_type}")

        return response

    def reset_prompt_cache(self, prompt):
        """
        重置提示缓存和关联状态
        
        当模型发生变化或者无法复用现有缓存时，需要完全重置缓存。
        
        Args:
            prompt (List[int]): 新提示的 token 序列，用于初始化重置后的缓存
        """
        logging.debug(f"*** Resetting cache. ***")
        # 更新模型标识
        self.prompt_cache.model_key = self.model_provider.model_key
        # 创建新的模型缓存
        self.prompt_cache.cache = make_prompt_cache(self.model_provider.model)
        # 如果有草稿模型，也创建其缓存
        if self.model_provider.draft_model is not None:
            self.prompt_cache.cache += make_prompt_cache(
                self.model_provider.draft_model
            )
        # 缓存新的提示
        self.prompt_cache.tokens = list(prompt)

    def get_prompt_cache(self, prompt):
        """
        智能获取提示缓存，复用公共前缀
        
        这个函数实现了高效的缓存策略：
        1. 比较新提示与缓存的公共前缀长度
        2. 尝试修剪缓存以匹配公共前缀（避免重复计算）
        3. 返回需要实际处理的部分
        
        Args:
            prompt (List[int]): 新提示的 token 序列
        
        Returns:
            List[int]: 需要处理的提示后缀
        """
        cache_len = len(self.prompt_cache.tokens)
        prompt_len = len(prompt)
        # 计算公共前缀长度
        com_prefix_len = common_prefix_len(self.prompt_cache.tokens, prompt)

        # 确保至少保留一个 token
        com_prefix_len = min(com_prefix_len, len(prompt) - 1)

        # 情况1：模型变化或无公共前缀，重置缓存
        if (
            self.prompt_cache.model_key != self.model_provider.model_key
            or com_prefix_len == 0
        ):
            self.reset_prompt_cache(prompt)

        # 情况2：缓存是提示的前缀，处理后缀
        elif com_prefix_len == cache_len:
            logging.debug(
                f"*** Cache is prefix of prompt (cache_len: {cache_len}, prompt_len: {prompt_len}). Processing suffix. ***"
            )
            prompt = prompt[com_prefix_len:]  # 获取后缀
            self.prompt_cache.tokens.extend(prompt)  # 更新缓存

        # 情况3：公共前缀短于缓存长度，尝试修剪
        elif com_prefix_len < cache_len:
            logging.debug(
                f"*** Common prefix ({com_prefix_len}) shorter than cache ({cache_len}). Attempting trim. ***"
            )

            if can_trim_prompt_cache(self.prompt_cache.cache):
                # 可以修剪缓存
                num_to_trim = cache_len - com_prefix_len
                logging.debug(f"    Trimming {num_to_trim} tokens from cache.")
                trim_prompt_cache(self.prompt_cache.cache, num_to_trim)
                self.prompt_cache.tokens = self.prompt_cache.tokens[:com_prefix_len]
                prompt = prompt[com_prefix_len:]
                self.prompt_cache.tokens.extend(prompt)
            else:
                # 无法修剪，重置缓存
                logging.debug(f"    Cache cannot be trimmed. Resetting cache.")
                self.reset_prompt_cache(prompt)

        # 不应该到达的情况
        else:
            logging.error(
                f"Unexpected cache state: com_prefix_len ({com_prefix_len}) > cache_len ({cache_len}). Resetting cache."
            )
            self.reset_prompt_cache(prompt)

        logging.debug(f"Returning {len(prompt)} tokens for processing.")
        return prompt

    def handle_completion(
        self,
        prompt: List[int],
        stop_id_sequences: List[List[int]],
    ):
        """
        处理文本生成的核心函数
        
        这个函数负责实际的文本生成过程，包括：
        - 采样器配置
        - 流式生成循环
        - 停止条件检查
        - 响应格式化和发送
        
        Args:
            prompt (List[int]): 提示的 token 序列
            stop_id_sequences (List[List[int]]): 停止词的 token 序列列表
        """
        tokens = []
        finish_reason = "length"  # 默认结束原因
        stop_sequence_suffix = None
        
        if self.stream:
            self.end_headers()
            logging.debug(f"Starting stream:")
        else:
            logging.debug(f"Starting completion:")
            
        token_logprobs = []
        top_tokens = []

        # 获取缓存优化的提示
        prompt = self.get_prompt_cache(prompt)

        text = ""
        tic = time.perf_counter()
        
        # 创建采样器
        sampler = make_sampler(
            self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            xtc_probability=self.xtc_probability,
            xtc_threshold=self.xtc_threshold,
            xtc_special_tokens=[
                self.tokenizer.eos_token_id,
                self.tokenizer.encode("\n"),
            ],
        )
        
        # 创建 logit 处理器
        logits_processors = make_logits_processors(
            self.logit_bias,
            self.repetition_penalty,
            self.repetition_context_size,
        )

        # 工具调用相关变量
        tool_calls = []
        tool_text = ""
        in_tool_call = False
        segment = ""

        # 创建保活回调函数，用于在长提示处理期间发送保活信号
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

        # 开始流式生成
        for gen_response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=self.prompt_cache.cache,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=self.num_draft_tokens,
            prompt_progress_callback=keepalive_callback,
        ):
            logging.debug(gen_response.text)

            # 处理工具调用
            if (
                self.tokenizer.has_tool_calling
                and gen_response.text == self.tokenizer.tool_call_start
            ):
                in_tool_call = True
            elif in_tool_call:
                if gen_response.text == self.tokenizer.tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = ""
                    in_tool_call = False
                else:
                    tool_text += gen_response.text
            else:
                # 普通文本生成
                text += gen_response.text
                segment += gen_response.text
                
            token = gen_response.token
            logprobs = gen_response.logprobs
            tokens.append(token)
            self.prompt_cache.tokens.append(token)

            # 处理概率信息（如果需要）
            if self.logprobs > 0:
                sorted_indices = mx.argpartition(-logprobs, kth=self.logprobs - 1)
                top_indices = sorted_indices[: self.logprobs]
                top_logprobs = logprobs[top_indices]
                top_token_info = zip(top_indices.tolist(), top_logprobs.tolist())
                top_tokens.append(tuple(top_token_info))

            token_logprobs.append(logprobs[token].item())

            # 检查停止条件
            stop_condition = stopping_criteria(
                tokens, stop_id_sequences, self.tokenizer.eos_token_id
            )
            if stop_condition.stop_met:
                finish_reason = "stop"
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(
                        tokens[-stop_condition.trim_length :]
                    )
                    text = text[: -len(stop_sequence_suffix)]
                segment = ""
                break

            # 流式输出处理
            if self.stream and not in_tool_call:
                # 检查是否与停止序列有重叠，如果有则继续生成直到确定
                if any(
                    (
                        sequence_overlap(tokens, sequence)
                        for sequence in stop_id_sequences
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

        # 更新最终结束原因
        if gen_response.finish_reason is not None:
            finish_reason = gen_response.finish_reason

        # 输出性能统计
        logging.debug(f"Prompt: {gen_response.prompt_tps:.3f} tokens-per-sec")
        logging.debug(f"Generation: {gen_response.generation_tps:.3f} tokens-per-sec")
        logging.debug(f"Peak memory: {gen_response.peak_memory:.3f} GB")

        # 发送最终响应
        if self.stream:
            # 流式响应的最后一部分
            response = self.generate_response(
                segment, finish_reason, tool_calls=tool_calls
            )
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()
            
            # 如果需要，发送使用统计
            if self.stream_options is not None and self.stream_options["include_usage"]:
                original_prompt_length = (
                    len(self.prompt_cache.tokens) - len(tokens) + len(prompt)
                )
                response = self.completion_usage_response(
                    original_prompt_length, len(tokens)
                )
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()
                
            # 发送结束标记
            self.wfile.write("data: [DONE]\n\n".encode())
            self.wfile.flush()
        else:
            # 非流式响应
            response = self.generate_response(
                text,
                finish_reason,
                len(prompt),
                len(tokens),
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
                tool_calls=tool_calls,
            )
            response_json = json.dumps(response).encode()
            indent = "\t"  # 反斜杠不能在 f-strings 中
            logging.debug(f"Outgoing Response: {json.dumps(response, indent=indent)}")

            # 发送 Content-Length 头部
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
        生成使用统计响应（仅用于流式响应的统计信息）
        
        Args:
            prompt_token_count (Optional[int]): 提示 token 数量
            completion_token_count (Optional[int]): 完成 token 数量
        
        Returns:
            dict: 使用统计响应字典
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

    def handle_chat_completions(self) -> List[int]:
        """
        处理聊天补全请求
        
        这个函数处理 OpenAI 格式的聊天补全请求：
        - 解析消息列表
        - 应用聊天模板
        - 返回 token 化的提示
        
        Returns:
            List[int]: 提示的 token 序列
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # 确定响应类型
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"
        
        # 如果分词器有聊天模板，使用它
        if self.tokenizer.chat_template:
            messages = body["messages"]
            process_message_content(messages)  # 处理消息内容格式
            prompt = self.tokenizer.apply_chat_template(
                messages,
                body.get("tools") or None,  # 工具定义
                add_generation_prompt=True,  # 添加生成提示
                **self.model_provider.cli_args.chat_template_args,
            )
        else:
            # 使用简单的聊天转换
            prompt = convert_chat(body["messages"], body.get("role_mapping"))
            prompt = self.tokenizer.encode(prompt)

        return prompt

    def handle_text_completions(self) -> List[int]:
        """
        处理文本补全请求
        
        这个函数处理简单的文本补全请求：
        - 从请求中获取提示文本
        - 将其 token 化
        
        Returns:
            List[int]: 提示的 token 序列
        """
        # 确定响应类型
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"
        assert "prompt" in self.body, "Request did not contain a prompt"
        return self.tokenizer.encode(self.body["prompt"])

    def do_GET(self):
        """
        处理 GET 请求的主入口函数
        
        支持的端点：
        - /v1/models: 获取可用模型列表
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
        
        简单返回 OK 状态，用于负载均衡器和服务发现。
        """
        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write('{"status": "ok"}'.encode())
        self.wfile.flush()

    def handle_models_request(self):
        """
        处理模型列表请求
        
        扫描 Hugging Face 缓存目录，查找已下载的 MLX 兼容模型。
        返回符合 MLX-LM 格式的模型列表。
        """
        self._set_completion_headers(200)
        self.end_headers()

        # 定义 MLX-LM 模型的必需文件
        files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

        # 解析路径，支持模型过滤
        parts = self.path.split("/")
        filter_repo_id = None
        if len(parts) > 3:
            filter_repo_id = "/".join(parts[3:])

        # 判断是否为 MLX-LM 模型的函数
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


def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=HTTPServer,
    handler_class=APIHandler,
):
    """
    启动 HTTP 服务器
    
    这个函数创建并启动一个 HTTP 服务器，用于处理 API 请求。
    
    Args:
        host (str): 服务器主机地址
        port (int): 服务器端口
        model_provider (ModelProvider): 模型提供者实例
        server_class: HTTP 服务器类，默认 HTTPServer
        handler_class: 请求处理器类，默认 APIHandler
    """
    server_address = (host, port)
    prompt_cache = PromptCache()
    
    # 获取地址信息
    infos = socket.getaddrinfo(
        *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    server_class.address_family, _, _, _, server_address = next(iter(infos))
    
    # 创建服务器实例
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            model_provider,
            prompt_cache=prompt_cache,
            system_fingerprint=get_system_fingerprint(),
            *args,
            **kwargs,
        ),
    )
    
    # 安全警告
    warnings.warn(
        "mlx_lm.server is not recommended for production as "
        "it only implements basic security checks."
    )
    
    logging.info(f"Starting httpd at {host} on port {port}...")
    httpd.serve_forever()


def main():
    """
    主函数，解析命令行参数并启动服务器
    
    使用示例：
        python -m mlx_lm.server --model ~/.mlx/models/mistral-7b --host 0.0.0.0 --port 8080
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
    
    # 服务器相关参数
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
    
    # 安全和配置参数
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
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

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # 启动服务器
    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    # 弃用警告
    print(
        "Calling `python -m mlx_lm.server...` directly is deprecated."
        " Use `mlx_lm.server...` or `python -m mlx_lm server ...` instead."
    )
    main()
