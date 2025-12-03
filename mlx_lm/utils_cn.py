# Copyright © 2023-2024 Apple Inc.

# =============================================================================
# MLX 模型工具模块 - 详细中文注释版本
# =============================================================================
# 
# 本模块是 MLX (Machine Learning eXecution) 框架的核心工具模块，
# 提供了模型加载、量化、保存、上传等完整的功能链。
# 
# MLX 是苹果公司开发的机器学习框架，专门为苹果芯片优化，
# 支持在 Mac 设备上进行高效的模型训练和推理。
#
# 主要功能模块：
# 1. 模型加载和初始化
# 2. 模型量化（减少模型大小和提高推理速度）
# 3. 模型保存和分片处理
# 4. 与 Hugging Face Hub 的集成
# 5. 分词器加载和处理
# =============================================================================

import copy  # 用于深拷贝对象，防止修改原始数据
import glob  # 用于文件路径匹配，查找符合特定模式的文件
import importlib  # 动态导入模块，支持根据配置加载不同的模型
import inspect  # 用于检查对象的结构，如函数参数
import json  # JSON 数据格式的读写
import logging  # 日志记录，用于调试和错误追踪
import os  # 操作系统接口，处理文件路径和环境变量
import shutil  # 高级文件操作，如复制和删除
from pathlib import Path  # 现代化的路径处理库
from textwrap import dedent  # 文本格式化，去除多余缩进
from typing import (  # 类型注解，提高代码可读性和IDE支持
    Any,  # 任意类型
    Callable,  # 可调用对象（函数）
    Dict,  # 字典类型
    Optional,  # 可选类型
    Tuple,  # 元组类型
    Type,  # 类型对象
    Union,  # 联合类型
)

import mlx.core as mx  # MLX 核心模块，提供张量计算等基础功能
import mlx.nn as nn  # MLX 神经网络模块，提供层和模型基类

# =============================================================================
# 条件导入：根据环境变量选择模型下载源
# =============================================================================
# 检查是否设置环境变量 MLXLM_USE_MODEL_SCALE 为 true
# ModelScope 是阿里巴巴的模型分发平台，在中国大陆访问更稳定
if os.getenv("MLXLM_USE_MODELSCOPE", "False").lower() == "true":
    try:
        # 如果使用 ModelScope，则导入其下载功能
        from modelscope import snapshot_download
    except ImportError:
        # 如果没有安装 modelscope，提示用户安装
        raise ImportError("Run `pip install modelscope` to use ModelScope.")
else:
    # 默认使用 Hugging Face Hub 作为模型下载源
    from huggingface_hub import snapshot_download

# MLX 工具函数导入
from mlx.utils import tree_flatten, tree_map, tree_reduce
# tree_flatten: 将嵌套的树结构展平为键值对列表
# tree_map: 对树结构的每个叶子节点应用函数
# tree_reduce: 对树结构的所有叶子节点进行归约操作

# Hugging Face 分词器基类
from transformers import PreTrainedTokenizer

# 本地模块导入
from .tokenizer_utils import TokenizerWrapper, load_tokenizer
# TokenizerWrapper: 分词器包装类，提供统一的接口
# load_tokenizer: 加载分词器的函数

from .tuner.utils import dequantize as dequantize_model
from .tuner.utils import get_total_parameters, load_adapters
# dequantize_model: 模型反量化，将量化后的权重恢复为原始精度
# get_total_parameters: 计算模型总参数数量
# load_adapters: 加载 LoRA 适配器，用于模型微调

# =============================================================================
# 模型类型映射常量
# =============================================================================
# 这个字典用于将不同的模型类型映射到基础实现类型
# 这样可以避免重复实现相似的模型架构
MODEL_REMAPPING = {
    "mistral": "llama",  # Mistral 模型基于 Llama 架构
    "llava": "mistral3",  # LLaVA 多模态模型基于 Mistral3
    "phi-msft": "phixtral",  # 微软 Phi 模型映射到 Phixtral
    "falcon_mamba": "mamba",  # Falcon Mamba 基于 Mamba 架构
    "kimi_k2": "deepseek_v3",  # Kimi K2 映射到 DeepSeek V3
    "qwen2_5_vl": "qwen2_vl",  # Qwen2.5 VL 映射到 Qwen2 VL
}

# 最大文件大小限制（GB）
# 用于控制模型保存时每个分片的最大大小，避免单个文件过大
MAX_FILE_SIZE_GB = 5


def _get_classes(config: dict):
    """
    根据配置动态获取模型类和模型参数类
    
    这个函数是模型加载的核心，它根据配置文件中的 model_type
    动态导入对应的模型模块，并返回模型类和参数类。
    
    Args:
        config (dict): 模型配置字典，必须包含 "model_type" 键
        
    Returns:
        tuple: (Model类, ModelArgs类)
            - Model类: 神经网络模型的类定义
            - ModelArgs类: 模型参数配置的类定义
            
    Raises:
        ValueError: 当模型类型不支持时抛出异常
        
    Example:
        >>> config = {"model_type": "llama"}
        >>> ModelClass, ModelArgsClass = _get_classes(config)
        >>> model_args = ModelArgsClass.from_dict(config)
        >>> model = ModelClass(model_args)
    """
    # 从配置中获取模型类型
    model_type = config["model_type"]
    
    # 使用映射表转换模型类型，确保使用正确的基础实现
    # 例如：mistral -> llama，避免重复实现
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    
    try:
        # 动态导入对应的模型模块
        # 例如：mlx_lm.models.llama
        arch = importlib.import_module(f"mlx_lm.models.{model_type}")
    except ImportError:
        # 如果导入失败，说明该模型类型不支持
        msg = f"Model type {model_type} not supported."
        logging.error(msg)  # 记录错误日志
        raise ValueError(msg)
    
    # 返回模型类和参数类
    # 每个模型模块都应该定义这两个类
    return arch.Model, arch.ModelArgs


def compute_bits_per_weight(model):
    """
    计算模型的比特每权重比率
    
    这个指标用于衡量模型的压缩效率，比特每权重越低，
    说明模型压缩得越好，占用的存储空间越小。
    
    Args:
        model: MLX 神经网络模型
        
    Returns:
        float: 比特每权重的数值
        
    Example:
        >>> model, _ = load("llama-7b")
        >>> bpw = compute_bits_per_weight(model)
        >>> print(f"模型压缩效率: {bpw:.2f} bits/weight")
    """
    # 使用 tree_reduce 遍历模型的所有参数
    # 对每个 mx.array 类型的参数，计算其字节大小
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, 
        model, 
        0
    )
    
    # 获取模型总参数数量
    model_params = get_total_parameters(model)
    
    # 计算比特每权重：总字节数 * 8 / 参数数量
    return model_bytes * 8 / model_params


def _download(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    确保模型在本地可用，如果不存在则从 Hugging Face Hub 下载
    
    这个函数会首先检查指定路径是否存在模型文件，
    如果不存在，则从 Hugging Face Hub 或 ModelScope 下载模型。
    
    Args:
        path_or_hf_repo (str): 本地路径或 Hugging Face 仓库ID
            - 本地路径示例："/path/to/model"
            - HF仓库示例："meta-llama/Llama-2-7b-hf"
        revision (str, optional): 版本标识，可以是分支名、标签或提交哈希
            - 示例："main", "v1.0.0", "abc123def"
            
    Returns:
        Path: 模型在本地的路径对象
        
    Note:
        下载时会过滤文件类型，只下载必要的文件：
        - JSON配置文件
        - 模型权重文件（safetensors格式）
        - Python代码文件
        - 分词器相关文件
    """
    # 将路径转换为 Path 对象，便于跨平台操作
    model_path = Path(path_or_hf_repo)
    
    # 检查本地是否存在模型
    if not model_path.exists():
        # 如果本地不存在，则从远程下载
        model_path = Path(
            snapshot_download(
                path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",           # JSON配置文件
                    "model*.safetensors", # 模型权重文件
                    "*.py",             # Python代码文件
                    "tokenizer.model",   # SentencePiece分词器
                    "*.tiktoken",        # Tiktoken分词器
                    "tiktoken.model",    # Tiktoken模型文件
                    "*.txt",            # 文本文件
                    "*.jsonl",          # JSON Lines文件
                    "*.jinja",          # Jinja模板文件
                ],
            )
        )
    
    return model_path


def hf_repo_to_path(hf_repo):
    """
    将 Hugging Face 仓库ID转换为本地路径
    
    这个函数用于获取已下载的 Hugging Face 模型的本地路径，
    假设模型已经被下载到缓存目录中。
    
    Args:
        hf_repo (str): Hugging Face 仓库ID
        
    Returns:
        Path: 模型在本地缓存中的路径
    """
    return Path(snapshot_download(hf_repo, local_files_only=True))


def load_config(model_path: Path) -> dict:
    """
    加载模型配置文件
    
    配置文件通常是 config.json，包含了模型架构、超参数等信息。
    
    Args:
        model_path (Path): 模型目录路径
        
    Returns:
        dict: 模型配置字典
        
    Raises:
        FileNotFoundError: 当配置文件不存在时抛出异常
        
    Example:
        >>> config = load_config(Path("/path/to/model"))
        >>> print(f"模型类型: {config['model_type']}")
        >>> print(f"隐藏层大小: {config['hidden_size']}")
    """
    try:
        # 打开并解析 JSON 配置文件
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        # 如果配置文件不存在，记录错误并抛出异常
        logging.error(f"Config file not found in {model_path}")
        raise
    
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    strict: bool = True,
    model_config: dict = {},
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
) -> Tuple[nn.Module, dict]:
    """
    从指定路径加载并初始化模型
    
    这是模型加载的核心函数，负责：
    1. 加载配置文件
    2. 加载模型权重
    3. 初始化模型结构
    4. 应用量化设置
    5. 加载权重到模型中
    
    Args:
        model_path (Path): 模型目录路径
        lazy (bool): 是否懒加载模式
            - False: 立即将所有参数加载到内存（默认）
            - True: 参数在需要时才加载，节省内存
        strict (bool): 是否严格模式
            - True: 权重不匹配时抛出异常（默认）
            - False: 允许部分权重缺失或多余
        model_config (dict, optional): 额外的模型配置参数
            用于覆盖或补充配置文件中的设置
        get_model_classes (Callable, optional): 获取模型类的函数
            默认使用 _get_classes，可以自定义以支持特殊模型
            
    Returns:
        Tuple[nn.Module, dict]: (模型实例, 配置字典)
        
    Raises:
        FileNotFoundError: 当权重文件不存在时抛出异常
        ValueError: 当模型类无法找到或实例化时抛出异常
        
    Example:
        >>> model, config = load_model(Path("/path/to/model"))
        >>> print(f"模型参数数量: {get_total_parameters(model)}")
        >>> output = model(input_tokens)
    """
    # 加载模型配置文件
    config = load_config(model_path)
    
    # 应用额外的配置参数，会覆盖原有配置
    config.update(model_config)
    
    # 查找所有模型权重文件（safetensors格式）
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    
    # 如果没有找到权重文件且处于严格模式，抛出异常
    if not weight_files and strict:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    
    # 加载所有权重文件并合并到一个字典中
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    
    # 根据配置获取模型类和参数类
    model_class, model_args_class = get_model_classes(config=config)
    
    # 从配置字典创建模型参数对象
    model_args = model_args_class.from_dict(config)
    
    # 实例化模型
    model = model_class(model_args)
    
    # 如果模型有 sanitize 方法，调用它来清理权重
    # 某些模型需要重命名或调整权重以匹配当前实现
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    
    def _quantize(quantization):
        """
        内部函数：应用量化配置
        
        量化是将模型权重从高精度（如FP32）转换为低精度（如INT4）
        的过程，可以显著减少模型大小和提高推理速度。
        """
        def class_predicate(p, m):
            """
            判断是否对某个参数进行量化
            
            Args:
                p: 参数路径（如 "model.layers.0.weight"）
                m: 对应的模块
                
            Returns:
                bool or dict: 是否量化，或量化参数
            """
            # 检查是否有针对特定层的量化配置
            if p in config["quantization"]:
                return config["quantization"][p]
            # 检查模块是否支持量化
            if not hasattr(m, "to_quantized"):
                return False
            # 检查是否有对应的缩放因子
            return f"{p}.scales" in weights
        
        # 应用量化
        nn.quantize(
            model,
            group_size=quantization["group_size"],  # 量化组大小
            bits=quantization["bits"],              # 量化位数
            mode=quantization.get("mode", "affine"), # 量化模式
            class_predicate=class_predicate,        # 量化判断函数
        )
    
    # 检查是否有量化配置
    if (quantization := config.get("quantization", None)) is not None:
        _quantize(quantization)
    elif quantization_config := config.get("quantization_config", False):
        # 处理旧版本的量化配置格式
        quant_method = quantization_config["quant_method"]
        if quant_method == "bitnet":
            # BitNet 量化方法
            from .models.bitlinear_layers import bitnet_quantize
            model = bitnet_quantize(model, quantization_config)
        elif quant_method == "mxfp4":
            # MXFP4 量化方法
            quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
    
    # 将权重加载到模型中
    model.load_weights(list(weights.items()), strict=strict)
    
    # 如果不是懒加载模式，立即将所有参数加载到内存
    if not lazy:
        mx.eval(model.parameters())
    
    # 设置模型为评估模式（关闭dropout等训练时特有的层）
    model.eval()
    
    return model, config


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: str = None,
) -> Union[
    Tuple[nn.Module, TokenizerWrapper],
    Tuple[nn.Module, TokenizerWrapper, Dict[str, Any]],
]:
    """
    统一的模型和分词器加载接口
    
    这是最常用的加载函数，它会：
    1. 下载或验证模型文件
    2. 加载模型和分词器
    3. 可选地加载 LoRA 适配器
    4. 返回可直接使用的模型和分词器
    
    Args:
        path_or_hf_repo (str): 本地路径或 Hugging Face 仓库ID
        tokenizer_config (dict, optional): 分词器配置参数
            - 可包含特殊token、最大长度等设置
        model_config (dict, optional): 模型配置参数
            - 用于覆盖模型配置文件中的设置
        adapter_path (str, optional): LoRA 适配器路径
            - 如果提供，会将适配器加载到模型上
        lazy (bool): 是否懒加载模式，默认为 False
        return_config (bool): 是否返回配置字典，默认为 False
        revision (str, optional): 模型版本，可以是分支名或标签
        
    Returns:
        Union[Tuple[nn.Module, TokenizerWrapper], 
              Tuple[nn.Module, TokenizerWrapper, Dict[str, Any]]]:
            - 如果 return_config=False: (模型, 分词器)
            - 如果 return_config=True: (模型, 分词器, 配置)
            
    Example:
        >>> # 基础用法
        >>> model, tokenizer = load("meta-llama/Llama-2-7b-hf")
        >>> 
        >>> # 带配置的用法
        >>> model, tokenizer = load(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     model_config={"use_cache": False}
        ... )
        >>> 
        >>> # 带 LoRA 适配器的用法
        >>> model, tokenizer = load(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     adapter_path="./my-lora-adapter"
        ... )
    """
    # 确保模型文件在本地可用
    model_path = _download(path_or_hf_repo, revision=revision)
    
    # 加载模型
    model, config = load_model(model_path, lazy, model_config=model_config)
    
    # 如果提供了适配器路径，加载 LoRA 适配器
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()  # 设置为评估模式
    
    # 加载分词器
    tokenizer = load_tokenizer(
        model_path, 
        tokenizer_config, 
        eos_token_ids=config.get("eos_token_id", None)
    )
    
    # 根据参数决定是否返回配置
    if return_config:
        return model, tokenizer, config
    else:
        return model, tokenizer


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    将模型权重分割成多个小文件
    
    这个函数用于处理大模型，将权重分割成多个文件，
    每个文件不超过指定大小，便于传输和存储。
    
    Args:
        weights (dict): 模型权重字典，键为参数名，值为权重张量
        max_file_size_gb (int): 每个分片的最大大小（GB），默认为 5GB
        
    Returns:
        list: 权重分片列表，每个元素是一个权重字典
        
    Example:
        >>> weights = {"layer1.weight": tensor1, "layer2.weight": tensor2, ...}
        >>> shards = make_shards(weights, max_file_size_gb=2)
        >>> print(f"分割成 {len(shards)} 个分片")
    """
    # 将GB转换为字节
    max_file_size_bytes = max_file_size_gb << 30  # 1 GB = 2^30 bytes
    
    shards = []  # 存储所有分片
    shard, shard_size = {}, 0  # 当前分片和当前分片大小
    
    # 遍历所有权重
    for k, v in weights.items():
        # 检查添加当前权重是否会超过大小限制
        if shard_size + v.nbytes > max_file_size_bytes:
            # 如果超过限制，保存当前分片并开始新分片
            shards.append(shard)
            shard, shard_size = {}, 0
        
        # 将权重添加到当前分片
        shard[k] = v
        shard_size += v.nbytes
    
    # 添加最后一个分片
    shards.append(shard)
    
    return shards


def create_model_card(path: Union[str, Path], hf_path: Union[str, Path, None]):
    """
    创建模型卡片文件（README.md）
    
    模型卡片是模型的说明文档，包含模型的基本信息、使用方法等。
    
    Args:
        path (Union[str, Path]): 模型的本地路径
        hf_path (Union[str, Path, None]): 原始 Hugging Face 模型路径
        
    Example:
        >>> create_model_card("./my-model", "meta-llama/Llama-2-7b-hf")
        >>> # 这会在 ./my-model/README.md 创建模型卡片
    """
    from huggingface_hub import ModelCard, ModelCardData
    
    if hf_path is None:
        # 如果没有原始模型，创建一个基础的模型卡片
        card = ModelCard.from_template(ModelCardData(language="en"))
    else:
        # 加载原始模型的卡片
        card = ModelCard.load(hf_path)
    
    # 设置 MLX 相关的元数据
    card.data.library_name = "mlx"
    card.data.pipeline_tag = "text-generation"
    
    # 添加 MLX 标签
    if card.data.tags is None:
        card.data.tags = ["mlx"]
    elif "mlx" not in card.data.tags:
        card.data.tags += ["mlx"]
    
    # 设置基础模型信息
    if hf_path is not None:
        card.data.base_model = str(hf_path)
    
    # 清空原有内容（后面会重新生成）
    card.text = ""
    
    # 保存模型卡片
    card.save(os.path.join(path, "README.md"))


def upload_to_hub(path: str, upload_repo: str):
    """
    将模型上传到 Hugging Face Hub
    
    这个函数会将本地模型上传到 Hugging Face Hub，
    包括模型权重、配置文件、分词器和说明文档。
    
    Args:
        path (str): 本地模型路径
        upload_repo (str): 目标仓库名称（格式：username/model-name）
        
    Example:
        >>> upload_to_hub("./my-llama-model", "myusername/my-llama-mlx")
        >>> # 模型将被上传到 https://huggingface.co/myusername/my-llama-mlx
    """
    from huggingface_hub import HfApi, ModelCard, logging
    
    from . import __version__
    
    # 设置日志级别为 INFO，显示上传进度
    logging.set_verbosity_info()
    
    # 加载现有模型卡片
    card_path = Path(path) / "README.md"
    card = ModelCard.load(card_path)
    
    hf_path = card.data.base_model
    
    # 生成模型来源信息
    if hf_path is not None:
        provenance = f"""
        This model [{upload_repo}](https://huggingface.co/{upload_repo}) was
        converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path})
        using mlx-lm version **{__version__}**.
        """
    else:
        provenance = ""
    
    # 生成新的模型卡片内容
    card.text = dedent(
        f"""
        # {upload_repo}
        {provenance}
        ## Use with mlx

        ```bash
        pip install mlx-lm
        ```

        ```python
        from mlx_lm import load, generate

        model, tokenizer = load("{upload_repo}")

        prompt = "hello"

        if tokenizer.chat_template is not None:
            messages = [{{"role": "user", "content": prompt}}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

        response = generate(model, tokenizer, prompt=prompt, verbose=True)
        ```
        """
    )
    
    # 保存更新后的模型卡片
    card.save(card_path)
    
    # 初始化 Hugging Face API
    api = HfApi()
    
    # 创建仓库（如果已存在则不报错）
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    
    # 上传整个模型文件夹
    api.upload_large_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
    )
    
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_model(
    save_path: Union[str, Path],
    model: nn.Module,
    *,
    donate_model: bool = False,
) -> None:
    """
    保存模型权重和元数据索引到指定目录
    
    这个函数会：
    1. 将模型权重分割成适当大小的文件
    2. 生成权重索引文件
    3. 保存所有文件到指定目录
    
    Args:
        save_path (Union[str, Path]): 保存路径
        model (nn.Module): 要保存的模型
        donate_model (bool): 是否"捐赠"模型内存
            - True: 保存后清空模型内存，节省内存使用
            - False: 保留模型在内存中
            
    Example:
        >>> model, _ = load("meta-llama/Llama-2-7b-hf")
        >>> save_model("./my-saved-model", model, donate_model=True)
        >>> # 模型将被保存到 ./my-saved-model/ 目录
    """
    # 确保保存路径是 Path 对象
    if isinstance(save_path, str):
        save_path = Path(save_path)
    
    # 创建保存目录（如果不存在）
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 将模型参数展平为字典
    weights = dict(tree_flatten(model.parameters()))
    
    # 将权重分割成多个文件
    shards = make_shards(weights)
    shards_count = len(shards)
    
    # 根据分片数量决定文件名格式
    if shards_count > 1:
        # 多分片：使用编号格式
        shard_file_format = "model-{:05d}-of-{:05d}.safetensors"
    else:
        # 单分片：使用简单名称
        shard_file_format = "model.safetensors"
    
    # 计算总大小和总参数数量
    total_size = sum(v.nbytes for v in weights.values())
    index_data = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": get_total_parameters(model),
        },
        "weight_map": {},  # 权重名到文件名的映射
    }
    
    # 如果需要"捐赠"模型内存，将模型参数替换为空数组
    if donate_model:
        model.update(tree_map(lambda _: mx.array([]), model.parameters()))
    
    # 清空权重字典引用，释放内存
    weights.clear()
    del weights
    
    # 保存每个分片
    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None  # 立即释放当前分片的引用
        
        # 生成分片文件名
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name
        
        # 保存分片为 safetensors 格式
        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})
        
        # 更新权重映射表
        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        
        # 释放分片内存
        del shard
    
    # 对权重映射进行排序，便于阅读
    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }
    
    # 保存权重索引文件
    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def quantize_model(
    model: nn.Module,
    config: dict,
    group_size: int,
    bits: int,
    mode: str = "affine",
    quant_predicate: Optional[Callable[[str, nn.Module], Union[bool, dict]]] = None,
) -> Tuple[nn.Module, dict]:
    """
    对模型权重应用量化
    
    量化是一种模型压缩技术，通过减少权重的表示精度来：
    1. 减少模型存储空间
    2. 提高推理速度（在某些硬件上）
    3. 降低内存使用
    
    Args:
        model (nn.Module): 要量化的模型
        config (dict): 模型配置字典
        group_size (int): 量化组大小
            - 常见值：32, 64, 128
            - 越大压缩率越高，但可能影响精度
        bits (int): 每个权重的位数
            - 常见值：4, 8
            - 越小模型越小，但精度损失越大
        mode (str): 量化模式
            - "affine": 仿射量化（默认）
            - "mxfp4": MX FP4 量化
        quant_predicate (Callable, optional): 量化判断函数
            用于自定义哪些层应该量化，如何量化
            
    Returns:
        Tuple[nn.Module, dict]: (量化后的模型, 更新后的配置)
        
    Example:
        >>> model, config = load("meta-llama/Llama-2-7b-hf")
        >>> quantized_model, quantized_config = quantize_model(
        ...     model, config, group_size=64, bits=4
        ... )
        >>> print(f"量化后比特每权重: {compute_bits_per_weight(quantized_model):.2f}")
    """
    # 深拷贝配置，避免修改原始配置
    quantized_config = copy.deepcopy(config)
    
    # 获取量化判断函数
    quant_predicate = quant_predicate or getattr(model, "quant_predicate", None)
    
    # 基础量化参数
    quant_params = {"group_size": group_size, "bits": bits, "mode": mode}
    
    # 检查是否已经有部分量化配置
    if "quantization" in quantized_config:
        fine_grained_config = True  # 已经有量化配置，使用细粒度控制
    else:
        fine_grained_config = False  # 没有量化配置，使用统一设置
        quantized_config["quantization"] = quant_params
    
    def wrapped_predicate(path, module):
        """
        包装的量化判断函数
        
        这个函数决定是否对特定模块进行量化，
        以及如何量化。
        """
        # 检查模块是否支持量化
        if not hasattr(module, "to_quantized"):
            return False
        
        # 检查权重维度是否能被组大小整除
        if module.weight.shape[-1] % group_size != 0:
            return False
        
        # 默认进行量化
        bool_or_params = True
        
        # 如果有自定义判断函数，调用它
        if quant_predicate is not None:
            bool_or_params = quant_predicate(path, module)
        
        # 处理自定义量化参数
        if isinstance(bool_or_params, dict):
            # 如果返回的是字典，保存到配置中
            quantized_config["quantization"][path] = bool_or_params
        elif fine_grained_config and bool_or_params:
            # 如果是细粒度配置且需要量化，使用默认参数
            quantized_config["quantization"][path] = quant_params
        
        return bool_or_params
    
    # 应用量化
    nn.quantize(
        model,
        group_size,
        bits,
        mode=mode,
        class_predicate=wrapped_predicate,
    )
    
    # 为了兼容 HF 模型树，同时设置 quantization_config
    quantized_config["quantization_config"] = quantized_config["quantization"]
    
    # 计算并打印量化后的比特每权重
    bpw = compute_bits_per_weight(model)
    print(f"[INFO] Quantized model with {bpw:.3f} bits per weight.")
    
    return model, quantized_config


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """
    保存模型配置到指定路径
    
    这个函数会清理配置中的无用键，对配置进行排序，
    然后保存为 JSON 格式。
    
    Args:
        config (dict): 模型配置字典
        config_path (Union[str, Path]): 配置文件保存路径
        
    Example:
        >>> config = {"model_type": "llama", "hidden_size": 4096}
        >>> save_config(config, "./my-model/config.json")
    """
    # 清理一些不需要的键
    config.pop("_name_or_path", None)  # HF 模型路径
    config.pop("vision_config", None)  # 视觉模型配置（如果有的话）
    
    # 确保 quantization_config 存在
    if "quantization" in config:
        config["quantization_config"] = config["quantization"]
    
    # 对配置进行排序，提高可读性
    config = dict(sorted(config.items()))
    
    # 写入配置文件
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def save(
    dst_path: Union[str, Path],
    src_path_or_repo: Union[str, Path],
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    config: Dict[str, Any],
    donate_model: bool = True,
):
    """
    完整保存模型、分词器和配置
    
    这是最完整的保存函数，会保存：
    1. 模型权重（分片）
    2. 模型配置
    3. 分词器文件
    4. 其他必要文件
    5. 模型卡片
    
    Args:
        dst_path (Union[str, Path]): 目标保存路径
        src_path_or_repo (Union[str, Path]): 源路径或仓库ID
        model (nn.Module): 要保存的模型
        tokenizer (TokenizerWrapper): 分词器
        config (Dict[str, Any]): 模型配置
        donate_model (bool): 是否"捐赠"模型内存，默认为 True
        
    Example:
        >>> model, tokenizer, config = load("meta-llama/Llama-2-7b-hf", return_config=True)
        >>> save("./my-model", "meta-llama/Llama-2-7b-hf", model, tokenizer, config)
    """
    # 处理源路径
    src_path = Path(src_path_or_repo)
    if not src_path.exists():
        # 如果本地不存在，假设是 HF 仓库ID
        hf_repo = src_path_or_repo
        src_path = hf_repo_to_path(hf_repo)
    else:
        hf_repo = None
    
    # 确保目标路径是 Path 对象
    dst_path = Path(dst_path)
    
    # 保存模型
    save_model(dst_path, model, donate_model=True)
    
    # 保存配置
    save_config(config, config_path=dst_path / "config.json")
    
    # 保存分词器
    tokenizer.save_pretrained(dst_path)
    
    # 复制其他必要的文件
    for p in ["*.py", "generation_config.json"]:
        for file in glob.glob(str(src_path / p)):
            shutil.copy(file, dst_path)
    
    # 创建模型卡片
    create_model_card(dst_path, hf_repo)


def common_prefix_len(list1, list2):
    """
    计算两个列表的公共前缀长度
    
    这个函数用于比较两个序列，找出它们从开头开始相同的部分。
    在模型处理中常用于比较 token 序列或路径。
    
    Args:
        list1: 第一个字符串列表
        list2: 第二个字符串列表
        
    Returns:
        int: 公共前缀的长度。如果列表为空或第一个元素不匹配，返回 0
        
    Example:
        >>> common_prefix_len([1, 2, 3, 4], [1, 2, 5, 6])
        >>> 2  # 前两个元素相同
        >>> 
        >>> common_prefix_len(["a", "b"], ["x", "y"])
        >>> 0  # 第一个元素就不同
        >>> 
        >>> common_prefix_len([1, 2], [1, 2, 3, 4])
        >>> 2  # 较短列表的长度
    """
    # 确定最大可能的公共前缀长度
    min_len = min(len(list1), len(list2))
    
    # 逐个元素比较
    for i in range(min_len):
        if list1[i] != list2[i]:
            # 发现不匹配的元素，返回当前索引
            return i
    
    # 在较短列表的长度内都没有不匹配，返回较短列表的长度
    return min_len


def does_model_support_input_embeddings(model: nn.Module) -> bool:
    """
    检查模型是否在调用签名中支持 input_embeddings 参数
    
    某些高级用例需要直接传入词嵌入向量而不是 token ID，
    这个函数检查模型是否支持这种用法。
    
    Args:
        model (nn.Module): 要检查的模型
        
    Returns:
        bool: 如果模型支持 input_embeddings 返回 True，否则返回 False
        
    Example:
        >>> model, _ = load("meta-llama/Llama-2-7b-hf")
        >>> if does_model_support_input_embeddings(model):
        ...     # 可以直接使用嵌入向量
        ...     embeddings = mx.random.normal((1, 10, 4096))
        ...     output = model(input_embeddings=embeddings)
        ... else:
        ...     # 需要使用 token ID
        ...     tokens = mx.array([[1, 2, 3, 4, 5]])
        ...     output = model(tokens)
    """
    try:
        # 获取模型 __call__ 方法的签名
        signature = inspect.signature(model.__call__)
        
        # 检查参数中是否包含 input_embeddings
        return "input_embeddings" in signature.parameters
    except (ValueError, TypeError):
        # 如果无法获取签名（某些特殊情况），返回 False
        return False
