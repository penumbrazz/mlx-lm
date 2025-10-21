# Copyright © 2023-2024 Apple Inc.
# 本文件是在原始 `generate.py` 基础上，为初学者添加了详细的中文注释版本。
# 通过对关键步骤逐段讲解，帮助理解生成式推理流程的整体结构与细节。

# ---------------------- #
# 标准库相关的导入
# ---------------------- #
# argparse: 解析命令行参数，方便从终端调用脚本。
# contextlib: 提供上下文管理器工具，便于使用 `with` 语句管理资源。
# functools: 提供函数式编程辅助工具，这里用于局部函数配置。
# json、sys、time: 分别负责 JSON 解析、系统交互和时间统计。
import argparse
import contextlib
import functools
import json
import sys
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

# ---------------------- #
# 第三方库导入
# ---------------------- #
# mlx.core / mlx.nn: MLX 提供的核心计算与神经网络模块。
# tree_reduce: 用于对嵌套结构中的数组做累计操作。
# PreTrainedTokenizer: transformers 提供的预训练分词器基类。
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from transformers import PreTrainedTokenizer

# ---------------------- #
# 项目内部模块导入
# ---------------------- #
# cache 模块提供 KV 缓存的创建与操作。
from .models import cache
from .models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    load_prompt_cache,
)
from .sample_utils import make_sampler  # 负责采样逻辑的构建（温度、top-p 等）
from .tokenizer_utils import TokenizerWrapper  # 对 tokenizer 做统一封装，便于处理
from .utils import does_model_support_input_embeddings, load  # 工具函数

# ---------------------- #
# 默认配置常量
# ---------------------- #
# 这些默认值决定了未显式传参时的生成行为，例如默认提示词、采样温度、最大生成长度等。
# 在阅读或修改时，可以将它们视为“兜底配置”。
DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.0
DEFAULT_TOP_K = 0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_MIN_TOKENS_TO_KEEP = 1
DEFAULT_SEED = None
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_QUANTIZED_KV_START = 5000


def str2bool(string):
    """将字符串形式的布尔值转换为实际布尔值。"""
    # 约定：只要不是代表 False 的字符，就视为 True，以提高容错性。
    return string.lower() not in ["false", "f"]


def setup_arg_parser():
    """创建并返回命令行参数解析器。"""
    # 通过 argparse 构建命令行参数解释器，允许用户灵活定制模型加载与生成行为。
    # 每一个 add_argument 调用都为脚本提供了一个可配置选项，下方注释说明其用途。
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    # ---- 模型与 Tokenizer 相关配置 ----
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--extra-eos-token",
        type=str,
        default=(),
        nargs="+",
        help="Add tokens in the list of eos tokens that stop generation.",
    )
    # ---- 输入提示、对话模板相关配置 ----
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--prefill-response",
        default=None,
        help="Prefill response to be used for the chat template",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    # ---- 采样控制参数（影响生成多样性与确定性）----
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--min-p", type=float, default=DEFAULT_MIN_P, help="Sampling min-p"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Sampling top-k"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0,
        help="Thresold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--min-tokens-to-keep",
        type=int,
        default=DEFAULT_MIN_TOKENS_TO_KEEP,
        help="Minimum tokens to keep for min-p sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--chat-template-config",
        help="Additional config for `apply_chat_template`. Should be a dictionary of"
        " string keys to values represented as a JSON decodable string.",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Log verbose output when 'True' or 'T' or only print the response when 'False' or 'F'",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--prompt-cache-file",
        type=str,
        default=None,
        help="A file containing saved KV caches to avoid recomputing them",
    )
    # ---- KV 缓存相关设置，控制显存占用与量化 ----
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. "
        "Defaults to no quantization.",
        default=None,
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        help="Group size for KV cache quantization.",
        default=64,
    )
    parser.add_argument(
        "--quantized-kv-start",
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards.",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
    )
    # ---- 推断加速相关参数（草稿模型用于推测式解码）----
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
    return parser


# A stream on the default device just for generation
# 专门用于推理的异步流，保证生成过程与主线程解耦，提高吞吐。
generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    上下文管理器：在限定时间内调整 Metal 的 wired memory 上限。

    注意：异步推理执行期间不应修改 wired limit；如果可能存在异步任务，请传入需要同步的流，
    从而在退出上下文管理器前保证安全同步。
    """
    # 这个上下文管理器会根据模型大小临时调整 Metal 的“wired memory”限制，
    # 以避免加载大模型时触发性能或内存问题。退出上下文时会恢复旧值。
    if not mx.metal.is_available():
        try:
            yield
        finally:
            pass
    else:
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
        )
        # 统计模型中所有参数的字节数，用于判断是否接近推荐的显存限制。
        max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
        if model_bytes > 0.9 * max_rec_size:
            model_mb = model_bytes // 2**20
            max_rec_mb = max_rec_size // 2**20
            print(
                f"[WARNING] Generating with a model that requires {model_mb} MB "
                f"which is close to the maximum recommended size of {max_rec_mb} "
                "MB. This can be slow. See the documentation for possible work-arounds: "
                "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
            )
        old_limit = mx.set_wired_limit(max_rec_size)
        try:
            yield
        finally:
            # 退出上下文时，确保所有流同步完成，再恢复之前的 wired limit。
            if streams is not None:
                for s in streams:
                    mx.synchronize(s)
            else:
                mx.synchronize()
            mx.set_wired_limit(old_limit)


@dataclass
class GenerationResponse:
    """
    :func:`stream_generate` 的单次输出结构。

    参数说明：
        text (str): 解码得到的新增文本片段，可能为空字符串。
        token (int): 最新生成的 token ID。
        from_draft (bool): 是否由草稿模型生成该 token。
        logprobs (mx.array): 对应 token 的对数概率向量。
        prompt_tokens (int): 提示词的 token 数量。
        prompt_tps (float): 处理提示词阶段的 tokens-per-second。
        generation_tokens (int): 已生成的 token 总量。
        generation_tps (float): 生成阶段的 tokens-per-second。
        peak_memory (float): 当前峰值显存占用（GB）。
        finish_reason (str): 返回此次响应的原因，可为 "length"、"stop" 或 `None`。
    """

    text: str
    token: int
    logprobs: mx.array
    from_draft: bool
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None

    # dataclass 会自动生成初始化、比较等方法。这里封装了单次生成步骤的完整信息，
    # 包括文本片段、对应 token、采样的 log 概率以及性能指标，便于上层逻辑使用。


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    """根据配置选择性地对 KV 缓存进行量化，以降低显存占用。"""
    # kv_bits 为 None 表示不启用量化，直接返回。
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        # 只对支持量化并且已经累积到指定步数之后的缓存进行转换。
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int], int]] = None,
    input_embeddings: Optional[mx.array] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    基于给定提示词逐步产生 token ID 的生成器。

    参数说明：
        prompt (mx.array): 输入提示词序列。
        model (nn.Module): 用于生成的模型。
        max_tokens (int): 最多生成的 token 数。若设为 ``-1`` 则视为无限生成，默认 ``256``。
        sampler (Callable[[mx.array], mx.array], 可选): 从对数概率分布中采样 token 的函数。默认 ``None`` 表示取 argmax。
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], 可选):
            对 logits 进行二次处理的函数列表，输入为历史 token 与当前 logits，输出处理后的 logits。
        max_kv_size (int, 可选): KV 缓存的最大容量，超出后（除前 4 个 token 外）会被覆盖。
        prompt_cache (List[Any], 可选): 预先计算好的提示缓存，若提供会在原地更新。
        prefill_step_size (int): 处理提示词时的分块大小。
        kv_bits (int, 可选): KV 缓存量化的位宽；为 ``None`` 时不进行量化。
        kv_group_size (int): KV 量化时的分组大小，默认 ``64``。
        quantized_kv_start (int): 当启用量化时，从第几个 token 开始对缓存量化，默认 ``0``。
        prompt_progress_callback (Callable[[int], int], 可选): 回调函数，传入已处理 token 数与总提示长度。
        input_embeddings (mx.array, 可选): 直接提供输入嵌入，可替代或配合提示 token 使用。

    生成值：
        Tuple[mx.array, mx.array]: 当前生成的 token 以及对应的对数概率向量。
    """
    # 作为底层 token 生成器，这个函数负责：
    # 1. 预处理提示词并填充 KV 缓存；
    # 2. 循环调用模型获取 logits；
    # 3. 对 logits 做概率归一化后，通过采样策略拿到下一个 token。
    if input_embeddings is not None:
        if not does_model_support_input_embeddings(model):
            raise ValueError("Model does not support input embeddings.")
        elif len(prompt) > 0 and len(prompt) != len(input_embeddings):
            raise ValueError(
                f"When providing input_embeddings, their sequence length ({len(input_embeddings)}) "
                f"must match the sequence length of the prompt ({len(prompt)}), or the "
                "prompt must be empty."
            )
    elif len(prompt) == 0:
        raise ValueError(
            "Either input_embeddings or prompt (or both) must be provided."
        )

    tokens = None

    # Create the KV cache for generation
    # KV 缓存用于保存注意力层中的 Key/Value，避免每次重复计算历史 token。
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        # 将 tokens（以及可选的输入嵌入）送入模型，并复用同一份缓存。
        if input_embeddings is not None:
            return model(
                input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
            )
        else:
            return model(input_tokens, cache=prompt_cache)

    def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array] = None):
        nonlocal tokens

        with mx.stream(generation_stream):
            # 单 token 推理置于独立的 stream，以便异步执行。
            logits = _model_call(
                input_tokens=input_tokens[None],
                input_embeddings=(
                    input_embeddings[None] if input_embeddings is not None else None
                ),
            )

            logits = logits[:, -1, :]

            if logits_processors and len(input_tokens) > 0:
                # 对历史 token 进行拼接后传入处理器，可以做重复惩罚、温度缩放等。
                tokens = (
                    mx.concat([tokens, input_tokens])
                    if tokens is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            # 根据配置对缓存进行量化。
            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    with mx.stream(generation_stream):
        total_prompt_tokens = (
            len(input_embeddings) if input_embeddings is not None else len(prompt)
        )
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        while total_prompt_tokens - prompt_processed_tokens > 1:
            # 预填充阶段：批量送入若干 token 构建缓存，可显著提速长提示的处理。
            n_to_process = min(prefill_step_size, prompt.size - 1)
            _model_call(
                input_tokens=prompt[:n_to_process][None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            # 让缓存保持最新状态，并在 Metal 上执行必要的同步。
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            input_embeddings = (
                input_embeddings[n_to_process:]
                if input_embeddings is not None
                else input_embeddings
            )
            mx.clear_cache()
            # 清理临时缓存，避免显存占用累积。

        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)

    mx.async_eval(y, logprobs)
    n = 0
    while True:
        # 每轮循环尝试生成下一个 token；为避免阻塞，调用 _step 后立即异步执行。
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        # 这里 yield 出的是当前 token 及其 log 概率，供上层消费。
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1


def speculative_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    *,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
    """
    使用主模型与草稿模型进行推测式解码的生成器。

    参数说明：
        prompt (mx.array): 输入提示词序列。
        model (nn.Module): 主模型，用于最终确认 token。
        draft_model (nn.Module): 草稿模型，用于提前生成候选 token。
        num_draft_tokens (int, 可选): 每轮草稿模型尝试生成的 token 数，默认 ``2``。
        max_tokens (int): 最多生成的 token 数，设为 ``-1`` 表示无限生成，默认 ``256``。
        sampler (Callable[[mx.array], mx.array], 可选): 从对数概率中采样 token 的函数。
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], 可选):
            对 logits 进行二次处理的函数列表。
        prompt_cache (List[Any], 可选): 预计算的提示缓存，必须支持裁剪以便回滚。
        prefill_step_size (int): 预填充阶段的分块大小。
        kv_bits (int, 可选): KV 缓存量化位宽，``None`` 表示不量化。
        kv_group_size (int): KV 量化的分组大小，默认 ``64``。
        quantized_kv_start (int): 启用量化后从哪个步骤开始量化缓存，默认 ``0``。

    生成值：
        Tuple[mx.array, mx.array, bool]: 包含生成 token、对应对数概率，以及是否来自草稿模型的标记。
    """
    # 该生成器实现“推测式解码”（speculative decoding）：
    # 使用草稿模型提前生成多个候选 token，再由主模型验证并接受，从而提升速度。

    y = prompt.astype(mx.uint32)
    prev_tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        model_cache = cache.make_prompt_cache(model)
        draft_cache = cache.make_prompt_cache(draft_model)
    else:
        model_cache = prompt_cache[: len(model.layers)]
        draft_cache = prompt_cache[len(model.layers) :]
    # 注意：如果传入共享缓存，需要根据模型层数切分给主模型与草稿模型。

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _process_and_sample(tokens, logits):
        # 统一封装 logits 的后处理与采样逻辑，便于主模型与草稿模型复用。
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs

    def _step(model, cache, y, n_predict=1):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=cache)
            logits = logits[:, -n_predict:, :]

            # 对缓存做量化，并根据需要逐 token 调用处理器。
            quantize_cache_fn(cache)
            if logits_processors:
                nonlocal prev_tokens
                out_y, out_logprobs = [], []
                if n_predict > 1:
                    y = y[: -(n_predict - 1)]
                for i in range(n_predict):
                    prev_tokens = (
                        mx.concat([prev_tokens, y]) if prev_tokens is not None else y
                    )
                    y, logprobs = _process_and_sample(prev_tokens, logits[:, i, :])
                    out_y.append(y)
                    out_logprobs.append(logprobs)
                return mx.concatenate(out_y, axis=0), mx.concatenate(
                    out_logprobs, axis=0
                )
            else:
                return _process_and_sample(None, logits.squeeze(0))

    def _prefill(model, cache, y):
        # 预填充阶段与普通生成类似，分块送入提示词并同步缓存。
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=cache)
            quantize_cache_fn(cache)
            mx.eval([c.state for c in cache])
            y = y[prefill_step_size:]
            mx.clear_cache()
        return y

    def _rewind_cache(num_draft, num_accept):
        # 若部分草稿 token 被拒绝，需要把缓存状态回滚到对应位置。
        cache.trim_prompt_cache(model_cache, num_draft - num_accept)
        cache.trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))

    def _draft_generate(y, num_draft):
        if num_draft == 0:
            return mx.array([], mx.uint32)
        ys = []
        for _ in range(num_draft):
            # 草稿模型不断前推，生成候选 token，随后交由主模型验证。
            y, _ = _step(draft_model, draft_cache, y)
            mx.async_eval(y)
            ys.append(y)
        return mx.concatenate(ys)

    with mx.stream(generation_stream):
        draft_y = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

    ntoks = 0
    # Set these so the finally block doesn't raise
    num_draft = 0
    n = 0
    try:
        while True:
            num_draft = min(max_tokens - ntoks, num_draft_tokens)
            # 草稿模型尝试一次生成 num_draft 个 token。
            draft_tokens = _draft_generate(draft_y, num_draft)
            if prev_tokens is not None:
                prev_tokens = prev_tokens[: prev_tokens.size - y.size - num_draft + 1]
            y = mx.concatenate([y, draft_tokens])
            # 主模型对草稿 token + 下一个真实 token 进行验证。
            tokens, logprobs = _step(model, model_cache, y, num_draft + 1)
            mx.eval(tokens, draft_tokens)
            draft_tokens = draft_tokens.tolist()
            tokens = tokens.tolist()
            n = 0
            while n < num_draft:
                tn, dtn, lpn = tokens[n], draft_tokens[n], logprobs[n]
                if tn != dtn:
                    # 一旦主模型拒绝草稿 token，则跳出等待重新生成。
                    break
                n += 1
                ntoks += 1
                yield tn, lpn, True
                if ntoks == max_tokens:
                    break
            if ntoks < max_tokens:
                ntoks += 1
                yield tokens[n], logprobs[n], False

            if ntoks == max_tokens:
                break

            y = mx.array([tokens[n]], mx.uint32)
            draft_y = y

            # If we accepted all the draft tokens, include the last
            # draft token in the next draft step since it hasn't been
            # processed yet by the draft model
            if n == num_draft:
                draft_y = mx.concatenate(
                    [mx.array(draft_tokens[-1:], mx.uint32), draft_y]
                )

            if prev_tokens is not None:
                prev_tokens = prev_tokens[: -max(num_draft - n, 1)]
            _rewind_cache(num_draft, n)
    finally:
        # 退出循环时确保缓存回滚，防止残留状态影响后续调用。
        _rewind_cache(num_draft, n)


def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    """
    基于提示词生成文本片段的高层生成器。

    参数说明：
        model (nn.Module): 用于生成的模型。
        tokenizer (PreTrainedTokenizer): 对输入/输出做分词与去分词的 tokenizer。
        prompt (Union[str, mx.array, List[int]]): 字符串提示或已编码的 token 序列。
        max_tokens (int): 最多生成的 token 数，默认 ``256``。
        draft_model (Optional[nn.Module]): 可选的草稿模型；提供时启用推测式解码，且需与主模型共用 tokenizer。
        kwargs: 其余参数将传递给 :func:`generate_step`，详见该函数。

    生成值：
        GenerationResponse: 包含新增文本片段及相关元数据的对象，详见 :class:`GenerationResponse`。
    """
    # 这个函数将底层 token 迭代器封装为返回文本片段的高层生成器。
    # 主要职责：
    # 1. 对 prompt 做 tokenizer 处理；
    # 2. 根据是否启用推测式解码选择不同的 token 流；
    # 3. 计算性能指标并产出 `GenerationResponse`。
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            # Try to infer if special tokens are needed
            # 对字符串 prompt 进行编码：若 tokenizer 没有自动添加 BOS，则手动添加。
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer

    kwargs["max_tokens"] = max_tokens

    # 根据是否提供 draft_model 选择普通生成或推测式生成。
    if draft_model is None:
        kwargs.pop("num_draft_tokens", None)
        token_generator = generate_step(prompt, model, **kwargs)
        # from_draft always false for non-speculative generation
        token_generator = (
            (token, logprobs, False) for token, logprobs in token_generator
        )
    else:
        kwargs.pop("max_kv_size", None)
        kwargs.pop("prompt_progress_callback", None)
        token_generator = speculative_generate_step(
            prompt, model, draft_model, **kwargs
        )
    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break

            # detokenizer 会把 token 转回字符串，并保存最近一次增量片段。
            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break

            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
            )

        detokenizer.finalize()
        # 最后一次返回时带上剩余片段，并标记终止原因（自然结束或达到长度上限）。
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
        )


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, List[int]],
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    使用模型生成完整响应的便捷接口。

    参数说明：
       model (nn.Module): 语言模型。
       tokenizer (PreTrainedTokenizer): 分词器。
       prompt (Union[str, List[int]]): 输入的字符串或已编码的 token 序列。
       verbose (bool): 若为 ``True``，则实时打印生成 token 与耗时信息，默认 ``False``。
       kwargs: 其他参数将传递给 :func:`stream_generate`，详见该函数。
    """
    # 这是用户最常调用的接口：封装了逐步生成的逻辑，并将输出拼接成完整字符串。
    if verbose:
        print("=" * 10)

    text = ""
    for response in stream_generate(model, tokenizer, prompt, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        # 把每次生成的片段累积到最终字符串。
        text += response.text

    if verbose:
        print()
        print("=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
            return
        # 打印提示/生成阶段的性能指标，便于观察推理效率。
        print(
            f"Prompt: {response.prompt_tokens} tokens, "
            f"{response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {response.generation_tokens} tokens, "
            f"{response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {response.peak_memory:.3f} GB")
    return text


def _left_pad_prompts(prompts, max_length=None):
    """将序列左侧填充 0，使得 batch 中的提示长度一致，便于张量化。"""
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])


@dataclass
class BatchStats:
    """
    保存批量生成过程中各项统计信息的数据类。

    参数说明：
        prompt_tokens (int): 已处理的提示 token 数。
        prompt_tps (float): 提示阶段的 tokens-per-second。
        prompt_time (float): 提示阶段耗时（秒）。
        generation_tokens (int): 已生成的 token 数。
        generation_tps (float): 生成阶段的 tokens-per-second。
        generation_time (float): 生成阶段耗时（秒）。
        peak_memory (float): 当前记录的峰值显存（GB）。
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0

    # 用于收集批量生成的性能指标，与单条生成的 `GenerationResponse` 类似，
    # 方便在批处理场景下统一查看统计数据。


@dataclass
class BatchResponse:
    """
    用于封装批量生成结果的数据类。

    参数说明：
        texts (List[str]): 每个提示对应的生成文本。
        stats (BatchStats): 批量生成时的统计信息。
    """

    texts: List[str]
    stats: BatchStats

    # 保存批量生成的文本结果与统计信息，供外部调用者使用。


@dataclass
class Batch:
    uids: List[int]
    y: mx.array
    logprobs: mx.array
    max_tokens: List[int]
    num_tokens: List[int]
    cache: List[Any]

    def __len__(self):
        return len(self.uids)

    def filter(self, keep_idx: List[int]):
        # 根据保留索引裁剪 batch，中途完成的样本会被剔除。
        self.uids = [self.uids[k] for k in keep_idx]
        self.max_tokens = [self.max_tokens[k] for k in keep_idx]
        self.num_tokens = [self.num_tokens[k] for k in keep_idx]
        keep_idx = mx.array(keep_idx, mx.int32)
        self.y = self.y[keep_idx]
        self.logprobs = self.logprobs[keep_idx]
        for c in self.cache:
            c.filter(keep_idx)

    def extend(self, other):
        # 将新的 batch 拼接到当前 batch，持续积累未完成的样本。
        self.uids.extend(other.uids)
        self.y = mx.concatenate([self.y, other.y])
        self.logprobs = mx.concatenate([self.logprobs, other.logprobs])
        self.num_tokens.extend(other.num_tokens)
        self.max_tokens.extend(other.max_tokens)
        for c, o in zip(self.cache, other.cache):
            c.extend(o)


def _make_cache(model, left_padding):
    """
    将单样本缓存列表转换为适用于批量推理的缓存结构。
    """
    # 批量生成时，每个样本的前缀长度不同，需要把单样本缓存包装成支持 batch 的形式。

    def to_batch_cache(c):
        if isinstance(c, KVCache):
            # 普通 KV 缓存 -> 批量 KV 缓存
            return BatchKVCache(left_padding)
        elif isinstance(c, ArraysCache):
            # ArraysCache 本身支持批量，只需指定左填充信息。
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, CacheList):
            # 嵌套缓存需要递归转换。
            return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        cache = model.make_cache()
        return [to_batch_cache(c) for c in cache]
    else:
        return [BatchKVCache(left_padding) for _ in model.layers]


class BatchGenerator:
    """负责批量提示的增量生成调度，内部维护一个活跃批次队列。"""

    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]

    def __init__(
        self,
        model,
        max_tokens: int = 128,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
    ):
        # 初始化批量生成器，配置预填充/补全阶段的批次大小，并准备统计结构。
        self.model = model
        self.unprocessed_prompts = []
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or set()
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size
        self._stats = BatchStats()

        # active_batch 保存当前正在增量生成的样本集合。
        self.active_batch = None

    def insert(self, prompts, max_tokens: Union[List[int], int, None] = None):
        uids = []

        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        for p, m in zip(prompts, max_tokens):
            # 将新提示加入等待队列，并分配唯一 uid，便于结果对齐。
            self.unprocessed_prompts.append((self.uid_count, p, m))
            uids.append(self.uid_count)
            self.uid_count += 1
        # Sort in ascending order of length
        self.unprocessed_prompts = sorted(
            self.unprocessed_prompts, key=lambda x: len(x[1])
        )
        return uids

    def _process_prompts(self, prompts):
        uids, inputs, max_tokens = zip(*prompts)
        lengths = [len(p) for p in inputs]
        max_length = max(lengths)
        batch_size = self.prefill_batch_size
        self._stats.prompt_tokens += sum(lengths)
        left_padding = [max_length - l for l in lengths]
        inputs = _left_pad_prompts(inputs, max_length=max_length)

        # 为当前批次创建批量化的 KV 缓存，并执行预填充阶段。
        prompt_cache = _make_cache(self.model, left_padding)

        while inputs.shape[1] > 1:
            n_to_process = min(self.prefill_step_size, inputs.shape[1] - 1)
            self.model(inputs[:, :n_to_process], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            inputs = inputs[:, n_to_process:]
            mx.clear_cache()

        # 预填充结束后，调用 _step 获取第一轮生成结果。
        y, logprobs = self._step(inputs, prompt_cache)
        mx.async_eval(y, logprobs)
        return Batch(
            list(uids), y, logprobs, list(max_tokens), [0] * len(uids), prompt_cache
        )

    def _step(self, input_tokens: mx.array, prompt_cache: List[Any]):
        # 对整个 batch 一次性生成下一个 token，并返回采样后的 token 与 logprob。
        logits = self.model(input_tokens, cache=prompt_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = self.sampler(logprobs)
        return sampled, logprobs

    def stats(self):
        # 调用该方法即可获取累计的性能统计信息。
        self._stats.prompt_tps = self._stats.prompt_tokens / self._stats.prompt_time
        self._stats.generation_tps = (
            self._stats.generation_tokens / self._stats.generation_time
        )
        self._stats.peak_memory = mx.get_peak_memory() / 1e9
        return self._stats

    def _next(self):
        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0
        num_to_add = self.completion_batch_size - num_active
        # 当活跃样本数量未达到 completion_batch_size 时，从等待队列中补齐。
        while num_to_add >= self.prefill_batch_size:
            prompts = self.unprocessed_prompts[: self.prefill_batch_size]
            # Finish processing the last examples of the last batch
            if len(prompts) == 0 and num_active > 0:
                break
            # No more prompts and no more completions, all done
            elif len(prompts) == 0:
                self.active_batch = None
                return []
            # Process prompts
            if batch is not None and not prompt_processing:
                # Finish any active completion tokens
                mx.eval(batch.y, batch.logprobs)
                self._stats.generation_time += time.perf_counter() - tic
                tic = time.perf_counter()

            batch = self._process_prompts(prompts)
            self.unprocessed_prompts = self.unprocessed_prompts[
                self.prefill_batch_size :
            ]
            prompt_processing = True
            # If there was no active batch, set it
            if self.active_batch is None:
                self.active_batch = batch
            else:
                self.active_batch.extend(batch)

            num_active = len(self.active_batch)
            num_to_add -= len(batch)

        batch = self.active_batch
        y, logprobs = batch.y, batch.logprobs
        # 调用 _step 生成下一批 token，并提前触发异步计算。
        batch.y, batch.logprobs = self._step(y[:, None], batch.cache)
        mx.async_eval(batch.y, batch.logprobs)

        y = y.tolist()
        toc = time.perf_counter()
        if prompt_processing:
            self._stats.prompt_time += toc - tic
        else:
            self._stats.generation_time += toc - tic
        keep_idx = []
        end_idx = []
        responses = []

        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            # 遍历当前 batch 的每个样本，判断是否达到结束条件。
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if t in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            responses.append(self.Response(uid, t, logprobs[e], finish_reason))

        # Remove any finished completions
        # 将完成的样本移出 active_batch，只保留仍需继续生成的样本。
        if len(end_idx):
            if len(keep_idx) > 0:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)
        return responses

    def next(self):
        # 对外暴露的接口：包装 `_next` 并在生成流上执行。
        with mx.stream(generation_stream):
            return self._next()


def batch_generate(
    model,
    tokenizer,
    prompts: List[int],
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    **kwargs,
) -> BatchResponse:
    """
    针对一组提示批量生成文本。

    参数说明：
       model (nn.Module): 语言模型。
       tokenizer (PreTrainedTokenizer): 分词器。
       prompts (List[List[int]]): 输入提示的 token 序列列表。
       verbose (bool): 若为 ``True``，打印生成进度与耗时信息，默认 ``False``。
       max_tokens (Union[int, List[int]]): 输出 token 的最大数量，可为单个整数或与提示等长的列表。
       kwargs: 其他参数将传递给 :obj:`BatchGenerator`，可进一步定制行为。
    """
    # 对外提供的批量推理入口：负责初始化 BatchGenerator 并汇总结果。

    gen = BatchGenerator(model, stop_tokens=tokenizer.eos_token_ids, **kwargs)
    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")

    with wired_limit(model, [generation_stream]):
        uids = gen.insert(prompts, max_tokens)
        results = {uid: [] for uid in uids}
        while responses := gen.next():
            # gen.next() 每次返回一批 token 响应，需要按 uid 聚合同一条样本的结果。
            for r in responses:
                if verbose and r.finish_reason != None:
                    fin += 1
                    print(
                        f"[batch_generate] Finished processing {fin}/{num_samples} ...",
                        end="\r",
                    )
                if r.finish_reason != "stop":
                    # 非终止 token 加入结果序列；遇到 stop token 则不再追加。
                    results[r.uid].append(r.token)
    if verbose:
        print(f"[batch_generate] Finished processing {fin}/{num_samples}")

    # Return results in correct order
    # 按照插入顺序还原文本，并返回统计信息。
    texts = [tokenizer.decode(results[uid]) for uid in uids]
    stats = gen.stats()
    if verbose:
        print(
            f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")
    return BatchResponse(texts, stats)


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    # Load the prompt cache and metadata if a cache file is provided
    using_cache = args.prompt_cache_file is not None
    if using_cache:
        # 如果提供了缓存文件，则直接加载以复用先前计算的 KV。
        prompt_cache, metadata = load_prompt_cache(
            args.prompt_cache_file,
            return_metadata=True,
        )
        if isinstance(prompt_cache[0], QuantizedKVCache):
            if args.kv_bits is not None and args.kv_bits != prompt_cache[0].bits:
                raise ValueError(
                    "--kv-bits does not match the kv cache loaded from --prompt-cache-file."
                )
            if args.kv_group_size != prompt_cache[0].group_size:
                raise ValueError(
                    "--kv-group-size does not match the kv cache loaded from --prompt-cache-file."
                )

    # Building tokenizer_config
    tokenizer_config = (
        {} if not using_cache else json.loads(metadata["tokenizer_config"])
    )
    tokenizer_config["trust_remote_code"] = True if args.trust_remote_code else None
    # 保证 tokenizer 的配置与缓存/用户参数一致，避免推理时不兼容。

    model_path = args.model
    if using_cache:
        if model_path is None:
            model_path = metadata["model"]
        elif model_path != metadata["model"]:
            raise ValueError(
                f"Providing a different model ({model_path}) than that "
                f"used to create the prompt cache ({metadata['model']}) "
                "is an error."
            )
    model_path = model_path or DEFAULT_MODEL

    model, tokenizer = load(
        model_path,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
    )
    for eos_token in args.extra_eos_token:
        # 用户可额外指定停止 token，适应不同格式的对话模板。
        tokenizer.add_eos_token(eos_token)

    template_kwargs = {}
    if args.chat_template_config is not None:
        template_kwargs = json.loads(args.chat_template_config)

    # 根据优先级决定使用何种聊天模板：用户默认模板 > 缓存中的模板。
    if args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template
    elif using_cache:
        tokenizer.chat_template = json.loads(metadata["chat_template"])

    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    prompt = sys.stdin.read() if prompt == "-" else prompt
    if not args.ignore_chat_template and tokenizer.chat_template is not None:
        # 构造 messages 列表，符合 tokenizer 预期的聊天格式。
        if args.system_prompt is not None:
            messages = [{"role": "system", "content": args.system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})

        has_prefill = args.prefill_response is not None
        if has_prefill:
            messages.append({"role": "assistant", "content": args.prefill_response})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=has_prefill,
            add_generation_prompt=not has_prefill,
            **template_kwargs,
        )

        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if using_cache:
            messages[-1]["content"] = "<query>"
            test_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=has_prefill,
                add_generation_prompt=not has_prefill,
            )
            prompt = prompt[test_prompt.index("<query>") :]
        prompt = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        prompt = tokenizer.encode(prompt)

    if args.draft_model is not None:
        draft_model, draft_tokenizer = load(args.draft_model)
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError("Draft model tokenizer does not match model tokenizer.")
    else:
        draft_model = None
    sampler = make_sampler(
        args.temp,
        args.top_p,
        args.min_p,
        args.min_tokens_to_keep,
        top_k=args.top_k,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
    )
    # make_sampler 根据温度、top-p、XTC 等参数构建采样策略，返回一个函数。
    # 将所有解析好的参数传入 `generate` 完成最终文本生成。
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        sampler=sampler,
        max_kv_size=args.max_kv_size,
        prompt_cache=prompt_cache if using_cache else None,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        draft_model=draft_model,
        num_draft_tokens=args.num_draft_tokens,
    )
    if not args.verbose:
        # 默认情况下直接打印最终生成的文本。
        print(response)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.generate...` directly is deprecated."
        " Use `mlx_lm.generate...` or `python -m mlx_lm generate ...` instead."
    )
    main()
