"""
分词器工具模块 - 中文注释版本

这个模块提供了各种分词器和解分词器的实现，特别专注于流式解分词功能。
流式解分词允许我们逐个处理token，这对于实时生成文本的场景非常有用。

主要组件：
1. StreamingDetokenizer - 流式解分词器基类，定义了统一接口
2. NaiveStreamingDetokenizer - 朴素但通用的流式解分词实现
3. SPMStreamingDetokenizer - 针对SentencePiece模型优化的实现
4. BPEStreamingDetokenizer - 针对BPE模型优化的实现
5. TokenizerWrapper - 包装器，整合HuggingFace分词器和流式解分词器
6. NewlineTokenizer - 特殊处理换行符的分词器

使用示例：
```python
# 加载分词器
tokenizer = load_tokenizer(model_path)

# 获取流式解分词器
detokenizer = tokenizer.detokenizer
detokenizer.reset()

# 逐个处理token
for token in generated_tokens:
    detokenizer.add_token(token)
    print(detokenizer.last_segment, end='')  # 实时输出新生成的文本片段

# 完成解分词
detokenizer.finalize()
print(detokenizer.text)  # 输出完整文本
```
"""

import json
from functools import partial
from json import JSONDecodeError
from typing import List

from transformers import AutoTokenizer, PreTrainedTokenizerFast


class StreamingDetokenizer:
    """
    流式解分词器基类
    
    这个类定义了流式解分词的统一接口，允许我们逐个token地解分词文本。
    流式解分词的优势在于可以实时显示生成的文本，而不需要等待所有token生成完毕。

    工作原理：
    1. 初始化时调用reset()重置状态
    2. 逐个添加token通过add_token()方法
    3. 随时可以通过text属性获取当前已解码的完整文本
    4. 通过last_segment属性获取自上次访问以来新生成的文本片段
    5. 最后调用finalize()确保所有剩余token都被正确解码

    属性说明：
    - text: 当前已解码的完整文本
    - tokens: 已添加的所有token列表
    - last_segment: 自上次访问以来新生成的文本片段
    - offset: 内部使用的偏移量，用于跟踪last_segment的起始位置

    使用示例：
    ```python
    detokenizer = StreamingDetokenizer()
    detokenizer.reset()
    
    for token in generate_tokens():
        detokenizer.add_token(token)
        print(detokenizer.last_segment, end='')  # 实时输出新片段
    
    detokenizer.finalize()
    print(detokenizer.text)  # 完整文本
    ```
    """

    # 使用__slots__优化内存使用，只定义必要的属性
    __slots__ = ("text", "tokens", "offset")

    def reset(self):
        """
        重置解分词器状态
        
        这个方法应该清除所有内部状态，准备开始新的解分词过程。
        通常在开始解分词新的token序列之前调用。
        
        子类必须实现这个方法。
        """
        raise NotImplementedError("子类必须实现reset方法")

    def add_token(self, token):
        """
        添加一个token到解分词器
        
        参数:
            token: 要添加的token（通常是整数ID）
            
        这个方法将新的token添加到内部状态中，并可能更新text属性。
        根据具体的实现策略，可能不会立即解码这个token。
        
        子类必须实现这个方法。
        """
        raise NotImplementedError("子类必须实现add_token方法")

    def finalize(self):
        """
        完成解分词过程
        
        这个方法确保所有未处理的token都被正确解码，并更新最终的text属性。
        在所有token都添加完毕后应该调用此方法。
        
        子类必须实现这个方法。
        """
        raise NotImplementedError("子类必须实现finalize方法")

    @property
    def last_segment(self):
        """
        获取自上次访问以来新生成的文本片段
        
        这个属性使用了智能的偏移量跟踪机制：
        1. 获取当前完整文本
        2. 从上次记录的偏移位置开始截取到末尾
        3. 更新偏移位置到当前文本末尾
        4. 返回截取的片段
        
        这样设计的好处是用户可以连续调用这个属性来获取增量更新的文本，
        而不需要自己手动跟踪位置。
        
        返回:
            str: 自上次访问以来新生成的文本片段
        """
        text = self.text
        segment = text[self.offset :]  # 从偏移位置截取到末尾
        self.offset = len(text)  # 更新偏移位置
        return segment


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    """
    朴素流式解分词器
    
    这是最简单但最通用的流式解分词实现。它依赖于底层分词器的decode()方法，
    因此适用于任何类型的分词器。

    实现原理：
    1. 维护一个当前token列表(_current_tokens)和当前文本(_current_text)
    2. 每次添加token时，尝试解码当前token列表
    3. 如果解码结果包含换行符或特殊字符，则将其合并到主文本中
    4. 否则继续累积token，等待更好的解码时机

    复杂度分析：
    - 时间复杂度：O(T²)，其中T是最长行的长度
    - 这是因为对于每个新token，都可能需要重新解码整个当前行
    - 虽然复杂度较高，但实现简单且兼容性好

    适用场景：
    - 需要兼容各种分词器的通用场景
    - 对性能要求不高的应用
    - 调试和测试阶段

    注意事项：
    - 在处理长文本时性能可能较差
    - 某些特殊字符的处理可能不够精确
    """

    def __init__(self, tokenizer):
        """
        初始化朴素流式解分词器
        
        参数:
            tokenizer: 底层分词器实例，必须提供decode()方法
            
        初始化过程：
        1. 保存分词器引用
        2. 测试分词器的decode方法（解码[0]确保方法可用）
        3. 调用reset()初始化所有状态变量
        """
        self._tokenizer = tokenizer
        # 测试分词器的decode方法是否正常工作
        self._tokenizer.decode([0])
        self.reset()

    def reset(self):
        """
        重置解分词器状态
        
        初始化所有内部状态变量：
        - offset: 用于last_segment属性的偏移跟踪
        - tokens: 已添加的所有token列表
        - _text: 已经确认的完整文本
        - _current_tokens: 当前正在处理的token列表
        - _current_text: 当前token列表的解码结果
        """
        self.offset = 0
        self.tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token):
        """
        添加一个token到解分词器
        
        参数:
            token: 要添加的token ID
            
        处理逻辑：
        1. 将token添加到当前token列表
        2. 将token添加到完整token列表
        3. 不立即解码，等待text属性访问时再处理
        """
        self._current_tokens.append(token)
        self.tokens.append(token)

    def finalize(self):
        """
        完成解分词过程
        
        将所有剩余的未处理token解码并合并到主文本中。
        这个方法确保在token流结束后，所有内容都被正确解码。
        """
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        """
        获取当前已解码的完整文本
        
        这是这个类的核心逻辑所在：
        1. 如果有当前token，尝试解码它们
        2. 检查解码结果是否以替换字符()结尾
        3. 如果是，说明解码不完整，暂时不添加到主文本
        4. 如果不是，检查是否需要清理空格
        5. 如果解码结果以换行符结尾，将其合并到主文本
        6. 返回主文本和当前文本的组合
        
        替换字符的含义：
        - 通常出现在UTF-8多字节字符被截断时
        - 表示当前字符还不完整，需要更多token
        
        返回:
            str: 当前已解码的完整文本
        """
        # 如果有当前token，尝试解码
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
            
            # 检查是否以替换字符结尾（表示UTF-8字符不完整）
            if self._current_text.endswith("\ufffd") or (
                self._tokenizer.clean_up_tokenization_spaces
                and len(self._current_text) > 0
                and self._current_text[-1] == " "
            ):
                # 如果不完整或需要清理空格，移除最后一个字符
                self._current_text = self._current_text[:-1]
        
        # 如果当前文本以换行符结尾，可以安全地合并到主文本
        if self._current_text and self._current_text[-1] == "\n":
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
            
        return self._text + self._current_text


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """
    SentencePiece流式解分词器
    
    针对SentencePiece模型优化的流式解分词器。SentencePiece是一种广泛使用的分词方法，
    特别适用于多语言模型。

    SentencePiece特点：
    - 使用特殊字符（▁，Unicode U+2581）表示词的开始
    - 支持字节级回退机制
    - 能够处理任意语言的文本

    优化原理：
    - 当下一个token以▁开头时，表示新词开始，可以安全地将之前累积的token解码
    - 这样可以实现线性复杂度O(T)，而不是朴素方法的O(T²)
    - 通过预先构建token映射表，避免重复的token到字符串转换

    复杂度分析：
    - 时间复杂度：O(T)，线性复杂度
    - 空间复杂度：O(V)，其中V是词汇表大小

    适用场景：
    - 使用SentencePiece分词的模型（如T5、mT5等）
    - 需要高性能流式解分词的场景
    - 多语言文本处理
    """

    def __init__(self, tokenizer, trim_space=True):
        """
        初始化SPM流式解分词器
        
        参数:
            tokenizer: SentencePiece分词器
            trim_space: 是否修剪开头的空格，默认为True
            
        初始化过程：
        1. 设置空格修剪选项
        2. 定义分隔符（▁字符）
        3. 构建token映射表（从token ID到字节的映射）
        4. 处理特殊字节token（如<0x20>表示空格）
        5. 调用reset()初始化状态
        """
        self.trim_space = trim_space
        # SentencePiece使用▁（Unicode U+2581）作为词开始标记
        self._sep = "\u2581".encode()

        # 构建token映射表：从token ID到字节的映射
        # 这样可以避免每次都查找词汇表
        max_token_id = max(tokenizer.vocab.values())
        self.tokenmap = [""] * (max_token_id + 1)
        
        for value, tokenid in tokenizer.vocab.items():
            if value.startswith("<0x"):
                # 处理字节token（如<0x20>表示空格）
                # 从十六进制字符串转换为字节值
                self.tokenmap[tokenid] = bytes([int(value[3:5], 16)])
            else:
                # 普通token直接编码为字节
                self.tokenmap[tokenid] = value.encode()

        self.reset()

    def reset(self):
        """
        重置解分词器状态
        
        初始化所有状态变量：
        - offset: 用于last_segment属性的偏移跟踪
        - _unflushed: 未刷新的字节缓冲区
        - text: 已解码的文本
        - tokens: 已添加的token列表
        """
        self.offset = 0
        self._unflushed = b""  # 字节缓冲区
        self.text = ""
        self.tokens = []

    def _try_flush(self, force=False):
        """
        尝试刷新缓冲区到文本
        
        这个方法是SPM解分词器的核心：
        1. 将▁替换为空格
        2. 解码字节为UTF-8文本
        3. 检查是否以替换字符结尾（表示UTF-8不完整）
        4. 如果不完整且不强制刷新，则等待更多token
        5. 如果完整或强制刷新，则添加到主文本
        
        参数:
            force: 是否强制刷新（在finalize时使用）
        """
        # 将▁替换为空格，并尝试解码为UTF-8
        text = self._unflushed.replace(self._sep, b" ").decode("utf-8", "replace")
        
        # 如果不强制刷新且文本以替换字符结尾，表示UTF-8字符不完整
        if not force and text.endswith("\ufffd"):
            return  # 等待更多token完成字符
            
        # 如果是第一个token且需要修剪空格，移除开头的空格
        if not self.text and self.trim_space and text and text[0] == " ":
            text = text[1:]
            
        self.text += text
        self._unflushed = b""

    def add_token(self, token):
        """
        添加一个token到解分词器
        
        参数:
            token: 要添加的token ID
            
        处理逻辑：
        1. 将token添加到完整列表
        2. 从映射表获取对应的字节表示
        3. 将字节添加到未刷新缓冲区
        4. 尝试刷新缓冲区（如果可能的话）
        """
        self.tokens.append(token)
        v = self.tokenmap[token]  # 获取token的字节表示
        self._unflushed += v  # 添加到缓冲区
        self._try_flush()  # 尝试刷新

    def finalize(self):
        """
        完成解分词过程
        
        强制刷新所有剩余的缓冲区内容到主文本。
        即使UTF-8字符不完整也会强制解码，确保没有遗漏内容。
        """
        self._try_flush(force=True)
        self._unflushed = b""


class BPEStreamingDetokenizer(StreamingDetokenizer):
    """
    BPE流式解分词器
    
    针对OpenAI风格的BPE（Byte Pair Encoding）模型优化的流式解分词器。
    BPE是一种常见的分词方法，特别适用于英文和其他使用空格分隔的语言。

    BPE特点：
    - 使用字节级别的编码
    - 可以处理任意Unicode字符
    - 通常需要特殊处理空格

    优化原理：
    - 类似于SPM解分词器，通过智能判断何时可以安全地解码token
    - 维护未刷新的字符串缓冲区
    - 当确定不会再有字符连接时，才将缓冲区内容刷新到主文本

    字节解码器：
    - BPE使用特殊的字节到字符的映射
    - 需要实现GPT-2中定义的字节解码算法
    - 这样可以正确处理UTF-8编码

    复杂度分析：
    - 时间复杂度：O(T)，线性复杂度
    - 空间复杂度：O(V)，其中V是词汇表大小

    适用场景：
    - GPT系列模型
    - 使用BPE分词的其他模型
    - 需要高性能的英文文本处理
    """

    # 类级别的字节解码器，所有实例共享
    _byte_decoder = None
    
    # 需要特殊处理的空格模式
    _space_matches = (".", "?", "!", ",", "n't", "'m", "'s", "'ve", "'re")

    def __init__(self, tokenizer):
        """
        初始化BPE流式解分词器
        
        参数:
            tokenizer: BPE分词器
            
        初始化过程：
        1. 保存分词器的空格清理设置
        2. 构建token映射表（从token ID到字符串的映射）
        3. 初始化状态变量
        4. 构建字节解码器（如果还没有的话）
        """
        self.clean_spaces = tokenizer.clean_up_tokenization_spaces

        # 构建token映射表：从token ID到字符串的映射
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()

        # 构建BPE字节解码器（基于OpenAI GPT-2的实现）
        self.make_byte_decoder()

    def reset(self):
        """
        重置解分词器状态
        
        初始化所有状态变量：
        - offset: 用于last_segment属性的偏移跟踪
        - _unflushed: 未刷新的字符串缓冲区
        - text: 已解码的文本
        - tokens: 已添加的token列表
        """
        self.offset = 0
        self._unflushed = ""  # 字符串缓冲区
        self.text = ""
        self.tokens = []

    def _decode_bytes(self, seq):
        """
        解码字节序列为UTF-8字符串
        
        这个方法实现了BPE特殊的字节解码逻辑：
        1. 遍历每个字符
        2. 使用字节解码器将字符转换为字节值
        3. 将字节值组合成字节数组
        4. 解码字节数组为UTF-8字符串
        
        参数:
            seq: 要解码的字符序列
            
        返回:
            str: 解码后的UTF-8字符串
        """
        barr = bytearray()  # 字节数组
        for c in seq:
            # 获取字符对应的字节值
            res = self._byte_decoder.get(c, False)
            if res:
                barr.append(res)
            else:
                # 如果没有映射，直接使用UTF-8编码
                barr.extend(bytes(c, "utf-8"))
        return barr.decode("utf-8", "replace")

    def _maybe_trim_space(self, current_text):
        """
        智能处理空格
        
        根据上下文决定是否需要修剪开头的空格：
        1. 如果当前文本为空，直接返回
        2. 如果开头不是空格，直接返回
        3. 如果是第一个文本且以空格开头，修剪空格
        4. 如果启用了空格清理且下一个字符是标点符号等，修剪空格
        
        参数:
            current_text: 当前要处理的文本
            
        返回:
            str: 处理后的文本
        """
        if len(current_text) == 0:
            return current_text
        elif current_text[0] != " ":
            return current_text
        elif not self.text:
            return current_text[1:]  # 第一个token，修剪开头的空格
        elif self.clean_spaces and current_text[1:].startswith(self._space_matches):
            return current_text[1:]  # 如果后面是标点符号，修剪空格
        return current_text

    def add_token(self, token):
        """
        添加一个token到解分词器
        
        参数:
            token: 要添加的token ID
            
        处理逻辑：
        1. 将token添加到完整列表
        2. 从映射表获取对应的字符串
        3. 将字符串添加到未刷新缓冲区
        4. 尝试解码缓冲区
        5. 检查是否可以安全刷新（UTF-8完整且不是单独的空格字节）
        """
        self.tokens.append(token)
        # 获取token的字符串表示，如果token超出范围使用"!"
        v = self.tokenmap[token] if token < len(self.tokenmap) else "!"
        self._unflushed += v
        text = self._decode_bytes(self._unflushed)

        # 检查是否可以安全刷新：
        # 1. UTF-8字符完整（不以替换字符结尾）
        # 2. 不是单独的空格字节（等待下一个token决定是否需要空格）
        if not text.endswith("\ufffd") and not (
            len(v) == 1 and self._byte_decoder[v[0]] == 32
        ):
            self.text += self._maybe_trim_space(text)
            self._unflushed = ""

    def finalize(self):
        """
        完成解分词过程
        
        强制解码所有剩余的缓冲区内容并合并到主文本。
        使用智能空格处理来确保输出的正确性。
        """
        # 强制解码剩余的未刷新内容
        current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
            "utf-8",
            "replace",
        )
        self.text += self._maybe_trim_space(current_text)
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls):
        """
        构建BPE字节解码器
        
        这个方法实现了OpenAI GPT-2中定义的字节解码算法。
        算法的基本思想是将可打印ASCII字符映射到对应的字节值，
        同时保留不可打印字符的特殊编码。

        映射规则：
        1. 字符0-33直接映射到字节0-33
        2. 字符34-126映射到字节33-125
        3. 字符127-161映射到字节0-34（重复映射）
        4. 字符162-255映射到字节35-128
        
        这种映射确保了：
        - 所有字节都有对应的字符表示
        - 避免了JSON转义的问题
        - 保持了可读性
        
        详细原理请参考：
        https://github.com/openai/gpt-2/blob/master/src/encoder.py
        """
        if cls._byte_decoder is not None:
            return  # 已经构建过了

        char_to_bytes = {}
        # 定义字符范围的边界
        limits = [
            0,        # 起始
            ord("!"), # 33
            ord("~") + 1, # 127
            ord("¡"), # 161
            ord("¬") + 1, # 172
            ord("®"),   # 174
            ord("ÿ") + 1, # 256
        ]
        n = 0
        # 构建字符到字节的映射
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                # 偶数索引：使用高字节映射
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                # 奇数索引：直接映射
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes


class TokenizerWrapper:
    """
    分词器包装器
    
    这个类是一个智能包装器，它将HuggingFace分词器和流式解分词器结合在一起。
    主要功能包括：
    1. 自动选择合适的流式解分词器
    2. 属性委托机制（将未处理的属性访问转发到底层分词器）
    3. EOS（结束序列）token管理
    4. 特殊token检测（思考token、工具调用token）

    设计模式：
    - 使用了代理模式（Proxy Pattern）
    - 通过__getattr__和__setattr__实现属性访问的透明转发
    - 提供了额外的功能而不改变原有接口

    特殊功能：
    - 检测模型是否支持思考模式（thinking）
    - 检测模型是否支持工具调用（tool calling）
    - 支持多个EOS token的管理

    使用示例：
    ```python
    # 包装器会自动选择合适的流式解分词器
    tokenizer = load_tokenizer(model_path)
    
    # 直接使用分词器的所有功能
    tokens = tokenizer.encode("Hello world")
    text = tokenizer.decode(tokens)
    
    # 获取流式解分词器
    detokenizer = tokenizer.detokenizer
    ```
    """

    def __init__(
        self, tokenizer, detokenizer_class=NaiveStreamingDetokenizer, eos_token_ids=None
    ):
        """
        初始化分词器包装器
        
        参数:
            tokenizer: HuggingFace分词器实例
            detokenizer_class: 要使用的流式解分词器类
            eos_token_ids: EOS token ID列表（可选）
            
        初始化过程：
        1. 保存分词器和流式解分词器类
        2. 设置EOS token集合
        3. 检测特殊token（思考和工具调用）
        """
        self._tokenizer = tokenizer
        self._detokenizer_class = detokenizer_class
        # 设置EOS token ID集合
        self._eos_token_ids = (
            set(eos_token_ids)
            if eos_token_ids is not None
            else {tokenizer.eos_token_id}
        )
        
        # 初始化特殊token检测标志
        self._think_start = None
        self._think_end = None
        self._tool_call_start = None
        self._tool_call_end = None

        # 定义思考模式的特殊token对
        THINK_TOKENS = [("<|thinking|>", "<|/thinking|>")]
        # 定义工具调用的特殊token对
        TOOL_CALL_TOKENS = [("<|tool_call|>", "<|/tool_call|>")]

        vocab = tokenizer.get_vocab()
        # 检测思考模式token
        for think_start, think_end in THINK_TOKENS:
            if think_start in vocab and think_end in vocab:
                self._think_start = think_start
                self._think_end = think_end
                break
        
        # 检测工具调用token（仅在聊天模板包含"tool"时）
        if tokenizer.chat_template and '"tool"' in tokenizer.chat_template:
            for tool_call_start, tool_call_end in TOOL_CALL_TOKENS:
                if tool_call_start in vocab and tool_call_end in vocab:
                    self._tool_call_start = tool_call_start
                    self._tool_call_end = tool_call_end
                    break

    def add_eos_token(self, token: str):
        """
        添加EOS token到集合中
        
        参数:
            token: 要添加的EOS token（可以是字符串或整数ID）
            
        这个方法允许动态添加额外的EOS token，这对于处理多语言模型
        或具有多种结束标记的模型很有用。
        """
        token_id = None
        try:
            token_id = int(token)
        except ValueError:
            token_id = self._tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            raise ValueError(f"'{token}' is not a token for this tokenizer")

        self._eos_token_ids.add(token_id)

    @property
    def has_thinking(self):
        """
        检查模型是否支持思考模式
        
        返回:
            bool: 如果支持思考模式返回True，否则返回False
        """
        return self._think_start is not None

    @property
    def think_start(self):
        """
        获取思考开始token
        
        返回:
            str or None: 思考开始token文本，如果不支持则返回None
        """
        return self._think_start

    @property
    def think_end(self):
        """
        获取思考结束token
        
        返回:
            str or None: 思考结束token文本，如果不支持则返回None
        """
        return self._think_end

    @property
    def has_tool_calling(self):
        """
        检查模型是否支持工具调用
        
        返回:
            bool: 如果支持工具调用返回True，否则返回False
        """
        return self._tool_call_start is not None

    @property
    def tool_call_start(self):
        """
        获取工具调用开始token
        
        返回:
            str or None: 工具调用开始token文本，如果不支持则返回None
        """
        return self._tool_call_start

    @property
    def tool_call_end(self):
        """
        获取工具调用结束token
        
        返回:
            str or None: 工具调用结束token文本，如果不支持则返回None
        """
        return self._tool_call_end

    @property
    def detokenizer(self):
        """
        获取流式解分词器实例
        
        返回:
            StreamingDetokenizer: 新的流式解分词器实例
            
        每次调用这个属性都会创建一个新的解分词器实例，
        确保每次解分词过程都是独立的。
        """
        return self._detokenizer_class(self)

    def __getattr__(self, attr):
        """
        属性访问委托机制
        
        这个方法实现了代理模式的核心功能：
        1. 如果访问的是detokenizer或eos_token_ids，返回内部属性
        2. 如果访问的是私有属性（以_开头），使用正常属性访问
        3. 否则，将属性访问委托给底层分词器
        
        参数:
            attr: 要访问的属性名
            
        返回:
            属性值或方法引用
        """
        if attr == "detokenizer":
            return self._detokenizer
        elif attr == "eos_token_ids":
            return self._eos_token_ids
        elif attr.startswith("_"):
            return self.__getattribute__(attr)
        else:
            return getattr(self._tokenizer, attr)

    def __setattr__(self, attr, value):
        """
        属性设置委托机制
        
        这个方法控制属性的设置：
        1. detokenizer属性不能被设置（只读）
        2. eos_token_ids可以被设置为集合或None
        3. 私有属性（以_开头）正常设置
        4. 其他属性委托给底层分词器
        
        参数:
            attr: 属性名
            value: 属性值
        """
        if attr in {"detokenizer", "eos_token_ids"}:
            if attr == "detokenizer":
                raise AttributeError("Cannot set the detokenizer.")
            elif attr == "eos_token_ids":
                self._eos_token_ids = set(value) if value is not None else set()
        elif attr.startswith("_"):
            super().__setattr__(attr, value)
        else:
            setattr(self._tokenizer, attr, value)


class NewlineTokenizer(PreTrainedTokenizerFast):
    """
    换行符处理分词器
    
    这个特殊分词器用于处理包含换行符的文本。它将换行符替换为特殊标记<n>，
    这样可以避免某些分词器在处理换行符时可能出现的问题。

    工作原理：
    1. 编码时：将\n替换为<n>
    2. 解码时：将<n>替换回\n
    3. 支持批量处理

    使用场景：
    - 需要精确控制换行符处理的模型
    - 某些对换行符敏感的分词器
    - 需要保持文本格式的应用

    注意事项：
    - 这是一个预处理包装器，不会改变底层分词器的核心逻辑
    - 仅在需要精确换行符控制时使用
    """

    def __init__(self, *args, **kwargs):
        """
        初始化换行符处理分词器
        
        参数:
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)

    def _preprocess_text(self, text):
        """
        预处理文本，将换行符替换为特殊标记
        
        参数:
            text: 要预处理的文本
            
        返回:
            str: 处理后的文本
        """
        return text.replace("\n", "<n>")

    def _postprocess_text(self, text):
        """
        后处理文本，将特殊标记替换回换行符
        
        参数:
            text: 要后处理的文本
            
        返回:
            str: 处理后的文本
        """
        return text.replace("<n>", "\n")

    def encode(self, text, **kwargs):
        """
        编码单个文本
        
        参数:
            text: 要编码的文本
            **kwargs: 传递给父类的其他参数
            
        返回:
            List[int]: 编码后的token ID列表
        """
        return super().encode(self._preprocess_text(text), **kwargs)

    def encode_batch(self, texts, **kwargs):
        """
        批量编码文本
        
        参数:
            texts: 要编码的文本列表
            **kwargs: 传递给父类的其他参数
            
        返回:
            List[List[int]]: 编码后的token ID列表的列表
        """
        return super().encode_batch([self._preprocess_text(t) for t in texts], **kwargs)

    def decode(self, *args, **kwargs):
        """
        解码token序列为文本
        
        参数:
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数
            
        返回:
            str: 解码后的文本
        """
        return self._postprocess_text(super().decode(*args, **kwargs))

    def batch_decode(self, *args, **kwargs):
        """
        批量解码token序列
        
        参数:
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数
            
        返回:
            List[str]: 解码后的文本列表
        """
        decoded = super().batch_decode(*args, **kwargs)
        return [self._postprocess_text(d) for d in decoded]


# 注册NewlineTokenizer到AutoTokenizer
AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)


def _match(a, b):
    """
    递归匹配两个数据结构是否相等
    
    这个函数用于深度比较两个数据结构（字典、列表等）是否相等。
    它比简单的==比较更严格，确保类型和结构都完全匹配。

    参数:
        a: 第一个数据结构
        b: 第二个数据结构
        
    返回:
        bool: 如果完全匹配返回True，否则返回False
    """
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))
    return a == b


def _is_spm_decoder(decoder):
    """
    检查是否为SPM解码器（带空格清理）
    
    通过匹配解码器的结构来判断是否为标准的SentencePiece解码器配置。
    
    参数:
        decoder: 解码器配置字典
        
    返回:
        bool: 如果是SPM解码器返回True，否则返回False
    """
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder):
    """
    检查是否为SPM解码器（不带空格清理）
    
    这是SPM解码器的变体，不包含空格清理步骤。
    
    参数:
        decoder: 解码器配置字典
        
    返回:
        bool: 如果是不带空格清理的SPM解码器返回True，否则返回False
    """
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder):
    """
    检查是否为BPE解码器
    
    通过检查解码器类型是否为ByteLevel来判断是否为BPE解码器。
    
    参数:
        decoder: 解码器配置字典
        
    返回:
        bool: 如果是BPE解码器返回True，否则返回False
    """
    return isinstance(decoder, dict) and decoder.get("type", None) == "ByteLevel"


def load_tokenizer(
    model_path, tokenizer_config_extra={}, return_tokenizer=True, eos_token_ids=None
):
    """
    加载HuggingFace分词器并自动选择合适的流式解分词器
    
    这个函数是模块的主要入口点，它：
    1. 加载指定路径的分词器
    2. 分析分词器配置，自动选择最佳的流式解分词器
    3. 创建包装器实例
    4. 可选地只返回解分词器类

    参数:
        model_path: 模型路径（Path对象）
        tokenizer_config_extra: 额外的分词器配置参数
        return_tokenizer: 是否返回完整的分词器包装器（True）或只返回解分词器类（False）
        eos_token_ids: EOS token ID列表（可选）
        
    返回:
        TokenizerWrapper 或 流式解分词器类
        
    注意：
        为了使用快速流式分词器，建议传递本地文件路径而不是Hugging Face仓库ID。
    """
    detokenizer_class = NaiveStreamingDetokenizer

    # 尝试读取tokenizer.json文件来分析解码器类型
    tokenizer_file = model_path / "tokenizer.json"
    if tokenizer_file.exists():
        with open(tokenizer_file, "r", encoding="utf-8") as fid:
            try:
                tokenizer_content = json.load(fid)
            except JSONDecodeError as e:
                raise JSONDecodeError("Failed to parse tokenizer.json", e.doc, e.pos)

        # 根据解码器类型选择合适的流式解分词器
        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = SPMStreamingDetokenizer
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
            elif _is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = BPEStreamingDetokenizer

    # 确保eos_token_ids是列表格式
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    # 根据参数返回不同的对象
    if return_tokenizer:
        return TokenizerWrapper(
            AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra),
            detokenizer_class,
            eos_token_ids=eos_token_ids,
        )
    else:
        return detokenizer_class


def no_bos_or_eos(sequence: List, bos: int, eos: int) -> List:
    """
    移除序列中的BOS和EOS token
    
    这个辅助函数用于清理token序列，移除可能存在的开头（BOS）和结尾（EOS）标记。
    
    参数:
        sequence: token序列
        bos: BOS token ID
        eos: EOS token ID
        
    返回:
        List[int]: 清理后的token序列
        
    处理逻辑：
    1. 如果序列以BOS token开头，移除它
    2. 如果序列以EOS token结尾，移除它
    3. 返回清理后的序列
    """
    removed_bos = sequence if sequence[0] != bos else sequence[1:]
    return removed_bos[:-1] if removed_bos[-1] == eos else removed_bos
