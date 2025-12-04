# Copyright © 2025 Apple Inc.

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

# Specify the checkpoint
checkpoint = "/Volumes/Data/lmstudio/lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit"

# Load the corresponding model and tokenizer
model, tokenizer = load(path_or_hf_repo=checkpoint)

# A batch of prompts
prompts = [
    "写一个关于狐狸的故事。",
    "写一个关于狐狸的故事。",
    "天空为什么是蓝色的?",
    "现在几点?",
    "珠穆朗玛峰有多高?",
]

sampler=make_sampler(temp=0.6, top_k=20, top_p=0.95)
# Apply the chat template and encode to tokens
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        add_generation_prompt=True,
    )
    for p in prompts
]

# Set `verbose=True` to see generation statistics
result = batch_generate(model, tokenizer, prompts, verbose=True, max_tokens=4096,sampler=sampler)

# The returned result contains texts completions in the same order as prompts
for result_str in result.texts:
    print("===")
    print(result_str)

# Example using prompt caches for faster generation
prompts2 = [
    "Could you summarize that?",
    "And what about the sea?",
    "Try again?",
    "And Mt Olympus?",
]
prompts2 = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        add_generation_prompt=True,
    )
    for p in prompts2
]

result2 = batch_generate(
    model, tokenizer, prompts2, verbose=False, prompt_caches=result.caches
)
print(result2.texts[-1])
