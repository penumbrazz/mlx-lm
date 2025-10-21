"""Run four batched generations with varying sampling settings."""

import argparse

import mlx.core as mx

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler


def main(model_name: str, max_tokens: int = 64) -> None:
    mx.random.seed(42)
    model, tokenizer = load(path_or_hf_repo=model_name)
    print(f"Using model: {model_name}")

    base_prompts = [
        "Write a story about 中国.",
    "Why is the sky blue?",
    "现在几点?",
    "珠穆朗玛峰有多高?",
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        for prompt in base_prompts
    ]

    sampler_configs = [
        {"temperature": 1.0, "top_k": 60, "min_p": 0.0, "top_p": 0.8},
        {"temperature": 2.0, "top_k": 70},
        {"temperature": 3.0, "top_k": 80},
    ]

    for idx, config in enumerate(sampler_configs, start=1):
        print(
            f"===== Batch {idx}: temp={config['temperature']}, top_k={config['top_k']}"
        )
        sampler = make_sampler(
            temp=config["temperature"],
            top_k=config["top_k"],
        )
        result = batch_generate(
            model,
            tokenizer,
            prompts,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        for prompt_text, completion in zip(base_prompts, result.texts):
            print(f"Prompt: {prompt_text}\n{completion}")
            print("-" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/Volumes/Data/lmstudio/lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit",
        help="Model to load",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per response",
    )
    args = parser.parse_args()
    main(args.model, max_tokens=args.max_tokens)