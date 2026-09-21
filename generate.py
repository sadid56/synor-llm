#!/usr/bin/env python3
"""
Synor AI — CLI Text Generation Engine.
"""

import argparse
import os
import sys
import torch

from synor.model import SynorLM
from synor.sampler import TextSampler
from synor.tokenizer import CharTokenizer
from synor.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text continuation using trained Synor Foundation Model"
    )
    parser.add_argument("--prompt", type=str, default="", help="Prompt text to condition generation")
    parser.add_argument("--tokens", type=int, default=300, help="Number of characters to generate (default: 300)")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (0.0 = greedy, 1.0 = creative)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling constraint (default: 40)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus sampling constraint (default: 0.9)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to checkpoint (default: checkpoints/best_model.pt)",
    )
    return parser.parse_args()


from synor.logger import chalk, print_banner, log_info, log_error


def resolve_checkpoint(checkpoint_path: str) -> str:
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    fallback = "checkpoints/latest.pt"
    if os.path.exists(fallback):
        return fallback
    log_error(f"No checkpoint found at '{checkpoint_path}' or '{fallback}'.")
    print(f"  {chalk.dim('👉 Train the model first:')} {chalk.bold.cyan('python3 train.py')}")
    sys.exit(1)


def main():
    args = parse_args()
    device = get_device()
    checkpoint_file = resolve_checkpoint(args.checkpoint)

    meta_path = "data/meta.pkl"
    if not os.path.exists(meta_path):
        log_error(f"Tokenizer metadata '{meta_path}' not found.")
        print(f"  {chalk.dim('👉 Run training once to generate vocabulary.')}")
        sys.exit(1)

    try:
        tokenizer = CharTokenizer.load(meta_path)
    except Exception as e:
        log_error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = SynorLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except Exception as e:
        log_error(f"Failed to load weights from '{checkpoint_file}': {e}")
        sys.exit(1)

    sampler = TextSampler(model=model, tokenizer=tokenizer, device=device)

    print_banner(
        "Synor AI — Generation Engine",
        "Autonomous Streaming Completion",
        {
            "Checkpoint": checkpoint_file,
            "Device": str(device).upper(),
            "Temperature": args.temp,
            "Top-K / Top-P": f"{args.top_k} / {args.top_p}",
        },
    )

    if args.prompt:
        sys.stdout.write(chalk.bold.bright_cyan(args.prompt))
        sys.stdout.flush()

    def stream_char(c: str):
        sys.stdout.write(chalk.bright_white(c))
        sys.stdout.flush()

    try:
        sampler.generate(
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            stream_callback=stream_char,
        )
    except KeyboardInterrupt:
        pass

    print(f"\n\n{chalk.dim('─' * 62)}\n")


if __name__ == "__main__":
    main()
