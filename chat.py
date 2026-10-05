#!/usr/bin/env python3
"""
Synor AI — Autonomous Interactive Console REPL.
100% Pure Neural Vanilla Transformer Generation.
Direct autoregressive decoding from model weights.
"""

import os
import sys
import torch

from synor.model import SynorLM
from synor.sampler import TextSampler
from synor.tokenizer import BaseTokenizer, CharTokenizer, load_tokenizer
from synor.utils import get_device
from synor.logger import chalk, print_banner, log_info, log_success, log_error


def resolve_checkpoint() -> str:
    if os.path.exists("checkpoints/best_model.pt"):
        return "checkpoints/best_model.pt"
    if os.path.exists("checkpoints/latest.pt"):
        return "checkpoints/latest.pt"
    log_error("No trained checkpoints found in 'checkpoints/'.")
    print(f"  {chalk.dim('👉 Train the model first:')} {chalk.bold.cyan('python3 train.py')}")
    sys.exit(1)


def main():
    device = get_device()
    checkpoint_path = resolve_checkpoint()

    meta_path = "data/meta.pkl"
    if not os.path.exists(meta_path):
        log_error(f"Tokenizer metadata '{meta_path}' not found.")
        print(f"  {chalk.dim('👉 Run python3 train.py to initialize vocabulary.')}")
        sys.exit(1)

    try:
        tokenizer = load_tokenizer(meta_path)
    except Exception as e:
        log_error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = SynorLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except Exception as e:
        log_error(f"Failed to load checkpoint '{checkpoint_path}': {e}")
        sys.exit(1)

    sampler = TextSampler(model=model, tokenizer=tokenizer, device=device)

    print_banner(
        "Synor AI — Pure Neural Companion",
        "Pure Autoregressive Transformer Generation (Vanilla Brain)",
        {
            "Checkpoint": checkpoint_path,
            "Compute Device": str(device).upper(),
            "Architecture": f"{model.get_num_params():,} Parameters (Zero Heuristics)",
            "Commands": "clear (clear screen) │ exit (quit)",
        },
    )

    def stream_char(c: str):
        sys.stdout.write(chalk.bright_white(c))
        sys.stdout.flush()

    while True:
        try:
            prompt_input = input(f"\n{chalk.bold.bg_magenta.white(' USER ')} ")
            prompt = prompt_input.strip()
            if not prompt:
                continue

            # Command: exit / quit
            if prompt.lower() in ["exit", "quit", "/exit", "/quit"]:
                print(f"\n{chalk.bold.yellow('👋 Session ended. Catch you later, bro!')}\n")
                break

            # Command: clear screen
            if prompt.lower() in ["clear", "cls", "/clear"]:
                os.system("clear" if os.name != "nt" else "cls")
                continue

            # Format dialogue prompt for pure neural generation
            formatted_prompt = f"User: {prompt}\nAssistant: "

            print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} ", end="", flush=True)

            generated_reply = sampler.generate(
                prompt=formatted_prompt,
                max_new_tokens=140,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2,
                stop_strings=["\n\n", "\nUser:", "\nAssistant:", "\n\nUser:", "<|endoftext|>"],
                stream_callback=stream_char,
            )
            print()

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{chalk.bold.yellow('👋 Session ended. Catch you later, bro!')}\n")
            break


if __name__ == "__main__":
    main()
