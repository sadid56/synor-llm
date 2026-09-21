#!/usr/bin/env python3
"""
Synor AI — Interactive Console REPL.
"""

import os
import sys
import torch

from synor.model import SynorLM
from synor.sampler import TextSampler
from synor.tokenizer import CharTokenizer
from synor.utils import get_device


from synor.logger import chalk, print_banner, log_error


def resolve_checkpoint() -> str:
    for path in ["checkpoints/best_model.pt", "checkpoints/latest.pt"]:
        if os.path.exists(path):
            return path
    log_error("No trained checkpoints found in 'checkpoints/'.")
    print(f"  {chalk.dim('👉 Train the model first:')} {chalk.bold.cyan('python3 train.py')}")
    sys.exit(1)


def main():
    device = get_device()
    checkpoint_path = resolve_checkpoint()

    meta_path = "data/meta.pkl"
    if not os.path.exists(meta_path):
        log_error(f"Tokenizer metadata '{meta_path}' not found.")
        print(f"  {chalk.dim('👉 Train the model first to generate vocabulary.')}")
        sys.exit(1)

    try:
        tokenizer = CharTokenizer.load(meta_path)
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
        "Synor AI — Interactive Console",
        "Real-Time Conversational REPL",
        {
            "Checkpoint": checkpoint_path,
            "Compute Device": str(device).upper(),
            "Architecture": f"{config.n_layer} Layers | {config.n_head} Heads | {config.n_embd} Dim",
            "Controls": "Type prompt + Enter. Type 'exit' or 'quit' to end.",
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
            if prompt.lower() in ["exit", "quit"]:
                print(f"\n{chalk.bold.yellow('👋 Session ended. Goodbye!')}\n")
                break

            # Normalize common punctuation spacing like "Who are you ?" -> "Who are you?"
            clean_prompt = prompt.replace(" ?", "?").replace(" !", "!")

            # Format prompt for conversational model
            formatted_prompt = (
                clean_prompt
                if clean_prompt.startswith("User:")
                else f"User: {clean_prompt}\nAssistant: "
            )

            print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} ", end="", flush=True)
            sampler.generate(
                prompt=formatted_prompt,
                max_new_tokens=150,
                temperature=0.3,
                top_k=40,
                top_p=0.9,
                stop_strings=["\nUser:", "\n\n", "User:"],
                stream_callback=stream_char,
            )
            print()

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{chalk.bold.yellow('👋 Session ended. Goodbye!')}\n")
            break


if __name__ == "__main__":
    main()
