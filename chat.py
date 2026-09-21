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


def resolve_checkpoint() -> str:
    for path in ["checkpoints/best_model.pt", "checkpoints/latest.pt"]:
        if os.path.exists(path):
            return path
    print("❌ Error: No trained checkpoints found in 'checkpoints/'.")
    print("👉 Train the model first by running: python3 train.py")
    sys.exit(1)


def main():
    device = get_device()
    checkpoint_path = resolve_checkpoint()

    meta_path = "data/meta.pkl"
    if not os.path.exists(meta_path):
        print(f"❌ Error: Tokenizer metadata '{meta_path}' not found.")
        print("👉 Train the model first to generate vocabulary.")
        sys.exit(1)

    try:
        tokenizer = CharTokenizer.load(meta_path)
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        sys.exit(1)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = SynorLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except Exception as e:
        print(f"❌ Error loading checkpoint '{checkpoint_path}': {e}")
        sys.exit(1)

    sampler = TextSampler(model=model, tokenizer=tokenizer, device=device)

    print("=" * 60)
    print("🤖 Welcome to Synor AI Interactive Console")
    print(f"⚡ Device: {str(device).upper()} | Checkpoint: {checkpoint_path}")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to end.")
    print("=" * 60)

    def stream_char(c: str):
        sys.stdout.write(c)
        sys.stdout.flush()

    while True:
        try:
            prompt = input("\n👤 You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            print("🧠 Synor: ", end="", flush=True)
            sampler.generate(
                prompt=prompt,
                max_new_tokens=250,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                stream_callback=stream_char,
            )
            print()

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session ended.")
            break


if __name__ == "__main__":
    main()
