#!/usr/bin/env python3
import argparse
import math
import os
import torch
from synor.config import get_preset
from synor.model import SynorLM
from synor.bpe_tokenizer import BPETokenizer
from synor.dataset import SFTDataset
from synor.utils import get_device


def evaluate(checkpoint_path: str = "checkpoints/best_model.pt", config_name: str = "100m", sft_dir: str = "data/sft"):
    device = get_device()
    tokenizer = BPETokenizer()

    cfg = get_preset(config_name)
    cfg.vocab_size = tokenizer.vocab_size

    model = SynorLM(cfg).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        return

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Model parameters: {model.get_num_params():,}")

    dataset = SFTDataset(sft_dir=sft_dir, tokenizer=tokenizer)
    eval_iters = 50
    batch_size = 4
    losses = []

    print(f"\nEvaluating validation loss on {eval_iters} batches...")
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = dataset.get_batch("val", batch_size, cfg.block_size, str(device))
            _, loss = model(x, y)
            losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(min(mean_loss, 20))

    print("\n" + "─" * 40)
    print(f"  Validation Loss: {mean_loss:.4f}")
    print(f"  Perplexity (PPL): {perplexity:.2f}")
    print("─" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config", default="100m")
    parser.add_argument("--sft-dir", default="data/sft")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.config, args.sft_dir)
