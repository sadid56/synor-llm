#!/usr/bin/env python3
"""
Synor AI — Training and Continuous Learning CLI.
"""

import argparse
import os
import sys

from synor.config import SynorConfig, get_preset
from synor.dataset import TextDataset
from synor.model import SynorLM
from synor.tokenizer import CharTokenizer
from synor.trainer import Trainer
from synor.utils import get_device, setup_logger

logger = setup_logger("synor.train")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or resume pretraining the Synor Foundation Model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium", "large"],
        help="Model preset (default: tiny)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Number of training steps to run (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per step (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Peak learning rate for AdamW (default: 5e-4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from 'checkpoints/latest.pt' if it exists",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to folder containing .txt training corpora (default: data/raw)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=200,
        help="Interval of steps between validation loss calculations (default: 200)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print("=" * 60)
    print("🧠 Synor AI — Foundation Model Training Pipeline")
    print(f"⚡ Compute Device: {str(device).upper()}")
    print(f"📦 Preset Scale: {args.config.upper()}")
    print("=" * 60)

    # 1. Tokenizer management
    meta_path = "data/meta.pkl"
    if os.path.exists(meta_path) and args.resume:
        try:
            tokenizer = CharTokenizer.load(meta_path)
            print(f"📖 Loaded existing vocabulary ({tokenizer.vocab_size} tokens) from {meta_path}")
        except Exception as e:
            logger.warning(f"Could not load {meta_path}: {e}. Initializing new tokenizer.")
            tokenizer = CharTokenizer()
    else:
        tokenizer = CharTokenizer()

    # 2. Dataset loading and verification
    try:
        dataset = TextDataset(raw_dir=args.data_dir, tokenizer=tokenizer)
    except Exception as e:
        print(f"\n❌ Dataset Error: {e}")
        sys.exit(1)

    stats = dataset.stats()
    print(f"📂 Corpus Statistics:")
    print(f"   - Files: {stats['num_files']}")
    print(f"   - Total characters: {stats['total_chars']:,}")
    print(f"   - Total tokens: {stats['total_tokens']:,}")
    print(f"   - Vocabulary size: {stats['vocab_size']}")

    try:
        tokenizer.save(meta_path)
    except Exception as e:
        logger.warning(f"Could not save tokenizer to {meta_path}: {e}")

    # 3. Model instantiation
    preset = get_preset(args.config)
    config = SynorConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=preset.block_size,
        n_embd=preset.n_embd,
        n_head=preset.n_head,
        n_layer=preset.n_layer,
        dropout=preset.dropout,
    )

    model = SynorLM(config)
    print(f"🧩 Model Architecture: {model.get_num_params():,} trainable parameters")

    # 4. Trainer execution
    trainer = Trainer(
        model=model,
        dataset=dataset,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir="checkpoints",
    )

    if args.resume:
        if trainer.load_checkpoint("latest.pt"):
            print(
                f"🔄 Resumed from 'checkpoints/latest.pt' at Step {trainer.iter_num:,} "
                f"(Best Val Loss: {trainer.best_val_loss:.4f})"
            )
        else:
            print("ℹ️  No valid checkpoint found in 'checkpoints/latest.pt'. Training from scratch.")

    trainer.train(
        additional_iters=args.iters,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
    )

    print("\n🎉 Training finished!")
    print("👉 Generate text: python3 generate.py --prompt 'ROMEO:'")
    print("👉 Chat with AI:  python3 chat.py")


if __name__ == "__main__":
    main()
