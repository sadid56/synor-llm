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


from synor.logger import (
    chalk,
    print_banner,
    log_info,
    log_success,
    log_warn,
    log_error,
)


def main():
    args = parse_args()
    device = get_device()

    print_banner(
        "Synor AI — Training Pipeline",
        "Generative Pretrained Transformer & Supervised Fine-Tuning",
        {
            "Compute Device": str(device).upper(),
            "Preset Scale": args.config.upper(),
            "Learning Rate": args.lr,
            "Target Steps": args.iters,
            "Batch Size": args.batch_size,
        },
    )

    # 1. Tokenizer management
    meta_path = "data/meta.pkl"
    if os.path.exists(meta_path) and args.resume:
        try:
            tokenizer = CharTokenizer.load(meta_path)
            log_info(f"Loaded existing vocabulary: {chalk.bold.yellow(f'{tokenizer.vocab_size} tokens')} from '{meta_path}'")
        except Exception as e:
            log_warn(f"Could not load {meta_path}: {e}. Initializing new tokenizer.")
            tokenizer = CharTokenizer()
    else:
        tokenizer = CharTokenizer()

    # 2. Dataset loading and verification
    try:
        dataset = TextDataset(raw_dir=args.data_dir, tokenizer=tokenizer)
    except Exception as e:
        log_error(f"Dataset Error: {e}")
        sys.exit(1)

    stats = dataset.stats()
    n_files = stats["num_files"]
    t_chars = stats["total_chars"]
    v_size = stats["vocab_size"]
    log_info(
        f"Corpus Loaded: {chalk.bold.yellow(f'{n_files} file(s)')} | "
        f"{chalk.bold.white(f'{t_chars:,}')} chars | "
        f"Vocab: {chalk.bold.cyan(str(v_size))}"
    )

    try:
        tokenizer.save(meta_path)
    except Exception as e:
        log_warn(f"Could not save tokenizer to {meta_path}: {e}")

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
    log_info(f"Model Initialized: {chalk.bold.bright_green(f'{model.get_num_params():,}')} trainable parameters")

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
            log_success(
                f"Resumed from 'checkpoints/latest.pt' at Step {chalk.bold.white(f'{trainer.iter_num:,}')} "
                f"(Best Val Loss: {chalk.bold.green(f'{trainer.best_val_loss:.4f}')})"
            )
        else:
            log_info("No existing checkpoint found in 'checkpoints/latest.pt'. Training from scratch.")

    print(f"\n{chalk.bold.cyan('─' * 62)}")
    trainer.train(
        additional_iters=args.iters,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
    )
    print(f"{chalk.bold.cyan('─' * 62)}\n")

    log_success("Training pipeline finished successfully!")
    print(f"  {chalk.dim('👉 Run chat:')}     {chalk.bold.cyan('python3 chat.py')}")
    print(f"  {chalk.dim('👉 Generate:')}     {chalk.bold.cyan('python3 generate.py --prompt \"User: Hi\"')}\n")


if __name__ == "__main__":
    main()
