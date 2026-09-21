#!/usr/bin/env python3
"""
Synor AI — Training and Continuous Learning CLI.
"""

import argparse
import os
import sys

# Prevent Apple Silicon MPS memory allocator crashes on large models
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

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
        default="100m",
        choices=["tiny", "small", "medium", "large", "100m"],
        help="Model preset (default: 100m)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="auto",
        choices=["auto", "bpe", "char"],
        help="Tokenizer type (default: auto: bpe for 100m, char for tiny)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrain",
        choices=["pretrain", "sft"],
        help="Training stage: pretrain (next-token on corpus) or sft (dialogue with masked target loss)",
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
        default=4,
        help="Batch size per micro-step (default: 4 for 100M MPS training)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate for AdamW (default: 3e-4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from 'checkpoints/latest.pt' if it exists",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/pretrain",
        help="Path to folder containing .txt pretrain corpora (default: data/pretrain)",
    )
    parser.add_argument(
        "--sft-dir",
        type=str,
        default="data/sft",
        help="Path to folder containing .txt dialogue corpora for SFT (default: data/sft)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Interval of steps between validation loss calculations (default: 50)",
    )
    return parser.parse_args()


from synor.bpe_tokenizer import BPETokenizer
from synor.dataset import TextDataset, SFTDataset
from synor.tokenizer import BaseTokenizer, CharTokenizer, load_tokenizer
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

    active_dir = args.sft_dir if args.stage == "sft" else args.data_dir
    if not os.path.exists(active_dir):
        if args.stage == "pretrain" and os.path.exists("data/raw"):
            active_dir = "data/raw"

    print_banner(
        "Synor AI — 100M Foundation Training Pipeline",
        "Generative Pretrained Transformer (Zero 3rd-Party Models)",
        {
            "Compute Device": str(device).upper(),
            "Preset Scale": args.config.upper(),
            "Stage": f"{args.stage.upper()} ({'Masked Loss' if args.stage == 'sft' else 'Causal LM'})",
            "Learning Rate": args.lr,
            "Target Steps": args.iters,
            "Micro Batch Size": args.batch_size,
            "Grad Accum Steps": args.grad_accum_steps,
            "Effective Batch": args.batch_size * args.grad_accum_steps,
            "Active Directory": active_dir,
        },
    )

    # 1. Tokenizer management
    meta_path = "data/meta.pkl"
    use_bpe = args.tokenizer == "bpe" or (args.tokenizer == "auto" and args.config in ["100m", "large"])

    if use_bpe:
        tokenizer = BPETokenizer()
        log_info(f"Using Byte-Pair Encoding (BPE) Sub-Word Tokenizer: {chalk.bold.yellow(f'{tokenizer.vocab_size:,} vocab tokens')}")
    elif os.path.exists(meta_path) and args.resume:
        try:
            tokenizer = load_tokenizer(meta_path)
            log_info(f"Loaded existing vocabulary: {chalk.bold.yellow(f'{tokenizer.vocab_size} tokens')} from '{meta_path}'")
        except Exception as e:
            log_warn(f"Could not load {meta_path}: {e}. Initializing CharTokenizer.")
            tokenizer = CharTokenizer()
    else:
        tokenizer = CharTokenizer()

    # 2. Dataset loading and verification
    try:
        if args.stage == "sft":
            dataset = SFTDataset(sft_dir=active_dir, tokenizer=tokenizer)
            stats = dataset.stats()
            num_pairs = stats.get("dialogue_pairs", 0)
            num_f = stats.get("num_files", 1)
            tot_t = stats.get("total_tokens", 0)
            log_info(
                f"SFT Dialogues Loaded: {chalk.bold.yellow(f'{num_pairs} pairs')} across "
                f"{chalk.bold.white(f'{num_f} file(s)')} | "
                f"Tokens: {chalk.bold.white(f'{tot_t:,}')} | "
                f"Masked Prompt Targets: {chalk.bold.cyan('-100 (Loss exclusively on Assistant replies)')}"
            )
        else:
            dataset = TextDataset(raw_dir=active_dir, tokenizer=tokenizer)
            stats = dataset.stats()
            n_files = stats["num_files"]
            t_chars = stats["total_chars"]
            t_toks = stats["total_tokens"]
            v_size = stats["vocab_size"]
            log_info(
                f"Pretrain Corpus Loaded: {chalk.bold.yellow(f'{n_files} file(s)')} | "
                f"{chalk.bold.white(f'{t_chars:,}')} chars | "
                f"Tokens: {chalk.bold.white(f'{t_toks:,}')} | "
                f"Vocab: {chalk.bold.cyan(str(v_size))}"
            )
    except Exception as e:
        log_error(f"Dataset Error: {e}")
        sys.exit(1)

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
        grad_accum_steps=args.grad_accum_steps,
    )
    print(f"{chalk.bold.cyan('─' * 62)}\n")

    log_success("Training pipeline finished successfully!")
    print(f"  {chalk.dim('👉 Run chat:')}     {chalk.bold.cyan('python3 chat.py')}")
    print(f"  {chalk.dim('👉 Generate:')}     {chalk.bold.cyan('python3 generate.py --prompt \"User: Hi\"')}\n")


if __name__ == "__main__":
    main()
