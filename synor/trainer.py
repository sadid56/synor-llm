import math
import os
import sys
import time
from typing import Dict, Optional
import torch
from torch.nn.utils import clip_grad_norm_

from synor.dataset import TextDataset
from synor.model import SynorLM
from synor.utils import save_checkpoint_atomic, setup_logger
from synor.logger import chalk, log_step, log_success, log_info, log_warn

logger = setup_logger("synor.trainer")


class Trainer:
    """
    Production-grade trainer for Synor foundation models.
    Supports continuous learning, cosine LR decay, and atomic state saving.
    """

    def __init__(
        self,
        model: SynorLM,
        dataset: TextDataset,
        device: torch.device,
        learning_rate: float = 5e-4,
        min_lr: float = 5e-5,
        warmup_iters: int = 100,
        weight_decay: float = 1e-2,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters

        # Separate weight decay for 2D weight matrices (linear layers) vs 1D biases and layer norms
        decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
        nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8
        )

        self.iter_num = 0
        self.best_val_loss = float("inf")

    def get_lr(self, it: int, max_iters: int) -> float:
        if it < self.warmup_iters:
            return self.learning_rate * (it + 1) / (self.warmup_iters + 1)
        if it > max_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (max_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    @torch.no_grad()
    def estimate_loss(self, eval_iters: int = 20, batch_size: int = 32) -> Dict[str, float]:
        self.model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = self.dataset.get_batch(
                    split, batch_size, self.model.config.block_size, str(self.device)
                )
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
        return out

    def save_checkpoint(self, filename: str) -> str:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.model.config,
        }
        target_path = os.path.join(self.checkpoint_dir, filename)
        save_checkpoint_atomic(checkpoint, target_path)
        return target_path

    def load_checkpoint(self, filename: str) -> bool:
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            return False

        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.iter_num = ckpt.get("iter_num", 0)
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            return True
        except Exception as e:
            logger.warning(f"Failed to load checkpoint '{path}': {e}. Starting fresh.")
            return False

    def train(
        self,
        additional_iters: int = 1000,
        batch_size: int = 32,
        eval_interval: int = 200,
        eval_iters: int = 20,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
    ) -> None:
        start_iter = self.iter_num
        max_iters = start_iter + additional_iters

        print(f"\n🚀 Training started from Step {start_iter:,} ➡️ Target: Step {max_iters:,}")
        self.model.train()
        t0 = time.time()

        try:
            for step in range(start_iter, max_iters):
                self.iter_num = step + 1

                # Update learning rate per schedule
                lr = self.get_lr(step, max_iters)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                self.optimizer.zero_grad(set_to_none=True)
                for _ in range(grad_accum_steps):
                    x, y = self.dataset.get_batch(
                        "train", batch_size, self.model.config.block_size, str(self.device)
                    )
                    _, loss = self.model(x, y)
                    if grad_accum_steps > 1:
                        loss = loss / grad_accum_steps
                    loss.backward()

                if grad_clip > 0.0:
                    clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()

                # Periodic cache cleanup to prevent MPS / CUDA memory fragmentation
                if (step + 1) % 5 == 0:
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                    elif self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # Periodic evaluation and checkpointing
                if (step + 1) % eval_interval == 0 or (step + 1) == max_iters:
                    dt = time.time() - t0
                    losses = self.estimate_loss(eval_iters, batch_size)
                    val_loss = losses["val"]

                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pt")

                    # Calculate dataset progress / percentage
                    data_pct = None
                    if hasattr(self.dataset, "samples") and len(self.dataset.samples) > 0:
                        total_samples_seen = (step + 1) * batch_size * grad_accum_steps
                        data_pct = (total_samples_seen / len(self.dataset.samples)) * 100.0
                    elif hasattr(self.dataset, "train_data") and len(self.dataset.train_data) > 0:
                        total_tokens_seen = (step + 1) * batch_size * grad_accum_steps * self.model.config.block_size
                        data_pct = (total_tokens_seen / len(self.dataset.train_data)) * 100.0

                    self.save_checkpoint("latest.pt")
                    log_step(step + 1, max_iters, losses["train"], val_loss, lr, dt, is_best=is_best, data_pct=data_pct)
                    t0 = time.time()

        except KeyboardInterrupt:
            print("\n")
            log_warn("Training interrupted by user. Safely saving checkpoint...")
            self.save_checkpoint("latest.pt")
            log_success("Saved state to 'checkpoints/latest.pt'. Resume anytime using --resume.")
            sys.exit(0)

        print()
        log_success(f"Training completed! Best validation loss: {self.best_val_loss:.4f}")
        log_info(f"Checkpoints saved to: '{self.checkpoint_dir}/'")
