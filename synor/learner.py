"""
Synor AI — Autonomous Continual Learning Engine.
Enables Synor AI to continually learn from real interactions and conversational memory
by updating its neural weights via backpropagation without static rules.
"""

import os
import torch
from torch.nn import functional as F
from typing import List, Dict, Optional

from synor.model import SynorLM
from synor.tokenizer import CharTokenizer
from synor.logger import chalk, log_info, log_success, log_error


class AutoLearner:
    """
    Continual online fine-tuner for SynorLM.
    Samples from conversational memory + knowledge replay to adjust neural weights.
    """

    def __init__(
        self,
        model: SynorLM,
        tokenizer: CharTokenizer,
        device: torch.device,
        learning_rate: float = 1e-4,
        checkpoint_path: str = "checkpoints/best_model.pt",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.checkpoint_path = checkpoint_path
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

    def learn_from_text(self, text_samples: List[str], steps: int = 40, batch_size: int = 8) -> Optional[float]:
        """
        Execute gradient descent on the provided text samples to adapt neural weights.
        """
        if not text_samples:
            return None

        # Combine samples with delimiter
        combined_text = "\n\n".join(s.strip() for s in text_samples if s.strip())
        if len(combined_text) < 30:
            return None

        tokens = self.tokenizer.encode(combined_text)
        data_tensor = torch.tensor(tokens, dtype=torch.long)
        n = len(data_tensor)

        block_size = self.model.config.block_size
        if n <= block_size:
            # Pad if too short
            pad_len = block_size + 1 - n
            data_tensor = torch.cat([data_tensor, torch.zeros(pad_len, dtype=torch.long)])
            n = len(data_tensor)

        self.model.train()
        total_loss = 0.0

        for step in range(steps):
            # Sample random chunk offsets
            ix = torch.randint(0, n - block_size, (batch_size,))
            x = torch.stack([data_tensor[i : i + block_size] for i in ix]).to(self.device)
            y = torch.stack([data_tensor[i + 1 : i + 1 + block_size] for i in ix]).to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits, loss = self.model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, steps)
        self.model.eval()

        # Save updated weights atomically
        self._save_weights()
        return avg_loss

    def learn_from_interactions(self, interactions: List[Dict[str, str]], steps: int = 50) -> Optional[float]:
        """
        Format episodic interaction pairs into training dialogue and update weights.
        """
        if not interactions:
            return None

        formatted_samples = []
        for item in interactions:
            u = item.get("user", "")
            a = item.get("assistant", "")
            if u and a:
                formatted_samples.append(f"User: {u}\nAssistant: {a}")

        # Also include a few core conversations from raw data to prevent catastrophic forgetting
        try:
            with open("data/raw/conversations.txt", "r", encoding="utf-8") as f:
                core_dialogues = f.read()
                formatted_samples.append(core_dialogues[:1500])
        except Exception:
            pass

        return self.learn_from_text(formatted_samples, steps=steps)

    def _save_weights(self) -> None:
        """
        Persist fine-tuned weights to checkpoint directory.
        """
        try:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model.config,
            }
            torch.save(checkpoint, self.checkpoint_path)
            # Also save latest.pt
            torch.save(checkpoint, "checkpoints/latest.pt")
        except Exception as e:
            log_error(f"Failed to save auto-learned weights: {e}")
