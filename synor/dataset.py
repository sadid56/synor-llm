import glob
import os
from typing import Dict, Optional, Tuple
import torch
from synor.tokenizer import BaseTokenizer


class TextDataset:
    """
    Scans raw text files, tokenizes text, and provides train/val data batches.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        tokenizer: Optional[BaseTokenizer] = None,
        train_split: float = 0.9,
    ):
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"Data directory '{raw_dir}' does not exist.")

        txt_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in '{raw_dir}'. Please add training corpora to this directory."
            )

        self.txt_files = txt_files
        self.raw_dir = raw_dir

        raw_texts = []
        for path in self.txt_files:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    if content.strip():
                        raw_texts.append(content)
            except Exception as e:
                raise IOError(f"Error reading file '{path}': {e}") from e

        if not raw_texts:
            raise ValueError(f"All .txt files in '{raw_dir}' are empty.")

        self.full_text = "\n\n".join(raw_texts)

        if tokenizer is None:
            raise ValueError("Tokenizer instance must be provided to TextDataset.")
        self.tokenizer = tokenizer

        if getattr(self.tokenizer, "vocab_size", 0) == 0 and hasattr(self.tokenizer, "fit"):
            self.tokenizer.fit(self.full_text)

        tokens = self.tokenizer.encode(self.full_text)
        self.data_tensor = torch.tensor(tokens, dtype=torch.long)

        split_idx = int(len(self.data_tensor) * train_split)
        self.train_data = self.data_tensor[:split_idx]
        self.val_data = self.data_tensor[split_idx:]

        if len(self.train_data) < 100:
            raise ValueError("Training data too small (< 100 tokens). Provide a larger corpus.")

    def get_batch(
        self, split: str = "train", batch_size: int = 32, block_size: int = 128, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        if len(data) <= block_size:
            raise ValueError(
                f"{split} dataset length ({len(data)}) must be strictly greater than block_size ({block_size})."
            )

        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

        if device != "cpu":
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        return x, y

    def stats(self) -> Dict[str, int]:
        return {
            "total_chars": len(self.full_text),
            "total_tokens": len(self.data_tensor),
            "train_tokens": len(self.train_data),
            "val_tokens": len(self.val_data),
            "vocab_size": getattr(self.tokenizer, "vocab_size", 0),
            "num_files": len(self.txt_files),
        }
