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

        if os.path.isfile(raw_dir):
            txt_files = [raw_dir]
        else:
            txt_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in '{raw_dir}'. Please add training corpora."
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


class SFTDataset:
    """
    Supervised Fine-Tuning (SFT) dataset for multi-turn and single-turn dialogues.
    Every dialogue sample starts cleanly at Position 0 with prompt tokens masked (-100).
    Loss is calculated exclusively on Assistant completions followed by <|endoftext|>.
    """

    def __init__(
        self,
        sft_dir: str = "data/sft",
        tokenizer: Optional[BaseTokenizer] = None,
        train_split: float = 0.9,
    ):
        if not os.path.exists(sft_dir):
            raise FileNotFoundError(f"SFT directory '{sft_dir}' does not exist.")

        if os.path.isfile(sft_dir):
            txt_files = [sft_dir]
            cache_file = sft_dir + ".cache.pt"
        else:
            txt_files = sorted(glob.glob(os.path.join(sft_dir, "*.txt")))
            cache_file = os.path.join(sft_dir, "sft_cache.pt")

        if not txt_files:
            raise FileNotFoundError(
                f"No .txt dialogue files found in '{sft_dir}'. Please provide SFT dialogue data."
            )

        self.txt_files = txt_files
        self.sft_dir = sft_dir

        if tokenizer is None:
            raise ValueError("Tokenizer instance must be provided to SFTDataset.")
        self.tokenizer = tokenizer
        self.eot_token = getattr(self.tokenizer, "eot_token", 50256)

        # Check if cache exists and is newer than all txt files
        use_cache = False
        if os.path.exists(cache_file):
            cache_mtime = os.path.getmtime(cache_file)
            if all(os.path.getmtime(p) <= cache_mtime for p in self.txt_files):
                use_cache = True

        if use_cache:
            try:
                cached_data = torch.load(cache_file, map_location="cpu", weights_only=False)
                self.samples = cached_data["samples"]
                self.total_tokens = cached_data["total_tokens"]
            except Exception:
                use_cache = False

        if not use_cache:
            self.samples = []
            total_tokens = 0

            for path in self.txt_files:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                dialogues = self._parse_dialogues(content)
                for user_msg, assistant_msg in dialogues:
                    prompt_text = f"User: {user_msg.strip()}\nAssistant: "
                    resp_text = f" {assistant_msg.strip()}<|endoftext|>"

                    p_tokens = self.tokenizer.encode(prompt_text)
                    r_tokens = self.tokenizer.encode(resp_text)

                    if not r_tokens:
                        continue

                    seq = p_tokens + r_tokens
                    tgt = ([-100] * len(p_tokens)) + r_tokens
                    total_tokens += len(seq)
                    self.samples.append((seq, tgt))

            if not self.samples:
                raise ValueError(f"No valid dialogue pairs extracted from '{sft_dir}'.")

            self.total_tokens = total_tokens
            try:
                torch.save({"samples": self.samples, "total_tokens": self.total_tokens}, cache_file)
            except Exception:
                pass

        split_idx = max(1, int(len(self.samples) * train_split))
        self.train_samples = self.samples[:split_idx]
        self.val_samples = self.samples[split_idx:] if split_idx < len(self.samples) else self.samples

    def _parse_dialogues(self, text: str) -> list:
        """Parse text file containing 'User: ...' and 'Assistant: ...' pairs."""
        dialogues = []
        blocks = text.split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            u_lines = []
            a_lines = []
            is_u = False
            is_a = False

            for line in lines:
                sline = line.strip()
                if sline.startswith("User:") or sline.startswith("user:"):
                    is_u = True
                    is_a = False
                    u_lines.append(sline[5:].strip())
                elif sline.startswith("Assistant:") or sline.startswith("assistant:") or sline.startswith("Synor:"):
                    prefix_len = 10 if sline.startswith("Assistant:") or sline.startswith("assistant:") else 6
                    is_a = True
                    is_u = False
                    a_lines.append(sline[prefix_len:].strip())
                elif is_u:
                    u_lines.append(sline)
                elif is_a:
                    a_lines.append(sline)

            u_text = " ".join(u_lines).strip()
            a_text = "\n".join(a_lines).strip()
            if u_text and a_text:
                dialogues.append((u_text, a_text))

        return dialogues

    def get_batch(
        self, split: str = "train", batch_size: int = 4, block_size: int = 512, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pool = self.train_samples if split == "train" else self.val_samples
        indices = torch.randint(0, len(pool), (batch_size,)).tolist()
        batch_items = [pool[idx] for idx in indices]

        raw_max = max(len(item[0]) for item in batch_items) + 1
        pad_target_len = min(block_size, raw_max)

        batch_x = []
        batch_y = []
        for seq, tgt in batch_items:
            s = seq[:pad_target_len]
            t = tgt[:pad_target_len]
            pad_len = pad_target_len - len(s)
            padded_x = s + [self.eot_token] * pad_len
            padded_y = t + [-100] * pad_len
            batch_x.append(padded_x)
            batch_y.append(padded_y)

        bx = torch.tensor(batch_x, dtype=torch.long, device=device)
        by = torch.tensor(batch_y, dtype=torch.long, device=device)

        x = bx[:, :-1].contiguous()
        y = by[:, 1:].contiguous()
        return x, y

    def stats(self) -> Dict[str, int]:
        return {
            "total_tokens": self.total_tokens,
            "train_pairs": len(self.train_samples),
            "val_pairs": len(self.val_samples),
            "dialogue_pairs": len(self.samples),
            "vocab_size": getattr(self.tokenizer, "vocab_size", 0),
            "num_files": len(self.txt_files),
        }

