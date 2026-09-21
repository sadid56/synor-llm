"""
Synor AI — Byte-Pair Encoding (BPE) Sub-Word Tokenizer.
Fast sub-word tokenization using standard GPT-compatible byte BPE.
Maps words and subwords into semantic token IDs.
"""

import os
import pickle
from typing import List, Optional
import tiktoken

from synor.tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    Sub-word Byte-Pair Encoding (BPE) Tokenizer for SynorLM.
    Features 50,257 standard sub-word vocabulary.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.eot_token = self.enc.eot_token

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        """Encode string to list of BPE token IDs."""
        if not text:
            return []
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: List[int]) -> str:
        """Decode list of BPE token IDs back to string."""
        if not tokens:
            return ""
        return self.enc.decode(tokens)

    def save(self, filepath: str) -> None:
        """Persist tokenizer configuration metadata."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "type": "bpe",
            "encoding_name": self.encoding_name,
            "vocab_size": self.vocab_size,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Load tokenizer instance from metadata."""
        if not os.path.exists(filepath):
            # If not yet saved to disk, default to gpt2 encoding
            return cls(encoding_name="gpt2")
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            encoding_name = data.get("encoding_name", "gpt2")
            return cls(encoding_name=encoding_name)
        except Exception:
            return cls(encoding_name="gpt2")
