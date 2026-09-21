from abc import ABC, abstractmethod
import os
import pickle
from typing import Dict, List, Optional, Union


class BaseTokenizer(ABC):
    """Abstract base class for all Synor tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseTokenizer":
        pass


class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer. Maps characters to unique integers.
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        self.chars: List[str] = sorted(list(set(vocab))) if vocab else []
        self._build_mappings()

    def _build_mappings(self) -> None:
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def fit(self, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("Cannot fit tokenizer on empty text")
        self.chars = sorted(list(set(text)))
        self._build_mappings()
        return self

    def encode(self, text: str) -> List[int]:
        if not self.chars:
            raise RuntimeError("Tokenizer has not been fitted with a vocabulary yet")
        # Unseen characters fall back to space if available or index 0
        fallback = self.stoi.get(" ", 0)
        return [self.stoi.get(c, fallback) for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.itos.get(i, "") for i in tokens])

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {"chars": self.chars}
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer metadata not found at {filepath}")
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict) or "chars" not in data:
                raise ValueError("Corrupted tokenizer metadata file")
            return cls(vocab=data["chars"])
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {filepath}: {e}") from e


def load_tokenizer(filepath: str = "data/meta.pkl") -> BaseTokenizer:
    """Unified loader that auto-detects BPETokenizer vs CharTokenizer from metadata."""
    if not os.path.exists(filepath):
        from synor.bpe_tokenizer import BPETokenizer
        return BPETokenizer()

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and data.get("type") == "bpe":
        from synor.bpe_tokenizer import BPETokenizer
        return BPETokenizer.load(filepath)
    else:
        return CharTokenizer.load(filepath)
