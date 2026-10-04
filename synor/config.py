from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SynorConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_embd: int = 768
    n_head: int = 12
    n_kv_head: Optional[int] = 4
    n_layer: int = 10
    hidden_dim: Optional[int] = 2048
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_kv_head is not None and self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")


PRESETS: Dict[str, SynorConfig] = {
    "tiny": SynorConfig(
        block_size=128,
        vocab_size=65,
        n_embd=192,
        n_head=6,
        n_kv_head=2,
        n_layer=4,
        hidden_dim=512,
        dropout=0.0,
    ),
    "small": SynorConfig(
        block_size=256,
        vocab_size=65,
        n_embd=384,
        n_head=6,
        n_kv_head=2,
        n_layer=6,
        hidden_dim=1024,
        dropout=0.0,
    ),
    "medium": SynorConfig(
        block_size=512,
        vocab_size=65,
        n_embd=512,
        n_head=8,
        n_kv_head=4,
        n_layer=8,
        hidden_dim=1376,
        dropout=0.0,
    ),
    "large": SynorConfig(
        block_size=1024,
        vocab_size=65,
        n_embd=768,
        n_head=12,
        n_kv_head=4,
        n_layer=12,
        hidden_dim=2048,
        dropout=0.0,
    ),
    "100m": SynorConfig(
        block_size=512,
        vocab_size=50257,
        n_embd=768,
        n_head=12,
        n_kv_head=4,
        n_layer=10,
        hidden_dim=2048,
        dropout=0.0,
    ),
}


def get_preset(name: str) -> SynorConfig:
    name = name.lower()
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    config = PRESETS[name]
    config.validate()
    return config
