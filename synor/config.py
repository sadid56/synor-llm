from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SynorConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_embd: int = 192
    n_head: int = 6
    n_layer: int = 4
    dropout: float = 0.1
    bias: bool = True

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
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
        n_layer=4,
        dropout=0.1,
    ),
    "small": SynorConfig(
        block_size=256,
        vocab_size=65,
        n_embd=384,
        n_head=6,
        n_layer=6,
        dropout=0.1,
    ),
    "medium": SynorConfig(
        block_size=512,
        vocab_size=65,
        n_embd=512,
        n_head=8,
        n_layer=8,
        dropout=0.1,
    ),
    "large": SynorConfig(
        block_size=1024,
        vocab_size=65,
        n_embd=768,
        n_head=12,
        n_layer=12,
        dropout=0.1,
    ),
}


def get_preset(name: str) -> SynorConfig:
    """Retrieve model preset configuration by name."""
    name = name.lower()
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    config = PRESETS[name]
    config.validate()
    return config
