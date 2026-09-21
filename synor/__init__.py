"""
Synor AI Foundation Model Framework.
"""

from synor.config import SynorConfig, get_preset, PRESETS
from synor.model import SynorLM
from synor.tokenizer import BaseTokenizer, CharTokenizer
from synor.dataset import TextDataset
from synor.trainer import Trainer
from synor.sampler import TextSampler
from synor.utils import get_device, save_checkpoint_atomic, setup_logger
from synor.logger import (
    chalk,
    print_banner,
    log_info,
    log_success,
    log_warn,
    log_error,
    log_step,
)

from synor.emotion import EmotionalState
from synor.search import search_engine, should_search_web

# Aliases for backward compatibility
MiniGPT = SynorLM
GPTConfig = SynorConfig

__version__ = "0.1.0"

__all__ = [
    "SynorLM",
    "SynorConfig",
    "MiniGPT",
    "GPTConfig",
    "get_preset",
    "PRESETS",
    "BaseTokenizer",
    "CharTokenizer",
    "TextDataset",
    "Trainer",
    "TextSampler",
    "get_device",
    "save_checkpoint_atomic",
    "setup_logger",
    "chalk",
    "print_banner",
    "log_info",
    "log_success",
    "log_warn",
    "log_error",
    "log_step",
    "EmotionalState",
]
