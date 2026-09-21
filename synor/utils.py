import os
import logging
import tempfile
import torch


def get_device() -> torch.device:
    """Detect and return the highest-performance compute device available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint_atomic(state: dict, target_path: str) -> None:
    """
    Save model checkpoint atomically to avoid corruption during interruption.
    Writes to a temporary file in the same directory first, then replaces atomically.
    """
    target_dir = os.path.dirname(target_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(delete=False, dir=target_dir, suffix=".tmp") as tmp_file:
        tmp_path = tmp_file.name
        
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, target_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise IOError(f"Failed to atomically save checkpoint to {target_path}: {e}") from e


def setup_logger(name: str = "synor") -> logging.Logger:
    """Configure a clean, standard console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
