"""Utility helpers for the EAN project."""

import os
import random
import logging
import yaml
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a nested dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure and return the root logger."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("EAN")


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: str = None,
) -> None:
    """Save a training checkpoint; optionally also copy as the best model."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        import shutil
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath: str, device: torch.device) -> Dict[str, Any]:
    """Load a checkpoint from disk onto the specified device."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    return torch.load(filepath, map_location=device)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str) -> None:
    """Create a directory (and any parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
