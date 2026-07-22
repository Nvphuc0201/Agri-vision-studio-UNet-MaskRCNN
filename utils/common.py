from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "cuda") -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    ensure_dir(Path(path).parent)
    torch.save(state, path)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_class_names_from_checkpoint(checkpoint: Dict[str, Any]) -> List[str]:
    config = checkpoint.get("config", {})
    if isinstance(config, dict):
        names = config.get("data", {}).get("class_names")
        if names:
            return list(names)
    return []
