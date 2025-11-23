# config.py
"""
Config loader for digital-twin project.

Provides a single entry `load_config(path=None)` that reads YAML config from
`configs/default.yaml` by default and returns a nested dict. Also exposes
`get_default_config()` for quick access.

This file also sets global random seeds for reproducibility when `seed` is present
in the config (it seeds Python, NumPy and torch if available).
"""

import os
import yaml
import random
import numpy as np

try:
    import torch
except Exception:
    torch = None

DEFAULT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config', 'defaults.yaml'))


def load_config(path=None):
    """Load YAML config and return a dict. Also sets global seeds if `seed` key present."""
    p = path or DEFAULT_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, 'r') as f:
        cfg = yaml.safe_load(f)

    # set reproducible seeds if provided
    seed = cfg.get('seed', None)
    if seed is not None:
        _set_seeds(seed)
    return cfg


def get_default_config():
    return load_config(DEFAULT_PATH)


def _set_seeds(seed):
    print(f"[config] Setting global random seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    try:
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


if __name__ == '__main__':
    print(load_config())
