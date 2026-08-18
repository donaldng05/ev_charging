"""Deterministic RNG seeding."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed the standard library and NumPy RNGs (and PYTHONHASHSEED)."""
    if seed < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
