"""Deterministic RNG seeding."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed the standard library RNG (and PYTHONHASHSEED).

    NumPy / XGBoost seeding is added when those dependencies land in M2-M3.
    """
    if seed < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
