"""Shared helpers for validated CSV artifact reading and writing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def select_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str = "table",
) -> pd.DataFrame:
    """Return columns in contract order, raising a consistent validation error."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        msg = f"{label} missing columns: {missing}"
        raise ValueError(msg)
    return frame.loc[:, list(columns)]


def write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    label: str = "table",
) -> None:
    """Create parent directories and write a contract-ordered CSV."""
    selected = select_columns(frame, columns, label=label) if columns is not None else frame
    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(path, index=False)


def read_csv(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    date_columns: Sequence[str] | None = None,
    label: str = "table",
) -> pd.DataFrame:
    """Read a CSV with optional column validation and UTC datetime parsing."""
    if not path.is_file():
        msg = f"{label} not found: {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    if columns is not None:
        frame = select_columns(frame, columns, label=label)
    if date_columns is not None:
        for c in date_columns:
            if c in frame.columns:
                frame[c] = pd.to_datetime(frame[c], utc=True, errors="coerce")
    return frame
