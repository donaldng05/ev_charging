"""Small shared helpers for validated CSV artifact writes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def select_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    """Return columns in contract order, raising a consistent validation error."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"{label} missing columns: {missing}"
        raise ValueError(msg)
    return frame.loc[:, list(columns)]


def write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    label: str = "CSV table",
) -> None:
    """Create parent directories and write a complete or contract-ordered CSV."""
    selected = select_columns(frame, columns, label=label) if columns is not None else frame
    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(path, index=False)
