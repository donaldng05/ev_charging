"""CSV read/write for derived ACN-Data tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chargeopt.data.validation import SESSION_COLUMNS

DATE_COLUMNS: tuple[str, ...] = ("start_time", "end_time", "done_charging_time")


def write_sessions_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.loc[:, list(SESSION_COLUMNS)]
    ordered.to_csv(path, index=False)


def read_sessions_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"session CSV not found: {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    for column in DATE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def write_demand_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_demand_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"demand CSV not found: {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame
