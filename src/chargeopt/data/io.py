"""CSV read/write for derived ACN-Data tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chargeopt.data.validation import SESSION_COLUMNS
from chargeopt.utils.io import write_csv

DATE_COLUMNS: tuple[str, ...] = ("start_time", "end_time", "done_charging_time")


def write_sessions_csv(frame: pd.DataFrame, path: Path) -> None:
    write_csv(frame, path, columns=SESSION_COLUMNS, label="session table")


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
    write_csv(frame, path, label="demand table")


def read_demand_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"demand CSV not found: {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def write_trips_csv(frame: pd.DataFrame, path: Path) -> None:
    write_csv(frame, path, label="trip table")


def read_trips_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"trip CSV not found: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)
