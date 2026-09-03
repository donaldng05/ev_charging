"""CSV read/write for derived ACN-Data tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chargeopt.data.validation import SESSION_COLUMNS
from chargeopt.utils.io import read_csv, write_csv

DATE_COLS = ("start_time", "end_time", "done_charging_time")


def write_sessions_csv(frame: pd.DataFrame, path: Path) -> None:
    write_csv(frame, path, columns=SESSION_COLUMNS, label="session table")


def read_sessions_csv(path: Path) -> pd.DataFrame:
    return read_csv(path, date_columns=DATE_COLS, label="session CSV")


def write_demand_csv(frame: pd.DataFrame, path: Path) -> None:
    write_csv(frame, path, label="demand table")


def read_demand_csv(path: Path) -> pd.DataFrame:
    return read_csv(path, date_columns=("timestamp",), label="demand CSV")


def write_trips_csv(frame: pd.DataFrame, path: Path) -> None:
    write_csv(frame, path, label="trip table")


def read_trips_csv(path: Path) -> pd.DataFrame:
    return read_csv(path, label="trip CSV")
