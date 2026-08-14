"""Validate derived session tables."""

from __future__ import annotations

import pandas as pd

from chargeopt.data.schemas import ChargingSession

SESSION_COLUMNS: tuple[str, ...] = (
    "session_id",
    "site_id",
    "station_id",
    "space_id",
    "start_time",
    "end_time",
    "done_charging_time",
    "duration_min",
    "energy_kwh",
    "day_of_week",
    "hour",
)


def validate_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in SESSION_COLUMNS if col not in frame.columns]
    if missing:
        msg = f"session table missing columns: {missing}"
        raise ValueError(msg)
    if frame.empty:
        msg = "session table is empty"
        raise ValueError(msg)
    if frame["session_id"].duplicated().any():
        msg = "session_id values must be unique"
        raise ValueError(msg)
    subset = frame.loc[:, list(SESSION_COLUMNS)].copy()
    for column in ("start_time", "end_time", "done_charging_time"):
        subset[column] = subset[column].astype(object).where(subset[column].notna(), None)
    records = subset.to_dict(orient="records")
    validated = [ChargingSession.model_validate(row).model_dump() for row in records]
    return pd.DataFrame(validated)
