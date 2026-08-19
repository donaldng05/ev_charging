"""Calibrate the synthetic simulation world from normalized ACN sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chargeopt.simulation.schemas import CalibrationStats


def calibrate_from_sessions(
    sessions: pd.DataFrame,
    *,
    timestep_minutes: int,
) -> CalibrationStats:
    """Summarize real arrivals, delivered energy, and connected duration."""
    if sessions.empty:
        msg = "cannot calibrate simulation from empty sessions"
        raise ValueError(msg)
    required = {"start_time", "end_time", "duration_min", "energy_kwh"}
    missing = sorted(required - set(sessions.columns))
    if missing:
        msg = f"session table missing calibration columns: {missing}"
        raise ValueError(msg)
    if timestep_minutes <= 0:
        msg = "timestep_minutes must be positive"
        raise ValueError(msg)

    if "hour" in sessions.columns:
        hours = sessions["hour"].astype(int)
    else:
        hours = pd.to_datetime(sessions["start_time"], utc=True).dt.hour
    counts = np.bincount(hours.to_numpy(), minlength=24).astype(float)
    hour_pmf = tuple((counts / counts.sum()).tolist())

    events: list[tuple[pd.Timestamp, int]] = []
    starts = pd.to_datetime(sessions["start_time"], utc=True)
    ends = pd.to_datetime(sessions["end_time"], utc=True)
    for start, finish in zip(starts, ends, strict=True):
        events.append((start, 1))
        events.append((finish, -1))
    concurrent = 0
    peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        concurrent += delta
        peak = max(peak, concurrent)

    return CalibrationStats(
        hour_pmf=hour_pmf,
        mean_duration_min=float(sessions["duration_min"].astype(float).mean()),
        mean_energy_kwh=float(sessions["energy_kwh"].astype(float).mean()),
        peak_concurrent=peak,
        n_sessions=len(sessions),
    )
