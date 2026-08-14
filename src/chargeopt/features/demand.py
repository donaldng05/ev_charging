"""Site-level 15-minute demand features and temporal split."""

from __future__ import annotations

from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from chargeopt.data.schemas import DemandInterval
from chargeopt.data.validation import validate_sessions

LAG_15M = 1
LAG_1H = 4
LAG_24H = 96
ROLL_1H = 4
ROLL_24H = 96


def _charging_end(row: pd.Series) -> pd.Timestamp:
    start = pd.Timestamp(row["start_time"])
    end = pd.Timestamp(row["end_time"])
    done = row["done_charging_time"]
    charge_end = pd.Timestamp(done) if pd.notna(done) else end
    if charge_end < start:
        return end
    return charge_end


def build_demand_table(
    sessions: pd.DataFrame,
    *,
    timestep_minutes: int,
    timezone_name: str,
    train_fraction: float,
    val_fraction: float,
) -> pd.DataFrame:
    if timestep_minutes != 15:
        msg = "MVP demand features require timestep_minutes=15"
        raise ValueError(msg)
    sessions = validate_sessions(sessions)
    tz = ZoneInfo(timezone_name)
    for column in ("start_time", "end_time", "done_charging_time"):
        sessions[column] = pd.to_datetime(sessions[column], utc=True)

    site_id = str(sessions["site_id"].iloc[0])
    freq = pd.Timedelta(minutes=timestep_minutes)
    range_start = sessions["start_time"].min().floor(f"{timestep_minutes}min")
    range_end = sessions["end_time"].max().ceil(f"{timestep_minutes}min")
    bins = pd.date_range(range_start, range_end, freq=freq, inclusive="left")
    energy_map: dict[pd.Timestamp, float] = dict.fromkeys(list(bins), 0.0)
    arrival_map: dict[pd.Timestamp, int] = dict.fromkeys(list(bins), 0)

    for _, row in sessions.iterrows():
        start = pd.Timestamp(row["start_time"])
        charge_end = _charging_end(row)
        arrival_bin = start.floor(f"{timestep_minutes}min")
        if arrival_bin in arrival_map:
            arrival_map[arrival_bin] += 1
        total_seconds = (charge_end - start).total_seconds()
        if total_seconds <= 0:
            continue
        cursor = arrival_bin
        while cursor < charge_end:
            bin_end = cursor + freq
            overlap_start = max(start, cursor)
            overlap_end = min(charge_end, bin_end)
            overlap = (overlap_end - overlap_start).total_seconds()
            if overlap > 0 and cursor in energy_map:
                energy_map[cursor] += float(row["energy_kwh"]) * (overlap / total_seconds)
            cursor = bin_end

    frame = pd.DataFrame(
        {
            "timestamp": bins,
            "site_id": site_id,
            "n_arrivals": [arrival_map[ts] for ts in bins],
            "energy_kwh": [energy_map[ts] for ts in bins],
        }
    )
    local = frame["timestamp"].dt.tz_convert(tz)
    frame["hour"] = local.dt.hour
    frame["day_of_week"] = local.dt.weekday
    frame["is_weekend"] = frame["day_of_week"] >= 5
    frame["month"] = local.dt.month
    shifted = frame["energy_kwh"].shift(1)
    frame["lag_15m"] = frame["energy_kwh"].shift(LAG_15M)
    frame["lag_1h"] = frame["energy_kwh"].shift(LAG_1H)
    frame["lag_24h"] = frame["energy_kwh"].shift(LAG_24H)
    frame["rolling_mean_1h"] = shifted.rolling(ROLL_1H, min_periods=1).mean()
    frame["rolling_mean_24h"] = shifted.rolling(ROLL_24H, min_periods=1).mean()
    frame["split"] = _temporal_split(len(frame), train_fraction, val_fraction)

    records: list[dict[str, Any]] = []
    raw_records = cast(list[dict[str, Any]], frame.to_dict(orient="records"))
    for record in raw_records:
        for key in ("lag_15m", "lag_1h", "lag_24h", "rolling_mean_1h", "rolling_mean_24h"):
            value = record[key]
            if value is None or pd.isna(value):
                record[key] = None
        records.append(record)
    validated = [DemandInterval.model_validate(record).model_dump() for record in records]
    return pd.DataFrame(validated)


def _temporal_split(n: int, train_fraction: float, val_fraction: float) -> list[str]:
    if n <= 0:
        return []
    if n == 1:
        return ["test"]
    if n == 2:
        return ["train", "test"]

    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    return labels
