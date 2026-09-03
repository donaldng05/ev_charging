"""Site-level 15-minute demand features and temporal split."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from chargeopt.data.schemas import DemandInterval
from chargeopt.data.validation import validate_sessions

LAG_15M, LAG_1H, LAG_24H, LAG_1W = 1, 4, 96, 96 * 7
ROLL_1H, ROLL_24H, ROLL_7D = 4, 96, 96 * 7
ERA_PRE_COVID, ERA_COVID, ERA_POST_COVID = "pre_covid", "covid", "post_covid"


def _as_utc(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def era_label(
    timestamp: datetime | pd.Timestamp, covid_start: datetime, covid_end: datetime
) -> str:
    ts, start, end = _as_utc(timestamp), _as_utc(covid_start), _as_utc(covid_end)
    return ERA_PRE_COVID if ts < start else ERA_COVID if ts < end else ERA_POST_COVID


def _charging_end(row: pd.Series) -> pd.Timestamp:
    start, end, done = (
        pd.Timestamp(row["start_time"]),
        pd.Timestamp(row["end_time"]),
        row["done_charging_time"],
    )
    charge_end = pd.Timestamp(done) if pd.notna(done) else end
    return end if charge_end < start else charge_end


def build_demand_table(
    sessions: pd.DataFrame,
    *,
    timestep_minutes: int,
    timezone_name: str,
    train_fraction: float,
    val_fraction: float,
    covid_start: datetime,
    covid_end: datetime,
) -> pd.DataFrame:
    if timestep_minutes != 15:
        raise ValueError("MVP demand features require timestep_minutes=15")
    sessions = validate_sessions(sessions)
    tz = ZoneInfo(timezone_name)
    for col in ("start_time", "end_time", "done_charging_time"):
        sessions[col] = pd.to_datetime(sessions[col], utc=True)

    site_id = str(sessions["site_id"].iloc[0])
    freq = pd.Timedelta(minutes=timestep_minutes)
    bins = pd.date_range(
        sessions["start_time"].min().floor(f"{timestep_minutes}min"),
        sessions["end_time"].max().ceil(f"{timestep_minutes}min"),
        freq=freq,
        inclusive="left",
    )
    energy_map: dict[pd.Timestamp, float] = dict.fromkeys(list(bins), 0.0)
    arrival_map: dict[pd.Timestamp, int] = dict.fromkeys(list(bins), 0)

    for _, row in sessions.iterrows():
        start = pd.Timestamp(row["start_time"])
        charge_end = _charging_end(row)
        arr_bin = start.floor(f"{timestep_minutes}min")
        if arr_bin in arrival_map:
            arrival_map[arr_bin] += 1
        total_sec = (charge_end - start).total_seconds()
        if total_sec <= 0:
            continue
        cur = arr_bin
        while cur < charge_end:
            overlap = (min(charge_end, cur + freq) - max(start, cur)).total_seconds()
            if overlap > 0 and cur in energy_map:
                energy_map[cur] += float(row["energy_kwh"]) * (overlap / total_sec)
            cur = cur + freq

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

    sh = frame["energy_kwh"].shift(1)
    frame["lag_15m"] = frame["energy_kwh"].shift(LAG_15M)
    frame["lag_1h"] = frame["energy_kwh"].shift(LAG_1H)
    frame["lag_24h"] = frame["energy_kwh"].shift(LAG_24H)
    frame["lag_1w"] = frame["energy_kwh"].shift(LAG_1W)
    frame["rolling_mean_1h"] = sh.rolling(ROLL_1H, min_periods=1).mean()
    frame["rolling_mean_24h"] = sh.rolling(ROLL_24H, min_periods=1).mean()
    frame["rolling_mean_7d"] = sh.rolling(ROLL_7D, min_periods=1).mean()
    frame["era"] = [era_label(ts, covid_start, covid_end) for ts in frame["timestamp"]]
    frame["split"] = temporal_split_labels(len(frame), train_fraction, val_fraction)

    null_keys = (
        "lag_15m",
        "lag_1h",
        "lag_24h",
        "lag_1w",
        "rolling_mean_1h",
        "rolling_mean_24h",
        "rolling_mean_7d",
    )
    records: list[dict[str, Any]] = []
    for r in cast(list[dict[str, Any]], frame.to_dict(orient="records")):
        for k in null_keys:
            if r[k] is None or pd.isna(r[k]):
                r[k] = None
        records.append(r)
    return pd.DataFrame([DemandInterval.model_validate(rec).model_dump() for rec in records])


def temporal_split_labels(n: int, train_fraction: float, val_fraction: float) -> list[str]:
    if n <= 0:
        return []
    if n == 1:
        return ["test"]
    if n == 2:
        return ["train", "test"]
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction))
    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1
    return ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
