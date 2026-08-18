"""Prediction artifacts and the M4 forecast lookup contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from chargeopt.data.schemas import DemandPrediction, EnergyPrediction

DEMAND_PREDICTION_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "split",
    "target",
    "prediction",
    "model",
    "seed",
)
ENERGY_PREDICTION_COLUMNS: tuple[str, ...] = (
    "trip_id",
    "split",
    "target",
    "prediction",
    "model",
    "seed",
)
METRICS_COLUMNS: tuple[str, ...] = ("model", "split", "mae", "rmse", "n")
ERROR_SLICE_COLUMNS: tuple[str, ...] = (
    "model",
    "split",
    "hour",
    "is_weekend",
    "era",
    "mae",
    "rmse",
    "n",
)


def write_demand_predictions(frame: pd.DataFrame, path: Path) -> None:
    _write_validated(frame, path, DEMAND_PREDICTION_COLUMNS, DemandPrediction)


def write_energy_predictions(frame: pd.DataFrame, path: Path) -> None:
    _write_validated(frame, path, ENERGY_PREDICTION_COLUMNS, EnergyPrediction)


def write_metrics(frame: pd.DataFrame, path: Path) -> None:
    missing = [column for column in METRICS_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"metrics table missing columns: {missing}"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(METRICS_COLUMNS)].to_csv(path, index=False)


def write_error_slices(frame: pd.DataFrame, path: Path) -> None:
    missing = [column for column in ERROR_SLICE_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"error-slice table missing columns: {missing}"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(ERROR_SLICE_COLUMNS)].to_csv(path, index=False)


def load_demand_forecast(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"demand forecast CSV not found: {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    missing = [column for column in DEMAND_PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"demand forecast missing columns: {missing}"
        raise ValueError(msg)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.loc[:, list(DEMAND_PREDICTION_COLUMNS)]


def lookup_predicted_congestion(
    forecast: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    model: str,
) -> float:
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    matched = forecast.loc[(forecast["timestamp"] == ts) & (forecast["model"] == model)]
    if matched.empty:
        msg = f"no {model!r} forecast at {ts.isoformat()}"
        raise KeyError(msg)
    return float(matched["prediction"].iloc[0])


def _write_validated(
    frame: pd.DataFrame,
    path: Path,
    columns: tuple[str, ...],
    schema: type[DemandPrediction] | type[EnergyPrediction],
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"prediction table missing columns: {missing}"
        raise ValueError(msg)
    records = cast(list[dict[str, Any]], frame.loc[:, list(columns)].to_dict(orient="records"))
    validated = [schema.model_validate(record).model_dump() for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validated).to_csv(path, index=False)
