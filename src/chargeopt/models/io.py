"""Prediction artifacts and the M4 forecast lookup contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel

from chargeopt.data.schemas import DemandPrediction, EnergyPrediction
from chargeopt.models.learners import RANDOM_FOREST
from chargeopt.utils.io import read_csv, select_columns, write_csv

DEMAND_PREDICTION_COLUMNS = ("timestamp", "split", "target", "prediction", "model", "seed")
ENERGY_PREDICTION_COLUMNS = ("trip_id", "split", "target", "prediction", "model", "seed")
METRICS_COLUMNS = ("model", "split", "mae", "rmse", "n")
ERROR_SLICE_COLUMNS = ("model", "split", "hour", "is_weekend", "era", "mae", "rmse", "n")


def _write_validated(
    frame: pd.DataFrame,
    path: Path,
    columns: tuple[str, ...],
    schema: type[BaseModel],
) -> None:
    selected = select_columns(frame, columns, label="prediction table")
    records = cast(list[dict[str, Any]], selected.to_dict(orient="records"))
    validated = [schema.model_validate(r).model_dump() for r in records]
    write_csv(pd.DataFrame(validated), path, label="prediction table")


def write_demand_predictions(frame: pd.DataFrame, path: Path) -> None:
    _write_validated(frame, path, DEMAND_PREDICTION_COLUMNS, DemandPrediction)


def write_energy_predictions(frame: pd.DataFrame, path: Path) -> None:
    _write_validated(frame, path, ENERGY_PREDICTION_COLUMNS, EnergyPrediction)


def write_metrics(frame: pd.DataFrame, path: Path) -> None:
    write_csv(frame, path, columns=METRICS_COLUMNS, label="metrics table")


def write_error_slices(frame: pd.DataFrame, path: Path) -> None:
    write_csv(frame, path, columns=ERROR_SLICE_COLUMNS, label="error-slice table")


def load_demand_forecast(path: Path) -> pd.DataFrame:
    return read_csv(
        path,
        columns=DEMAND_PREDICTION_COLUMNS,
        date_columns=("timestamp",),
        label="demand forecast CSV",
    )


def lookup_predicted_congestion(
    forecast: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    model: str = RANDOM_FOREST,
) -> float:
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    matched = forecast.loc[(forecast["timestamp"] == ts) & (forecast["model"] == model)]
    if matched.empty:
        msg = f"no {model!r} forecast at {ts.isoformat()}"
        raise KeyError(msg)
    return float(matched["prediction"].iloc[0])
