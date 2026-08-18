"""Forecast artifact contract consumed by a stub M4 policy lookup."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chargeopt.models.demand import RANDOM_FOREST
from chargeopt.models.io import (
    load_demand_forecast,
    lookup_predicted_congestion,
    write_demand_predictions,
)


def _forecast_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2018-09-11T20:00:00", periods=3, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "split": ["test"] * 6,
            "target": [10.0, 12.0, 8.0, 10.0, 12.0, 8.0],
            "prediction": [9.5, 11.0, 7.5, 4.0, 4.0, 4.0],
            "model": [RANDOM_FOREST] * 3 + ["last_observation"] * 3,
            "seed": [42] * 6,
        }
    )


def test_load_demand_forecast_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "demand_predictions.csv"
    write_demand_predictions(_forecast_frame(), path)
    loaded = load_demand_forecast(path)
    assert set(loaded.columns) == {
        "timestamp",
        "split",
        "target",
        "prediction",
        "model",
        "seed",
    }
    assert loaded["timestamp"].dt.tz is not None


def test_stub_policy_looks_up_predicted_congestion(tmp_path: Path) -> None:
    path = tmp_path / "demand_predictions.csv"
    write_demand_predictions(_forecast_frame(), path)
    forecast = load_demand_forecast(path)
    timestamp = pd.Timestamp("2018-09-11T20:15:00", tz="UTC")

    def stub_policy_congestion_score(when: pd.Timestamp) -> float:
        return lookup_predicted_congestion(forecast, when, model=RANDOM_FOREST)

    assert stub_policy_congestion_score(timestamp) == 11.0


def test_lookup_missing_timestamp_raises(tmp_path: Path) -> None:
    path = tmp_path / "demand_predictions.csv"
    write_demand_predictions(_forecast_frame(), path)
    forecast = load_demand_forecast(path)
    with pytest.raises(KeyError, match="forecast"):
        lookup_predicted_congestion(
            forecast,
            pd.Timestamp("2019-01-01T00:00:00", tz="UTC"),
            model=RANDOM_FOREST,
        )
