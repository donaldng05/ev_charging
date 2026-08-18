"""Demand error slices by hour, weekend, and era."""

from __future__ import annotations

import pandas as pd

from chargeopt.models.metrics import error_slices_from_predictions


def test_error_slices_group_by_hour_weekend_and_era() -> None:
    timestamps = pd.date_range("2018-09-07T08:00:00", periods=4, freq="15min", tz="UTC")
    demand = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hour": timestamps.hour,
            "is_weekend": True,
            "era": ["pre_covid"] * 4,
        }
    )
    predictions = pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "split": ["test"] * 8,
            "target": [10.0] * 8,
            "prediction": [9.0] * 4 + [8.0] * 4,
            "model": ["random_forest"] * 4 + ["last_observation"] * 4,
            "seed": [42] * 8,
        }
    )
    slices = error_slices_from_predictions(predictions, demand)
    assert {"model", "split", "hour", "is_weekend", "era", "mae", "rmse", "n"} <= set(
        slices.columns
    )
    rf = slices.loc[slices["model"] == "random_forest"].iloc[0]
    assert rf["mae"] == 1.0
    assert rf["era"] == "pre_covid"
    assert bool(rf["is_weekend"]) is True
    assert rf["n"] == 4
