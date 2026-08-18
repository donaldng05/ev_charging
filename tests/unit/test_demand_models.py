"""Next-hour demand target, chronological baselines, and Random Forest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chargeopt.data.io import read_sessions_csv
from chargeopt.features.demand import build_demand_table
from chargeopt.models.baselines import (
    fit_historical_average,
    last_observation_forecast,
    predict_historical_average,
)
from chargeopt.models.demand import (
    DEMAND_FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_next_hour_target,
    horizon_bins,
    train_and_predict_demand,
)

FIXTURE = Path("tests/fixtures/acn_sessions.csv")


def _demand_frame(n: int = 20) -> pd.DataFrame:
    timestamps = pd.date_range("2018-09-05", periods=n, freq="15min", tz="UTC")
    energy = np.arange(n, dtype=float)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    hour = timestamps.hour
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_id": "caltech",
            "n_arrivals": np.zeros(n, dtype=int),
            "energy_kwh": energy,
            "hour": hour,
            "day_of_week": timestamps.dayofweek,
            "is_weekend": timestamps.dayofweek >= 5,
            "month": timestamps.month,
            "lag_15m": pd.Series(energy).shift(1),
            "lag_1h": pd.Series(energy).shift(4),
            "lag_24h": pd.Series(energy).shift(8),
            "rolling_mean_1h": pd.Series(energy).shift(1).rolling(4, min_periods=1).mean(),
            "rolling_mean_24h": pd.Series(energy).shift(1).rolling(8, min_periods=1).mean(),
            "split": splits,
        }
    )


def test_horizon_bins_from_config_minutes() -> None:
    assert horizon_bins(60, 15) == 4


def test_horizon_bins_rejects_non_multiple() -> None:
    with pytest.raises(ValueError, match="multiple"):
        horizon_bins(50, 15)


def test_next_hour_target_sums_future_bins() -> None:
    frame = _demand_frame(20)
    labeled = add_next_hour_target(frame, horizon_minutes=60, timestep_minutes=15)
    first = labeled.iloc[0]
    expected = float(frame["energy_kwh"].iloc[1:5].sum())
    assert first[TARGET_COLUMN] == expected


def test_next_hour_target_drops_cross_split_rows() -> None:
    frame = _demand_frame(20)
    labeled = add_next_hour_target(frame, horizon_minutes=60, timestep_minutes=15)
    train_end = frame.loc[frame["split"] == "train"].iloc[-4:]
    kept = set(labeled.loc[labeled["split"] == "train", "timestamp"])
    for timestamp in train_end["timestamp"]:
        assert timestamp not in kept


def test_demand_features_exclude_current_and_future_energy() -> None:
    overlap = set(DEMAND_FEATURE_COLUMNS) & set(FORBIDDEN_FEATURE_COLUMNS)
    assert overlap == set()


def test_last_observation_is_prior_hour_total() -> None:
    frame = _demand_frame(12)
    frame["rolling_mean_1h"] = 2.5
    predicted = last_observation_forecast(frame, n_bins=4)
    assert (predicted == 10.0).all()


def test_historical_average_fits_train_only() -> None:
    frame = _demand_frame(48)
    labeled = add_next_hour_target(frame, horizon_minutes=60, timestep_minutes=15)
    assert {"val", "test"} <= set(labeled["split"])
    train = labeled.loc[labeled["split"] == "train"]
    fitted = fit_historical_average(train, target_column=TARGET_COLUMN)
    pred_one = predict_historical_average(labeled, fitted)

    leaked = labeled.copy()
    leaked.loc[leaked["split"] != "train", TARGET_COLUMN] = 1_000.0
    leaked_train = leaked.loc[leaked["split"] == "train"]
    fitted_leaked = fit_historical_average(leaked_train, target_column=TARGET_COLUMN)
    pred_two = predict_historical_average(leaked, fitted_leaked)
    pd.testing.assert_series_equal(pred_one, pred_two, check_names=False)


def test_train_and_predict_demand_is_reproducible() -> None:
    sessions = read_sessions_csv(FIXTURE)
    demand = build_demand_table(
        sessions,
        timestep_minutes=15,
        timezone_name="America/Los_Angeles",
        train_fraction=0.7,
        val_fraction=0.15,
    )
    kwargs = {
        "timestep_minutes": 15,
        "horizon_minutes": 60,
        "n_estimators": 20,
        "max_depth": 4,
        "min_samples_leaf": 1,
        "seed": 42,
    }
    first, _ = train_and_predict_demand(demand, **kwargs)
    second, _ = train_and_predict_demand(demand, **kwargs)
    forest = first.loc[first["model"] == "random_forest", "prediction"].to_numpy()
    forest_again = second.loc[second["model"] == "random_forest", "prediction"].to_numpy()
    np.testing.assert_allclose(forest, forest_again)


def test_demand_predictions_include_required_columns() -> None:
    sessions = read_sessions_csv(FIXTURE)
    demand = build_demand_table(
        sessions,
        timestep_minutes=15,
        timezone_name="America/Los_Angeles",
        train_fraction=0.7,
        val_fraction=0.15,
    )
    predictions, metrics = train_and_predict_demand(
        demand,
        timestep_minutes=15,
        horizon_minutes=60,
        n_estimators=20,
        max_depth=4,
        min_samples_leaf=1,
        seed=42,
    )
    required = {"timestamp", "split", "target", "prediction", "model", "seed"}
    assert required <= set(predictions.columns)
    assert set(predictions["model"]) >= {"last_observation", "historical_average", "random_forest"}
    assert set(predictions["split"]) <= {"train", "val", "test"}
    assert "test" in set(predictions["split"])
    assert {"model", "split", "mae", "rmse", "n"} <= set(metrics.columns)
    assert (metrics["split"] == "test").any()
