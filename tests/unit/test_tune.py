"""Walk-forward demand tuning never sees the chronological test split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chargeopt.config import HyperparameterGrid, load_config
from chargeopt.features.energy import generate_synthetic_trips
from chargeopt.models.demand import horizon_bins
from chargeopt.models.tune import (
    expanding_window_splits,
    param_grid,
    select_best_params,
    tune_demand_random_forest,
    tune_energy_random_forest,
)


def _demand_frame(n: int = 20) -> pd.DataFrame:
    timestamps = pd.date_range("2018-09-05", periods=n, freq="15min", tz="UTC")
    energy = np.arange(n, dtype=float)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_id": "caltech",
            "n_arrivals": np.zeros(n, dtype=int),
            "energy_kwh": energy,
            "hour": timestamps.hour,
            "day_of_week": timestamps.dayofweek,
            "is_weekend": timestamps.dayofweek >= 5,
            "month": timestamps.month,
            "lag_15m": pd.Series(energy).shift(1),
            "lag_1h": pd.Series(energy).shift(4),
            "lag_24h": pd.Series(energy).shift(8),
            "rolling_mean_1h": pd.Series(energy).shift(1).rolling(4, min_periods=1).mean(),
            "rolling_mean_24h": pd.Series(energy).shift(1).rolling(8, min_periods=1).mean(),
            "lag_1w": pd.Series(energy).shift(8),
            "rolling_mean_7d": pd.Series(energy).shift(1).rolling(8, min_periods=1).mean(),
            "era": ["pre_covid"] * n,
            "split": splits,
        }
    )


def _tiny_search() -> HyperparameterGrid:
    return HyperparameterGrid(n_estimators=[8, 12], max_depth=[2], min_samples_leaf=[1])


def test_param_grid_is_cartesian_product() -> None:
    grid = HyperparameterGrid(n_estimators=[10, 20], max_depth=[2, 4], min_samples_leaf=[1])
    combos = param_grid(grid)
    assert len(combos) == 4
    assert {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1} in combos


def test_expanding_window_splits_increase_and_respect_gap() -> None:
    gap = 4
    folds = expanding_window_splits(40, n_splits=3, gap=gap)
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        assert train_idx.tolist() == sorted(train_idx.tolist())
        assert val_idx.tolist() == sorted(val_idx.tolist())
        assert int(train_idx.max()) < int(val_idx.min())
        assert int(val_idx.min()) - int(train_idx.max()) >= gap + 1


def test_select_best_params_uses_mean_mae_then_rmse() -> None:

    metrics = pd.DataFrame(
        [
            {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 1, "mae": 2.0, "rmse": 6.0},
            {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 1, "mae": 4.0, "rmse": 4.0},
            {"n_estimators": 20, "max_depth": 2, "min_samples_leaf": 1, "mae": 4.0, "rmse": 4.0},
            {"n_estimators": 20, "max_depth": 2, "min_samples_leaf": 1, "mae": 4.0, "rmse": 5.0},
        ]
    )
    best = select_best_params(metrics)
    assert best == {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 1}


def test_tune_demand_ignores_test_energy() -> None:
    frame = _demand_frame(120)
    gap = horizon_bins(60, 15)
    kwargs = {
        "timestep_minutes": 15,
        "horizon_minutes": 60,
        "n_splits": 2,
        "search": _tiny_search(),
        "seed": 42,
        "gap": gap,
    }
    best, folds, val_mae = tune_demand_random_forest(frame, **kwargs)
    assert best.keys() == {"n_estimators", "max_depth", "min_samples_leaf"}
    assert not folds.empty
    assert val_mae >= 0

    poisoned = frame.copy()
    poisoned.loc[poisoned["split"] == "test", "energy_kwh"] = 1_000.0
    best_poisoned, _, _ = tune_demand_random_forest(poisoned, **kwargs)
    assert best == best_poisoned
    assert set(folds["fold"].unique()) <= set(range(2))


def test_tune_energy_ignores_test_split() -> None:
    spec = load_config().models.energy.model_copy(
        update={"n_trips": 80, "n_estimators": 12, "max_depth": 3}
    )
    trips = generate_synthetic_trips(spec, seed=42, train_fraction=0.7, val_fraction=0.15)
    search = HyperparameterGrid(n_estimators=[8, 12], max_depth=[2], min_samples_leaf=[1])
    best, folds, val_mae = tune_energy_random_forest(trips, spec=spec, search=search, seed=42)
    poisoned = trips.copy()
    poisoned.loc[poisoned["split"] == "test", "energy_kwh"] = 1_000.0
    best_poisoned, _, _ = tune_energy_random_forest(poisoned, spec=spec, search=search, seed=42)
    assert best == best_poisoned
    assert val_mae >= 0
    assert not folds.empty
