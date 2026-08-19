"""Walk-forward demand tuning never sees the chronological test split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chargeopt.config import LEARNER_NAMES, load_config
from chargeopt.features.energy import generate_synthetic_trips
from chargeopt.models.demand import horizon_bins
from chargeopt.models.tune import (
    expanding_window_splits,
    param_grid,
    resolve_learner_names,
    select_best_params,
    tune_demand_learner,
    tune_demand_learners,
    tune_energy_learner,
    tune_energy_learners,
)
from tests.unit.model_helpers import fast_learners


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


def test_param_grid_is_cartesian_product() -> None:
    combos = param_grid({"n_estimators": [10, 20], "max_depth": [2, 4], "min_samples_leaf": [1]})
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


def test_select_best_params_keeps_float_alpha() -> None:
    metrics = pd.DataFrame(
        [
            {"alpha": 0.1, "mae": 2.0, "rmse": 3.0},
            {"alpha": 0.1, "mae": 4.0, "rmse": 4.0},
            {"alpha": 10.0, "mae": 5.0, "rmse": 5.0},
        ]
    )
    best = select_best_params(metrics, ("alpha",))
    assert best == {"alpha": 0.1}
    assert isinstance(best["alpha"], float)


def test_resolve_learner_names_filters_or_returns_all() -> None:
    assert resolve_learner_names(None) == LEARNER_NAMES
    assert resolve_learner_names("ridge") == ("ridge",)


def test_tune_demand_ignores_test_energy() -> None:
    frame = _demand_frame(120)
    gap = horizon_bins(60, 15)
    search = {"n_estimators": [8, 12], "max_depth": [2], "min_samples_leaf": [1]}
    kwargs = {
        "learner": "random_forest",
        "search": search,
        "timestep_minutes": 15,
        "horizon_minutes": 60,
        "n_splits": 2,
        "seed": 42,
        "gap": gap,
    }
    best, folds, val_mae = tune_demand_learner(frame, **kwargs)
    assert best.keys() == {"n_estimators", "max_depth", "min_samples_leaf"}
    assert set(folds["model"]) == {"random_forest"}
    assert not folds.empty
    assert val_mae >= 0

    poisoned = frame.copy()
    poisoned.loc[poisoned["split"] == "test", "energy_kwh"] = 1_000.0
    best_poisoned, _, _ = tune_demand_learner(poisoned, **kwargs)
    assert best == best_poisoned
    assert set(folds["fold"].unique()) <= set(range(2))


def test_tune_demand_ridge_ignores_test_energy() -> None:
    frame = _demand_frame(120)
    kwargs = {
        "learner": "ridge",
        "search": {"alpha": [0.1, 1.0]},
        "timestep_minutes": 15,
        "horizon_minutes": 60,
        "n_splits": 2,
        "seed": 42,
        "gap": horizon_bins(60, 15),
    }
    best, folds, _ = tune_demand_learner(frame, **kwargs)
    poisoned = frame.copy()
    poisoned.loc[poisoned["split"] == "test", "energy_kwh"] = 1_000.0
    best_poisoned, _, _ = tune_demand_learner(poisoned, **kwargs)
    assert best == best_poisoned
    assert isinstance(best["alpha"], float)
    assert set(folds["model"]) == {"ridge"}


def test_tune_demand_learners_writes_model_column() -> None:
    frame = _demand_frame(120)
    best, folds, val_mae = tune_demand_learners(
        frame,
        learners=fast_learners(),
        timestep_minutes=15,
        horizon_minutes=60,
        n_splits=2,
        seed=42,
        gap=horizon_bins(60, 15),
        names=("ridge", "random_forest"),
    )
    assert set(best) == {"ridge", "random_forest"}
    assert set(val_mae) == {"ridge", "random_forest"}
    assert set(folds["model"]) == {"ridge", "random_forest"}


def test_tune_energy_ignores_test_split() -> None:
    spec = load_config().models.energy.model_copy(
        update={"n_trips": 80, "learners": fast_learners(n_estimators=12, max_depth=3)}
    )
    trips = generate_synthetic_trips(spec, seed=42, train_fraction=0.7, val_fraction=0.15)
    search = {"n_estimators": [8, 12], "max_depth": [2], "min_samples_leaf": [1]}
    best, folds, val_mae = tune_energy_learner(
        trips, spec=spec, learner="random_forest", search=search, seed=42
    )
    poisoned = trips.copy()
    poisoned.loc[poisoned["split"] == "test", "energy_kwh"] = 1_000.0
    best_poisoned, _, _ = tune_energy_learner(
        poisoned, spec=spec, learner="random_forest", search=search, seed=42
    )
    assert best == best_poisoned
    assert val_mae >= 0
    assert not folds.empty
    assert set(folds["model"]) == {"random_forest"}


def test_tune_energy_learners_covers_selected_names() -> None:
    spec = load_config().models.energy.model_copy(
        update={"n_trips": 80, "learners": fast_learners(n_estimators=12, max_depth=3)}
    )
    trips = generate_synthetic_trips(spec, seed=42, train_fraction=0.7, val_fraction=0.15)
    best, folds, val_mae = tune_energy_learners(
        trips, spec=spec, seed=42, names=("ridge", "elasticnet")
    )
    assert set(best) == {"ridge", "elasticnet"}
    assert set(val_mae) == {"ridge", "elasticnet"}
    assert set(folds["model"]) == {"ridge", "elasticnet"}
