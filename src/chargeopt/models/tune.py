"""Walk-forward hyperparameter search. Demand never uses shuffled folds or the test split."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from chargeopt.config import EnergyModelConfig, HyperparameterGrid
from chargeopt.models.demand import (
    DEMAND_FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_next_hour_target,
    fit_random_forest,
    predict_random_forest,
)
from chargeopt.models.energy import fit_residual_forest, predict_residual_forest
from chargeopt.models.metrics import regression_metrics

PARAM_KEYS: tuple[str, ...] = ("n_estimators", "max_depth", "min_samples_leaf")


def param_grid(search: HyperparameterGrid) -> list[dict[str, int]]:
    return [
        {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
        for n_estimators, max_depth, min_samples_leaf in product(
            search.n_estimators,
            search.max_depth,
            search.min_samples_leaf,
        )
    ]


def expanding_window_splits(
    n_samples: int,
    *,
    n_splits: int,
    gap: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    dummy = np.zeros(n_samples)
    return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(dummy)]


def select_best_params(fold_metrics: pd.DataFrame) -> dict[str, int]:
    grouped = (
        fold_metrics.groupby(list(PARAM_KEYS), sort=True)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"))
        .sort_values(["mae", "rmse"], kind="mergesort")
    )
    if grouped.empty:
        msg = "no fold metrics to select hyperparameters from"
        raise ValueError(msg)
    best = grouped.index[0]
    if not isinstance(best, tuple):
        best = (best,)
    return {key: int(value) for key, value in zip(PARAM_KEYS, best, strict=True)}


def tune_demand_random_forest(
    demand: pd.DataFrame,
    *,
    timestep_minutes: int,
    horizon_minutes: int,
    n_splits: int,
    search: HyperparameterGrid,
    seed: int,
    gap: int,
) -> tuple[dict[str, int], pd.DataFrame, float]:
    labeled = add_next_hour_target(
        demand,
        horizon_minutes=horizon_minutes,
        timestep_minutes=timestep_minutes,
    )
    train = labeled.loc[labeled["split"] == "train"].reset_index(drop=True)
    val = labeled.loc[labeled["split"] == "val"]
    if train.empty:
        msg = "no train rows remain after the next-hour split mask"
        raise ValueError(msg)
    if val.empty:
        msg = "no val rows remain after the next-hour split mask"
        raise ValueError(msg)

    rows: list[dict[str, Any]] = []
    for params in param_grid(search):
        for fold, (train_idx, fold_val_idx) in enumerate(
            expanding_window_splits(len(train), n_splits=n_splits, gap=gap)
        ):
            fold_train = train.iloc[train_idx]
            fold_val = train.iloc[fold_val_idx]
            imputer, forest = fit_random_forest(
                fold_train,
                feature_columns=DEMAND_FEATURE_COLUMNS,
                target_column=TARGET_COLUMN,
                seed=seed,
                **params,
            )
            predicted = predict_random_forest(
                fold_val,
                imputer=imputer,
                forest=forest,
                feature_columns=DEMAND_FEATURE_COLUMNS,
            )
            stats = regression_metrics(
                fold_val[TARGET_COLUMN].reset_index(drop=True),
                pd.Series(predicted),
            )
            rows.append({"fold": fold, **params, **stats})

    fold_metrics = pd.DataFrame(rows)
    best = select_best_params(fold_metrics)
    imputer, forest = fit_random_forest(
        train,
        feature_columns=DEMAND_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        seed=seed,
        **best,
    )
    val_pred = predict_random_forest(
        val,
        imputer=imputer,
        forest=forest,
        feature_columns=DEMAND_FEATURE_COLUMNS,
    )
    val_stats = regression_metrics(val[TARGET_COLUMN].reset_index(drop=True), pd.Series(val_pred))
    return best, fold_metrics, float(val_stats["mae"])


def tune_energy_random_forest(
    trips: pd.DataFrame,
    *,
    spec: EnergyModelConfig,
    search: HyperparameterGrid,
    seed: int,
) -> tuple[dict[str, int], pd.DataFrame, float]:
    train = trips.loc[trips["split"] == "train"]
    val = trips.loc[trips["split"] == "val"]
    if train.empty:
        msg = "synthetic trips have no train split"
        raise ValueError(msg)
    if val.empty:
        msg = "synthetic trips have no val split"
        raise ValueError(msg)

    rows: list[dict[str, Any]] = []
    for params in param_grid(search):
        imputer, forest = fit_residual_forest(train, spec=spec, seed=seed, **params)
        predicted = predict_residual_forest(val, spec=spec, imputer=imputer, forest=forest)
        stats = regression_metrics(val["energy_kwh"].reset_index(drop=True), pd.Series(predicted))
        rows.append({"fold": 0, **params, **stats})

    fold_metrics = pd.DataFrame(rows)
    best = select_best_params(fold_metrics)
    imputer, forest = fit_residual_forest(train, spec=spec, seed=seed, **best)
    val_pred = predict_residual_forest(val, spec=spec, imputer=imputer, forest=forest)
    val_stats = regression_metrics(val["energy_kwh"].reset_index(drop=True), pd.Series(val_pred))
    return best, fold_metrics, float(val_stats["mae"])
