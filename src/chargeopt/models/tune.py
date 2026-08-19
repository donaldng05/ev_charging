"""Walk-forward hyperparameter search. Demand never uses shuffled folds or the test split."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from chargeopt.config import LEARNER_NAMES, EnergyModelConfig, LearnerSuite
from chargeopt.models.demand import (
    DEMAND_FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_next_hour_target,
)
from chargeopt.models.energy import fit_residual_learner, predict_residual_learner
from chargeopt.models.learners import fit_learner, predict_learner
from chargeopt.models.metrics import regression_metrics

METRIC_COLUMNS = frozenset({"fold", "mae", "rmse", "n", "model"})


def param_grid(search: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(search)
    if not keys:
        msg = "search grid is empty"
        raise ValueError(msg)
    return [
        dict(zip(keys, combo, strict=True)) for combo in product(*(search[key] for key in keys))
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


def select_best_params(
    fold_metrics: pd.DataFrame,
    param_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    keys = list(param_keys) if param_keys is not None else _infer_param_keys(fold_metrics)
    grouped = (
        fold_metrics.groupby(keys, sort=True)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"))
        .sort_values(["mae", "rmse"], kind="mergesort")
    )
    if grouped.empty:
        msg = "no fold metrics to select hyperparameters from"
        raise ValueError(msg)
    best = grouped.index[0]
    if not isinstance(best, tuple):
        best = (best,)
    return {key: _coerce_param(value) for key, value in zip(keys, best, strict=True)}


def resolve_learner_names(requested: str | None) -> tuple[str, ...]:
    if requested is None:
        return LEARNER_NAMES
    if requested not in LEARNER_NAMES:
        allowed = ", ".join(LEARNER_NAMES)
        msg = f"unknown learner {requested!r}; expected one of: {allowed}"
        raise ValueError(msg)
    return (requested,)


def tune_demand_learner(
    demand: pd.DataFrame,
    *,
    learner: str,
    search: Mapping[str, Sequence[Any]],
    timestep_minutes: int,
    horizon_minutes: int,
    n_splits: int,
    seed: int,
    gap: int,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
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
            fitted = fit_learner(
                fold_train,
                name=learner,
                feature_columns=DEMAND_FEATURE_COLUMNS,
                target_column=TARGET_COLUMN,
                params=params,
                seed=seed,
            )
            predicted = predict_learner(
                fold_val,
                fitted,
                feature_columns=DEMAND_FEATURE_COLUMNS,
            )
            stats = regression_metrics(
                fold_val[TARGET_COLUMN].reset_index(drop=True),
                pd.Series(predicted),
            )
            rows.append({"model": learner, "fold": fold, **params, **stats})

    fold_metrics = pd.DataFrame(rows)
    best = select_best_params(fold_metrics, list(search.keys()))
    fitted = fit_learner(
        train,
        name=learner,
        feature_columns=DEMAND_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        params=best,
        seed=seed,
    )
    val_pred = predict_learner(val, fitted, feature_columns=DEMAND_FEATURE_COLUMNS)
    val_stats = regression_metrics(val[TARGET_COLUMN].reset_index(drop=True), pd.Series(val_pred))
    return best, fold_metrics, float(val_stats["mae"])


def tune_demand_learners(
    demand: pd.DataFrame,
    *,
    learners: LearnerSuite,
    timestep_minutes: int,
    horizon_minutes: int,
    n_splits: int,
    seed: int,
    gap: int,
    names: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, float]]:
    selected = tuple(names) if names is not None else LEARNER_NAMES
    best_by_model: dict[str, dict[str, Any]] = {}
    val_mae: dict[str, float] = {}
    frames: list[pd.DataFrame] = []
    for name in selected:
        best, folds, mae = tune_demand_learner(
            demand,
            learner=name,
            search=learners.search_for(name),
            timestep_minutes=timestep_minutes,
            horizon_minutes=horizon_minutes,
            n_splits=n_splits,
            seed=seed,
            gap=gap,
        )
        best_by_model[name] = best
        val_mae[name] = mae
        frames.append(folds)
    return best_by_model, pd.concat(frames, ignore_index=True), val_mae


def tune_energy_learner(
    trips: pd.DataFrame,
    *,
    spec: EnergyModelConfig,
    learner: str,
    search: Mapping[str, Sequence[Any]],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
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
        fitted = fit_residual_learner(train, spec=spec, name=learner, params=params, seed=seed)
        predicted = predict_residual_learner(val, spec=spec, fitted=fitted)
        stats = regression_metrics(val["energy_kwh"].reset_index(drop=True), pd.Series(predicted))
        rows.append({"model": learner, "fold": 0, **params, **stats})

    fold_metrics = pd.DataFrame(rows)
    best = select_best_params(fold_metrics, list(search.keys()))
    fitted = fit_residual_learner(train, spec=spec, name=learner, params=best, seed=seed)
    val_pred = predict_residual_learner(val, spec=spec, fitted=fitted)
    val_stats = regression_metrics(val["energy_kwh"].reset_index(drop=True), pd.Series(val_pred))
    return best, fold_metrics, float(val_stats["mae"])


def tune_energy_learners(
    trips: pd.DataFrame,
    *,
    spec: EnergyModelConfig,
    seed: int,
    names: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, float]]:
    selected = tuple(names) if names is not None else LEARNER_NAMES
    best_by_model: dict[str, dict[str, Any]] = {}
    val_mae: dict[str, float] = {}
    frames: list[pd.DataFrame] = []
    for name in selected:
        best, folds, mae = tune_energy_learner(
            trips,
            spec=spec,
            learner=name,
            search=spec.learners.search_for(name),
            seed=seed,
        )
        best_by_model[name] = best
        val_mae[name] = mae
        frames.append(folds)
    return best_by_model, pd.concat(frames, ignore_index=True), val_mae


def _infer_param_keys(fold_metrics: pd.DataFrame) -> list[str]:
    return [column for column in fold_metrics.columns if column not in METRIC_COLUMNS]


def _coerce_param(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
