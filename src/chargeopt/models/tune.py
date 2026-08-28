"""Walk-forward hyperparameter search. Demand never uses shuffled folds or the test split."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
        raise ValueError("search grid is empty")
    return [dict(zip(keys, combo, strict=True)) for combo in product(*(search[k] for k in keys))]


def expanding_window_splits(
    n_samples: int,
    *,
    n_splits: int,
    gap: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    return list(splitter.split(np.zeros(n_samples)))


def _coerce_param(value: Any) -> Any:
    return (
        int(value)
        if isinstance(value, np.integer)
        else float(value)
        if isinstance(value, np.floating)
        else value
    )


def select_best_params(
    fold_metrics: pd.DataFrame,
    param_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    keys = (
        list(param_keys)
        if param_keys is not None
        else [c for c in fold_metrics.columns if c not in METRIC_COLUMNS]
    )
    grouped = (
        fold_metrics.groupby(keys, sort=True)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"))
        .sort_values(["mae", "rmse"], kind="mergesort")
    )
    if grouped.empty:
        raise ValueError("no fold metrics to select hyperparameters from")
    best = grouped.index[0]
    best_tuple = best if isinstance(best, tuple) else (best,)
    return {k: _coerce_param(v) for k, v in zip(keys, best_tuple, strict=True)}


def resolve_learner_names(requested: str | None) -> tuple[str, ...]:
    if requested is None:
        return LEARNER_NAMES
    if requested not in LEARNER_NAMES:
        allowed = ", ".join(LEARNER_NAMES)
        raise ValueError(f"unknown learner {requested!r}; expected one of: {allowed}")
    return (requested,)


def _tune_grid(
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    search: Mapping[str, Sequence[Any]],
    learner: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fit_fn: Callable[[pd.DataFrame, dict[str, Any]], Any],
    predict_fn: Callable[[pd.DataFrame, Any], np.ndarray],
    target_col: str,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    for params in param_grid(search):
        for fold, (t_idx, v_idx) in enumerate(splits):
            fold_train, fold_val = train.iloc[t_idx], train.iloc[v_idx]
            fitted = fit_fn(fold_train, params)
            pred = predict_fn(fold_val, fitted)
            stats = regression_metrics(fold_val[target_col].reset_index(drop=True), pd.Series(pred))
            rows.append({"model": learner, "fold": fold, **params, **stats})
    fold_metrics = pd.DataFrame(rows)
    best = select_best_params(fold_metrics, list(search.keys()))
    fitted = fit_fn(train, best)
    val_pred = predict_fn(val, fitted)
    val_stats = regression_metrics(val[target_col].reset_index(drop=True), pd.Series(val_pred))
    return best, fold_metrics, float(val_stats["mae"])


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
        demand, horizon_minutes=horizon_minutes, timestep_minutes=timestep_minutes
    )
    train = labeled.loc[labeled["split"] == "train"].reset_index(drop=True)
    val = labeled.loc[labeled["split"] == "val"]
    if train.empty or val.empty:
        raise ValueError("train/val rows missing after next-hour split mask")

    return _tune_grid(
        train,
        val,
        search=search,
        learner=learner,
        splits=expanding_window_splits(len(train), n_splits=n_splits, gap=gap),
        fit_fn=lambda df, p: fit_learner(
            df,
            name=learner,
            feature_columns=DEMAND_FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            params=p,
            seed=seed,
        ),
        predict_fn=lambda df, f: predict_learner(df, f, feature_columns=DEMAND_FEATURE_COLUMNS),
        target_col=TARGET_COLUMN,
    )


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
    results = [
        (
            name,
            *tune_demand_learner(
                demand,
                learner=name,
                search=learners.search_for(name),
                timestep_minutes=timestep_minutes,
                horizon_minutes=horizon_minutes,
                n_splits=n_splits,
                seed=seed,
                gap=gap,
            ),
        )
        for name in selected
    ]
    return (
        {name: best for name, best, _, _ in results},
        pd.concat([folds for _, _, folds, _ in results], ignore_index=True),
        {name: mae for name, _, _, mae in results},
    )


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
    if train.empty or val.empty:
        raise ValueError("synthetic trips have no train/val split")

    dummy_splits = [(np.arange(len(train)), np.arange(len(train)))]
    return _tune_grid(
        train,
        val,
        search=search,
        learner=learner,
        splits=dummy_splits,
        fit_fn=lambda df, p: fit_residual_learner(df, spec=spec, name=learner, params=p, seed=seed),
        predict_fn=lambda df, f: predict_residual_learner(df, spec=spec, fitted=f),
        target_col="energy_kwh",
    )


def tune_energy_learners(
    trips: pd.DataFrame,
    *,
    spec: EnergyModelConfig,
    seed: int,
    names: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, float]]:
    selected = tuple(names) if names is not None else LEARNER_NAMES
    results = [
        (
            name,
            *tune_energy_learner(
                trips, spec=spec, learner=name, search=spec.learners.search_for(name), seed=seed
            ),
        )
        for name in selected
    ]
    return (
        {name: best for name, best, _, _ in results},
        pd.concat([folds for _, _, folds, _ in results], ignore_index=True),
        {name: mae for name, _, _, mae in results},
    )
