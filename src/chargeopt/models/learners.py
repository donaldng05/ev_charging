"""Shared sklearn regressor registry for demand and residual energy models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chargeopt.config import LEARNER_NAMES

RANDOM_FOREST = "random_forest"
RIDGE = "ridge"
ELASTICNET = "elasticnet"
EXTRA_TREES = "extra_trees"
HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
LINEAR_LEARNERS = frozenset({RIDGE, ELASTICNET})


@dataclass(frozen=True)
class FittedLearner:
    name: str
    pipeline: Pipeline


def build_estimator(name: str, params: Mapping[str, Any], seed: int) -> BaseEstimator:
    if name not in LEARNER_NAMES:
        raise ValueError(f"unknown learner {name!r}; expected one of: {', '.join(LEARNER_NAMES)}")
    if name in (RANDOM_FOREST, EXTRA_TREES):
        cls = RandomForestRegressor if name == RANDOM_FOREST else ExtraTreesRegressor
        return cls(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed,
            n_jobs=1,
        )
    if name == RIDGE:
        return Ridge(alpha=float(params["alpha"]))
    if name == ELASTICNET:
        return ElasticNet(
            alpha=float(params["alpha"]), l1_ratio=float(params["l1_ratio"]), max_iter=10_000
        )
    return HistGradientBoostingRegressor(
        max_iter=int(params["max_iter"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=seed,
        early_stopping=False,
    )


def build_pipeline(name: str, params: Mapping[str, Any], seed: int) -> Pipeline:
    steps: list[tuple[str, BaseEstimator]] = [("imputer", SimpleImputer(strategy="median"))]
    if name in LINEAR_LEARNERS:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", build_estimator(name, params, seed)))
    return Pipeline(steps)


def fit_learner(
    train: pd.DataFrame,
    *,
    name: str,
    feature_columns: tuple[str, ...],
    target_column: str,
    params: Mapping[str, Any],
    seed: int,
) -> FittedLearner:
    pipeline = build_pipeline(name, params, seed)
    features = train.loc[:, list(feature_columns)].astype(float)
    pipeline.fit(features, train[target_column].to_numpy(dtype=float))
    return FittedLearner(name=name, pipeline=pipeline)


def predict_learner(
    frame: pd.DataFrame,
    fitted: FittedLearner,
    *,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    features = frame.loc[:, list(feature_columns)].astype(float)
    return np.asarray(fitted.pipeline.predict(features), dtype=float)
