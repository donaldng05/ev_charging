"""Chronological demand baselines. Fit statistics come from train only."""

from __future__ import annotations

from typing import NamedTuple, cast

import pandas as pd


class HistoricalAverage(NamedTuple):
    by_hour: dict[int, float]
    global_mean: float


def last_observation_forecast(demand: pd.DataFrame, n_bins: int) -> pd.Series:
    """Persist the last observed hour: rolling mean of prior bins times horizon bins."""
    return demand["rolling_mean_1h"].astype(float) * n_bins


def weekly_naive_forecast(target: pd.Series, *, n_week_bins: int) -> pd.Series:
    """Same next-hour window from one week earlier; uses only already-observed energy."""
    if n_week_bins < 1:
        msg = "n_week_bins must be >= 1"
        raise ValueError(msg)
    return target.astype(float).shift(n_week_bins)


def fit_historical_average(train: pd.DataFrame, *, target_column: str) -> HistoricalAverage:
    means = train.groupby("hour", observed=True)[target_column].mean()
    by_hour = {int(cast(int, hour)): float(value) for hour, value in means.items()}
    return HistoricalAverage(by_hour=by_hour, global_mean=float(train[target_column].mean()))


def predict_historical_average(demand: pd.DataFrame, fitted: HistoricalAverage) -> pd.Series:
    mapped = demand["hour"].map(fitted.by_hour)
    return mapped.fillna(fitted.global_mean).astype(float)
