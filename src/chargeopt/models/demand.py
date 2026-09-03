"""Next-hour demand target, split mask, and sklearn demand forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chargeopt.config import LEARNER_NAMES, LearnerSuite
from chargeopt.features.demand import LAG_1W
from chargeopt.models.baselines import (
    fit_historical_average,
    last_observation_forecast,
    predict_historical_average,
    weekly_naive_forecast,
)
from chargeopt.models.learners import fit_learner, predict_learner
from chargeopt.models.metrics import metrics_from_predictions

TARGET_COLUMN = "target_next_hour_energy_kwh"
LAST_OBSERVATION = "last_observation"
HISTORICAL_AVERAGE = "historical_average"
WEEKLY_NAIVE = "weekly_naive"
RANDOM_FOREST = "random_forest"
DEMAND_BASELINES: tuple[str, ...] = (LAST_OBSERVATION, HISTORICAL_AVERAGE, WEEKLY_NAIVE)

DEMAND_FEATURE_COLUMNS: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "lag_15m",
    "lag_1h",
    "lag_24h",
    "lag_1w",
    "rolling_mean_1h",
    "rolling_mean_24h",
    "rolling_mean_7d",
)
FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "energy_kwh",
    "n_arrivals",
    TARGET_COLUMN,
    "split",
    "timestamp",
    "site_id",
    "era",
)


def horizon_bins(horizon_minutes: int, timestep_minutes: int) -> int:
    if timestep_minutes <= 0 or horizon_minutes % timestep_minutes != 0:
        raise ValueError("horizon_minutes must be a positive multiple of timestep_minutes")
    n_bins = horizon_minutes // timestep_minutes
    if n_bins < 1:
        raise ValueError("horizon must cover at least one timestep")
    return n_bins


def add_next_hour_target(
    demand: pd.DataFrame, *, horizon_minutes: int, timestep_minutes: int
) -> pd.DataFrame:
    n_bins = horizon_bins(horizon_minutes, timestep_minutes)
    energy = demand["energy_kwh"].astype(float)
    target = energy.shift(-1)
    for step in range(2, n_bins + 1):
        target = target + energy.shift(-step)
    same_split = pd.Series(True, index=demand.index)
    for step in range(1, n_bins + 1):
        fut = demand["split"].shift(-step)
        same_split &= fut.notna() & fut.eq(demand["split"])
    labeled = demand.copy()
    labeled[TARGET_COLUMN] = target
    return labeled.loc[same_split].reset_index(drop=True)


def train_and_predict_demand(
    demand: pd.DataFrame,
    *,
    timestep_minutes: int,
    horizon_minutes: int,
    learners: LearnerSuite,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = add_next_hour_target(
        demand, horizon_minutes=horizon_minutes, timestep_minutes=timestep_minutes
    )
    fit_frame = labeled.loc[labeled["split"].isin(["train", "val"])]
    if fit_frame.empty:
        raise ValueError("no train/val rows remain after the next-hour split mask")

    n_bins = horizon_bins(horizon_minutes, timestep_minutes)
    hist_avg = fit_historical_average(fit_frame, target_column=TARGET_COLUMN)
    frames = [
        _demand_prediction_frame(
            labeled, last_observation_forecast(labeled, n_bins).to_numpy(), LAST_OBSERVATION, seed
        ),
        _demand_prediction_frame(
            labeled,
            predict_historical_average(labeled, hist_avg).to_numpy(),
            HISTORICAL_AVERAGE,
            seed,
        ),
        _demand_prediction_frame(
            labeled,
            weekly_naive_forecast(labeled[TARGET_COLUMN], n_week_bins=LAG_1W).to_numpy(),
            WEEKLY_NAIVE,
            seed,
        ),
    ]
    for name in LEARNER_NAMES:
        fitted = fit_learner(
            fit_frame,
            name=name,
            feature_columns=DEMAND_FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            params=learners.params_for(name),
            seed=seed,
        )
        pred = predict_learner(labeled, fitted, feature_columns=DEMAND_FEATURE_COLUMNS)
        frames.append(_demand_prediction_frame(labeled, pred, name, seed))

    predictions = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["target", "prediction"])
        .reset_index(drop=True)
    )
    return predictions, metrics_from_predictions(predictions)


def _demand_prediction_frame(
    labeled: pd.DataFrame, prediction: np.ndarray, model: str, seed: int
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": labeled["timestamp"].to_numpy(),
            "split": labeled["split"].to_numpy(),
            "target": labeled[TARGET_COLUMN].to_numpy(),
            "prediction": prediction,
            "model": model,
            "seed": seed,
        }
    )
