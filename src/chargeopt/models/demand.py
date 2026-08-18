"""Next-hour demand target, split mask, and Random Forest forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from chargeopt.models.baselines import (
    fit_historical_average,
    last_observation_forecast,
    predict_historical_average,
)
from chargeopt.models.metrics import metrics_from_predictions

TARGET_COLUMN = "target_next_hour_energy_kwh"
LAST_OBSERVATION = "last_observation"
HISTORICAL_AVERAGE = "historical_average"
RANDOM_FOREST = "random_forest"

DEMAND_FEATURE_COLUMNS: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "lag_15m",
    "lag_1h",
    "lag_24h",
    "rolling_mean_1h",
    "rolling_mean_24h",
)

FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "energy_kwh",
    "n_arrivals",
    TARGET_COLUMN,
    "split",
    "timestamp",
    "site_id",
)


def horizon_bins(horizon_minutes: int, timestep_minutes: int) -> int:
    if timestep_minutes <= 0 or horizon_minutes % timestep_minutes != 0:
        msg = "horizon_minutes must be a positive multiple of timestep_minutes"
        raise ValueError(msg)
    n_bins = horizon_minutes // timestep_minutes
    if n_bins < 1:
        msg = "horizon must cover at least one timestep"
        raise ValueError(msg)
    return n_bins


def add_next_hour_target(
    demand: pd.DataFrame,
    *,
    horizon_minutes: int,
    timestep_minutes: int,
) -> pd.DataFrame:
    n_bins = horizon_bins(horizon_minutes, timestep_minutes)
    energy = demand["energy_kwh"].astype(float)
    target = energy.shift(-1)
    for step in range(2, n_bins + 1):
        target = target + energy.shift(-step)
    same_split = pd.Series(True, index=demand.index)
    for step in range(1, n_bins + 1):
        future_split = demand["split"].shift(-step)
        same_split &= future_split.notna() & future_split.eq(demand["split"])
    labeled = demand.copy()
    labeled[TARGET_COLUMN] = target
    return labeled.loc[same_split].reset_index(drop=True)


def fit_random_forest(
    train: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_column: str,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
) -> tuple[SimpleImputer, RandomForestRegressor]:
    imputer = SimpleImputer(strategy="median")
    features = train.loc[:, list(feature_columns)].astype(float)
    imputed = imputer.fit_transform(features)
    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=1,
    )
    forest.fit(imputed, train[target_column].to_numpy(dtype=float))
    return imputer, forest


def predict_random_forest(
    frame: pd.DataFrame,
    *,
    imputer: SimpleImputer,
    forest: RandomForestRegressor,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    features = frame.loc[:, list(feature_columns)].astype(float)
    imputed = imputer.transform(features)
    predicted: np.ndarray = forest.predict(imputed)
    return predicted


def train_and_predict_demand(
    demand: pd.DataFrame,
    *,
    timestep_minutes: int,
    horizon_minutes: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = add_next_hour_target(
        demand,
        horizon_minutes=horizon_minutes,
        timestep_minutes=timestep_minutes,
    )
    n_bins = horizon_bins(horizon_minutes, timestep_minutes)
    train = labeled.loc[labeled["split"] == "train"]
    if train.empty:
        msg = "no train rows remain after the next-hour split mask"
        raise ValueError(msg)

    last_obs = last_observation_forecast(labeled, n_bins)
    historical = fit_historical_average(train, target_column=TARGET_COLUMN)
    hist_pred = predict_historical_average(labeled, historical)
    imputer, forest = fit_random_forest(
        train,
        feature_columns=DEMAND_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )
    rf_pred = predict_random_forest(
        labeled,
        imputer=imputer,
        forest=forest,
        feature_columns=DEMAND_FEATURE_COLUMNS,
    )

    predictions = pd.concat(
        [
            _demand_prediction_frame(labeled, last_obs.to_numpy(), LAST_OBSERVATION, seed),
            _demand_prediction_frame(labeled, hist_pred.to_numpy(), HISTORICAL_AVERAGE, seed),
            _demand_prediction_frame(labeled, rf_pred, RANDOM_FOREST, seed),
        ],
        ignore_index=True,
    )
    predictions = predictions.dropna(subset=["target", "prediction"]).reset_index(drop=True)
    return predictions, metrics_from_predictions(predictions)


def _demand_prediction_frame(
    labeled: pd.DataFrame,
    prediction: np.ndarray,
    model: str,
    seed: int,
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
