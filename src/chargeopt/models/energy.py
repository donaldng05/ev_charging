"""Physics baseline and residual sklearn trip-energy models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from chargeopt.config import LEARNER_NAMES, EnergyModelConfig
from chargeopt.features.energy import generate_synthetic_trips
from chargeopt.models.learners import FittedLearner, fit_learner, predict_learner
from chargeopt.models.metrics import metrics_from_predictions, regression_metrics

PHYSICS = "physics"
RANDOM_FOREST = "random_forest"
RESIDUAL_COLUMN = "energy_residual_kwh"
ENERGY_FEATURE_COLUMNS: tuple[str, ...] = ("distance_km", "temperature_c")


def physics_energy(distance_km: pd.Series, rate_kwh_per_km: float) -> pd.Series:
    return distance_km.astype(float) * rate_kwh_per_km


def residual_energy_target(trips: pd.DataFrame, rate_kwh_per_km: float) -> pd.Series:
    return trips["energy_kwh"].astype(float) - physics_energy(trips["distance_km"], rate_kwh_per_km)


def fit_residual_learner(
    train: pd.DataFrame, *, spec: EnergyModelConfig, name: str, params: Mapping[str, Any], seed: int
) -> FittedLearner:
    frame = train.copy()
    frame[RESIDUAL_COLUMN] = residual_energy_target(train, spec.rate_kwh_per_km)
    return fit_learner(
        frame,
        name=name,
        feature_columns=ENERGY_FEATURE_COLUMNS,
        target_column=RESIDUAL_COLUMN,
        params=params,
        seed=seed,
    )


def predict_residual_learner(
    trips: pd.DataFrame, *, spec: EnergyModelConfig, fitted: FittedLearner
) -> np.ndarray:
    res = predict_learner(trips, fitted, feature_columns=ENERGY_FEATURE_COLUMNS)
    phys = physics_energy(trips["distance_km"], spec.rate_kwh_per_km).to_numpy()
    return np.asarray(np.maximum(phys + res, 0.0), dtype=float)


def train_and_predict_energy(
    trips: pd.DataFrame, *, spec: EnergyModelConfig, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = trips.loc[trips["split"] == "train"]
    if train.empty:
        raise ValueError("synthetic trips have no train split")

    phys = physics_energy(trips["distance_km"], spec.rate_kwh_per_km)
    frames = [_energy_prediction_frame(trips, phys.to_numpy(), PHYSICS, seed)]
    for name in LEARNER_NAMES:
        fitted = fit_residual_learner(
            train, spec=spec, name=name, params=spec.learners.params_for(name), seed=seed
        )
        frames.append(
            _energy_prediction_frame(
                trips, predict_residual_learner(trips, spec=spec, fitted=fitted), name, seed
            )
        )
    predictions = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["target", "prediction"])
        .reset_index(drop=True)
    )
    return predictions, metrics_from_predictions(predictions)


def evaluate_energy_cold_holdout(
    train: pd.DataFrame,
    *,
    spec: EnergyModelConfig,
    seed: int,
    temperature_c: float,
    n_trips: int,
    train_fraction: float,
    val_fraction: float,
) -> pd.DataFrame:
    cold = generate_synthetic_trips(
        spec.model_copy(update={"n_trips": n_trips}),
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        temperature_c=temperature_c,
    )
    phys = physics_energy(cold["distance_km"], spec.rate_kwh_per_km).to_numpy()
    rows = [
        {
            "model": PHYSICS,
            "split": "cold",
            **regression_metrics(cold["energy_kwh"], pd.Series(phys)),
        }
    ]
    for name in LEARNER_NAMES:
        fitted = fit_residual_learner(
            train, spec=spec, name=name, params=spec.learners.params_for(name), seed=seed
        )
        pred = predict_residual_learner(cold, spec=spec, fitted=fitted)
        rows.append(
            {
                "model": name,
                "split": "cold",
                **regression_metrics(cold["energy_kwh"], pd.Series(pred)),
            }
        )
    return pd.DataFrame(rows)


def _energy_prediction_frame(
    trips: pd.DataFrame, prediction: np.ndarray, model: str, seed: int
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": trips["trip_id"].to_numpy(),
            "split": trips["split"].to_numpy(),
            "target": trips["energy_kwh"].to_numpy(),
            "prediction": prediction,
            "model": model,
            "seed": seed,
        }
    )
