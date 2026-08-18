"""Physics baseline and Random Forest trip-energy models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chargeopt.config import EnergyModelConfig
from chargeopt.models.demand import fit_random_forest, predict_random_forest
from chargeopt.models.metrics import metrics_from_predictions

PHYSICS = "physics"
RANDOM_FOREST = "random_forest"
ENERGY_FEATURE_COLUMNS: tuple[str, ...] = (
    "distance_km",
    "duration_min",
    "temperature_c",
)


def physics_energy(distance_km: pd.Series, rate_kwh_per_km: float) -> pd.Series:
    return distance_km.astype(float) * rate_kwh_per_km


def train_and_predict_energy(
    trips: pd.DataFrame,
    *,
    spec: EnergyModelConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = trips.loc[trips["split"] == "train"]
    if train.empty:
        msg = "synthetic trips have no train split"
        raise ValueError(msg)

    physics = physics_energy(trips["distance_km"], spec.rate_kwh_per_km)
    imputer, forest = fit_random_forest(
        train,
        feature_columns=ENERGY_FEATURE_COLUMNS,
        target_column="energy_kwh",
        n_estimators=spec.n_estimators,
        max_depth=spec.max_depth,
        min_samples_leaf=spec.min_samples_leaf,
        seed=seed,
    )
    rf_pred = predict_random_forest(
        trips,
        imputer=imputer,
        forest=forest,
        feature_columns=ENERGY_FEATURE_COLUMNS,
    )
    predictions = pd.concat(
        [
            _energy_prediction_frame(trips, physics.to_numpy(), PHYSICS, seed),
            _energy_prediction_frame(trips, rf_pred, RANDOM_FOREST, seed),
        ],
        ignore_index=True,
    )
    predictions = predictions.dropna(subset=["target", "prediction"]).reset_index(drop=True)
    return predictions, metrics_from_predictions(predictions)


def _energy_prediction_frame(
    trips: pd.DataFrame,
    prediction: np.ndarray,
    model: str,
    seed: int,
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
