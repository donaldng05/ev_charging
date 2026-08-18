"""Physics baseline and residual Random Forest trip-energy models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from chargeopt.config import EnergyModelConfig
from chargeopt.features.energy import generate_synthetic_trips
from chargeopt.models.demand import fit_random_forest, predict_random_forest
from chargeopt.models.metrics import metrics_from_predictions, regression_metrics

PHYSICS = "physics"
RANDOM_FOREST = "random_forest"
RESIDUAL_COLUMN = "energy_residual_kwh"
ENERGY_FEATURE_COLUMNS: tuple[str, ...] = (
    "distance_km",
    "temperature_c",
)


def physics_energy(distance_km: pd.Series, rate_kwh_per_km: float) -> pd.Series:
    return distance_km.astype(float) * rate_kwh_per_km


def residual_energy_target(trips: pd.DataFrame, rate_kwh_per_km: float) -> pd.Series:
    return trips["energy_kwh"].astype(float) - physics_energy(trips["distance_km"], rate_kwh_per_km)


def fit_residual_forest(
    train: pd.DataFrame,
    *,
    spec: EnergyModelConfig,
    seed: int,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
) -> tuple[SimpleImputer, RandomForestRegressor]:
    residual_frame = train.copy()
    residual_frame[RESIDUAL_COLUMN] = residual_energy_target(train, spec.rate_kwh_per_km)
    return fit_random_forest(
        residual_frame,
        feature_columns=ENERGY_FEATURE_COLUMNS,
        target_column=RESIDUAL_COLUMN,
        n_estimators=n_estimators or spec.n_estimators,
        max_depth=max_depth or spec.max_depth,
        min_samples_leaf=min_samples_leaf or spec.min_samples_leaf,
        seed=seed,
    )


def predict_residual_forest(
    trips: pd.DataFrame,
    *,
    spec: EnergyModelConfig,
    imputer: SimpleImputer,
    forest: RandomForestRegressor,
) -> np.ndarray:
    residual = predict_random_forest(
        trips,
        imputer=imputer,
        forest=forest,
        feature_columns=ENERGY_FEATURE_COLUMNS,
    )
    physics = physics_energy(trips["distance_km"], spec.rate_kwh_per_km).to_numpy()
    clipped = np.asarray(np.maximum(physics + residual, 0.0), dtype=float)
    return clipped


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
    imputer, forest = fit_residual_forest(train, spec=spec, seed=seed)
    rf_pred = predict_residual_forest(trips, spec=spec, imputer=imputer, forest=forest)
    predictions = pd.concat(
        [
            _energy_prediction_frame(trips, physics.to_numpy(), PHYSICS, seed),
            _energy_prediction_frame(trips, rf_pred, RANDOM_FOREST, seed),
        ],
        ignore_index=True,
    )
    predictions = predictions.dropna(subset=["target", "prediction"]).reset_index(drop=True)
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
    cold_spec = spec.model_copy(update={"n_trips": n_trips})
    cold = generate_synthetic_trips(
        cold_spec,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        temperature_c=temperature_c,
    )
    imputer, forest = fit_residual_forest(train, spec=spec, seed=seed)
    physics = physics_energy(cold["distance_km"], spec.rate_kwh_per_km).to_numpy()
    rf_pred = predict_residual_forest(cold, spec=spec, imputer=imputer, forest=forest)
    rows = []
    for model, prediction in ((PHYSICS, physics), (RANDOM_FOREST, rf_pred)):
        stats = regression_metrics(cold["energy_kwh"], pd.Series(prediction))
        rows.append({"model": model, "split": "cold", **stats})
    return pd.DataFrame(rows)


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
