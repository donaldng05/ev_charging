"""Synthetic trip generation, physics baseline, and energy Random Forest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chargeopt.config import load_config
from chargeopt.features.energy import generate_synthetic_trips
from chargeopt.models.energy import ENERGY_FEATURE_COLUMNS, physics_energy, train_and_predict_energy


def _energy_spec(*, n_trips: int = 80):
    spec = load_config().models.energy
    return spec.model_copy(update={"n_trips": n_trips, "n_estimators": 30, "max_depth": 4})


def test_physics_energy_is_rate_times_distance() -> None:
    distance = pd.Series([10.0, 20.0, 5.0])
    predicted = physics_energy(distance, rate_kwh_per_km=0.18)
    pd.testing.assert_series_equal(predicted, distance * 0.18)


def test_synthetic_trips_are_seeded_and_split_chronologically() -> None:
    spec = _energy_spec()
    first = generate_synthetic_trips(spec, seed=42, train_fraction=0.7, val_fraction=0.15)
    second = generate_synthetic_trips(spec, seed=42, train_fraction=0.7, val_fraction=0.15)
    pd.testing.assert_frame_equal(first, second)
    assert list(first["split"].unique()) == ["train", "val", "test"]
    splits = first["split"].tolist()
    assert splits == sorted(splits, key=["train", "val", "test"].index)


def test_energy_features_are_distance_duration_and_temperature() -> None:
    assert ENERGY_FEATURE_COLUMNS == ("distance_km", "duration_min", "temperature_c")


def test_energy_forest_fits_train_only_and_beats_physics_on_test() -> None:
    spec = _energy_spec(n_trips=200)
    trips = generate_synthetic_trips(spec, seed=42, train_fraction=0.7, val_fraction=0.15)
    predictions, metrics = train_and_predict_energy(trips, spec=spec, seed=42)
    assert set(predictions["model"]) == {"physics", "random_forest"}
    required = {"trip_id", "split", "target", "prediction", "model", "seed"}
    assert required <= set(predictions.columns)

    test = metrics.loc[metrics["split"] == "test"].set_index("model")
    assert float(test.loc["random_forest", "mae"]) < float(test.loc["physics", "mae"])


def test_energy_forest_is_reproducible() -> None:
    spec = _energy_spec(n_trips=80)
    trips = generate_synthetic_trips(spec, seed=7, train_fraction=0.7, val_fraction=0.15)
    first, _ = train_and_predict_energy(trips, spec=spec, seed=7)
    second, _ = train_and_predict_energy(trips, spec=spec, seed=7)
    rf = first.loc[first["model"] == "random_forest", "prediction"].to_numpy()
    rf_again = second.loc[second["model"] == "random_forest", "prediction"].to_numpy()
    np.testing.assert_allclose(rf, rf_again)
