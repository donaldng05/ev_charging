"""Seeded synthetic trip generation for the energy model."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from chargeopt.config import EnergyModelConfig
from chargeopt.data.schemas import SyntheticTrip
from chargeopt.features.demand import temporal_split_labels

TRIP_COLUMNS: tuple[str, ...] = (
    "trip_id",
    "distance_km",
    "duration_min",
    "speed_kmh",
    "temperature_c",
    "energy_kwh",
    "split",
)


def generate_synthetic_trips(
    spec: EnergyModelConfig,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    temperature_c: float | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_trips = spec.n_trips
    distance = np.clip(
        rng.normal(spec.distance_km_mean, spec.distance_km_std, n_trips),
        0.5,
        None,
    )
    duration = np.clip(
        rng.normal(spec.duration_min_mean, spec.duration_min_std, n_trips),
        1.0,
        None,
    )
    if temperature_c is None:
        temperature = rng.normal(spec.temperature_mean_c, spec.temperature_std_c, n_trips)
    else:
        temperature = np.full(n_trips, temperature_c)
    speed_kmh = distance / (duration / 60.0)
    cold_delta = np.maximum(spec.temperature_reference_c - temperature, 0.0)
    noise = rng.normal(0.0, spec.noise_std_kwh, n_trips)
    energy = spec.rate_kwh_per_km * distance * (1.0 + spec.cold_penalty_per_c * cold_delta) + noise
    energy = np.maximum(energy, 0.0)
    splits = temporal_split_labels(n_trips, train_fraction, val_fraction)
    frame = pd.DataFrame(
        {
            "trip_id": [f"trip-{index:05d}" for index in range(n_trips)],
            "distance_km": distance,
            "duration_min": duration,
            "speed_kmh": speed_kmh,
            "temperature_c": temperature,
            "energy_kwh": energy,
            "split": splits,
        }
    )
    records = cast(list[dict[str, Any]], frame.to_dict(orient="records"))
    validated = [SyntheticTrip.model_validate(record).model_dump() for record in records]
    return pd.DataFrame(validated)
