"""Deterministic energy consumption for simulated fleet trips."""

import numpy as np

from chargeopt.config import EnergyModelConfig
from chargeopt.features.energy import generative_trip_energy


def deterministic_trip_energy(
    distance_km: float,
    temperature_c: float,
    spec: EnergyModelConfig,
) -> float:
    """Apply the M2 energy contract without synthetic training noise."""
    predicted = generative_trip_energy(
        np.asarray([distance_km], dtype=float),
        np.asarray([temperature_c], dtype=float),
        spec,
    )
    return float(predicted[0])
