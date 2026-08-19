"""Deterministic trip-energy calculations for simulation."""

import numpy as np
import pytest

from chargeopt.config import load_config
from chargeopt.features.energy import generative_trip_energy
from chargeopt.simulation.energy import deterministic_trip_energy


def test_generative_energy_applies_cold_penalty_without_noise() -> None:
    spec = load_config().models.energy
    distance = np.asarray([10.0])
    temperature = np.asarray([-10.0])

    energy = generative_trip_energy(distance, temperature, spec)

    np.testing.assert_allclose(energy, [2.34])


def test_simulation_energy_is_deterministic_and_cold_uses_more() -> None:
    spec = load_config().models.energy

    warm_first = deterministic_trip_energy(10.0, 20.0, spec)
    warm_second = deterministic_trip_energy(10.0, 20.0, spec)
    cold = deterministic_trip_energy(10.0, -10.0, spec)

    assert warm_first == warm_second
    assert warm_first == pytest.approx(1.8)
    assert cold > warm_first
