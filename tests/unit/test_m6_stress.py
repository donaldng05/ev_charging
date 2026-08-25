"""M6 stress overrides preserve the normal simulation contract."""

from pathlib import Path

import pandas as pd

from chargeopt.config import load_config
from chargeopt.data.io import read_sessions_csv
from chargeopt.simulation.engine import run_simulation


def test_explicit_normal_overrides_match_legacy_simulation() -> None:
    config = load_config()
    sessions = read_sessions_csv(Path("tests/fixtures/acn_sessions.csv"))

    legacy = run_simulation(config, sessions=sessions, seed=42)
    explicit_normal = run_simulation(
        config,
        sessions=sessions,
        seed=42,
        temperature_c=config.models.energy.temperature_mean_c,
        trip_rate_multiplier=config.simulation.trip_rate_multiplier,
        station_availability=None,
    )

    assert legacy.metrics == explicit_normal.metrics
    pd.testing.assert_frame_equal(legacy.stations, explicit_normal.stations)
    pd.testing.assert_frame_equal(legacy.vehicle_ticks, explicit_normal.vehicle_ticks)
    pd.testing.assert_frame_equal(legacy.station_ticks, explicit_normal.station_ticks)


def test_stress_availability_reduces_active_capacity() -> None:
    config = load_config()
    sessions = read_sessions_csv(Path("tests/fixtures/acn_sessions.csv"))

    normal = run_simulation(config, sessions=sessions, seed=42)
    stress = run_simulation(
        config,
        sessions=sessions,
        seed=42,
        temperature_c=config.stress.temperature_c,
        trip_rate_multiplier=(
            config.simulation.trip_rate_multiplier * config.stress.demand_multiplier
        ),
        station_availability=config.stress.station_availability,
    )

    assert len(normal.stations) == len(stress.stations) == config.simulation.n_stations
    assert list(
        normal.stations[["station_id", "x_km", "y_km"]].itertuples(index=False, name=None)
    ) == list(stress.stations[["station_id", "x_km", "y_km"]].itertuples(index=False, name=None))
    assert (stress.stations["n_chargers"] == 0).sum() == 2
    assert stress.stations["n_chargers"].sum() < normal.stations["n_chargers"].sum()
