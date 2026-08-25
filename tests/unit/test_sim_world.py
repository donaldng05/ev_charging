"""Seeded synthetic station, vehicle, and itinerary generation."""

from itertools import pairwise
from pathlib import Path

import numpy as np

from chargeopt.config import load_config
from chargeopt.data.io import read_sessions_csv
from chargeopt.simulation.calibration import calibrate_from_sessions
from chargeopt.simulation.world import (
    _repair_departures,
    build_fleet,
    build_stations,
    build_trips,
    charging_service_ticks,
)

FIXTURE = Path("tests/fixtures/acn_sessions.csv")


def test_world_builders_use_config_and_do_not_reuse_acn_evse_ids() -> None:
    config = load_config()
    sessions = read_sessions_csv(FIXTURE)
    calibration = calibrate_from_sessions(
        sessions,
        timestep_minutes=config.simulation.timestep_minutes,
    )

    stations = build_stations(config.simulation, calibration, seed=42)
    vehicles = build_fleet(config.simulation, stations)

    assert len(stations) == config.simulation.n_stations
    assert {station.station_id for station in stations} == {
        f"sim-{index:02d}" for index in range(config.simulation.n_stations)
    }
    assert not ({station.station_id for station in stations} & set(sessions["station_id"]))
    assert all(0 <= station.x_km <= config.simulation.metro_span_km for station in stations)
    assert all(0 <= station.y_km <= config.simulation.metro_span_km for station in stations)
    assert all(station.n_chargers >= config.simulation.n_chargers_min for station in stations)
    assert (
        max(station.n_chargers for station in stations)
        - min(station.n_chargers for station in stations)
        <= 1
    )
    assert len({station.price_per_kwh for station in stations}) == len(stations)
    assert len(vehicles) == config.simulation.fleet_size
    assert vehicles[11].home_station_id == "sim-01"


def test_station_capacity_uses_energy_service_time_not_connected_duration() -> None:
    config = load_config()
    calibration = calibrate_from_sessions(
        read_sessions_csv(FIXTURE),
        timestep_minutes=config.simulation.timestep_minutes,
    )
    long_dwell = calibration.model_copy(
        update={"mean_duration_min": calibration.mean_duration_min * 100}
    )
    high_energy = calibration.model_copy(update={"mean_energy_kwh": 1000.0})

    assert charging_service_ticks(config.simulation, calibration) == 1
    baseline = build_stations(config.simulation, calibration, seed=42)
    dwell_only = build_stations(config.simulation, long_dwell, seed=42)
    energy_heavy = build_stations(config.simulation, high_energy, seed=42)

    assert [station.n_chargers for station in dwell_only] == [
        station.n_chargers for station in baseline
    ]
    assert sum(station.n_chargers for station in energy_heavy) > sum(
        station.n_chargers for station in baseline
    )


def test_trip_generation_is_seeded_ordered_and_within_horizon() -> None:
    config = load_config()
    calibration = calibrate_from_sessions(
        read_sessions_csv(FIXTURE),
        timestep_minutes=config.simulation.timestep_minutes,
    )
    stations = build_stations(config.simulation, calibration, seed=42)
    vehicles = build_fleet(config.simulation, stations)

    first = build_trips(
        config.simulation,
        config.models.energy,
        calibration,
        vehicles,
        seed=42,
        temperature_c=config.models.energy.temperature_mean_c,
    )
    second = build_trips(
        config.simulation,
        config.models.energy,
        calibration,
        vehicles,
        seed=42,
        temperature_c=config.models.energy.temperature_mean_c,
    )

    assert first == second
    assert len(first) == config.simulation.fleet_size * config.simulation.trips_per_vehicle
    for vehicle in vehicles:
        itinerary = [trip for trip in first if trip.vehicle_id == vehicle.vehicle_id]
        assert [trip.trip_index for trip in itinerary] == list(
            range(config.simulation.trips_per_vehicle)
        )
        assert all(
            current.departure_tick + current.duration_ticks < following.departure_tick
            for current, following in pairwise(itinerary)
        )
        assert all(
            trip.departure_tick + trip.duration_ticks <= config.simulation.steps_per_day
            for trip in itinerary
        )


def test_trip_rate_multiplier_scales_effective_itinerary_size() -> None:
    config = load_config()
    calibration = calibrate_from_sessions(
        read_sessions_csv(FIXTURE),
        timestep_minutes=config.simulation.timestep_minutes,
    )
    stations = build_stations(config.simulation, calibration, seed=42)
    vehicles = build_fleet(config.simulation, stations)

    trips = build_trips(
        config.simulation,
        config.models.energy,
        calibration,
        vehicles,
        seed=42,
        temperature_c=config.models.energy.temperature_mean_c,
        trip_rate_multiplier=7.0,
    )

    assert len(trips) == config.simulation.fleet_size * 14


def test_repair_departures_keeps_late_itinerary_inside_horizon() -> None:
    durations = np.asarray([1, 2])

    departures = _repair_departures(
        np.asarray([0, 95]),
        durations,
        horizon_ticks=96,
    )

    assert departures[0] >= 0
    assert departures[-1] + int(durations[-1]) <= 96
    assert departures[0] + int(durations[0]) < departures[1]
