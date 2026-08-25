"""Seeded synthetic station, vehicle, and itinerary generation."""

from itertools import pairwise
from pathlib import Path

import numpy as np

from chargeopt.config import load_config
from chargeopt.data.io import read_sessions_csv
from chargeopt.simulation.calibration import calibrate_from_sessions
from chargeopt.simulation.world import (
    _repair_departures,
    apply_station_availability,
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


def test_station_availability_disables_exact_fraction_deterministically() -> None:
    config = load_config()
    calibration = calibrate_from_sessions(
        read_sessions_csv(FIXTURE),
        timestep_minutes=config.simulation.timestep_minutes,
    )
    stations = build_stations(config.simulation, calibration, seed=42)

    first = apply_station_availability(stations, availability=0.8, seed=42)
    second = apply_station_availability(stations, availability=0.8, seed=42)

    assert sum(station.n_chargers > 0 for station in first) == 8
    assert sum(station.n_chargers == 0 for station in first) == 2
    assert [
        (station.station_id, station.x_km, station.y_km, station.n_chargers) for station in first
    ] == [
        (station.station_id, station.x_km, station.y_km, station.n_chargers) for station in second
    ]
    assert [(station.station_id, station.x_km, station.y_km) for station in first] == [
        (station.station_id, station.x_km, station.y_km) for station in stations
    ]


def test_stress_demand_increases_itineraries_and_cold_temperature_energy() -> None:
    config = load_config()
    calibration = calibrate_from_sessions(
        read_sessions_csv(FIXTURE),
        timestep_minutes=config.simulation.timestep_minutes,
    )
    stations = build_stations(config.simulation, calibration, seed=42)
    vehicles = build_fleet(config.simulation, stations)
    normal_trips = build_trips(
        config.simulation,
        config.models.energy,
        calibration,
        vehicles,
        seed=42,
        temperature_c=20.0,
        trip_rate_multiplier=1.0,
    )
    cold_trips = build_trips(
        config.simulation,
        config.models.energy,
        calibration,
        vehicles,
        seed=42,
        temperature_c=-10.0,
        trip_rate_multiplier=1.0,
    )
    stress_trips = build_trips(
        config.simulation,
        config.models.energy,
        calibration,
        vehicles,
        seed=42,
        temperature_c=-10.0,
        trip_rate_multiplier=1.5,
    )
    cold_normal_load = sum(trip.energy_kwh for trip in normal_trips)
    cold_stress_load = sum(trip.energy_kwh for trip in stress_trips)

    assert len(stress_trips) > len(normal_trips)
    assert sum(trip.energy_kwh for trip in cold_trips) > cold_normal_load
    assert cold_stress_load > cold_normal_load


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
