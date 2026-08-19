"""Seeded synthetic station, vehicle, and itinerary generation."""

from itertools import pairwise
from pathlib import Path

from chargeopt.config import load_config
from chargeopt.data.io import read_sessions_csv
from chargeopt.simulation.calibration import calibrate_from_sessions
from chargeopt.simulation.world import build_fleet, build_stations, build_trips

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
