"""Discrete-time driving, queue, charging, and SOC transitions."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from chargeopt.config import load_config
from chargeopt.data.io import read_sessions_csv
from chargeopt.simulation.engine import run_simulation, run_world
from chargeopt.simulation.schemas import FleetTrip, SimStation, VehicleState, VehicleStatus


def _simulation(*, fleet_size: int, horizon_hours: int = 1):
    return load_config().simulation.model_copy(
        update={
            "fleet_size": fleet_size,
            "n_stations": 1,
            "horizon_hours": horizon_hours,
            "charger_power_kw": 4.0,
            "battery_kwh": 10.0,
            "soc_initial": 0.8,
            "soc_min": 0.1,
            "soc_charge_target": 0.9,
            "start_day": date(2018, 9, 5),
        }
    )


def _station() -> SimStation:
    return SimStation(
        station_id="sim-00",
        x_km=1.0,
        y_km=2.0,
        n_chargers=1,
        power_kw=4.0,
        price_per_kwh=0.25,
    )


def _vehicles(count: int) -> list[VehicleState]:
    return [
        VehicleState(
            vehicle_id=f"vehicle-{index:03d}",
            battery_kwh=10.0,
            soc=0.8,
            status=VehicleStatus.IDLE,
            home_station_id="sim-00",
            x_km=1.0,
            y_km=2.0,
        )
        for index in range(count)
    ]


def _trip(vehicle_id: str, *, energy_kwh: float = 1.0, duration_ticks: int = 1) -> FleetTrip:
    return FleetTrip(
        vehicle_id=vehicle_id,
        trip_index=0,
        departure_tick=0,
        distance_km=5.0,
        duration_ticks=duration_ticks,
        energy_kwh=energy_kwh,
    )


def test_driving_consumes_energy_per_tick_and_soc_stays_bounded() -> None:
    result = run_world(
        _simulation(fleet_size=1),
        stations=[_station()],
        vehicles=_vehicles(1),
        trips=[_trip("vehicle-000", energy_kwh=2.0, duration_ticks=2)],
        seed=42,
        timezone_name="America/Los_Angeles",
    )

    ticks = result.vehicle_ticks.loc[result.vehicle_ticks["vehicle_id"] == "vehicle-000"]
    assert ticks.iloc[0]["status"] == "driving"
    assert ticks.iloc[0]["soc"] == pytest.approx(0.7)
    assert ticks.iloc[1]["soc"] == pytest.approx(0.7)
    assert ticks["soc"].between(0.0, 1.0).all()
    assert result.metrics.energy_usage_kwh == pytest.approx(2.0)


def test_fifo_queue_never_exceeds_station_capacity() -> None:
    vehicles = _vehicles(3)
    result = run_world(
        _simulation(fleet_size=3),
        stations=[_station()],
        vehicles=vehicles,
        trips=[_trip(vehicle.vehicle_id) for vehicle in vehicles],
        seed=42,
        timezone_name="America/Los_Angeles",
    )

    station_ticks = result.station_ticks
    assert (station_ticks["occupancy"] <= 1).all()
    assert station_ticks.iloc[0]["occupancy"] == 1
    assert station_ticks.iloc[0]["queue_len"] == 2
    at_tick_one = result.vehicle_ticks.loc[result.vehicle_ticks["tick"] == 1].set_index(
        "vehicle_id"
    )
    assert at_tick_one.loc["vehicle-001", "status"] == "charging"
    assert at_tick_one.loc["vehicle-002", "status"] == "queued"
    assert result.metrics.avg_wait_minutes == pytest.approx(30.0)


def test_insufficient_soc_skips_trip_and_records_violation() -> None:
    result = run_world(
        _simulation(fleet_size=1),
        stations=[_station()],
        vehicles=_vehicles(1),
        trips=[_trip("vehicle-000", energy_kwh=7.5)],
        seed=42,
        timezone_name="America/Los_Angeles",
    )

    vehicle = result.vehicle_ticks.loc[result.vehicle_ticks["vehicle_id"] == "vehicle-000"]
    assert (vehicle["status"] == "idle").all()
    assert (vehicle["soc"] == 0.8).all()
    assert result.metrics.soc_violations == 1


def test_seeded_default_run_has_30_vehicles_and_96_ticks() -> None:
    config = load_config()
    sessions = read_sessions_csv(Path("tests/fixtures/acn_sessions.csv"))

    first = run_simulation(config, sessions=sessions, seed=42)
    second = run_simulation(config, sessions=sessions, seed=42)

    assert len(first.vehicle_ticks) == 30 * 96
    assert len(first.station_ticks) == 10 * 96
    assert first.vehicle_ticks["vehicle_id"].nunique() == 30
    assert first.vehicle_ticks["tick"].nunique() == 96
    assert first.metrics == second.metrics
    charging_ticks = first.station_ticks.loc[first.station_ticks["energy_delivered_kwh"] > 0]
    assert (charging_ticks["occupancy"] > 0).all()
    expected_utilization = first.station_ticks["occupancy"].sum() / (
        first.stations["n_chargers"].sum() * 96
    )
    assert first.metrics.station_utilization == pytest.approx(expected_utilization)
    pd.testing.assert_frame_equal(first.vehicle_ticks, second.vehicle_ticks)
