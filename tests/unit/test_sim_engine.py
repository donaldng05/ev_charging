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
    assert ticks["drove_this_tick"].tolist()[:2] == [True, True]
    assert ticks["charged_this_tick"].tolist()[:2] == [False, True]
    assert not ticks["queued_this_tick"].any()
    assert not ticks["stranded_this_tick"].any()
    assert ticks["soc"].between(0.0, 1.0).all()
    assert result.metrics.energy_usage_kwh == pytest.approx(2.0)
    assert result.metrics.vehicle_idle_minutes == 0


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
    assert result.metrics.vehicle_idle_minutes == pytest.approx(90.0)


def test_station_queue_snapshot_matches_vehicle_queue_activity() -> None:
    vehicles = _vehicles(3)
    result = run_world(
        _simulation(fleet_size=3),
        stations=[_station()],
        vehicles=vehicles,
        trips=[_trip(vehicle.vehicle_id) for vehicle in vehicles],
        seed=42,
        timezone_name="America/Los_Angeles",
    )

    queued_vehicle_count = result.vehicle_ticks.loc[
        result.vehicle_ticks["tick"] == 1,
        "queued_this_tick",
    ].sum()
    station_queue_len = result.station_ticks.loc[
        result.station_ticks["tick"] == 1,
        "queue_len",
    ].item()

    assert station_queue_len == queued_vehicle_count


def test_departure_release_promotes_fifo_queue_before_new_arrivals() -> None:
    vehicles = _vehicles(3)
    trips = [
        _trip("vehicle-000"),
        FleetTrip(
            vehicle_id="vehicle-000",
            trip_index=1,
            departure_tick=1,
            distance_km=5.0,
            duration_ticks=2,
            energy_kwh=1.0,
        ),
        _trip("vehicle-001"),
        _trip("vehicle-002", energy_kwh=1.0, duration_ticks=2),
    ]

    result = run_world(
        _simulation(fleet_size=3),
        stations=[_station()],
        vehicles=vehicles,
        trips=trips,
        seed=42,
        timezone_name="America/Los_Angeles",
    )

    at_tick_one = result.vehicle_ticks.loc[result.vehicle_ticks["tick"] == 1].set_index(
        "vehicle_id"
    )
    assert at_tick_one.loc["vehicle-001", "status"] == "charging"
    assert at_tick_one.loc["vehicle-002", "status"] == "queued"
    assert at_tick_one.loc["vehicle-001", "charged_this_tick"]
    assert at_tick_one.loc["vehicle-002", "queued_this_tick"]


def test_insufficient_soc_records_planned_trip_as_stranded_delay() -> None:
    result = run_world(
        _simulation(fleet_size=1),
        stations=[_station()],
        vehicles=_vehicles(1),
        trips=[_trip("vehicle-000", energy_kwh=7.5, duration_ticks=2)],
        seed=42,
        timezone_name="America/Los_Angeles",
    )

    vehicle = result.vehicle_ticks.loc[result.vehicle_ticks["vehicle_id"] == "vehicle-000"]
    assert vehicle["status"].tolist()[:3] == ["stranded", "stranded", "idle"]
    assert vehicle["stranded_this_tick"].tolist()[:3] == [True, True, False]
    assert (vehicle["soc"] == 0.8).all()
    assert result.metrics.soc_violations == 1
    assert result.metrics.vehicle_idle_minutes == 30.0


def test_same_tick_charging_is_not_counted_as_policy_delay() -> None:
    result = run_world(
        _simulation(fleet_size=1),
        stations=[_station().model_copy(update={"power_kw": 8.0})],
        vehicles=_vehicles(1),
        trips=[_trip("vehicle-000", energy_kwh=1.0)],
        seed=42,
        timezone_name="America/Los_Angeles",
    )

    first_tick = result.vehicle_ticks.iloc[0]
    assert first_tick["status"] == "idle"
    assert first_tick["drove_this_tick"]
    assert first_tick["charged_this_tick"]
    assert not first_tick["queued_this_tick"]
    assert not first_tick["stranded_this_tick"]
    assert result.metrics.vehicle_idle_minutes == 0


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
    queued_vehicle_ticks = int(first.vehicle_ticks["queued_this_tick"].sum())
    assert int(first.station_ticks["queue_len"].sum()) == queued_vehicle_ticks
    occupancy_limit = first.station_ticks.merge(
        first.stations[["station_id", "n_chargers"]],
        on="station_id",
        how="left",
    )
    assert (occupancy_limit["occupancy"] <= occupancy_limit["n_chargers"]).all()
    expected_utilization = first.station_ticks["occupancy"].sum() / (
        first.stations["n_chargers"].sum() * 96
    )
    assert first.metrics.station_utilization == pytest.approx(expected_utilization)
    queued_minutes = first.vehicle_ticks["queued_this_tick"].sum() * 15
    stranded_minutes = first.vehicle_ticks["stranded_this_tick"].sum() * 15
    assert first.metrics.vehicle_idle_minutes == queued_minutes + stranded_minutes
    charge_sessions = (
        first.vehicle_ticks["drove_this_tick"]
        & (first.vehicle_ticks["charged_this_tick"] | first.vehicle_ticks["queued_this_tick"])
    ).sum()
    assert first.metrics.avg_wait_minutes == pytest.approx(
        queued_minutes / charge_sessions if charge_sessions else 0.0
    )
    pd.testing.assert_frame_equal(first.vehicle_ticks, second.vehicle_ticks)
