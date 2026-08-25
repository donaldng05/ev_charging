"""M3 station choosers: home routing and concentrated-routing probe."""

import pytest

from chargeopt.optimization import ConcentratedStationChooser, HomeStationChooser
from chargeopt.simulation.policy import (
    ConcentratedStationChooser as LegacyConcentratedStationChooser,
)
from chargeopt.simulation.policy import (
    HomeStationChooser as LegacyHomeStationChooser,
)
from chargeopt.simulation.schemas import SimStation, VehicleState, VehicleStatus


def _station(station_id: str, *, x_km: float = 0.0) -> SimStation:
    return SimStation(
        station_id=station_id,
        x_km=x_km,
        y_km=0.0,
        n_chargers=1,
        power_kw=150.0,
        price_per_kwh=0.2,
    )


def _vehicle(*, home_station_id: str) -> VehicleState:
    return VehicleState(
        vehicle_id="vehicle-000",
        battery_kwh=60.0,
        soc=0.7,
        status=VehicleStatus.IDLE,
        home_station_id=home_station_id,
        x_km=0.0,
        y_km=0.0,
    )


def test_home_chooser_returns_assigned_station() -> None:
    stations = [_station("sim-00"), _station("sim-01", x_km=1.0)]
    vehicle = _vehicle(home_station_id="sim-01")

    chosen = HomeStationChooser().choose(
        vehicle,
        stations,
        tick=0,
        occupancy={"sim-00": 1, "sim-01": 0},
        queues={"sim-00": ("vehicle-001",), "sim-01": ()},
    )

    assert chosen == "sim-01"


def test_simulation_policy_imports_remain_compatible() -> None:
    assert LegacyHomeStationChooser is HomeStationChooser
    assert LegacyConcentratedStationChooser is ConcentratedStationChooser


def test_home_chooser_rejects_unknown_home_station() -> None:
    stations = [_station("sim-00")]
    vehicle = _vehicle(home_station_id="missing")

    with pytest.raises(ValueError, match="unknown home station"):
        HomeStationChooser().choose(
            vehicle,
            stations,
            tick=0,
            occupancy={"sim-00": 0},
            queues={"sim-00": ()},
        )


def test_concentrated_chooser_always_picks_lexicographically_first_station() -> None:
    stations = [_station("sim-01", x_km=1.0), _station("sim-00")]
    vehicle = _vehicle(home_station_id="sim-01")

    chosen = ConcentratedStationChooser().choose(
        vehicle,
        stations,
        tick=12,
        occupancy={"sim-00": 1, "sim-01": 0},
        queues={"sim-00": ("vehicle-009",), "sim-01": ()},
    )

    assert chosen == "sim-00"
