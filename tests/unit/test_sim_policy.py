"""M3 station choosers: home routing and concentrated-routing probe."""

from pathlib import Path

import pytest

from chargeopt.config import PolicyName, PolicyScoringConfig, load_config
from chargeopt.data.io import read_sessions_csv
from chargeopt.optimization import (
    CheapestStationChooser,
    ConcentratedStationChooser,
    HomeStationChooser,
    MLInformedStationChooser,
    NearestStationChooser,
    build_station_chooser,
)
from chargeopt.simulation.engine import run_simulation
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


def test_nearest_chooser_prefers_free_closest_station_and_breaks_ties() -> None:
    stations = [_station("sim-01", x_km=1.0), _station("sim-00", x_km=-1.0)]
    vehicle = _vehicle(home_station_id="sim-00")

    chosen = NearestStationChooser().choose(
        vehicle,
        stations,
        tick=0,
        occupancy={"sim-00": 0, "sim-01": 0},
        queues={"sim-00": (), "sim-01": ()},
    )

    assert chosen == "sim-00"


def test_cheapest_chooser_uses_free_capacity_before_price() -> None:
    stations = [
        _station("sim-00").model_copy(update={"price_per_kwh": 0.20}),
        _station("sim-01", x_km=1.0).model_copy(update={"price_per_kwh": 0.45}),
    ]

    chosen = CheapestStationChooser().choose(
        _vehicle(home_station_id="sim-00"),
        stations,
        tick=0,
        occupancy={"sim-00": 1, "sim-01": 0},
        queues={"sim-00": (), "sim-01": ()},
    )

    assert chosen == "sim-01"


def test_policy_falls_back_to_all_stations_when_every_charger_is_full() -> None:
    stations = [
        _station("sim-01").model_copy(update={"price_per_kwh": 0.45}),
        _station("sim-00").model_copy(update={"price_per_kwh": 0.20}),
    ]

    chosen = CheapestStationChooser().choose(
        _vehicle(home_station_id="sim-01"),
        stations,
        tick=0,
        occupancy={"sim-00": 1, "sim-01": 1},
        queues={"sim-00": ("vehicle-001",), "sim-01": ()},
    )

    assert chosen == "sim-00"


def test_ml_forecast_changes_selection_by_scaling_queue_penalty() -> None:
    stations = [
        _station("sim-00", x_km=0.0),
        _station("sim-01", x_km=10.0),
    ]
    vehicle = _vehicle(home_station_id="sim-00")
    scoring = PolicyScoringConfig(
        distance_weight=1.0,
        price_weight=0.0,
        queue_weight=0.8,
        forecast_weight=1.0,
        forecast_scale_kwh=10.0,
    )

    low_forecast = MLInformedStationChooser(scoring=scoring, forecast_by_tick={0: 0.0})
    high_forecast = MLInformedStationChooser(scoring=scoring, forecast_by_tick={0: 10.0})
    state = {"sim-00": 0, "sim-01": 0}
    queues = {"sim-00": ("vehicle-001",), "sim-01": ()}

    assert (
        low_forecast.choose(vehicle, stations, tick=0, occupancy=state, queues=queues) == "sim-00"
    )
    assert (
        high_forecast.choose(vehicle, stations, tick=0, occupancy=state, queues=queues) == "sim-01"
    )


def test_ml_chooser_requires_a_forecast_for_the_current_tick() -> None:
    chooser = MLInformedStationChooser(
        scoring=PolicyScoringConfig(
            distance_weight=1.0,
            price_weight=1.0,
            queue_weight=1.0,
            forecast_weight=1.0,
            forecast_scale_kwh=10.0,
        ),
        forecast_by_tick={},
    )

    with pytest.raises(KeyError, match="tick 3"):
        chooser.choose(
            _vehicle(home_station_id="sim-00"),
            [_station("sim-00")],
            tick=3,
            occupancy={"sim-00": 0},
            queues={"sim-00": ()},
        )


def test_policy_factory_resolves_all_m4_policies() -> None:
    scoring = PolicyScoringConfig(
        distance_weight=1.0,
        price_weight=1.0,
        queue_weight=1.0,
        forecast_weight=1.0,
        forecast_scale_kwh=10.0,
    )

    assert isinstance(
        build_station_chooser(PolicyName.NEAREST, scoring=scoring),
        NearestStationChooser,
    )
    assert isinstance(
        build_station_chooser(PolicyName.CHEAPEST, scoring=scoring),
        CheapestStationChooser,
    )
    assert isinstance(
        build_station_chooser(
            PolicyName.ML_INFORMED,
            scoring=scoring,
            forecast_by_tick={0: 1.0},
        ),
        MLInformedStationChooser,
    )


def test_all_m4_policies_are_deterministic_for_a_seeded_world() -> None:
    config = load_config()
    sessions = read_sessions_csv(Path("tests/fixtures/acn_sessions.csv"))
    forecast = {tick: 1.0 for tick in range(config.simulation.steps_per_day)}

    for policy in config.experiment.policies:
        chooser = build_station_chooser(
            policy,
            scoring=config.optimization,
            forecast_by_tick=forecast if policy is PolicyName.ML_INFORMED else None,
        )
        first = run_simulation(config, sessions=sessions, seed=42, chooser=chooser)
        second = run_simulation(config, sessions=sessions, seed=42, chooser=chooser)
        assert first.metrics == second.metrics


def test_congestion_profile_makes_policy_queue_behavior_visible_on_fixture() -> None:
    config = load_config(Path("configs/congestion.yaml"))
    sessions = read_sessions_csv(Path("tests/fixtures/acn_sessions.csv"))
    forecast = {tick: 1.0 for tick in range(config.simulation.steps_per_day)}

    for policy in config.experiment.policies:
        chooser = build_station_chooser(
            policy,
            scoring=config.optimization,
            forecast_by_tick=forecast if policy is PolicyName.ML_INFORMED else None,
        )
        result = run_simulation(config, sessions=sessions, seed=42, chooser=chooser)

        assert result.metrics.station_utilization > 0.30
        assert result.metrics.avg_wait_minutes > 0.0
        assert result.metrics.soc_violations == 0
