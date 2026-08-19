"""Policy-neutral M3 calibration gate, evaluated from metric tables."""

from pathlib import Path

import pandas as pd
import pytest

from chargeopt.config import load_config
from chargeopt.data.io import read_sessions_csv
from chargeopt.simulation.engine import SimResult
from chargeopt.simulation.report import (
    CONCENTRATED_ROUTING,
    HOME_ROUTING,
    evaluate_calibration_gate,
    metrics_row,
    run_calibration,
)
from chargeopt.simulation.schemas import SimMetrics


def _metrics(**overrides: object) -> SimMetrics:
    payload: dict[str, object] = {
        "seed": 42,
        "energy_cost": 100.0,
        "avg_wait_minutes": 0.25,
        "soc_violations": 0,
        "energy_usage_kwh": 120.0,
        "station_utilization": 0.0625,
        "vehicle_idle_minutes": 15.0,
    }
    payload.update(overrides)
    return SimMetrics.model_validate(payload)


def _result(*, queue_lens: list[int], metrics: SimMetrics | None = None) -> SimResult:
    return SimResult(
        stations=pd.DataFrame(
            [{"station_id": "sim-00", "n_chargers": 1, "x_km": 0.0, "y_km": 0.0}]
        ),
        vehicle_ticks=pd.DataFrame(),
        station_ticks=pd.DataFrame({"queue_len": queue_lens}),
        metrics=metrics or _metrics(),
    )


def _home_row(
    seed: int,
    *,
    wait: float = 0.25,
    utilization: float = 0.0625,
    soc_violations: int = 0,
    peak_queue: int = 1,
) -> dict[str, object]:
    return {
        "seed": seed,
        "routing": HOME_ROUTING,
        "peak_queue": peak_queue,
        "energy_cost": 100.0,
        "avg_wait_minutes": wait,
        "soc_violations": soc_violations,
        "energy_usage_kwh": 120.0,
        "station_utilization": utilization,
        "vehicle_idle_minutes": 15.0,
    }


def _probe_row(*, wait: float = 58.5, peak_queue: int = 12) -> dict[str, object]:
    return {
        "seed": 42,
        "routing": CONCENTRATED_ROUTING,
        "peak_queue": peak_queue,
        "energy_cost": 100.0,
        "avg_wait_minutes": wait,
        "soc_violations": 0,
        "energy_usage_kwh": 120.0,
        "station_utilization": 0.5,
        "vehicle_idle_minutes": 3510.0,
    }


def test_metrics_row_adds_routing_and_peak_queue() -> None:
    row = metrics_row(_result(queue_lens=[0, 2, 1]), HOME_ROUTING)

    assert row["routing"] == HOME_ROUTING
    assert row["peak_queue"] == 2
    assert row["seed"] == 42
    assert row["avg_wait_minutes"] == pytest.approx(0.25)


def test_metrics_row_empty_station_ticks_has_zero_peak_queue() -> None:
    row = metrics_row(_result(queue_lens=[]), CONCENTRATED_ROUTING)

    assert row["routing"] == CONCENTRATED_ROUTING
    assert row["peak_queue"] == 0


def test_calibration_gate_passes_frozen_thresholds() -> None:
    home = pd.DataFrame(_home_row(seed) for seed in range(42, 52))
    gate = evaluate_calibration_gate(home, _probe_row())

    assert gate["gate_passed"] is True
    assert gate["soc_violations"] == 0
    assert gate["median_utilization"] == pytest.approx(0.0625)
    assert gate["seeds_with_queue"] == 10
    assert gate["mean_wait_minutes"] == pytest.approx(0.25)
    assert gate["probe_wait_delta"] == pytest.approx(58.25)
    assert gate["probe_peak_queue"] == 12


def test_calibration_gate_fails_when_any_check_breaks() -> None:
    home = pd.DataFrame(
        [
            _home_row(42, wait=12.0, utilization=0.01, soc_violations=1, peak_queue=0),
            _home_row(43, wait=12.0, utilization=0.01, peak_queue=0),
            _home_row(44, wait=12.0, utilization=0.01, peak_queue=0),
        ]
    )
    gate = evaluate_calibration_gate(home, _probe_row(wait=12.5, peak_queue=1))

    assert gate["gate_passed"] is False
    assert gate["checks"]["zero_soc_violations"] is False
    assert gate["checks"]["utilization_in_band"] is False
    assert gate["checks"]["mean_wait_cap"] is False
    assert gate["checks"]["probe_wait"] is False
    assert gate["checks"]["probe_peak_queue"] is False


def test_calibration_gate_requires_queue_exposure_across_seeds() -> None:
    home = pd.DataFrame(_home_row(seed, wait=0.0, peak_queue=0) for seed in range(42, 52))
    gate = evaluate_calibration_gate(home, _probe_row())

    assert gate["gate_passed"] is False
    assert gate["checks"]["queue_exposure"] is False


def test_run_calibration_emits_home_rows_and_one_concentrated_probe() -> None:
    config = load_config().model_copy(
        update={
            "experiment": load_config().experiment.model_copy(update={"seeds": [42, 43]}),
        }
    )
    sessions = read_sessions_csv(Path("tests/fixtures/acn_sessions.csv"))

    report = run_calibration(config, sessions)

    assert list(report.metrics["routing"]) == [
        HOME_ROUTING,
        HOME_ROUTING,
        CONCENTRATED_ROUTING,
    ]
    assert list(report.metrics["seed"]) == [42, 43, 42]
    assert report.home_result.metrics.seed == 42
    assert "gate_passed" in report.gate
