"""Policy-neutral multi-seed calibration report for the M3 simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import pandas as pd

from chargeopt.config import AppConfig
from chargeopt.optimization.policy import ConcentratedStationChooser, HomeStationChooser
from chargeopt.simulation.engine import SimResult, run_simulation

HOME_ROUTING = "home"
CONCENTRATED_ROUTING = "concentrated"

MIN_UTILIZATION = 0.05
MAX_UTILIZATION = 0.20
MIN_QUEUE_SEEDS = 3
MAX_MEAN_WAIT_MINUTES = 10.0
MIN_PROBE_WAIT_DELTA = 15.0
MIN_PROBE_PEAK_QUEUE = 3


class MetricsRow(TypedDict):
    seed: int
    routing: str
    peak_queue: int
    energy_cost: float
    avg_wait_minutes: float
    soc_violations: int
    energy_usage_kwh: float
    station_utilization: float
    vehicle_idle_minutes: float


class CalibrationChecks(TypedDict):
    zero_soc_violations: bool
    utilization_in_band: bool
    queue_exposure: bool
    mean_wait_cap: bool
    probe_wait: bool
    probe_peak_queue: bool


class CalibrationGate(TypedDict):
    gate_passed: bool
    soc_violations: int
    median_utilization: float
    seeds_with_queue: int
    mean_wait_minutes: float
    probe_wait_delta: float
    probe_peak_queue: int
    checks: CalibrationChecks


@dataclass(frozen=True)
class CalibrationReport:
    metrics: pd.DataFrame
    home_result: SimResult
    probe_row: MetricsRow
    gate: CalibrationGate


def metrics_row(result: SimResult, routing: str) -> MetricsRow:
    """Attach routing label and observed peak queue to one run's metrics."""
    ticks = result.station_ticks
    peak = int(ticks["queue_len"].max()) if not ticks.empty and "queue_len" in ticks.columns else 0
    m = result.metrics
    return {
        "seed": m.seed,
        "routing": routing,
        "peak_queue": peak,
        "energy_cost": m.energy_cost,
        "avg_wait_minutes": m.avg_wait_minutes,
        "soc_violations": m.soc_violations,
        "energy_usage_kwh": m.energy_usage_kwh,
        "station_utilization": m.station_utilization,
        "vehicle_idle_minutes": m.vehicle_idle_minutes,
    }


def evaluate_calibration_gate(home_frame: pd.DataFrame, probe_row: MetricsRow) -> CalibrationGate:
    """Score the frozen EXPERIMENTS.md normal-scenario calibration gate."""
    soc_v = int(home_frame["soc_violations"].sum())
    med_u = float(home_frame["station_utilization"].median())
    q_seeds = int(((home_frame["peak_queue"] > 0) | (home_frame["avg_wait_minutes"] > 0)).sum())
    mean_wait = float(home_frame["avg_wait_minutes"].mean())
    probe_wait_delta = probe_row["avg_wait_minutes"] - mean_wait
    checks: CalibrationChecks = {
        "zero_soc_violations": soc_v == 0,
        "utilization_in_band": MIN_UTILIZATION <= med_u <= MAX_UTILIZATION,
        "queue_exposure": q_seeds >= MIN_QUEUE_SEEDS,
        "mean_wait_cap": mean_wait <= MAX_MEAN_WAIT_MINUTES,
        "probe_wait": probe_wait_delta >= MIN_PROBE_WAIT_DELTA,
        "probe_peak_queue": probe_row["peak_queue"] >= MIN_PROBE_PEAK_QUEUE,
    }
    return {
        "gate_passed": all(checks.values()),
        "soc_violations": soc_v,
        "median_utilization": med_u,
        "seeds_with_queue": q_seeds,
        "mean_wait_minutes": mean_wait,
        "probe_wait_delta": probe_wait_delta,
        "probe_peak_queue": probe_row["peak_queue"],
        "checks": checks,
    }


def run_calibration(config: AppConfig, sessions: pd.DataFrame) -> CalibrationReport:
    """Run home-station seeds plus one concentrated-routing probe."""
    if not config.experiment.seeds:
        raise ValueError("experiment.seeds must contain at least one seed")
    home_chooser = HomeStationChooser()
    home_result: SimResult | None = None
    rows: list[MetricsRow] = []
    for s in config.experiment.seeds:
        res = run_simulation(config, sessions=sessions, seed=s, chooser=home_chooser)
        if home_result is None:
            home_result = res
        rows.append(metrics_row(res, HOME_ROUTING))

    probe = run_simulation(
        config,
        sessions=sessions,
        seed=config.experiment.seeds[0],
        chooser=ConcentratedStationChooser(),
    )
    probe_row = metrics_row(probe, CONCENTRATED_ROUTING)
    rows.append(probe_row)
    metrics = pd.DataFrame(rows)
    assert home_result is not None
    return CalibrationReport(
        metrics=metrics,
        home_result=home_result,
        probe_row=probe_row,
        gate=evaluate_calibration_gate(metrics.loc[metrics["routing"] == HOME_ROUTING], probe_row),
    )
