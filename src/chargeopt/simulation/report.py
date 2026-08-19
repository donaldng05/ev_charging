"""Policy-neutral multi-seed calibration report for the M3 simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import pandas as pd

from chargeopt.config import AppConfig
from chargeopt.simulation.engine import SimResult, run_simulation
from chargeopt.simulation.policy import ConcentratedStationChooser, HomeStationChooser

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
    if result.station_ticks.empty or "queue_len" not in result.station_ticks.columns:
        peak_queue = 0
    else:
        peak_queue = int(result.station_ticks["queue_len"].max())
    return {
        "seed": result.metrics.seed,
        "routing": routing,
        "peak_queue": peak_queue,
        "energy_cost": result.metrics.energy_cost,
        "avg_wait_minutes": result.metrics.avg_wait_minutes,
        "soc_violations": result.metrics.soc_violations,
        "energy_usage_kwh": result.metrics.energy_usage_kwh,
        "station_utilization": result.metrics.station_utilization,
        "vehicle_idle_minutes": result.metrics.vehicle_idle_minutes,
    }


def evaluate_calibration_gate(
    home_frame: pd.DataFrame,
    probe_row: MetricsRow,
) -> CalibrationGate:
    """Score the frozen EXPERIMENTS.md normal-scenario calibration gate."""
    soc_violations = int(home_frame["soc_violations"].sum())
    median_utilization = float(home_frame["station_utilization"].median())
    queued = (home_frame["peak_queue"] > 0) | (home_frame["avg_wait_minutes"] > 0)
    seeds_with_queue = int(queued.sum())
    mean_wait_minutes = float(home_frame["avg_wait_minutes"].mean())
    probe_wait_delta = probe_row["avg_wait_minutes"] - mean_wait_minutes
    checks: CalibrationChecks = {
        "zero_soc_violations": soc_violations == 0,
        "utilization_in_band": MIN_UTILIZATION <= median_utilization <= MAX_UTILIZATION,
        "queue_exposure": seeds_with_queue >= MIN_QUEUE_SEEDS,
        "mean_wait_cap": mean_wait_minutes <= MAX_MEAN_WAIT_MINUTES,
        "probe_wait": probe_wait_delta >= MIN_PROBE_WAIT_DELTA,
        "probe_peak_queue": probe_row["peak_queue"] >= MIN_PROBE_PEAK_QUEUE,
    }
    return {
        "gate_passed": all(checks.values()),
        "soc_violations": soc_violations,
        "median_utilization": median_utilization,
        "seeds_with_queue": seeds_with_queue,
        "mean_wait_minutes": mean_wait_minutes,
        "probe_wait_delta": probe_wait_delta,
        "probe_peak_queue": probe_row["peak_queue"],
        "checks": checks,
    }


def run_calibration(config: AppConfig, sessions: pd.DataFrame) -> CalibrationReport:
    """Run home-station seeds plus one concentrated-routing probe."""
    home_chooser = HomeStationChooser()
    rows: list[MetricsRow] = []
    home_result: SimResult | None = None
    for seed in config.experiment.seeds:
        result = run_simulation(config, sessions=sessions, seed=seed, chooser=home_chooser)
        if home_result is None:
            home_result = result
        rows.append(metrics_row(result, HOME_ROUTING))
    if home_result is None:
        msg = "experiment.seeds must contain at least one seed"
        raise ValueError(msg)
    probe = run_simulation(
        config,
        sessions=sessions,
        seed=config.experiment.seeds[0],
        chooser=ConcentratedStationChooser(),
    )
    probe_row = metrics_row(probe, CONCENTRATED_ROUTING)
    rows.append(probe_row)
    metrics = pd.DataFrame(rows)
    home_frame = metrics.loc[metrics["routing"] == HOME_ROUTING]
    return CalibrationReport(
        metrics=metrics,
        home_result=home_result,
        probe_row=probe_row,
        gate=evaluate_calibration_gate(home_frame, probe_row),
    )
