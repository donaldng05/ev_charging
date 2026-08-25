"""M5 multi-policy evaluation, aggregation, and reproducibility artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import pandas as pd

from chargeopt.config import AppConfig, MetricName, PolicyName, resolve_data_path
from chargeopt.models.io import load_demand_forecast, lookup_predicted_congestion
from chargeopt.optimization import build_station_chooser
from chargeopt.simulation.engine import run_simulation
from chargeopt.utils.experiment import config_hash, experiment_id, git_sha
from chargeopt.utils.io import select_columns, write_csv
from chargeopt.utils.seed import set_seed

METRIC_COLUMNS: tuple[str, ...] = tuple(metric.value for metric in MetricName)
RAW_RESULT_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "config_hash",
    "git_sha",
    "policy",
    "seed",
    *METRIC_COLUMNS,
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "policy",
    "metric",
    "n",
    "mean",
    "std",
    "worst",
    "ci_low",
    "ci_high",
)
MAXIMIZE_FOR_WORST: frozenset[str] = frozenset(
    {
        MetricName.ENERGY_COST.value,
        MetricName.AVG_WAIT_MINUTES.value,
        MetricName.SOC_VIOLATIONS.value,
        MetricName.ENERGY_USAGE_KWH.value,
        MetricName.VEHICLE_IDLE_MINUTES.value,
    }
)
SCHEMA_VERSION = "m5-evaluation-v1"


@dataclass(frozen=True)
class EvaluationReport:
    """Raw runs, summary statistics, and metadata for one evaluation command."""

    raw_results: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, object]


def build_forecast_by_tick(config: AppConfig) -> dict[int, float]:
    """Load the configured demand model forecast for every simulation tick."""
    path = resolve_data_path(config.models.demand.predictions_path)
    forecast = load_demand_forecast(path)
    model = config.models.demand.decision_model
    start = datetime.combine(
        config.simulation.start_day,
        time.min,
        tzinfo=ZoneInfo(config.data.timezone),
    )
    values: dict[int, float] = {}
    for tick in range(config.simulation.steps_per_day):
        timestamp = pd.Timestamp(
            start + timedelta(minutes=tick * config.simulation.timestep_minutes)
        )
        values[tick] = lookup_predicted_congestion(forecast, timestamp, model=model)
    return values


def run_evaluation(
    config: AppConfig,
    *,
    sessions: pd.DataFrame,
    policies: Sequence[PolicyName] | None = None,
    seeds: Sequence[int] | None = None,
    commit_sha: str | None = None,
) -> EvaluationReport:
    """Run the selected deterministic policy/seed matrix in memory."""
    selected_policies = _select_policies(config, policies)
    selected_seeds = _select_seeds(config, seeds)
    resolved_sha = commit_sha or git_sha() or "unknown"
    resolved_config_hash = config_hash(config)
    forecast_by_tick = (
        build_forecast_by_tick(config) if PolicyName.ML_INFORMED in selected_policies else None
    )

    rows: list[dict[str, object]] = []
    for policy in selected_policies:
        for seed in selected_seeds:
            set_seed(seed)
            chooser = build_station_chooser(
                policy,
                scoring=config.optimization,
                forecast_by_tick=forecast_by_tick,
            )
            result = run_simulation(config, sessions=sessions, seed=seed, chooser=chooser)
            metrics = result.metrics.model_dump()
            rows.append(
                {
                    "experiment_id": experiment_id(
                        config,
                        seed=seed,
                        policy=policy,
                        commit_sha=resolved_sha,
                    ),
                    "config_hash": resolved_config_hash,
                    "git_sha": resolved_sha,
                    "policy": policy.value,
                    "seed": seed,
                    **{metric: metrics[metric] for metric in METRIC_COLUMNS},
                }
            )

    raw_results = pd.DataFrame(rows, columns=list(RAW_RESULT_COLUMNS))
    summary = summarize_results(
        raw_results,
        confidence_level=config.evaluation.confidence_level,
    )
    metadata: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": resolved_config_hash,
        "git_sha": resolved_sha,
        "config": config.model_dump(mode="json"),
        "policies": [policy.value for policy in selected_policies],
        "seeds": list(selected_seeds),
        "metric_worst_direction": {
            metric: ("max" if metric in MAXIMIZE_FOR_WORST else "min") for metric in METRIC_COLUMNS
        },
        "confidence_level": config.evaluation.confidence_level,
        "n_runs": len(raw_results),
    }
    return EvaluationReport(raw_results=raw_results, summary=summary, metadata=metadata)


def summarize_results(raw_results: pd.DataFrame, *, confidence_level: float) -> pd.DataFrame:
    """Aggregate raw policy/seed metrics with deterministic confidence intervals."""
    _validate_raw_results(raw_results)
    if not 0 < confidence_level < 1 or not math.isfinite(confidence_level):
        msg = "confidence_level must be finite and in (0, 1)"
        raise ValueError(msg)
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rows: list[dict[str, object]] = []
    for policy in raw_results["policy"].drop_duplicates().tolist():
        policy_frame = raw_results.loc[raw_results["policy"] == policy]
        for metric in METRIC_COLUMNS:
            values = policy_frame[metric].astype(float)
            n = len(values)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            margin = z_value * std / math.sqrt(n) if n else 0.0
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "worst": float(values.max() if metric in MAXIMIZE_FOR_WORST else values.min()),
                    "ci_low": mean - margin,
                    "ci_high": mean + margin,
                }
            )
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def write_evaluation_artifacts(report: EvaluationReport, config: AppConfig) -> dict[str, str]:
    """Persist validated raw results, summaries, and metadata to configured paths."""
    _validate_raw_results(report.raw_results)
    _validate_summary(report.summary)
    raw_path = resolve_data_path(config.evaluation.raw_results_path)
    summary_path = resolve_data_path(config.evaluation.summary_path)
    metadata_path = resolve_data_path(config.evaluation.metadata_path)
    write_csv(
        report.raw_results,
        raw_path,
        columns=RAW_RESULT_COLUMNS,
        label="evaluation results",
    )
    write_csv(
        report.summary,
        summary_path,
        columns=SUMMARY_COLUMNS,
        label="evaluation summary",
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(report.metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "raw_results": str(raw_path),
        "summary": str(summary_path),
        "metadata": str(metadata_path),
    }


def read_evaluation_results(path: Path) -> pd.DataFrame:
    """Read and validate a raw M5 evaluation artifact."""
    frame = _read_evaluation_csv(path, RAW_RESULT_COLUMNS, "evaluation results")
    _validate_raw_results(frame)
    return frame


def read_evaluation_summary(path: Path) -> pd.DataFrame:
    """Read and validate an aggregated M5 evaluation artifact."""
    frame = _read_evaluation_csv(path, SUMMARY_COLUMNS, "evaluation summary")
    _validate_summary(frame)
    return frame


def _select_policies(
    config: AppConfig, policies: Sequence[PolicyName] | None
) -> tuple[PolicyName, ...]:
    selected = tuple(config.experiment.policies if policies is None else policies)
    configured = set(config.experiment.policies)
    unknown = [policy.value for policy in selected if policy not in configured]
    if unknown:
        msg = f"policies are not configured: {unknown}"
        raise ValueError(msg)
    if not selected:
        raise ValueError("at least one policy must be selected")
    return selected


def _select_seeds(config: AppConfig, seeds: Sequence[int] | None) -> tuple[int, ...]:
    selected = tuple(config.experiment.seeds if seeds is None else seeds)
    configured = set(config.experiment.seeds)
    unknown = [seed for seed in selected if seed not in configured]
    if unknown:
        msg = f"seeds are not configured: {unknown}"
        raise ValueError(msg)
    if not selected:
        raise ValueError("at least one seed must be selected")
    return selected


def _read_evaluation_csv(path: Path, columns: Sequence[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        msg = f"{label} not found: {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    _validate_exact_columns(frame, columns, label)
    return select_columns(frame, columns, label=label)


def _validate_raw_results(frame: pd.DataFrame) -> None:
    _validate_exact_columns(frame, RAW_RESULT_COLUMNS, "evaluation results")
    if frame.empty:
        raise ValueError("evaluation results must contain at least one row")
    allowed_policies = {policy.value for policy in PolicyName}
    if not set(frame["policy"].astype(str)).issubset(allowed_policies):
        raise ValueError("evaluation results contain an unknown policy")
    if frame["experiment_id"].isna().any() or frame["config_hash"].isna().any():
        raise ValueError("evaluation results contain missing reproducibility metadata")
    _require_finite(frame, ("seed", *METRIC_COLUMNS), "evaluation results")
    seeds = frame["seed"].astype(float)
    if not (seeds == seeds.round()).all():
        raise ValueError("evaluation result seeds must be integers")
    if frame.duplicated(subset=["policy", "seed"]).any():
        raise ValueError("evaluation results contain duplicate policy/seed rows")


def _validate_summary(frame: pd.DataFrame) -> None:
    _validate_exact_columns(frame, SUMMARY_COLUMNS, "evaluation summary")
    if frame.empty:
        raise ValueError("evaluation summary must contain at least one row")
    allowed_policies = {policy.value for policy in PolicyName}
    allowed_metrics = set(METRIC_COLUMNS)
    if not set(frame["policy"].astype(str)).issubset(allowed_policies):
        raise ValueError("evaluation summary contains an unknown policy")
    if not set(frame["metric"].astype(str)).issubset(allowed_metrics):
        raise ValueError("evaluation summary contains an unknown metric")
    _require_finite(
        frame,
        ("n", "mean", "std", "worst", "ci_low", "ci_high"),
        "evaluation summary",
    )
    counts = frame["n"].astype(float)
    if not ((counts >= 1) & (counts == counts.round())).all():
        raise ValueError("evaluation summary counts must be positive integers")
    if (frame["std"].astype(float) < 0).any():
        raise ValueError("evaluation summary standard deviations must be non-negative")
    if (frame["ci_low"].astype(float) > frame["ci_high"].astype(float)).any():
        raise ValueError("evaluation summary confidence intervals are inverted")
    if frame.duplicated(subset=["policy", "metric"]).any():
        raise ValueError("evaluation summary contains duplicate policy/metric rows")


def _validate_exact_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    expected = set(columns)
    actual = set(frame.columns)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    if extra:
        raise ValueError(f"{label} has unexpected columns: {extra}")


def _require_finite(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"{label} column {column!r} must contain finite numbers")
