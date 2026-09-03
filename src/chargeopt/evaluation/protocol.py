"""M5 multi-policy evaluation, aggregation, and reproducibility artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from statistics import NormalDist
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from chargeopt.config import AppConfig, MetricName, PolicyName, resolve_data_path
from chargeopt.models.io import load_demand_forecast, lookup_predicted_congestion
from chargeopt.optimization import build_station_chooser
from chargeopt.simulation.engine import run_simulation
from chargeopt.utils.experiment import config_hash, experiment_id, git_sha
from chargeopt.utils.io import select_columns, write_csv
from chargeopt.utils.seed import set_seed

METRIC_COLUMNS: tuple[str, ...] = tuple(m.value for m in MetricName)


class ScenarioName(StrEnum):
    NORMAL = "normal"
    STRESS = "stress"


RAW_RESULT_COLUMNS = (
    "experiment_id",
    "config_hash",
    "git_sha",
    "scenario",
    "policy",
    "seed",
    *METRIC_COLUMNS,
)
SUMMARY_COLUMNS = ("scenario", "policy", "metric", "n", "mean", "std", "worst", "ci_low", "ci_high")
ROBUSTNESS_COLUMNS = (
    "policy",
    "metric",
    "n",
    "normal_mean",
    "stress_mean",
    "stress_minus_normal",
    "delta_std",
    "delta_ci_low",
    "delta_ci_high",
    "robustness_ratio",
)
MAXIMIZE_FOR_WORST = frozenset(
    {
        MetricName.ENERGY_COST.value,
        MetricName.AVG_WAIT_MINUTES.value,
        MetricName.SOC_VIOLATIONS.value,
        MetricName.ENERGY_USAGE_KWH.value,
        MetricName.VEHICLE_IDLE_MINUTES.value,
    }
)
SCHEMA_VERSION = "m6-evaluation-v1"


@dataclass(frozen=True)
class EvaluationReport:
    raw_results: pd.DataFrame
    summary: pd.DataFrame
    robustness: pd.DataFrame
    metadata: dict[str, object]


def build_forecast_by_tick(config: AppConfig) -> dict[int, float]:
    forecast = load_demand_forecast(resolve_data_path(config.models.demand.predictions_path))
    model = config.models.demand.decision_model
    start = datetime.combine(
        config.simulation.start_day, time.min, tzinfo=ZoneInfo(config.data.timezone)
    )
    step = config.simulation.timestep_minutes
    return {
        tick: lookup_predicted_congestion(
            forecast, pd.Timestamp(start + timedelta(minutes=tick * step)), model=model
        )
        for tick in range(config.simulation.steps_per_day)
    }


def run_evaluation(
    config: AppConfig,
    *,
    sessions: pd.DataFrame,
    policies: Sequence[PolicyName] | None = None,
    seeds: Sequence[int] | None = None,
    commit_sha: str | None = None,
    scenarios: Sequence[ScenarioName] = (ScenarioName.NORMAL,),
) -> EvaluationReport:
    sel_policies = _select_policies(config, policies)
    sel_seeds = _select_seeds(config, seeds)
    sel_scenarios = _select_scenarios(scenarios)
    res_sha, res_hash = commit_sha or git_sha() or "unknown", config_hash(config)
    f_by_tick = build_forecast_by_tick(config) if PolicyName.ML_INFORMED in sel_policies else None

    rows: list[dict[str, object]] = []
    for sc in sel_scenarios:
        temp_c, trip_mult, avail = _scenario_overrides(config, sc)
        for pol in sel_policies:
            for s in sel_seeds:
                set_seed(s)
                ch = build_station_chooser(
                    pol, scoring=config.optimization, forecast_by_tick=f_by_tick
                )
                res = run_simulation(
                    config,
                    sessions=sessions,
                    seed=s,
                    chooser=ch,
                    temperature_c=temp_c,
                    trip_rate_multiplier=trip_mult,
                    station_availability=avail,
                )
                m = res.metrics.model_dump()
                rows.append(
                    {
                        "experiment_id": experiment_id(
                            config, seed=s, policy=pol, commit_sha=res_sha, scenario=sc.value
                        ),
                        "config_hash": res_hash,
                        "git_sha": res_sha,
                        "scenario": sc.value,
                        "policy": pol.value,
                        "seed": s,
                        **{col: m[col] for col in METRIC_COLUMNS},
                    }
                )

    raw = pd.DataFrame(rows, columns=list(RAW_RESULT_COLUMNS))
    conf = config.evaluation.confidence_level
    summary = summarize_results(raw, confidence_level=conf)
    robustness = (
        build_robustness(raw, confidence_level=conf)
        if ScenarioName.STRESS in sel_scenarios
        else pd.DataFrame(columns=list(ROBUSTNESS_COLUMNS))
    )

    meta: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": res_hash,
        "git_sha": res_sha,
        "config": config.model_dump(mode="json"),
        "scenarios": [s.value for s in sel_scenarios],
        "policies": [p.value for p in sel_policies],
        "seeds": list(sel_seeds),
        "metric_worst_direction": {
            col: ("max" if col in MAXIMIZE_FOR_WORST else "min") for col in METRIC_COLUMNS
        },
        "confidence_level": conf,
        "n_runs": len(raw),
    }
    if ScenarioName.STRESS in sel_scenarios:
        meta["stress"] = {
            "demand_multiplier": config.stress.demand_multiplier,
            "temperature_c": config.stress.temperature_c,
            "station_availability": config.stress.station_availability,
            "availability_rule": "ceil(n_stations * availability) active stations",
            "ratio_rule": "stress_mean / normal_mean; NA when normal_mean is zero",
        }
    return EvaluationReport(raw_results=raw, summary=summary, robustness=robustness, metadata=meta)


def summarize_results(raw_results: pd.DataFrame, *, confidence_level: float) -> pd.DataFrame:
    _validate_raw_results(raw_results)
    if not 0 < confidence_level < 1 or not math.isfinite(confidence_level):
        raise ValueError("confidence_level must be finite and in (0, 1)")
    z_val = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rows: list[dict[str, Any]] = []
    for (scenario, policy), group in raw_results.groupby(["scenario", "policy"], sort=False):
        for m in METRIC_COLUMNS:
            vals = group[m].astype(float)
            n, mean = len(vals), float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            margin = z_val * std / math.sqrt(n) if n else 0.0
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "metric": m,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "worst": float(vals.max() if m in MAXIMIZE_FOR_WORST else vals.min()),
                    "ci_low": mean - margin,
                    "ci_high": mean + margin,
                }
            )
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def build_robustness(raw_results: pd.DataFrame, *, confidence_level: float) -> pd.DataFrame:
    _validate_raw_results(raw_results)
    if not 0 < confidence_level < 1 or not math.isfinite(confidence_level):
        raise ValueError("confidence_level must be finite and in (0, 1)")
    if set(raw_results["scenario"]) != {ScenarioName.NORMAL.value, ScenarioName.STRESS.value}:
        raise ValueError("robustness requires both normal and stress scenarios")
    z_val = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rows: list[dict[str, Any]] = []
    for policy, pgroup in raw_results.groupby("policy", sort=False):
        norm = pgroup.loc[pgroup["scenario"] == "normal"].set_index("seed")
        stress = pgroup.loc[pgroup["scenario"] == "stress"].set_index("seed")
        paired = norm.index.intersection(stress.index)
        for m in METRIC_COLUMNS:
            n_vals, s_vals = norm.loc[paired, m].astype(float), stress.loc[paired, m].astype(float)
            deltas = s_vals - n_vals
            n, d_mean = len(deltas), float(deltas.mean())
            d_std = float(deltas.std(ddof=1)) if n > 1 else 0.0
            margin = z_val * d_std / math.sqrt(n) if n else 0.0
            n_mean, s_mean = float(n_vals.mean()), float(s_vals.mean())
            ratio = None if math.isclose(n_mean, 0.0, abs_tol=1e-12) else s_mean / n_mean
            rows.append(
                {
                    "policy": policy,
                    "metric": m,
                    "n": n,
                    "normal_mean": n_mean,
                    "stress_mean": s_mean,
                    "stress_minus_normal": d_mean,
                    "delta_std": d_std,
                    "delta_ci_low": d_mean - margin,
                    "delta_ci_high": d_mean + margin,
                    "robustness_ratio": ratio,
                }
            )
    return pd.DataFrame(rows, columns=list(ROBUSTNESS_COLUMNS))


def write_evaluation_artifacts(report: EvaluationReport, config: AppConfig) -> dict[str, str]:
    _validate_raw_results(report.raw_results)
    _validate_summary(report.summary)
    if not report.robustness.empty:
        _validate_robustness(report.robustness)
    cfg_eval = config.evaluation
    paths = {
        k: resolve_data_path(getattr(cfg_eval, f"{k}_path"))
        for k in ("raw_results", "summary", "metadata")
    }
    write_csv(
        report.raw_results,
        paths["raw_results"],
        columns=RAW_RESULT_COLUMNS,
        label="evaluation results",
    )
    write_csv(report.summary, paths["summary"], columns=SUMMARY_COLUMNS, label="evaluation summary")
    res_paths = {k: str(v) for k, v in paths.items()}
    if not report.robustness.empty:
        rob_path = resolve_data_path(cfg_eval.robustness_path)
        write_csv(
            report.robustness, rob_path, columns=ROBUSTNESS_COLUMNS, label="evaluation robustness"
        )
        res_paths["robustness"] = str(rob_path)
    report.metadata["artifact_paths"] = res_paths
    paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    paths["metadata"].write_text(
        json.dumps(report.metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return res_paths


def _read_eval(path: Path, cols: Sequence[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    _check_cols(frame, cols, label)
    return select_columns(frame, cols, label=label)


def read_evaluation_results(path: Path) -> pd.DataFrame:
    f = _read_eval(path, RAW_RESULT_COLUMNS, "evaluation results")
    _validate_raw_results(f)
    return f


def read_evaluation_summary(path: Path) -> pd.DataFrame:
    f = _read_eval(path, SUMMARY_COLUMNS, "evaluation summary")
    _validate_summary(f)
    return f


def read_evaluation_robustness(path: Path) -> pd.DataFrame:
    f = _read_eval(path, ROBUSTNESS_COLUMNS, "evaluation robustness")
    _validate_robustness(f)
    return f


def _select_policies(
    config: AppConfig, policies: Sequence[PolicyName] | None
) -> tuple[PolicyName, ...]:
    sel = tuple(config.experiment.policies if policies is None else policies)
    unknown = [p.value for p in sel if p not in set(config.experiment.policies)]
    if unknown:
        raise ValueError(f"policies are not configured: {unknown}")
    if not sel:
        raise ValueError("at least one policy must be selected")
    return sel


def _select_seeds(config: AppConfig, seeds: Sequence[int] | None) -> tuple[int, ...]:
    sel = tuple(config.experiment.seeds if seeds is None else seeds)
    unknown = [s for s in sel if s not in set(config.experiment.seeds)]
    if unknown:
        raise ValueError(f"seeds are not configured: {unknown}")
    if not sel:
        raise ValueError("at least one seed must be selected")
    return sel


def _select_scenarios(scenarios: Sequence[ScenarioName]) -> tuple[ScenarioName, ...]:
    sel = tuple(scenarios)
    if not sel:
        raise ValueError("at least one scenario must be selected")
    if len(set(sel)) != len(sel):
        raise ValueError("scenarios must be unique")
    return sel


def _scenario_overrides(
    config: AppConfig, scenario: ScenarioName
) -> tuple[float | None, float | None, float | None]:
    if scenario is ScenarioName.NORMAL:
        return None, None, None
    if scenario is ScenarioName.STRESS:
        return (
            config.stress.temperature_c,
            config.simulation.trip_rate_multiplier * config.stress.demand_multiplier,
            config.stress.station_availability,
        )
    raise ValueError(f"unsupported scenario {scenario!r}")


def _check_cols(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    exp, act = set(columns), set(frame.columns)
    missing, extra = sorted(exp - act), sorted(act - exp)
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    if extra:
        raise ValueError(f"{label} has unexpected columns: {extra}")


def _check_finite(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for c in columns:
        vals = pd.to_numeric(frame[c], errors="coerce")
        if vals.isna().any() or not vals.map(math.isfinite).all():
            raise ValueError(f"{label} column {c!r} must contain finite numbers")


def _validate_base(frame: pd.DataFrame, cols: Sequence[str], label: str, dups: list[str]) -> None:
    _check_cols(frame, cols, label)
    if frame.empty:
        raise ValueError(f"{label} must contain at least one row")
    if not set(frame["policy"].astype(str)).issubset({p.value for p in PolicyName}):
        raise ValueError(f"{label} contains an unknown policy")
    if "metric" in frame.columns and not set(frame["metric"].astype(str)).issubset(
        set(METRIC_COLUMNS)
    ):
        raise ValueError(f"{label} contains an unknown metric")
    if "scenario" in frame.columns and not set(frame["scenario"].astype(str)).issubset(
        {s.value for s in ScenarioName}
    ):
        raise ValueError(f"{label} contains an unknown scenario")
    if "n" in frame.columns:
        counts = frame["n"].astype(float)
        if not ((counts >= 1) & (counts == counts.round())).all():
            raise ValueError(f"{label} counts must be positive integers")
    if frame.duplicated(subset=dups).any():
        raise ValueError(f"{label} contains duplicate {'/'.join(dups)} rows")


def _validate_raw_results(frame: pd.DataFrame) -> None:
    _validate_base(frame, RAW_RESULT_COLUMNS, "evaluation results", ["scenario", "policy", "seed"])
    if frame["experiment_id"].isna().any() or frame["config_hash"].isna().any():
        raise ValueError("evaluation results contain missing reproducibility metadata")
    _check_finite(frame, ("seed", *METRIC_COLUMNS), "evaluation results")
    seeds = frame["seed"].astype(float)
    if not (seeds == seeds.round()).all():
        raise ValueError("evaluation result seeds must be integers")


def _validate_summary(frame: pd.DataFrame) -> None:
    _validate_base(frame, SUMMARY_COLUMNS, "evaluation summary", ["scenario", "policy", "metric"])
    _check_finite(frame, ("n", "mean", "std", "worst", "ci_low", "ci_high"), "evaluation summary")
    if (frame["std"].astype(float) < 0).any():
        raise ValueError("evaluation summary standard deviations must be non-negative")
    if (frame["ci_low"].astype(float) > frame["ci_high"].astype(float)).any():
        raise ValueError("evaluation summary confidence intervals are inverted")


def _validate_robustness(frame: pd.DataFrame) -> None:
    _validate_base(frame, ROBUSTNESS_COLUMNS, "evaluation robustness", ["policy", "metric"])
    _check_finite(
        frame,
        (
            "n",
            "normal_mean",
            "stress_mean",
            "stress_minus_normal",
            "delta_std",
            "delta_ci_low",
            "delta_ci_high",
        ),
        "evaluation robustness",
    )
    if (frame["delta_std"].astype(float) < 0).any():
        raise ValueError("evaluation robustness delta standard deviations must be non-negative")
    ratio = pd.to_numeric(frame["robustness_ratio"], errors="coerce")
    normal_zero = frame["normal_mean"].astype(float).abs() <= 1e-12
    if ratio[~normal_zero].isna().any() or not ratio[~normal_zero].map(math.isfinite).all():
        raise ValueError("non-zero normal means require finite robustness ratios")
