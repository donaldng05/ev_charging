"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from chargeopt import __version__
from chargeopt.config import (
    LEARNER_NAMES,
    AppConfig,
    PolicyName,
    RuntimeSettings,
    load_config,
    resolve_data_path,
)
from chargeopt.data.acn import iter_acn_sessions, localize_naive, snapshot_sessions
from chargeopt.data.io import read_demand_csv, read_sessions_csv, write_demand_csv, write_trips_csv
from chargeopt.evaluation import (
    ScenarioName,
    build_forecast_by_tick,
    run_evaluation,
    write_evaluation_artifacts,
)
from chargeopt.features.demand import build_demand_table
from chargeopt.features.energy import generate_synthetic_trips
from chargeopt.models.demand import DEMAND_BASELINES, horizon_bins, train_and_predict_demand
from chargeopt.models.energy import PHYSICS, evaluate_energy_cold_holdout, train_and_predict_energy
from chargeopt.models.io import (
    write_demand_predictions,
    write_energy_predictions,
    write_error_slices,
    write_metrics,
)
from chargeopt.models.metrics import (
    best_learned,
    error_slices_from_predictions,
    learned_beats_baselines,
    test_mae_by_model,
)
from chargeopt.models.tune import (
    param_grid,
    resolve_learner_names,
    tune_demand_learners,
    tune_energy_learners,
)
from chargeopt.optimization import build_station_chooser
from chargeopt.simulation.engine import run_simulation
from chargeopt.simulation.io import write_simulation_artifacts
from chargeopt.simulation.report import run_calibration
from chargeopt.utils.io import write_csv
from chargeopt.utils.log import configure_logging
from chargeopt.utils.seed import set_seed

POLICY_ALIASES = {
    "nearest": PolicyName.NEAREST,
    "cheapest": PolicyName.CHEAPEST,
    "ml": PolicyName.ML_INFORMED,
    "ml_informed": PolicyName.ML_INFORMED,
}


def _parse_policy(value: str) -> PolicyName:
    key = value.lower()
    if key not in POLICY_ALIASES:
        allowed = ", ".join(sorted(POLICY_ALIASES))
        raise argparse.ArgumentTypeError(f"unknown policy {value!r}; expected one of: {allowed}")
    return POLICY_ALIASES[key]


def _parse_positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite number, got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive finite number, got {value!r}")
    return parsed


def _add_config_and_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: first seed in the config)."
    )


def _add_learner_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--learner",
        choices=list(LEARNER_NAMES),
        default=None,
        help="Tune one learner (default: all configured learners).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chargeopt", description="EV fleet charging intelligence experiments."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    exp = sub.add_parser("experiment", help="Run the configured policy and seed evaluation matrix.")
    exp.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML (default: configs/default.yaml).",
    )
    exp.add_argument(
        "--policy",
        type=_parse_policy,
        default=None,
        help="Optional policy filter: nearest, cheapest, ml / ml_informed.",
    )
    exp.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: first seed in the config)."
    )
    exp.add_argument(
        "--stress", action="store_true", help="Run paired normal and declared stress scenarios."
    )

    sim = sub.add_parser("simulate", help="Run one seeded 24-hour synthetic fleet simulation.")
    _add_config_and_seed(sim)
    sim.add_argument(
        "--policy",
        type=_parse_policy,
        default=None,
        help="M4 station policy: nearest, cheapest, ml / ml_informed (default: M3 home routing).",
    )
    sim.add_argument(
        "--trip-rate-multiplier",
        type=_parse_positive_finite_float,
        default=None,
        help="Override configured trip load multiplier for this simulation.",
    )
    sim.add_argument(
        "--all-seeds",
        action="store_true",
        help="Run every experiment seed under home routing plus the concentrated-routing probe.",
    )

    data_sub = sub.add_parser("data", help="ACN-Data ingest and demand features.").add_subparsers(
        dest="data_command"
    )
    data_sub.add_parser("pull", help="Snapshot Caltech sessions to CSV.").add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML (default: configs/default.yaml).",
    )
    data_sub.add_parser(
        "features", help="Build the 15-minute demand table from a session CSV snapshot."
    ).add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML (default: configs/default.yaml).",
    )

    models_sub = sub.add_parser("models", help="Train demand and energy models.").add_subparsers(
        dest="models_command"
    )
    demand = models_sub.add_parser(
        "demand", help="Fit demand baselines and sklearn learners on the 15-minute demand table."
    )
    _add_config_and_seed(demand)
    energy = models_sub.add_parser(
        "energy",
        help="Generate synthetic trips and fit physics plus sklearn residual energy models.",
    )
    _add_config_and_seed(energy)

    tune_sub = models_sub.add_parser(
        "tune", help="Search hyperparameters without touching the test split."
    ).add_subparsers(dest="tune_command")
    for t_name, t_help in [
        ("demand", "Walk-forward learner search on chronological train folds."),
        ("energy", "Validate-split learner search for trip energy."),
    ]:
        p = tune_sub.add_parser(t_name, help=t_help)
        _add_config_and_seed(p)
        _add_learner_flag(p)

    return parser


def _load_from_args(args: argparse.Namespace) -> tuple[RuntimeSettings, AppConfig]:
    runtime = RuntimeSettings()
    config_path = args.config
    if config_path is None and "config_path" in runtime.model_fields_set:
        config_path = runtime.config_path
    config = load_config(config_path)
    configure_logging(runtime.log_level or config.logging.level)
    return runtime, config


def _resolve_seed(args: argparse.Namespace, config: AppConfig) -> int:
    return int(args.seed) if args.seed is not None else config.experiment.seeds[0]


def _print_json(payload: object) -> None:
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def run_experiment(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    sessions = read_sessions_csv(resolve_data_path(config.data.snapshot_path))
    report = run_evaluation(
        config,
        sessions=sessions,
        policies=(args.policy,) if args.policy is not None else None,
        seeds=(args.seed,) if args.seed is not None else None,
        scenarios=(
            (ScenarioName.NORMAL, ScenarioName.STRESS) if args.stress else (ScenarioName.NORMAL,)
        ),
    )
    paths = write_evaluation_artifacts(report, config)
    _print_json(
        {
            "status": "ok",
            "n_runs": len(report.raw_results),
            "n_policies": len(report.raw_results["policy"].unique()),
            "n_seeds": len(report.raw_results["seed"].unique()),
            "n_scenarios": len(report.raw_results["scenario"].unique()),
            "scenarios": report.metadata["scenarios"],
            "policies": report.metadata["policies"],
            "seeds": report.metadata["seeds"],
            "config_hash": report.metadata["config_hash"],
            "git_sha": report.metadata["git_sha"],
            "paths": paths,
        }
    )
    return 0


def run_simulate(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    if args.trip_rate_multiplier is not None:
        simulation = config.simulation.model_copy(
            update={"trip_rate_multiplier": args.trip_rate_multiplier}
        )
        config = config.model_copy(update={"simulation": simulation})
    if args.all_seeds and args.policy is not None:
        raise ValueError(
            "--policy cannot be combined with --all-seeds; use one policy per simulation"
        )

    sessions = read_sessions_csv(resolve_data_path(config.data.snapshot_path))
    paths = {
        "stations": resolve_data_path(config.simulation.stations_path),
        "run": resolve_data_path(config.simulation.run_path),
        "station_ticks": resolve_data_path(config.simulation.station_ticks_path),
        "metrics": resolve_data_path(config.simulation.metrics_path),
    }

    if args.all_seeds:
        report = run_calibration(config, sessions)
        write_simulation_artifacts(
            report.home_result,
            stations_path=paths["stations"],
            run_path=paths["run"],
            station_ticks_path=paths["station_ticks"],
            metrics_path=paths["metrics"],
            metrics=report.metrics,
        )
        probe = report.probe_row
        _print_json(
            {
                "status": "ok",
                "gate_passed": report.gate["gate_passed"],
                "trip_rate_multiplier": config.simulation.trip_rate_multiplier,
                "n_ticks": config.simulation.steps_per_day,
                "n_vehicles": config.simulation.fleet_size,
                "home": {
                    "n_seeds": len(config.experiment.seeds),
                    "median_utilization": report.gate["median_utilization"],
                    "mean_wait_minutes": report.gate["mean_wait_minutes"],
                    "soc_violations": report.gate["soc_violations"],
                    "seeds_with_queue": report.gate["seeds_with_queue"],
                },
                "probe": {
                    "seed": probe["seed"],
                    "avg_wait_minutes": probe["avg_wait_minutes"],
                    "peak_queue": probe["peak_queue"],
                    "wait_delta": report.gate["probe_wait_delta"],
                },
                "paths": {k: str(v) for k, v in paths.items()},
            }
        )
        return 0

    seed = _resolve_seed(args, config)
    set_seed(seed)
    chooser = None
    if args.policy is not None:
        f_by_tick = build_forecast_by_tick(config) if args.policy.value == "ml_informed" else None
        chooser = build_station_chooser(
            args.policy, scoring=config.optimization, forecast_by_tick=f_by_tick
        )

    result = run_simulation(config, sessions=sessions, seed=seed, chooser=chooser)
    write_simulation_artifacts(
        result,
        stations_path=paths["stations"],
        run_path=paths["run"],
        station_ticks_path=paths["station_ticks"],
        metrics_path=paths["metrics"],
    )
    _print_json(
        {
            "status": "ok",
            "seed": seed,
            "policy": args.policy.value if args.policy is not None else "home",
            "trip_rate_multiplier": config.simulation.trip_rate_multiplier,
            "n_ticks": config.simulation.steps_per_day,
            "n_vehicles": config.simulation.fleet_size,
            "metrics": result.metrics.model_dump(),
            "paths": {k: str(v) for k, v in paths.items()},
        }
    )
    return 0


def run_data_pull(args: argparse.Namespace) -> int:
    runtime, config = _load_from_args(args)
    path = resolve_data_path(config.data.snapshot_path)
    frame = snapshot_sessions(
        iter_acn_sessions(
            site=config.data.site,
            start=localize_naive(config.data.start, config.data.timezone),
            end=localize_naive(config.data.end, config.data.timezone),
            token=runtime.acn_token,
        ),
        site=config.data.site,
        path=path,
    )
    _print_json(
        {
            "status": "ok",
            "site": config.data.site,
            "n_sessions": len(frame),
            "snapshot_path": str(path),
        }
    )
    return 0


def run_data_features(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    processed = resolve_data_path(config.data.processed_path)
    demand = build_demand_table(
        read_sessions_csv(resolve_data_path(config.data.snapshot_path)),
        timestep_minutes=config.simulation.timestep_minutes,
        timezone_name=config.data.timezone,
        train_fraction=config.data.train_fraction,
        val_fraction=config.data.val_fraction,
        covid_start=localize_naive(config.data.covid_start, config.data.timezone),
        covid_end=localize_naive(config.data.covid_end, config.data.timezone),
    )
    write_demand_csv(demand, processed)
    _print_json(
        {
            "status": "ok",
            "n_intervals": len(demand),
            "processed_path": str(processed),
            "splits": demand["split"].value_counts().to_dict(),
        }
    )
    return 0


def run_models_demand(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    seed = _resolve_seed(args, config)
    set_seed(seed)
    demand = read_demand_csv(resolve_data_path(config.data.processed_path))
    predictions, metrics = train_and_predict_demand(
        demand,
        timestep_minutes=config.simulation.timestep_minutes,
        horizon_minutes=config.models.demand.horizon_minutes,
        learners=config.models.demand.learners,
        seed=seed,
    )
    pred_path, met_path, sl_path = (
        resolve_data_path(config.models.demand.predictions_path),
        resolve_data_path(config.models.demand.metrics_path),
        resolve_data_path(config.models.demand.error_slices_path),
    )
    write_demand_predictions(predictions, pred_path)
    write_metrics(metrics, met_path)
    write_error_slices(error_slices_from_predictions(predictions, demand), sl_path)
    test_mae = test_mae_by_model(metrics)
    _print_json(
        {
            "status": "ok",
            "n_rows": len(predictions),
            "seed": seed,
            "test_mae": test_mae,
            "learned_beats_baselines": learned_beats_baselines(
                test_mae, learners=LEARNER_NAMES, baselines=DEMAND_BASELINES
            ),
            "best_learned": best_learned(test_mae, LEARNER_NAMES),
            "decision_model": config.models.demand.decision_model,
            "predictions_path": str(pred_path),
            "metrics_path": str(met_path),
            "error_slices_path": str(sl_path),
        }
    )
    return 0


def run_models_energy(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    seed = _resolve_seed(args, config)
    set_seed(seed)
    trips = generate_synthetic_trips(
        config.models.energy,
        seed=seed,
        train_fraction=config.data.train_fraction,
        val_fraction=config.data.val_fraction,
    )
    trips_path = resolve_data_path(config.models.energy.trips_path)
    write_trips_csv(trips, trips_path)
    predictions, metrics = train_and_predict_energy(trips, spec=config.models.energy, seed=seed)
    pred_path, met_path = (
        resolve_data_path(config.models.energy.predictions_path),
        resolve_data_path(config.models.energy.metrics_path),
    )
    write_energy_predictions(predictions, pred_path)
    write_metrics(metrics, met_path)
    cold_metrics = evaluate_energy_cold_holdout(
        trips.loc[trips["split"] == "train"],
        spec=config.models.energy,
        seed=seed,
        temperature_c=config.stress.temperature_c,
        n_trips=config.models.energy.cold_holdout_n_trips,
        train_fraction=config.data.train_fraction,
        val_fraction=config.data.val_fraction,
    )
    cold_path = resolve_data_path(config.models.energy.cold_metrics_path)
    write_metrics(cold_metrics, cold_path)
    test_mae = test_mae_by_model(metrics)
    _print_json(
        {
            "status": "ok",
            "n_rows": len(predictions),
            "n_trips": len(trips),
            "seed": seed,
            "test_mae": test_mae,
            "cold_mae": {str(r["model"]): float(r["mae"]) for _, r in cold_metrics.iterrows()},
            "learned_beats_baselines": learned_beats_baselines(
                test_mae, learners=LEARNER_NAMES, baselines=(PHYSICS,)
            ),
            "best_learned": best_learned(test_mae, LEARNER_NAMES),
            "trips_path": str(trips_path),
            "predictions_path": str(pred_path),
            "metrics_path": str(met_path),
            "cold_metrics_path": str(cold_path),
        }
    )
    return 0


def _handle_tune(
    names: tuple[str, ...],
    best: Mapping[str, Mapping[str, Any]],
    folds: pd.DataFrame,
    mae: Mapping[str, float],
    path: Path,
    seed: int,
    learners_cfg: Any,
    n_splits: int | None = None,
) -> None:
    write_csv(folds, path, label="tune metrics")
    payload: dict[str, Any] = {
        "status": "ok",
        "learners": list(names),
        "best_params": best,
        "val_mae": mae,
        "n_combos": {n: len(param_grid(learners_cfg.search_for(n))) for n in names},
        "tune_metrics_path": str(path),
        "seed": seed,
    }
    if n_splits is not None:
        payload["n_splits"] = n_splits
    _print_json(payload)


def run_models_tune_demand(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    seed = _resolve_seed(args, config)
    set_seed(seed)
    names = resolve_learner_names(args.learner)
    demand = read_demand_csv(resolve_data_path(config.data.processed_path))
    gap = horizon_bins(config.models.demand.horizon_minutes, config.simulation.timestep_minutes)
    best_params, fold_metrics, val_mae = tune_demand_learners(
        demand,
        learners=config.models.demand.learners,
        timestep_minutes=config.simulation.timestep_minutes,
        horizon_minutes=config.models.demand.horizon_minutes,
        n_splits=config.models.demand.n_splits,
        seed=seed,
        gap=gap,
        names=names,
    )
    _handle_tune(
        names,
        best_params,
        fold_metrics,
        val_mae,
        resolve_data_path(config.models.demand.tune_metrics_path),
        seed,
        config.models.demand.learners,
        n_splits=config.models.demand.n_splits,
    )
    return 0


def run_models_tune_energy(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    seed = _resolve_seed(args, config)
    set_seed(seed)
    names = resolve_learner_names(args.learner)
    trips = generate_synthetic_trips(
        config.models.energy,
        seed=seed,
        train_fraction=config.data.train_fraction,
        val_fraction=config.data.val_fraction,
    )
    best_params, fold_metrics, val_mae = tune_energy_learners(
        trips, spec=config.models.energy, seed=seed, names=names
    )
    _handle_tune(
        names,
        best_params,
        fold_metrics,
        val_mae,
        resolve_data_path(config.models.energy.tune_metrics_path),
        seed,
        config.models.energy.learners,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "experiment":
        return run_experiment(args)
    if args.command == "simulate":
        return run_simulate(args)
    if args.command == "data":
        if args.data_command == "pull":
            return run_data_pull(args)
        if args.data_command == "features":
            return run_data_features(args)
        print("usage: chargeopt data {pull,features}", file=sys.stderr)
        return 2
    if args.command == "models":
        if args.models_command == "demand":
            return run_models_demand(args)
        if args.models_command == "energy":
            return run_models_energy(args)
        if args.models_command == "tune":
            if args.tune_command == "demand":
                return run_models_tune_demand(args)
            if args.tune_command == "energy":
                return run_models_tune_energy(args)
            print("usage: chargeopt models tune {demand,energy}", file=sys.stderr)
            return 2
        print("usage: chargeopt models {demand,energy,tune}", file=sys.stderr)
        return 2
    build_parser().print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
