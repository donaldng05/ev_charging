"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

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
from chargeopt.data.io import (
    read_demand_csv,
    read_sessions_csv,
    write_demand_csv,
    write_trips_csv,
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
from chargeopt.simulation.engine import run_simulation
from chargeopt.simulation.io import write_simulation_artifacts
from chargeopt.simulation.report import run_calibration
from chargeopt.utils.experiment import experiment_id, git_sha
from chargeopt.utils.log import configure_logging
from chargeopt.utils.seed import set_seed

POLICY_ALIASES: dict[str, PolicyName] = {
    "nearest": PolicyName.NEAREST,
    "cheapest": PolicyName.CHEAPEST,
    "ml": PolicyName.ML_INFORMED,
    "ml_informed": PolicyName.ML_INFORMED,
}


def _parse_policy(value: str) -> PolicyName:
    key = value.lower()
    if key not in POLICY_ALIASES:
        allowed = ", ".join(sorted(POLICY_ALIASES))
        msg = f"unknown policy {value!r}; expected one of: {allowed}"
        raise argparse.ArgumentTypeError(msg)
    return POLICY_ALIASES[key]


def _add_learner_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--learner",
        choices=list(LEARNER_NAMES),
        default=None,
        help="Tune one learner (default: all configured learners).",
    )


def _add_config_and_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: first seed in the config).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chargeopt",
        description="EV fleet charging intelligence experiments.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    experiment = subparsers.add_parser(
        "experiment",
        help="Resolve config and print the experiment identity (runner lands in M5).",
    )
    experiment.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML (default: configs/default.yaml).",
    )
    experiment.add_argument(
        "--policy",
        type=_parse_policy,
        default=PolicyName.ML_INFORMED,
        help="Charging policy: nearest, cheapest, ml / ml_informed.",
    )
    experiment.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: first seed in the config).",
    )

    simulate = subparsers.add_parser(
        "simulate",
        help="Run one seeded 24-hour synthetic fleet simulation.",
    )
    _add_config_and_seed(simulate)
    simulate.add_argument(
        "--all-seeds",
        action="store_true",
        help="Run every experiment seed under home routing plus the concentrated-routing probe.",
    )

    data = subparsers.add_parser("data", help="ACN-Data ingest and demand features.")
    data_sub = data.add_subparsers(dest="data_command")
    pull = data_sub.add_parser("pull", help="Snapshot Caltech sessions to CSV.")
    pull.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML (default: configs/default.yaml).",
    )
    features = data_sub.add_parser(
        "features",
        help="Build the 15-minute demand table from a session CSV snapshot.",
    )
    features.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML (default: configs/default.yaml).",
    )

    models = subparsers.add_parser("models", help="Train demand and energy models.")
    models_sub = models.add_subparsers(dest="models_command")
    demand = models_sub.add_parser(
        "demand",
        help="Fit demand baselines and sklearn learners on the 15-minute demand table.",
    )
    _add_config_and_seed(demand)
    energy = models_sub.add_parser(
        "energy",
        help="Generate synthetic trips and fit physics plus sklearn residual energy models.",
    )
    _add_config_and_seed(energy)
    tune = models_sub.add_parser(
        "tune", help="Search hyperparameters without touching the test split."
    )
    tune_sub = tune.add_subparsers(dest="tune_command")
    tune_demand = tune_sub.add_parser(
        "demand",
        help="Walk-forward learner search on chronological train folds.",
    )
    _add_config_and_seed(tune_demand)
    _add_learner_flag(tune_demand)
    tune_energy = tune_sub.add_parser(
        "energy",
        help="Validate-split learner search for trip energy.",
    )
    _add_config_and_seed(tune_energy)
    _add_learner_flag(tune_energy)
    return parser


def _load_from_args(args: argparse.Namespace) -> tuple[RuntimeSettings, AppConfig]:
    runtime = RuntimeSettings()
    config_path = args.config
    if config_path is None and "config_path" in runtime.model_fields_set:
        config_path = runtime.config_path
    config = load_config(config_path)
    log_level = runtime.log_level or config.logging.level
    configure_logging(log_level)
    return runtime, config


def _resolve_seed(args: argparse.Namespace, config: AppConfig) -> int:
    if args.seed is not None:
        return int(args.seed)
    return config.experiment.seeds[0]


def run_experiment(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    seed = _resolve_seed(args, config)
    set_seed(seed)
    run_id = experiment_id(config, seed=seed, policy=args.policy, commit_sha=git_sha())

    payload = {
        "status": "not_implemented",
        "message": "Experiment runner lands in M5. Config resolved successfully.",
        "experiment_id": run_id,
        "policy": args.policy.value,
        "seed": seed,
        "config": config.model_dump(mode="json"),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def run_simulate(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    sessions = read_sessions_csv(resolve_data_path(config.data.snapshot_path))
    stations_path = resolve_data_path(config.simulation.stations_path)
    run_path = resolve_data_path(config.simulation.run_path)
    station_ticks_path = resolve_data_path(config.simulation.station_ticks_path)
    metrics_path = resolve_data_path(config.simulation.metrics_path)
    if args.all_seeds:
        report = run_calibration(config, sessions)
        write_simulation_artifacts(
            report.home_result,
            stations_path=stations_path,
            run_path=run_path,
            station_ticks_path=station_ticks_path,
            metrics_path=metrics_path,
            metrics=report.metrics,
        )
        probe = report.probe_row
        payload = {
            "status": "ok",
            "gate_passed": report.gate["gate_passed"],
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
            "paths": {
                "stations": str(stations_path),
                "run": str(run_path),
                "station_ticks": str(station_ticks_path),
                "metrics": str(metrics_path),
            },
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    seed = _resolve_seed(args, config)
    set_seed(seed)
    result = run_simulation(config, sessions=sessions, seed=seed)
    write_simulation_artifacts(
        result,
        stations_path=stations_path,
        run_path=run_path,
        station_ticks_path=station_ticks_path,
        metrics_path=metrics_path,
    )
    payload = {
        "status": "ok",
        "seed": seed,
        "n_ticks": config.simulation.steps_per_day,
        "n_vehicles": config.simulation.fleet_size,
        "metrics": result.metrics.model_dump(),
        "paths": {
            "stations": str(stations_path),
            "run": str(run_path),
            "station_ticks": str(station_ticks_path),
            "metrics": str(metrics_path),
        },
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def run_data_pull(args: argparse.Namespace) -> int:
    runtime, config = _load_from_args(args)
    start = localize_naive(config.data.start, config.data.timezone)
    end = localize_naive(config.data.end, config.data.timezone)
    path = resolve_data_path(config.data.snapshot_path)
    frame = snapshot_sessions(
        iter_acn_sessions(
            site=config.data.site,
            start=start,
            end=end,
            token=runtime.acn_token,
        ),
        site=config.data.site,
        path=path,
    )
    payload = {
        "status": "ok",
        "site": config.data.site,
        "n_sessions": len(frame),
        "snapshot_path": str(path),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def run_data_features(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    snapshot = resolve_data_path(config.data.snapshot_path)
    processed = resolve_data_path(config.data.processed_path)
    sessions = read_sessions_csv(snapshot)
    demand = build_demand_table(
        sessions,
        timestep_minutes=config.simulation.timestep_minutes,
        timezone_name=config.data.timezone,
        train_fraction=config.data.train_fraction,
        val_fraction=config.data.val_fraction,
        covid_start=localize_naive(config.data.covid_start, config.data.timezone),
        covid_end=localize_naive(config.data.covid_end, config.data.timezone),
    )
    write_demand_csv(demand, processed)
    payload = {
        "status": "ok",
        "n_intervals": len(demand),
        "processed_path": str(processed),
        "splits": demand["split"].value_counts().to_dict(),
    }
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
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
    pred_path = resolve_data_path(config.models.demand.predictions_path)
    metrics_path = resolve_data_path(config.models.demand.metrics_path)
    write_demand_predictions(predictions, pred_path)
    write_metrics(metrics, metrics_path)
    slices = error_slices_from_predictions(predictions, demand)
    slices_path = resolve_data_path(config.models.demand.error_slices_path)
    write_error_slices(slices, slices_path)
    test_mae = test_mae_by_model(metrics)
    payload = {
        "status": "ok",
        "n_rows": len(predictions),
        "seed": seed,
        "test_mae": test_mae,
        "learned_beats_baselines": learned_beats_baselines(
            test_mae,
            learners=LEARNER_NAMES,
            baselines=DEMAND_BASELINES,
        ),
        "best_learned": best_learned(test_mae, LEARNER_NAMES),
        "decision_model": config.models.demand.decision_model,
        "predictions_path": str(pred_path),
        "metrics_path": str(metrics_path),
        "error_slices_path": str(slices_path),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
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
    predictions, metrics = train_and_predict_energy(
        trips,
        spec=config.models.energy,
        seed=seed,
    )
    pred_path = resolve_data_path(config.models.energy.predictions_path)
    metrics_path = resolve_data_path(config.models.energy.metrics_path)
    write_energy_predictions(predictions, pred_path)
    write_metrics(metrics, metrics_path)
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
    cold_lookup = {str(row["model"]): float(row["mae"]) for _, row in cold_metrics.iterrows()}
    payload = {
        "status": "ok",
        "n_rows": len(predictions),
        "n_trips": len(trips),
        "seed": seed,
        "test_mae": test_mae,
        "cold_mae": cold_lookup,
        "learned_beats_baselines": learned_beats_baselines(
            test_mae,
            learners=LEARNER_NAMES,
            baselines=(PHYSICS,),
        ),
        "best_learned": best_learned(test_mae, LEARNER_NAMES),
        "trips_path": str(trips_path),
        "predictions_path": str(pred_path),
        "metrics_path": str(metrics_path),
        "cold_metrics_path": str(cold_path),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def run_models_tune_demand(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    seed = _resolve_seed(args, config)
    set_seed(seed)
    names = resolve_learner_names(args.learner)
    demand = read_demand_csv(resolve_data_path(config.data.processed_path))
    gap = horizon_bins(
        config.models.demand.horizon_minutes,
        config.simulation.timestep_minutes,
    )
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
    path = resolve_data_path(config.models.demand.tune_metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(path, index=False)
    payload = {
        "status": "ok",
        "learners": list(names),
        "best_params": best_params,
        "val_mae": val_mae,
        "n_combos": {
            name: len(param_grid(config.models.demand.learners.search_for(name))) for name in names
        },
        "n_splits": config.models.demand.n_splits,
        "tune_metrics_path": str(path),
        "seed": seed,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
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
        trips,
        spec=config.models.energy,
        seed=seed,
        names=names,
    )
    path = resolve_data_path(config.models.energy.tune_metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(path, index=False)
    payload = {
        "status": "ok",
        "learners": list(names),
        "best_params": best_params,
        "val_mae": val_mae,
        "n_combos": {
            name: len(param_grid(config.models.energy.learners.search_for(name))) for name in names
        },
        "tune_metrics_path": str(path),
        "seed": seed,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
