"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from chargeopt import __version__
from chargeopt.config import (
    AppConfig,
    PolicyName,
    RuntimeSettings,
    load_config,
    resolve_data_path,
)
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


def run_experiment(args: argparse.Namespace) -> int:
    _, config = _load_from_args(args)
    seed = args.seed if args.seed is not None else config.experiment.seeds[0]
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


def run_data_pull(args: argparse.Namespace) -> int:
    from chargeopt.data.acn import iter_acn_sessions, localize_naive, snapshot_sessions

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
    from chargeopt.data.io import read_sessions_csv, write_demand_csv
    from chargeopt.features.demand import build_demand_table

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "experiment":
        return run_experiment(args)
    if args.command == "data":
        if args.data_command == "pull":
            return run_data_pull(args)
        if args.data_command == "features":
            return run_data_features(args)
        print("usage: chargeopt data {pull,features}", file=sys.stderr)
        return 2
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
