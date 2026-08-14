"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from chargeopt import __version__
from chargeopt.config import PolicyName, RuntimeSettings, load_config
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
    return parser


def run_experiment(args: argparse.Namespace) -> int:
    runtime = RuntimeSettings()
    config_path = args.config
    if config_path is None and "config_path" in runtime.model_fields_set:
        config_path = runtime.config_path
    config = load_config(config_path)
    log_level = runtime.log_level or config.logging.level
    configure_logging(log_level)

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "experiment":
        return run_experiment(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
