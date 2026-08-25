# EV Fleet Charging Intelligence

Given historical charging behavior and the current state of a small EV fleet, can we predict charging demand and make better charging decisions than simple heuristics?

This repository is the MVP for that question: two classical ML models, a deterministic charging policy, a discrete-time fleet simulator, and a reproducible evaluation protocol. The pipeline is:

**data → prediction → decision → simulated fleet behavior → business/engineering metrics.**

MVP 1 is intentionally small: one region, one fleet (~30 EVs), ~10 stations, 15-minute steps, a 24-hour horizon. No RL, deep learning, streaming, or production serving.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```text
uv sync --frozen
```

## Quality checks

```text
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

Install git hooks (optional):

```text
uv run pre-commit install
```

## CLI

```text
uv run chargeopt --help
uv run chargeopt experiment --config configs/default.yaml --policy ml --seed 42
uv run chargeopt simulate --config configs/default.yaml --seed 42
uv run chargeopt simulate --config configs/default.yaml --policy nearest --seed 42
uv run chargeopt simulate --config configs/default.yaml --policy cheapest --seed 42
uv run chargeopt simulate --config configs/default.yaml --policy ml --seed 42
uv run chargeopt simulate --config configs/congestion.yaml --policy ml --seed 42
uv run chargeopt simulate --config configs/default.yaml --all-seeds
uv run chargeopt data pull
uv run chargeopt data features
uv run chargeopt models demand
uv run chargeopt models energy
uv run chargeopt models tune demand
uv run chargeopt models tune energy
```

The experiment runner is a placeholder until M5. It already loads and validates `configs/default.yaml`, seeds the process, and prints a stable experiment id. M4 policy simulations run one policy at a time through `chargeopt simulate --policy`; `ml` is an alias for `ml_informed` and consumes the configured Random Forest demand forecast.

`chargeopt simulate` reads the normalized ACN session snapshot, calibrates a
synthetic 10-station world, and runs the configured 30-vehicle fleet for 96
15-minute ticks. Default mode writes one seed. `--all-seeds` runs every
`experiment.seeds` value under home-station routing plus one concentrated-routing
probe, writes `data/processed/sim_metrics.csv` (one row per seed and routing),
and prints the frozen normal-scenario calibration gate. Vehicle-tick activity
flags make driving, charging, queued, and stranded time auditable even when
status changes within one tick.

`configs/congestion.yaml` is a separate M4 load profile. It keeps the normal
30-vehicle, 10-station hardware but generates 14 effective trips per vehicle
with `trip_rate_multiplier: 7.0`, producing measurable queue pressure without
changing the default scenario.

`chargeopt models demand` reads the processed 15-minute demand CSV and writes
gitignored prediction, metrics, and error-slice artifacts.
`chargeopt models energy` generates synthetic trips, writes physics plus residual
Random Forest predictions, and a −10°C holdout metrics CSV.
`chargeopt models tune demand|energy` prints `best_params` JSON and writes fold
metrics; it does not rewrite `configs/default.yaml`. Copy winners into the frozen
`n_estimators` / `max_depth` / `min_samples_leaf` fields after a full-history
demand pull.

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Dataset](docs/DATASET.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Assumptions](docs/ASSUMPTIONS.md)
