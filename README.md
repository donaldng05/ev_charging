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
uv run chargeopt data pull
uv run chargeopt data features
uv run chargeopt models demand
uv run chargeopt models energy
```

The experiment runner is a placeholder until M5. It already loads and validates `configs/default.yaml`, seeds the process, and prints a stable experiment id.

`chargeopt models demand` reads the processed 15-minute demand CSV and writes gitignored prediction/metrics artifacts. `chargeopt models energy` generates synthetic trips, then writes physics and Random Forest energy predictions.

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Dataset](docs/DATASET.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Assumptions](docs/ASSUMPTIONS.md)
