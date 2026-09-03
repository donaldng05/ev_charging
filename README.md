# EV Fleet Charging Intelligence

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pydantic V2](https://img.shields.io/badge/Pydantic-V2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![NumPy 2.0](https://img.shields.io/badge/NumPy-2.0-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas 3.0](https://img.shields.io/badge/Pandas-3.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Dataset: Caltech ACN-Data](https://img.shields.io/badge/Data-Caltech%20ACN--Data-FF6F00)](https://ev.caltech.edu/dataset)
[![Modeling: Physics-Informed ML](https://img.shields.io/badge/Modeling-Physics--Informed%20ML-blueviolet)](docs/ARCHITECTURE.md)
[![Engine: Discrete-Time Simulation](https://img.shields.io/badge/Engine-Discrete--Time%20Simulation-4B0082)](docs/ARCHITECTURE.md)


> **Can predictive machine learning and congestion-aware dispatch outperform standard heuristics in commercial EV fleet charging?**

This repository provides an end-to-end simulation, machine learning, and decision-optimization platform to evaluate that question. Rather than evaluating ML models on standalone loss metrics (e.g. RMSE), it connects demand forecasts directly to fleet routing decisions, simulating full operational dynamics (battery state-of-charge, charger wait queues, dynamic tariffs, and vehicle idle times) under both standard and distribution-shift stress conditions.

---

## ⚡ Key Results at a Glance

Evaluated across 10 random seeds (96 15-minute decision steps per 24h horizon, 30 EVs, 10 stations) under both nominal operation and a severe **distribution-shift stress test** (1.5× trip demand, −10°C cold battery drain, 20% charger outages):

| Policy | Decision Logic | Normal Cost ($) | Stress Cost ($) | Avg Wait (min) | SOC Violations | Key Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Nearest** | Shortest distance to available charger | $161.02 | $202.22 | 0.03 | **0** | Fast charger access, but ignores high energy prices. |
| **Cheapest** | Lowest $/kWh rate regardless of location | **$108.47** | **$154.72** | 0.05 | **0** | Minimizes energy bill, but incurs highest queue and idle delays. |
| **ML-Informed** | Balances distance, tariff, & **predicted queue pressure** | $122.80 | $166.17 | **0.03** | **0** | **Optimal trade-off**: Matches Nearest's minimal wait while cutting energy cost substantially below Nearest. |

*Full methodology, paired confidence intervals, and statistical robustness ratios are documented in [docs/M6_REPORT.md](docs/M6_REPORT.md).*

---

## 🏗️ System Architecture

```text
       Real-World Ingest (Caltech/JPL ACN-Data)
                          │
                          ▼
            Feature Engineering & Lags
          (Cyclical time, lag_1h, lag_24h, lag_1w)
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
  Station Demand Forecast         Trip Energy Predictor
   (Tuned Random Forest)         (Kinematic Physics + Residuals)
          │                               │
          └───────────────┬───────────────┘
                          ▼
             Charging Policy Layer
      (Nearest vs. Cheapest vs. ML-Informed)
                          │
                          ▼
          Discrete-Time Fleet Simulator
  (15-min ticks, 96 steps, FIFO queues, SOC kinetics, outages)
                          │
                          ▼
        Reproducible Evaluation Protocol
     (Multi-seed matrix, paired CIs, Git SHA & config hash)
```

---

## 📁 Codebase Hierarchy

The codebase is organized into modular, strictly-typed packages under `src/chargeopt/`:

```text
src/chargeopt/
├── cli.py                     # Unified table-driven CLI (experiment, simulate, data, models)
├── config.py                  # Pydantic V2 strictly-typed settings & parameter boundaries
│
├── data/                      # Data ingestion & schema validation
│   ├── acn.py                 # Real-world ACN-Data API/snapshot ingestion & timezone handling
│   ├── schemas.py             # Pydantic models for raw and normalized session records
│   ├── validation.py          # Data-cleaning rules (duration bounds, energy sanity checks)
│   └── io.py                  # Type-checked CSV reading and writing
│
├── features/                  # Feature engineering pipelines
│   ├── demand.py              # 15-minute time-series binning, cyclical temporal & lag features
│   └── energy.py              # Synthetic trip generation with temperature & speed dynamics
│
├── models/                    # ML forecasting & physics-informed modeling
│   ├── demand.py              # Station demand forecasting pipeline (train/test temporal split)
│   ├── energy.py              # Kinematic physics model + residual Random Forest for cold-weather drain
│   ├── learners.py            # Learner registry (RandomForest, Ridge, ElasticNet, ExtraTrees, HGB)
│   ├── tune.py                # Hyperparameter grid-search engine with cross-validation
│   ├── metrics.py             # Statistical error analysis (MAE, RMSE, R²), slices, & baselines
│   ├── baselines.py           # Naive historical lag and mean benchmark baselines
│   └── io.py                  # Model prediction, metric, and error-slice serialization
│
├── optimization/              # Decision policies
│   └── policy.py              # StationChooser protocol: Nearest, Cheapest, and ML-Informed policies
│
├── simulation/                # Discrete-event fleet simulator
│   ├── engine.py              # 15-minute tick loop (96 steps): movement, charging, FIFO queues, SOC
│   ├── world.py               # Spatial grid, station coordinates, charger specs, and vehicle entities
│   ├── calibration.py         # Calibrates arrival rates & session durations from empirical data
│   ├── schemas.py             # Simulation domain models (Vehicle, Station, Trip, TickRecord)
│   ├── energy.py              # Battery drain and constant-rate charging kinetics
│   └── report.py              # Aggregates run-level KPIs (utilization, wait time, cost, SOC violations)
│
├── evaluation/                # Benchmarking & statistical protocol
│   └── protocol.py            # Multi-seed matrix runner, 95% confidence intervals, & paired stress tests
│
└── utils/                     # Supporting utilities
    ├── experiment.py          # Git SHA & configuration hash tracking for 100% reproducibility
    ├── seed.py                # Deterministic PRNG seeding across Python, NumPy, and scikit-learn
    ├── log.py                 # Structured logging configuration
    └── io.py                  # Atomic file reading/writing helpers
```

---

## 🚀 Quickstart

### 1. Prerequisites & Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+:

```bash
# Clone the repository
git clone https://github.com/donaldng05/ev_charging.git
cd ev_charging

# Install dependencies with locked versions
uv sync --frozen
```

### 2. Run Verification & Quality Suite

The codebase maintains strict static typing and a 95% test coverage requirement:

```bash
uv run pytest               # Run 159 tests (unit, integration, calibration, reproducibility)
uv run ruff check .         # Fast linting
uv run ruff format --check . # Code formatting check
uv run mypy src             # Strict static type check
```

---

## 💻 CLI Usage

All functionality is accessible via the unified `chargeopt` CLI:

### Run Benchmark Experiments (Multi-Seed Matrix)

```bash
# Run full evaluation matrix across all policies and seeds
uv run chargeopt experiment --config configs/default.yaml

# Run paired distribution-shift stress test (-10°C, 1.5x load, 80% station availability)
uv run chargeopt experiment --config configs/default.yaml --stress

# Filter to a specific policy or seed
uv run chargeopt experiment --config configs/default.yaml --policy ml --seed 42
```

### Run Discrete-Time Fleet Simulation

```bash
# Simulate fleet behavior under a specific policy
uv run chargeopt simulate --config configs/default.yaml --policy ml --seed 42
uv run chargeopt simulate --config configs/default.yaml --policy cheapest --seed 42
uv run chargeopt simulate --config configs/default.yaml --policy nearest --seed 42

# Run calibration gate across all seeds
uv run chargeopt simulate --config configs/default.yaml --all-seeds

# Run high-congestion scenario (14 trips/EV)
uv run chargeopt simulate --config configs/congestion.yaml --policy ml --seed 42
```

### Data Pipeline & Model Training

```bash
# Download and process ACN-Data session records
uv run chargeopt data pull
uv run chargeopt data features

# Train demand forecaster and physics-informed energy model
uv run chargeopt models demand
uv run chargeopt models energy

# Tune hyperparameters across candidate learners
uv run chargeopt models tune demand
uv run chargeopt models tune energy
```

---

## 🛡️ Engineering & Design Highlights

- **Strict Type Safety**: 100% typed with Python 3.12 type hints, validated under `mypy --strict` with Pydantic V2 integration.
- **Physics-Informed ML**: Hybrid energy prediction combines kinematic mechanics (aerodynamic drag, rolling resistance, HVAC load) with residual Random Forest learning for cold-weather temperature holdouts (−10°C).
- **100% Reproducible Experiments**: Every evaluation run records the exact Git commit SHA, full YAML configuration hash, and deterministic PRNG seeds.
- **Robustness Under Distribution Shift**: Explicitly evaluates policy failure modes under peak travel multipliers, sub-zero temperatures, and partial infrastructure downtime rather than assuming ideal stationary conditions.
- **High Test Standards**: 159 tests covering data validation, model contracts, calibration gates, simulation invariants, and CLI argument parsing, backed by automated coverage enforcement.

---

## 📖 Deep-Dive Documentation

- [Architecture & System Design](docs/ARCHITECTURE.md): Detailed pipeline design, component contracts, and scope boundaries.
- [Experimental Protocol](docs/EXPERIMENTS.md): Formal research hypotheses, metric definitions, and evaluation methodology.
- [Distribution-Shift Stress Report](docs/M6_REPORT.md): Complete findings, confidence intervals, and paired stress analysis.
- [Dataset & Ingestion Guide](docs/DATASET.md): Real-world Caltech/JPL ACN-Data snapshot characteristics and preprocessing.
- [Engineering Assumptions](docs/ASSUMPTIONS.md): Explicit documentation of simulation mechanics, hardware parameters, and simplifications.
