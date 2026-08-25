# Architecture

MVP 1 is one end-to-end pipeline. Later phases extend these modules; they should not require a rewrite.

```text
Historical data
      │
      ▼
Data processing
      │
      ├──────────────┐
      ▼              ▼
Demand forecast   Energy prediction
      │              │
      └──────┬───────┘
             ▼
      Charging policy
             │
             ▼
       Fleet simulator
             │
             ▼
        Evaluation
```

The ML models must change a downstream charging decision. Forecast RMSE alone is not a result.

## Module map

| Package | Milestone | Responsibility |
| --- | --- | --- |
| `chargeopt.config` | M0 | Frozen experiment YAML → Pydantic |
| `chargeopt.data` | M1 | Load, validate, split |
| `chargeopt.features` | M1–M2 | Demand and energy features |
| `chargeopt.models` | M2 | Demand forecast + trip energy |
| `chargeopt.simulation` | M3 | Vehicles, stations, queues, SOC, simulation mechanics |
| `chargeopt.optimization` | M3–M4 | Station-selection contract and nearest / cheapest / ML-informed policies |
| `chargeopt.evaluation` | M5–M6 | Policy/seed matrix, reproducible summaries, stress test |
| `chargeopt.cli` | M0–M5 | `chargeopt experiment`, `simulate`, M4 policy selection, `data pull|features`, `models demand|energy|tune` |

Config lives in [`configs/default.yaml`](../configs/default.yaml). Code must not hardcode fleet size, timestep, or horizon.

## In MVP 1

- 2 modeling tasks: station demand (next hour) and trip energy, each compared across sklearn learners (`random_forest`, `ridge`, `elasticnet`, `extra_trees`, `hist_gradient_boosting`) plus naive/physics baselines
- 3 policies: nearest, cheapest, ML-informed
- 1 discrete-time simulator (15-minute ticks, 24-hour horizon)
- 6 metrics: energy cost, wait, SOC violations, energy usage, station utilization, idle time
- 1 stress scenario: 1.5× demand, −10°C, 80% station availability

## M3 simulator flow

`chargeopt simulate` calibrates an arrival-hour distribution and mean connected
duration/energy from the normalized ACN session CSV. It then creates synthetic
city stations, generic EVs, and seeded daily itineraries. The 15-minute engine
applies deterministic trip consumption, FIFO station queues, charger capacity,
constant-rate charging, and SOC bounds for 96 ticks.

The M3 station chooser sends each EV to its assigned home station through the
small `chargeopt.optimization` protocol boundary. `--all-seeds` also runs a
concentrated-routing sensitivity probe that sends every vehicle to `sim-00`; that
probe is not an M4 policy. M4 adds nearest, cheapest, and ML-informed policies
without changing simulation mechanics. The former `chargeopt.simulation.policy`
module remains a compatibility import for existing callers. ACN EVSE
identifiers are never used as geo-distributed simulator station identifiers.

M4 adds nearest, cheapest, and ML-informed choosers without changing the world
transition mechanics. Policies first consider stations with free charger
capacity, then fall back to all stations when every charger is occupied so FIFO
queue behavior remains observable. Distance, price, and queue features are
normalized within the candidate set and ties resolve by `station_id`.

The demand forecast is site-level rather than station-specific. The
ML-informed policy therefore uses the forecast to scale the current queue
penalty: `forecast_pressure = clamp(predicted_kwh / forecast_scale_kwh, 0, 1)`.
Forecast loading stays in the CLI/integration layer; policy classes receive
plain tick-indexed numeric values and do not import model-training code.

## M5 evaluation flow

`chargeopt experiment` loads the normalized ACN session snapshot once, selects
the configured policy and seed matrix, and runs each policy through the same
simulation interface. The runner keeps detailed `simulate` artifacts separate
and writes compact evaluation results under `evaluation.*` paths in the active
YAML profile. Raw rows contain policy, seed, experiment id, config hash, Git
SHA, and all six metrics. Summary rows contain mean, sample standard deviation,
metric-direction worst case, and a 95% normal-approximation interval.

## Out of MVP 1

RL, transformers, GNNs, deep learning requirement, Kafka, Spark, CUDA, Kubernetes, real-time dashboards, Tesla-specific telemetry, CARLA.

## Dataset contracts

ACN-Data (Caltech sessions) is the real charging-behavior source. Vehicle SOC/trips and geo-distributed sim stations are synthetic. See [DATASET.md](DATASET.md). ACN-Sim is not used in MVP 1.
