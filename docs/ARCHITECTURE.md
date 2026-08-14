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
| `chargeopt.simulation` | M3 | Vehicles, stations, queues, SOC |
| `chargeopt.optimization` | M4 | Nearest / cheapest / ML-informed policies |
| `chargeopt.evaluation` | M5–M6 | Metrics, multi-seed runs, stress test |
| `chargeopt.cli` | M0 / M5 | `chargeopt experiment ...` |

Config lives in [`configs/default.yaml`](../configs/default.yaml). Code must not hardcode fleet size, timestep, or horizon.

## In MVP 1

- 2 models: station demand (next hour) and trip energy
- 3 policies: nearest, cheapest, ML-informed
- 1 discrete-time simulator (15-minute ticks, 24-hour horizon)
- 6 metrics: energy cost, wait, SOC violations, energy usage, station utilization, idle time
- 1 stress scenario: 1.5× demand, −10°C, 80% station availability

## Out of MVP 1

RL, transformers, GNNs, deep learning requirement, Kafka, Spark, CUDA, Kubernetes, real-time dashboards, Tesla-specific telemetry, CARLA.

## Dataset contracts

Deferred until a dataset is chosen. Do not invent Parquet schemas ahead of the columns.
