# Assumptions

These are frozen for MVP 1. Changing them is a new experiment, not a silent code edit.

## Fleet and infrastructure

- One synthetic metro region, one fleet.
- Generic EV profiles. No Tesla-specific vehicle models or proprietary telemetry.
- About 30 vehicles and 10 stations (see `configs/default.yaml`).
- Stations are homogeneous except location, number of chargers, charging power, and price.
- One charger power in the default config (150 kW). No mixed connector types.

## Time

- Discrete time at 15-minute intervals.
- Simulation horizon is 24 hours (96 steps).

## Charging physics

- Constant charge rate: energy per tick is `power_kw * timestep_hours`.
- No charging curves, tapering, battery temperature, or connector constraints.

## Energy and demand

- Trip energy starts from a fixed-rate baseline (`kWh/km`) plus an ML regressor once M2 lands.
- Demand forecasting predicts next-hour station energy demand, not individual sessions.
- Vehicle telemetry may be synthetic and calibrated against public charging behavior. That distinction must stay explicit once a dataset is chosen.

## Decisions

- Policies are deterministic scores. No RL, MPC, or stochastic optimization in MVP 1.
- The ML-informed policy may use demand forecasts as a congestion feature. That is the only required ML → decision link.

## Evaluation

- Temporal splits only, once data exists. No random shuffle of time series.
- Multi-seed reporting. No single-run leaderboard.
- One distribution-shift scenario (high demand, cold weather, reduced charger availability).

## Explicitly out of scope

No Transformers, GNNs, Spark, Kafka, CUDA, Kubernetes, FastAPI, live dashboards, CARLA, or Alpamayo.
