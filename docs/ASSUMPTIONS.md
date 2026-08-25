# Assumptions

These are frozen for MVP 1. Changing them is a new experiment, not a silent code edit.

## Fleet and infrastructure

- One hybrid region: real Caltech ACN-Data for demand, synthetic metro stations for the fleet simulator. Do not treat Caltech EVSEs as city-scale stations.
- Generic EV profiles. No Tesla-specific vehicle models or proprietary telemetry.
- About 30 vehicles and 10 stations (see `configs/default.yaml`).
- Stations are homogeneous except location, number of chargers, charging power, and price.
- One charger power in the default config (150 kW). No mixed connector types.
- Charger counts use ACN mean delivered energy converted to 15-minute charging
  service ticks. ACN connected duration remains a diagnostic because the
  simulator does not model plugged-idle parking.
- Generic 60 kWh batteries start at 70% SOC, preserve a 10% minimum reserve,
  and leave charging at a 90% target.
- Each vehicle has two seeded trips per day and an evenly assigned synthetic
  home station. Two trips is the first policy-neutral candidate that passed the
  frozen normal-scenario calibration gate. M3 home routing always returns
  vehicles to that assigned station. The concentrated-routing probe is an M3
  sensitivity check, not a station-selection policy; nearest, cheapest, and
  ML-informed policies land in M4.
- Synthetic locations occupy a 12 km square and station prices span
  $0.20–$0.45/kWh.

## Time

- Discrete time at 15-minute intervals.
- Simulation horizon is 24 hours (96 steps).

## Charging physics

- Constant charge rate: energy per tick is `power_kw * timestep_hours`.
- No charging curves, tapering, battery temperature, or connector constraints.

## Energy and demand

- Trip energy starts from a fixed-rate baseline (`kWh/km`) plus an ML regressor once M2 lands.
- M3 uses the same noise-free rate × distance × cold-penalty relationship used
  to generate M2 training targets. Fitted estimators are not serialized.
- Demand forecasting predicts next-hour site energy demand, not individual sessions.
- Vehicle telemetry is synthetic. Charging demand is calibrated from public ACN-Data (Caltech). That distinction stays explicit.

## Decisions

- Policies are deterministic scores. No RL, MPC, or stochastic optimization in MVP 1.
- The ML-informed policy may use demand forecasts as a congestion feature. That is the only required ML → decision link.
- M4 policy weights are frozen in `configs/default.yaml` under `optimization`.
  Distance, price, and queue features are min-max normalized among candidate
  stations; ties resolve lexicographically by `station_id`.
- Because the demand forecast is site-level, not station-level, the ML-informed
  policy applies forecast pressure to the current queue term rather than
  inventing station-specific predictions: `clamp(predicted_kwh /
  forecast_scale_kwh, 0, 1)`.
- A policy considers stations with free chargers first. If all stations are
  full, it ranks all stations and allows the simulator's FIFO queue to record
  the resulting wait.

## Evaluation

- Temporal splits only, once data exists. No random shuffle of time series.
- Multi-seed reporting. No single-run leaderboard.
- One distribution-shift scenario (high demand, cold weather, reduced charger availability).

## Explicitly out of scope

No Transformers, GNNs, Spark, Kafka, CUDA, Kubernetes, FastAPI, live dashboards, CARLA, or Alpamayo.
