# Experiments

Define the protocol before looking at policy rankings. Do not tune the system until it "wins" and then report that run.

## Research question

Given historical charging behavior and the current state of a small EV fleet, can we predict charging demand and make better charging decisions than simple heuristics?

## Shared setup

All policies run on the same simulated fleet. Numbers come from [`configs/default.yaml`](../configs/default.yaml):

- 30 vehicles, 10 stations
- 15-minute resolution, 24-hour horizon (96 decision points)
- homogeneous chargers except location, charger count, power, and price

Run each policy across the configured seeds (10 by default). Report mean, standard deviation, and worst-case—not a single seed.

## Policies

| Id | Rule |
| --- | --- |
| `nearest` | Closest charger with capacity |
| `cheapest` | Lowest energy price, ignoring predicted congestion |
| `ml_informed` | Score stations with distance, price, current queue, **and predicted future congestion** |

`ml` on the CLI is an alias for `ml_informed`.

The comparison that matters is nearest vs ML-informed, with cheapest as a second baseline. Everything except the policy stays identical.

## Metrics

| Metric | Meaning |
| --- | --- |
| `energy_cost` | Total charging energy cost |
| `avg_wait_minutes` | Average time waiting for a charger |
| `soc_violations` | Vehicles that cannot complete a trip given SOC |
| `energy_usage_kwh` | Energy consumed by the fleet |
| `station_utilization` | Occupied charger-time / available charger-time |
| `vehicle_idle_minutes` | Total policy-induced delay: queued plus stranded vehicle-minutes |

`vehicle_idle_minutes` excludes normal parked/off-duty time. Queue delay is
also reported as `avg_wait_minutes`; that overlap is intentional because one is
a fleet total and the other is an average per charging session.

## Normal-scenario calibration gate

The normal scenario is frozen before M4 policy rankings. Across configured
seeds it must have zero SOC violations, 5%–20% median aggregate utilization,
queue exposure in at least 3 of 10 seeds, and no more than 10 minutes mean
home-station wait. A concentrated-routing sensitivity probe must add at least
15 minutes average wait and produce a peak queue of at least 3 vehicles.

The first candidate passing these policy-neutral checks is retained. The
calibration does not inspect nearest, cheapest, or ML-informed rankings.

```text
uv run chargeopt simulate --config configs/default.yaml --all-seeds
```

That command writes gitignored `data/processed/sim_metrics.csv` (home-station
rows for every configured seed, plus one concentrated-routing probe on the
first seed) and prints `gate_passed` with median utilization, seeds with
queues, mean wait, probe wait delta, and probe peak queue. Fixture tests do
not require a passing gate; freeze numbers from a local Caltech snapshot run.

## Stress scenario

After the IID comparison, take the same policies under:

- demand × 1.5
- temperature −10°C
- station availability 80%

Report degradation vs the normal scenario:

`robustness_ratio = performance_stress / performance_normal`

A useful MVP result can be: the policy works under IID conditions and degrades under a demand spike. That is allowed. Pretending robustness is solved is not.

## Command shape (M5)

```text
uv run chargeopt experiment --config configs/default.yaml --policy ml --seed 42
```

Until M5 this command resolves config and prints an experiment id; it does not simulate.
