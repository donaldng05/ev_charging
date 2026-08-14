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
| `vehicle_idle_minutes` | Time spent waiting or stranded, not driving or charging |

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
