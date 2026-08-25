# M6 Distribution-Shift Stress Evaluation

## Method

The official M6 run used `configs/default.yaml`, the repository's local ACN
snapshot, all three configured policies (`nearest`, `cheapest`, and
`ml_informed`), and seeds 42–51. The command executed was:

```text
chargeopt experiment --config configs/default.yaml --stress
```

This produced 60 in-memory simulations: 3 policies × 10 seeds × 2 paired
scenarios. Normal and stress runs used the same policy and seed. Stress changed
only the declared inputs: demand multiplier `1.5`, temperature `−10°C`, and
80% station availability. With 10 stations, two seed-selected stations were
disabled by setting their effective charger count to zero; station identities
and locations were retained.

The raw and summary artifacts are gitignored and were written to the paths in
the resolved config. The recorded configuration hash was
`a54e1d33962a08a96d8db0481ff098da9b719bfd5916035ecba24a4f8f93a622`. The CLI
reported Git SHA `cbd94de3a8ebd932675066dbd307b8d230af8e78` at evaluation time.

## Results

Means below are over the ten paired seeds. `stress − normal` is the signed
paired difference. Ratios are `stress_mean / normal_mean`; `NA` means the
normal mean is zero.

| Policy | Metric | Normal mean | Stress mean | Stress − normal | Ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| nearest | energy cost | 161.019 | 202.221 | +41.202 | 1.256 |
| cheapest | energy cost | 108.469 | 154.720 | +46.251 | 1.426 |
| ml_informed | energy cost | 122.800 | 166.165 | +43.365 | 1.353 |
| nearest | average wait (min) | 0.000 | 0.033 | +0.033 | NA |
| cheapest | average wait (min) | 0.000 | 0.050 | +0.050 | NA |
| ml_informed | average wait (min) | 0.000 | 0.033 | +0.033 | NA |
| nearest | SOC violations | 0.000 | 0.000 | +0.000 | NA |
| cheapest | SOC violations | 0.000 | 0.000 | +0.000 | NA |
| ml_informed | SOC violations | 0.000 | 0.000 | +0.000 | NA |
| nearest | energy usage (kWh) | 132.643 | 251.629 | +118.986 | 1.897 |
| cheapest | energy usage (kWh) | 132.643 | 251.629 | +118.986 | 1.897 |
| ml_informed | energy usage (kWh) | 132.643 | 251.629 | +118.986 | 1.897 |
| nearest | station utilization | 0.0625 | 0.1172 | +0.0547 | 1.875 |
| cheapest | station utilization | 0.0625 | 0.1172 | +0.0547 | 1.875 |
| ml_informed | station utilization | 0.0625 | 0.1172 | +0.0547 | 1.875 |
| nearest | vehicle idle (min) | 0.000 | 3.000 | +3.000 | NA |
| cheapest | vehicle idle (min) | 0.000 | 4.500 | +4.500 | NA |
| ml_informed | vehicle idle (min) | 0.000 | 3.000 | +3.000 | NA |

## Uncertainty and interpretation

Paired 95% normal-approximation intervals used the sample standard deviation of
the ten seed-level differences. For energy cost, the delta intervals were
`[33.815, 48.589]` for nearest, `[37.563, 54.940]` for cheapest, and
`[34.713, 52.018]` for ML-informed. The ML-informed policy matched nearest on
stress wait and idle time and improved on the cheapest baseline, but it did not
beat cheapest on normal or stress energy cost. This M6 run therefore does not
show a general ML advantage; it shows limited congestion robustness under this
specific low-utilization normal calibration and declared stress.

## Limitations and threats to validity

- The simulator fleet, station geography, prices, trips, and SOC are synthetic;
  ACN-Data calibrates aggregate arrival, duration, and energy behavior only.
- The default normal scenario is lightly loaded, so policy differences in wait
  are near a floor. The separate `configs/congestion.yaml` profile remains the
  M4 congestion experiment and is not substituted into M6.
- The demand forecast is site-level rather than station-level, and the M6 run
  does not retrain or retune models, policy weights, or forecast scaling.
- Station availability is a one-time deterministic capacity reduction, not a
  time-varying outage process. The normal confidence intervals are unbounded
  normal approximations and are not claims of calibrated predictive coverage.
- Ten seeds and one local snapshot are useful for reproducibility but do not
  establish broad external validity. The zero normal means for several metrics
  make relative ratios undefined; the report retains absolute paired deltas.
