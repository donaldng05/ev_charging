# Dataset

Public [ACN-Data](https://acnportal.readthedocs.io/en/latest/acndata/data_client.html) calibrates charging demand. Individual vehicle telemetry (SOC, battery capacity, trips, next-trip energy) is synthetically generated. Do not treat ACN-Data as Tesla telemetry or as a city-scale station network.

MVP ingest site is **caltech** only. JPL and office001 are later robustness splits.

## Hybrid mapping

Caltech is **one site with many EVSEs**. `stationID` / `spaceID` are garage-scale chargers, not independent public stations with city-scale distances.

| Piece | Source |
| --- | --- |
| Charging sessions, energy, duration, arrivals | ACN-Data (real) |
| 15-minute site-level demand | Derived from Caltech sessions |
| ~10 geo-distributed sim stations | Synthetic, calibrated from ACN arrival / energy / duration stats |
| Vehicle SOC, battery kWh, trips | Synthetic |
| Weather, electricity price, inter-station GPS | Not in ACN-Data |

## ACN-Data raw fields (observed)

Pulled via `DataClient.get_sessions_by_time("caltech", ...)` for the ingest
window in `configs/default.yaml` (`2018-09-05` to `2018-10-31`). A registered
`CHARGEOPT_ACN_TOKEN` is required for this range; `DEMO_TOKEN` may only cover a
short demo week. Observed keys:

| Field | Example / notes |
| --- | --- |
| `_id` | API document id |
| `sessionID` | Unique session string |
| `stationID` | EVSE id, e.g. `2-39-79-383` |
| `spaceID` | Stall label, e.g. `CA-492` |
| `siteID` | Numeric site code (`0002` for Caltech), not the API name `caltech` |
| `clusterID` | Cluster within the site |
| `connectionTime` | Plug-in (timezone-aware) |
| `disconnectTime` | Unplug |
| `doneChargingTime` | Charge complete; may be null |
| `kWhDelivered` | Energy delivered |
| `timezone` | `America/Los_Angeles` |
| `userID` | Often null |
| `userInputs` | Null or a list of requested-energy dicts; not used in MVP 1 |

## Derived session CSV

Canonical snapshot: `data/raw/acn_caltech_sessions.csv` (gitignored). Not pickle. Not Parquet.

| Column | From |
| --- | --- |
| `session_id` | `sessionID` |
| `site_id` | API site name (`caltech`), not raw `siteID` |
| `station_id` | `stationID` |
| `space_id` | `spaceID` |
| `start_time` | `connectionTime` (ISO-8601 with offset) |
| `end_time` | `disconnectTime` |
| `done_charging_time` | `doneChargingTime` (empty if missing) |
| `duration_min` | `(end_time - start_time)` in minutes |
| `energy_kwh` | `kWhDelivered` |
| `day_of_week` | Monday=0 from `start_time` in site timezone |
| `hour` | 0–23 from `start_time` |

## Demand CSV

`data/processed/acn_caltech_demand_15min.csv` is **site-level** (one series for Caltech), 15-minute bins:

- `n_arrivals` — sessions whose `start_time` falls in the bin
- `energy_kwh` — `kWhDelivered` spread uniformly over `[start_time, done_charging_time]` (fallback: `end_time`)
- calendar: `hour`, `day_of_week`, `is_weekend`, `month`
- lags: `lag_15m`, `lag_1h`, `lag_24h`
- rolling means: `rolling_mean_1h`, `rolling_mean_24h`
- `split` — `train` / `val` / `test`

Holidays are out of MVP 1.

## Demand forecast target

The stored demand CSV does not include a forecast column. At train time M2 derives
`target_next_hour_energy_kwh` as the sum of `energy_kwh` over the next
`models.demand.horizon_minutes` (60 minutes → four 15-minute bins). Rows whose
horizon crosses a `split` boundary are dropped so train labels never use val/test
energy.

Demand features for the learned model are calendar fields plus lags and rolling
means already in the table. Current-bin `energy_kwh` and `n_arrivals` are not
features.

## Demand prediction CSV

Canonical artifact: `data/processed/demand_predictions.csv` (gitignored).

| Column | Notes |
| --- | --- |
| `timestamp` | Forecast issue time (bin start, UTC) |
| `split` | `train` / `val` / `test` of that timestamp |
| `target` | Next-hour energy (kWh) |
| `prediction` | Model output |
| `model` | `last_observation`, `historical_average`, or `random_forest` |
| `seed` | Config/CLI seed |

## Synthetic trips

Canonical artifact: `data/processed/synthetic_trips.csv` (gitignored). Generated
with the experiment seed; not ACN-Data.

| Column | Notes |
| --- | --- |
| `trip_id` | `trip-00000` … sequential generation order |
| `distance_km` | Sampled, clipped to a positive minimum |
| `duration_min` | Sampled, clipped to a positive minimum |
| `speed_kmh` | `distance_km / (duration_min / 60)` |
| `temperature_c` | Synthetic ambient temperature |
| `energy_kwh` | Rate × distance × cold penalty + noise |
| `split` | Chronological generation-index split (same fractions as demand) |

Physics baseline predicts `rate_kwh_per_km * distance_km`. The Random Forest uses
`distance_km` and `temperature_c` only (`duration_min` is sampled but is not in
the generative formula). Energy predictions use
`trip_id` in place of `timestamp` with the same `split`, `target`, `prediction`,
`model`, `seed` contract (`model` is `physics` or `random_forest`).

## Temporal split

Never randomly shuffle sessions or demand intervals. Split the sorted 15-minute timeline by fraction (`train_fraction`, `val_fraction` in config; test is the remainder). Calendar cut dates are **not** frozen until a longer snapshot exists; the rule is time-order, not a specific year.

## What is not in this dataset

SOC, battery capacity, current location, next trip, electricity price, and
lat/lon between independent public stations. Synthetic trips and temperatures
are generated in M2; they are not observed ACN fields.
