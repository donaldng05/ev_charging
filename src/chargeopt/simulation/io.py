"""CSV persistence for one simulator run."""

from pathlib import Path

import pandas as pd

from chargeopt.simulation.engine import SimResult
from chargeopt.simulation.report import HOME_ROUTING, metrics_row
from chargeopt.utils.io import read_csv, write_csv

STATION_COLUMNS = ("station_id", "x_km", "y_km", "n_chargers", "power_kw", "price_per_kwh")
VEHICLE_TICK_COLUMNS = (
    "tick",
    "timestamp",
    "vehicle_id",
    "status",
    "soc",
    "station_id",
    "trip_index",
    "x_km",
    "y_km",
    "drove_this_tick",
    "charged_this_tick",
    "queued_this_tick",
    "stranded_this_tick",
)
STATION_TICK_COLUMNS = (
    "tick",
    "timestamp",
    "station_id",
    "occupancy",
    "queue_len",
    "energy_delivered_kwh",
)
METRIC_COLUMNS = (
    "seed",
    "routing",
    "peak_queue",
    "energy_cost",
    "avg_wait_minutes",
    "soc_violations",
    "energy_usage_kwh",
    "station_utilization",
    "vehicle_idle_minutes",
)


def write_simulation_artifacts(
    result: SimResult,
    *,
    stations_path: Path,
    run_path: Path,
    station_ticks_path: Path,
    metrics_path: Path,
    metrics: pd.DataFrame | None = None,
) -> None:
    """Write canonical station, vehicle-tick, station-tick, and metric CSVs."""
    write_csv(result.stations, stations_path, columns=STATION_COLUMNS, label="simulation artifact")
    write_csv(
        result.vehicle_ticks, run_path, columns=VEHICLE_TICK_COLUMNS, label="simulation artifact"
    )
    write_csv(
        result.station_ticks,
        station_ticks_path,
        columns=STATION_TICK_COLUMNS,
        label="simulation artifact",
    )
    frame = metrics if metrics is not None else pd.DataFrame([metrics_row(result, HOME_ROUTING)])
    write_csv(frame, metrics_path, columns=METRIC_COLUMNS, label="simulation artifact")


def read_sim_stations(path: Path) -> pd.DataFrame:
    return read_csv(path, columns=STATION_COLUMNS, label="simulation artifact")


def read_vehicle_ticks(path: Path) -> pd.DataFrame:
    return read_csv(
        path, columns=VEHICLE_TICK_COLUMNS, date_columns=("timestamp",), label="simulation artifact"
    )


def read_station_ticks(path: Path) -> pd.DataFrame:
    return read_csv(
        path, columns=STATION_TICK_COLUMNS, date_columns=("timestamp",), label="simulation artifact"
    )


def read_sim_metrics(path: Path) -> pd.DataFrame:
    return read_csv(path, columns=METRIC_COLUMNS, label="simulation artifact")
