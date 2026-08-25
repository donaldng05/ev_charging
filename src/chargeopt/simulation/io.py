"""CSV persistence for one simulator run."""

from pathlib import Path

import pandas as pd

from chargeopt.simulation.engine import SimResult
from chargeopt.simulation.report import HOME_ROUTING, metrics_row
from chargeopt.utils.io import select_columns, write_csv

STATION_COLUMNS: tuple[str, ...] = (
    "station_id",
    "x_km",
    "y_km",
    "n_chargers",
    "power_kw",
    "price_per_kwh",
)
VEHICLE_TICK_COLUMNS: tuple[str, ...] = (
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
STATION_TICK_COLUMNS: tuple[str, ...] = (
    "tick",
    "timestamp",
    "station_id",
    "occupancy",
    "queue_len",
    "energy_delivered_kwh",
)
METRIC_COLUMNS: tuple[str, ...] = (
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
    _write(result.stations, stations_path, STATION_COLUMNS)
    _write(result.vehicle_ticks, run_path, VEHICLE_TICK_COLUMNS)
    _write(result.station_ticks, station_ticks_path, STATION_TICK_COLUMNS)
    frame = metrics if metrics is not None else pd.DataFrame([metrics_row(result, HOME_ROUTING)])
    _write(frame, metrics_path, METRIC_COLUMNS)


def read_sim_stations(path: Path) -> pd.DataFrame:
    return _read(path, STATION_COLUMNS)


def read_vehicle_ticks(path: Path) -> pd.DataFrame:
    frame = _read(path, VEHICLE_TICK_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def read_station_ticks(path: Path) -> pd.DataFrame:
    frame = _read(path, STATION_TICK_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def read_sim_metrics(path: Path) -> pd.DataFrame:
    return _read(path, METRIC_COLUMNS)


def _write(frame: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> None:
    write_csv(frame, path, columns=columns, label="simulation artifact")


def _read(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        msg = f"simulation artifact not found: {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    return select_columns(frame, columns, label="simulation artifact")
