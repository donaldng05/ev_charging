"""Seeded construction of synthetic stations, vehicles, and fleet trips."""

from __future__ import annotations

import math

import numpy as np

from chargeopt.config import EnergyModelConfig, SimulationConfig
from chargeopt.simulation.energy import deterministic_trip_energy
from chargeopt.simulation.schemas import (
    CalibrationStats,
    FleetTrip,
    SimStation,
    VehicleState,
    VehicleStatus,
)


def build_stations(
    spec: SimulationConfig, calibration: CalibrationStats, *, seed: int
) -> list[SimStation]:
    """Create synthetic city stations sized from ACN charging service time."""
    rng = np.random.default_rng(seed)
    arrival_rate = spec.fleet_size * spec.trips_per_vehicle / spec.steps_per_day
    total_chargers = max(
        spec.n_stations * spec.n_chargers_min,
        math.ceil(arrival_rate * charging_service_ticks(spec, calibration)),
    )
    base, remainder = divmod(total_chargers, spec.n_stations)
    prices = np.linspace(spec.price_per_kwh_min, spec.price_per_kwh_max, spec.n_stations)
    locs = rng.uniform(0.0, spec.metro_span_km, size=(spec.n_stations, 2))
    return [
        SimStation(
            station_id=f"sim-{i:02d}",
            x_km=float(locs[i, 0]),
            y_km=float(locs[i, 1]),
            n_chargers=max(spec.n_chargers_min, base + (i < remainder)),
            power_kw=spec.charger_power_kw,
            price_per_kwh=float(prices[i]),
        )
        for i in range(spec.n_stations)
    ]


def apply_station_availability(
    stations: list[SimStation], *, availability: float, seed: int
) -> list[SimStation]:
    """Disable a deterministic fraction of stations for a stress scenario."""
    if not math.isfinite(availability) or not 0 < availability <= 1:
        raise ValueError("station availability must be finite and in (0, 1]")
    if not stations:
        raise ValueError("at least one station is required")
    n_active = max(1, math.ceil(len(stations) * availability))
    active = set(
        map(int, np.random.default_rng(seed).choice(len(stations), size=n_active, replace=False))
    )
    return [
        s.model_copy(update={"n_chargers": s.n_chargers if i in active else 0})
        for i, s in enumerate(stations)
    ]


def charging_service_ticks(spec: SimulationConfig, calibration: CalibrationStats) -> int:
    """Convert mean delivered ACN energy to simulator charging ticks."""
    return max(
        1,
        math.ceil(
            calibration.mean_energy_kwh / (spec.charger_power_kw * spec.timestep_minutes / 60.0)
        ),
    )


def build_fleet(spec: SimulationConfig, stations: list[SimStation]) -> list[VehicleState]:
    """Create generic EVs assigned evenly to home stations."""
    if len(stations) != spec.n_stations:
        raise ValueError("station count must match simulation.n_stations")
    return [
        VehicleState(
            vehicle_id=f"vehicle-{i:03d}",
            battery_kwh=spec.battery_kwh,
            soc=spec.soc_initial,
            status=VehicleStatus.IDLE,
            home_station_id=stations[i % len(stations)].station_id,
            x_km=stations[i % len(stations)].x_km,
            y_km=stations[i % len(stations)].y_km,
        )
        for i in range(spec.fleet_size)
    ]


def build_trips(
    simulation: SimulationConfig,
    energy: EnergyModelConfig,
    calibration: CalibrationStats,
    vehicles: list[VehicleState],
    *,
    seed: int,
    temperature_c: float,
    trip_rate_multiplier: float = 1.0,
) -> list[FleetTrip]:
    """Create reproducible daily itineraries using ACN's arrival-hour shape."""
    if trip_rate_multiplier <= 0:
        raise ValueError("trip_rate_multiplier must be positive")
    rng = np.random.default_rng(seed)
    n_trips = max(1, round(simulation.trips_per_vehicle * trip_rate_multiplier))
    tph = 60 // simulation.timestep_minutes
    trips: list[FleetTrip] = []
    for v in vehicles:
        dist = np.clip(
            rng.normal(energy.distance_km_mean, energy.distance_km_std, n_trips), 0.5, None
        )
        dur_min = np.clip(
            rng.normal(energy.duration_min_mean, energy.duration_min_std, n_trips), 1.0, None
        )
        dur_ticks = np.maximum(1, np.ceil(dur_min / simulation.timestep_minutes).astype(int))
        hours = rng.choice(24, size=n_trips, p=calibration.hour_pmf)
        raw = hours * tph + rng.integers(0, tph, size=n_trips)
        order = np.argsort(raw, kind="stable")
        deps = _repair_departures(
            raw[order].astype(int), dur_ticks[order], simulation.steps_per_day
        )
        for idx, (dep, d, dur) in enumerate(zip(deps, dist[order], dur_ticks[order], strict=True)):
            trips.append(
                FleetTrip(
                    vehicle_id=v.vehicle_id,
                    trip_index=idx,
                    departure_tick=int(dep),
                    distance_km=float(d),
                    duration_ticks=int(dur),
                    energy_kwh=deterministic_trip_energy(float(d), temperature_c, energy),
                )
            )
    return trips


def _repair_departures(raw: np.ndarray, durations: np.ndarray, horizon_ticks: int) -> list[int]:
    if int(durations.sum()) + len(durations) - 1 > horizon_ticks:
        raise ValueError("trip itinerary cannot fit within simulation horizon")
    deps = [int(raw[0])]
    for i in range(1, len(raw)):
        deps.append(max(int(raw[i]), deps[i - 1] + int(durations[i - 1]) + 1))
    deps[-1] = min(deps[-1], horizon_ticks - int(durations[-1]))
    for i in range(len(deps) - 2, -1, -1):
        deps[i] = min(deps[i], deps[i + 1] - int(durations[i]) - 1)
    return deps
