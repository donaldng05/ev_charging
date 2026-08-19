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
    spec: SimulationConfig,
    calibration: CalibrationStats,
    *,
    seed: int,
) -> list[SimStation]:
    """Create synthetic city stations sized from ACN charging service time."""
    rng = np.random.default_rng(seed)
    arrival_rate_per_tick = spec.fleet_size * spec.trips_per_vehicle / spec.steps_per_day
    mean_service_ticks = charging_service_ticks(spec, calibration)
    total_chargers = max(
        spec.n_stations * spec.n_chargers_min,
        math.ceil(arrival_rate_per_tick * mean_service_ticks),
    )
    base, remainder = divmod(total_chargers, spec.n_stations)
    prices = np.linspace(
        spec.price_per_kwh_min,
        spec.price_per_kwh_max,
        spec.n_stations,
    )
    locations = rng.uniform(0.0, spec.metro_span_km, size=(spec.n_stations, 2))
    return [
        SimStation(
            station_id=f"sim-{index:02d}",
            x_km=float(locations[index, 0]),
            y_km=float(locations[index, 1]),
            n_chargers=max(spec.n_chargers_min, base + (index < remainder)),
            power_kw=spec.charger_power_kw,
            price_per_kwh=float(prices[index]),
        )
        for index in range(spec.n_stations)
    ]


def charging_service_ticks(
    spec: SimulationConfig,
    calibration: CalibrationStats,
) -> int:
    """Convert mean delivered ACN energy to simulator charging ticks."""
    energy_per_tick = spec.charger_power_kw * spec.timestep_minutes / 60.0
    return max(1, math.ceil(calibration.mean_energy_kwh / energy_per_tick))


def build_fleet(
    spec: SimulationConfig,
    stations: list[SimStation],
) -> list[VehicleState]:
    """Create generic EVs assigned evenly to home stations."""
    if len(stations) != spec.n_stations:
        msg = "station count must match simulation.n_stations"
        raise ValueError(msg)
    vehicles: list[VehicleState] = []
    for index in range(spec.fleet_size):
        home = stations[index % len(stations)]
        vehicles.append(
            VehicleState(
                vehicle_id=f"vehicle-{index:03d}",
                battery_kwh=spec.battery_kwh,
                soc=spec.soc_initial,
                status=VehicleStatus.IDLE,
                home_station_id=home.station_id,
                x_km=home.x_km,
                y_km=home.y_km,
            )
        )
    return vehicles


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
        msg = "trip_rate_multiplier must be positive"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    trips_per_vehicle = max(
        1,
        round(simulation.trips_per_vehicle * trip_rate_multiplier),
    )
    ticks_per_hour = 60 // simulation.timestep_minutes
    trips: list[FleetTrip] = []
    for vehicle in vehicles:
        distances = np.clip(
            rng.normal(energy.distance_km_mean, energy.distance_km_std, trips_per_vehicle),
            0.5,
            None,
        )
        durations_min = np.clip(
            rng.normal(energy.duration_min_mean, energy.duration_min_std, trips_per_vehicle),
            1.0,
            None,
        )
        duration_ticks = np.maximum(
            1,
            np.ceil(durations_min / simulation.timestep_minutes).astype(int),
        )
        hours = rng.choice(24, size=trips_per_vehicle, p=calibration.hour_pmf)
        within_hour = rng.integers(0, ticks_per_hour, size=trips_per_vehicle)
        raw_departures = hours * ticks_per_hour + within_hour
        order = np.argsort(raw_departures, kind="stable")
        ordered_departures = raw_departures[order].astype(int)
        ordered_distances = distances[order]
        ordered_durations = duration_ticks[order]
        departures = _repair_departures(
            ordered_departures,
            ordered_durations,
            simulation.steps_per_day,
        )
        for trip_index, (departure, distance, duration) in enumerate(
            zip(
                departures,
                ordered_distances,
                ordered_durations,
                strict=True,
            )
        ):
            trips.append(
                FleetTrip(
                    vehicle_id=vehicle.vehicle_id,
                    trip_index=trip_index,
                    departure_tick=int(departure),
                    distance_km=float(distance),
                    duration_ticks=int(duration),
                    energy_kwh=deterministic_trip_energy(
                        float(distance),
                        temperature_c,
                        energy,
                    ),
                )
            )
    return trips


def _repair_departures(
    raw: np.ndarray,
    durations: np.ndarray,
    horizon_ticks: int,
) -> list[int]:
    required = int(durations.sum()) + len(durations) - 1
    if required > horizon_ticks:
        msg = "trip itinerary cannot fit within simulation horizon"
        raise ValueError(msg)
    departures = [int(raw[0])]
    for index in range(1, len(raw)):
        earliest = departures[index - 1] + int(durations[index - 1]) + 1
        departures.append(max(int(raw[index]), earliest))
    departures[-1] = min(
        departures[-1],
        horizon_ticks - int(durations[-1]),
    )
    for index in range(len(departures) - 2, -1, -1):
        latest = departures[index + 1] - int(durations[index]) - 1
        departures[index] = min(departures[index], latest)
    return departures
