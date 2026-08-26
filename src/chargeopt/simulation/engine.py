"""Deterministic discrete-time fleet simulation engine."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from chargeopt.config import AppConfig, SimulationConfig
from chargeopt.optimization.policy import HomeStationChooser, StationChooser
from chargeopt.simulation.calibration import calibrate_from_sessions
from chargeopt.simulation.metrics import build_metrics
from chargeopt.simulation.schemas import (
    FleetTrip,
    SimMetrics,
    SimStation,
    StationTick,
    VehicleState,
    VehicleStatus,
    VehicleTick,
)
from chargeopt.simulation.world import (
    apply_station_availability,
    build_fleet,
    build_stations,
    build_trips,
)


@dataclass(frozen=True)
class SimResult:
    stations: pd.DataFrame
    vehicle_ticks: pd.DataFrame
    station_ticks: pd.DataFrame
    metrics: SimMetrics


def run_simulation(
    config: AppConfig,
    *,
    sessions: pd.DataFrame,
    seed: int,
    chooser: StationChooser | None = None,
    temperature_c: float | None = None,
    trip_rate_multiplier: float | None = None,
    station_availability: float | None = None,
) -> SimResult:
    """Calibrate, construct, and run one seeded synthetic fleet day."""
    simulation = config.simulation
    calibration = calibrate_from_sessions(
        sessions,
        timestep_minutes=simulation.timestep_minutes,
    )
    stations = build_stations(simulation, calibration, seed=seed)
    if station_availability is not None:
        stations = apply_station_availability(
            stations,
            availability=station_availability,
            seed=seed,
        )
    vehicles = build_fleet(simulation, stations)
    trips = build_trips(
        simulation,
        config.models.energy,
        calibration,
        vehicles,
        seed=seed,
        temperature_c=(
            config.models.energy.temperature_mean_c if temperature_c is None else temperature_c
        ),
        trip_rate_multiplier=(
            simulation.trip_rate_multiplier
            if trip_rate_multiplier is None
            else trip_rate_multiplier
        ),
    )
    return run_world(
        simulation,
        stations=stations,
        vehicles=vehicles,
        trips=trips,
        seed=seed,
        timezone_name=config.data.timezone,
        chooser=chooser,
    )


def run_world(
    spec: SimulationConfig,
    *,
    stations: list[SimStation],
    vehicles: list[VehicleState],
    trips: list[FleetTrip],
    seed: int,
    timezone_name: str,
    chooser: StationChooser | None = None,
) -> SimResult:
    """Run a fully constructed world for the configured number of ticks."""
    if len(vehicles) != spec.fleet_size:
        msg = "vehicle count must match simulation.fleet_size"
        raise ValueError(msg)
    if len(stations) != spec.n_stations:
        msg = "station count must match simulation.n_stations"
        raise ValueError(msg)
    chooser = chooser or HomeStationChooser()
    station_by_id = {station.station_id: station for station in stations}
    if len(station_by_id) != len(stations):
        msg = "station ids must be unique"
        raise ValueError(msg)
    states = {vehicle.vehicle_id: vehicle.model_copy(deep=True) for vehicle in vehicles}
    if len(states) != len(vehicles):
        msg = "vehicle ids must be unique"
        raise ValueError(msg)

    itineraries: dict[str, list[FleetTrip]] = defaultdict(list)
    trip_by_key: dict[tuple[str, int], FleetTrip] = {}
    for trip in trips:
        if trip.vehicle_id not in states:
            msg = f"trip references unknown vehicle {trip.vehicle_id!r}"
            raise ValueError(msg)
        itineraries[trip.vehicle_id].append(trip)
        trip_by_key[(trip.vehicle_id, trip.trip_index)] = trip
    for itinerary in itineraries.values():
        itinerary.sort(key=lambda trip: trip.trip_index)

    next_trip = {vehicle_id: 0 for vehicle_id in states}
    occupancy: dict[str, list[str]] = {station.station_id: [] for station in stations}
    queues: dict[str, deque[str]] = {station.station_id: deque() for station in stations}
    vehicle_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    start = datetime.combine(
        spec.start_day,
        time.min,
        tzinfo=ZoneInfo(timezone_name),
    )
    energy_cost = 0.0
    energy_usage = 0.0
    soc_violations = 0
    queued_ticks = 0
    policy_delay_ticks = 0
    occupied_ticks = 0
    charge_sessions = 0
    stranded_until: dict[str, int] = {}

    for tick in range(spec.steps_per_day):
        timestamp = start + timedelta(minutes=tick * spec.timestep_minutes)
        _release_stranded(tick, states, stranded_until)
        _release_for_departures(
            tick,
            states,
            itineraries,
            next_trip,
            occupancy,
            queues,
        )
        for station in stations:
            _promote_queue(
                station.station_id,
                station_by_id,
                states,
                occupancy,
                queues,
            )
        for vehicle_id in states:
            state = states[vehicle_id]
            itinerary = itineraries.get(vehicle_id, [])
            position = next_trip[vehicle_id]
            if position >= len(itinerary):
                continue
            trip = itinerary[position]
            if trip.departure_tick != tick or state.status is not VehicleStatus.IDLE:
                continue
            available_kwh = (state.soc - spec.soc_min) * state.battery_kwh
            next_trip[vehicle_id] += 1
            if available_kwh + 1e-12 < trip.energy_kwh:
                soc_violations += 1
                state.trip_index = trip.trip_index
                state.status = VehicleStatus.STRANDED
                stranded_until[vehicle_id] = tick + trip.duration_ticks
                continue
            state.status = VehicleStatus.DRIVING
            state.station_id = None
            state.trip_index = trip.trip_index
            state.remaining_travel_ticks = trip.duration_ticks

        drove_this_tick: set[str] = set()
        for state in states.values():
            if state.status is not VehicleStatus.DRIVING:
                continue
            drove_this_tick.add(state.vehicle_id)
            trip = trip_by_key[(state.vehicle_id, state.trip_index)]
            consumed = trip.energy_kwh / trip.duration_ticks
            next_soc = state.soc - consumed / state.battery_kwh
            if next_soc < 0:
                soc_violations += 1
            actual_consumed = min(consumed, state.soc * state.battery_kwh)
            energy_usage += actual_consumed
            state.soc = max(0.0, next_soc)
            state.remaining_travel_ticks -= 1
            if state.remaining_travel_ticks == 0:
                station_id = chooser.choose(
                    state,
                    stations,
                    tick=tick,
                    occupancy={key: len(value) for key, value in occupancy.items()},
                    queues={key: tuple(value) for key, value in queues.items()},
                )
                if station_id not in station_by_id:
                    msg = f"chooser returned unknown station {station_id!r}"
                    raise ValueError(msg)
                station = station_by_id[station_id]
                state.station_id = station_id
                state.x_km = station.x_km
                state.y_km = station.y_km
                charge_sessions += 1
                if len(occupancy[station_id]) < station.n_chargers:
                    occupancy[station_id].append(state.vehicle_id)
                    state.status = VehicleStatus.CHARGING
                else:
                    queues[station_id].append(state.vehicle_id)
                    state.status = VehicleStatus.QUEUED

        station_energy = {station.station_id: 0.0 for station in stations}
        station_occupancy = {
            station.station_id: len(occupancy[station.station_id]) for station in stations
        }
        station_queue_lengths = {
            station.station_id: len(queues[station.station_id]) for station in stations
        }
        charged_this_tick = {
            vehicle_id
            for station_occupancy_ids in occupancy.values()
            for vehicle_id in station_occupancy_ids
        }
        queued_this_tick = {
            vehicle_id for station_queue in queues.values() for vehicle_id in station_queue
        }
        stranded_this_tick = {
            vehicle_id
            for vehicle_id, state in states.items()
            if state.status is VehicleStatus.STRANDED
        }
        queued_ticks += len(queued_this_tick)
        policy_delay_ticks += len(queued_this_tick | stranded_this_tick)
        for station in stations:
            station_id = station.station_id
            for vehicle_id in list(occupancy[station_id]):
                state = states[vehicle_id]
                capacity_left = (spec.soc_charge_target - state.soc) * state.battery_kwh
                delivered = min(
                    station.power_kw * spec.timestep_minutes / 60.0,
                    max(0.0, capacity_left),
                )
                state.soc = min(
                    1.0,
                    state.soc + delivered / state.battery_kwh,
                )
                station_energy[station_id] += delivered
                energy_cost += delivered * station.price_per_kwh
                if state.soc + 1e-12 >= spec.soc_charge_target:
                    occupancy[station_id].remove(vehicle_id)
                    state.status = VehicleStatus.IDLE
                    state.station_id = None
            _promote_queue(
                station_id,
                station_by_id,
                states,
                occupancy,
                queues,
            )

        for state in states.values():
            vehicle_rows.append(
                VehicleTick(
                    tick=tick,
                    timestamp=timestamp,
                    vehicle_id=state.vehicle_id,
                    status=state.status,
                    soc=state.soc,
                    station_id=state.station_id,
                    trip_index=state.trip_index,
                    x_km=state.x_km,
                    y_km=state.y_km,
                    drove_this_tick=state.vehicle_id in drove_this_tick,
                    charged_this_tick=state.vehicle_id in charged_this_tick,
                    queued_this_tick=state.vehicle_id in queued_this_tick,
                    stranded_this_tick=state.vehicle_id in stranded_this_tick,
                ).model_dump()
            )
        for station in stations:
            station_id = station.station_id
            occupied_ticks += station_occupancy[station_id]
            station_rows.append(
                StationTick(
                    tick=tick,
                    timestamp=timestamp,
                    station_id=station_id,
                    occupancy=station_occupancy[station_id],
                    queue_len=station_queue_lengths[station_id],
                    energy_delivered_kwh=station_energy[station_id],
                ).model_dump()
            )

    available_ticks = sum(station.n_chargers for station in stations) * spec.steps_per_day
    metrics = build_metrics(
        seed=seed,
        energy_cost=energy_cost,
        queued_vehicle_ticks=queued_ticks,
        charge_sessions=charge_sessions,
        soc_violations=soc_violations,
        energy_usage_kwh=energy_usage,
        occupied_charger_ticks=occupied_ticks,
        available_charger_ticks=available_ticks,
        policy_delay_vehicle_ticks=policy_delay_ticks,
        timestep_minutes=spec.timestep_minutes,
    )
    return SimResult(
        stations=pd.DataFrame([station.model_dump() for station in stations]),
        vehicle_ticks=pd.DataFrame(vehicle_rows),
        station_ticks=pd.DataFrame(station_rows),
        metrics=metrics,
    )


def _release_stranded(
    tick: int,
    states: dict[str, VehicleState],
    stranded_until: dict[str, int],
) -> None:
    completed = [
        vehicle_id for vehicle_id, release_tick in stranded_until.items() if tick >= release_tick
    ]
    for vehicle_id in completed:
        states[vehicle_id].status = VehicleStatus.IDLE
        del stranded_until[vehicle_id]


def _release_for_departures(
    tick: int,
    states: dict[str, VehicleState],
    itineraries: dict[str, list[FleetTrip]],
    next_trip: dict[str, int],
    occupancy: dict[str, list[str]],
    queues: dict[str, deque[str]],
) -> None:
    for vehicle_id, state in states.items():
        position = next_trip[vehicle_id]
        itinerary = itineraries.get(vehicle_id, [])
        if position >= len(itinerary) or itinerary[position].departure_tick != tick:
            continue
        if state.status is VehicleStatus.CHARGING and state.station_id is not None:
            occupancy[state.station_id].remove(vehicle_id)
            state.status = VehicleStatus.IDLE
            state.station_id = None
        elif state.status is VehicleStatus.QUEUED and state.station_id is not None:
            queues[state.station_id].remove(vehicle_id)
            state.status = VehicleStatus.IDLE
            state.station_id = None


def _promote_queue(
    station_id: str,
    stations: dict[str, SimStation],
    states: dict[str, VehicleState],
    occupancy: dict[str, list[str]],
    queues: dict[str, deque[str]],
) -> None:
    while queues[station_id] and len(occupancy[station_id]) < stations[station_id].n_chargers:
        vehicle_id = queues[station_id].popleft()
        occupancy[station_id].append(vehicle_id)
        states[vehicle_id].status = VehicleStatus.CHARGING
