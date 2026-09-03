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
    VehicleState,
    VehicleStatus,
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
    sim = config.simulation
    cal = calibrate_from_sessions(sessions, timestep_minutes=sim.timestep_minutes)
    stations = build_stations(sim, cal, seed=seed)
    if station_availability is not None:
        stations = apply_station_availability(
            stations, availability=station_availability, seed=seed
        )
    vehicles = build_fleet(sim, stations)
    trips = build_trips(
        sim,
        config.models.energy,
        cal,
        vehicles,
        seed=seed,
        temperature_c=config.models.energy.temperature_mean_c
        if temperature_c is None
        else temperature_c,
        trip_rate_multiplier=sim.trip_rate_multiplier
        if trip_rate_multiplier is None
        else trip_rate_multiplier,
    )
    return run_world(
        sim,
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
        raise ValueError("vehicle count must match simulation.fleet_size")
    if len(stations) != spec.n_stations:
        raise ValueError("station count must match simulation.n_stations")
    chooser = chooser or HomeStationChooser()
    st_by_id = {s.station_id: s for s in stations}
    if len(st_by_id) != len(stations):
        raise ValueError("station ids must be unique")
    states = {v.vehicle_id: v.model_copy(deep=True) for v in vehicles}
    if len(states) != len(vehicles):
        raise ValueError("vehicle ids must be unique")

    itineraries: dict[str, list[FleetTrip]] = defaultdict(list)
    trip_by_key: dict[tuple[str, int], FleetTrip] = {}
    for t in trips:
        if t.vehicle_id not in states:
            raise ValueError(f"trip references unknown vehicle {t.vehicle_id!r}")
        itineraries[t.vehicle_id].append(t)
        trip_by_key[(t.vehicle_id, t.trip_index)] = t
    for itin in itineraries.values():
        itin.sort(key=lambda t: t.trip_index)

    next_trip = {v_id: 0 for v_id in states}
    occupancy: dict[str, list[str]] = {s.station_id: [] for s in stations}
    queues: dict[str, deque[str]] = {s.station_id: deque() for s in stations}
    vehicle_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    start = datetime.combine(spec.start_day, time.min, tzinfo=ZoneInfo(timezone_name))
    step_delta = timedelta(minutes=spec.timestep_minutes)
    energy_cost = energy_usage = 0.0
    soc_violations = queued_ticks = policy_delay_ticks = occupied_ticks = charge_sessions = 0
    stranded_until: dict[str, int] = {}

    for tick in range(spec.steps_per_day):
        timestamp = start + tick * step_delta

        # 1. Release stranded vehicles whose trip duration has elapsed
        for vid in [v for v, rel in stranded_until.items() if tick >= rel]:
            states[vid].status = VehicleStatus.IDLE
            del stranded_until[vid]

        # 2. Release vehicles that must depart for their next trip
        for vid, state in states.items():
            itin = itineraries.get(vid, [])
            pos = next_trip[vid]
            if (
                pos < len(itin)
                and itin[pos].departure_tick == tick
                and state.station_id is not None
            ):
                if state.status is VehicleStatus.CHARGING and vid in occupancy[state.station_id]:
                    occupancy[state.station_id].remove(vid)
                elif state.status is VehicleStatus.QUEUED and vid in queues[state.station_id]:
                    queues[state.station_id].remove(vid)
                state.status, state.station_id = VehicleStatus.IDLE, None

        # 3. Promote queued vehicles into newly freed chargers
        for sid in st_by_id:
            _promote_queue(sid, st_by_id, states, occupancy, queues)

        # 4. Departures & Driving initialization
        for vid, state in states.items():
            itin = itineraries.get(vid, [])
            pos = next_trip[vid]
            if (
                pos < len(itin)
                and itin[pos].departure_tick == tick
                and state.status is VehicleStatus.IDLE
            ):
                trip = itin[pos]
                next_trip[vid] += 1
                avail_kwh = (state.soc - spec.soc_min) * state.battery_kwh
                if avail_kwh + 1e-12 < trip.energy_kwh:
                    soc_violations += 1
                    state.trip_index, state.status = trip.trip_index, VehicleStatus.STRANDED
                    stranded_until[vid] = tick + trip.duration_ticks
                else:
                    state.status, state.station_id = VehicleStatus.DRIVING, None
                    state.trip_index, state.remaining_travel_ticks = (
                        trip.trip_index,
                        trip.duration_ticks,
                    )

        # 5. Driving progression & Arrival at charging stations
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
            energy_usage += min(consumed, state.soc * state.battery_kwh)
            state.soc = max(0.0, next_soc)
            state.remaining_travel_ticks -= 1
            if state.remaining_travel_ticks == 0:
                sid = chooser.choose(
                    state,
                    stations,
                    tick=tick,
                    occupancy={k: len(v) for k, v in occupancy.items()},
                    queues={k: tuple(v) for k, v in queues.items()},
                )
                if sid not in st_by_id:
                    raise ValueError(f"chooser returned unknown station {sid!r}")
                target_st = st_by_id[sid]
                state.station_id, state.x_km, state.y_km = sid, target_st.x_km, target_st.y_km
                charge_sessions += 1
                if len(occupancy[sid]) < target_st.n_chargers:
                    occupancy[sid].append(state.vehicle_id)
                    state.status = VehicleStatus.CHARGING
                else:
                    queues[sid].append(state.vehicle_id)
                    state.status = VehicleStatus.QUEUED

        # Snapshot occupancy, queues, and activity states before charging energy delivery
        station_energy = {s.station_id: 0.0 for s in stations}
        st_occupancy = {s.station_id: len(occupancy[s.station_id]) for s in stations}
        st_queues = {s.station_id: len(queues[s.station_id]) for s in stations}
        charged_this_tick = {v for occ in occupancy.values() for v in occ}
        queued_this_tick = {v for q in queues.values() for v in q}
        stranded_this_tick = {v for v, st in states.items() if st.status is VehicleStatus.STRANDED}
        queued_ticks += len(queued_this_tick)
        policy_delay_ticks += len(queued_this_tick | stranded_this_tick)

        # 6. Station energy delivery & Charging progression
        for s in stations:
            sid = s.station_id
            for vid in list(occupancy[sid]):
                v_state = states[vid]
                cap_left = (spec.soc_charge_target - v_state.soc) * v_state.battery_kwh
                delivered = min(s.power_kw * spec.timestep_minutes / 60.0, max(0.0, cap_left))
                v_state.soc = min(1.0, v_state.soc + delivered / v_state.battery_kwh)
                station_energy[sid] += delivered
                energy_cost += delivered * s.price_per_kwh
                if v_state.soc + 1e-12 >= spec.soc_charge_target:
                    occupancy[sid].remove(vid)
                    v_state.status, v_state.station_id = VehicleStatus.IDLE, None
            _promote_queue(sid, st_by_id, states, occupancy, queues)

        # 7. Collect tick rows and state counters
        for v_state in states.values():
            vehicle_rows.append(
                {
                    "tick": tick,
                    "timestamp": timestamp,
                    "vehicle_id": v_state.vehicle_id,
                    "status": v_state.status.value,
                    "soc": v_state.soc,
                    "station_id": v_state.station_id,
                    "trip_index": v_state.trip_index,
                    "x_km": v_state.x_km,
                    "y_km": v_state.y_km,
                    "drove_this_tick": v_state.vehicle_id in drove_this_tick,
                    "charged_this_tick": v_state.vehicle_id in charged_this_tick,
                    "queued_this_tick": v_state.vehicle_id in queued_this_tick,
                    "stranded_this_tick": v_state.vehicle_id in stranded_this_tick,
                }
            )
        for s in stations:
            sid = s.station_id
            occupied_ticks += st_occupancy[sid]
            station_rows.append(
                {
                    "tick": tick,
                    "timestamp": timestamp,
                    "station_id": sid,
                    "occupancy": st_occupancy[sid],
                    "queue_len": st_queues[sid],
                    "energy_delivered_kwh": station_energy[sid],
                }
            )

    metrics = build_metrics(
        seed=seed,
        energy_cost=energy_cost,
        queued_vehicle_ticks=queued_ticks,
        charge_sessions=charge_sessions,
        soc_violations=soc_violations,
        energy_usage_kwh=energy_usage,
        occupied_charger_ticks=occupied_ticks,
        available_charger_ticks=sum(s.n_chargers for s in stations) * spec.steps_per_day,
        policy_delay_vehicle_ticks=policy_delay_ticks,
        timestep_minutes=spec.timestep_minutes,
    )
    return SimResult(
        stations=pd.DataFrame([s.model_dump() for s in stations]),
        vehicle_ticks=pd.DataFrame(vehicle_rows),
        station_ticks=pd.DataFrame(station_rows),
        metrics=metrics,
    )


def _promote_queue(
    sid: str,
    stations: dict[str, SimStation],
    states: dict[str, VehicleState],
    occupancy: dict[str, list[str]],
    queues: dict[str, deque[str]],
) -> None:
    while queues[sid] and len(occupancy[sid]) < stations[sid].n_chargers:
        vid = queues[sid].popleft()
        occupancy[sid].append(vid)
        states[vid].status = VehicleStatus.CHARGING
