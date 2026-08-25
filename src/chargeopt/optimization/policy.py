"""Station-selection contracts and deterministic policy implementations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from chargeopt.config import PolicyName, PolicyScoringConfig

if TYPE_CHECKING:
    from chargeopt.simulation.schemas import SimStation, VehicleState


class StationChooser(Protocol):
    def choose(
        self,
        vehicle: VehicleState,
        stations: Sequence[SimStation],
        *,
        tick: int,
        occupancy: Mapping[str, int],
        queues: Mapping[str, tuple[str, ...]],
    ) -> str:
        """Return one station id for an arriving vehicle."""
        ...


class HomeStationChooser:
    """M3 routing: every vehicle returns to its assigned synthetic station."""

    def choose(
        self,
        vehicle: VehicleState,
        stations: Sequence[SimStation],
        *,
        tick: int,
        occupancy: Mapping[str, int],
        queues: Mapping[str, tuple[str, ...]],
    ) -> str:
        del tick, occupancy, queues
        station_ids = {station.station_id for station in stations}
        if vehicle.home_station_id not in station_ids:
            msg = f"unknown home station {vehicle.home_station_id!r}"
            raise ValueError(msg)
        return vehicle.home_station_id


def _candidate_stations(
    stations: Sequence[SimStation],
    occupancy: Mapping[str, int],
) -> list[SimStation]:
    if not stations:
        msg = "at least one station is required"
        raise ValueError(msg)
    available = [
        station for station in stations if occupancy.get(station.station_id, 0) < station.n_chargers
    ]
    return available or list(stations)


def _distance(vehicle: VehicleState, station: SimStation) -> float:
    return math.hypot(vehicle.x_km - station.x_km, vehicle.y_km - station.y_km)


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high):
        return {key: 0.0 for key in values}
    span = high - low
    return {key: (value - low) / span for key, value in values.items()}


class NearestStationChooser:
    """Choose the closest station with a free charger, deterministically."""

    def choose(
        self,
        vehicle: VehicleState,
        stations: Sequence[SimStation],
        *,
        tick: int,
        occupancy: Mapping[str, int],
        queues: Mapping[str, tuple[str, ...]],
    ) -> str:
        del tick, queues
        candidates = _candidate_stations(stations, occupancy)
        return min(
            candidates,
            key=lambda station: (_distance(vehicle, station), station.station_id),
        ).station_id


class CheapestStationChooser:
    """Choose the lowest-price station with deterministic tie-breaking."""

    def choose(
        self,
        vehicle: VehicleState,
        stations: Sequence[SimStation],
        *,
        tick: int,
        occupancy: Mapping[str, int],
        queues: Mapping[str, tuple[str, ...]],
    ) -> str:
        del vehicle, tick, queues
        candidates = _candidate_stations(stations, occupancy)
        return min(
            candidates,
            key=lambda station: (station.price_per_kwh, station.station_id),
        ).station_id


class MLInformedStationChooser:
    """Score stations using current state and a site-level congestion forecast."""

    def __init__(
        self,
        *,
        scoring: PolicyScoringConfig,
        forecast_by_tick: Mapping[int, float],
    ) -> None:
        self.scoring = scoring
        self.forecast_by_tick = forecast_by_tick

    def choose(
        self,
        vehicle: VehicleState,
        stations: Sequence[SimStation],
        *,
        tick: int,
        occupancy: Mapping[str, int],
        queues: Mapping[str, tuple[str, ...]],
    ) -> str:
        candidates = _candidate_stations(stations, occupancy)
        try:
            predicted_kwh = float(self.forecast_by_tick[tick])
        except KeyError as exc:
            msg = f"no demand forecast for simulation tick {tick}"
            raise KeyError(msg) from exc
        if not math.isfinite(predicted_kwh):
            msg = f"demand forecast for simulation tick {tick} is not finite"
            raise ValueError(msg)
        forecast_pressure = min(
            1.0,
            max(0.0, predicted_kwh) / self.scoring.forecast_scale_kwh,
        )
        distances = {station.station_id: _distance(vehicle, station) for station in candidates}
        prices = {station.station_id: station.price_per_kwh for station in candidates}
        queue_lengths = {
            station.station_id: float(len(queues.get(station.station_id, ())))
            for station in candidates
        }
        normalized_distance = _normalize(distances)
        normalized_price = _normalize(prices)
        normalized_queue = _normalize(queue_lengths)
        queue_factor = 1.0 + self.scoring.forecast_weight * forecast_pressure

        def score(station: SimStation) -> tuple[float, str]:
            station_id = station.station_id
            value = (
                self.scoring.distance_weight * normalized_distance[station_id]
                + self.scoring.price_weight * normalized_price[station_id]
                + self.scoring.queue_weight * normalized_queue[station_id] * queue_factor
            )
            return value, station_id

        return min(candidates, key=score).station_id


class ConcentratedStationChooser:
    """M3 sensitivity probe: every vehicle routes to one station."""

    def choose(
        self,
        vehicle: VehicleState,
        stations: Sequence[SimStation],
        *,
        tick: int,
        occupancy: Mapping[str, int],
        queues: Mapping[str, tuple[str, ...]],
    ) -> str:
        del vehicle, tick, occupancy, queues
        return min(station.station_id for station in stations)


def build_station_chooser(
    policy: PolicyName,
    *,
    scoring: PolicyScoringConfig,
    forecast_by_tick: Mapping[int, float] | None = None,
) -> StationChooser:
    """Construct one deterministic chooser for the selected M4 policy."""
    if policy is PolicyName.NEAREST:
        return NearestStationChooser()
    if policy is PolicyName.CHEAPEST:
        return CheapestStationChooser()
    if policy is PolicyName.ML_INFORMED:
        if forecast_by_tick is None:
            msg = "ml_informed policy requires a demand forecast"
            raise ValueError(msg)
        return MLInformedStationChooser(
            scoring=scoring,
            forecast_by_tick=forecast_by_tick,
        )
    msg = f"unsupported policy {policy!r}"
    raise ValueError(msg)
