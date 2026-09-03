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


def _distance(vehicle: VehicleState, station: SimStation) -> float:
    return math.hypot(vehicle.x_km - station.x_km, vehicle.y_km - station.y_km)


def _candidate_stations(
    stations: Sequence[SimStation], occupancy: Mapping[str, int]
) -> list[SimStation]:
    active = [s for s in stations if s.n_chargers > 0]
    if not active:
        raise ValueError("at least one station is required")
    return [s for s in active if occupancy.get(s.station_id, 0) < s.n_chargers] or active


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high):
        return {k: 0.0 for k in values}
    span = high - low
    return {k: (v - low) / span for k, v in values.items()}


class HomeStationChooser:
    """M3 routing: every vehicle returns to its assigned synthetic station."""

    def choose(
        self,
        vehicle: VehicleState,
        stations: Sequence[SimStation],
        *,
        tick: int = 0,
        occupancy: Mapping[str, int] | None = None,
        queues: Mapping[str, tuple[str, ...]] | None = None,
    ) -> str:
        del tick, occupancy, queues
        st_map = {s.station_id: s for s in stations}
        home = st_map.get(vehicle.home_station_id)
        if home is None:
            raise ValueError(f"unknown home station {vehicle.home_station_id!r}")
        if home.n_chargers > 0:
            return home.station_id
        active = [s for s in stations if s.n_chargers > 0]
        if not active:
            raise ValueError("at least one station is required")
        return min(active, key=lambda s: (_distance(vehicle, s), s.station_id)).station_id


class NearestStationChooser:
    """Choose the closest station with a free charger, deterministically."""

    def choose(
        self,
        vehicle: VehicleState,
        stations: Sequence[SimStation],
        *,
        tick: int = 0,
        occupancy: Mapping[str, int] | None = None,
        queues: Mapping[str, tuple[str, ...]] | None = None,
    ) -> str:
        del tick, queues
        return min(
            _candidate_stations(stations, occupancy or {}),
            key=lambda s: (_distance(vehicle, s), s.station_id),
        ).station_id


class CheapestStationChooser:
    """Choose the lowest-price station with deterministic tie-breaking."""

    def choose(
        self,
        vehicle: VehicleState,
        stations: Sequence[SimStation],
        *,
        tick: int = 0,
        occupancy: Mapping[str, int] | None = None,
        queues: Mapping[str, tuple[str, ...]] | None = None,
    ) -> str:
        del vehicle, tick, queues
        return min(
            _candidate_stations(stations, occupancy or {}),
            key=lambda s: (s.price_per_kwh, s.station_id),
        ).station_id


class MLInformedStationChooser:
    """Score stations using current state and a site-level congestion forecast."""

    def __init__(
        self, *, scoring: PolicyScoringConfig, forecast_by_tick: Mapping[int, float]
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
        if tick not in self.forecast_by_tick:
            raise KeyError(f"no demand forecast for simulation tick {tick}")
        kwh = float(self.forecast_by_tick[tick])
        if not math.isfinite(kwh):
            raise ValueError(f"demand forecast for simulation tick {tick} is not finite")
        pressure = min(1.0, max(0.0, kwh) / self.scoring.forecast_scale_kwh)
        norm_d = _normalize({s.station_id: _distance(vehicle, s) for s in candidates})
        norm_p = _normalize({s.station_id: s.price_per_kwh for s in candidates})
        norm_q = _normalize(
            {s.station_id: float(len(queues.get(s.station_id, ()))) for s in candidates}
        )
        q_factor = 1.0 + self.scoring.forecast_weight * pressure

        def score(s: SimStation) -> tuple[float, str]:
            sid = s.station_id
            return (
                self.scoring.distance_weight * norm_d[sid]
                + self.scoring.price_weight * norm_p[sid]
                + self.scoring.queue_weight * norm_q[sid] * q_factor,
                sid,
            )

        return min(candidates, key=score).station_id


class ConcentratedStationChooser:
    """M3 sensitivity probe: every vehicle routes to one station."""

    def choose(
        self,
        vehicle: VehicleState,
        stations: Sequence[SimStation],
        *,
        tick: int = 0,
        occupancy: Mapping[str, int] | None = None,
        queues: Mapping[str, tuple[str, ...]] | None = None,
    ) -> str:
        del vehicle, tick, occupancy, queues
        active = [s for s in stations if s.n_chargers > 0]
        if not active:
            raise ValueError("at least one station is required")
        return min(s.station_id for s in active)


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
            raise ValueError("ml_informed policy requires a demand forecast")
        return MLInformedStationChooser(scoring=scoring, forecast_by_tick=forecast_by_tick)
    raise ValueError(f"unsupported policy {policy!r}")
