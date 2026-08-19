"""Station-selection boundary used by M3 and implemented by M4 policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

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
