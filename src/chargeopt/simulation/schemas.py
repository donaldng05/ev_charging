"""Pydantic contracts for the synthetic fleet simulator. Not ACN EVSE identity."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VehicleStatus(StrEnum):
    IDLE = "idle"
    DRIVING = "driving"
    QUEUED = "queued"
    CHARGING = "charging"
    STRANDED = "stranded"


class SimStation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    station_id: str
    x_km: float
    y_km: float
    n_chargers: int = Field(ge=0)
    power_kw: float = Field(gt=0)
    price_per_kwh: float = Field(ge=0)


class VehicleState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vehicle_id: str
    battery_kwh: float = Field(gt=0)
    soc: float = Field(ge=0, le=1)
    status: VehicleStatus
    home_station_id: str
    station_id: str | None = None
    x_km: float
    y_km: float
    trip_index: int = -1
    remaining_travel_ticks: int = Field(ge=0, default=0)


class FleetTrip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vehicle_id: str
    trip_index: int = Field(ge=0)
    departure_tick: int = Field(ge=0)
    distance_km: float = Field(gt=0)
    duration_ticks: int = Field(ge=1)
    energy_kwh: float = Field(ge=0)


class VehicleTick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tick: int = Field(ge=0)
    timestamp: datetime
    vehicle_id: str
    status: VehicleStatus
    soc: float = Field(ge=0, le=1)
    station_id: str | None = None
    trip_index: int = -1
    x_km: float
    y_km: float
    drove_this_tick: bool
    charged_this_tick: bool
    queued_this_tick: bool
    stranded_this_tick: bool


class StationTick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tick: int = Field(ge=0)
    timestamp: datetime
    station_id: str
    occupancy: int = Field(ge=0)
    queue_len: int = Field(ge=0)
    energy_delivered_kwh: float = Field(ge=0)


class SimMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    energy_cost: float = Field(ge=0)
    avg_wait_minutes: float = Field(ge=0)
    soc_violations: int = Field(ge=0)
    energy_usage_kwh: float = Field(ge=0)
    station_utilization: float = Field(ge=0, le=1)
    vehicle_idle_minutes: float = Field(ge=0)


class CalibrationStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour_pmf: tuple[float, ...]
    mean_duration_min: float = Field(ge=0)
    mean_energy_kwh: float = Field(ge=0)
    peak_concurrent: int = Field(ge=0)
    n_sessions: int = Field(ge=1)

    @field_validator("hour_pmf")
    @classmethod
    def hour_pmf_covers_day(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) != 24:
            msg = "hour_pmf must have 24 hour bins"
            raise ValueError(msg)
        if any(item < 0 for item in value):
            msg = "hour_pmf values must be >= 0"
            raise ValueError(msg)
        total = sum(value)
        if abs(total - 1.0) > 1e-9:
            msg = "hour_pmf must sum to 1"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def peak_within_sessions(self) -> CalibrationStats:
        if self.peak_concurrent > self.n_sessions:
            msg = "peak_concurrent cannot exceed n_sessions"
            raise ValueError(msg)
        return self
