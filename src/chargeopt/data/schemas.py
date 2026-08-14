"""Pydantic contracts for ACN-Data-derived tables. See docs/DATASET.md."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChargingSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    site_id: str
    station_id: str
    space_id: str
    start_time: datetime
    end_time: datetime
    done_charging_time: datetime | None = None
    duration_min: float = Field(ge=0)
    energy_kwh: float = Field(ge=0)
    day_of_week: int = Field(ge=0, le=6)
    hour: int = Field(ge=0, le=23)

    @model_validator(mode="after")
    def end_after_start(self) -> ChargingSession:
        if self.end_time < self.start_time:
            msg = "end_time must be >= start_time"
            raise ValueError(msg)
        if self.done_charging_time is not None and self.done_charging_time < self.start_time:
            msg = "done_charging_time must be >= start_time"
            raise ValueError(msg)
        return self


class Station(BaseModel):
    """EVSE identity from ACN-Data. No invented lat/lon."""

    model_config = ConfigDict(extra="forbid")

    station_id: str
    site_id: str
    space_id: str | None = None


class DemandInterval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    site_id: str
    n_arrivals: int = Field(ge=0)
    energy_kwh: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: bool
    month: int = Field(ge=1, le=12)
    lag_15m: float | None = None
    lag_1h: float | None = None
    lag_24h: float | None = None
    rolling_mean_1h: float | None = None
    rolling_mean_24h: float | None = None
    split: Literal["train", "val", "test"]

    @field_validator("lag_15m", "lag_1h", "lag_24h", "rolling_mean_1h", "rolling_mean_24h")
    @classmethod
    def lags_non_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            msg = "lag and rolling features must be >= 0"
            raise ValueError(msg)
        return value
