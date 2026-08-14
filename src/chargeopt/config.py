"""Load and validate experiment configuration.

MVP numbers live in YAML. Code must not hardcode fleet size, timestep, or horizon.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PolicyName(StrEnum):
    NEAREST = "nearest"
    CHEAPEST = "cheapest"
    ML_INFORMED = "ml_informed"


class MetricName(StrEnum):
    ENERGY_COST = "energy_cost"
    AVG_WAIT_MINUTES = "avg_wait_minutes"
    SOC_VIOLATIONS = "soc_violations"
    ENERGY_USAGE_KWH = "energy_usage_kwh"
    STATION_UTILIZATION = "station_utilization"
    VEHICLE_IDLE_MINUTES = "vehicle_idle_minutes"


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str
    fleet_size: int = Field(ge=1)
    n_stations: int = Field(ge=1)
    timestep_minutes: int
    horizon_hours: int = Field(ge=1)
    charger_power_kw: float = Field(gt=0)

    @field_validator("timestep_minutes")
    @classmethod
    def timestep_must_divide_hour(cls, value: int) -> int:
        if value not in (15, 30):
            msg = "timestep_minutes must be 15 or 30 for MVP"
            raise ValueError(msg)
        return value

    @property
    def steps_per_day(self) -> int:
        return (self.horizon_hours * 60) // self.timestep_minutes


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policies: list[PolicyName] = Field(min_length=1)
    metrics: list[MetricName] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)

    @field_validator("policies")
    @classmethod
    def policies_unique(cls, value: list[PolicyName]) -> list[PolicyName]:
        if len(set(value)) != len(value):
            msg = "experiment.policies must be unique"
            raise ValueError(msg)
        return value

    @field_validator("metrics")
    @classmethod
    def metrics_unique(cls, value: list[MetricName]) -> list[MetricName]:
        if len(set(value)) != len(value):
            msg = "experiment.metrics must be unique"
            raise ValueError(msg)
        return value

    @field_validator("seeds")
    @classmethod
    def seeds_unique(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            msg = "experiment.seeds must be unique"
            raise ValueError(msg)
        return value


class StressConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    demand_multiplier: float = Field(gt=0)
    temperature_c: float
    station_availability: float = Field(gt=0, le=1)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            msg = f"logging.level must be one of {sorted(allowed)}"
            raise ValueError(msg)
        return level


class AppConfig(BaseModel):
    """Frozen experiment contract loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    simulation: SimulationConfig
    experiment: ExperimentConfig
    stress: StressConfig
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def require_mvp_policies(self) -> Self:
        required = {PolicyName.NEAREST, PolicyName.CHEAPEST, PolicyName.ML_INFORMED}
        missing = required - set(self.experiment.policies)
        if missing:
            names = ", ".join(sorted(p.value for p in missing))
            msg = f"experiment.policies must include MVP policies; missing: {names}"
            raise ValueError(msg)
        return self


class RuntimeSettings(BaseSettings):
    """Process-level overrides. Experiment numbers still come from YAML."""

    model_config = SettingsConfigDict(env_prefix="CHARGEOPT_", extra="ignore")

    config_path: Path = Path("configs/default.yaml")
    log_level: str | None = None


def default_config_path() -> Path:
    """Repo-root `configs/default.yaml` when running from a checkout."""
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def resolve_config_path(path: Path | None = None) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            msg = f"config file not found: {resolved}"
            raise FileNotFoundError(msg)
        return resolved

    cwd_candidate = Path("configs/default.yaml")
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()

    packaged = default_config_path()
    if packaged.is_file():
        return packaged

    msg = "config file not found: configs/default.yaml"
    raise FileNotFoundError(msg)


def load_config(path: Path | str | None = None) -> AppConfig:
    config_path = resolve_config_path(Path(path) if path is not None else None)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"config root must be a mapping: {config_path}"
        raise ValueError(msg)
    return AppConfig.model_validate(raw)
