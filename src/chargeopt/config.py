"""Load and validate experiment configuration.

MVP numbers live in YAML. Code must not hardcode fleet size, timestep, or horizon.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
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
    battery_kwh: float = Field(gt=0)
    soc_initial: float = Field(ge=0, le=1)
    soc_min: float = Field(ge=0, le=1)
    soc_charge_target: float = Field(ge=0, le=1)
    trips_per_vehicle: int = Field(ge=1)
    trip_rate_multiplier: float = Field(gt=0)
    metro_span_km: float = Field(gt=0)
    price_per_kwh_min: float = Field(ge=0)
    price_per_kwh_max: float = Field(ge=0)
    n_chargers_min: int = Field(ge=1)
    start_day: date
    stations_path: Path
    run_path: Path
    station_ticks_path: Path
    metrics_path: Path

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if self.timestep_minutes not in (15, 30):
            raise ValueError("timestep_minutes must be 15 or 30 for MVP")
        if not math.isfinite(self.trip_rate_multiplier):
            raise ValueError("trip_rate_multiplier must be finite")
        if self.soc_min >= self.soc_charge_target:
            raise ValueError("soc_min must be below soc_charge_target")
        if self.price_per_kwh_min > self.price_per_kwh_max:
            raise ValueError("price_per_kwh_min must be <= price_per_kwh_max")
        return self

    @property
    def steps_per_day(self) -> int:
        return (self.horizon_hours * 60) // self.timestep_minutes


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policies: list[PolicyName] = Field(min_length=1)
    metrics: list[MetricName] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)

    @field_validator("policies", "metrics", "seeds")
    @classmethod
    def list_items_unique(cls, value: list[Any], info: ValidationInfo) -> list[Any]:
        if len(set(value)) != len(value):
            raise ValueError(f"experiment.{info.field_name} must be unique")
        return value


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_results_path: Path
    summary_path: Path
    robustness_path: Path
    metadata_path: Path
    confidence_level: float = Field(gt=0, lt=1)


class StressConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    demand_multiplier: float = Field(gt=0)
    temperature_c: float
    station_availability: float = Field(gt=0, le=1)


class PolicyScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    distance_weight: float = Field(ge=0)
    price_weight: float = Field(ge=0)
    queue_weight: float = Field(ge=0)
    forecast_weight: float = Field(ge=0)
    forecast_scale_kwh: float = Field(gt=0)

    @model_validator(mode="after")
    def require_nonzero_score(self) -> Self:
        if self.distance_weight + self.price_weight + self.queue_weight <= 0:
            raise ValueError("at least one station score weight must be positive")
        return self


LEARNER_NAMES: tuple[str, ...] = (
    "random_forest",
    "ridge",
    "elasticnet",
    "extra_trees",
    "hist_gradient_boosting",
)


class TreeSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: list[int] = Field(min_length=1)
    max_depth: list[int] = Field(min_length=1)
    min_samples_leaf: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def check_positive(self) -> Self:
        for f in ("n_estimators", "max_depth", "min_samples_leaf"):
            if any(x < 1 for x in getattr(self, f)):
                raise ValueError("search grid values must be >= 1")
        return self


class TreeLearnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    min_samples_leaf: int = Field(ge=1)
    search: TreeSearch


class RidgeSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: list[float] = Field(min_length=1)

    @field_validator("alpha")
    @classmethod
    def alpha_positive(cls, v: list[float]) -> list[float]:
        if any(x <= 0 for x in v):
            raise ValueError("search grid values must be > 0")
        return v


class RidgeLearnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(gt=0)
    search: RidgeSearch


class ElasticNetSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: list[float] = Field(min_length=1)
    l1_ratio: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def check_ranges(self) -> Self:
        if any(x <= 0 for x in self.alpha):
            raise ValueError("search grid values must be > 0")
        if any(x < 0 or x > 1 for x in self.l1_ratio):
            raise ValueError("l1_ratio search values must be in [0, 1]")
        return self


class ElasticNetLearnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(gt=0)
    l1_ratio: float = Field(ge=0, le=1)
    search: ElasticNetSearch


class HistGradientBoostingSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_iter: list[int] = Field(min_length=1)
    max_depth: list[int] = Field(min_length=1)
    learning_rate: list[float] = Field(min_length=1)
    min_samples_leaf: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def check_ranges(self) -> Self:
        for f in ("max_iter", "max_depth", "min_samples_leaf"):
            if any(x < 1 for x in getattr(self, f)):
                raise ValueError("search grid values must be >= 1")
        if any(x <= 0 for x in self.learning_rate):
            raise ValueError("search grid values must be > 0")
        return self


class HistGradientBoostingLearnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_iter: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    min_samples_leaf: int = Field(ge=1)
    search: HistGradientBoostingSearch


class LearnerSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    random_forest: TreeLearnerConfig
    ridge: RidgeLearnerConfig
    elasticnet: ElasticNetLearnerConfig
    extra_trees: TreeLearnerConfig
    hist_gradient_boosting: HistGradientBoostingLearnerConfig

    def params_for(self, name: str) -> dict[str, Any]:
        p = self.learner(name).model_dump()
        p.pop("search")
        return p

    def search_for(self, name: str) -> dict[str, list[Any]]:
        return self.learner(name).search.model_dump()

    def learner(
        self, name: str
    ) -> (
        TreeLearnerConfig
        | RidgeLearnerConfig
        | ElasticNetLearnerConfig
        | HistGradientBoostingLearnerConfig
    ):
        if name in LEARNER_NAMES:
            return cast(
                TreeLearnerConfig
                | RidgeLearnerConfig
                | ElasticNetLearnerConfig
                | HistGradientBoostingLearnerConfig,
                getattr(self, name),
            )
        raise ValueError(f"unknown learner {name!r}; expected one of: {', '.join(LEARNER_NAMES)}")


class DemandModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon_minutes: int = Field(gt=0)
    n_splits: int = Field(ge=2)
    decision_model: str
    learners: LearnerSuite
    predictions_path: Path
    metrics_path: Path
    tune_metrics_path: Path
    error_slices_path: Path

    @field_validator("decision_model")
    @classmethod
    def decision_model_is_learner(cls, value: str) -> str:
        if value not in LEARNER_NAMES:
            raise ValueError(
                f"models.demand.decision_model must be one of: {', '.join(LEARNER_NAMES)}"
            )
        return value


class EnergyModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_trips: int = Field(ge=1)
    rate_kwh_per_km: float = Field(gt=0)
    distance_km_mean: float = Field(gt=0)
    distance_km_std: float = Field(gt=0)
    duration_min_mean: float = Field(gt=0)
    duration_min_std: float = Field(gt=0)
    temperature_mean_c: float
    temperature_std_c: float = Field(gt=0)
    temperature_reference_c: float
    cold_penalty_per_c: float = Field(ge=0)
    noise_std_kwh: float = Field(ge=0)
    learners: LearnerSuite
    cold_holdout_n_trips: int = Field(ge=1)
    trips_path: Path
    predictions_path: Path
    metrics_path: Path
    tune_metrics_path: Path
    cold_metrics_path: Path


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    demand: DemandModelConfig
    energy: EnergyModelConfig


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site: str
    timezone: str
    start: datetime
    end: datetime
    snapshot_path: Path
    processed_path: Path
    train_fraction: float = Field(gt=0, lt=1)
    val_fraction: float = Field(gt=0, lt=1)
    covid_start: datetime
    covid_end: datetime

    @model_validator(mode="after")
    def validate_data(self) -> Self:
        allowed = {"caltech", "jpl", "office001"}
        if self.site not in allowed:
            raise ValueError(f"data.site must be one of {sorted(allowed)}")
        if self.end <= self.start:
            raise ValueError("data.end must be after data.start")
        if self.covid_end <= self.covid_start:
            raise ValueError("data.covid_end must be after data.covid_start")
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("data.train_fraction + data.val_fraction must be < 1")
        return self


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"logging.level must be one of {sorted(allowed)}")
        return level


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    simulation: SimulationConfig
    data: DataConfig
    experiment: ExperimentConfig
    evaluation: EvaluationConfig
    stress: StressConfig
    optimization: PolicyScoringConfig
    models: ModelsConfig
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        required = {PolicyName.NEAREST, PolicyName.CHEAPEST, PolicyName.ML_INFORMED}
        missing = required - set(self.experiment.policies)
        if missing:
            names = ", ".join(sorted(p.value for p in missing))
            raise ValueError(f"experiment.policies must include MVP policies; missing: {names}")
        if self.models.demand.horizon_minutes % self.simulation.timestep_minutes != 0:
            raise ValueError(
                "models.demand.horizon_minutes must be divisible by simulation.timestep_minutes"
            )
        return self


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHARGEOPT_", extra="ignore")
    config_path: Path = Path("configs/default.yaml")
    log_level: str | None = None
    acn_token: str = "DEMO_TOKEN"


def default_config_path() -> Path:
    return project_root() / "configs" / "default.yaml"


def resolve_config_path(path: Path | None = None) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"config file not found: {resolved}")
        return resolved
    for p in (Path("configs/default.yaml"), default_config_path()):
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError("config file not found: configs/default.yaml")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_path(path: Path) -> Path:
    return path if path.is_absolute() else project_root() / path


def load_config(path: Path | str | None = None) -> AppConfig:
    config_path = resolve_config_path(Path(path) if path is not None else None)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {config_path}")
    return AppConfig.model_validate(raw)
