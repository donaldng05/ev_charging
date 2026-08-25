"""Tests for YAML → Pydantic config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from chargeopt.config import (
    LEARNER_NAMES,
    AppConfig,
    LearnerSuite,
    PolicyName,
    default_config_path,
    load_config,
    resolve_config_path,
)


def test_default_config_round_trip() -> None:
    config = load_config()
    assert config.simulation.fleet_size == 30
    assert config.simulation.n_stations == 10
    assert config.simulation.timestep_minutes == 15
    assert config.simulation.horizon_hours == 24
    assert config.simulation.steps_per_day == 96
    assert config.simulation.charger_power_kw == 150.0
    assert config.simulation.battery_kwh == 60.0
    assert config.simulation.soc_initial == 0.7
    assert config.simulation.soc_min == 0.1
    assert config.simulation.soc_charge_target == 0.9
    assert config.simulation.trips_per_vehicle == 2
    assert config.simulation.trip_rate_multiplier == 1.0
    assert config.simulation.metro_span_km == 12.0
    assert config.simulation.price_per_kwh_min == 0.20
    assert config.simulation.price_per_kwh_max == 0.45
    assert config.simulation.n_chargers_min == 1
    assert config.simulation.start_day.isoformat() == "2018-09-05"
    assert config.simulation.stations_path.as_posix() == "data/processed/sim_stations.csv"
    assert config.simulation.run_path.as_posix() == "data/processed/sim_run.csv"
    assert config.simulation.station_ticks_path.as_posix() == "data/processed/sim_station_ticks.csv"
    assert config.simulation.metrics_path.as_posix() == "data/processed/sim_metrics.csv"
    assert set(config.experiment.policies) >= {
        PolicyName.NEAREST,
        PolicyName.CHEAPEST,
        PolicyName.ML_INFORMED,
    }
    assert config.stress.demand_multiplier == 1.5
    assert config.stress.temperature_c == -10.0
    assert config.stress.station_availability == 0.8
    assert config.optimization.distance_weight == 1.0
    assert config.optimization.price_weight == 1.0
    assert config.optimization.queue_weight == 1.0
    assert config.optimization.forecast_weight == 1.0
    assert config.optimization.forecast_scale_kwh == 10.0
    assert config.data.site == "caltech"
    assert config.data.train_fraction + config.data.val_fraction < 1
    assert config.models.demand.horizon_minutes == 60
    assert config.models.demand.decision_model == "random_forest"
    assert config.models.demand.learners.random_forest.n_estimators == 200
    assert config.models.demand.learners.random_forest.max_depth == 12
    assert config.models.demand.learners.random_forest.min_samples_leaf == 8
    assert config.models.demand.n_splits == 4
    assert config.models.demand.learners.ridge.alpha == 0.1
    assert config.models.demand.learners.elasticnet.l1_ratio == 0.8
    assert config.models.demand.learners.extra_trees.n_estimators == 100
    assert config.models.demand.learners.hist_gradient_boosting.max_iter == 100
    assert config.data.start.isoformat() == "2018-05-01T00:00:00"
    assert config.data.end.isoformat() == "2026-08-17T00:00:00"
    assert config.data.covid_start.isoformat() == "2020-03-01T00:00:00"
    assert config.data.covid_end.isoformat() == "2021-09-01T00:00:00"
    assert config.models.energy.n_trips == 2000
    assert config.models.energy.rate_kwh_per_km == 0.18
    assert config.models.energy.cold_penalty_per_c == 0.01
    assert config.models.energy.noise_std_kwh == 0.05


def test_load_config_from_explicit_path(tmp_path: Path) -> None:
    source = Path("configs/default.yaml").read_text(encoding="utf-8")
    path = tmp_path / "copy.yaml"
    path.write_text(source, encoding="utf-8")
    config = load_config(path)
    assert config.simulation.region == "caltech_hybrid"


def test_congestion_profile_loads_with_separate_artifacts() -> None:
    config = load_config(Path("configs/congestion.yaml"))

    assert config.simulation.trip_rate_multiplier == 7.0
    assert config.simulation.region == "caltech_hybrid_congestion"
    assert config.simulation.run_path.as_posix() == "data/processed/congestion_sim_run.csv"
    assert config.simulation.metrics_path.as_posix() == "data/processed/congestion_sim_metrics.csv"


def test_missing_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("simulation:\n  region: x\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_unknown_key_fails(tmp_path: Path) -> None:
    config = load_config()
    payload = config.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_invalid_timestep_fails() -> None:
    config = load_config()
    payload = config.model_dump(mode="json")
    payload["simulation"]["timestep_minutes"] = 5
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_trip_rate_multiplier_must_be_positive_and_finite() -> None:
    payload = load_config().model_dump(mode="json")
    payload["simulation"]["trip_rate_multiplier"] = 0.0
    with pytest.raises(ValidationError, match="trip_rate_multiplier"):
        AppConfig.model_validate(payload)

    payload["simulation"]["trip_rate_multiplier"] = float("inf")
    with pytest.raises(ValidationError, match="finite"):
        AppConfig.model_validate(payload)


def test_duplicate_seeds_fail() -> None:
    config = load_config()
    payload = config.model_dump(mode="json")
    payload["experiment"]["seeds"] = [1, 1]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_missing_mvp_policy_fails() -> None:
    config = load_config()
    payload = config.model_dump(mode="json")
    payload["experiment"]["policies"] = ["nearest", "cheapest"]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_resolve_config_path_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_config_path(tmp_path / "nope.yaml")


def test_non_mapping_config_fails(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_invalid_log_level_fails() -> None:
    payload = load_config().model_dump(mode="json")
    payload["logging"]["level"] = "verbose"
    with pytest.raises(ValidationError, match=r"logging\.level"):
        AppConfig.model_validate(payload)


def test_duplicate_policies_fail() -> None:
    payload = load_config().model_dump(mode="json")
    payload["experiment"]["policies"] = ["nearest", "nearest", "cheapest", "ml_informed"]
    with pytest.raises(ValidationError, match="unique"):
        AppConfig.model_validate(payload)


def test_duplicate_metrics_fail() -> None:
    payload = load_config().model_dump(mode="json")
    payload["experiment"]["metrics"] = ["energy_cost", "energy_cost"]
    with pytest.raises(ValidationError, match="unique"):
        AppConfig.model_validate(payload)


def test_default_config_path_exists() -> None:
    assert default_config_path().is_file()


def test_resolve_falls_back_to_repo_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path(None) == default_config_path()


def test_demand_horizon_must_divide_timestep() -> None:
    payload = load_config().model_dump(mode="json")
    payload["models"]["demand"]["horizon_minutes"] = 50
    with pytest.raises(ValidationError, match="horizon_minutes"):
        AppConfig.model_validate(payload)


def test_unknown_learner_key_fails() -> None:
    payload = load_config().model_dump(mode="json")
    payload["models"]["demand"]["learners"]["svm"] = {"alpha": 1.0, "search": {"alpha": [1.0]}}
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_soc_min_must_be_below_charge_target() -> None:
    payload = load_config().model_dump(mode="json")
    payload["simulation"]["soc_min"] = 0.95
    payload["simulation"]["soc_charge_target"] = 0.9
    with pytest.raises(ValidationError, match="soc_min"):
        AppConfig.model_validate(payload)


def test_price_min_must_not_exceed_max() -> None:
    payload = load_config().model_dump(mode="json")
    payload["simulation"]["price_per_kwh_min"] = 0.50
    payload["simulation"]["price_per_kwh_max"] = 0.20
    with pytest.raises(ValidationError, match="price_per_kwh"):
        AppConfig.model_validate(payload)


def test_policy_scoring_requires_a_positive_station_weight() -> None:
    payload = load_config().model_dump(mode="json")
    payload["optimization"]["distance_weight"] = 0.0
    payload["optimization"]["price_weight"] = 0.0
    payload["optimization"]["queue_weight"] = 0.0
    with pytest.raises(ValidationError, match="score weight"):
        AppConfig.model_validate(payload)


def test_policy_forecast_scale_must_be_positive() -> None:
    payload = load_config().model_dump(mode="json")
    payload["optimization"]["forecast_scale_kwh"] = 0.0
    with pytest.raises(ValidationError, match="forecast_scale_kwh"):
        AppConfig.model_validate(payload)


def test_invalid_decision_model_fails() -> None:
    payload = load_config().model_dump(mode="json")
    payload["models"]["demand"]["decision_model"] = "last_observation"
    with pytest.raises(ValidationError, match="decision_model"):
        AppConfig.model_validate(payload)


def test_learner_suite_names_match_config() -> None:
    config = load_config()
    assert tuple(LearnerSuite.model_fields) == LEARNER_NAMES
    assert set(config.models.demand.learners.params_for("ridge")) == {"alpha"}
    assert set(config.models.energy.learners.params_for("elasticnet")) == {"alpha", "l1_ratio"}
