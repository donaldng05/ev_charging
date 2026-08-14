"""Tests for YAML → Pydantic config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from chargeopt.config import (
    AppConfig,
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
    assert set(config.experiment.policies) >= {
        PolicyName.NEAREST,
        PolicyName.CHEAPEST,
        PolicyName.ML_INFORMED,
    }
    assert config.stress.demand_multiplier == 1.5
    assert config.stress.temperature_c == -10.0
    assert config.stress.station_availability == 0.8
    assert config.data.site == "caltech"
    assert config.data.train_fraction + config.data.val_fraction < 1


def test_load_config_from_explicit_path(tmp_path: Path) -> None:
    source = Path("configs/default.yaml").read_text(encoding="utf-8")
    path = tmp_path / "copy.yaml"
    path.write_text(source, encoding="utf-8")
    config = load_config(path)
    assert config.simulation.region == "caltech_hybrid"


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
