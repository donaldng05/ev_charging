"""CLI smoke tests."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest

from chargeopt.cli import main


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "chargeopt" in captured.out
    assert "experiment" in captured.out
    assert "models" in captured.out


def test_experiment_resolves_default_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["experiment", "--policy", "ml", "--seed", "42"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "not_implemented"
    assert payload["policy"] == "ml_informed"
    assert payload["seed"] == 42
    assert payload["config"]["simulation"]["fleet_size"] == 30
    assert "experiment_id" in payload


def test_data_pull_writes_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import datetime

    raw = {
        "sessionID": "s-cli",
        "stationID": "2-39-79-383",
        "spaceID": "CA-492",
        "connectionTime": datetime(2018, 9, 5, 15, 0, tzinfo=UTC),
        "disconnectTime": datetime(2018, 9, 5, 17, 0, tzinfo=UTC),
        "doneChargingTime": datetime(2018, 9, 5, 16, 0, tzinfo=UTC),
        "kWhDelivered": 5.0,
    }

    def fake_iter(**kwargs: object) -> list[dict[str, object]]:
        return [raw]

    monkeypatch.setattr("chargeopt.cli.iter_acn_sessions", fake_iter)
    source = Path("configs/default.yaml").read_text(encoding="utf-8")
    yaml_path = tmp_path / "cfg.yaml"
    snapshot = tmp_path / "sessions.csv"
    patched = source.replace(
        "snapshot_path: data/raw/acn_caltech_sessions.csv",
        f"snapshot_path: {snapshot.as_posix()}",
    )
    yaml_path.write_text(patched, encoding="utf-8")
    assert main(["data", "pull", "--config", str(yaml_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_sessions"] == 1
    assert snapshot.is_file()


def test_unknown_policy_fails() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["experiment", "--policy", "rl"])
    assert excinfo.value.code != 0


def test_data_features_from_fixture(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = Path("configs/default.yaml").read_text(encoding="utf-8")
    yaml_path = tmp_path / "cfg.yaml"
    processed = tmp_path / "demand.csv"
    snapshot = Path("tests/fixtures/acn_sessions.csv").resolve()
    patched = source.replace(
        "snapshot_path: data/raw/acn_caltech_sessions.csv",
        f"snapshot_path: {snapshot.as_posix()}",
    ).replace(
        "processed_path: data/processed/acn_caltech_demand_15min.csv",
        f"processed_path: {processed.as_posix()}",
    )
    yaml_path.write_text(patched, encoding="utf-8")
    assert main(["data", "features", "--config", str(yaml_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["n_intervals"] > 0
    assert processed.is_file()


def test_models_demand_from_fixture(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = Path("configs/default.yaml").read_text(encoding="utf-8")
    yaml_path = tmp_path / "cfg.yaml"
    processed = tmp_path / "demand.csv"
    predictions = tmp_path / "demand_predictions.csv"
    metrics = tmp_path / "demand_metrics.csv"
    snapshot = Path("tests/fixtures/acn_sessions.csv").resolve()
    patched = (
        source.replace(
            "snapshot_path: data/raw/acn_caltech_sessions.csv",
            f"snapshot_path: {snapshot.as_posix()}",
        )
        .replace(
            "processed_path: data/processed/acn_caltech_demand_15min.csv",
            f"processed_path: {processed.as_posix()}",
        )
        .replace(
            "predictions_path: data/processed/demand_predictions.csv",
            f"predictions_path: {predictions.as_posix()}",
        )
        .replace(
            "metrics_path: data/processed/demand_metrics.csv",
            f"metrics_path: {metrics.as_posix()}",
        )
        .replace("n_estimators: 200", "n_estimators: 20", 1)
    )
    yaml_path.write_text(patched, encoding="utf-8")
    assert main(["data", "features", "--config", str(yaml_path)]) == 0
    capsys.readouterr()
    assert main(["models", "demand", "--config", str(yaml_path), "--seed", "42"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert "random_forest" in payload["test_mae"]
    assert predictions.is_file()
    assert metrics.is_file()


def test_models_energy_writes_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = Path("configs/default.yaml").read_text(encoding="utf-8")
    yaml_path = tmp_path / "cfg.yaml"
    trips = tmp_path / "trips.csv"
    predictions = tmp_path / "energy_predictions.csv"
    metrics = tmp_path / "energy_metrics.csv"
    patched = (
        source.replace("n_trips: 2000", "n_trips: 80")
        .replace(
            "trips_path: data/processed/synthetic_trips.csv",
            f"trips_path: {trips.as_posix()}",
        )
        .replace(
            "predictions_path: data/processed/energy_predictions.csv",
            f"predictions_path: {predictions.as_posix()}",
        )
        .replace(
            "metrics_path: data/processed/energy_metrics.csv",
            f"metrics_path: {metrics.as_posix()}",
        )
    )
    yaml_path.write_text(patched, encoding="utf-8")
    assert main(["models", "energy", "--config", str(yaml_path), "--seed", "42"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["n_trips"] == 80
    assert "physics" in payload["test_mae"]
    assert trips.is_file()
    assert predictions.is_file()


def test_models_requires_subcommand() -> None:
    assert main(["models"]) == 2
