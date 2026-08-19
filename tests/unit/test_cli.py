"""CLI smoke tests."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest

from chargeopt.cli import main
from chargeopt.config import LEARNER_NAMES


def _shrink_learner_grids(source: str) -> str:
    return (
        source.replace("n_estimators: 200", "n_estimators: 20")
        .replace("n_estimators: 100", "n_estimators: 20")
        .replace("n_estimators: [100, 200]", "n_estimators: [8]")
        .replace("max_depth: [6, 8, 12]", "max_depth: [2]")
        .replace("min_samples_leaf: [2, 8]", "min_samples_leaf: [1]")
        .replace("max_iter: 200", "max_iter: 20")
        .replace("max_iter: 100", "max_iter: 20")
        .replace("max_iter: [100, 200]", "max_iter: [20]")
        .replace("max_depth: [3, 6]", "max_depth: [2]")
        .replace("learning_rate: [0.05, 0.1]", "learning_rate: [0.1]")
        .replace("min_samples_leaf: [10, 20]", "min_samples_leaf: [1]")
        .replace("alpha: [0.1, 1.0, 10.0, 100.0]", "alpha: [0.1, 1.0]")
        .replace("alpha: [0.1, 1.0, 10.0]", "alpha: [1.0]")
        .replace("l1_ratio: [0.2, 0.5, 0.8]", "l1_ratio: [0.5]")
        .replace("n_splits: 4", "n_splits: 2")
    )


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
    source = _shrink_learner_grids(Path("configs/default.yaml").read_text(encoding="utf-8"))
    yaml_path = tmp_path / "cfg.yaml"
    processed = tmp_path / "demand.csv"
    predictions = tmp_path / "demand_predictions.csv"
    metrics = tmp_path / "demand_metrics.csv"
    slices = tmp_path / "demand_error_slices.csv"
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
        .replace(
            "error_slices_path: data/processed/demand_error_slices.csv",
            f"error_slices_path: {slices.as_posix()}",
        )
    )
    yaml_path.write_text(patched, encoding="utf-8")
    assert main(["data", "features", "--config", str(yaml_path)]) == 0
    capsys.readouterr()
    assert main(["models", "demand", "--config", str(yaml_path), "--seed", "42"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert set(LEARNER_NAMES) <= set(payload["test_mae"])
    assert payload["decision_model"] == "random_forest"
    assert "learned_beats_baselines" in payload
    assert payload["best_learned"] in LEARNER_NAMES
    assert predictions.is_file()
    assert metrics.is_file()
    assert slices.is_file()


def test_models_energy_writes_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = Path("configs/default.yaml").read_text(encoding="utf-8")
    yaml_path = tmp_path / "cfg.yaml"
    trips = tmp_path / "trips.csv"
    predictions = tmp_path / "energy_predictions.csv"
    metrics = tmp_path / "energy_metrics.csv"
    cold_metrics = tmp_path / "energy_cold_metrics.csv"
    patched = (
        _shrink_learner_grids(source)
        .replace("n_trips: 2000", "n_trips: 80")
        .replace("cold_holdout_n_trips: 200", "cold_holdout_n_trips: 40")
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
        .replace(
            "cold_metrics_path: data/processed/energy_cold_metrics.csv",
            f"cold_metrics_path: {cold_metrics.as_posix()}",
        )
    )
    yaml_path.write_text(patched, encoding="utf-8")
    assert main(["models", "energy", "--config", str(yaml_path), "--seed", "42"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["n_trips"] == 80
    assert "physics" in payload["test_mae"]
    assert set(LEARNER_NAMES) <= set(payload["test_mae"])
    assert "learned_beats_baselines" in payload
    assert payload["best_learned"] in LEARNER_NAMES
    assert "cold_mae" in payload
    assert trips.is_file()
    assert predictions.is_file()
    assert cold_metrics.is_file()


def test_models_requires_subcommand() -> None:
    assert main(["models"]) == 2


def test_simulate_writes_structured_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = Path("configs/default.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "simulation.yaml"
    stations = tmp_path / "stations.csv"
    run = tmp_path / "run.csv"
    station_ticks = tmp_path / "station_ticks.csv"
    metrics = tmp_path / "metrics.csv"
    snapshot = Path("tests/fixtures/acn_sessions.csv").resolve()
    patched = (
        source.replace(
            "snapshot_path: data/raw/acn_caltech_sessions.csv",
            f"snapshot_path: {snapshot.as_posix()}",
        )
        .replace(
            "stations_path: data/processed/sim_stations.csv",
            f"stations_path: {stations.as_posix()}",
        )
        .replace(
            "run_path: data/processed/sim_run.csv",
            f"run_path: {run.as_posix()}",
        )
        .replace(
            "station_ticks_path: data/processed/sim_station_ticks.csv",
            f"station_ticks_path: {station_ticks.as_posix()}",
        )
        .replace(
            "metrics_path: data/processed/sim_metrics.csv",
            f"metrics_path: {metrics.as_posix()}",
        )
    )
    config_path.write_text(patched, encoding="utf-8")

    assert main(["simulate", "--config", str(config_path), "--seed", "42"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["seed"] == 42
    assert payload["n_ticks"] == 96
    assert payload["n_vehicles"] == 30
    assert set(payload["metrics"]) == {
        "seed",
        "energy_cost",
        "avg_wait_minutes",
        "soc_violations",
        "energy_usage_kwh",
        "station_utilization",
        "vehicle_idle_minutes",
    }
    assert all(path.is_file() for path in (stations, run, station_ticks, metrics))
    run_columns = set(run.read_text(encoding="utf-8").splitlines()[0].split(","))
    assert {
        "drove_this_tick",
        "charged_this_tick",
        "queued_this_tick",
        "stranded_this_tick",
    } <= run_columns
    metrics_header = set(metrics.read_text(encoding="utf-8").splitlines()[0].split(","))
    assert {"routing", "peak_queue"} <= metrics_header


def test_simulate_all_seeds_writes_home_and_probe_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = Path("configs/default.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "simulation.yaml"
    stations = tmp_path / "stations.csv"
    run = tmp_path / "run.csv"
    station_ticks = tmp_path / "station_ticks.csv"
    metrics = tmp_path / "metrics.csv"
    snapshot = Path("tests/fixtures/acn_sessions.csv").resolve()
    patched = (
        source.replace(
            "snapshot_path: data/raw/acn_caltech_sessions.csv",
            f"snapshot_path: {snapshot.as_posix()}",
        )
        .replace(
            "stations_path: data/processed/sim_stations.csv",
            f"stations_path: {stations.as_posix()}",
        )
        .replace(
            "run_path: data/processed/sim_run.csv",
            f"run_path: {run.as_posix()}",
        )
        .replace(
            "station_ticks_path: data/processed/sim_station_ticks.csv",
            f"station_ticks_path: {station_ticks.as_posix()}",
        )
        .replace(
            "metrics_path: data/processed/sim_metrics.csv",
            f"metrics_path: {metrics.as_posix()}",
        )
        .replace(
            "seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]",
            "seeds: [42, 43]",
        )
    )
    config_path.write_text(patched, encoding="utf-8")

    assert main(["simulate", "--config", str(config_path), "--all-seeds"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert "gate_passed" in payload
    assert payload["home"]["n_seeds"] == 2
    assert payload["probe"]["seed"] == 42
    assert all(path.is_file() for path in (stations, run, station_ticks, metrics))
    header = metrics.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "routing" in header
    assert "peak_queue" in header
    rows = metrics.read_text(encoding="utf-8").splitlines()[1:]
    assert len(rows) == 3
    routings = [row.split(",")[header.index("routing")] for row in rows]
    assert routings == ["home", "home", "concentrated"]


def test_models_tune_requires_target() -> None:
    assert main(["models", "tune"]) == 2


def test_models_tune_demand_from_fixture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _shrink_learner_grids(Path("configs/default.yaml").read_text(encoding="utf-8"))
    yaml_path = tmp_path / "cfg.yaml"
    processed = tmp_path / "demand.csv"
    predictions = tmp_path / "demand_predictions.csv"
    metrics = tmp_path / "demand_metrics.csv"
    tune_metrics = tmp_path / "demand_tune.csv"
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
        .replace(
            "tune_metrics_path: data/processed/demand_tune.csv",
            f"tune_metrics_path: {tune_metrics.as_posix()}",
        )
    )
    yaml_path.write_text(patched, encoding="utf-8")
    assert main(["data", "features", "--config", str(yaml_path)]) == 0
    capsys.readouterr()
    assert main(["models", "tune", "demand", "--config", str(yaml_path), "--seed", "42"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert set(payload["best_params"]) == set(LEARNER_NAMES)
    assert set(payload["best_params"]["random_forest"]) == {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
    }
    assert set(payload["best_params"]["ridge"]) == {"alpha"}
    assert tune_metrics.is_file()
    assert "model" in tune_metrics.read_text(encoding="utf-8").splitlines()[0]


def test_models_tune_energy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _shrink_learner_grids(Path("configs/default.yaml").read_text(encoding="utf-8"))
    yaml_path = tmp_path / "cfg.yaml"
    trips = tmp_path / "trips.csv"
    tune_metrics = tmp_path / "energy_tune.csv"
    patched = (
        source.replace("n_trips: 2000", "n_trips: 80")
        .replace(
            "trips_path: data/processed/synthetic_trips.csv",
            f"trips_path: {trips.as_posix()}",
        )
        .replace(
            "tune_metrics_path: data/processed/energy_tune.csv",
            f"tune_metrics_path: {tune_metrics.as_posix()}",
        )
    )
    yaml_path.write_text(patched, encoding="utf-8")
    assert main(["models", "tune", "energy", "--config", str(yaml_path), "--seed", "42"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert set(payload["best_params"]) == set(LEARNER_NAMES)
    assert set(payload["best_params"]["random_forest"]) == {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
    }
    assert tune_metrics.is_file()
    assert "model" in tune_metrics.read_text(encoding="utf-8").splitlines()[0]


def test_models_tune_demand_filters_learner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _shrink_learner_grids(Path("configs/default.yaml").read_text(encoding="utf-8"))
    yaml_path = tmp_path / "cfg.yaml"
    processed = tmp_path / "demand.csv"
    tune_metrics = tmp_path / "demand_tune.csv"
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
            "tune_metrics_path: data/processed/demand_tune.csv",
            f"tune_metrics_path: {tune_metrics.as_posix()}",
        )
    )
    yaml_path.write_text(patched, encoding="utf-8")
    assert main(["data", "features", "--config", str(yaml_path)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "models",
                "tune",
                "demand",
                "--config",
                str(yaml_path),
                "--seed",
                "42",
                "--learner",
                "ridge",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["learners"] == ["ridge"]
    assert set(payload["best_params"]) == {"ridge"}
    assert set(payload["best_params"]["ridge"]) == {"alpha"}
    assert set(payload["val_mae"]) == {"ridge"}
