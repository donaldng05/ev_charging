"""M5 policy-matrix evaluation and artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from chargeopt.cli import main
from chargeopt.config import MetricName, PolicyName, load_config
from chargeopt.data.io import read_sessions_csv
from chargeopt.evaluation import (
    ScenarioName,
    build_robustness,
    read_evaluation_results,
    read_evaluation_robustness,
    read_evaluation_summary,
    run_evaluation,
    summarize_results,
    write_evaluation_artifacts,
)
from chargeopt.models.io import write_demand_predictions


def _config(tmp_path: Path, *, seeds: list[int] | None = None):
    config = load_config()
    evaluation = config.evaluation.model_copy(
        update={
            "raw_results_path": tmp_path / "results.csv",
            "summary_path": tmp_path / "summary.csv",
            "robustness_path": tmp_path / "robustness.csv",
            "metadata_path": tmp_path / "metadata.json",
        }
    )
    data = config.data.model_copy(
        update={"snapshot_path": Path("tests/fixtures/acn_sessions.csv").resolve()}
    )
    experiment = config.experiment.model_copy(update={"seeds": seeds or [42, 43]})
    return config.model_copy(
        update={"data": data, "evaluation": evaluation, "experiment": experiment}
    )


def _write_forecast(config, path: Path) -> None:
    timestamps = pd.date_range(
        "2018-09-05T07:00:00Z",
        periods=config.simulation.steps_per_day,
        freq="15min",
    )
    write_demand_predictions(
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "split": ["test"] * len(timestamps),
                "target": [1.0] * len(timestamps),
                "prediction": [1.0] * len(timestamps),
                "model": [config.models.demand.decision_model] * len(timestamps),
                "seed": [42] * len(timestamps),
            }
        ),
        path,
    )


def test_full_matrix_is_deterministic_and_writes_separate_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prediction_path = tmp_path / "demand_predictions.csv"
    _write_forecast(config, prediction_path)
    config = config.model_copy(
        update={
            "models": config.models.model_copy(
                update={
                    "demand": config.models.demand.model_copy(
                        update={"predictions_path": prediction_path}
                    )
                }
            )
        }
    )
    sessions = read_sessions_csv(config.data.snapshot_path)

    first = run_evaluation(config, sessions=sessions, commit_sha="abc123")
    second = run_evaluation(config, sessions=sessions, commit_sha="abc123")

    pd.testing.assert_frame_equal(first.raw_results, second.raw_results)
    assert list(first.raw_results["policy"]) == [
        "nearest",
        "nearest",
        "cheapest",
        "cheapest",
        "ml_informed",
        "ml_informed",
    ]
    assert list(first.raw_results["seed"]) == [42, 43, 42, 43, 42, 43]
    assert len(first.summary) == len(config.experiment.policies) * len(MetricName)
    assert first.metadata["n_runs"] == 6

    paths = write_evaluation_artifacts(first, config)
    assert all(Path(path).is_file() for path in paths.values())
    assert len(read_evaluation_results(Path(paths["raw_results"]))) == 6
    assert len(read_evaluation_summary(Path(paths["summary"]))) == 18
    metadata = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "m6-evaluation-v1"
    assert metadata["git_sha"] == "abc123"


def test_evaluation_filters_select_configured_subset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sessions = read_sessions_csv(config.data.snapshot_path)

    report = run_evaluation(
        config,
        sessions=sessions,
        policies=[PolicyName.CHEAPEST],
        seeds=[43],
        commit_sha="abc123",
    )

    assert report.raw_results["policy"].tolist() == ["cheapest"]
    assert report.raw_results["seed"].tolist() == [43]

    with pytest.raises(ValueError, match="not configured"):
        run_evaluation(config, sessions=sessions, seeds=[999], commit_sha="abc123")


def test_summary_uses_metric_directions_and_single_seed_ci() -> None:
    raw = pd.DataFrame(
        [
            {
                "experiment_id": "a",
                "config_hash": "h",
                "git_sha": "g",
                "scenario": "normal",
                "policy": "nearest",
                "seed": 42,
                "energy_cost": 1.0,
                "avg_wait_minutes": 2.0,
                "soc_violations": 0,
                "energy_usage_kwh": 10.0,
                "station_utilization": 0.4,
                "vehicle_idle_minutes": 3.0,
            },
            {
                "experiment_id": "b",
                "config_hash": "h",
                "git_sha": "g",
                "scenario": "normal",
                "policy": "nearest",
                "seed": 43,
                "energy_cost": 3.0,
                "avg_wait_minutes": 4.0,
                "soc_violations": 1,
                "energy_usage_kwh": 12.0,
                "station_utilization": 0.2,
                "vehicle_idle_minutes": 5.0,
            },
        ]
    )

    summary = summarize_results(raw, confidence_level=0.95)
    cost = summary.loc[summary["metric"] == "energy_cost"].iloc[0]
    utilization = summary.loc[summary["metric"] == "station_utilization"].iloc[0]

    assert cost["n"] == 2
    assert cost["mean"] == pytest.approx(2.0)
    assert cost["std"] == pytest.approx(2**0.5)
    assert cost["worst"] == pytest.approx(3.0)
    assert cost["ci_low"] < cost["mean"] < cost["ci_high"]
    assert utilization["worst"] == pytest.approx(0.2)

    single = summarize_results(raw.iloc[[0]], confidence_level=0.95)
    assert (single["std"] == 0.0).all()
    assert (single["ci_low"] == single["mean"]).all()
    assert (single["ci_high"] == single["mean"]).all()


def test_missing_ml_forecast_fails_before_writing_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sessions = read_sessions_csv(config.data.snapshot_path)
    config = config.model_copy(
        update={
            "models": config.models.model_copy(
                update={
                    "demand": config.models.demand.model_copy(
                        update={"predictions_path": tmp_path / "missing.csv"}
                    )
                }
            )
        }
    )

    with pytest.raises(FileNotFoundError, match="forecast"):
        run_evaluation(
            config,
            sessions=sessions,
            policies=[PolicyName.ML_INFORMED],
            seeds=[42],
            commit_sha="abc123",
        )
    assert not Path(config.evaluation.raw_results_path).exists()


def test_artifact_readers_reject_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"policy": ["nearest"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        read_evaluation_results(path)


def test_artifact_readers_reject_extra_or_non_numeric_columns(tmp_path: Path) -> None:
    extra_path = tmp_path / "extra.csv"
    extra = {
        column: [0]
        for column in (
            "experiment_id",
            "config_hash",
            "git_sha",
            "scenario",
            "policy",
            "seed",
            "energy_cost",
            "avg_wait_minutes",
            "soc_violations",
            "energy_usage_kwh",
            "station_utilization",
            "vehicle_idle_minutes",
        )
    }
    extra.update({"unexpected": [1], "policy": ["nearest"], "scenario": ["normal"]})
    pd.DataFrame(extra).to_csv(extra_path, index=False)

    with pytest.raises(ValueError, match="unexpected columns"):
        read_evaluation_results(extra_path)

    bad_path = tmp_path / "bad-values.csv"
    extra.pop("unexpected")
    extra["energy_cost"] = ["not-a-number"]
    pd.DataFrame(extra).to_csv(bad_path, index=False)

    with pytest.raises(ValueError, match="finite numbers"):
        read_evaluation_results(bad_path)


def test_cli_experiment_reports_artifacts_and_matrix_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, seeds=[42])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json")),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "experiment",
                "--config",
                str(config_path),
                "--policy",
                "cheapest",
                "--seed",
                "42",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "ok"
    assert payload["n_runs"] == 1
    assert payload["n_policies"] == 1
    assert payload["n_seeds"] == 1
    assert payload["policies"] == ["cheapest"]
    assert payload["seeds"] == [42]
    assert payload["n_scenarios"] == 1
    assert len(payload["config_hash"]) == 64
    assert set(payload["paths"]) == {"raw_results", "summary", "metadata"}


def test_stress_evaluation_is_paired_and_writes_robustness_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path, seeds=[42, 43])
    prediction_path = tmp_path / "demand_predictions.csv"
    _write_forecast(config, prediction_path)
    config = config.model_copy(
        update={
            "models": config.models.model_copy(
                update={
                    "demand": config.models.demand.model_copy(
                        update={"predictions_path": prediction_path}
                    )
                }
            )
        }
    )
    sessions = read_sessions_csv(config.data.snapshot_path)

    report = run_evaluation(
        config,
        sessions=sessions,
        commit_sha="abc123",
        scenarios=(ScenarioName.NORMAL, ScenarioName.STRESS),
    )

    assert len(report.raw_results) == 12
    assert report.raw_results.groupby("scenario").size().to_dict() == {
        "normal": 6,
        "stress": 6,
    }
    assert report.raw_results.duplicated(subset=["scenario", "policy", "seed"]).sum() == 0
    pairs = report.raw_results.groupby(["policy", "seed"])["scenario"].agg(tuple)
    assert set(pairs) == {("normal", "stress")}
    assert report.raw_results["experiment_id"].nunique() == 12
    assert len(report.summary) == 36
    assert len(report.robustness) == 18
    assert report.metadata["stress"]["demand_multiplier"] == 1.5

    paths = write_evaluation_artifacts(report, config)
    assert Path(paths["robustness"]).is_file()
    robustness = read_evaluation_robustness(Path(paths["robustness"]))
    assert list(robustness.columns) == [
        "policy",
        "metric",
        "n",
        "normal_mean",
        "stress_mean",
        "stress_minus_normal",
        "delta_std",
        "delta_ci_low",
        "delta_ci_high",
        "robustness_ratio",
    ]
    assert (robustness["n"] == 2).all()


def test_robustness_uses_paired_deltas_and_na_for_zero_baselines() -> None:
    rows = []
    for scenario, values in (
        ("normal", {"energy_cost": 2.0, "avg_wait_minutes": 0.0}),
        ("stress", {"energy_cost": 4.0, "avg_wait_minutes": 3.0}),
    ):
        for seed in (42, 43):
            rows.append(
                {
                    "experiment_id": f"{scenario}-{seed}",
                    "config_hash": "h",
                    "git_sha": "g",
                    "scenario": scenario,
                    "policy": "nearest",
                    "seed": seed,
                    "energy_cost": values["energy_cost"],
                    "avg_wait_minutes": values["avg_wait_minutes"],
                    "soc_violations": 0.0,
                    "energy_usage_kwh": 1.0,
                    "station_utilization": 0.5,
                    "vehicle_idle_minutes": 0.0,
                }
            )

    robustness = build_robustness(
        pd.DataFrame(rows),
        confidence_level=0.95,
    )
    cost = robustness.loc[robustness["metric"] == "energy_cost"].iloc[0]
    wait = robustness.loc[robustness["metric"] == "avg_wait_minutes"].iloc[0]
    assert cost["n"] == 2
    assert cost["stress_minus_normal"] == pytest.approx(2.0)
    assert cost["robustness_ratio"] == pytest.approx(2.0)
    assert cost["delta_std"] == pytest.approx(0.0)
    assert cost["delta_ci_low"] == pytest.approx(cost["stress_minus_normal"])
    assert pd.isna(wait["robustness_ratio"])
    assert wait["stress_minus_normal"] == pytest.approx(3.0)


def test_cli_stress_reports_scenarios_and_robustness_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, seeds=[42])
    prediction_path = tmp_path / "demand_predictions.csv"
    _write_forecast(config, prediction_path)
    config = config.model_copy(
        update={
            "models": config.models.model_copy(
                update={
                    "demand": config.models.demand.model_copy(
                        update={"predictions_path": prediction_path}
                    )
                }
            )
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json")),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "experiment",
                "--config",
                str(config_path),
                "--policy",
                "cheapest",
                "--seed",
                "42",
                "--stress",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_runs"] == 2
    assert payload["n_scenarios"] == 2
    assert set(payload["scenarios"]) == {"normal", "stress"}
    assert set(payload["paths"]) == {"raw_results", "summary", "metadata", "robustness"}
