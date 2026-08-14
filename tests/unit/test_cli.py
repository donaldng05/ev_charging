"""CLI smoke tests."""

from __future__ import annotations

import json

import pytest

from chargeopt.cli import main


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "chargeopt" in captured.out
    assert "experiment" in captured.out


def test_experiment_resolves_default_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["experiment", "--policy", "ml", "--seed", "42"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "not_implemented"
    assert payload["policy"] == "ml_informed"
    assert payload["seed"] == 42
    assert payload["config"]["simulation"]["fleet_size"] == 30
    assert "experiment_id" in payload


def test_unknown_policy_fails() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["experiment", "--policy", "rl"])
    assert excinfo.value.code != 0
