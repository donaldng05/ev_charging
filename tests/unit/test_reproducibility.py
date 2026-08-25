"""Tests for reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import pytest

from chargeopt.config import PolicyName, load_config
from chargeopt.utils.experiment import config_hash, experiment_id, git_sha
from chargeopt.utils.seed import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(42)
    first = [random.random() for _ in range(5)]
    set_seed(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_seed_reproduces_numpy() -> None:
    set_seed(42)
    first = np.random.random(5)
    set_seed(42)
    second = np.random.random(5)
    assert np.array_equal(first, second)


def test_set_seed_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)


def test_experiment_id_is_stable() -> None:
    config = load_config()
    first = experiment_id(config, seed=42, policy=PolicyName.ML_INFORMED, commit_sha="abc")
    second = experiment_id(config, seed=42, policy=PolicyName.ML_INFORMED, commit_sha="abc")
    assert first == second
    assert len(first) == 12


def test_config_hash_is_stable_and_full_length() -> None:
    config = load_config()

    assert config_hash(config) == config_hash(config)
    assert len(config_hash(config)) == 64


def test_experiment_id_changes_with_seed() -> None:
    config = load_config()
    a = experiment_id(config, seed=42, policy=PolicyName.NEAREST, commit_sha="abc")
    b = experiment_id(config, seed=43, policy=PolicyName.NEAREST, commit_sha="abc")
    assert a != b


def test_experiment_id_changes_with_scenario() -> None:
    config = load_config()
    normal = experiment_id(
        config,
        seed=42,
        policy=PolicyName.NEAREST,
        commit_sha="abc",
        scenario="normal",
    )
    stress = experiment_id(
        config,
        seed=42,
        policy=PolicyName.NEAREST,
        commit_sha="abc",
        scenario="stress",
    )
    assert normal != stress


def test_git_sha_is_string_or_none() -> None:
    sha = git_sha()
    assert sha is None or (isinstance(sha, str) and len(sha) >= 7)
