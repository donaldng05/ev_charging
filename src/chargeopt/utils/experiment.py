"""Stable experiment identity from config, seed, policy, and git SHA."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chargeopt.config import AppConfig, PolicyName


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    sha = result.stdout.strip()
    return sha or None


def config_hash(config: AppConfig) -> str:
    """Return the full SHA-256 hash of the canonical resolved configuration."""
    encoded = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def experiment_id(
    config: AppConfig,
    *,
    seed: int,
    policy: PolicyName,
    commit_sha: str | None = None,
    scenario: str = "normal",
) -> str:
    payload = {
        "config_hash": config_hash(config),
        "seed": seed,
        "policy": policy.value,
        "scenario": scenario,
        "git_sha": commit_sha or "unknown",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
