"""15-minute demand features and temporal split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chargeopt.data.io import read_sessions_csv
from chargeopt.features.demand import _temporal_split, build_demand_table

FIXTURE = Path("tests/fixtures/acn_sessions.csv")


def test_demand_table_site_level_and_temporal_split() -> None:
    sessions = read_sessions_csv(FIXTURE)
    demand = build_demand_table(
        sessions,
        timestep_minutes=15,
        timezone_name="America/Los_Angeles",
        train_fraction=0.7,
        val_fraction=0.15,
    )
    assert demand["site_id"].nunique() == 1
    assert (demand["n_arrivals"] >= 0).all()
    assert (demand["energy_kwh"] >= 0).all()
    assert abs(float(demand["energy_kwh"].sum()) - float(sessions["energy_kwh"].sum())) < 1e-6
    assert demand["n_arrivals"].sum() == len(sessions)
    assert list(demand["split"].unique()) == ["train", "val", "test"]
    splits = demand["split"].tolist()
    assert splits == sorted(splits, key=["train", "val", "test"].index)
    assert demand["timestamp"].is_monotonic_increasing


def test_lags_do_not_use_future_energy() -> None:
    sessions = read_sessions_csv(FIXTURE)
    demand = build_demand_table(
        sessions,
        timestep_minutes=15,
        timezone_name="America/Los_Angeles",
        train_fraction=0.7,
        val_fraction=0.15,
    )
    energy = demand["energy_kwh"]
    lag = demand["lag_15m"]
    comparable = lag.notna()
    assert (lag[comparable].to_numpy() == energy.shift(1)[comparable].to_numpy()).all()


def test_arrival_lands_in_start_bin() -> None:
    sessions = read_sessions_csv(FIXTURE)
    demand = build_demand_table(
        sessions,
        timestep_minutes=15,
        timezone_name="America/Los_Angeles",
        train_fraction=0.7,
        val_fraction=0.15,
    )
    start = pd.Timestamp(sessions.iloc[0]["start_time"]).floor("15min")
    row = demand.loc[demand["timestamp"] == start].iloc[0]
    assert row["n_arrivals"] >= 1
    assert row["energy_kwh"] > 0


def test_demand_table_rejects_30_minute_bins() -> None:
    sessions = read_sessions_csv(FIXTURE)
    try:
        build_demand_table(
            sessions,
            timestep_minutes=30,
            timezone_name="America/Los_Angeles",
            train_fraction=0.7,
            val_fraction=0.15,
        )
    except ValueError as exc:
        assert str(exc) == "MVP demand features require timestep_minutes=15"
    else:
        raise AssertionError("30-minute demand bins must be rejected")


def test_temporal_split_handles_small_inputs() -> None:
    assert _temporal_split(0, 0.7, 0.15) == []
    assert _temporal_split(1, 0.7, 0.15) == ["test"]
    assert _temporal_split(2, 0.7, 0.15) == ["train", "test"]
