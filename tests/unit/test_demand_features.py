"""15-minute demand features and temporal split."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from chargeopt.data.io import read_sessions_csv
from chargeopt.features.demand import (
    LAG_1W,
    ROLL_7D,
    build_demand_table,
    era_label,
    temporal_split_labels,
)

FIXTURE = Path("tests/fixtures/acn_sessions.csv")
COVID_START = datetime(2020, 3, 1, tzinfo=ZoneInfo("America/Los_Angeles"))
COVID_END = datetime(2021, 9, 1, tzinfo=ZoneInfo("America/Los_Angeles"))


def _build_demand() -> pd.DataFrame:
    return build_demand_table(
        read_sessions_csv(FIXTURE),
        timestep_minutes=15,
        timezone_name="America/Los_Angeles",
        train_fraction=0.7,
        val_fraction=0.15,
        covid_start=COVID_START,
        covid_end=COVID_END,
    )


def test_demand_table_site_level_and_temporal_split() -> None:
    sessions = read_sessions_csv(FIXTURE)
    demand = _build_demand()
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
    demand = _build_demand()
    energy = demand["energy_kwh"]
    lag = demand["lag_15m"]
    comparable = lag.notna()
    assert (lag[comparable].to_numpy() == energy.shift(1)[comparable].to_numpy()).all()


def test_arrival_lands_in_start_bin() -> None:
    sessions = read_sessions_csv(FIXTURE)
    demand = _build_demand()
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
            covid_start=COVID_START,
            covid_end=COVID_END,
        )
    except ValueError as exc:
        assert str(exc) == "MVP demand features require timestep_minutes=15"
    else:
        raise AssertionError("30-minute demand bins must be rejected")


def test_temporal_split_handles_small_inputs() -> None:
    assert temporal_split_labels(0, 0.7, 0.15) == []
    assert temporal_split_labels(1, 0.7, 0.15) == ["test"]
    assert temporal_split_labels(2, 0.7, 0.15) == ["train", "test"]


def test_weekly_lags_do_not_use_future_energy() -> None:
    demand = _build_demand()
    energy = demand["energy_kwh"]
    lag = demand["lag_1w"]
    comparable = lag.notna()
    expected = energy.shift(LAG_1W)
    assert (lag[comparable].to_numpy() == expected[comparable].to_numpy()).all()
    rolling = demand["rolling_mean_7d"]
    shifted = energy.shift(1)
    expected_roll = shifted.rolling(ROLL_7D, min_periods=1).mean()
    roll_ok = rolling.notna()
    assert (rolling[roll_ok].to_numpy() == expected_roll[roll_ok].to_numpy()).all()


def test_era_label_uses_half_open_covid_window() -> None:
    tz = ZoneInfo("America/Los_Angeles")
    assert era_label(datetime(2019, 12, 31, tzinfo=tz), COVID_START, COVID_END) == "pre_covid"
    assert era_label(COVID_START, COVID_START, COVID_END) == "covid"
    assert era_label(datetime(2021, 8, 31, tzinfo=tz), COVID_START, COVID_END) == "covid"
    assert era_label(COVID_END, COVID_START, COVID_END) == "post_covid"


def test_demand_table_labels_fixture_as_pre_covid() -> None:
    demand = _build_demand()
    assert set(demand["era"]) == {"pre_covid"}
