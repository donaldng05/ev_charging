"""ACN-Data session ingest and schema tests (no live API)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from chargeopt.data.acn import (
    iter_chunked_sessions,
    iter_time_chunks,
    normalize_session,
    snapshot_sessions,
)
from chargeopt.data.io import read_sessions_csv
from chargeopt.data.schemas import ChargingSession, Station
from chargeopt.data.validation import validate_sessions

FIXTURE = Path("tests/fixtures/acn_sessions.csv")

RAW_SESSION = {
    "_id": "5bc93a46f9af8b0dc677d7fe",
    "userInputs": None,
    "sessionID": "2_39_79_383_2018-09-05 07:01:52.123265",
    "stationID": "2-39-79-383",
    "spaceID": "CA-492",
    "siteID": "0002",
    "clusterID": "0039",
    "connectionTime": datetime(2018, 9, 5, 7, 1, 52, tzinfo=UTC),
    "disconnectTime": datetime(2018, 9, 7, 11, 54, 0, tzinfo=UTC),
    "kWhDelivered": 13.071,
    "doneChargingTime": datetime(2018, 9, 5, 9, 15, 4, tzinfo=UTC),
    "timezone": "America/Los_Angeles",
    "userID": None,
}


def test_normalize_session_maps_acn_fields() -> None:
    row = normalize_session(RAW_SESSION, site="caltech")
    assert row["session_id"] == RAW_SESSION["sessionID"]
    assert row["site_id"] == "caltech"
    assert row["station_id"] == "2-39-79-383"
    assert row["space_id"] == "CA-492"
    assert row["energy_kwh"] == 13.071
    assert row["duration_min"] > 0
    ChargingSession.model_validate(row)


def test_normalize_session_drops_done_time_before_connection() -> None:
    raw = {**RAW_SESSION, "doneChargingTime": datetime(2018, 9, 4, tzinfo=UTC)}

    row = normalize_session(raw, site="caltech")

    assert row["done_charging_time"] is None


def test_normalize_session_drops_done_time_after_disconnect() -> None:
    raw = {**RAW_SESSION, "doneChargingTime": datetime(2018, 9, 8, tzinfo=UTC)}

    row = normalize_session(raw, site="caltech")

    assert row["done_charging_time"] is None


def test_snapshot_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sessions.csv"
    frame = snapshot_sessions([RAW_SESSION], site="caltech", path=path)
    assert path.is_file()
    loaded = read_sessions_csv(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["session_id"] == RAW_SESSION["sessionID"]
    validate_sessions(frame)


def test_fixture_csv_validates() -> None:
    frame = validate_sessions(read_sessions_csv(FIXTURE))
    assert len(frame) == 6
    assert set(frame["site_id"]) == {"caltech"}


def test_charging_session_rejects_negative_energy() -> None:
    row = normalize_session(RAW_SESSION, site="caltech")
    row["energy_kwh"] = -1
    with pytest.raises(ValidationError):
        ChargingSession.model_validate(row)


def test_charging_session_rejects_end_before_start() -> None:
    row = normalize_session(RAW_SESSION, site="caltech")
    row["end_time"] = row["start_time"]
    row["start_time"] = datetime(2019, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError):
        ChargingSession.model_validate(row)


def test_station_has_no_invented_coordinates() -> None:
    station = Station(station_id="2-39-79-383", site_id="caltech", space_id="CA-492")
    assert "latitude" not in station.model_dump()


def test_validate_sessions_rejects_duplicates() -> None:
    frame = read_sessions_csv(FIXTURE)
    frame = frame.iloc[[0, 0]]
    with pytest.raises(ValueError, match="unique"):
        validate_sessions(frame)


def test_iter_time_chunks_are_half_open_and_cover_the_window() -> None:
    tz = ZoneInfo("America/Los_Angeles")
    start = datetime(2018, 5, 1, tzinfo=tz)
    end = datetime(2018, 7, 15, tzinfo=tz)
    chunks = list(iter_time_chunks(start, end, days=30))
    assert chunks[0] == (start, start + timedelta(days=30))
    assert chunks[-1][1] == end
    for index, (chunk_start, chunk_end) in enumerate(chunks[:-1]):
        assert chunk_end == chunks[index + 1][0]
        assert chunk_end > chunk_start
    covered = sum((stop - begin).total_seconds() for begin, stop in chunks)
    assert covered == (end - start).total_seconds()


def test_iter_chunked_sessions_dedupes_boundary_ids() -> None:
    calls: list[tuple[datetime, datetime]] = []

    def fetch(*, site: str, start: datetime, end: datetime) -> list[dict[str, object]]:
        assert site == "caltech"
        calls.append((start, end))
        shared = {
            "sessionID": "shared-boundary",
            "stationID": "2-39-79-383",
            "spaceID": "CA-492",
            "connectionTime": start,
            "disconnectTime": start + timedelta(hours=1),
            "kWhDelivered": 1.0,
        }
        unique = {
            **shared,
            "sessionID": f"unique-{start.isoformat()}",
        }
        return [shared, unique]

    tz = ZoneInfo("America/Los_Angeles")
    start = datetime(2018, 5, 1, tzinfo=tz)
    end = datetime(2018, 6, 15, tzinfo=tz)
    sessions = list(
        iter_chunked_sessions(
            fetch,
            site="caltech",
            start=start,
            end=end,
            chunk_days=30,
        )
    )
    ids = [str(row["sessionID"]) for row in sessions]
    assert ids.count("shared-boundary") == 1
    assert len(ids) == len(set(ids))
    assert len(calls) >= 2
