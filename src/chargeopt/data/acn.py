"""Pull ACN-Data sessions and normalize to the derived CSV schema."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from chargeopt.data.io import write_sessions_csv
from chargeopt.data.validation import validate_sessions

LOGGER = logging.getLogger(__name__)

PULL_CHUNK_DAYS = 30
SessionFetch = Callable[..., Iterable[dict[str, Any]]]


def localize_naive(value: datetime, timezone_name: str) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=ZoneInfo(timezone_name))
    return value


def iter_time_chunks(
    start: datetime,
    end: datetime,
    *,
    days: int = PULL_CHUNK_DAYS,
) -> Iterator[tuple[datetime, datetime]]:
    if days < 1:
        msg = "chunk days must be >= 1"
        raise ValueError(msg)
    if end <= start:
        msg = "end must be after start"
        raise ValueError(msg)
    cursor = start
    step = timedelta(days=days)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        yield cursor, chunk_end
        cursor = chunk_end


def iter_chunked_sessions(
    fetch: SessionFetch,
    *,
    site: str,
    start: datetime,
    end: datetime,
    chunk_days: int = PULL_CHUNK_DAYS,
) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for chunk_start, chunk_end in iter_time_chunks(start, end, days=chunk_days):
        for raw in fetch(site=site, start=chunk_start, end=chunk_end):
            session_id = str(raw.get("sessionID", ""))
            if session_id and session_id in seen:
                continue
            if session_id:
                seen.add(session_id)
            yield raw


def iter_acn_sessions(
    *,
    site: str,
    start: datetime,
    end: datetime,
    token: str,
) -> Iterator[dict[str, Any]]:
    from acnportal.acndata import DataClient  # type: ignore[import-untyped]

    client = DataClient(token)

    def fetch(*, site: str, start: datetime, end: datetime) -> Iterable[dict[str, Any]]:
        sessions = client.get_sessions_by_time(site, start=start, end=end)
        return cast(Iterable[dict[str, Any]], sessions)

    yield from iter_chunked_sessions(fetch, site=site, start=start, end=end)


def _as_datetime(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.Timestamp(pd.to_datetime(value, utc=True))
    if pd.isna(ts):
        return None
    parsed = ts.to_pydatetime()
    if not isinstance(parsed, datetime):
        return None
    return parsed


def normalize_session(raw: dict[str, Any], *, site: str) -> dict[str, Any]:
    start = _as_datetime(raw.get("connectionTime"))
    end = _as_datetime(raw.get("disconnectTime"))
    if start is None or end is None:
        msg = f"session missing connectionTime/disconnectTime: {raw.get('sessionID')}"
        raise ValueError(msg)
    done = _as_datetime(raw.get("doneChargingTime"))
    if done is not None and not start <= done <= end:
        LOGGER.warning(
            "ignoring invalid doneChargingTime for session %s",
            raw.get("sessionID"),
        )
        done = None
    duration_min = (end - start).total_seconds() / 60.0
    local_start = start.astimezone(ZoneInfo("America/Los_Angeles"))
    energy = raw.get("kWhDelivered")
    if energy is None:
        msg = f"session missing kWhDelivered: {raw.get('sessionID')}"
        raise ValueError(msg)
    return {
        "session_id": str(raw["sessionID"]),
        "site_id": site,
        "station_id": str(raw["stationID"]),
        "space_id": str(raw["spaceID"]),
        "start_time": start,
        "end_time": end,
        "done_charging_time": done,
        "duration_min": duration_min,
        "energy_kwh": float(energy),
        "day_of_week": int(local_start.weekday()),
        "hour": int(local_start.hour),
    }


def sessions_frame_from_raw(raw_sessions: Iterable[dict[str, Any]], *, site: str) -> pd.DataFrame:
    rows = [normalize_session(raw, site=site) for raw in raw_sessions]
    if not rows:
        msg = "no ACN-Data sessions returned for the requested window"
        raise ValueError(msg)
    return pd.DataFrame(rows)


def snapshot_sessions(
    raw_sessions: Iterable[dict[str, Any]],
    *,
    site: str,
    path: Path,
) -> pd.DataFrame:
    frame = validate_sessions(sessions_frame_from_raw(raw_sessions, site=site))
    write_sessions_csv(frame, path)
    return frame
