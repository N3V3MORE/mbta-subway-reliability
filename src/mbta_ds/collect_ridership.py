"""Collector for MBTA gated station entries (ArcGIS FeatureServer).

Daily, half-hourly gate validations for every gated subway / light-rail / Silver
Line station. This is the demand signal used by Track B and as a demand proxy
feature in Track A.

Coverage note
-------------
Probing the service shows its single ``GSE`` table currently spans
2024-08-25 -> 2026-06-30, which is *shorter* than the LAMP performance archive
(2019-09-15 -> present). Ridership is therefore the binding constraint on the
shared analysis window, and :func:`coverage` is queried live rather than
hard-coded so the pipeline keeps working as MBTA publishes more data.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

from . import config
from .http import get_json, make_session
from .progress import Progress

log = logging.getLogger(__name__)

RAW_PATH = config.RIDERSHIP_RAW_DIR / "gated_station_entries_raw.parquet"
DAILY_PATH = config.RIDERSHIP_RAW_DIR / "gated_station_entries_daily.parquet"

OUT_FIELDS = "service_date,time_period,stop_id,station_name,route_or_line,gated_entries"


# ---------------------------------------------------------------------------
# Coverage discovery
# ---------------------------------------------------------------------------
def coverage(session: requests.Session | None = None) -> tuple[date, date]:
    """Return the ``(first, last)`` service date actually present in the service.

    Two single-row queries (one sorted ascending, one descending) are far cheaper
    than a ``returnCountOnly`` scan on a table this large.
    """
    session = session or make_session()
    bounds: list[date] = []
    for order in ("ASC", "DESC"):
        payload = get_json(
            config.GATED_ENTRIES_QUERY_URL,
            session,
            params={
                "where": "1=1",
                "outFields": "service_date",
                "returnGeometry": "false",
                "orderByFields": f"service_date {order}",
                "resultRecordCount": 1,
                "f": "json",
            },
            timeout=120,
        )
        features = payload.get("features", [])
        if not features:
            raise RuntimeError("gated entries service returned no rows at all")
        bounds.append(_epoch_ms_to_date(features[0]["attributes"]["service_date"]))
    return bounds[0], bounds[1]


def _epoch_ms_to_date(value: object) -> date:
    """Convert the ArcGIS epoch-millisecond date encoding to a calendar date.

    ``service_date`` is reported as epoch milliseconds anchored at UTC midnight,
    and the timestamp encodes the service date itself -- so the UTC date
    component is what we want, with no timezone shift applied.
    """
    return pd.Timestamp(int(value), unit="ms", tz="UTC").date()


# ---------------------------------------------------------------------------
# Paged fetch
# ---------------------------------------------------------------------------
def _build_where(start: date, end: date) -> str:
    return (
        "service_date >= timestamp '{start} 00:00:00' AND "
        "service_date <= timestamp '{end} 23:59:59'"
    ).format(start=start.isoformat(), end=end.isoformat())


def _count(session: requests.Session, where: str) -> int:
    payload = get_json(
        config.GATED_ENTRIES_QUERY_URL,
        session,
        params={"where": where, "returnCountOnly": "true", "f": "json"},
        timeout=180,
    )
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error: {payload['error']}")
    return int(payload.get("count", 0))


def _fetch_page(
    session: requests.Session,
    where: str,
    offset: int,
    page_size: int,
) -> list[dict]:
    payload = get_json(
        config.GATED_ENTRIES_QUERY_URL,
        session,
        params={
            "where": where,
            "outFields": OUT_FIELDS,
            "returnGeometry": "false",
            "orderByFields": "ObjectId",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "json",
        },
        timeout=180,
    )
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error at offset {offset}: {payload['error']}")
    return [f.get("attributes", {}) for f in payload.get("features", [])]


def fetch_gated_entries(
    start: date,
    end: date,
    *,
    session: requests.Session | None = None,
    page_size: int = config.GATED_ENTRIES_PAGE_SIZE,
    workers: int = 6,
) -> pd.DataFrame:
    """Fetch gated entries in ``[start, end]``.

    The total row count is established first so every page can then be requested
    concurrently -- a serial walk of ~150 pages is noticeably slow, and the
    service is read-only and rate-limit tolerant.
    """
    session = session or make_session()
    where = _build_where(start, end)

    total = _count(session, where)
    if total == 0:
        raise RuntimeError(f"no gated entries in {start} -> {end}")

    offsets = list(range(0, total, page_size))
    log.info("gated entries: %s rows in %d page(s)", f"{total:,}", len(offsets))

    rows: list[dict] = []
    with Progress(len(offsets), "gated-entry pages", log_every=25) as progress:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_page, make_session(), where, off, page_size): off
                for off in offsets
            }
            failures = 0
            for future in as_completed(futures):
                offset = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    log.error("page at offset %d failed: %s", offset, exc)
                progress.step(extra=f"{len(rows):,} rows so far")
            if failures:
                log.warning("%d of %d pages failed", failures, len(offsets))

    if not rows:
        raise RuntimeError(f"all pages failed for {start} -> {end}")
    if len(rows) < total:
        log.warning("fetched %s of %s rows; some pages failed", f"{len(rows):,}", f"{total:,}")

    return _normalise(pd.DataFrame(rows))


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean types once, at the collection boundary."""
    out = frame.copy()

    out["service_date"] = out["service_date"].map(_epoch_ms_to_date)
    out["gated_entries"] = pd.to_numeric(out["gated_entries"], errors="coerce")

    # `time_period` arrives wrapped as "(00:30:00)"; normalise to "HH:MM:SS" and
    # expose the components as numbers so clustering can bin on them directly.
    cleaned = out["time_period"].astype(str).str.strip().str.strip("()")
    parts = cleaned.str.split(":", expand=True)
    out["time_period"] = cleaned
    out["period_hour"] = pd.to_numeric(parts[0], errors="coerce")
    out["period_minute"] = pd.to_numeric(parts[1], errors="coerce")

    for column in ("stop_id", "station_name", "route_or_line"):
        out[column] = out[column].astype(str).str.strip()

    out = out[out["gated_entries"].notna() & out["period_hour"].notna()]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def collect(
    days: int = 90,
    *,
    start: date | None = None,
    end: date | None = None,
    session: requests.Session | None = None,
    refresh: bool = False,
) -> dict:
    """Fetch, cache and summarise ridership for a window of ``days`` days.

    The window is clamped to the service's real coverage so callers never have to
    know about the publication lag.
    """
    config.ensure_dirs()
    session = session or make_session()

    first, last = coverage(session)
    if start is not None and end is not None:
        window_start, window_end = start, end
    else:
        requested_end = end or last
        window_end = min(requested_end, last)
        window_start = max(window_end - timedelta(days=days - 1), first)
        if window_end != requested_end:
            log.info(
                "clamping ridership window end %s -> %s (service coverage ends %s)",
                requested_end, window_end, last,
            )

    if window_start > window_end:
        raise RuntimeError(
            f"window {window_start} -> {window_end} falls outside "
            f"ridership coverage {first} -> {last}"
        )
    if window_start < first or window_end > last:
        log.warning(
            "ridership window %s -> %s is partly outside coverage %s -> %s; "
            "the service will simply return the rows it has",
            window_start, window_end, first, last,
        )

    raw = fetch_gated_entries(window_start, window_end, session=session)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(RAW_PATH, index=False)

    daily = (
        raw.groupby(["service_date", "stop_id", "station_name", "route_or_line"],
                    as_index=False)["gated_entries"]
        .sum()
        .rename(columns={"gated_entries": "daily_entries"})
    )
    daily.to_parquet(DAILY_PATH, index=False)

    log.info(
        "cached ridership: %s raw rows, %s daily rows",
        f"{len(raw):,}", f"{len(daily):,}",
    )
    return {
        "service_start": window_start,
        "service_end": window_end,
        "raw_rows": len(raw),
        "daily_rows": len(daily),
        "stations": int(daily["station_name"].nunique()),
        "coverage": (first, last),
    }


def load_daily() -> pd.DataFrame:
    """Load one row per station per day of gated entries."""
    if not DAILY_PATH.exists():
        raise FileNotFoundError(f"{DAILY_PATH} missing; run the `collect` stage first")
    return pd.read_parquet(DAILY_PATH)


def load_raw() -> pd.DataFrame:
    """Load the half-hourly station-entry table."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"{RAW_PATH} missing; run the `collect` stage first")
    return pd.read_parquet(RAW_PATH)
