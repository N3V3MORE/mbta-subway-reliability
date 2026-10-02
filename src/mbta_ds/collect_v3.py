"""Live collector for the MBTA V3 API (the "we collected this ourselves" dataset).

Polls ``/predictions``, ``/vehicles`` and ``/alerts`` for the subway routes and
appends timestamped snapshots to JSONL, one gzip file per resource and local
day (``data/raw/v3/<resource>/<date>.jsonl.gz``). JSONL is used deliberately: it
is line-appendable, crash-safe, and needs no schema negotiation, which matters for
a process designed to run in the background for weeks. Each append is its own
gzip member, so a file cut short by a crash loses at most one poll, and the files
are about a tenth of the plain-text size (~1 GB a day uncompressed).

One poll issues three requests. At the default 60-second interval that is
3 requests/minute -- comfortably inside the 20 request/minute anonymous limit,
so the collector works with or without an API key.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from . import config
from .http import get_json, make_session

log = logging.getLogger(__name__)

RESOURCES = ("predictions", "vehicles", "alerts")

_URLS = {
    "predictions": config.V3_PREDICTIONS_URL,
    "vehicles": config.V3_VEHICLES_URL,
    "alerts": config.V3_ALERTS_URL,
}

#: Filters are built from the explicit subway route list rather than
#: ``filter[route_type]`` so the collector does not depend on an undocumented
#: filter and never silently absorbs bus data.
_PARAMS = {
    "predictions": {"filter[route]": ",".join(config.SUBWAY_ROUTES)},
    "vehicles": {"filter[route]": ",".join(config.SUBWAY_ROUTES)},
    "alerts": {"filter[route]": ",".join(config.SUBWAY_ROUTES)},
}


def _scalarise(value: object) -> object:
    """Collapse nested JSON:API values so they can live in a flat table."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def flatten_records(payload: dict, resource: str, fetch_ts: float) -> list[dict]:
    """Flatten a JSON:API document into flat row dicts.

    Attributes become columns; each to-one relationship becomes ``<name>_id``.
    Nested structures are JSON-encoded so nothing is lost to the flattening.
    """
    rows: list[dict] = []
    for item in payload.get("data") or []:
        row: dict = {
            "_resource": resource,
            "_fetch_ts": fetch_ts,
            "_fetched_at": datetime.fromtimestamp(fetch_ts, tz=UTC).isoformat(),
            "id": item.get("id"),
        }
        for key, value in (item.get("attributes") or {}).items():
            row[key] = _scalarise(value)
        for name, relationship in (item.get("relationships") or {}).items():
            data = (relationship or {}).get("data")
            if isinstance(data, dict) and data.get("id"):
                row[f"{name}_id"] = data["id"]
        rows.append(row)
    return rows


def daily_path(resource: str, fetch_ts: float, root: Path | None = None) -> Path:
    """Where a poll at ``fetch_ts`` is stored: one file per resource and local day."""
    day = pd.Timestamp(fetch_ts, unit="s", tz="UTC").tz_convert(config.SERVICE_TZ).date()
    return (root or config.V3_RAW_DIR) / resource / f"{day}.jsonl.gz"


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, default=str) + "\n" for row in rows)
    # Appending a new gzip member per poll keeps each write self-contained.
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(text)


def poll_once(
    session: requests.Session,
    *,
    resources: tuple[str, ...] = RESOURCES,
) -> dict[str, int]:
    """Fetch one snapshot of each resource and append it to disk."""
    fetch_ts = time.time()
    counts: dict[str, int] = {}
    for resource in resources:
        try:
            payload = get_json(_URLS[resource], session, params=_PARAMS[resource], timeout=45)
        except Exception as exc:  # noqa: BLE001 - one bad resource must not kill the loop
            log.warning("poll of /%s failed: %s", resource, exc)
            counts[resource] = 0
            continue
        rows = flatten_records(payload, resource, fetch_ts)
        _append_jsonl(daily_path(resource, fetch_ts), rows)
        counts[resource] = len(rows)
    return counts


def fetch_stations() -> pd.DataFrame:
    """Fetch subway station coordinates and names from the V3 API.

    Needed because the LAMP static stops table carries no latitude/longitude, so
    any map of the network has to come from the API. This is also the cleanest
    demonstration that the API key path works end to end.
    """
    config.ensure_dirs()
    session = make_session(config.get_api_key())
    dest = config.V3_RAW_DIR / "stations.parquet"

    payload = get_json(
        config.V3_BASE + "/stops",
        session,
        params={
            "filter[route]": ",".join(config.SUBWAY_ROUTES),
            "fields[stop]": "name,latitude,longitude,municipality,platform_code,"
                            "wheelchair_boarding,location_type,parent_station",
        },
        timeout=60,
    )
    rows = []
    for item in payload.get("data") or []:
        attributes = item.get("attributes") or {}
        rows.append({
            "stop_id": item.get("id"),
            "stop_name": attributes.get("name"),
            "latitude": attributes.get("latitude"),
            "longitude": attributes.get("longitude"),
            "municipality": attributes.get("municipality"),
            "location_type": attributes.get("location_type"),
            "parent_station": attributes.get("parent_station"),
            "wheelchair_boarding": attributes.get("wheelchair_boarding"),
        })

    frame = pd.DataFrame(rows).dropna(subset=["latitude", "longitude"])
    if frame.empty:
        raise RuntimeError("V3 /stops returned no usable coordinates")
    frame.to_parquet(dest, index=False)
    log.info("cached %d station coordinates to %s", len(frame), dest.name)
    return frame


def load_stations() -> pd.DataFrame:
    """Load cached station coordinates."""
    dest = config.V3_RAW_DIR / "stations.parquet"
    if not dest.exists():
        return fetch_stations()
    return pd.read_parquet(dest)


def minutes_until(clock: str, now: pd.Timestamp | None = None) -> float:
    """Minutes from ``now`` to the next ``HH:MM``, Boston time.

    The daily scheduled run stops at the end of the service day (03:00), so it
    never overlaps the next morning's run, whenever it was started.
    """
    now = now if now is not None else pd.Timestamp.now(tz=config.SERVICE_TZ)
    hour, minute = (int(part) for part in clock.split(":"))
    target = now.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    if target <= now:
        target += pd.Timedelta(days=1)
    return (target - now).total_seconds() / 60


def collect(
    minutes: float = 10.0,
    *,
    interval: float = 60.0,
    resources: tuple[str, ...] = RESOURCES,
) -> dict:
    """Poll the V3 API for ``minutes`` at ``interval`` seconds between polls.

    Returns a summary including total rows written, so the caller can assert the
    collector actually produced data.
    """
    config.ensure_dirs()
    key = config.get_api_key()
    session = make_session(key)
    log.info(
        "live collector starting: %.1f min at %.0fs interval (%s)",
        minutes, interval, "API key" if key else "anonymous, 20 req/min limit",
    )

    deadline = time.monotonic() + minutes * 60
    polls = 0
    totals = dict.fromkeys(resources, 0)

    while True:
        counts = poll_once(session, resources=resources)
        polls += 1
        for resource, count in counts.items():
            totals[resource] += count
        log.info(
            "poll %d: %s",
            polls,
            ", ".join(f"{name}={count}" for name, count in counts.items()),
        )
        if time.monotonic() + interval > deadline:
            break
        time.sleep(interval)

    summary = {"polls": polls, **{f"{k}_rows": v for k, v in totals.items()}}
    log.info("live collector finished: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Reading back what the collector wrote
# ---------------------------------------------------------------------------
def live_files(resource: str, root: Path | None = None) -> list[Path]:
    """Every file holding ``resource``: the daily gzip files, and the single
    plain JSONL file earlier versions of the collector wrote."""
    root = root or config.V3_RAW_DIR
    legacy = root / f"{resource}.jsonl"
    return ([legacy] if legacy.exists() else []) + sorted((root / resource).glob("*.jsonl.gz"))


def load_live(resource: str, root: Path | None = None) -> pd.DataFrame:
    """Load every collected snapshot of a resource into a DataFrame, values as written."""
    if resource not in RESOURCES:
        raise ValueError(f"unknown resource {resource!r}; expected one of {RESOURCES}")
    files = live_files(resource, root)
    if not files:
        raise FileNotFoundError(f"no {resource} snapshots; run `python -m mbta_ds.cli live --minutes 5` first")
    return pd.concat([pd.read_json(path, lines=True, dtype=False, convert_dates=False,
                                   compression="gzip" if path.suffix == ".gz" else None)
                      for path in files], ignore_index=True)


def live_prediction_delays() -> pd.DataFrame:
    """Turn collected predictions into a delay-change table.

    Each row is a (trip, stop) prediction observed at some poll. Grouping by the
    pair lets us measure how much the predicted arrival time *moves* between
    successive polls -- prediction volatility, which is a genuinely
    real-time-only quantity that the historical archive cannot provide.
    """
    frame = load_live("predictions")
    if frame.empty:
        return frame

    out = frame.copy()
    for column in ("arrival_time", "departure_time"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce", utc=True)

    out["_fetched_at"] = pd.to_datetime(out["_fetched_at"], errors="coerce", utc=True)
    out = out[out["arrival_time"].notna()]

    key = [c for c in ("trip_id", "stop_id", "direction_id") if c in out.columns]
    if not key:
        return out

    out = out.sort_values(key + ["_fetched_at"])
    out["predicted_arrival_epoch"] = out["arrival_time"].astype("int64") // 10**9
    grouped = out.groupby(key, dropna=False)
    out["poll_index"] = grouped.cumcount()
    out["arrival_revision_seconds"] = grouped["predicted_arrival_epoch"].diff()
    out["polls_observed"] = grouped["predicted_arrival_epoch"].transform("size")
    return out
