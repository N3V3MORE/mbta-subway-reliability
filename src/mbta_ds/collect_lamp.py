"""Collector for the LAMP public performance archive (no API key required).

MBTA's LAMP platform publishes one Parquet file per service date containing one
row per ``trip_id`` x ``stop_id``, with both the *actual* observed stop timestamp
and the *scheduled* arrival/departure time. That pairing is what makes delay
modelling possible: delay is the difference between the two.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

from . import config
from .http import download, make_session
from .progress import Progress

log = logging.getLogger(__name__)

#: Columns we actually use. Keeping the projection narrow cuts peak memory on
#: multi-month windows by roughly two thirds versus reading every column.
PERFORMANCE_COLUMNS = (
    "service_date",
    "start_time",
    "route_id",
    "branch_route_id",
    "trunk_route_id",
    "trip_id",
    "direction_id",
    "direction",
    "direction_destination",
    "stop_count",
    "vehicle_id",
    "stop_id",
    "parent_station",
    "stop_sequence",
    "move_timestamp",
    "stop_timestamp",
    "travel_time_seconds",
    "dwell_time_seconds",
    "headway_branch_seconds",
    "headway_trunk_seconds",
    "scheduled_arrival_time",
    "scheduled_departure_time",
    "scheduled_travel_time",
    "scheduled_headway_branch",
    "scheduled_headway_trunk",
)

INDEX_PATH = config.RAW_DIR / "lamp_index.csv"
ALERTS_PATH = config.ALERTS_RAW_DIR / "LAMP_RT_ALERTS.parquet"
STOPS_PATH = config.STATIC_RAW_DIR / "LAMP_static_stops.parquet"


# ---------------------------------------------------------------------------
# Index / window resolution
# ---------------------------------------------------------------------------
def fetch_index(
    session: requests.Session | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the LAMP catalogue of published performance files.

    Columns: ``size_bytes``, ``last_modified``, ``service_date``, ``file_url``.
    """
    session = session or make_session()
    if refresh or not INDEX_PATH.exists():
        log.info("fetching LAMP performance index")
        download(config.SUBWAY_PERF_INDEX_URL, INDEX_PATH, session, skip_if_exists=False)

    index = pd.read_csv(INDEX_PATH)
    index["service_date"] = pd.to_datetime(index["service_date"]).dt.date
    index = index.sort_values("service_date").reset_index(drop=True)
    return index


def coverage(index: pd.DataFrame) -> tuple[date, date]:
    """Return the ``(first, last)`` service date the LAMP catalogue covers."""
    return index["service_date"].min(), index["service_date"].max()


def resolve_window(
    index: pd.DataFrame,
    days: int,
    *,
    end: date | None = None,
) -> tuple[date, date]:
    """Resolve a ``(start, end)`` service-date window from the catalogue.

    ``days`` counts backwards from the most recent published date (or from
    ``end`` when given). The window is clamped to what actually exists so a
    request for more history than LAMP holds degrades gracefully.
    """
    first, last_available = coverage(index)
    selected = index["service_date"][index["service_date"] <= (end or last_available)]
    if selected.empty:
        raise ValueError(f"no LAMP data at or before {end}")
    last = selected.max()

    start = last - pd.Timedelta(days=days - 1)
    start = start.date() if hasattr(start, "date") else start
    return max(start, first), last


# ---------------------------------------------------------------------------
# Performance files
# ---------------------------------------------------------------------------
def download_performance(
    index: pd.DataFrame,
    start: date,
    end: date,
    *,
    session: requests.Session | None = None,
    workers: int = 4,
) -> list[Path]:
    """Download every per-date performance Parquet in ``[start, end]``.

    Files already present are skipped, so re-running is cheap and incremental.
    """
    config.LAMP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = session or make_session()

    selected = index[(index["service_date"] >= start) & (index["service_date"] <= end)]
    if selected.empty:
        raise ValueError(f"no LAMP files between {start} and {end}")

    jobs: list[tuple[str, Path]] = []
    for row in selected.itertuples():
        url = row.file_url if isinstance(row.file_url, str) and row.file_url else config.SUBWAY_PERF_URL_TMPL.format(
            service_date=config.format_service_date(row.service_date)
        )
        dest = config.LAMP_RAW_DIR / f"{config.format_service_date(row.service_date)}.parquet"
        jobs.append((url, dest))

    pending = [(u, d) for u, d in jobs if not (d.exists() and d.stat().st_size > 0)]
    log.info(
        "LAMP performance: %d dates in window, %d already cached, %d to download",
        len(jobs), len(jobs) - len(pending), len(pending),
    )

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download, url, dest, make_session()): dest
                for url, dest in pending
            }
            failures = 0
            with Progress(len(pending), "LAMP downloads", log_every=20) as progress:
                for future in as_completed(futures):
                    dest = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        failures += 1
                        log.error("failed to download %s: %s", dest.name, exc)
                        progress.step(dest.name, extra="FAILED")
                        continue
                    progress.step(dest.name)
            if failures:
                # A missing day would otherwise be silently absent from the analysis.
                raise RuntimeError(
                    f"{failures} of {len(pending)} LAMP downloads failed; re-run the "
                    "collect stage (cached files are kept)"
                )

    return [dest for _, dest in jobs if dest.exists() and dest.stat().st_size > 0]


def load_performance(
    start: date,
    end: date,
    *,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load cached performance Parquet files for ``[start, end]`` into one frame.

    Missing dates are skipped and reported, so a partial cache still produces a
    usable training set.
    """
    wanted = columns or PERFORMANCE_COLUMNS
    paths = sorted(config.LAMP_RAW_DIR.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no LAMP parquet in {config.LAMP_RAW_DIR}; run the `collect` stage first"
        )

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for path in paths:
        try:
            service_date = config.parse_service_date(path.stem)
        except ValueError:
            continue
        if not (start <= service_date <= end):
            continue
        try:
            # Project at read time: only the wanted columns are decoded.
            available = set(pq.read_schema(path).names)
            frame = pd.read_parquet(path, columns=[c for c in wanted if c in available])
        except Exception as exc:  # noqa: BLE001 - a corrupt cache file must not be fatal
            log.warning("unreadable parquet %s (%s)", path.name, exc)
            missing.append(path.stem)
            continue
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"no readable LAMP parquet between {start} and {end}")
    if missing:
        log.warning("skipped %d unreadable file(s): %s", len(missing), ", ".join(missing))

    combined = pd.concat(frames, ignore_index=True, copy=False)
    log.info("loaded %s rows across %d LAMP file(s)", f"{len(combined):,}", len(frames))
    return combined


# ---------------------------------------------------------------------------
# Static reference tables and alerts
# ---------------------------------------------------------------------------
def covers(path: Path, end: date) -> bool:
    """Whether a cached archive can hold records up to ``end``.

    The alerts and stops tables are snapshots of an ever-growing archive, so one
    downloaded on or before the window's last day cannot cover the whole window.
    """
    return path.exists() and date.fromtimestamp(path.stat().st_mtime) > end


def download_static(session: requests.Session | None = None, *, workers: int = 4,
                    refresh: bool = False) -> list[Path]:
    """Download the static tables in :data:`config.STATIC_TABLES` (the stops table)."""
    config.STATIC_RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = session or make_session()

    def _one(table: str) -> Path:
        url = config.STATIC_TABLE_URL_TMPL.format(table=table)
        dest = config.STATIC_RAW_DIR / f"{table}.parquet"
        return download(url, dest, make_session(), skip_if_exists=not refresh)

    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, t): t for t in config.STATIC_TABLES}
        for future in as_completed(futures):
            table = futures[future]
            try:
                paths.append(future.result())
            except Exception as exc:  # noqa: BLE001
                # Static tables enrich the analysis but are not essential, so a
                # failure here must not abort the collection stage.
                log.warning("failed to download %s: %s", table, exc)
    return paths


def download_alerts(session: requests.Session | None = None, *, refresh: bool = False) -> Path | None:
    """Download the archived GTFS-Realtime alerts feed (text + cause/effect labels)."""
    config.ALERTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = session or make_session()
    try:
        return download(config.ALERTS_URL, ALERTS_PATH, session, skip_if_exists=not refresh)
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to download alerts: %s", exc)
        return None


def load_static(table: str, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a cached static table by name (e.g. ``LAMP_static_stops``).

    The stops table holds every static-schedule version (~7.5M rows), so callers
    should pass ``columns`` to decode only what they need.
    """
    path = config.STATIC_RAW_DIR / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the `collect` stage first")
    return pd.read_parquet(path, columns=columns)


def load_alerts(*, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the archived alerts feed (~5M rows back to 2019; pass ``columns``)."""
    if not ALERTS_PATH.exists():
        raise FileNotFoundError(f"{ALERTS_PATH} missing; run the `collect` stage first")
    if columns is not None:
        available = set(pq.read_schema(ALERTS_PATH).names)
        columns = [c for c in columns if c in available]
    return pd.read_parquet(ALERTS_PATH, columns=columns)


def collect(
    days: int = 90,
    *,
    start: date | None = None,
    end: date | None = None,
    include_static: bool = True,
    include_alerts: bool = True,
    refresh: bool = False,
) -> dict:
    """Full collection stage for this source. Returns a small summary dict.

    ``start``/``end`` override the length-based window; the CLI uses this to
    download exactly the window shared with the ridership dataset. The stops and
    alerts snapshots are re-downloaded when ``refresh`` is set or when the cached
    copy predates the window's end (:func:`covers`).
    """
    config.ensure_dirs()
    session = make_session()
    index = fetch_index(session)

    if start is None or end is None:
        resolved_start, resolved_end = resolve_window(index, days, end=end)
        start = start or resolved_start
        end = end or resolved_end
    log.info("LAMP window: %s -> %s", start, end)

    paths = download_performance(index, start, end, session=session)
    summary = {
        "service_start": start,
        "service_end": end,
        "files": len(paths),
        "coverage": coverage(index),
    }
    if include_static:
        summary["static_files"] = len(download_static(
            session, refresh=refresh or not covers(STOPS_PATH, end)))
    if include_alerts:
        summary["alerts"] = download_alerts(
            session, refresh=refresh or not covers(ALERTS_PATH, end)) is not None
    return summary
