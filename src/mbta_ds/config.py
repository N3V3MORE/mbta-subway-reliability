"""Central configuration: paths, dataset URLs, and environment loading.

Everything path- or endpoint-related lives here so the rest of the package never
hard-codes a URL or assumes a working directory.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# config.py -> mbta_ds -> src -> <repo root>
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

#: A named run (``MBTA_RUN=winter``) keeps everything tied to its date window --
#: the window, ridership, weather, processed tables and reports -- in folders of
#: its own, so several windows coexist. Downloads that do not depend on the
#: window (daily performance files, stops, alerts) stay shared.
RUN = os.environ.get("MBTA_RUN", "").strip()
RUN_DIR = DATA_DIR / "runs" / RUN if RUN else DATA_DIR
RUN_RAW_DIR = RUN_DIR if RUN else RAW_DIR
PROCESSED_DIR = RUN_DIR / "processed"
#: Everything a person reads: the report page, its figures and its tables.
REPORTS_DIR = ROOT / "reports" / RUN if RUN else ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

LAMP_RAW_DIR = RAW_DIR / "lamp"          # per-service-date performance parquet
STATIC_RAW_DIR = RAW_DIR / "static"      # the stops table
ALERTS_RAW_DIR = RAW_DIR / "alerts"
RIDERSHIP_RAW_DIR = RUN_RAW_DIR / "ridership"
WEATHER_RAW_DIR = RUN_RAW_DIR / "weather"
V3_RAW_DIR = RAW_DIR / "v3"              # live collector output

ALL_DIRS = [
    RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR,
    LAMP_RAW_DIR, STATIC_RAW_DIR, ALERTS_RAW_DIR,
    RIDERSHIP_RAW_DIR, WEATHER_RAW_DIR, V3_RAW_DIR,
]


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Source A -- LAMP public performance archive (no API key required)
# --------------------------------------------------------------------------
LAMP_BASE = "https://performancedata.mbta.com/lamp"

# index.csv lists every published service date, its size, and its file URL.
SUBWAY_PERF_INDEX_URL = f"{LAMP_BASE}/subway-on-time-performance-v1/index.csv"
SUBWAY_PERF_URL_TMPL = (
    LAMP_BASE
    + "/subway-on-time-performance-v1/{service_date}-subway-on-time-performance-v1.parquet"
)

# Static GTFS-derived tables and the archived GTFS-Realtime alerts feed.
STATIC_TABLE_URL_TMPL = LAMP_BASE + "/tableau/rail/{table}.parquet"
#: Only the stops table is read (for station names). LAMP also publishes
#: routes / trips / stop_times, but nothing in the pipeline uses them and
#: stop_times alone is ~2.3 GB, so they are not downloaded.
STATIC_TABLES = ("LAMP_static_stops",)
ALERTS_URL = LAMP_BASE + "/tableau/alerts/LAMP_RT_ALERTS.parquet"

# --------------------------------------------------------------------------
# Source B -- MBTA V3 API (live data; API key optional)
# --------------------------------------------------------------------------
V3_BASE = "https://api-v3.mbta.com"
V3_PREDICTIONS_URL = f"{V3_BASE}/predictions"
V3_VEHICLES_URL = f"{V3_BASE}/vehicles"
V3_ALERTS_URL = f"{V3_BASE}/alerts"

#: Route IDs treated as "subway" (heavy rail + light rail + Mattapan trolley).
SUBWAY_ROUTES = (
    "Red", "Orange", "Blue", "Green-B", "Green-C", "Green-D", "Green-E", "Mattapan",
)

# --------------------------------------------------------------------------
# Source C -- MBTA gated station entries (ArcGIS Hub / FeatureServer)
# --------------------------------------------------------------------------
GATED_ENTRIES_QUERY_URL = (
    "https://services1.arcgis.com/ceiitspzDAHrdGO1/arcgis/rest/services"
    "/GSE/FeatureServer/0/query"
)
GATED_ENTRIES_PAGE_SIZE = 2000  # FeatureServer maxRecordCount

# --------------------------------------------------------------------------
# Source D -- Open-Meteo historical weather (free, no key)
# --------------------------------------------------------------------------
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_LAT = 42.3601   # downtown Boston
WEATHER_LON = -71.0589
WEATHER_HOURLY_VARS = (
    "temperature_2m",
    "precipitation",
    "snowfall",
    "windspeed_10m",
    "relative_humidity_2m",
)

# --------------------------------------------------------------------------
# Domain constants
# --------------------------------------------------------------------------
SERVICE_TZ = "America/New_York"
#: The MBTA service day runs 03:00 -> 02:59 the following day, so GTFS times for
#: post-midnight trips exceed 86400 seconds (see ``clean.compute_delay``).
SECONDS_PER_DAY = 86_400

#: A train is "late" for classification purposes beyond this threshold.
LATE_THRESHOLD_SECONDS = 300


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
def load_env(path: Path | None = None) -> None:
    """Populate ``os.environ`` from a ``.env`` file (existing vars win).

    A deliberately tiny loader so the project needs no extra dependency.
    """
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str | None:
    """Return the MBTA V3 API key if configured, else ``None``.

    ``None`` is a supported mode: the live collector stays under the anonymous
    rate limit and every bulk source is public.
    """
    load_env()
    key = os.environ.get("MBTA_API_KEY", "").strip()
    return key or None


# --------------------------------------------------------------------------
# Date-window helpers
# --------------------------------------------------------------------------
def parse_service_date(value: object) -> date:
    """Parse a LAMP service date (``int`` ``20260131`` or ``str``) into a date."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if text.isdigit() and len(text) == 8:
        return datetime.strptime(text, "%Y%m%d").date()
    return datetime.fromisoformat(text).date()


def format_service_date(value: date) -> str:
    """Format a date the way LAMP file names and filters expect (``YYYY-MM-DD``)."""
    return value.strftime("%Y-%m-%d")



# --------------------------------------------------------------------------
# Analysis-window manifest
# --------------------------------------------------------------------------
# The collection stage resolves ONE window shared by every source (LAMP is the
# long archive, ridership the short one). Persisting it means later stages cannot
# silently disagree about which dates are under analysis.
WINDOW_PATH = RUN_DIR / "analysis_window.json"


def save_window(start: date, end: date) -> None:
    """Persist the resolved analysis window for downstream stages."""
    import json

    WINDOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    WINDOW_PATH.write_text(
        json.dumps({"start": start.isoformat(), "end": end.isoformat()}, indent=2),
        encoding="utf-8",
    )


def load_window() -> tuple[date, date] | None:
    """Read the persisted analysis window, or ``None`` if collection has not run."""
    import json

    if not WINDOW_PATH.exists():
        return None
    data = json.loads(WINDOW_PATH.read_text(encoding="utf-8"))
    return date.fromisoformat(data["start"]), date.fromisoformat(data["end"])
