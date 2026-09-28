"""Collector for hourly Boston weather (Open-Meteo historical archive).

Weather is the main *exogenous* feature in the delay model: snow and heavy rain
are the obvious candidates for pushing headways off schedule. Open-Meteo's
archive endpoint is free, keyless, and timezone-aware, which keeps this source
fully reproducible.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from . import config
from .http import get_json, make_session

log = logging.getLogger(__name__)

OUT_PATH = config.WEATHER_RAW_DIR / "boston_hourly.parquet"


def fetch_weather(
    start: date,
    end: date,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch hourly weather for ``[start, end]`` in local (Eastern) time.

    The archive lags real time by a few days. When the requested window runs past
    what is published, the window is clipped and the shortfall is logged -- the
    feature stage then imputes the tail and flags it rather than dropping rows.
    """
    session = session or make_session()
    params = {
        "latitude": config.WEATHER_LAT,
        "longitude": config.WEATHER_LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(config.WEATHER_HOURLY_VARS),
        "timezone": config.SERVICE_TZ,
    }

    try:
        payload = get_json(config.OPEN_METEO_ARCHIVE_URL, session, params=params, timeout=120)
    except Exception as exc:  # noqa: BLE001
        # Most likely cause: the window extends past the archive's publish lag.
        log.warning("weather request failed (%s); retrying with a 7-day buffer", exc)
        buffered_end = pd.Timestamp(end) - pd.Timedelta(days=7)
        params["end_date"] = buffered_end.date().isoformat()
        payload = get_json(config.OPEN_METEO_ARCHIVE_URL, session, params=params, timeout=120)

    if "hourly" not in payload:
        raise RuntimeError(f"unexpected Open-Meteo response: {list(payload)[:10]}")

    hourly = payload["hourly"]
    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    for name in config.WEATHER_HOURLY_VARS:
        frame[name] = pd.to_numeric(pd.Series(hourly.get(name)), errors="coerce")

    frame = frame.rename(
        columns={
            "temperature_2m": "temperature_c",
            "precipitation": "precip_mm",
            "snowfall": "snowfall_cm",
            "windspeed_10m": "wind_kph",
            "relative_humidity_2m": "humidity_pct",
        }
    )

    if frame["timestamp"].max().date() < end:
        log.warning(
            "weather archive only covers up to %s (requested %s); "
            "later hours will be imputed",
            frame["timestamp"].max().date(), end,
        )
    return frame


def collect(days: int, *, end: date | None = None, refresh: bool = False) -> dict:
    """Fetch and cache hourly weather covering the analysis window."""
    config.ensure_dirs()

    if end is None:
        end = pd.Timestamp.now(tz="UTC").date()  # Timestamp.utcnow() is deprecated
    start = pd.Timestamp(end) - pd.Timedelta(days=days - 1)
    start = start.date()
    # The last service day runs past midnight (to ~01:30), so fetch the next day too.
    end = (pd.Timestamp(end) + pd.Timedelta(days=1)).date()

    if not refresh and OUT_PATH.exists():
        cached = pd.read_parquet(OUT_PATH)
        if not cached.empty:
            covered = (
                cached["timestamp"].min().date() <= start
                and cached["timestamp"].max() >= pd.Timestamp(end) + pd.Timedelta(hours=23)
            )
            if covered:
                log.info("weather cache already covers %s -> %s", start, end)
                return {
                    "service_start": cached["timestamp"].min().date(),
                    "service_end": cached["timestamp"].max().date(),
                    "rows": len(cached),
                }

    frame = fetch_weather(start, end)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    log.info("cached %s hourly weather rows to %s", f"{len(frame):,}", OUT_PATH.name)
    return {
        "service_start": frame["timestamp"].min().date(),
        "service_end": frame["timestamp"].max().date(),
        "rows": len(frame),
    }


def load_weather() -> pd.DataFrame:
    """Load cached hourly weather."""
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"{OUT_PATH} missing; run the `collect` stage first")
    return pd.read_parquet(OUT_PATH)
