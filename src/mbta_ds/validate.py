"""Data-validation stage: physical and bookkeeping checks on every data source.

Each check is a measured value compared against a limit. Every check runs and is
reported, so one pass shows every problem at once; the CLI exits non-zero if a
*hard* check fails. *Info* checks describe the data without judging it (planned
shutdowns, for instance, are real events rather than errors).
"""

from __future__ import annotations

import logging
import operator

import pandas as pd

from . import clean, collect_ridership, collect_weather, config, features

log = logging.getLogger(__name__)

OUT_PATH = config.PROCESSED_DIR / "validation.csv"

#: A route-day with fewer records than this share of the route's usual count for
#: that weekday is reported as a (likely planned) service reduction.
DIVERSION_SHARE = 0.6

_OPS = {"<=": operator.le, ">=": operator.ge, "==": operator.eq}


def _check(source: str, name: str, value: float, op: str, limit: float, *,
           hard: bool = True) -> dict:
    return {"source": source, "check": name, "value": round(float(value), 4), "op": op,
            "limit": limit, "hard": hard, "ok": bool(_OPS[op](value, limit))}


def check_clean(frame: pd.DataFrame) -> list[dict]:
    """Physical consistency of the tidy trip-stop table."""
    trip = frame.sort_values(clean.TRIP_ORDER).groupby(clean.TRIP_KEY, sort=False)
    dates = pd.to_datetime(pd.Series(frame["service_date_parsed"].unique()))
    arrivals = frame[clean.is_arrival(frame)]

    per_day = frame.groupby(["service_date_parsed", "route_id"]).size().unstack(fill_value=0)
    weekday = pd.to_datetime(pd.Series(per_day.index)).dt.dayofweek.to_numpy()
    reduced = (per_day / per_day.groupby(weekday).transform("median")) < DIVERSION_SHARE

    both = frame["move_timestamp"].notna() & frame["travel_time_seconds"].notna()
    travel_matches = (frame["stop_timestamp"] - frame["move_timestamp"])[both] == \
        frame.loc[both, "travel_time_seconds"]
    c = "clean"
    return [
        _check(c, "service dates missing inside the window",
               (dates.max() - dates.min()).days + 1 - len(dates), "==", 0),
        _check(c, "scheduled time runs backwards within a trip",
               (trip["scheduled_arrival_time"].diff() < 0).sum(), "==", 0),
        _check(c, "share of arrivals timestamped before the previous stop (flagged)",
               frame["time_inconsistent"].mean(), "<=", 0.001),
        _check(c, "duplicate (service_date, trip_id, stop_id)",
               frame.duplicated(clean.TRIP_KEY + ["stop_id"]).sum(), "==", 0),
        _check(c, "duplicate visits under a second platform id",
               frame.duplicated(clean.TRIP_KEY + ["parent_station", "scheduled_arrival_time"]).sum(),
               "==", 0),
        _check(c, "negative travel, dwell or headway values",
               (frame[["travel_time_seconds", "dwell_time_seconds", "headway_trunk_seconds"]] < 0)
               .sum().sum(), "==", 0),
        _check(c, "share of rows where travel_time == stop_ts - move_ts",
               travel_matches.mean(), ">=", 0.999),
        _check(c, "trips running more than 30 min early on average (mismatched)",
               (trip["delay_seconds"].median() < -clean.MISMATCH_EARLY_SECONDS).sum(), "==", 0),
        _check(c, "share of arrivals more than 1 h off schedule",
               arrivals["delay_outlier"].mean(), "<=", 0.005),
        _check(c, "|median arrival delay| in seconds (sanity)",
               abs(arrivals["delay_seconds"].median()), "<=", 120),
        _check(c, "stations left as raw ids (name lookup failed)",
               frame["station_name"].str.startswith("place-").sum(), "==", 0),
        _check(c, "route-days below 60% of usual volume (shutdowns)",
               reduced.to_numpy().sum(), ">=", 0, hard=False),
    ]


def check_ridership(raw: pd.DataFrame, n_days: int) -> list[dict]:
    r = "ridership"
    return [
        _check(r, "negative gated entries", (raw["gated_entries"] < 0).sum(), "==", 0),
        _check(r, "duplicate (date, stop, line, period)",
               raw.duplicated(["service_date", "stop_id", "route_or_line", "time_period"]).sum(),
               "==", 0),
        _check(r, "service dates covered", raw["service_date"].nunique(), "==", n_days),
        _check(r, "half-hour periods other than :00 / :30",
               (~raw["period_minute"].isin([0, 30])).sum(), "==", 0),
    ]


def check_weather(weather: pd.DataFrame) -> list[dict]:
    w = "weather"
    steps = weather["timestamp"].diff().dropna()
    return [
        _check(w, "gaps or repeats in the hourly series",
               (steps != pd.Timedelta(hours=1)).sum(), "==", 0),
        _check(w, "missing values", weather.isna().sum().sum(), "==", 0),
        _check(w, "temperatures outside -30..45 C",
               (~weather["temperature_c"].between(-30, 45)).sum(), "==", 0),
        _check(w, "humidity outside 0..100 %",
               (~weather["humidity_pct"].between(0, 100)).sum(), "==", 0),
        _check(w, "negative precipitation or snowfall",
               (weather[["precip_mm", "snowfall_cm"]] < 0).sum().sum(), "==", 0),
    ]


def check_features(frame: pd.DataFrame) -> list[dict]:
    f = "features"
    declared = features.all_feature_columns()
    return [
        _check(f, "target columns among the features",
               len(set(declared) & set(features.TARGET_COLUMNS)), "==", 0),
        _check(f, "declared features absent from the table",
               len(set(declared) - set(frame.columns)), "==", 0),
        _check(f, "fraction_through_trip outside 0..1",
               (~frame["fraction_through_trip"].between(0, 1)).sum(), "==", 0),
        # Origins are not targets, so every target row has a previous stop.
        _check(f, "share of rows without a previous-stop delay",
               frame["prev_delay_1"].isna().mean(), "<=", 0.001),
    ]


def run() -> dict:
    """Run every check, persist the report, and summarise the outcome."""
    clean_frame = clean.load()
    n_days = clean_frame["service_date_parsed"].nunique()
    results = check_clean(clean_frame) + check_features(features.load())
    for loader, checker in ((collect_ridership.load_raw, lambda d: check_ridership(d, n_days)),
                            (collect_weather.load_weather, check_weather)):
        try:
            results += checker(loader())
        except FileNotFoundError as exc:
            log.warning("skipping checks: %s", exc)

    report = pd.DataFrame(results)
    report.to_csv(OUT_PATH, index=False)
    for row in report.itertuples():
        status = "info" if not row.hard else ("ok  " if row.ok else "FAIL")
        log.info("%s %-10s %-62s %12s %s %s", status, row.source, row.check, row.value, row.op, row.limit)
    failed = report[report["hard"] & ~report["ok"]]
    log.info("validation: %d checks, %d hard failure(s) -> %s", len(report), len(failed), OUT_PATH.name)
    return {"checks": len(report), "failed": failed["check"].tolist()}
