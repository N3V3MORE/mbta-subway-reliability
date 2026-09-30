"""Where the T loses time: running time lost between stations, platform time lost at them.

Every train's trip is split into the stretches of track it ran (one stop to the
next) and the platforms it stood at. Each is compared with how long that stretch
or platform takes on a good run: its own 10th percentile over the window, on the
same line and in the same direction. Time beyond that is *lost*. Adding it up
over every train shows which few places cost the network the most time, and
whether a line loses it moving or standing.

The timetable is not the yardstick. It schedules whole minutes between stops
(mostly 2 or 3) and pads some stretches, so trains appear to "gain" time there;
a stretch's own good runs are the fair comparison. "Lost" is time beyond a good
run, not time wasted: a crowded platform at rush hour needs longer doors-open.

Two measures per place, because they answer different questions:

* ``median_lost`` -- what a typical train loses there (how bad the place is);
* ``lost_minutes_per_day`` -- train-minutes lost there per service day (what it
  costs the network; a busy place scores high even at a small loss per train).

The source records running time (``travel_time_seconds``: from leaving the
previous stop to arriving) and platform time (``dwell_time_seconds``) for each
stop, and matches its own timestamps 99.85% of the time. It records no platform
time at a trip's first stop, so time lost waiting at an origin is not counted.

Run with ``python -m mbta_ds.cli segments``; results go to the run's processed
folder (``segments.json`` and two parquet tables) and ``reports/tables``.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import clean, config, features

log = logging.getLogger(__name__)

SUMMARY_PATH = config.PROCESSED_DIR / "segments.json"
STRETCHES_PATH = config.PROCESSED_DIR / "segments_stretches.parquet"
PLATFORMS_PATH = config.PROCESSED_DIR / "segments_platforms.parquet"

#: A good run: this quantile of a place's own times.
REFERENCE_QUANTILE = 0.10
#: A stop pair must be at least this share of departures from its station (line
#: and direction), and seen this often, to be a real stretch. Rarer pairs span a
#: stop that went unrecorded, so their "running time" covers two stretches.
MIN_PAIR_SHARE = 0.02
MIN_TRAINS = 300
#: Longer "running times" are recording faults (up to 16 h in spring), not trains.
MAX_RUNNING_SECONDS = 3_600
#: A station is a terminal, in a direction, if most trips reaching it end there.
TERMINAL_SHARE = 0.5

LINE_KEY = ["line", "direction_id"]
STRETCH_KEY = LINE_KEY + ["from_station", "to_station"]
PLATFORM_KEY = LINE_KEY + ["station"]

_COLUMNS = ["service_date", "trip_id", "run", "route_id", "direction_id", "parent_station", "station_name",
            "scheduled_arrival_time", "stop_sequence", "travel_time_seconds", "dwell_time_seconds",
            "scheduled_hour", "is_weekend", "time_inconsistent"]


def _line(route_id: pd.Series) -> pd.Series:
    """Trunk line: the Green Line branches share their tunnel, Mattapan is its own."""
    return route_id.astype(str).str.split("-").str[0]


def traversals(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per train per stretch it ran: the stop it left, the stop it reached, how long it took.

    Consecutive stops of one run (one vehicle), in scheduled order; a time that
    runs backwards (``time_inconsistent``) is not a running time.
    """
    frame = frame.sort_values(clean.TRIP_ORDER, kind="stable")
    run = frame.groupby(clean.RUN_KEY, sort=False)
    out = pd.DataFrame({
        "line": _line(frame["route_id"]), "direction_id": frame["direction_id"],
        "from_station": run["parent_station"].shift(1), "to_station": frame["parent_station"],
        "service_date": frame["service_date"], "scheduled_hour": frame["scheduled_hour"],
        "is_weekend": frame["is_weekend"], "seconds": frame["travel_time_seconds"],
    })
    ok = out["from_station"].notna() & out["seconds"].notna() & ~frame["time_inconsistent"]
    ok &= out["seconds"] <= MAX_RUNNING_SECONDS
    out = out[ok]
    # Keep real stretches only (see MIN_PAIR_SHARE).
    counts = out.groupby(STRETCH_KEY).size()
    share = counts / counts.groupby(level=LINE_KEY + ["from_station"]).transform("sum")
    real = share[(share >= MIN_PAIR_SHARE) & (counts >= MIN_TRAINS)].index
    return out[pd.MultiIndex.from_frame(out[STRETCH_KEY]).isin(real)].reset_index(drop=True)


def platform_visits(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per train per platform it stood at and then left for another stop, with how long it stood.

    Time at a run's last stop is a layover before the next trip, not time lost:
    at Cleveland Circle it is a median 578 s. It is 0.2% of platform records and
    is left out.
    """
    frame = frame.sort_values(clean.TRIP_ORDER, kind="stable")
    frame = frame[frame.groupby(clean.RUN_KEY, sort=False).cumcount(ascending=False) > 0]
    out = pd.DataFrame({
        "line": _line(frame["route_id"]), "direction_id": frame["direction_id"], "station": frame["parent_station"],
        "service_date": frame["service_date"], "scheduled_hour": frame["scheduled_hour"],
        "is_weekend": frame["is_weekend"], "seconds": frame["dwell_time_seconds"],
    })
    out = out[out["seconds"].notna()]
    counts = out.groupby(PLATFORM_KEY)["seconds"].transform("size")
    return out[counts >= MIN_TRAINS].reset_index(drop=True)


def add_lost(rows: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Seconds beyond the place's good run (its REFERENCE_QUANTILE), never below zero."""
    reference = rows.groupby(key)["seconds"].quantile(REFERENCE_QUANTILE).rename("reference")
    rows = rows.join(reference, on=key)
    return rows.assign(lost=(rows["seconds"] - rows["reference"]).clip(lower=0))


def summarise(rows: pd.DataFrame, key: list[str], days: int) -> pd.DataFrame:
    """Per place: trains, its good-run time, what a typical and a bad train lose, and the daily total."""
    by = rows.groupby(key)
    table = pd.DataFrame({
        "trains": by.size(),
        "reference_seconds": by["reference"].first(),
        "median_seconds": by["seconds"].median(),
        "median_lost": by["lost"].median(),
        "p90_lost": by["lost"].quantile(0.9),
        "mean_lost": by["lost"].mean(),
        "lost_minutes_per_day": by["lost"].sum() / 60 / days,
    }).reset_index()
    return table.sort_values("lost_minutes_per_day", ascending=False).reset_index(drop=True)


def terminals(frame: pd.DataFrame) -> set[tuple[str, int, str]]:
    """(line, direction, station) where most trips reaching the station end there."""
    frame = frame.sort_values(clean.TRIP_ORDER, kind="stable")
    last = frame.groupby(clean.TRIP_KEY, sort=False).cumcount(ascending=False) == 0
    ends = last.groupby([_line(frame["route_id"]), frame["direction_id"], frame["parent_station"]]).mean()
    return {tuple(k) for k in ends[ends >= TERMINAL_SHARE].index}


def _peak(rows: pd.DataFrame) -> pd.Series:
    return rows["scheduled_hour"].isin(features.PEAK_HOURS) & ~rows["is_weekend"].astype(bool)


def network_summary(running: pd.DataFrame, platform: pd.DataFrame, stretches: pd.DataFrame,
                    platforms: pd.DataFrame, ends: set, days: int) -> dict:
    """The headline numbers: how much is lost, where, and when."""
    run_total, plat_total = running["lost"].sum(), platform["lost"].sum()
    # Concentration: the share of all lost time in the worst tenth of places.
    places = pd.concat([stretches["lost_minutes_per_day"], platforms["lost_minutes_per_day"]]).sort_values(ascending=False)
    top = max(1, len(places) // 10)
    into_end = stretches.apply(lambda r: (r["line"], r["direction_id"], r["to_station"]) in ends, axis=1)
    both = pd.concat([running.assign(kind="running"), platform.assign(kind="platform")], ignore_index=True)
    peak = _peak(both)
    lines = both.groupby(["line", "kind"])["lost"].agg(["mean", "sum"]).unstack("kind")
    hours = both.groupby(["scheduled_hour", "kind"])["lost"].mean().unstack("kind")
    return {
        "days": days,
        "lost_minutes_per_day": float((run_total + plat_total) / 60 / days),
        "running_share": float(run_total / (run_total + plat_total)),
        "stretches": len(stretches),
        "platforms": len(platforms),
        "top_tenth_share": float(places.head(top).sum() / places.sum()),
        "per_visit": {  # seconds lost by a train at one stretch or platform
            "peak_median": float(both.loc[peak, "lost"].median()), "offpeak_median": float(both.loc[~peak, "lost"].median()),
            "peak_mean": float(both.loc[peak, "lost"].mean()), "offpeak_mean": float(both.loc[~peak, "lost"].mean()),
        },
        "into_terminal": {  # a typical train's running time lost on the last stretch into a terminal
            "median_lost": float(stretches.loc[into_end, "median_lost"].median()),
            "elsewhere_median_lost": float(stretches.loc[~into_end, "median_lost"].median()),
            "stretches": int(into_end.sum()),
        },
        "by_line": [{"line": line, "running_mean": float(row[("mean", "running")]),
                     "platform_mean": float(row[("mean", "platform")]),
                     "running_share": float(row[("sum", "running")] / (row[("sum", "running")] + row[("sum", "platform")]))}
                    for line, row in lines.iterrows()],
        "by_hour": [{"hour": int(hour), "running_mean": float(row.get("running", np.nan)),
                     "platform_mean": float(row.get("platform", np.nan))} for hour, row in hours.iterrows()],
    }


def run() -> dict:
    """Compute where the run's trains lost time and persist the tables and summary."""
    config.ensure_dirs()
    frame = pd.read_parquet(clean.OUT_PATH, columns=_COLUMNS)
    days = int(frame["service_date"].nunique())
    names = frame.drop_duplicates("parent_station").set_index("parent_station")["station_name"].astype(str)

    running = add_lost(traversals(frame), STRETCH_KEY)
    platform = add_lost(platform_visits(frame), PLATFORM_KEY)
    stretches = summarise(running, STRETCH_KEY, days)
    platforms = summarise(platform, PLATFORM_KEY, days)
    ends = terminals(frame)
    summary = network_summary(running, platform, stretches, platforms, ends, days)

    stretches["into_terminal"] = [(r.line, r.direction_id, r.to_station) in ends for r in stretches.itertuples()]
    stretches.to_parquet(STRETCHES_PATH, index=False)
    platforms.to_parquet(PLATFORMS_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readable = {"from_station": "From", "to_station": "To", "station": "Station", "line": "Line",
                "direction_id": "Direction", "trains": "Trains", "reference_seconds": "Good run (s)",
                "median_lost": "Typical train loses (s)", "p90_lost": "1 in 10 loses (s)",
                "lost_minutes_per_day": "Train-minutes lost a day"}
    for table, name in ((stretches, "segments_running"), (platforms, "segments_platforms")):
        out = table.replace({c: names for c in ("from_station", "to_station", "station") if c in table})
        out[[c for c in readable if c in out]].rename(columns=readable).round(1).to_csv(
            config.TABLES_DIR / f"{name}.csv", index=False)

    log.info("time lost: %.0f train-minutes a day, %.0f%% of it running; worst tenth of places carry %.0f%%",
             summary["lost_minutes_per_day"], 100 * summary["running_share"], 100 * summary["top_tenth_share"])
    return {"stretches": len(stretches), "platforms": len(platforms),
            "lost_minutes_per_day": round(summary["lost_minutes_per_day"])}


def load() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """The persisted summary and the stretch and platform tables."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{SUMMARY_PATH} missing; run the `segments` stage first")
    return (json.loads(SUMMARY_PATH.read_text(encoding="utf-8")),
            pd.read_parquet(STRETCHES_PATH), pd.read_parquet(PLATFORMS_PATH))
