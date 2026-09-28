"""Cleaning stage: turn raw LAMP performance records into a tidy delay table.

Every decision here was made after inspecting real data rather than assumed up
front. The measurements that drove them (on a 5-day sample, 320,735 rows) are
recorded next to each rule, and each run's row counts are written by :class:`Ledger`.

Key findings that shape the cleaning
------------------------------------
1. ``trip_id`` comes in three namespaces. ``numeric`` trips are scheduled service
   and their delay distribution is textbook (median +63 s, 5th-95th percentile
   -458 s to +956 s). ``ADDED-*`` trips are created at runtime and ``NONREV-*``
   trips are non-revenue moves; neither has a meaningful entry in the planned
   schedule, and their "delays" are garbage (medians of -9,817 s and -18,734 s).
   They are therefore excluded -- not because they are outliers, but because the
   quantity "delay" is not defined for them.
2. ``trip_id`` is reused across service dates (2,123 of 13,305 trip ids appear on
   more than one date), so any trip-level grouping must key on
   ``(service_date, trip_id)``. Grouping on ``trip_id`` alone silently mixes days
   and produces nonsense.
3. ``parent_station`` holds a *stop id* (e.g. ``place-alfcl``), not a name, despite
   the data dictionary's wording. Names come from ``LAMP_static_stops``.
4. The service day runs 03:00 -> 02:59, so post-midnight trips carry GTFS times
   above 86400 s. GTFS measures times from "noon minus 12 hours" of the service
   date, which is local midnight on every day except the two DST transitions,
   where it is 23:00 or 01:00. Anchoring there and adding the raw GTFS time
   handles post-midnight trips and DST without special-casing. (Anchoring at
   midnight instead put every delay on 8 March 2026 off by -3,600 s.)
5. ``stop_sequence`` is not a reliable trip order. On 3,830 trips (mostly Green-E
   into Heath Street) the *final* stop is labelled ``stop_sequence == 1``, so
   ordering by it puts the end of the trip first and hands the lag features the
   future. The scheduled arrival time is monotone by construction and is used
   instead (:data:`TRIP_ORDER`).
6. The first stop of a trip is not an arrival delay: trains reach the origin
   platform and wait, a median 190 s before the scheduled departure. Those rows
   are kept as lag sources but excluded wherever delay is measured
   (:func:`is_arrival`); left in, they made terminals look punctual.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import collect_lamp, config

log = logging.getLogger(__name__)

OUT_PATH = config.PROCESSED_DIR / "trips_clean.parquet"
LEDGER_PATH = config.PROCESSED_DIR / "clean_ledger.csv"

#: Trip-id namespaces that have no usable planned-schedule match.
EXCLUDED_TRIP_PREFIXES = ("ADDED-", "NONREV-")

#: Beyond this absolute delay a row is kept but flagged rather than dropped, so
#: the report can quantify how much of the tail the models are fighting.
DELAY_OUTLIER_SECONDS = 3600
#: Dwell longer than this is physically implausible for a subway train.
DWELL_OUTLIER_SECONDS = 1800
#: A whole trip running this far *ahead* of its timetable would have had to pass
#: the 3-5 trains scheduled in front of it on the same track. It is a vehicle
#: matched to the wrong scheduled trip (435 trips, typically offset by ~an hour).
MISMATCH_EARLY_SECONDS = 1800
#: A single stop this far from *both* neighbouring stops of its run, while those
#: neighbours agree with each other to within :data:`GLITCH_AGREE_SECONDS`, is a
#: bad record rather than a train: a real hold-up raises the delay and it *stays*
#: raised at the next stop. At a run's first or last stop the one neighbour and the
#: stop beyond it are used. Designed on the spring window (1,232 stops in 3.2M).
GLITCH_SECONDS = 1800
GLITCH_AGREE_SECONDS = 600

#: Default target threshold: "late" means more than 5 minutes behind schedule.
LATE_THRESHOLD_SECONDS = config.LATE_THRESHOLD_SECONDS

#: The true order of stops within a trip (see finding 5 above).
TRIP_KEY = ["service_date", "trip_id"]
TRIP_ORDER = TRIP_KEY + ["scheduled_arrival_time", "stop_sequence"]
#: One vehicle's stretch of a trip. On 300 Green Line trips the vehicle changes
#: mid-trip (delay jumps a median 68 s at the hand-over, against 24 s between
#: ordinary stops), so "the previous stop" must never reach across trains.
RUN_KEY = TRIP_KEY + ["run"]


def isolated_glitches(frame: pd.DataFrame) -> pd.Series:
    """Stops whose delay disagrees with both neighbours while they agree.

    ``frame`` must be in trip order with a ``run`` column. Left in, one such
    record (e.g. a Red Line stop logged 3.4 h early inside an on-time trip) is
    copied forward as the next stop's "previous delay" and costs a prediction
    hours of error.
    """
    delay = frame["delay_seconds"]
    group = frame.groupby(RUN_KEY, sort=False)["delay_seconds"]
    prev, nxt = group.shift(1), group.shift(-1)
    prev2, nxt2 = group.shift(2), group.shift(-2)
    far = lambda a, b: (a - b).abs() > GLITCH_SECONDS          # noqa: E731
    near = lambda a, b: (a - b).abs() <= GLITCH_AGREE_SECONDS  # noqa: E731
    interior = far(delay, prev) & far(delay, nxt) & near(prev, nxt)
    first = prev.isna() & far(delay, nxt) & near(nxt, nxt2)
    last = nxt.isna() & far(delay, prev) & near(prev, prev2)
    return interior | first | last


def is_arrival(frame: pd.DataFrame) -> pd.Series:
    """Rows whose ``delay_seconds`` is a genuine arrival delay.

    Excludes the trip origin (a platform wait, not a delay) and observations
    timestamped before the previous stop's, which cannot be physically true.
    """
    return ~(frame["is_origin"] | frame["time_inconsistent"])


@dataclass
class Ledger:
    """Records how many rows each step removed, for the report."""

    steps: dict[str, int] = field(default_factory=dict)

    def record(self, step: str, before: int, after: int) -> None:
        removed = before - after
        self.steps[step] = removed
        log.info("  %-42s rows %s -> %s  (removed %s)",
                 step, f"{before:,}", f"{after:,}", f"{removed:,}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"step": k, "rows_removed": v} for k, v in self.steps.items()]
        )


# ---------------------------------------------------------------------------
# Step 1 -- schedule anchoring
# ---------------------------------------------------------------------------
def _service_date_table(service_dates: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    """Parse each *distinct* service date once and return ``(codes, table)``.

    A 90-day window has ~3M rows but only ~90 dates, so parsing strings per
    distinct date instead of per row avoids millions of redundant conversions.
    ``table.iloc[codes]`` broadcasts the per-date values back to the rows.
    """
    codes, uniques = pd.factorize(service_dates, sort=False)
    if (codes < 0).any():
        raise ValueError("service_date contains missing values")
    day = pd.to_datetime(pd.Series(uniques).astype("int64").astype(str), format="%Y%m%d")
    # GTFS anchor: local noon minus 12 h. Noon is never ambiguous or skipped.
    noon = (day + pd.Timedelta(hours=12)).dt.tz_localize(config.SERVICE_TZ)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    table = pd.DataFrame({
        "midnight_epoch": (noon - epoch) // pd.Timedelta(seconds=1) - 12 * 3600,
        "day_of_week": day.dt.dayofweek.astype("int8"),
        "date": day.dt.date,
    })
    return codes, table


def service_midnight_epoch(service_dates: pd.Series) -> pd.Series:
    """Convert ``YYYYMMDD`` ints to the GTFS schedule anchor in epoch seconds.

    GTFS times are measured from "noon minus 12 hours" of the service date: local
    midnight on ordinary days, one hour earlier (spring forward) or later (fall
    back) on DST transition days. Every scheduled time is added to this anchor.
    """
    codes, table = _service_date_table(service_dates)
    return pd.Series(
        table["midnight_epoch"].to_numpy()[codes], index=service_dates.index, dtype="int64"
    )


def compute_delay(frame: pd.DataFrame) -> pd.Series:
    """Return delay in seconds: observed stop time minus scheduled arrival.

    Positive means late. Post-midnight trips need no special handling because
    their GTFS times already exceed 86400 s (23,971 of 316,427 rows in the
    5-day sample did).
    """
    midnight = service_midnight_epoch(frame["service_date"])
    scheduled_epoch = midnight + frame["scheduled_arrival_time"]
    return frame["stop_timestamp"] - scheduled_epoch


# ---------------------------------------------------------------------------
# Step 2 -- enrichment from the static GTFS tables
# ---------------------------------------------------------------------------
def _stop_name_lookup(override: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a ``stop_id -> stop_name`` lookup from the static stops table.

    The table carries every static-schedule version since 2019 (750 of them), and
    stations get renamed: keeping the *first* row per id yields 2019 names such as
    "Packards Corner" and "Science Park", which then fail to match the current
    names the V3 API returns for the station map. The latest version wins.
    """
    if override is not None:
        return override
    try:
        stops = collect_lamp.load_static(
            "LAMP_static_stops", columns=["stop_id", "stop_name", "static_version_key"]
        )
    except FileNotFoundError:
        log.warning("static stops table unavailable; station names will be ids")
        return pd.DataFrame(columns=["stop_id", "stop_name"])

    return (
        stops.dropna(subset=["stop_id", "stop_name"])
        .sort_values("static_version_key", kind="stable")
        .drop_duplicates(subset=["stop_id"], keep="last")[["stop_id", "stop_name"]]
    )


def _attach_stop_names(frame: pd.DataFrame,
                       lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resolve ``parent_station`` (really a stop id) to a human-readable name.

    Rows whose parent id has no record fall back to their platform-level
    ``stop_id``, then to the raw id. Lookups are label-aligned ``map`` calls, so
    they stay correct whatever the frame's index looks like after filtering.
    """
    lookup = _stop_name_lookup(lookup)
    if lookup.empty:
        frame["station_name"] = frame["parent_station"]
        return frame

    names = lookup.drop_duplicates(subset=["stop_id"], keep="last").set_index("stop_id")["stop_name"]
    frame["station_name"] = (
        frame["parent_station"].map(names)
        .combine_first(frame["stop_id"].map(names))
        .combine_first(frame["parent_station"])
    )
    return frame


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def build(
    *,
    window: tuple[date, date] | None = None,
    refresh: bool = False,
    raw_frame: pd.DataFrame | None = None,
    stop_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run every cleaning step and return the tidy delay table.

    ``raw_frame`` and ``stop_lookup`` inject inputs directly and suppress caching.
    They exist so the cleaning rules can be unit-tested against small synthetic
    frames without a network download or a filesystem dependency.
    """
    in_memory = raw_frame is not None
    if not in_memory and OUT_PATH.exists() and not refresh:
        log.info("using cached clean table %s", OUT_PATH.name)
        return pd.read_parquet(OUT_PATH)

    config.ensure_dirs()
    ledger = Ledger()

    if in_memory:
        raw = raw_frame
        log.info("cleaning %s injected rows", f"{len(raw):,}")
    else:
        index = collect_lamp.fetch_index()
        # Prefer the window the collection stage resolved and persisted; fall back
        # to a 90-day LAMP window when running this stage standalone.
        resolved = (window or config.load_window()
                    or collect_lamp.resolve_window(index, days=90))
        start, end = resolved
        log.info("cleaning window %s -> %s", start, end)
        raw = collect_lamp.load_performance(start, end)
    ledger.record("load raw performance rows", len(raw), len(raw))

    # Every rule below filters into a new frame, so the input is never mutated
    # and an up-front defensive copy is unnecessary.
    frame = raw

    # --- rule 1: keep only scheduled revenue service ---------------------
    before = len(frame)
    is_excluded = frame["trip_id"].astype(str).str.startswith(EXCLUDED_TRIP_PREFIXES)
    frame = frame[~is_excluded]
    ledger.record(
        f"drop non-scheduled trip namespaces {EXCLUDED_TRIP_PREFIXES}", before, len(frame)
    )

    # --- rule 2: a delay needs both an observation and a schedule --------
    before = len(frame)
    frame = frame[frame["stop_timestamp"].notna() & frame["scheduled_arrival_time"].notna()]
    ledger.record("drop rows missing observed or scheduled time", before, len(frame))

    # --- rule 3: de-duplicate the (service date, trip, stop) key ---------
    before = len(frame)
    frame = (
        # Stable sort: ties keep their input order on every CPU and numpy build.
        frame.sort_values("stop_timestamp", kind="stable")
        .drop_duplicates(subset=["service_date", "trip_id", "stop_id"], keep="first")
    )
    ledger.record("drop duplicate (service_date, trip_id, stop_id)", before, len(frame))

    # One scheduled visit recorded under two platform ids (Kenmore, Ashmont,
    # Mattapan): a second copy would become its own "previous stop".
    before = len(frame)
    frame = frame.drop_duplicates(
        subset=TRIP_KEY + ["parent_station", "scheduled_arrival_time"], keep="first"
    )
    ledger.record("drop duplicate visits under a second platform id", before, len(frame))

    # --- derived columns -------------------------------------------------
    # Rule 2 guarantees both operands, so delay is always defined here. Dates are
    # parsed once per distinct service date and broadcast back to the rows.
    frame = frame.copy()
    date_codes, dates = _service_date_table(frame["service_date"])
    midnight = dates["midnight_epoch"].to_numpy()[date_codes]
    frame["scheduled_epoch"] = midnight + frame["scheduled_arrival_time"]
    frame["delay_seconds"] = frame["stop_timestamp"] - frame["scheduled_epoch"]
    frame["arrival_local"] = pd.to_datetime(
        frame["stop_timestamp"], unit="s", utc=True
    ).dt.tz_convert(config.SERVICE_TZ)

    # Scheduled time-of-day, folded into 00:00-23:59. Post-midnight trips carry
    # times > 86400 s, so the modulo is what makes "hour of day" meaningful.
    frame["scheduled_hour"] = ((frame["scheduled_arrival_time"] // 3600) % 24).astype("int16")
    frame["scheduled_seconds_of_day"] = (
        frame["scheduled_arrival_time"] % config.SECONDS_PER_DAY
    ).astype("int32")

    frame["day_of_week"] = dates["day_of_week"].to_numpy()[date_codes]
    frame["is_weekend"] = frame["day_of_week"] >= 5
    frame["service_date_parsed"] = dates["date"].to_numpy()[date_codes]

    frame["direction_id"] = frame["direction_id"].astype("int8")
    frame["late"] = frame["delay_seconds"] > LATE_THRESHOLD_SECONDS

    # --- rule 4: flag, do not silently delete, implausible values --------
    frame["delay_outlier"] = frame["delay_seconds"].abs() > DELAY_OUTLIER_SECONDS
    frame["dwell_implausible"] = (
        frame["dwell_time_seconds"].notna()
        & ((frame["dwell_time_seconds"] > DWELL_OUTLIER_SECONDS)
           | (frame["dwell_time_seconds"] < 0))
    )
    if frame["dwell_implausible"].any():
        frame.loc[frame["dwell_implausible"], "dwell_time_seconds"] = np.nan

    # --- trip order, origins, and physically impossible timestamps -------
    frame = frame.sort_values(TRIP_ORDER).reset_index(drop=True)
    trip = frame.groupby(TRIP_KEY, sort=False)
    handover = frame["vehicle_id"].ne(trip["vehicle_id"].shift()) & (trip.cumcount() > 0)
    frame["run"] = handover.groupby([frame[k] for k in TRIP_KEY]).cumsum().astype("int8")

    # --- single stops that disagree with the rest of their run -------------
    before = len(frame)
    # Repeated until none remain: removing one bad stop can expose its neighbour.
    while (glitch := isolated_glitches(frame)).any():
        frame = frame[~glitch].reset_index(drop=True)
    ledger.record("drop isolated stops >30 min off both neighbours", before, len(frame))

    # --- whole trips matched to the wrong timetable entry ------------------
    # After the single-stop rule, so a trip's median is judged on its real stops.
    before = len(frame)
    trip_median = frame.groupby(TRIP_KEY)["delay_seconds"].transform("median")
    frame = frame[trip_median >= -MISMATCH_EARLY_SECONDS].reset_index(drop=True)
    ledger.record("drop trips matched to the wrong schedule (>30 min early)",
                  before, len(frame))

    run = frame.groupby(RUN_KEY, sort=False)
    frame["stop_index"] = run.cumcount().astype("int16")
    # A run's first stop has no previous stop *on the same train*: at a trip
    # origin it is a platform wait, at a hand-over the delay belongs to a new train.
    frame["is_origin"] = frame["stop_index"] == 0
    frame["time_inconsistent"] = run["stop_timestamp"].diff() < 0

    frame = _attach_stop_names(frame, stop_lookup)

    ledger.record("final clean rows", len(frame), len(frame))
    if in_memory:
        # Never let an injected test frame overwrite the real cached table.
        return frame

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    ledger.to_frame().to_csv(LEDGER_PATH, index=False)

    log.info("clean table: %s rows, %s cols -> %s", f"{len(frame):,}",
             frame.shape[1], OUT_PATH.name)
    return frame


def summary(frame: pd.DataFrame | None = None) -> dict:
    """Headline statistics; delay figures describe genuine arrivals only."""
    frame = build() if frame is None else frame
    arrivals = frame[is_arrival(frame)]
    delay = arrivals["delay_seconds"]
    return {
        "rows": len(frame),
        "arrival_rows": len(arrivals),
        "origin_rows": int(frame["is_origin"].sum()),
        "time_inconsistent_rows": int(frame["time_inconsistent"].sum()),
        "service_start": str(frame["service_date_parsed"].min()),
        "service_end": str(frame["service_date_parsed"].max()),
        "routes": frame["route_id"].nunique(),
        "stations": frame["station_name"].nunique(),
        "median_delay_seconds": float(delay.median()),
        "mean_delay_seconds": float(delay.mean()),
        "p05_delay_seconds": float(delay.quantile(0.05)),
        "p95_delay_seconds": float(delay.quantile(0.95)),
        "late_pct": float(arrivals["late"].mean() * 100),
        "delay_outlier_pct": float(arrivals["delay_outlier"].mean() * 100),
    }


def run(*, refresh: bool = False) -> dict:
    """Stage entry point used by the CLI."""
    frame = build(refresh=refresh)
    stats = summary(frame)
    log.info("clean summary: %s", stats)
    return stats


def load() -> pd.DataFrame:
    """Load the cached clean table, building it if absent."""
    if not OUT_PATH.exists():
        return build()
    return pd.read_parquet(OUT_PATH)
