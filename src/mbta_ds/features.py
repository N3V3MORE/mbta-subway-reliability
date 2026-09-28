"""Feature extraction: build the leakage-safe design matrix for the delay model.

The central discipline in this module is **causal ordering**. A feature for the
arrival at stop *k* may only use information that exists before the train reaches
stop *k*. Concretely:

* Lag features come from the *previous* stops of the same trip, shifted with
  ``groupby(...).shift(1)`` -- never from the current or later rows.
* Demand is merged from a *lagged* rolling window of prior days, never the same
  day's total (which is unknown until the day ends).
* Target encoding is deliberately **not** done here. Categorical identifiers are
  handed to the model as raw categories so any encoding happens inside a
  train-only pipeline and cannot leak test information.
* Weather is joined on the *scheduled* hour. This is a nowcast, not a forecast,
  and is flagged as a limitation in the README; the trainer runs a no-weather
  ablation to quantify how much it actually buys.

Feature groups are declared explicitly (:data:`FEATURE_GROUPS`) so the trainer can
run ablations without re-deriving anything.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from . import clean, collect_lamp, collect_ridership, collect_weather, config
from .progress import Progress

log = logging.getLogger(__name__)

OUT_PATH = config.PROCESSED_DIR / "features.parquet"

#: Row cap for ``--quick`` runs. A normal build keeps every arrival: the trainer
#: samples its *training* rows and scores on every test-period row.
QUICK_MAX_ROWS = 150_000

#: Window for the line-wide lateness features.
LINE_WINDOW_SECONDS = 900

#: Columns that identify a row but must never be used as features.
KEY_COLUMNS = (
    "service_date", "service_date_parsed", "trip_id", "stop_id",
    "vehicle_id", "stop_sequence",
)
TARGET_COLUMNS = ("delay_seconds", "late", "delay_outlier")

CATEGORICAL_FEATURES = ("route_id", "trunk_route_id", "direction_id", "station_name")

#: Numeric feature columns, grouped by theme so ablations are declarative.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "schedule": (
        "stop_sequence",
        "stop_count",
        "fraction_through_trip",
        "scheduled_seconds_of_day",
        "scheduled_hour",
        "scheduled_elapsed_seconds",
        "scheduled_travel_time",
        "scheduled_headway_branch",
        "scheduled_headway_trunk",
    ),
    # The train's own state at the last stop it has reached. With a horizon of k
    # stops, "prev" means k stops back: the latest thing known at prediction time.
    "propagation": (
        "prev_delay_1",
        "prev_delay_2",
        "prev_dwell_seconds",
        "prev_travel_time_seconds",
        "prev_headway_seconds",
        "delay_trend",
        "scheduled_seconds_ahead",
    ),
    # `days_since_window_start` is deliberately NOT a model feature. It is a
    # monotone index of the window, so under a temporal split its train and test
    # ranges never overlap and a tree splitting on it extrapolates blindly (it
    # raised test MAE from 272 s to 442 s). It is still built into the table so
    # `model_delay.run_calendar_diagnostic` can reproduce that finding.
    "calendar": (
        "day_of_week",
        "is_weekend",
        "is_peak",
    ),
    "demand": (
        "station_entries_lag7",
        "station_entries_trend",
        "demand_missing",
    ),
    "weather": (
        "temperature_c",
        "precip_mm",
        "snowfall_cm",
        "wind_kph",
        "humidity_pct",
        "is_precip",
        "is_snow",
        "weather_missing",
    ),
    "alerts": (
        "route_alerts_active",
        "route_alert_severity_max",
    ),
    # Other trains on the line, as they stood at prediction time (see
    # `attach_network`): delays spread down a line, not only along a trip.
    "network": (
        "leader_delay",
        "leader_age_seconds",
        "line_late_share_15m",
        "line_arrivals_15m",
    ),
}
CATEGORICAL_GROUP = ("categorical",)

#: Built into the table for diagnostics only; see the note on "calendar" above.
DIAGNOSTIC_COLUMNS = ("days_since_window_start",)

#: Peak hours by the MBTA's own definition of weekday peaks.
PEAK_HOURS = (7, 8, 9, 16, 17, 18)

#: Alert effects that describe station facilities rather than train movement.
#: About 93% of subway alerts in the 2026 window are elevator/escalator outages,
#: which carry no information about whether a train will run late.
NON_SERVICE_ALERT_EFFECTS = ("ACCESSIBILITY_ISSUE",)


def all_feature_columns(groups: tuple[str, ...] | None = None) -> list[str]:
    """Return the flat list of feature columns for the requested groups."""
    names = list(groups) if groups is not None else list(FEATURE_GROUPS) + list(CATEGORICAL_GROUP)
    columns: list[str] = []
    for name in names:
        if name == "categorical":
            columns.extend(CATEGORICAL_FEATURES)
        else:
            columns.extend(FEATURE_GROUPS[name])
    return columns


# ---------------------------------------------------------------------------
# Propagation features
# ---------------------------------------------------------------------------
def add_propagation_features(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add the train's own state as known ``horizon`` stops before the target.

    With ``horizon=1`` this is the previous stop; with ``horizon=5`` the model is
    asked to predict five stops ahead, knowing only what had happened by then.
    ``known_at`` is that moment: departure from stop *target - horizon* (the next
    row's ``move_timestamp``), falling back to the arrival there when unrecorded.

    Grouping key is ``(service_date, trip_id)`` and **not** ``trip_id`` alone:
    trip ids are reused across service dates, so grouping on the id alone would
    leak across days. Stops are ordered by scheduled time, not ``stop_sequence``,
    which sometimes labels the final stop as the first; and a trip finished by
    another vehicle is split into runs, so no lag reaches across trains.
    """
    k = horizon
    out = frame.sort_values(clean.TRIP_ORDER)
    group = out.groupby(clean.RUN_KEY if "run" in out else clean.TRIP_KEY, sort=False)

    out["prev_delay_1"] = group["delay_seconds"].shift(k)
    out["prev_delay_2"] = group["delay_seconds"].shift(k + 1)
    out["prev_dwell_seconds"] = group["dwell_time_seconds"].shift(k)
    out["prev_travel_time_seconds"] = group["travel_time_seconds"].shift(k)
    out["prev_headway_seconds"] = group["headway_trunk_seconds"].shift(k)
    # Slope of delay over the last two known stops: losing or gaining time?
    out["delay_trend"] = out["prev_delay_1"] - out["prev_delay_2"]
    out["scheduled_seconds_ahead"] = (
        out["scheduled_arrival_time"] - group["scheduled_arrival_time"].shift(k)
    )
    departed = group["move_timestamp"].shift(k - 1) if k > 1 else out["move_timestamp"]
    out["known_at"] = departed.combine_first(group["stop_timestamp"].shift(k))
    return out


def attach_network(frame: pd.DataFrame) -> pd.DataFrame:
    """Add what *other* trains on the line had done by ``known_at``.

    Every observed arrival in ``frame`` is an event. For each row, on the same
    line and direction and strictly before ``known_at`` (so nothing from the
    future is used):

    * ``leader_delay`` / ``leader_age_seconds`` -- the delay of the last train to
      reach this row's station, and how long before ``known_at`` it did so. A
      long age means a gap: often a blockage further up the line.
    * ``line_late_share_15m`` / ``line_arrivals_15m`` -- the share of arrivals on
      the line more than 5 minutes late in the preceding 15 minutes, and how
      many arrivals there were.
    """
    line = ["trunk_route_id", "direction_id"]
    events = (frame.loc[frame["stop_timestamp"].notna(),
                        line + ["parent_station", "stop_timestamp", "delay_seconds"]]
              .astype({"stop_timestamp": "float64"}).sort_values("stop_timestamp"))
    rows = frame.loc[frame["known_at"].notna(), line + ["parent_station", "known_at"]]
    rows = rows.astype({"known_at": "float64"}).sort_values("known_at")

    leader = pd.merge_asof(
        rows, events.rename(columns={"stop_timestamp": "leader_ts", "delay_seconds": "leader_delay"}),
        left_on="known_at", right_on="leader_ts", by=line + ["parent_station"],
        allow_exact_matches=False,
    )
    out = frame.copy()
    out["leader_delay"] = pd.Series(leader["leader_delay"].to_numpy(), index=rows.index)
    out["leader_age_seconds"] = pd.Series(
        (leader["known_at"] - leader["leader_ts"]).to_numpy(), index=rows.index)

    # Running counts of arrivals and late arrivals per line; the difference of
    # two as-of lookups is the count inside the window.
    events = events.assign(n=1, late=(events["delay_seconds"] > config.LATE_THRESHOLD_SECONDS))
    events[["n", "late"]] = events.groupby(line)[["n", "late"]].cumsum()

    def counts_before(times: pd.Series) -> pd.DataFrame:
        probe = rows[line].assign(t=times.to_numpy()).sort_values("t")
        hit = pd.merge_asof(probe, events[line + ["stop_timestamp", "n", "late"]],
                            left_on="t", right_on="stop_timestamp", by=line,
                            allow_exact_matches=False)
        return hit.set_index(probe.index)[["n", "late"]].fillna(0).reindex(rows.index)

    now, then = counts_before(rows["known_at"]), counts_before(rows["known_at"] - LINE_WINDOW_SECONDS)
    arrivals = now["n"] - then["n"]
    out["line_arrivals_15m"] = arrivals
    out["line_late_share_15m"] = (now["late"] - then["late"]) / arrivals.where(arrivals > 0)
    return out


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def attach_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Join hourly weather on the scheduled local hour."""
    out = frame.copy()
    try:
        weather = collect_weather.load_weather()
    except FileNotFoundError:
        log.warning("no weather cache; weather features will be missing")
        for column in FEATURE_GROUPS["weather"]:
            out[column] = np.nan
        return out

    scheduled_local = pd.to_datetime(
        out["scheduled_epoch"], unit="s", utc=True
    ).dt.tz_convert(config.SERVICE_TZ).dt.tz_localize(None)
    out["_sched_hour"] = scheduled_local.dt.floor("h")

    # Timestamps are naive local time, so the autumn DST fall-back repeats 01:00.
    # A duplicated key would silently duplicate every train row in that hour.
    weather = weather.copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather = weather.drop_duplicates(subset=["timestamp"], keep="first")
    merged = out.merge(weather, left_on="_sched_hour", right_on="timestamp", how="left")
    merged = merged.drop(columns=["_sched_hour", "timestamp"])

    merged["is_precip"] = (merged["precip_mm"].fillna(0) > 0).astype("int8")
    merged["is_snow"] = (merged["snowfall_cm"].fillna(0) > 0).astype("int8")
    merged["weather_missing"] = merged["temperature_c"].isna().astype("int8")
    return merged


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------
def attach_demand(frame: pd.DataFrame) -> pd.DataFrame:
    """Join a *lagged* station-demand signal from the ridership dataset.

    Same-day entries are not usable (they are only known once the day is over),
    so this uses the mean of the previous seven days, plus the difference between
    that mean and the mean of the seven days before it as a trend term.
    """
    out = frame.copy()
    try:
        daily = collect_ridership.load_daily()
    except FileNotFoundError:
        log.warning("no ridership cache; demand features will be missing")
        for column in FEATURE_GROUPS["demand"]:
            out[column] = np.nan
        return out

    # Multi-line stations appear once per line; sum them back to station level,
    # then lay the result out on a complete daily calendar (date x station). A
    # row-wise shift would treat "the previous row" as "yesterday", silently
    # reaching further back whenever a station is missing a day.
    per_station = daily.groupby(["service_date", "stop_id"])["daily_entries"].sum()
    wide = per_station.unstack("stop_id")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))

    lag1 = wide.shift(1)                       # strictly before the service date
    lag7 = lag1.rolling(7, min_periods=1).mean()
    prev7 = lag1.shift(7).rolling(7, min_periods=1).mean()

    def _long(values: pd.DataFrame, name: str) -> pd.DataFrame:
        return (values.rename_axis(index="service_date", columns="parent_station")
                .reset_index().melt(id_vars="service_date", var_name="parent_station",
                                    value_name=name))

    lookup = _long(lag7, "station_entries_lag7")
    lookup["station_entries_trend"] = _long(lag7 - prev7, "trend")["trend"].to_numpy()
    # The clean/feature tables key on an int like 20260630 while the ridership
    # cache carries real dates; normalise to the int key or the merge silently
    # produces all-NaN.
    lookup["service_date"] = lookup["service_date"].dt.strftime("%Y%m%d").astype("int64")
    merged = out.merge(lookup, on=["service_date", "parent_station"], how="left")

    # Not every subway stop is gated: Green Line surface stops west of Kenmore and
    # parts of the Green Line Extension have no faregates at all, so they have no
    # ridership signal. That absence is informative, so it is flagged rather than
    # imputed to zero here.
    merged["demand_missing"] = merged["station_entries_lag7"].isna().astype("int8")
    missing_rate = float(merged["demand_missing"].mean())
    if missing_rate > 0:
        log.info("%.1f%% of rows belong to stations with no gated-entry signal",
                 missing_rate * 100)
    return merged


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def _without_alert_features(out: pd.DataFrame, reason: str) -> pd.DataFrame:
    log.warning("%s; alert features will be missing", reason)
    for column in FEATURE_GROUPS["alerts"]:
        out[column] = np.nan
    return out


def attach_alerts(
    frame: pd.DataFrame,
    *,
    max_span_days: int = 7,
    alerts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join the service alerts already in force when the scheduled hour begins.

    **Causality.** These alerts are *reactive*, not published in advance: in the
    2026 window the median gap between an alert's creation and the start of its
    active period is zero. An alert is therefore only counted from the first hour
    boundary at or after its start. Flooring the start instead would let an alert
    raised at 08:50 inform an 08:10 arrival -- information from the future.

    **Relevance.** Elevator and escalator outages (:data:`NON_SERVICE_ALERT_EFFECTS`)
    are ~93% of subway alerts and say nothing about train running, so they are
    excluded rather than left to drown out the service-affecting ones.

    **Cost.** The archive holds ~5M rows back to 2019, exploded along both
    ``informed_entity`` and ``active_period``; only ~15k are subway alerts that
    overlap the window. Those are selected *before* expanding alerts into hourly
    bins, and spans are capped at ``max_span_days``.

    ``alerts`` injects the archive directly, for tests.
    """
    out = frame.copy()
    route_col = "informed_entity.route_id"
    start_col = "active_period.start_datetime"
    end_col = "active_period.end_datetime"

    if alerts is None:
        try:
            alerts = collect_lamp.load_alerts(
                columns=[route_col, start_col, end_col, "severity", "id", "effect"]
            )
        except FileNotFoundError:
            return _without_alert_features(out, "no alerts archive")
    if route_col not in alerts.columns or start_col not in alerts.columns:
        return _without_alert_features(out, "alerts archive lacks expected columns")

    work = alerts.rename(columns={route_col: "route_id", start_col: "alert_start",
                                  end_col: "alert_end", "id": "alert_id"})
    if "alert_end" not in work.columns:
        work["alert_end"] = pd.NaT
    if "severity" not in work.columns:
        work["severity"] = np.nan
    if "alert_id" not in work.columns:
        work["alert_id"] = np.arange(len(work))

    # --- select what matters before any expansion ------------------------
    routes = set(out["route_id"].astype(str).unique())
    work = work[work["route_id"].astype(str).str.strip().isin(routes)]
    if "effect" in work.columns:
        work = work[~work["effect"].isin(NON_SERVICE_ALERT_EFFECTS)]

    work = work.assign(
        route_id=work["route_id"].astype(str).str.strip(),
        alert_start=pd.to_datetime(work["alert_start"], errors="coerce"),
        alert_end=pd.to_datetime(work["alert_end"], errors="coerce"),
    )
    work = work[work["alert_start"].notna()]

    dates = pd.to_datetime(out["service_date"].astype("int64").astype(str), format="%Y%m%d")
    window_start = dates.min()
    # Service days run past midnight, so the window's last hour is on the next day.
    window_end = dates.max() + pd.Timedelta(days=2)
    # Open-ended alerts run to the end of the window, capped at `max_span_days`.
    capped_end = work["alert_start"] + pd.Timedelta(days=max_span_days)
    work["alert_end"] = work["alert_end"].fillna(window_end).clip(upper=capped_end)
    work = work[(work["alert_end"] >= window_start) & (work["alert_start"] <= window_end)]
    work = work.drop_duplicates(subset=["alert_id", "route_id", "alert_start", "alert_end"])

    # --- expand each alert into the hours it is *known* to be in force ----
    first_hour = work["alert_start"].dt.ceil("h")
    hours = ((work["alert_end"].dt.floor("h") - first_hour) // pd.Timedelta(hours=1)) + 1
    keep = hours >= 1
    work, first_hour, hours = work[keep], first_hour[keep], hours[keep].astype(int)
    if work.empty:
        # The archive was read fine and simply holds nothing relevant: that is a
        # genuine zero, not missing data.
        log.info("alerts: no service-affecting alerts in the window")
        out["route_alerts_active"] = 0.0
        out["route_alert_severity_max"] = 0.0
        return out

    repeat = np.repeat(np.arange(len(work)), hours.to_numpy())
    offsets = np.arange(len(repeat)) - np.repeat(np.cumsum(hours.to_numpy()) - hours.to_numpy(),
                                                  hours.to_numpy())
    exploded = pd.DataFrame({
        "route_id": work["route_id"].to_numpy()[repeat],
        "hour": first_hour.to_numpy()[repeat] + pd.to_timedelta(offsets, unit="h"),
        "alert_id": work["alert_id"].to_numpy()[repeat],
        "severity": work["severity"].to_numpy()[repeat],
    })
    grid = (
        exploded.groupby(["route_id", "hour"], as_index=False)
        .agg(route_alerts_active=("alert_id", "nunique"),
             route_alert_severity_max=("severity", "max"))
    )
    log.info("alerts: %s service-affecting alert spans -> %s route-hours",
             f"{len(work):,}", f"{len(grid):,}")

    # The archive is microsecond-resolution; align both keys before merging.
    grid["hour"] = grid["hour"].astype("datetime64[ns]")
    scheduled_local = pd.to_datetime(
        out["scheduled_epoch"], unit="s", utc=True
    ).dt.tz_convert(config.SERVICE_TZ).dt.tz_localize(None)
    out["hour"] = scheduled_local.dt.floor("h").astype("datetime64[ns]")
    out["route_id"] = out["route_id"].astype(str)

    merged = out.merge(grid, on=["route_id", "hour"], how="left")
    # No alert in force means zero alerts at severity zero -- not "unknown", which
    # median imputation would otherwise fill with a typical *alert's* severity.
    merged["route_alerts_active"] = merged["route_alerts_active"].fillna(0)
    merged["route_alert_severity_max"] = merged["route_alert_severity_max"].fillna(0)
    return merged.drop(columns=["hour"])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def stratified_sample(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """Randomly subsample to ``max_rows``, proportionally across service dates.

    Stratifying by date keeps every day represented, so the temporal split and
    the date-based grouping used by the propagation features stay intact. The
    per-date loop is written explicitly rather than via ``groupby.apply`` so the
    sampling behaviour is unambiguous across pandas versions.
    """
    if max_rows is None or len(frame) <= max_rows:
        return frame

    frac = max_rows / len(frame)
    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []
    for _, part in frame.groupby("service_date", sort=False):
        n = max(1, int(round(len(part) * frac)))
        n = min(n, len(part))
        picked.append(rng.choice(part.index.to_numpy(), size=n, replace=False))
    sampled = frame.loc[np.concatenate(picked)].sort_index()
    log.info("subsampled %s -> %s rows (%.1f%%), stratified by service date",
             f"{len(frame):,}", f"{len(sampled):,}", frac * 100)
    return sampled.reset_index(drop=True)


def build(
    *,
    max_rows: int | None = None,
    seed: int = 506,
    refresh: bool = False,
    clean_frame: pd.DataFrame | None = None,
    no_cache: bool = False,
    horizon: int = 1,
) -> pd.DataFrame:
    """Build and cache the feature table: one row per predictable arrival.

    ``horizon`` is how many stops ahead the target lies (1 = the next stop). Only
    the default horizon is cached; others are built in memory for experiments.
    ``clean_frame`` injects an input directly and, like ``no_cache``, suppresses
    writing the cache so tests never clobber the real artefact.
    """
    write_cache = not no_cache and clean_frame is None and horizon == 1
    if write_cache and OUT_PATH.exists() and not refresh:
        log.info("using cached feature table %s", OUT_PATH.name)
        return pd.read_parquet(OUT_PATH)

    config.ensure_dirs()
    frame = clean.load() if clean_frame is None else clean_frame

    # ------------------------------------------------------------------
    # Order matters. The row-wise features are computed on the FULL clean
    # table and only then is the table subsampled. Subsampling first would
    # corrupt every lag feature: `shift(1)` would return the previous
    # *surviving* stop rather than the previous stop, silently changing what
    # "the delay at the last stop" means and inflating the missing rate.
    # ------------------------------------------------------------------
    log.info("building features for %s clean rows", f"{len(frame):,}")

    with Progress(6, "feature build") as progress:
        frame = add_propagation_features(frame, horizon=horizon)
        progress.step("delay-propagation lags", extra=f"{len(frame):,} rows")

        # Needs every train's arrivals as events, so it runs before any filtering.
        frame = attach_network(frame)
        progress.step("other trains on the line")

        # Scheduled elapsed time is measured from the trip's first scheduled stop.
        frame["scheduled_elapsed_seconds"] = (
            frame["scheduled_arrival_time"] - frame["start_time"]
        )
        # stop_sequence counts in steps of 10 on most lines (up to 710 on a
        # 32-stop trip), so it cannot be divided by stop_count; the stop's
        # position in trip order can.
        frame["fraction_through_trip"] = (
            frame["stop_index"] / (frame["stop_count"] - 1).clip(lower=1)
        ).clip(upper=1)
        frame["is_peak"] = (
            frame["scheduled_hour"].isin(PEAK_HOURS) & (~frame["is_weekend"])
        ).astype("int8")

        # service_date is an int like 20260630, so subtracting dates must go
        # through real timestamps -- the raw difference is only a day count
        # within a month.
        window_min = pd.to_datetime(frame["service_date_parsed"]).min()
        frame["days_since_window_start"] = (
            pd.to_datetime(frame["service_date_parsed"]) - window_min
        ).dt.days.astype("int32")

        # Origin and impossible rows have served as lag sources above; they are
        # not arrival delays, so they are not targets. Nor is a stop fewer than
        # `horizon` stops into its run, where nothing about the train is known yet.
        targets = clean.is_arrival(frame) & frame["prev_delay_1"].notna()
        log.info("keeping %s target rows (dropped %s origin / inconsistent / "
                 "too-early rows)", f"{targets.sum():,}", f"{(~targets).sum():,}")
        frame = stratified_sample(frame[targets], max_rows, seed)
        progress.step("schedule, calendar and targets", extra=f"{len(frame):,} rows")

        frame = attach_demand(frame)
        progress.step("ridership demand join")

        frame = attach_weather(frame)
        progress.step("weather join")

    frame = attach_alerts(frame)

    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("category")

    # A feature declared in FEATURE_GROUPS but never produced would otherwise be
    # dropped silently by the projection below, quietly shrinking the model's
    # input. Fail loudly instead.
    declared = all_feature_columns()
    absent = [c for c in declared if c not in frame.columns]
    if absent:
        raise KeyError(
            f"declared features missing from the built table: {absent}. "
            "Check PERFORMANCE_COLUMNS in collect_lamp.py and the join steps here."
        )

    # Cast numerics down: the full matrix is ~40 columns wide and float64 would
    # double the memory footprint for no modelling benefit.
    for column in declared:
        if frame[column].dtype == "float64":
            frame[column] = frame[column].astype("float32")

    keep = (
        list(dict.fromkeys(
            KEY_COLUMNS + TARGET_COLUMNS + tuple(declared) + DIAGNOSTIC_COLUMNS
            + ("parent_station", "station_name", "direction_destination", "arrival_local")
        ))
    )
    keep = [c for c in keep if c in frame.columns]
    frame = frame[keep].reset_index(drop=True)

    if not write_cache:
        return frame

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    log.info("feature table: %s rows x %s cols -> %s", f"{len(frame):,}",
             frame.shape[1], OUT_PATH.name)
    return frame


def load() -> pd.DataFrame:
    """Load the cached feature table, building it if absent."""
    if not OUT_PATH.exists():
        return build()
    return pd.read_parquet(OUT_PATH)


def summary(frame: pd.DataFrame | None = None) -> dict:
    """Return coverage and missingness diagnostics for the feature table."""
    frame = load() if frame is None else frame
    features = all_feature_columns()
    missing = {
        column: round(float(frame[column].isna().mean() * 100), 2)
        for column in features
        if column in frame.columns and frame[column].isna().any()
    }
    return {
        "rows": int(len(frame)),
        "date_min": str(frame["service_date_parsed"].min()),
        "date_max": str(frame["service_date_parsed"].max()),
        "n_features": len(features),
        "missing_pct": missing,
    }


def run(*, quick: bool = False, refresh: bool = False) -> dict:
    """Stage entry point used by the CLI."""
    frame = build(max_rows=QUICK_MAX_ROWS if quick else None, refresh=refresh)
    stats = summary(frame)
    log.info("feature summary: %s", stats)
    return stats
