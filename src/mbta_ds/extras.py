"""Supporting analyses behind the README's narrative claims.

The main stages produce the headline tables. This stage recomputes the smaller
findings the README quotes, so every number there comes from code:

* how sticky delay is along a trip (slope of delay on the previous stop's);
* how much delay changes on a trip's final hop versus mid-trip;
* the Heath Street pairing artefact on Green-E;
* the worst days of the test period and what the rows there looked like;
* the cross-season transfer test (spring model on winter and back), when both
  runs exist;
* an independent re-derivation of delay, the previous-stop delay and the weather
  join on a random sample, using different code from the pipeline.

Run with ``python -m mbta_ds.cli extras`` after ``train``; results are written to
``extras.json`` in the run's processed folder and as CSV under ``reports/tables``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import clean, collect_weather, config, features
from . import model_delay as md

log = logging.getLogger(__name__)

OUT_PATH = config.PROCESSED_DIR / "extras.json"
MAIN_PROCESSED = config.DATA_DIR / "processed"
WINTER_PROCESSED = config.DATA_DIR / "runs" / "winter" / "processed"
HEATH, LECHMERE, MEDFORD = "place-hsmnl", "place-lech", "place-mdftf"
REDERIVE_SAMPLE = 2_000


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(clean.TRIP_ORDER, kind="stable")


# ---------------------------------------------------------------------------
# Delay dynamics along a trip
# ---------------------------------------------------------------------------
def stickiness(feats: pd.DataFrame) -> pd.DataFrame:
    """Per line: least-squares slope of delay on the previous stop's delay, and
    the share of hops that recover more than a minute."""
    rows = []
    for route, part in feats.groupby(feats["route_id"].astype(str)):
        x = part["prev_delay_1"].to_numpy(float)
        y = part["delay_seconds"].to_numpy(float)
        slope = np.polyfit(x, y, 1)[0]
        rows.append({"route_id": route, "slope": float(slope), "n": len(part),
                     "share_recovering_over_60s": float(np.mean(y - x < -60))})
    table = pd.DataFrame(rows)
    change = feats["delay_seconds"] - feats["prev_delay_1"]
    table.attrs["network_share_recovering_over_60s"] = float(np.mean(change < -60))
    return table


def final_hop(frame: pd.DataFrame) -> pd.DataFrame:
    """Change in delay on each run's last hop versus mid-trip hops, per line."""
    arrivals = _ordered(frame)
    group = arrivals.groupby(clean.RUN_KEY, sort=False)
    arrivals = arrivals.assign(change=group["delay_seconds"].diff(),
                               last=group.cumcount(ascending=False) == 0)
    arrivals = arrivals[clean.is_arrival(arrivals) & arrivals["change"].notna()]
    out = arrivals.groupby([arrivals["route_id"].astype(str), "last"])["change"].agg(["median", "mean", "size"])
    return out.unstack("last").set_axis(
        ["mid_median", "final_median", "mid_mean", "final_mean", "mid_n", "final_n"], axis=1
    ).reset_index()


def heath_street(frame: pd.DataFrame) -> dict:
    """Green-E towards Medford/Tufts: departure offset from Heath Street, the
    scheduled gap between departures, and running time against the timetable."""
    e = _ordered(frame[(frame["route_id"] == "Green-E") & (frame["direction_id"] == 1)])
    group = e.groupby(clean.RUN_KEY, sort=False)
    e = e.assign(depart=group["move_timestamp"].shift(-1))
    origin = e[(e["parent_station"] == HEATH) & e["is_origin"]].copy()
    anchor = clean.service_midnight_epoch(origin["service_date"])
    origin["offset"] = origin["depart"] - (anchor + origin["scheduled_departure_time"])
    origin["day"] = np.where(origin["is_weekend"], "weekend", "weekday")
    origin = origin.sort_values(["service_date", "scheduled_departure_time"])
    origin["gap"] = origin.groupby("service_date")["scheduled_departure_time"].diff()

    runs = e[e["parent_station"].isin([LECHMERE, MEDFORD])]
    wide = runs.pivot_table(index=clean.RUN_KEY + ["is_weekend"], columns="parent_station",
                            values=["stop_timestamp", "scheduled_arrival_time"], aggfunc="first").dropna()
    actual = (wide[("stop_timestamp", MEDFORD)] - wide[("stop_timestamp", LECHMERE)]) / 60
    sched = (wide[("scheduled_arrival_time", MEDFORD)] - wide[("scheduled_arrival_time", LECHMERE)]) / 60
    day = np.where(wide.index.get_level_values("is_weekend"), "weekend", "weekday")

    hourly = origin.assign(hour=(origin["scheduled_departure_time"] // 3600) % 24).groupby("hour").agg(
        offset=("offset", "median"), gap=("gap", "median"), n=("offset", "size"))
    hourly = hourly[hourly["n"] >= 30].dropna()

    second = e[e["stop_index"] == 1]
    other = _ordered(frame[(frame["route_id"] == "Green-E") & (frame["direction_id"] == 0)])
    other_second = other[other["stop_index"] == 1]
    iqr = lambda s: float(s.quantile(0.75) - s.quantile(0.25)) / 60   # noqa: E731
    return {
        "median_departure_offset_min": (origin.groupby("day")["offset"].median() / 60).round(2).to_dict(),
        "median_scheduled_gap_min": (origin.groupby("day")["gap"].median() / 60).round(2).to_dict(),
        "lechmere_to_medford_actual_min": actual.groupby(day).median().round(2).to_dict(),
        "lechmere_to_medford_scheduled_min": sched.groupby(day).median().round(2).to_dict(),
        "hourly_offset_vs_gap_correlation": float(hourly["offset"].corr(hourly["gap"])),
        "second_stop_delay_iqr_min": {"towards Medford/Tufts": iqr(second["delay_seconds"]),
                                      "towards Heath Street": iqr(other_second["delay_seconds"])},
    }


# ---------------------------------------------------------------------------
# The test period's worst days
# ---------------------------------------------------------------------------
def worst_days(frame: pd.DataFrame, top: int = 3) -> dict:
    """Daily MAE of the change model, the level (absolute-error) model and
    persistence, and what the worst days' worst line looked like."""
    p = pd.read_parquet(md.PREDICTIONS_PATH)
    daily = p.assign(
        change=(p["predicted_delay"] - p["delay_seconds"]).abs(),
        level=(p["level_predicted_delay"] - p["delay_seconds"]).abs(),
        persistence=(p["persistence_delay"] - p["delay_seconds"]).abs(),
    ).groupby("service_date_parsed")[["change", "level", "persistence"]].mean()
    worst = daily.sort_values("level", ascending=False).head(top)
    lines = []
    for day in worst.index:
        rows = p[p["service_date_parsed"] == day]
        err = (rows["level_predicted_delay"] - rows["delay_seconds"]).abs()
        route = err.groupby(rows["route_id"].astype(str)).sum().idxmax()
        on_day = frame[(frame["service_date_parsed"] == day) & (frame["route_id"] == route)
                       & clean.is_arrival(frame)]
        lines.append({
            "date": str(day), "route_id": route,
            "trips": int(on_day[clean.TRIP_KEY].drop_duplicates().shape[0]),
            "median_delay_hours": round(float(on_day["delay_seconds"].median()) / 3600, 2),
            "median_headway_min": round(float(on_day["headway_trunk_seconds"].median()) / 60, 1),
            "median_scheduled_headway_min": round(float(on_day["scheduled_headway_trunk"].median()) / 60, 1),
        })
    return {"daily": daily.round(1).reset_index().astype({"service_date_parsed": str}).to_dict("records"),
            "worst": worst.round(1).reset_index().astype({"service_date_parsed": str}).to_dict("records"),
            "worst_lines": lines,
            "level_model_max_prediction_s": float(p["level_predicted_delay"].max()),
            "max_actual_delay_s": float(p["delay_seconds"].max())}


# ---------------------------------------------------------------------------
# Cross-season transfer
# ---------------------------------------------------------------------------
def _align_categories(*frames: pd.DataFrame) -> None:
    """Give each categorical column the same categories (and codes) in every frame,
    so a model fitted on one season reads the other's categories correctly."""
    for column in features.CATEGORICAL_FEATURES:
        values = pd.concat([f[column].astype(object) for f in frames])
        union = pd.Index(sorted(values.dropna().unique(), key=str))
        for f in frames:
            f[column] = pd.Categorical(f[column].astype(object), categories=union)


def cross_season() -> list[dict] | None:
    """Train the headline model on one season, score it on the other's test period."""
    paths = {"spring": MAIN_PROCESSED / "features.parquet",
             "winter": WINTER_PROCESSED / "features.parquet"}
    if not all(p.exists() for p in paths.values()):
        log.info("cross-season test skipped: needs both the main and the winter run")
        return None
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    _align_categories(*frames.values())
    splits = {name: md.temporal_split(f) for name, f in frames.items()}
    for split in splits.values():
        split.train = features.stratified_sample(split.train, md.TRAIN_ROWS, md.SEED)
    numeric = sorted(set(md._columns(splits["spring"].train)[0]) & set(md._columns(splits["winter"].train)[0]))
    categorical = list(features.CATEGORICAL_FEATURES)
    cols = numeric + categorical

    rows = []
    trainings = {"spring": splits["spring"].train, "winter": splits["winter"].train,
                 "winter, whole window": features.stratified_sample(frames["winter"], md.TRAIN_ROWS, md.SEED)}
    for name, train in trainings.items():
        model = md._fit(md._change_boost(False), train, numeric, categorical, "delay_seconds")
        row = {"trained_on": name}
        for season, split in splits.items():
            y = split.test["delay_seconds"].to_numpy()
            row[f"{season}_test_mae"] = md._mae(model.predict(split.test[cols]), y)
        rows.append(row)
    rows.append({"trained_on": "persistence", **{
        f"{season}_test_mae": md._mae(split.test["prev_delay_1"].to_numpy(float),
                                      split.test["delay_seconds"].to_numpy())
        for season, split in splits.items()}})
    return rows


# ---------------------------------------------------------------------------
# Independent re-derivation
# ---------------------------------------------------------------------------
def rederive(frame: pd.DataFrame, feats: pd.DataFrame, n: int = REDERIVE_SAMPLE) -> dict:
    """Recompute derived values for a random sample with plain Python.

    Delay uses ``zoneinfo`` and the GTFS definition (noon minus 12 h) directly;
    the previous-stop delay walks each sampled run in scheduled order; the
    weather is looked up by hand for the hour of the prediction moment.
    """
    tz = ZoneInfo(config.SERVICE_TZ)
    sample = feats.sample(min(n, len(feats)), random_state=md.SEED)
    trips = sample[clean.TRIP_KEY].drop_duplicates()
    sub = frame.merge(trips, on=clean.TRIP_KEY)
    runs = {key: sorted(group.itertuples(), key=lambda r: (r.scheduled_arrival_time, r.stop_sequence))
            for key, group in sub.groupby(clean.RUN_KEY)}
    by_stop = sub.set_index(["service_date", "trip_id", "stop_id"])
    weather = collect_weather.load_weather().drop_duplicates("timestamp").set_index("timestamp")["temperature_c"]

    delay_ok = prev_ok = weather_ok = weather_checked = 0
    for row in sample.itertuples():
        src = by_stop.loc[(row.service_date, row.trip_id, row.stop_id)]
        day = datetime.strptime(str(row.service_date), "%Y%m%d")
        anchor = datetime(day.year, day.month, day.day, 12, tzinfo=tz) - timedelta(hours=12)
        delay = src["stop_timestamp"] - (anchor.timestamp() + src["scheduled_arrival_time"])
        delay_ok += int(delay == row.delay_seconds)
        ordered = runs[(row.service_date, row.trip_id, src["run"])]
        position = [r.stop_id for r in ordered].index(row.stop_id)
        prev_ok += int(position > 0 and ordered[position - 1].delay_seconds == row.prev_delay_1)
        hour = datetime.fromtimestamp(row.known_at, tz).replace(minute=0, second=0, tzinfo=None)
        if pd.Timestamp(hour) in weather.index:
            weather_checked += 1
            weather_ok += int(np.float32(weather[pd.Timestamp(hour)]) == np.float32(row.temperature_c))
    return {"sample": len(sample), "delay_matches": delay_ok, "prev_delay_matches": prev_ok,
            "weather_checked": weather_checked, "weather_matches": weather_ok}


def run() -> dict:
    """Compute every supporting analysis and persist them."""
    config.ensure_dirs()
    frame = clean.load()
    feats = features.load()
    stickiness_table = stickiness(feats)
    out = {
        "stickiness": stickiness_table.to_dict("records"),
        "network_share_recovering_over_60s": stickiness_table.attrs["network_share_recovering_over_60s"],
        "final_hop": final_hop(frame).to_dict("records"),
        "heath_street": heath_street(frame),
        "worst_days": worst_days(frame),
        "rederivation": rederive(frame, feats),
        "cross_season": cross_season() if not config.RUN else None,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for name in ("stickiness", "final_hop", "cross_season"):
        if out[name]:
            pd.DataFrame(out[name]).to_csv(config.TABLES_DIR / f"extras_{name}.csv", index=False)
    log.info("extras -> %s", OUT_PATH.name)
    return {k: (len(v) if isinstance(v, list) else "ok") for k, v in out.items() if v is not None}


def load() -> dict:
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"{OUT_PATH} missing; run the `extras` stage first")
    return json.loads(OUT_PATH.read_text(encoding="utf-8"))
