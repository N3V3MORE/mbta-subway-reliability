"""Export the network and observed train runs for the 3D map in ``map/``.

Two kinds of JSON file, both written to ``map/public/data/``:

* ``network.json``: the lines, one track geometry per branch (MBTA's canonical
  shapes), every station with its reliability and demand results, and the list
  of replay days.
* ``replay-<date>.json``: every observed trip on one service day. For each stop
  it records the station, when the train arrived and left, how late it was for
  riders there (``clean.lateness``), its delay against the timetable, and the
  model's prediction of that delay, plus the service incidents the MBTA reported
  that day (its alerts archive): when, on which lines and stations, and what.

Positions in the app are interpolated only between a trip's *observed* stops,
so no train is ever drawn where the data does not place it. Shapes and line
colours come from the V3 API and are cached next to the station coordinates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean, collect_lamp, config
from .features import NON_SERVICE_ALERT_EFFECTS
from .collect_v3 import load_stations
from .http import get_json, make_session

log = logging.getLogger(__name__)

OUT_DIR = config.ROOT / "map" / "public" / "data"
ROUTES_CACHE = config.V3_RAW_DIR / "routes.json"

#: (run, service date, label, note). The run names the folder under data/runs/;
#: "" is the main spring run. Each date lies inside its run's test period, so the
#: predictions shown were made by a model that never saw that day.
DEFAULT_DAYS = (
    ("", "2026-06-10", "Spring weekday", "A Wednesday in the spring test period."),
    ("winter", "2026-02-23", "Snowstorm", "The 23 February storm, from the winter run."),
    ("holdout", "2026-09-16", "Autumn holdout", "A Wednesday in the untouched July-September holdout."),
)

#: A station must appear on this share of a branch's trips to belong to it, so a
#: diverted trip cannot graft a stray station onto the branch.
MIN_BRANCH_SHARE = 0.02
#: Stations further than this from their branch's shape are left off it.
MAX_OFFSET_M = 250.0

_TRIP_COLUMNS = [
    "service_date", "trip_id", "route_id", "branch_route_id", "direction_destination",
    "stop_id", "parent_station", "stop_sequence", "stop_timestamp", "move_timestamp",
    "delay_seconds", "lateness_seconds", "late", "is_origin", "time_inconsistent",
    "scheduled_epoch", "scheduled_arrival_time",
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def decode_polyline(encoded: str) -> np.ndarray:
    """Decode a Google encoded polyline into an ``(n, 2)`` array of lon, lat."""
    values, shift, result = [], 0, 0
    for char in encoded:
        byte = ord(char) - 63
        result |= (byte & 0x1F) << shift
        shift += 5
        if byte < 0x20:
            values.append(~(result >> 1) if result & 1 else result >> 1)
            shift, result = 0, 0
    lat_lon = np.cumsum(np.array(values, dtype=float).reshape(-1, 2), axis=0) / 1e5
    return lat_lon[:, ::-1]


def _planar(lon_lat: np.ndarray, origin_lat: float) -> np.ndarray:
    """Project lon/lat to local metres. Accurate to well under 1% across Boston."""
    scale = np.array([111_320.0 * np.cos(np.radians(origin_lat)), 110_540.0])
    return lon_lat * scale


def cumulative_metres(lon_lat: np.ndarray, origin_lat: float) -> np.ndarray:
    """Distance along a polyline at each vertex, in metres."""
    steps = np.linalg.norm(np.diff(_planar(lon_lat, origin_lat), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def project(line: np.ndarray, points: np.ndarray, origin_lat: float) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto a polyline: distance along it, and distance from it."""
    xy = _planar(line, origin_lat)
    pts = _planar(points, origin_lat)
    start, seg = xy[:-1], np.diff(xy, axis=0)
    length2 = np.maximum((seg ** 2).sum(axis=1), 1e-9)
    # (points x segments) parameter of the nearest point on each segment.
    t = np.clip(((pts[:, None, :] - start) * seg).sum(axis=2) / length2, 0.0, 1.0)
    offset = np.linalg.norm(start + t[..., None] * seg - pts[:, None, :], axis=2)
    best = offset.argmin(axis=1)
    rows = np.arange(len(points))
    along = cumulative_metres(line, origin_lat)[best] + t[rows, best] * np.sqrt(length2[best])
    return along, offset[rows, best]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _run_dir(run: str) -> Path:
    return config.DATA_DIR / "runs" / run if run else config.DATA_DIR


def fetch_routes(refresh: bool = False) -> dict:
    """Line names, colours and canonical shapes from V3, cached on first use."""
    if ROUTES_CACHE.exists() and not refresh:
        return json.loads(ROUTES_CACHE.read_text(encoding="utf-8"))
    session = make_session(config.get_api_key())
    routes = get_json(config.V3_BASE + "/routes", session, params={
        "filter[id]": ",".join(config.SUBWAY_ROUTES),
        "fields[route]": "color,long_name,short_name,sort_order",
    })
    payload = {"routes": {item["id"]: item["attributes"] for item in routes["data"]}, "shapes": {}}
    for route in config.SUBWAY_ROUTES:
        shapes = get_json(config.V3_BASE + "/shapes", session, params={"filter[route]": route})
        # "canonical-" shapes are MBTA's reference geometry for each branch and
        # direction; the rest are shuttles and diversions.
        payload["shapes"][route] = sorted(
            item["attributes"]["polyline"] for item in shapes["data"]
            if item["id"].startswith("canonical-"))
    ROUTES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ROUTES_CACHE.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def load_trips(run: str, service_date: int | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(
        _run_dir(run) / "processed" / "trips_clean.parquet", columns=_TRIP_COLUMNS,
        filters=[("service_date", "==", service_date)] if service_date else None)
    frame["branch"] = frame["branch_route_id"].astype("object").fillna(frame["route_id"].astype("object"))
    return frame


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def build_lines(routes: dict) -> list[dict]:
    lines = []
    for route_id in config.SUBWAY_ROUTES:
        attrs = routes[route_id]
        short = attrs.get("short_name") or attrs["long_name"].removesuffix(" Line")
        lines.append({"id": route_id, "name": attrs["long_name"], "short": short,
                      "color": "#" + attrs["color"]})
    return lines


def build_patterns(trips: pd.DataFrame, coords: pd.DataFrame, shapes: dict[str, list[np.ndarray]],
                   origin_lat: float) -> list[dict]:
    """One geometry per branch, with its stations in order along the track.

    ``coords`` is indexed by station id with ``longitude``/``latitude`` columns;
    ``shapes`` maps each route to its candidate lon/lat polylines. The branch's
    shape is the one its stations sit closest to, which is what separates
    Ashmont from Braintree on the shared Red Line trunk.
    """
    visits = trips.drop_duplicates(["branch", "trip_id", "parent_station"])
    per_branch = visits.groupby("branch")["trip_id"].nunique()
    share = visits.groupby(["branch", "parent_station"]).size() / per_branch
    route_of = trips.drop_duplicates("branch").set_index("branch")["route_id"].astype(str)

    patterns = []
    for branch in sorted(per_branch.index):
        # Trips with no branch id fall back to their route; where the route has
        # real branches those trips ride one of them instead of a line of their own.
        if branch == route_of[branch] and (route_of == branch).sum() > 1:
            continue
        members = share.loc[branch]
        station_ids = members.index[members >= MIN_BRANCH_SHARE].intersection(coords.index)
        points = coords.loc[station_ids, ["longitude", "latitude"]].to_numpy()
        best = None
        for line in shapes.get(route_of[branch], []):
            along, offset = project(line, points, origin_lat)
            score = np.percentile(offset, 95)
            if best is None or score < best[0]:
                best = (score, line, along, offset)
        if best is None:
            log.warning("no canonical shape for %s; branch skipped", branch)
            continue
        _, line, along, offset = best
        keep = offset <= MAX_OFFSET_M
        if not keep.all():
            log.warning("%s: %d station(s) too far from the shape: %s", branch,
                        (~keep).sum(), list(station_ids[~keep]))
        order = np.argsort(along[keep])
        patterns.append({
            "id": branch,
            "line": route_of[branch],
            "coords": np.round(line, 5).tolist(),
            "dist": np.round(cumulative_metres(line, origin_lat)).astype(int).tolist(),
            "stations": list(station_ids[keep][order]),
            "at": np.round(along[keep][order]).astype(int).tolist(),
        })
    return patterns


def build_stations(coords: pd.DataFrame, trips: pd.DataFrame, run: str = "") -> list[dict]:
    """Every station on a branch, with the clustering results of ``run``."""
    processed = _run_dir(run) / "processed"
    clusters = pd.read_parquet(processed / "station_clusters.parquet").set_index("station_name")
    profiles = pd.read_parquet(processed / "demand_profiles.parquet")
    profile_cols = [f"p{i:02d}" for i in range(48)]
    served = trips.groupby("parent_station")["route_id"].agg(
        lambda s: sorted(set(s.astype(str)), key=config.SUBWAY_ROUTES.index))

    def num(value: object, digits: int = 0) -> float | int | None:
        if pd.isna(value):
            return None
        return round(float(value), digits) if digits else int(round(float(value)))

    stations = []
    for station_id, row in coords.loc[coords.index.intersection(served.index)].iterrows():
        name = row["stop_name"]
        stats = clusters.loc[name] if name in clusters.index else None
        profile = profiles.loc[name, profile_cols] if name in profiles.index else None
        stations.append({
            "id": station_id,
            "name": name,
            "coords": [round(row["longitude"], 5), round(row["latitude"], 5)],
            "lines": served[station_id],
            "onTime": None if stats is None else num(stats["on_time_rate"], 3),
            "median": None if stats is None else num(stats["median_lateness"]),
            "p90": None if stats is None else num(stats["p90_lateness"]),
            "entries": None if stats is None else num(stats["total_entries"]),
            "reliability": None if stats is None else stats["reliability_cluster"],
            "demand": None if stats is None or pd.isna(stats["demand_cluster"]) else stats["demand_cluster"],
            "profile": None if profile is None else np.round(profile.to_numpy(float), 4).tolist(),
        })
    return sorted(stations, key=lambda s: s["name"])


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def build_replay(day: pd.DataFrame, predictions: pd.DataFrame, station_index: dict[str, int],
                 patterns: list[dict]) -> dict:
    """Pack one service day of observed trips into compact per-trip arrays.

    Times are seconds after the start of the service day, the same anchor the
    timetable uses, so times after midnight run past 86,400 instead of wrapping.
    """
    day = day[day["parent_station"].isin(station_index)].sort_values(["trip_id", "stop_sequence"])
    # Stop times must strictly increase along a trip. A record that goes back in
    # time (or repeats a station) cannot be drawn as motion, so it is dropped.
    before = day.groupby("trip_id")["stop_timestamp"].cummax().groupby(day["trip_id"]).shift()
    repeat = day["parent_station"].eq(day.groupby("trip_id")["parent_station"].shift())
    day = day[(before.isna() | (day["stop_timestamp"] > before)) & ~repeat]
    day = day.merge(predictions, on=["trip_id", "stop_id"], how="left")

    anchor = (day["scheduled_epoch"] - day["scheduled_arrival_time"]).mode().iloc[0]
    by_line: dict[str, list[tuple[int, set[str]]]] = {}
    for index, pattern in enumerate(patterns):
        by_line.setdefault(pattern["line"], []).append((index, set(pattern["stations"])))

    trips = []
    for trip_id, rows in day.groupby("trip_id", sort=False):
        stops = list(rows["parent_station"])
        candidates = by_line.get(str(rows["route_id"].iloc[0]), [])
        if not candidates:
            continue
        # The branch that contains the most of the trip's stops carries it.
        pattern, members = max(candidates, key=lambda c: sum(s in c[1] for s in stops))
        rows = rows[rows["parent_station"].isin(members)]
        if len(rows) < 2:
            continue
        arrive = (rows["stop_timestamp"].to_numpy() - anchor).round()
        # A train leaves stop i when it starts moving toward stop i + 1.
        leave = np.append(rows["move_timestamp"].to_numpy()[1:] - anchor, np.nan).round()
        leave = np.where((leave >= arrive) & (leave <= np.append(arrive[1:], np.inf)), leave, arrive)
        predicted = rows["predicted_delay"].round()
        trips.append({
            "id": str(trip_id),
            "line": str(rows["route_id"].iloc[0]),
            "pattern": pattern,
            "dest": rows["direction_destination"].iloc[0],
            "s": [station_index[s] for s in rows["parent_station"]],
            "a": arrive.astype(int).tolist(),
            "w": (leave - arrive).astype(int).tolist(),
            "l": rows["lateness_seconds"].round().astype(int).tolist(),
            "d": rows["delay_seconds"].round().astype(int).tolist(),
            "p": [None if pd.isna(v) else int(v) for v in predicted],
        })
    trips.sort(key=lambda trip: trip["a"][0])

    scored = day.dropna(subset=["predicted_delay"])
    return {
        "anchor": int(anchor),
        "summary": {
            "trips": len(trips),
            "stops": sum(len(trip["s"]) for trip in trips),
            "late": round(float(day.loc[clean.is_arrival(day), "late"].mean()), 3),
            "maeModel": round(float((scored["predicted_delay"] - scored["delay_seconds"]).abs().mean()), 1),
            "maePersistence": round(float((scored["persistence_delay"] - scored["delay_seconds"]).abs().mean()), 1),
        },
        "trips": trips,
    }


_ALERT_COLUMNS = ["id", "cause", "effect", "header_text.translation.text", "created_datetime",
                  "closed_datetime", "active_period.end_datetime", "informed_entity.route_id",
                  "informed_entity.stop_id"]
#: The service day runs from 03:00 to 03:00 the next morning, local time.
SERVICE_DAY_START = pd.Timedelta(hours=3)


def build_incidents(alerts: pd.DataFrame, date: str, anchor: int, station_index: dict[str, int],
                    stop_parent: dict[str, str]) -> list[dict]:
    """Service incidents reported on one service day, for the replay timeline.

    The archive has one row per alert, informed entity and update; an alert
    starts when it was created and ends when it was closed (or, if never closed,
    when its active period ran out). Times are seconds after the replay's anchor,
    like the trips'. Elevator and escalator outages are left out, as in the
    features. Alert times are local, without a zone.
    """
    if alerts.empty:
        return []
    alerts = alerts[alerts["informed_entity.route_id"].isin(config.SUBWAY_ROUTES)
                    & ~alerts["effect"].isin(NON_SERVICE_ALERT_EFFECTS)]
    start = pd.Timestamp(date) + SERVICE_DAY_START
    created = pd.to_datetime(alerts["created_datetime"], errors="coerce")
    alerts = alerts[(created >= start) & (created < start + pd.Timedelta(days=1))]

    def seconds(moment) -> int | None:
        if pd.isna(moment):
            return None
        epoch = pd.Timestamp(moment).tz_localize(config.SERVICE_TZ, ambiguous="NaT",
                                                 nonexistent="shift_forward")
        return None if pd.isna(epoch) else int(epoch.timestamp()) - anchor

    incidents = []
    for alert_id, rows in alerts.groupby("id", sort=False):
        rows = rows.sort_values("created_datetime")
        end = pd.to_datetime(rows["closed_datetime"], errors="coerce").max()
        if pd.isna(end):
            end = pd.to_datetime(rows["active_period.end_datetime"], errors="coerce").max()
        stops = {stop_parent.get(str(s), str(s)) for s in rows["informed_entity.stop_id"].dropna()}
        headers = rows["header_text.translation.text"].dropna()
        incidents.append({
            "id": str(alert_id),
            "start": seconds(rows["created_datetime"].iloc[0]),
            "end": seconds(end),
            "lines": sorted(rows["informed_entity.route_id"].astype(str).unique()),
            "stations": sorted(station_index[s] for s in stops if s in station_index),
            "effect": str(rows["effect"].iloc[-1]),
            "cause": None if pd.isna(rows["cause"].iloc[-1]) else str(rows["cause"].iloc[-1]),
            "text": str(headers.iloc[-1]) if len(headers) else "",
        })
    return sorted((i for i in incidents if i["start"] is not None), key=lambda i: i["start"])


def load_day_alerts() -> pd.DataFrame:
    """The alerts archive, or an empty frame when it has not been downloaded."""
    try:
        return collect_lamp.load_alerts(columns=_ALERT_COLUMNS)
    except FileNotFoundError:
        log.warning("no alerts archive: replays will have no incidents")
        return pd.DataFrame(columns=_ALERT_COLUMNS)


def _dump(path: Path, payload: dict) -> int:
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return len(text)


def run(days: tuple[tuple[str, str, str, str], ...] = DEFAULT_DAYS, out_dir: Path = OUT_DIR,
        refresh: bool = False) -> dict:
    """Write network.json and one replay file per day that has data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    routes = fetch_routes(refresh)
    coords = load_stations().query("location_type == 1").set_index("stop_id")
    trips = load_trips("")
    origin_lat = float(coords["latitude"].mean())

    shapes = {route: [decode_polyline(s) for s in encoded] for route, encoded in routes["shapes"].items()}
    patterns = build_patterns(trips, coords, shapes, origin_lat)
    stations = build_stations(coords, trips)
    station_index = {station["id"]: i for i, station in enumerate(stations)}
    all_coords = np.array([c for p in patterns for c in p["coords"]])
    window = json.loads((config.DATA_DIR / "analysis_window.json").read_text(encoding="utf-8"))

    alerts = load_day_alerts()
    stop_parent = dict(zip(trips["stop_id"].astype(str), trips["parent_station"].astype(str)))
    replays, written = [], {}
    for run_name, date, label, note in days:
        service_date = int(date.replace("-", ""))
        processed = _run_dir(run_name) / "processed"
        if not (processed / "trips_clean.parquet").exists():
            log.warning("skipping %s: run %r has not been built", date, run_name or "spring")
            continue
        day = load_trips(run_name, service_date)
        if day.empty:
            log.warning("skipping %s: no trips in run %r", date, run_name or "spring")
            continue
        predictions = pd.read_parquet(
            processed / "predictions.parquet",
            columns=["service_date_parsed", "trip_id", "stop_id", "predicted_delay", "persistence_delay"])
        predictions = predictions[predictions.pop("service_date_parsed").astype(str) == date]
        replay = build_replay(day, predictions, station_index, patterns)
        replay["incidents"] = build_incidents(alerts, date, replay["anchor"], station_index, stop_parent)
        replay["summary"]["incidents"] = len(replay["incidents"])
        file = f"replay-{date}.json"
        written[file] = _dump(out_dir / file, {"date": date, "label": label, **replay})
        replays.append({"date": date, "label": label, "note": note, "file": file, **replay["summary"]})

    network = {
        "window": [window["start"], window["end"]],
        "bounds": [all_coords.min(axis=0).round(4).tolist(), all_coords.max(axis=0).round(4).tolist()],
        "lines": build_lines(routes["routes"]),
        "patterns": patterns,
        "stations": stations,
        "replays": replays,
    }
    written["network.json"] = _dump(out_dir / "network.json", network)
    for file, size in written.items():
        log.info("wrote %s (%.0f KB)", file, size / 1024)
    return {"out_dir": str(out_dir), "files": written, "patterns": len(patterns),
            "stations": len(stations), "replays": [r["date"] for r in replays]}
