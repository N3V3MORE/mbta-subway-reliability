"""Cross-check our cleaned timestamps against TransitMatters' published data.

TransitMatters (dashboard.transitmatters.org) derives travel times from the same
MBTA archive independently. If our loading, cleaning and timezone handling are
right, the same trains appear in both at the same times. For a stretch of each
heavy-rail line on a few test-period days, each of their trips is matched to ours
by arrival time, and the arrival and travel times are compared.

Optional stage (`python -m mbta_ds.cli crosscheck`): it needs a third-party web
service, so it is not part of `make all`.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import clean, config
from .http import get_json, make_session

log = logging.getLogger(__name__)

API = "https://dashboard-api.labs.transitmatters.org/api/traveltimes/{date}"
OUT_PATH = config.PROCESSED_DIR / "crosscheck.csv"
ROUTES = ("Red", "Orange", "Blue")
DATES = ("2026-06-10", "2026-06-17", "2026-06-24")
STOPS_APART = 8
MATCH_TOLERANCE_SECONDS = 60


def _local(epoch: pd.Series) -> pd.Series:
    return pd.to_datetime(epoch, unit="s", utc=True).dt.tz_convert(config.SERVICE_TZ).dt.tz_localize(None)


def our_trips(frame: pd.DataFrame, route: str) -> tuple[str, str, pd.DataFrame]:
    """The busiest origin platform on ``route`` (direction 0) and the stop
    ``STOPS_APART`` later, with each run's departure from one and arrival at the other."""
    rows = frame[(frame["route_id"] == route) & (frame["direction_id"] == 0)]
    runs = rows.groupby(clean.RUN_KEY, sort=False)
    rows = rows.assign(depart=runs["move_timestamp"].shift(-1), later_stop=runs["stop_id"].shift(-STOPS_APART),
                       later_arrival=runs["stop_timestamp"].shift(-STOPS_APART))
    rows = rows[rows["later_stop"].notna()]   # the origin needs a stop that far ahead
    origin = rows["stop_id"].value_counts().index[0]
    rows = rows[rows["stop_id"] == origin]
    dest = rows["later_stop"].value_counts().index[0]
    rows = rows[rows["later_stop"] == dest].dropna(subset=["depart", "later_arrival"])
    trips = pd.DataFrame({"arr": _local(rows["later_arrival"]),
                          "ours_travel": rows["later_arrival"] - rows["depart"]})
    return origin, dest, trips.sort_values("arr")


def run() -> dict:
    """Compare every TransitMatters trip on the sample days with ours."""
    frame = clean.load().sort_values(clean.TRIP_ORDER)
    session = make_session()
    rows = []
    for route in ROUTES:
        origin, dest, ours = our_trips(frame, route)
        for day in DATES:
            theirs = pd.DataFrame(get_json(API.format(date=day), session,
                                           params={"from_stop": origin, "to_stop": dest}))
            if theirs.empty:
                continue
            theirs = (theirs.assign(arr=pd.to_datetime(theirs["arr_dt"]))
                      .sort_values("arr")[["arr", "travel_time_sec"]])
            matched = pd.merge_asof(theirs, ours.assign(ours_arr=ours["arr"]), on="arr",
                                    direction="nearest",
                                    tolerance=pd.Timedelta(seconds=MATCH_TOLERANCE_SECONDS))
            hit = matched["ours_arr"].notna()
            rows.append({
                "route": route, "date": day, "from_stop": origin, "to_stop": dest,
                "their_trips": len(theirs), "matched": int(hit.sum()),
                "median_arrival_gap_s": float((matched.loc[hit, "arr"] - matched.loc[hit, "ours_arr"])
                                              .dt.total_seconds().abs().median()),
                "median_travel_their_s": float(theirs["travel_time_sec"].median()),
                "median_travel_ours_s": float(matched.loc[hit, "ours_travel"].median()),
            })
            log.info("%s %s: %d/%d trips matched", route, day, rows[-1]["matched"], len(theirs))
    table = pd.DataFrame(rows).assign(match_rate=lambda t: t["matched"] / t["their_trips"])
    table.to_csv(OUT_PATH, index=False)
    return {"pairs": len(table), "match_rate": float(table["matched"].sum() / table["their_trips"].sum())}
