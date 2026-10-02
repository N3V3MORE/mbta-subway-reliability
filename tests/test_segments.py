"""Tests for where the T loses time: stretches, platforms, lost time and terminals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import segments


def _trip(trip_id: str, stations: list[str], running: list[float], dwell: list[float | None], *,
          route: str = "Red", date: int = 20260601, hour: int = 8) -> list[dict]:
    """Rows of one trip in the clean table's shape. `running[i]` is the time into stop i."""
    return [{"service_date": date, "trip_id": trip_id, "run": 0, "route_id": route, "direction_id": 0,
             "parent_station": s, "station_name": s.upper(), "scheduled_arrival_time": 30_000 + 120 * i,
             "stop_sequence": 10 * (i + 1), "travel_time_seconds": running[i], "dwell_time_seconds": dwell[i],
             "scheduled_hour": hour, "is_weekend": False, "time_inconsistent": False}
            for i, s in enumerate(stations)]


def _frame(trips: int = 400, slow_every: int = 10) -> pd.DataFrame:
    """A-B-C trips: A->B always 60 s, B->C 60 s except every `slow_every`th trip at 180 s."""
    rows = []
    for t in range(trips):
        b_to_c = 180.0 if t % slow_every == 0 else 60.0
        rows += _trip(f"t{t}", ["A", "B", "C"], [np.nan, 60.0, b_to_c], [None, 30.0 + (t % 2) * 20, None])
    return pd.DataFrame(rows)


def test_traversals_pair_consecutive_stops_of_one_run():
    rows = segments.traversals(_frame())
    assert set(zip(rows["from_station"], rows["to_station"])) == {("A", "B"), ("B", "C")}
    assert (rows["line"] == "Red").all() and len(rows) == 800


def test_a_rare_pair_spanning_an_unrecorded_stop_is_not_a_stretch():
    frame = _frame()
    # Three trips never recorded B, so they appear to run A->C in one go.
    skipped = pd.DataFrame([row for i in range(3) for row in _trip(f"s{i}", ["A", "C"], [np.nan, 130.0], [None, None])])
    rows = segments.traversals(pd.concat([frame, skipped], ignore_index=True))
    assert ("A", "C") not in set(zip(rows["from_station"], rows["to_station"]))


def test_recording_faults_and_backwards_times_are_not_running_times():
    frame = _frame()
    frame.loc[(frame["trip_id"] == "t1") & (frame["parent_station"] == "C"), "travel_time_seconds"] = 57_000
    frame.loc[(frame["trip_id"] == "t2") & (frame["parent_station"] == "C"), "time_inconsistent"] = True
    rows = segments.traversals(frame)
    assert rows["seconds"].max() <= segments.MAX_RUNNING_SECONDS and len(rows) == 798


def test_lost_time_is_measured_from_the_places_own_good_run():
    rows = segments.add_lost(segments.traversals(_frame()), segments.STRETCH_KEY)
    bc = rows[rows["to_station"] == "C"]
    assert bc["reference"].iloc[0] == 60  # a good run, not the slow tenth
    assert sorted(bc["lost"].unique()) == [0, 120]
    assert (rows.loc[rows["to_station"] == "B", "lost"] == 0).all()


def test_summary_per_place_and_per_day():
    frame = _frame()
    running = segments.add_lost(segments.traversals(frame), segments.STRETCH_KEY)
    table = segments.summarise(running, segments.STRETCH_KEY, days=1).set_index("to_station")
    # 40 of 400 trains lose 120 s on B->C: 80 train-minutes on the one day.
    assert table.loc["C", "lost_minutes_per_day"] == pytest.approx(80)
    assert table.loc["C", "median_lost"] == 0 and table.loc["C", "p90_lost"] > 0
    assert table.index[0] == "C"  # sorted by daily total


def test_platform_time_lost_is_counted_per_station():
    visits = segments.add_lost(segments.platform_visits(_frame()), segments.PLATFORM_KEY)
    assert set(visits["station"]) == {"B"}  # no platform time is recorded at the origin or the end
    assert sorted(visits["lost"].unique()) == [0, 20]


def test_a_layover_at_the_last_stop_is_not_platform_time_lost():
    frame = _frame()
    frame.loc[frame["parent_station"] == "C", "dwell_time_seconds"] = 600.0
    assert set(segments.platform_visits(frame)["station"]) == {"B"}


def test_terminals_are_where_most_trips_end():
    assert segments.terminals(_frame()) == {("Red", 0, "C")}
