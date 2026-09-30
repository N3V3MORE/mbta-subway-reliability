"""Tests for the 3D map export: geometry helpers, branch building, replay packing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import export_map

# Four stations 1.1 km apart due north, then a fifth that branches north-east.
COORDS = pd.DataFrame(
    {"longitude": [-71.0, -71.0, -71.0, -71.0, -70.99], "latitude": [42.30, 42.31, 42.32, 42.33, 42.33],
     "stop_name": ["A", "B", "C", "D", "E"]},
    index=["pA", "pB", "pC", "pD", "pE"],
)
LAT = 42.3
TRUNK_TO_D = np.array([[-71.0, 42.30], [-71.0, 42.33]])
TRUNK_TO_E = np.array([[-71.0, 42.30], [-71.0, 42.32], [-70.99, 42.33]])


def _trip_rows(trip_id: str, branch: str | None, stations: list[str], *, start: int = 1_000) -> list[dict]:
    return [{"trip_id": trip_id, "route_id": "Red", "branch_route_id": branch, "parent_station": s,
             "stop_id": f"{s}-{trip_id}", "stop_sequence": i * 10, "stop_timestamp": float(start + 120 * i)}
            for i, s in enumerate(stations)]


def test_decode_polyline_matches_the_reference_example():
    # Google's documented example: (38.5, -120.2), (40.7, -120.95), (43.252, -126.453).
    decoded = export_map.decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@")
    np.testing.assert_allclose(decoded, [[-120.2, 38.5], [-120.95, 40.7], [-126.453, 43.252]])


def test_project_gives_distance_along_and_off_the_line():
    points = np.array([[-71.0, 42.31], [-70.999, 42.32]])
    along, offset = export_map.project(TRUNK_TO_D, points, LAT)
    np.testing.assert_allclose(along, [1_105.4, 2_210.8], rtol=1e-3)
    assert offset[0] == pytest.approx(0, abs=1e-6)
    assert offset[1] == pytest.approx(82, rel=0.02)  # 0.001 degrees of longitude at 42.3 N


def test_branches_pick_the_shape_their_stations_lie_on():
    trips = pd.DataFrame(
        _trip_rows("a1", "Red-A", ["pA", "pB", "pC", "pD"])
        + _trip_rows("b1", "Red-B", ["pA", "pB", "pC", "pE"])
        # No branch id: rides the trunk, so it must not become a pattern of its own.
        + _trip_rows("x1", None, ["pA", "pB"]))
    trips["branch"] = trips["branch_route_id"].fillna(trips["route_id"])
    patterns = export_map.build_patterns(trips, COORDS, {"Red": [TRUNK_TO_D, TRUNK_TO_E]}, LAT)

    assert [p["id"] for p in patterns] == ["Red-A", "Red-B"]
    assert patterns[0]["stations"] == ["pA", "pB", "pC", "pD"]
    assert patterns[1]["stations"] == ["pA", "pB", "pC", "pE"]
    assert patterns[1]["coords"] == TRUNK_TO_E.tolist()
    assert all(np.diff(p["at"]).min() > 0 for p in patterns)


def test_replay_keeps_observed_order_and_joins_predictions():
    patterns = [{"line": "Red", "stations": ["pA", "pB", "pC", "pD"]}]
    day = pd.DataFrame(_trip_rows("t1", "Red-A", ["pA", "pB", "pC", "pD"]))
    day["direction_destination"] = "Alewife"
    day["move_timestamp"] = day["stop_timestamp"] - 30
    day["delay_seconds"] = [0.0, 40.4, 80.0, 130.0]
    day["lateness_seconds"] = [0.0, 20.0, 400.0, 360.2]
    day["late"] = day["lateness_seconds"] > 300
    day["is_origin"] = [True, False, False, False]
    day["time_inconsistent"] = False
    day["scheduled_arrival_time"] = 36_000.0
    day["scheduled_epoch"] = 1_000_000.0 + day["scheduled_arrival_time"]
    # A record that goes back in time cannot be drawn as motion.
    day.loc[2, "stop_timestamp"] = 1_050.0
    predictions = pd.DataFrame({"trip_id": ["t1"], "stop_id": ["pB-t1"], "predicted_delay": [35.6], "persistence_delay": [0.0]})

    replay = export_map.build_replay(day, predictions, {s: i for i, s in enumerate(COORDS.index)}, patterns)

    trip, = replay["trips"]
    assert replay["anchor"] == 1_000_000
    assert trip["s"] == [0, 1, 3]
    assert trip["a"] == [-999_000, -998_880, -998_640]
    # Leaving stop i is the move toward stop i + 1; the last stop has no departure.
    assert trip["w"] == [90, 210, 0]
    assert trip["d"] == [0, 40, 130]
    assert trip["l"] == [0, 20, 360]
    # Share of arrivals (the origin excluded) that riders found late: B no, D yes.
    assert replay["summary"]["late"] == 0.5
    assert trip["p"] == [None, 36, None]
    assert replay["summary"]["maeModel"] == pytest.approx(4.8)


def test_incidents_are_the_days_service_alerts_with_start_end_and_stations():
    rows = [
        # Raised 07:10, updated, then closed 07:34: one incident.
        {"id": 1, "cause": "TECHNICAL_PROBLEM", "effect": "DETOUR", "header_text.translation.text": "Shuttles",
         "created_datetime": "2026-09-16 07:10:00", "closed_datetime": None,
         "active_period.end_datetime": "2026-09-16 09:10:00", "informed_entity.route_id": "Green-C",
         "informed_entity.stop_id": "70223"},
        {"id": 1, "cause": "TECHNICAL_PROBLEM", "effect": "DETOUR", "header_text.translation.text": "Shuttles",
         "created_datetime": "2026-09-16 07:10:00", "closed_datetime": "2026-09-16 07:34:00",
         "active_period.end_datetime": None, "informed_entity.route_id": "Green-C",
         "informed_entity.stop_id": "70223"},
        # Never closed: ends with its active period.
        {"id": 2, "cause": None, "effect": "OTHER_EFFECT", "header_text.translation.text": "Delays",
         "created_datetime": "2026-09-16 14:23:00", "closed_datetime": None,
         "active_period.end_datetime": "2026-09-16 16:23:00", "informed_entity.route_id": "Red",
         "informed_entity.stop_id": None},
        # Left out: an elevator outage, a bus alert, and one raised the day before.
        {"id": 3, "cause": None, "effect": "ACCESSIBILITY_ISSUE", "header_text.translation.text": "Elevator",
         "created_datetime": "2026-09-16 08:00:00", "closed_datetime": None,
         "active_period.end_datetime": None, "informed_entity.route_id": "Red", "informed_entity.stop_id": None},
        {"id": 4, "cause": None, "effect": "DETOUR", "header_text.translation.text": "Bus",
         "created_datetime": "2026-09-16 08:00:00", "closed_datetime": None,
         "active_period.end_datetime": None, "informed_entity.route_id": "1", "informed_entity.stop_id": None},
        {"id": 5, "cause": None, "effect": "OTHER_EFFECT", "header_text.translation.text": "Yesterday",
         "created_datetime": "2026-09-16 02:00:00", "closed_datetime": None,
         "active_period.end_datetime": None, "informed_entity.route_id": "Red", "informed_entity.stop_id": None},
    ]
    alerts = pd.DataFrame(rows)
    for col in ("created_datetime", "closed_datetime", "active_period.end_datetime"):
        alerts[col] = pd.to_datetime(alerts[col])
    # Service-day anchor: local midnight of 16 September (EDT, UTC-4).
    anchor = int(pd.Timestamp("2026-09-16", tz="America/New_York").timestamp())
    incidents = export_map.build_incidents(alerts, "2026-09-16", anchor, {"place-clmnl": 7},
                                           {"70223": "place-clmnl"})
    assert [i["id"] for i in incidents] == ["1", "2"]
    shuttle, delay = incidents
    assert (shuttle["start"], shuttle["end"]) == (7 * 3600 + 600, 7 * 3600 + 34 * 60)
    assert shuttle["stations"] == [7] and shuttle["lines"] == ["Green-C"]
    assert delay["end"] == 16 * 3600 + 23 * 60 and delay["stations"] == []


def test_no_alerts_archive_means_no_incidents():
    assert export_map.build_incidents(pd.DataFrame(), "2026-09-16", 0, {}, {}) == []
