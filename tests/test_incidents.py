"""Minute-level incident features: what each prediction could have known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import incidents


def _alert_rows():
    """Alert 1 on the Red Line: raised 08:05 announcing 15 minutes at Park Street,
    revised 08:20 to 30 minutes, closed 08:40. Alert 2 is an elevator outage."""
    common = {"id": 1, "cause": "TECHNICAL_PROBLEM", "effect": "OTHER_EFFECT",
              "created_datetime": "2026-06-10 08:05", "informed_entity.route_id": "Red"}
    return pd.DataFrame([
        {**common, "header_text.translation.text": "Red Line: Delays of about 15 minutes",
         "last_modified_datetime": "2026-06-10 08:05", "closed_datetime": None,
         "active_period.end_datetime": "2026-06-10 10:05", "informed_entity.stop_id": "70075"},
        {**common, "header_text.translation.text": "Red Line: Delays of about 30 minutes",
         "last_modified_datetime": "2026-06-10 08:20", "closed_datetime": None,
         "active_period.end_datetime": "2026-06-10 10:20", "informed_entity.stop_id": "70075"},
        {**common, "header_text.translation.text": "Red Line: Delays of about 30 minutes",
         "last_modified_datetime": "2026-06-10 08:40", "closed_datetime": "2026-06-10 08:40",
         "active_period.end_datetime": None, "informed_entity.stop_id": "70075"},
        {"id": 2, "cause": "MAINTENANCE", "effect": "ACCESSIBILITY_ISSUE", "created_datetime": "2026-06-10 08:00",
         "informed_entity.route_id": "Red", "header_text.translation.text": "Elevator unavailable",
         "last_modified_datetime": "2026-06-10 08:00", "closed_datetime": None,
         "active_period.end_datetime": "2026-06-10 18:00", "informed_entity.stop_id": None},
    ])


def _rows_at(*clock_times, route="Red", station="place-pktrm"):
    known = [pd.Timestamp(f"2026-06-10 {t}", tz="America/New_York").timestamp() for t in clock_times]
    return pd.DataFrame({"service_date": 20260610, "known_at": known, "route_id": route,
                         "parent_station": station})


@pytest.mark.parametrize("text, minutes", [
    ("Red Line: Delays of about 15 minutes due to a signal problem", 15.0),
    ("Green Line D Branch: Delays of up to 20 minutes", 20.0),
    ("Shuttle buses are replacing service", 0.0),
    (None, 0.0),
])
def test_reads_the_announced_delay(text, minutes):
    assert incidents.announced_delay(text) == minutes


def test_each_prediction_sees_the_alert_as_it_stood_then():
    table = incidents.versions(_alert_rows(), {"70075": "place-pktrm"})
    assert len(table) == 2   # two versions in force; the elevator and the close row are not
    out = incidents.add_features(_rows_at("08:00", "08:10", "08:25", "08:45"), table)
    assert out["incident_active"].tolist() == [0, 1, 1, 0]
    # 08:10 must not see the 08:20 revision to 30 minutes.
    assert out["incident_delay_minutes"].tolist() == [0, 15, 30, 0]
    assert out["incident_minutes_since"].tolist()[1:3] == [5, 20]
    assert np.isnan(out["incident_minutes_since"].iloc[0])
    assert out["incident_technical"].tolist() == [0, 1, 1, 0]
    assert out["incident_at_station"].tolist() == [0, 1, 1, 0]


def test_other_lines_and_stations_are_unaffected():
    table = incidents.versions(_alert_rows(), {"70075": "place-pktrm"})
    other_line = incidents.add_features(_rows_at("08:10", route="Orange"), table)
    other_station = incidents.add_features(_rows_at("08:10", station="place-harsq"), table)
    assert other_line["incident_active"].iloc[0] == 0
    assert other_station["incident_active"].iloc[0] == 1
    assert other_station["incident_at_station"].iloc[0] == 0


def test_an_unclosed_version_expires():
    rows = _alert_rows().iloc[[0]].assign(**{"active_period.end_datetime": None})
    table = incidents.versions(rows, {})
    assert (table["end"] - table["start"]).iloc[0] == pd.Timedelta(hours=incidents.MAX_VERSION_HOURS)
