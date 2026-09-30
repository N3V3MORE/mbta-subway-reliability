"""Tests for the plain-language story's data tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import story


def _arrivals() -> pd.DataFrame:
    # Red: half its arrivals 10+ min late. Orange: all on time for riders, though
    # far behind the timetable (drift), which must not count.
    lateness = [0.0, 30.0, 700.0, 900.0, 0.0, 10.0, 20.0, 30.0]
    return pd.DataFrame({
        "route_id": ["Red"] * 4 + ["Orange"] * 4,
        "scheduled_hour": [8, 8, 9, 9] * 2,
        "lateness_seconds": lateness,
        "late": [v > 300 for v in lateness],
        "delay_seconds": lateness[:4] + [3000.0] * 4,
        "station_name": ["A", "B", "A", "B"] * 2,
    })


class TestDelayBands:
    def test_rows_sum_to_one_and_most_late_line_is_last(self):
        table = story.delay_bands(_arrivals())
        assert np.allclose(table.sum(axis=1), 1.0)
        assert list(table.index) == ["Orange", "Red"]
        assert table.loc["Red", "10+ min late"] == pytest.approx(0.5)

    def test_band_edges_match_the_late_label(self):
        # "late" is more than 5 minutes, so exactly 5 minutes is still "1-5".
        frame = _arrivals().assign(lateness_seconds=300.0)
        assert story.delay_bands(frame)["1-5 min late"].eq(1.0).all()


class TestLateByHour:
    def test_thin_cells_are_left_blank(self):
        share = story.late_by_hour(_arrivals(), min_arrivals=3)
        assert share.isna().all().all()

    def test_share_and_service_day_column_order(self):
        share = story.late_by_hour(_arrivals(), min_arrivals=1)
        assert list(share.columns) == story.SERVICE_HOURS
        assert share.loc["Red", 9] == pytest.approx(1.0)
        assert share.index[0] == "Red"


class TestLateStations:
    def test_ranks_by_share_late(self):
        table = story.late_stations(_arrivals(), min_arrivals=1)
        assert table["late_share"].is_monotonic_decreasing
        assert table["late_share"].max() == pytest.approx(0.25)
