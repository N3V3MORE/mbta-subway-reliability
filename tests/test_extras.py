"""Tests for the supporting analyses behind the report's narrative claims."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import extras


def test_stickiness_slope_and_recovery_share():
    prev = np.array([0.0, 100.0, 200.0, 300.0])
    feats = pd.DataFrame({"route_id": "Red", "prev_delay_1": prev,
                          "delay_seconds": prev + [0.0, 0.0, -90.0, 0.0]})
    table = extras.stickiness(feats)
    assert table.loc[0, "share_recovering_over_60s"] == pytest.approx(0.25)
    assert table.attrs["network_share_recovering_over_60s"] == pytest.approx(0.25)
    assert 0.5 < table.loc[0, "slope"] < 1.0


def test_final_hop_separates_the_last_hop_of_each_run():
    frame = pd.DataFrame({
        "service_date": 20260601, "trip_id": "1", "run": 0, "route_id": "Red",
        "scheduled_arrival_time": [0, 60, 120, 180], "stop_sequence": [1, 2, 3, 4],
        "delay_seconds": [0.0, 10.0, 20.0, 80.0],
        "is_origin": [True, False, False, False], "time_inconsistent": False,
    })
    row = extras.final_hop(frame).iloc[0]
    assert row["final_median"] == 60.0
    assert row["mid_median"] == 10.0
