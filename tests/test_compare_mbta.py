"""The MBTA's prediction-accuracy rule, applied to our predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbta_ds import compare_mbta


def test_bins_by_predicted_seconds_and_applies_each_window():
    predicted = np.array([100.0, 100.0, 100.0, 300.0, 300.0, 1000.0, 2000.0, -5.0])
    actual = np.array([160.0, 161.0, 40.0, 420.0, 421.0, 1360.0, 2000.0, -5.0])
    table = compare_mbta.accuracy_by_bin(predicted, actual).set_index("bin")
    # 0-3 min: +60 and -60 pass, +61 fails.
    assert table.loc["0-3 min", "predictions"] == 3 and table.loc["0-3 min", "accurate"] == 2
    # 3-6 min: up to 120 s late passes, 121 s fails.
    assert table.loc["3-6 min", "accurate"] == 1
    # 12-30 min: 360 s late passes. Beyond 30 min or in the past: not scored.
    assert table.loc["12-30 min", "accurate"] == 1
    assert table["predictions"].sum() == 6


def test_early_windows_are_asymmetric():
    table = compare_mbta.accuracy_by_bin(np.array([500.0, 500.0]), np.array([350.0, 349.0]))
    assert table.set_index("bin").loc["6-12 min", "accurate"] == 1   # 150 s early ok, 151 s not


def test_mbta_weeks_are_counted_only_when_wholly_inside_the_period(tmp_path, monkeypatch):
    path = tmp_path / "acc.csv"
    pd.DataFrame({"weekly": ["2026-06-19", "2026-06-26", "2026-07-03", "2026-06-26"],
                  "mode": ["subway", "subway", "subway", "bus"], "route_id": ["Red"] * 3 + ["NA"],
                  "bin": ["0-3 min"] * 4, "arrival_departure": ["blended"] * 3 + ["departure"],
                  "num_predictions": [10, 20, 40, 1000], "num_accurate_predictions": [5, 10, 20, 0]}
                 ).to_csv(path, index=False)
    monkeypatch.setattr(compare_mbta, "RAW_PATH", path)
    monkeypatch.setattr(compare_mbta, "download", lambda *a, **k: path)
    # 17-30 June: the week ending 25 June (labelled 26 June) is inside; the one
    # labelled 19 June starts on the 12th and the one labelled 3 July ends on 2 July.
    table = compare_mbta.mbta_accuracy("2026-06-17", "2026-06-30", session=object())
    assert table["predictions"].tolist() == [20] and table["weeks"].tolist() == [1]
