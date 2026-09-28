"""Tests for the tail metrics: early-warning scoring, calibration and ranges."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import model_delay, model_tail


class TestEarlyWarningMetrics:
    def test_recall_at_precision_on_a_perfect_ranking(self):
        y = np.array([0, 0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        assert model_tail.recall_at_precision(y, score, 0.5) == 1.0

    def test_recall_at_precision_when_no_threshold_is_precise_enough(self):
        y = np.array([1, 0, 0, 0])
        score = np.array([0.1, 0.9, 0.8, 0.7])   # the positive ranks last
        # Catching it means flagging all four (precision 0.25 < 0.5): recall 0.
        assert model_tail.recall_at_precision(y, score, 0.5) == 0.0
        assert model_tail.recall_at_precision(y, score, 0.25) == 1.0

    def test_no_positives_gives_nan(self):
        assert np.isnan(model_tail.recall_at_precision(np.zeros(3), np.arange(3.0), 0.5))

    def test_calibration_bins_compare_predicted_with_observed(self):
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.01, 0.01, 0.9, 0.9])
        table = model_tail.calibration_table(y, proba).set_index("bin")
        assert table["observed"].tolist() == [0.0, 1.0]
        assert table["arrivals"].tolist() == [2, 2]


class TestRanges:
    def test_band_coverage_and_width(self):
        y = np.array([0.0, 50.0, 100.0, 500.0])
        low, high = np.full(4, -10.0), np.full(4, 110.0)
        metrics = model_tail.band_metrics(y, low, high)
        assert metrics["coverage"] == 0.75
        assert metrics["median_width_seconds"] == 120.0

    def test_pinball_loss_penalises_under_prediction_more_at_high_quantiles(self):
        y = np.array([100.0])
        under = model_tail.pinball_loss(y, np.array([0.0]), 0.9)
        over = model_tail.pinball_loss(y, np.array([200.0]), 0.9)
        assert under == pytest.approx(90.0)
        assert over == pytest.approx(10.0)


def test_day_type_flags_route_days_with_many_big_delays():
    frame = pd.DataFrame({
        "route_id": ["Red"] * 5 + ["Blue"] * 5,
        "service_date": [1] * 10,
        "delay_seconds": [700, 0, 0, 0, 0] + [0] * 5,   # Red: 20% over 10 min
    })
    assert model_delay.day_type(frame).tolist() == ["disrupted"] * 5 + ["normal"] * 5
