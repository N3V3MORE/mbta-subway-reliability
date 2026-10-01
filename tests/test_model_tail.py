"""Tests for the tail metrics: early-warning scoring, calibration and ranges."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import model_delay, model_tail


class TestEarlyWarningMetrics:
    def test_horizon_evaluation_handles_an_unknown_previous_lateness(self, monkeypatch):
        from mbta_ds import features
        from .test_model_delay import _toy_frame

        frame = _toy_frame()
        frame["lateness_seconds"] = np.where(np.arange(len(frame)) % 2, 700.0, 100.0)
        frame.loc[frame["service_date"] == frame["service_date"].max() - 1, "lateness_seconds"] = 100.0
        frame.loc[frame.index[::5], "prev_lateness_1"] = np.nan
        monkeypatch.setattr(features, "build", lambda **kwargs: frame.copy())

        class FixedModel:
            def predict_proba(self, X):
                return np.tile([0.5, 0.5], (len(X), 1))

            def predict(self, X):
                return np.zeros(len(X))

        def fit(model, rows, numeric, categorical, target):
            if target == "loses_5min":
                assert rows["prev_lateness_1"].notna().all()
            return FixedModel()

        monkeypatch.setattr(model_delay, "_fit", fit)
        result = model_tail.evaluate_horizon(1, quick=True)
        baseline = next(row for row in result["warning"]
                        if row["method"] == "baseline: how late the train is now"
                        and row["subset"] == "all arrivals")
        assert baseline["n"] == model_delay.temporal_split(frame).n_test
        assert np.isfinite(baseline["pr_auc"])

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


def test_day_type_flags_route_days_where_riders_waited_long():
    frame = pd.DataFrame({
        "route_id": ["Red"] * 5 + ["Blue"] * 5,
        "service_date": [1] * 10,
        # Red: 20% of arrivals kept riders waiting 10+ min extra. Blue: an hour
        # behind the timetable all day, but at the planned spacing (drift).
        "lateness_seconds": [700, 0, 0, 0, 0] + [0] * 5,
        "delay_seconds": [700, 0, 0, 0, 0] + [3600] * 5,
    })
    assert model_delay.day_type(frame).tolist() == ["disrupted"] * 5 + ["normal"] * 5


class TestConformalMargin:
    def test_widens_a_too_narrow_band_to_the_requested_coverage(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 20_000)
        # A band of +-0.5 covers ~38%; calibration must widen it to ~80%.
        margin = model_tail.conformal_margin(y[:10_000], np.full(10_000, -0.5),
                                             np.full(10_000, 0.5), 0.8)
        held_out = y[10_000:]
        coverage = np.mean((held_out >= -0.5 - margin) & (held_out <= 0.5 + margin))
        assert margin > 0
        assert abs(coverage - 0.8) < 0.02

    def test_narrows_a_too_wide_band(self):
        y = np.random.default_rng(1).normal(0, 1, 10_000)
        assert model_tail.conformal_margin(y, np.full(10_000, -5.0), np.full(10_000, 5.0), 0.8) < 0
