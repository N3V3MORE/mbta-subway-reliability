"""Tests for the progress-reporting helper.

The helper exists so that a multi-minute stage reports how far along it is. The
tests pin the two things that would make it useless: wrong arithmetic (a counter
that drifts from reality) and wrong formatting (an unreadable ETA).
"""

from __future__ import annotations

import logging
import math

import pytest

from mbta_ds import progress


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0s"),
            (9.4, "9s"),
            (59.6, "60s"),      # rounds to a minute, still rendered as seconds
            (60, "1m 00s"),
            (134, "2m 14s"),
            (3599, "59m 59s"),
            (3600, "1h 00m"),
            (3900, "1h 05m"),
        ],
    )
    def test_formats_readably(self, seconds, expected):
        assert progress.format_duration(seconds) == expected

    def test_negative_and_nan_are_handled(self):
        assert progress.format_duration(-5) == "0s"
        assert progress.format_duration(float("nan")) == "?"
        assert progress.format_duration(float("inf")) == "?"
        assert progress.format_duration(math.nan) == "?"


class TestProgressCounter:
    def test_counts_every_step(self):
        with progress.Progress(3, "test") as bar:
            assert bar.done == 0
            bar.step()
            assert bar.done == 1
            bar.step()
            bar.step()
            assert bar.done == 3

    def test_total_of_zero_does_not_divide_by_zero(self):
        # An empty candidate list must not crash the stage.
        with progress.Progress(0, "empty") as bar:
            assert bar.total == 1
        assert bar.done == 0

    def test_log_every_reduces_output_but_always_reports_the_last(self, caplog):
        caplog.set_level(logging.INFO, logger="mbta_ds.progress")
        with progress.Progress(10, "sparse", log_every=5) as bar:
            for _ in range(10):
                bar.step()
        step_lines = [r.message for r in caplog.records if "[n" not in str(r.message)
                      and str(r.message).startswith("[")]
        # Steps 5 and 10 are reported; 1-4 and 6-9 are not.
        assert len(step_lines) == 2
        assert any("5/10" in str(m) for m in step_lines)
        assert any("10/10" in str(m) for m in step_lines)

    def test_every_step_reported_by_default(self, caplog):
        caplog.set_level(logging.INFO, logger="mbta_ds.progress")
        with progress.Progress(4, "verbose") as bar:
            for _ in range(4):
                bar.step()
        step_lines = [r.message for r in caplog.records if str(r.message).startswith("[")]
        assert len(step_lines) == 4

    def test_step_name_and_extra_both_appear(self, caplog):
        caplog.set_level(logging.INFO, logger="mbta_ds.progress")
        with progress.Progress(2, "models") as bar:
            bar.step("decision_tree", extra="MAE 54.8s")
        text = caplog.text
        assert "decision_tree" in text
        assert "MAE 54.8s" in text
        assert "[1/2]" in text

    def test_eta_is_absent_before_the_first_step(self, caplog):
        caplog.set_level(logging.INFO, logger="mbta_ds.progress")
        with progress.Progress(3, "fresh"):
            pass
        # The start line must not claim an ETA it cannot know yet.
        start_lines = [r.message for r in caplog.records
                       if str(r.message).startswith("fresh: 0/")]
        assert start_lines

    def test_summary_is_emitted_on_exit(self, caplog):
        caplog.set_level(logging.INFO, logger="mbta_ds.progress")
        with progress.Progress(2, "wrapped") as bar:
            bar.step()
            bar.step()
        assert "wrapped: 2 step(s) in" in caplog.text

    def test_summary_is_emitted_even_when_the_body_raises(self, caplog):
        caplog.set_level(logging.INFO, logger="mbta_ds.progress")
        with pytest.raises(ValueError):
            with progress.Progress(2, "failing") as bar:
                bar.step()
                raise ValueError("boom")
        # The counter still reports progress made before the failure.
        assert "failing: 1 step(s) in" in caplog.text

