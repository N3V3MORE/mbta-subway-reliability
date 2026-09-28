"""Tests for the data-validation checks: clean data passes, broken data fails."""

from __future__ import annotations

import pandas as pd

from mbta_ds import validate


def _failed(results: list[dict]) -> set[str]:
    return {r["check"] for r in results if r["hard"] and not r["ok"]}


def _weather(hours: int = 48) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-06-01", periods=hours, freq="h"),
        "temperature_c": 20.0, "precip_mm": 0.0, "snowfall_cm": 0.0,
        "wind_kph": 10.0, "humidity_pct": 60.0,
    })


class TestCleanChecks:
    def test_synthetic_clean_table_passes(self, clean_frame):
        assert _failed(validate.check_clean(clean_frame)) == set()

    def test_duplicate_rows_fail(self, clean_frame):
        doubled = pd.concat([clean_frame, clean_frame.iloc[[3]]], ignore_index=True)
        failed = _failed(validate.check_clean(doubled))
        assert "duplicate (service_date, trip_id, stop_id)" in failed

    def test_unresolved_station_ids_fail(self, clean_frame):
        broken = clean_frame.assign(station_name="place-unknown")
        assert "stations left as raw ids (name lookup failed)" in _failed(
            validate.check_clean(broken))


class TestWeatherChecks:
    def test_complete_series_passes(self):
        assert _failed(validate.check_weather(_weather())) == set()

    def test_gap_and_bad_humidity_fail(self):
        broken = _weather().drop(index=[5]).reset_index(drop=True)
        broken.loc[0, "humidity_pct"] = 140.0
        failed = _failed(validate.check_weather(broken))
        assert "gaps or repeats in the hourly series" in failed
        assert "humidity outside 0..100 %" in failed


class TestRidershipChecks:
    def test_negative_entries_fail(self):
        raw = pd.DataFrame({
            "service_date": ["2026-06-01", "2026-06-01"], "stop_id": ["a", "b"],
            "route_or_line": ["Red", "Red"], "time_period": ["08:00:00", "08:00:00"],
            "period_minute": [0, 0], "gated_entries": [10.0, -1.0],
        })
        assert "negative gated entries" in _failed(validate.check_ridership(raw, n_days=1))
