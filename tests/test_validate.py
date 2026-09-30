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

    def test_a_day_empty_at_the_source_is_reported_not_failed(self, clean_frame):
        from datetime import timedelta

        day = clean_frame["service_date_parsed"].iloc[0]
        later = clean_frame.assign(service_date_parsed=day + timedelta(days=2))
        gap = pd.concat([clean_frame, later], ignore_index=True)     # day+1 missing
        lost = "service dates lost by the pipeline"
        assert lost in _failed(validate.check_clean(gap))
        assert lost not in _failed(validate.check_clean(gap, frozenset({day + timedelta(days=1)})))

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


class TestWindowChecks:
    def test_day_lost_at_the_window_edge_fails(self, clean_frame):
        from datetime import timedelta

        day = clean_frame["service_date_parsed"].iloc[0]
        window = (day, day + timedelta(days=1))          # the table only has `day`
        failed = _failed(validate.check_clean(clean_frame, window=window))
        assert "service dates lost by the pipeline" in failed

    def test_table_from_another_window_fails(self, clean_frame):
        from datetime import timedelta

        day = clean_frame["service_date_parsed"].iloc[0]
        window = (day + timedelta(days=30), day + timedelta(days=31))
        assert "service dates outside the analysis window" in _failed(
            validate.check_clean(clean_frame, window=window))

    def test_ridership_must_cover_the_window_dates_not_just_their_count(self):
        from datetime import date

        raw = pd.DataFrame({
            "service_date": [date(2026, 1, 1)], "stop_id": ["a"], "route_or_line": ["Red"],
            "time_period": ["08:00:00"], "period_minute": [0], "gated_entries": [10.0],
        })
        window = (date(2026, 6, 1), date(2026, 6, 1))
        assert "window dates missing from ridership" in _failed(
            validate.check_ridership(raw, window=window))

    def test_weather_must_reach_the_morning_after_the_window(self):
        from datetime import date

        window = (date(2026, 6, 1), date(2026, 6, 1))
        assert _failed(validate.check_weather(_weather(48), window)) == set()
        assert "window hours without weather" in _failed(validate.check_weather(_weather(24), window))

    def test_alerts_archive_must_reach_the_end_of_the_window(self):
        from datetime import date

        window = (date(2026, 6, 1), date(2026, 6, 30))
        stale = pd.DataFrame({"last_modified_datetime": pd.to_datetime(["2026-06-20 09:00"])})
        fresh = pd.DataFrame({"last_modified_datetime": pd.to_datetime(["2026-06-30 23:50"])})
        assert "window days after the alerts archive ends" in _failed(validate.check_alerts(stale, window))
        assert _failed(validate.check_alerts(fresh, window)) == set()
