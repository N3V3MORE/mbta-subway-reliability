"""Tests for feature extraction.

The most important test in this file is the leakage guard: no target-derived
column may appear in the feature matrix, and every lag feature must come from an
*earlier* stop of the *same* trip on the *same* service date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import clean, config, features

from .conftest import SERVICE_DATE, make_raw_trip


class TestLeakageGuards:
    def test_no_target_column_is_also_a_feature(self):
        declared = set(features.all_feature_columns())
        assert declared.isdisjoint(set(features.TARGET_COLUMNS))

    def test_monotone_time_index_is_not_a_model_feature(self):
        """Under a temporal split its train and test ranges never overlap."""
        assert "days_since_window_start" not in features.all_feature_columns()

    def test_time_index_is_still_built_for_the_diagnostic(self, clean_frame):
        built = features.build(max_rows=None, clean_frame=clean_frame,
                               refresh=True, no_cache=True)
        assert "days_since_window_start" in built.columns

    def test_origin_rows_are_lag_sources_but_not_targets(self, clean_frame):
        built = features.build(max_rows=None, clean_frame=clean_frame,
                               refresh=True, no_cache=True)
        assert len(built) == clean_frame["is_origin"].eq(False).sum()
        assert built["prev_delay_1"].notna().all()

    def test_fraction_through_trip_is_a_fraction(self, clean_frame):
        built = features.build(max_rows=None, clean_frame=clean_frame,
                               refresh=True, no_cache=True)
        assert built["fraction_through_trip"].between(0, 1).all()
        assert built["fraction_through_trip"].max() == 1.0

    def test_key_columns_exclude_identifiers_but_features_use_stop_sequence(self):
        # stop_sequence is a legitimate feature (position along the trip) even
        # though it also appears in the key set, but the raw identifiers must not
        # be features.
        declared = set(features.all_feature_columns())
        assert "trip_id" not in declared
        assert "vehicle_id" not in declared
        assert "service_date" not in declared


class TestPropagationFeatures:
    def _with_delay(self, trip_id: str, **kwargs) -> pd.DataFrame:
        """Raw synthetic rows plus the delay column the feature builder expects."""
        from mbta_ds import clean

        frame = pd.DataFrame(make_raw_trip(trip_id, SERVICE_DATE, **kwargs))
        frame["delay_seconds"] = clean.compute_delay(frame)
        return frame

    def test_previous_delay_comes_from_the_previous_stop(self):
        frame = self._with_delay("2001", delay_schedule=(10, 20, 30, 40, 50))
        frame = features.add_propagation_features(frame)
        frame = frame.sort_values("stop_sequence")

        assert np.isnan(frame["prev_delay_1"].iloc[0])  # first stop has no previous
        assert list(frame["prev_delay_1"].iloc[1:]) == [10, 20, 30, 40]
        assert list(frame["prev_delay_2"].iloc[2:]) == [10, 20, 30]

    def test_delay_trend_is_the_difference_of_the_two_lags(self):
        frame = self._with_delay("2002", delay_schedule=(10, 40, 70, 100, 130))
        frame = features.add_propagation_features(frame)
        frame = frame.sort_values("stop_sequence")
        # Every step adds 30s of delay, so the trend is a constant +30.
        assert list(frame["delay_trend"].dropna()) == [30, 30, 30]

    def test_lags_never_cross_service_dates(self):
        """Trip ids repeat across days, so a lag must not bleed backwards in time.

        The first stop of the *second* day must have no previous delay even though
        the same trip_id ran on the first day.
        """
        from mbta_ds import clean

        raw = pd.DataFrame(
            make_raw_trip("2003", SERVICE_DATE, delay_schedule=(500,) * 5)
            + make_raw_trip("2003", SERVICE_DATE + 1, delay_schedule=(1,) * 5)
        )
        raw["delay_seconds"] = clean.compute_delay(raw)
        frame = features.add_propagation_features(raw)

        second_day = frame[frame["service_date"] == SERVICE_DATE + 1]
        second_day = second_day.sort_values("stop_sequence")
        assert np.isnan(second_day["prev_delay_1"].iloc[0])
        # And the values that do exist must be from the same day only.
        assert set(second_day["prev_delay_1"].dropna()) == {1.0}

    def test_lags_follow_scheduled_order_not_stop_sequence(self):
        """A mislabelled terminal (stop_sequence 1 at the end) must not leak."""
        from mbta_ds import clean

        rows = make_raw_trip("2006", SERVICE_DATE, delay_schedule=(10, 20, 30, 40, 999))
        rows[-1]["stop_sequence"] = 1
        frame = pd.DataFrame(rows)
        frame["delay_seconds"] = clean.compute_delay(frame)
        frame = features.add_propagation_features(frame)
        # The terminal's 999 s delay is never anyone's "previous stop".
        assert 999 not in set(frame["prev_delay_1"].dropna())
        assert frame.loc[frame["delay_seconds"] == 999, "prev_delay_1"].item() == 40

    def test_lags_never_cross_trips(self):
        from mbta_ds import clean

        raw = pd.DataFrame(
            make_raw_trip("2004", SERVICE_DATE, delay_schedule=(100,) * 5)
            + make_raw_trip("2005", SERVICE_DATE, delay_schedule=(7,) * 5)
        )
        raw["delay_seconds"] = clean.compute_delay(raw)
        frame = features.add_propagation_features(raw)
        other = frame[frame["trip_id"] == "2005"].sort_values("stop_sequence")
        assert np.isnan(other["prev_delay_1"].iloc[0])


def _arrivals_at(*local_times: str, route: str = "Red") -> pd.DataFrame:
    """Minimal rows with a scheduled arrival at each given local time."""
    stamps = pd.to_datetime(list(local_times)).tz_localize(config.SERVICE_TZ)
    return pd.DataFrame({
        "service_date": SERVICE_DATE,
        "route_id": route,
        "scheduled_epoch": (stamps.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC"))
        // pd.Timedelta(seconds=1),
    })


def _alert(start: str, end: str | None, *, route: str = "Red", alert_id: int = 1,
           effect: str = "DELAY", severity: int = 5) -> dict:
    return {
        "id": alert_id,
        "informed_entity.route_id": route,
        "active_period.start_datetime": pd.Timestamp(start),
        "active_period.end_datetime": pd.Timestamp(end) if end else pd.NaT,
        "severity": severity,
        "effect": effect,
    }


class TestAlertFeatures:
    def test_alert_is_not_visible_before_it_starts(self):
        """An alert raised at 08:50 must not inform an 08:10 arrival."""
        arrivals = _arrivals_at("2026-06-26 08:10", "2026-06-26 09:10")
        alerts = pd.DataFrame([_alert("2026-06-26 08:50", "2026-06-26 11:00")])
        out = features.attach_alerts(arrivals, alerts=alerts)
        assert out["route_alerts_active"].tolist() == [0, 1]
        assert out["route_alert_severity_max"].tolist() == [0, 5]

    def test_alert_starting_on_the_hour_counts_for_that_hour(self):
        arrivals = _arrivals_at("2026-06-26 09:10")
        alerts = pd.DataFrame([_alert("2026-06-26 09:00", "2026-06-26 10:00")])
        out = features.attach_alerts(arrivals, alerts=alerts)
        assert out["route_alerts_active"].tolist() == [1]

    def test_accessibility_alerts_are_ignored(self):
        arrivals = _arrivals_at("2026-06-26 09:10")
        alerts = pd.DataFrame([
            _alert("2026-06-26 07:00", "2026-06-26 12:00", effect="ACCESSIBILITY_ISSUE"),
        ])
        out = features.attach_alerts(arrivals, alerts=alerts)
        assert out["route_alerts_active"].tolist() == [0]

    def test_other_routes_and_duplicate_rows_do_not_count(self):
        arrivals = _arrivals_at("2026-06-26 09:10")
        alerts = pd.DataFrame([
            _alert("2026-06-26 07:00", "2026-06-26 12:00", alert_id=1),
            _alert("2026-06-26 07:00", "2026-06-26 12:00", alert_id=1),  # exploded twin
            _alert("2026-06-26 07:00", "2026-06-26 12:00", alert_id=2, route="Orange"),
        ])
        out = features.attach_alerts(arrivals, alerts=alerts)
        assert out["route_alerts_active"].tolist() == [1]

    def test_open_ended_alert_is_capped(self):
        arrivals = _arrivals_at("2026-06-26 09:10")
        long_ago = pd.DataFrame([_alert("2026-06-01 07:00", None)])
        out = features.attach_alerts(arrivals, alerts=long_ago, max_span_days=7)
        assert out["route_alerts_active"].tolist() == [0]

    def test_backdated_alert_counts_from_its_creation(self):
        """An alert created at 09:40 but stamped as starting 08:00 was unknown at 09:10."""
        arrivals = _arrivals_at("2026-06-26 09:10", "2026-06-26 10:10")
        alert = _alert("2026-06-26 08:00", "2026-06-26 12:00")
        alert["created_datetime"] = pd.Timestamp("2026-06-26 09:40")
        out = features.attach_alerts(arrivals, alerts=pd.DataFrame([alert]))
        assert out["route_alerts_active"].tolist() == [0, 1]

    def test_matched_on_prediction_moment_not_scheduled_hour(self):
        """Predicted at 08:40 for a 09:10 arrival: a 09:00 alert did not exist yet."""
        arrivals = _arrivals_at("2026-06-26 09:10")
        at = pd.Timestamp("2026-06-26 08:40", tz=config.SERVICE_TZ).value // 10**9
        arrivals["known_at"] = float(at)
        alerts = pd.DataFrame([_alert("2026-06-26 09:00", "2026-06-26 12:00")])
        out = features.attach_alerts(arrivals, alerts=alerts)
        assert out["route_alerts_active"].tolist() == [0]

    def test_row_count_is_preserved(self):
        arrivals = _arrivals_at("2026-06-26 08:10", "2026-06-26 09:10", "2026-06-26 09:40")
        alerts = pd.DataFrame([
            _alert("2026-06-26 08:30", "2026-06-26 11:00", alert_id=1),
            _alert("2026-06-26 08:45", "2026-06-26 11:00", alert_id=2, severity=7),
        ])
        out = features.attach_alerts(arrivals, alerts=alerts)
        assert len(out) == 3
        assert out["route_alerts_active"].tolist() == [0, 2, 2]
        assert out["route_alert_severity_max"].tolist() == [0, 7, 7]


def _events(rows: list[tuple]) -> pd.DataFrame:
    """(parent_station, direction, stop_ts, delay, known_at) rows on the Red line."""
    return pd.DataFrame(rows, columns=["parent_station", "direction_id", "stop_timestamp",
                                       "delay_seconds", "known_at"]).assign(trunk_route_id="Red")


class TestNetworkFeatures:
    def test_train_ahead_is_the_last_one_before_prediction_time(self):
        frame = _events([
            ("place-a", 0, 1_000.0, 60.0, np.nan),     # earlier train at A
            ("place-a", 0, 1_500.0, 400.0, np.nan),    # the train just ahead
            ("place-a", 0, 2_100.0, 999.0, np.nan),    # arrives AFTER we predict
            ("place-a", 1, 1_900.0, 777.0, np.nan),    # other direction
            ("place-a", 0, 2_500.0, 90.0, 2_000.0),    # our train, predicted at t=2000
        ])
        row = features.attach_network(frame).iloc[-1]
        assert row["leader_delay"] == 400.0
        assert row["leader_age_seconds"] == 500.0

    def test_other_lines_do_not_count(self):
        frame = _events([("place-a", 0, 1_500.0, 400.0, np.nan),
                         ("place-a", 0, 2_500.0, 90.0, 2_000.0)])
        frame.loc[0, "trunk_route_id"] = "Orange"
        assert np.isnan(features.attach_network(frame).iloc[-1]["leader_delay"])

    def test_line_lateness_uses_the_last_15_minutes_only(self):
        frame = _events([
            ("place-x", 0, 100.0, 900.0, np.nan),      # late, but 30 min before
            ("place-y", 0, 1_300.0, 600.0, np.nan),    # late, inside the window
            ("place-z", 0, 1_500.0, 30.0, np.nan),     # on time, inside the window
            ("place-a", 0, 2_500.0, 90.0, 2_000.0),
        ])
        row = features.attach_network(frame).iloc[-1]
        assert row["line_arrivals_15m"] == 2
        assert row["line_late_share_15m"] == 0.5


def _runs(rows: list[tuple]) -> pd.DataFrame:
    """(trip, vehicle, sched, stop_ts, delay, known_at) rows on one service date."""
    return pd.DataFrame(rows, columns=["trip_id", "vehicle_id", "scheduled_epoch", "stop_timestamp",
                                       "delay_seconds", "known_at"]).assign(service_date=SERVICE_DATE, run=0)


class TestVehicleHistory:
    def test_previous_trip_of_the_same_train(self):
        frame = _runs([
            ("A", "v1", 1_000, 1_100, 100.0, np.nan),
            ("A", "v1", 1_500, 1_700, 200.0, 1_150.0),   # A ends at 1700, 200 s late
            ("B", "v1", 2_000, 2_060, 60.0, np.nan),
            ("B", "v1", 2_300, 2_400, 100.0, 2_100.0),
            ("C", "v2", 2_000, 2_010, 10.0, 2_050.0),    # another train: no history
        ])
        out = features.attach_vehicle_history(frame).set_index(["trip_id", "stop_timestamp"])
        assert out.loc[("B", 2_400), "vehicle_prev_trip_delay"] == 200.0
        assert out.loc[("B", 2_400), "vehicle_layover_seconds"] == 2_000 - 1_700
        assert np.isnan(out.loc[("A", 1_700), "vehicle_prev_trip_delay"])
        assert np.isnan(out.loc[("C", 2_010), "vehicle_prev_trip_delay"])

    def test_previous_trip_is_hidden_until_it_has_ended(self):
        """Terminal noise can make trip A 'end' after trip B starts: not yet known."""
        frame = _runs([
            ("A", "v1", 1_000, 1_000, 0.0, np.nan),
            ("A", "v1", 1_500, 2_100, 50.0, 1_050.0),    # A's last record at 2100
            ("B", "v1", 2_000, 2_000, 0.0, np.nan),
            ("B", "v1", 2_300, 2_300, 0.0, 2_050.0),     # predicted at 2050 < 2100
            ("B", "v1", 2_600, 2_600, 0.0, 2_350.0),     # predicted at 2350 > 2100
        ])
        b = features.attach_vehicle_history(frame).query("trip_id == 'B'")
        assert np.isnan(b["vehicle_prev_trip_delay"].iloc[1])
        assert b["vehicle_prev_trip_delay"].iloc[2] == 50.0

    def test_impossible_overlap_means_not_the_same_train(self):
        frame = _runs([
            ("A", "v1", 1_000, 1_000, 0.0, np.nan),
            ("A", "v1", 9_000, 9_000, 0.0, 1_100.0),     # "ends" hours after B starts
            ("B", "v1", 2_000, 2_000, 0.0, np.nan),
            ("B", "v1", 9_500, 9_500, 0.0, 9_400.0),
        ])
        b = features.attach_vehicle_history(frame).query("trip_id == 'B'")
        assert b["vehicle_prev_trip_delay"].isna().all()


class TestHorizon:
    def test_lags_reach_back_k_stops(self):
        from mbta_ds import clean

        frame = pd.DataFrame(make_raw_trip("2010", SERVICE_DATE, delay_schedule=(10, 20, 30, 40, 50)))
        frame["delay_seconds"] = clean.compute_delay(frame)
        out = features.add_propagation_features(frame, horizon=3).sort_values("stop_sequence")
        assert out["prev_delay_1"].tolist()[3:] == [10, 20]
        assert out["prev_delay_1"].iloc[:3].isna().all()
        # Known at departure from the stop 3 back = move time of the stop after it.
        assert out["known_at"].iloc[3] == frame.sort_values("stop_sequence")["move_timestamp"].iloc[1]
        assert out["scheduled_seconds_ahead"].iloc[3] == 3 * 120


class TestDemandFeatures:
    def test_lag_uses_calendar_days_across_a_gap(self, monkeypatch):
        """Lags are over calendar days, not over whichever rows happen to exist.

        A row-wise shift has no row for a day the station reported nothing, so
        trains that day got no demand signal at all even with six days of history.
        """
        daily = pd.DataFrame({
            "service_date": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-04"]).date,
            "stop_id": "place-a",
            "station_name": "A",
            "route_or_line": "Red",
            "daily_entries": [100.0, 200.0, 400.0],
        })
        monkeypatch.setattr(features.collect_ridership, "load_daily", lambda: daily)
        frame = pd.DataFrame({
            "service_date": [20260603, 20260604],
            "parent_station": "place-a",
        })
        out = features.attach_demand(frame).set_index("service_date")
        # 06-03 has no ridership row of its own, but its history does: 06-01..02.
        assert out.loc[20260603, "station_entries_lag7"] == 150.0
        # 06-04 sees the same history (06-03 is empty) and never its own value.
        assert out.loc[20260604, "station_entries_lag7"] == 150.0

    def test_first_day_has_no_history(self, monkeypatch):
        daily = pd.DataFrame({
            "service_date": pd.to_datetime(["2026-06-01", "2026-06-02"]).date,
            "stop_id": "place-a", "station_name": "A", "route_or_line": "Red",
            "daily_entries": [100.0, 200.0],
        })
        monkeypatch.setattr(features.collect_ridership, "load_daily", lambda: daily)
        frame = pd.DataFrame({"service_date": [20260601], "parent_station": "place-a"})
        out = features.attach_demand(frame)
        assert out["demand_missing"].tolist() == [1]


class TestWeatherJoin:
    def test_repeated_local_hour_does_not_duplicate_rows(self, monkeypatch):
        """The DST fall-back repeats 01:00 in naive local time."""
        weather = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-11-01 01:00", "2026-11-01 01:00"]),
            "temperature_c": [5.0, 4.0], "precip_mm": [0.0, 0.0],
            "snowfall_cm": [0.0, 0.0], "wind_kph": [10.0, 10.0],
            "humidity_pct": [80.0, 80.0],
        })
        monkeypatch.setattr(features.collect_weather, "load_weather", lambda: weather)
        stamp = pd.Timestamp("2026-11-01 01:20").tz_localize(config.SERVICE_TZ, ambiguous=True)
        frame = pd.DataFrame({"scheduled_epoch": [stamp.value // 10**9]})
        out = features.attach_weather(frame)
        assert len(out) == 1


class TestStratifiedCap:
    def test_no_cap_returns_everything(self):
        frame = pd.DataFrame({"service_date": [1, 1, 2, 2], "v": range(4)})
        assert len(features.stratified_sample(frame, None, 1)) == 4
        assert len(features.stratified_sample(frame, 10, 1)) == 4

    def test_cap_keeps_every_date_represented(self):
        frame = pd.DataFrame({
            "service_date": np.repeat([10, 11, 12, 13, 14], 100),
            "v": np.arange(500),
        })
        capped = features.stratified_sample(frame, 100, seed=506)
        assert len(capped) <= 110
        assert set(capped["service_date"]) == {10, 11, 12, 13, 14}

    def test_cap_is_deterministic(self):
        frame = pd.DataFrame({
            "service_date": np.repeat([1, 2, 3], 50),
            "v": np.arange(150),
        })
        first = features.stratified_sample(frame, 30, seed=506)
        second = features.stratified_sample(frame, 30, seed=506)
        assert first["v"].tolist() == second["v"].tolist()


class TestFeatureGroups:
    def test_all_feature_columns_flat_matches_groups(self):
        flat = features.all_feature_columns()
        expected = sum(len(v) for v in features.FEATURE_GROUPS.values())
        expected += len(features.CATEGORICAL_FEATURES)
        assert len(flat) == expected

    def test_requesting_a_subset_of_groups_works(self):
        columns = features.all_feature_columns(groups=("schedule",))
        assert columns == list(features.FEATURE_GROUPS["schedule"])

    def test_categorical_group_adds_categoricals(self):
        columns = features.all_feature_columns(groups=("categorical",))
        assert columns == list(features.CATEGORICAL_FEATURES)

    def test_ablation_order_covers_every_regression_group_at_the_end(self):
        from mbta_ds.model_delay import ABLATION_ORDER

        final = set(ABLATION_ORDER[-1][1])
        regression = set(features.FEATURE_GROUPS) - set(features.LATENESS_ONLY_GROUPS)
        assert final == regression | {"categorical"}

    def test_previous_lateness_is_the_last_stop_not_this_one(self, clean_frame):
        out = features.add_propagation_features(clean_frame)
        for _, trip in out.groupby(clean.RUN_KEY, sort=False):
            assert trip["prev_lateness_1"].iloc[1:].tolist() == trip["lateness_seconds"].iloc[:-1].tolist()
            assert np.isnan(trip["prev_lateness_1"].iloc[0])

    def test_ablation_order_is_cumulative(self):
        from mbta_ds.model_delay import ABLATION_ORDER

        for earlier, later in zip(ABLATION_ORDER, ABLATION_ORDER[1:]):
            assert set(earlier[1]) <= set(later[1])


class TestBuildGuard:
    def test_missing_declared_feature_raises(self, monkeypatch, clean_frame):
        """A declared feature that is never produced must fail loudly."""
        monkeypatch.setitem(
            features.FEATURE_GROUPS, "schedule",
            features.FEATURE_GROUPS["schedule"] + ("definitely_not_a_column",),
        )
        with pytest.raises((KeyError, ValueError)):
            features.build(max_rows=None, clean_frame=clean_frame,
                           refresh=True, no_cache=True)

