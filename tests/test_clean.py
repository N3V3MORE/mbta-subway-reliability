"""Tests for the cleaning stage.

These cover the rules that were derived from inspecting real data, because they
are the ones most likely to regress silently:

* the local-midnight schedule anchor (including DST and post-midnight trips),
* exclusion of trip namespaces with no planned-schedule match,
* de-duplication on the (service_date, trip_id, stop_id) key,
* station-name resolution from a parent-station id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import clean, config

from .conftest import SERVICE_DATE, _epoch, make_raw_trip


class TestServiceMidnightAnchor:
    def test_midnight_is_local_not_utc(self):
        # 2026-06-26 is EDT (UTC-4), so local midnight is 04:00 UTC.
        midnight = pd.Timestamp("2026-06-26", tz=config.SERVICE_TZ)
        expected = float(midnight.value // 10**9)
        actual = clean.service_midnight_epoch(pd.Series([20260626])).iloc[0]
        assert actual == expected

    def test_spring_forward_anchor_is_noon_minus_12h(self):
        """US DST starts 2026-03-08. GTFS counts from noon - 12 h = 23:00 EST the
        day before, an hour before local midnight; a midnight anchor put every
        delay that day off by -3,600 s in the real archive."""
        midnight = pd.Timestamp("2026-03-08", tz=config.SERVICE_TZ)
        actual = clean.service_midnight_epoch(pd.Series([20260308])).iloc[0]
        assert actual == float(midnight.value // 10**9) - 3600
        # 08:00:00 in the timetable is 08:00 on the (EDT) wall clock.
        eight = pd.Timestamp("2026-03-08 08:00", tz=config.SERVICE_TZ)
        assert actual + 8 * 3600 == eight.value // 10**9

    def test_fall_back_anchor_is_noon_minus_12h(self):
        """DST ends 2025-11-02: noon - 12 h is 01:00 EDT, an hour after local midnight."""
        midnight = pd.Timestamp("2025-11-02", tz=config.SERVICE_TZ)
        actual = clean.service_midnight_epoch(pd.Series([20251102])).iloc[0]
        assert actual == float(midnight.value // 10**9) + 3600
        eight = pd.Timestamp("2025-11-02 08:00", tz=config.SERVICE_TZ)
        assert actual + 8 * 3600 == eight.value // 10**9

    def test_vectorised_over_multiple_dates(self):
        result = clean.service_midnight_epoch(pd.Series([20260626, 20260627]))
        assert result.iloc[1] - result.iloc[0] == config.SECONDS_PER_DAY


class TestComputeDelay:
    def test_on_time_train_has_zero_delay(self):
        frame = pd.DataFrame(make_raw_trip("9001", SERVICE_DATE,
                                           delay_schedule=(0,) * 5))
        assert (clean.compute_delay(frame) == 0).all()

    def test_late_and_early_are_signed(self):
        frame = pd.DataFrame(make_raw_trip("9002", SERVICE_DATE,
                                           delay_schedule=(120, -60, 0, 0, 0)))
        delay = clean.compute_delay(frame)
        assert delay.iloc[0] == 120
        assert delay.iloc[1] == -60

    def test_post_midnight_trip_needs_no_special_case(self):
        """A 27-hour GTFS time (>86400s) must resolve to the following morning.

        This is the quirk the whole cleaning stage is built around: the service
        date stays on the prior calendar day while the GTFS time rolls past 24h.
        """
        rows = make_raw_trip("9003", SERVICE_DATE, start_seconds=97_200,
                             stops=3, delay_schedule=(300,) * 3)
        frame = pd.DataFrame(rows)
        delay = clean.compute_delay(frame)
        assert (delay == 300).all()

        # And the resolved local time really is the next calendar day.
        local = pd.to_datetime(frame["stop_timestamp"], unit="s", utc=True) \
            .dt.tz_convert(config.SERVICE_TZ)
        assert local.iloc[0].date() == pd.Timestamp("2026-06-27").date()


class TestBuildRules:
    def test_excludes_added_and_nonrev_trips(self, clean_frame):
        assert not clean_frame["trip_id"].str.startswith(("ADDED-", "NONREV-")).any()
        assert set(clean_frame["trip_id"].unique()) == {"1001", "1002"}

    def test_drops_rows_without_an_observation(self, clean_frame):
        assert clean_frame["stop_timestamp"].notna().all()
        assert "stop-null" not in set(clean_frame["stop_id"])

    def test_deduplicates_on_service_date_trip_stop(self, clean_frame):
        key = ["service_date", "trip_id", "stop_id"]
        assert not clean_frame.duplicated(subset=key).any()

    def test_resolves_station_names_from_parent_station(self, clean_frame):
        # parent_station holds an id like "place-0"; the name must come from the
        # static stops lookup rather than being used directly.
        assert clean_frame["station_name"].iloc[0] == "Station 0"
        assert clean_frame["station_name"].str.startswith("Station").all()

    def test_falls_back_to_platform_name_on_a_filtered_index(self):
        """The fallback must stay row-aligned when the index is not 0..n-1.

        By this point the frame has been filtered and sorted, so its index is a
        shuffled subset; positional and label-based lookups must not be mixed.
        """
        frame = pd.DataFrame({
            "parent_station": ["place-a", "place-unknown", "place-b"],
            "stop_id": ["70001", "70099", "70002"],
        }, index=[17, 3, 250])
        lookup = pd.DataFrame({
            "stop_id": ["place-a", "place-b", "70099"],
            "stop_name": ["Alpha", "Bravo", "Platform 99"],
        })
        named = clean._attach_stop_names(frame, lookup)
        assert named["station_name"].tolist() == ["Alpha", "Platform 99", "Bravo"]

    def test_unresolvable_ids_keep_the_raw_id(self):
        frame = pd.DataFrame({"parent_station": ["place-zzz"], "stop_id": ["1"]})
        lookup = pd.DataFrame({"stop_id": ["place-a"], "stop_name": ["Alpha"]})
        assert clean._attach_stop_names(frame, lookup)["station_name"].tolist() == ["place-zzz"]

    def test_latest_static_version_names_win(self, monkeypatch):
        """Stations get renamed; the 2019 name must not beat the current one."""
        stops = pd.DataFrame({
            "stop_id": ["place-spmnl", "place-spmnl", "place-a"],
            "stop_name": ["Science Park/West End", "Science Park", "Alpha"],
            "static_version_key": [1_700_000_000, 1_560_000_000, 1_560_000_000],
        })
        monkeypatch.setattr(clean.collect_lamp, "load_static", lambda *a, **k: stops)
        lookup = clean._stop_name_lookup().set_index("stop_id")["stop_name"]
        assert lookup["place-spmnl"] == "Science Park/West End"
        assert lookup["place-a"] == "Alpha"

    def test_flags_but_keeps_delay_outliers(self):
        rows = make_raw_trip("9004", SERVICE_DATE, stops=4, delay_schedule=(9000,) * 4)
        frame = clean.build(raw_frame=pd.DataFrame(rows),
                            stop_lookup=pd.DataFrame({"stop_id": [], "stop_name": []}))
        # The row survives, but is flagged rather than silently removed.
        assert len(frame) == 4
        assert frame["delay_outlier"].all()

    def test_labels_late_beyond_threshold(self):
        rows = make_raw_trip("9005", SERVICE_DATE, delay_schedule=(0, 301, 0, 0, 0))
        frame = clean.build(raw_frame=pd.DataFrame(rows),
                            stop_lookup=pd.DataFrame({"stop_id": [], "stop_name": []}))
        assert frame["late"].sum() == 1
        assert frame.loc[frame["late"], "delay_seconds"].iloc[0] == 301

    def test_does_not_write_cache_for_injected_frames(self, raw_frame, stop_lookup,
                                                     tmp_path, monkeypatch):
        monkeypatch.setattr(clean, "OUT_PATH", tmp_path / "never_written.parquet")
        clean.build(raw_frame=raw_frame, stop_lookup=stop_lookup)
        assert not (tmp_path / "never_written.parquet").exists()

    def test_derived_calendar_columns(self, clean_frame):
        # 2026-06-26 is a Friday.
        assert (clean_frame["day_of_week"] == 4).all()
        assert not clean_frame["is_weekend"].any()
        assert clean_frame["scheduled_hour"].between(0, 23).all()

    def test_scheduled_seconds_of_day_folds_post_midnight(self):
        rows = make_raw_trip("9006", SERVICE_DATE, start_seconds=90_000)
        frame = clean.build(raw_frame=pd.DataFrame(rows),
                            stop_lookup=pd.DataFrame({"stop_id": [], "stop_name": []}))
        # 90000 s is 25:00, which must fold to 01:00.
        assert (frame["scheduled_hour"] == 1).all()


def _build(rows: list[dict]) -> pd.DataFrame:
    return clean.build(raw_frame=pd.DataFrame(rows),
                       stop_lookup=pd.DataFrame({"stop_id": [], "stop_name": []}))


class TestTripOrder:
    def test_final_stop_mislabelled_as_first_is_reordered(self):
        """LAMP sometimes gives the *last* stop stop_sequence == 1 (Heath Street)."""
        rows = make_raw_trip("9100", SERVICE_DATE, delay_schedule=(0, 60, 120, 180, 240))
        rows[-1]["stop_sequence"] = 1   # the terminal arrival, mislabelled
        frame = _build(rows).sort_values("stop_index")
        assert frame["stop_id"].tolist() == [f"stop-{i}" for i in range(5)]
        assert frame["is_origin"].tolist() == [True, False, False, False, False]
        assert not frame["time_inconsistent"].any()

    def test_same_visit_under_two_platform_ids_is_kept_once(self):
        rows = make_raw_trip("9101", SERVICE_DATE)
        twin = dict(rows[2], stop_id="stop-2-other-platform")
        frame = _build(rows + [twin])
        assert len(frame) == 5

    def test_arrival_before_previous_stop_is_flagged(self):
        rows = make_raw_trip("9102", SERVICE_DATE)
        rows[3]["stop_timestamp"] = rows[1]["stop_timestamp"] - 10
        frame = _build(rows).sort_values("stop_index")
        assert frame["time_inconsistent"].tolist() == [False, False, False, True, False]

    def test_is_arrival_excludes_origin_and_inconsistent_rows(self):
        rows = make_raw_trip("9103", SERVICE_DATE)
        rows[3]["stop_timestamp"] = rows[1]["stop_timestamp"] - 10
        frame = _build(rows).sort_values("stop_index")
        assert clean.is_arrival(frame).tolist() == [False, True, True, False, True]

    def test_trip_an_hour_ahead_of_schedule_is_a_mismatch(self):
        """A whole trip 68 min early (seen on Green-D) is the wrong timetable entry."""
        mismatched = make_raw_trip("9104", SERVICE_DATE, delay_schedule=(-4050,) * 5)
        early_but_real = make_raw_trip("9105", SERVICE_DATE, delay_schedule=(-480,) * 5)
        frame = _build(mismatched + early_but_real)
        assert set(frame["trip_id"]) == {"9105"}

    def test_vehicle_handover_starts_a_new_run(self):
        """A trip finished by another train must not lag across the two trains."""
        rows = make_raw_trip("9106", SERVICE_DATE, delay_schedule=(0, 30, 60, 900, 930))
        for row in rows[3:]:
            row["vehicle_id"] = "v-other"
        frame = _build(rows).sort_values("scheduled_arrival_time")
        assert frame["run"].tolist() == [0, 0, 0, 1, 1]
        assert frame["stop_index"].tolist() == [0, 1, 2, 0, 1]
        assert frame["is_origin"].tolist() == [True, False, False, True, False]

        from mbta_ds import features
        lagged = features.add_propagation_features(frame).sort_values("scheduled_arrival_time")
        assert np.isnan(lagged["prev_delay_1"].iloc[3])   # not the other train's 60 s

    def test_summary_describes_arrivals_only(self, clean_frame):
        stats = clean.summary(clean_frame)
        assert stats["origin_rows"] == 2
        assert stats["arrival_rows"] == stats["rows"] - 2


class TestLedger:
    def test_records_every_step(self, raw_frame, stop_lookup):
        ledger = clean.Ledger()
        ledger.record("a", 10, 7)
        ledger.record("b", 7, 5)
        frame = ledger.to_frame()
        assert list(frame["step"]) == ["a", "b"]
        assert list(frame["rows_removed"]) == [3, 2]


class TestIsolatedGlitches:
    @staticmethod
    def _run(delays):
        return pd.DataFrame({"service_date": 20260601, "trip_id": "1", "run": 0,
                             "delay_seconds": [float(d) for d in delays]})

    def test_single_stop_hours_off_is_flagged(self):
        flags = clean.isolated_glitches(self._run([60, 90, -12085, 96, 120]))
        assert flags.tolist() == [False, False, True, False, False]

    def test_a_real_hold_up_that_persists_is_kept(self):
        flags = clean.isolated_glitches(self._run([60, 90, 2400, 2430, 2450]))
        assert not flags.any()

    def test_glitched_first_and_last_stops_are_flagged(self):
        assert clean.isolated_glitches(self._run([-9000, 30, 40, 50])).tolist() == [True, False, False, False]
        assert clean.isolated_glitches(self._run([30, 40, 50, 9000])).tolist() == [False, False, False, True]
