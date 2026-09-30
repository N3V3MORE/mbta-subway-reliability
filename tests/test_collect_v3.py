"""The live collector's storage: one gzip file per resource and local day."""

from __future__ import annotations

import gzip

import pandas as pd

from mbta_ds import collect_v3


def test_polls_go_to_the_local_day_they_were_made(tmp_path):
    # 03:30 UTC on 1 October is 23:30 on 30 September in Boston.
    ts = pd.Timestamp("2026-10-01 03:30", tz="UTC").timestamp()
    assert collect_v3.daily_path("predictions", ts, tmp_path) == tmp_path / "predictions" / "2026-09-30.jsonl.gz"


def test_appends_accumulate_and_load_back_with_the_legacy_file(tmp_path):
    ts = pd.Timestamp("2026-10-01 14:00", tz="UTC").timestamp()
    path = collect_v3.daily_path("alerts", ts, tmp_path)
    collect_v3._append_jsonl(path, [{"id": "a", "_fetch_ts": ts}])
    collect_v3._append_jsonl(path, [{"id": "b", "_fetch_ts": ts + 60}])
    collect_v3._append_jsonl(path, [])            # an empty poll writes nothing
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert len(handle.readlines()) == 2
    (tmp_path / "alerts.jsonl").write_text('{"id": "old", "_fetch_ts": 1}\n', encoding="utf-8")
    frame = collect_v3.load_live("alerts", tmp_path)
    assert frame["id"].tolist() == ["old", "a", "b"]


def test_until_counts_to_the_next_occurrence_of_the_clock_time():
    tz = "America/New_York"
    assert collect_v3.minutes_until("03:00", pd.Timestamp("2026-09-30 05:00", tz=tz)) == 22 * 60
    assert collect_v3.minutes_until("03:00", pd.Timestamp("2026-09-30 02:30", tz=tz)) == 30
