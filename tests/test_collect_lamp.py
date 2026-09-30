"""Tests for the LAMP collector's cache freshness rule (no network)."""

from __future__ import annotations

import os
from datetime import date, datetime

from mbta_ds import collect_lamp


def test_a_snapshot_taken_on_or_before_the_window_end_cannot_cover_it(tmp_path):
    archive = tmp_path / "alerts.parquet"
    assert not collect_lamp.covers(archive, date(2026, 6, 30))   # not downloaded yet

    archive.write_bytes(b"x")
    stamp = datetime(2026, 6, 30, 23, 0).timestamp()
    os.utime(archive, (stamp, stamp))
    assert not collect_lamp.covers(archive, date(2026, 6, 30))   # same day: incomplete

    stamp = datetime(2026, 7, 2, 9, 0).timestamp()
    os.utime(archive, (stamp, stamp))
    assert collect_lamp.covers(archive, date(2026, 6, 30))
