"""Tests for configuration helpers: date parsing, env loading, window manifest."""

from __future__ import annotations

import os
from datetime import date

import pytest

from mbta_ds import config


def test_parse_service_date_accepts_int_and_string():
    assert config.parse_service_date(20260131) == date(2026, 1, 31)
    assert config.parse_service_date("20260131") == date(2026, 1, 31)
    assert config.parse_service_date("2026-01-31") == date(2026, 1, 31)


def test_parse_service_date_rejects_garbage():
    with pytest.raises(ValueError):
        config.parse_service_date("not-a-date")


def test_format_service_date_is_iso():
    assert config.format_service_date(date(2026, 6, 30)) == "2026-06-30"


def test_load_env_reads_values_without_clobbering_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "MBTA_API_KEY=from-file\n"
        "OTHER=value-with-quotes\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MBTA_API_KEY", "already-set")
    monkeypatch.delenv("OTHER", raising=False)

    config.load_env(env_file)

    # An existing environment variable must win over the file.
    assert os.environ["MBTA_API_KEY"] == "already-set"
    assert os.environ["OTHER"] == "value-with-quotes"


def test_get_api_key_returns_none_when_unset(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("MBTA_API_KEY", raising=False)
    assert config.get_api_key() is None


def test_window_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "WINDOW_PATH", tmp_path / "analysis_window.json")

    assert config.load_window() is None
    config.save_window(date(2026, 4, 2), date(2026, 6, 30))
    assert config.load_window() == (date(2026, 4, 2), date(2026, 6, 30))


def test_named_run_gets_its_own_folders_but_shares_downloads(monkeypatch):
    import importlib

    monkeypatch.setenv("MBTA_RUN", "winter")
    winter = importlib.reload(config)
    try:
        assert winter.PROCESSED_DIR == winter.DATA_DIR / "runs" / "winter" / "processed"
        assert winter.WINDOW_PATH.parent == winter.DATA_DIR / "runs" / "winter"
        assert winter.RIDERSHIP_RAW_DIR.parent == winter.DATA_DIR / "runs" / "winter"
        assert winter.REPORTS_DIR == winter.ROOT / "reports" / "winter"
        assert winter.LAMP_RAW_DIR == winter.DATA_DIR / "raw" / "lamp"   # shared
    finally:
        monkeypatch.delenv("MBTA_RUN")
        importlib.reload(config)
    assert config.PROCESSED_DIR == config.DATA_DIR / "processed"


def test_service_day_constants_are_consistent():
    # The MBTA service day starts at 03:00 and GTFS times may exceed 24 hours,
    # which is exactly why the cleaning anchor must be local midnight.
    assert config.SECONDS_PER_DAY == 86_400
    assert config.LATE_THRESHOLD_SECONDS == 300
