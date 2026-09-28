"""Shared fixtures: small synthetic frames shaped like the real MBTA data.

The tests never touch the network or the downloaded cache. Instead they build
frames with the same columns and quirks as the real sources, which is both faster
and more reliable than depending on data that changes daily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbta_ds import config

# A fixed service date used everywhere so assertions are deterministic.
SERVICE_DATE = 20260626  # a Friday
SERVICE_DATE_INT = SERVICE_DATE


def _epoch(date_int: int, seconds_after_local_midnight: int) -> float:
    """Local-midnight epoch plus an offset in seconds, in Eastern time."""
    midnight = pd.Timestamp(str(date_int)).tz_localize(config.SERVICE_TZ)
    return float(midnight.value // 10**9 + seconds_after_local_midnight)


def make_raw_trip(
    trip_id: str,
    service_date: int,
    *,
    start_seconds: int = 22_680,
    stops: int = 5,
    delay_schedule: tuple[int, ...] | None = None,
    stop_seconds: int = 120,
) -> list[dict]:
    """Build one trip's raw rows, one per stop."""
    delays = delay_schedule or tuple(30 * i for i in range(stops))
    rows = []
    for index in range(stops):
        scheduled = start_seconds + index * stop_seconds
        observed = scheduled + delays[index]
        rows.append({
            "service_date": service_date,
            "trip_id": trip_id,
            "stop_id": f"stop-{index}",
            "parent_station": f"place-{index}",
            "stop_sequence": index + 1,
            "stop_count": stops,
            "route_id": "Red",
            "trunk_route_id": "Red",
            "branch_route_id": "Red-A",
            "direction_id": False,
            "direction": "South",
            "direction_destination": "Ashmont",
            "vehicle_id": f"v-{trip_id}",
            "start_time": start_seconds,
            "scheduled_arrival_time": float(scheduled),
            "scheduled_departure_time": float(scheduled + 30),
            "stop_timestamp": _epoch(service_date, observed),
            "move_timestamp": _epoch(service_date, observed - 60),
            "travel_time_seconds": 60.0,
            "dwell_time_seconds": 30.0,
            "headway_trunk_seconds": 480.0,
            "headway_branch_seconds": 480.0,
            "scheduled_travel_time": 60.0,
            "scheduled_headway_branch": 480.0,
            "scheduled_headway_trunk": 480.0,
        })
    return rows


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A small raw frame: 2 regular trips, 1 ADDED-, 1 NONREV-, plus dirty rows."""
    rows: list[dict] = []
    rows += make_raw_trip("1001", SERVICE_DATE)
    rows += make_raw_trip("1002", SERVICE_DATE, start_seconds=30_000,
                          delay_schedule=(0, -60, 120, 300, 600))
    # Trips with no usable planned-schedule match.
    rows += make_raw_trip("ADDED-555", SERVICE_DATE, delay_schedule=(-9000,) * 5)
    rows += make_raw_trip("NONREV-777", SERVICE_DATE, delay_schedule=(-18000,) * 5)

    frame = pd.DataFrame(rows)

    # A duplicate of the first stop of trip 1001 (must be de-duplicated).
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    # A row with no observation (must be dropped).
    blank = frame.iloc[1].copy()
    blank["stop_id"] = "stop-null"
    blank["stop_timestamp"] = np.nan
    frame = pd.concat([frame, blank.to_frame().T], ignore_index=True)
    return frame


@pytest.fixture
def stop_lookup() -> pd.DataFrame:
    """station id -> station name, matching the synthetic trips."""
    return pd.DataFrame({
        "stop_id": [f"place-{i}" for i in range(6)],
        "stop_name": [f"Station {i}" for i in range(6)],
    })


@pytest.fixture
def clean_frame(raw_frame, stop_lookup) -> pd.DataFrame:
    """The cleaned frame produced from the synthetic raw rows."""
    from mbta_ds import clean

    return clean.build(raw_frame=raw_frame, stop_lookup=stop_lookup)
