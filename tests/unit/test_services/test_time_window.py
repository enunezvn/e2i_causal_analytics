# tests/unit/test_services/test_time_window.py
from datetime import datetime, timezone

import pytest

from src.services.time_window import WindowParseError, parse_window

NOW = datetime(2026, 6, 20, tzinfo=timezone.utc)


def test_rolling_months():
    w = parse_window("past 3 months", now=NOW)
    assert w.kind == "rolling"
    assert w.end == NOW
    assert w.start == datetime(2026, 3, 20, tzinfo=timezone.utc)


def test_rolling_days():
    w = parse_window("last 90 days", now=NOW)
    assert w.start == datetime(2026, 3, 22, tzinfo=timezone.utc)
    assert w.end == NOW


def test_absolute_quarter():
    w = parse_window("Q1 2025", now=NOW)
    assert w.kind == "absolute"
    assert w.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2025, 4, 1, tzinfo=timezone.utc)


def test_absolute_year():
    w = parse_window("2024", now=NOW)
    assert w.start == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_explicit_dict():
    w = parse_window({"start": "2025-01-01", "end": "2025-02-01"}, now=NOW)
    assert w.start == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_none_passthrough():
    assert parse_window(None, now=NOW) is None


def test_invalid_raises():
    with pytest.raises(WindowParseError):
        parse_window("the time of legends", now=NOW)


def test_start_after_end_raises():
    with pytest.raises(WindowParseError):
        parse_window({"start": "2025-05-01", "end": "2025-01-01"}, now=NOW)


def test_to_params_iso():
    w = parse_window("Q1 2025", now=NOW)
    assert w.start_iso == "2025-01-01T00:00:00+00:00"
    assert w.end_iso == "2025-04-01T00:00:00+00:00"


def test_single_month():
    w = parse_window("March 2025", now=NOW)
    assert w.kind == "absolute"
    assert w.start == datetime(2025, 3, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2025, 4, 1, tzinfo=timezone.utc)


def test_month_range():
    w = parse_window("Jan-Mar 2025", now=NOW)
    assert w.kind == "absolute"
    assert w.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2025, 4, 1, tzinfo=timezone.utc)


def test_iso_date_range():
    w = parse_window("2025-01-01 to 2025-06-01", now=NOW)
    assert w.kind == "absolute"
    assert w.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2025, 6, 1, tzinfo=timezone.utc)


def test_rolling_trailing_weeks():
    w = parse_window("trailing 3 weeks", now=NOW)
    assert w.kind == "rolling"
    assert w.end == NOW
    assert w.start == datetime(2026, 5, 30, tzinfo=timezone.utc)


def test_rolling_previous_years():
    w = parse_window("previous 2 years", now=NOW)
    assert w.kind == "rolling"
    assert w.end == NOW
    assert w.start == datetime(2024, 6, 20, tzinfo=timezone.utc)
