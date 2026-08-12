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


def test_rolling_bare_year():
    """Bare 'last year' (no count) means one year — the phrase users actually
    type in chat (session_1784387374342 asked 'over the last year')."""
    w = parse_window("last year", now=NOW)
    assert w.kind == "rolling"
    assert w.end == NOW
    assert w.start == datetime(2025, 6, 20, tzinfo=timezone.utc)


def test_rolling_bare_month_week_day():
    assert parse_window("past month", now=NOW).start == datetime(2026, 5, 20, tzinfo=timezone.utc)
    assert parse_window("previous week", now=NOW).start == datetime(
        2026, 6, 13, tzinfo=timezone.utc
    )
    assert parse_window("last day", now=NOW).start == datetime(2026, 6, 19, tzinfo=timezone.utc)


def test_rolling_bare_unit_label_is_normalized():
    assert parse_window("last year", now=NOW).label == "last 1 years"


# --- Calendar-aligned phrases (#1546): the 2026-08-11 eval's only FAIL (4.5)
# --- and a stall (2.4) came from 'this month' / 'this quarter' / 'last
# --- quarter' raising WindowParseError.


def test_this_month():
    w = parse_window("this month", now=NOW)
    assert w.kind == "absolute"
    assert w.start == datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert w.label == "Jun 2026"
    # lower() at entry already folds case
    assert parse_window("This Month", now=NOW).start == w.start


def test_this_quarter():
    w = parse_window("this quarter", now=NOW)
    assert w.kind == "absolute"
    assert w.start == datetime(2026, 4, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert w.label == "Q2 2026"


def test_last_quarter():
    w = parse_window("last quarter", now=NOW)
    assert w.kind == "absolute"
    assert w.start == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2026, 4, 1, tzinfo=timezone.utc)
    assert w.label == "Q1 2026"


def test_last_quarter_crosses_year_boundary():
    w = parse_window("last quarter", now=datetime(2026, 2, 15, tzinfo=timezone.utc))
    assert w.start == datetime(2025, 10, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert w.label == "Q4 2025"


def test_this_year():
    w = parse_window("this year", now=NOW)
    assert w.kind == "absolute"
    assert w.start == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2027, 1, 1, tzinfo=timezone.utc)
    assert w.label == "2026"


def test_this_week_starts_monday():
    # NOW (2026-06-20) is a Saturday; its ISO week starts Monday 2026-06-15.
    w = parse_window("this week", now=NOW)
    assert w.kind == "absolute"
    assert w.start == datetime(2026, 6, 15, tzinfo=timezone.utc)
    assert w.end == datetime(2026, 6, 22, tzinfo=timezone.utc)


def test_current_and_previous_synonyms():
    this_q = parse_window("this quarter", now=NOW)
    assert parse_window("current quarter", now=NOW).as_dict() == this_q.as_dict()
    last_q = parse_window("last quarter", now=NOW)
    assert parse_window("previous quarter", now=NOW).as_dict() == last_q.as_dict()


def test_last_month_stays_rolling():
    """'last month' predates #1546 and is pinned ROLLING (now - 1 month .. now):
    the rolling branch matches first, so the calendar branch must not change
    its meaning. Only 'quarter' has a last/previous calendar form."""
    w = parse_window("last month", now=NOW)
    assert w.kind == "rolling"
    assert w.end == NOW
    assert w.start == datetime(2026, 5, 20, tzinfo=timezone.utc)
