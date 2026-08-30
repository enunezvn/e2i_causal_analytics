"""#1834 — the gap_analyzer ``time_period`` grammar, pinned against a frozen clock.

Measured on prod (2026-08-30): the request DEFAULT ``current_quarter`` was never
parsed by ``SupabaseDataConnector._parse_time_period`` — it, the documented
``2024-Q3`` form and any garbage all fell through to a silent "last 90 days"
window (Jun 1–Aug 30 vs Mar 2–May 31), so the "current quarter" label on
/ai-insights disagreed with the arithmetic and nothing told the user.

These tests pin the ONE shared grammar (``src.utils.gap_time_period``)
that the connector, the API model and the gap_detector node all consume. The
clock is injected as a plain ``today`` parameter — production callers omit it
and get ``date.today()``; no module-level monkeypatching of ``datetime`` is
needed, so the production code path stays honest and the tests stay hermetic.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.utils.gap_time_period import (
    ACCEPTED_FORMS,
    TimePeriodError,
    resolve_time_period,
)

TODAY = date(2026, 8, 30)  # the day the defect was measured; Q3-2026, month 2 of 3


def _iso(resolved) -> tuple[str, str, str, str]:
    return (
        resolved.period_start.isoformat(),
        resolved.period_end.isoformat(),
        resolved.prior_start.isoformat(),
        resolved.prior_end.isoformat(),
    )


# ---------------------------------------------------------------------------
# Relative calendar-quarter forms (the DEFAULT lives here)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_current_quarter_is_quarter_start_to_today_with_preceding_full_quarter_prior():
    """The default: quarter start → today; prior = the preceding FULL calendar quarter."""
    resolved = resolve_time_period("current_quarter", today=TODAY)
    assert _iso(resolved) == ("2026-07-01", "2026-08-30", "2026-04-01", "2026-06-30")


@pytest.mark.unit
@pytest.mark.parametrize("label", ["previous_quarter", "last_quarter"])
def test_previous_quarter_aliases_resolve_to_preceding_full_quarter(label):
    resolved = resolve_time_period(label, today=TODAY)
    assert _iso(resolved) == ("2026-04-01", "2026-06-30", "2026-01-01", "2026-03-31")


@pytest.mark.unit
def test_current_quarter_on_first_day_of_quarter_is_a_single_day_window():
    """Oct 1: the quarter has one day of data; the prior window is still Q3 in full."""
    resolved = resolve_time_period("current_quarter", today=date(2026, 10, 1))
    assert _iso(resolved) == ("2026-10-01", "2026-10-01", "2026-07-01", "2026-09-30")


@pytest.mark.unit
def test_previous_quarter_in_q1_rolls_the_year_back():
    resolved = resolve_time_period("previous_quarter", today=date(2027, 2, 14))
    assert _iso(resolved) == ("2026-10-01", "2026-12-31", "2026-07-01", "2026-09-30")


# ---------------------------------------------------------------------------
# Explicit calendar-quarter forms: Q#_YYYY and YYYY-Q#
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("label", ["Q3_2026", "2026-Q3", "q3_2026", "2026-q3"])
def test_explicit_quarter_forms_are_full_calendar_quarters(label):
    resolved = resolve_time_period(label, today=TODAY)
    assert _iso(resolved) == ("2026-07-01", "2026-09-30", "2026-04-01", "2026-06-30")


@pytest.mark.unit
def test_q1_prior_is_q4_of_the_previous_year():
    resolved = resolve_time_period("Q1_2026", today=TODAY)
    assert _iso(resolved) == ("2026-01-01", "2026-03-31", "2025-10-01", "2025-12-31")


@pytest.mark.unit
def test_documented_2024_q3_form_from_the_api_docstring_is_accepted():
    """The API field description advertised ``'2024-Q3'`` for years; it must parse."""
    resolved = resolve_time_period("2024-Q3", today=TODAY)
    assert _iso(resolved) == ("2024-07-01", "2024-09-30", "2024-04-01", "2024-06-30")


# ---------------------------------------------------------------------------
# YTD / MTD
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("label", ["YTD", "ytd"])
def test_ytd_prior_is_the_same_span_of_the_previous_year(label):
    resolved = resolve_time_period(label, today=TODAY)
    assert _iso(resolved) == ("2026-01-01", "2026-08-30", "2025-01-01", "2025-08-30")


@pytest.mark.unit
def test_ytd_on_leap_day_clamps_the_prior_end_to_feb_28():
    resolved = resolve_time_period("YTD", today=date(2028, 2, 29))
    assert _iso(resolved) == ("2028-01-01", "2028-02-29", "2027-01-01", "2027-02-28")


@pytest.mark.unit
@pytest.mark.parametrize("label", ["MTD", "mtd"])
def test_mtd_prior_is_the_preceding_full_calendar_month(label):
    """A day-shifted MTD prior (Jul 2–Jul 31) would contain ZERO monthly-grain rows."""
    resolved = resolve_time_period(label, today=TODAY)
    assert _iso(resolved) == ("2026-08-01", "2026-08-30", "2026-07-01", "2026-07-31")


@pytest.mark.unit
def test_mtd_in_january_rolls_the_year_back():
    resolved = resolve_time_period("MTD", today=date(2027, 1, 9))
    assert _iso(resolved) == ("2027-01-01", "2027-01-09", "2026-12-01", "2026-12-31")


# ---------------------------------------------------------------------------
# Explicit YYYY-MM-DD_YYYY-MM-DD ranges — length-shift, aligned to the monthly grain
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_explicit_month_aligned_range_prior_is_the_same_number_of_whole_months():
    """Rows sit on the 1st of each month; a month-aligned range shifts by whole months."""
    resolved = resolve_time_period("2026-07-01_2026-08-30", today=TODAY)
    assert _iso(resolved) == ("2026-07-01", "2026-08-30", "2026-05-01", "2026-06-30")


@pytest.mark.unit
def test_explicit_february_range_prior_still_contains_january_first():
    """The old day-shift gave Jan 4–Jan 31 for a Feb range: zero monthly rows."""
    resolved = resolve_time_period("2026-02-01_2026-02-28", today=TODAY)
    assert _iso(resolved) == ("2026-02-01", "2026-02-28", "2026-01-01", "2026-01-31")


@pytest.mark.unit
def test_explicit_non_aligned_range_keeps_the_inclusive_length_shift():
    """31 inclusive days in → 31 inclusive days out, ending the day before the start."""
    resolved = resolve_time_period("2026-07-15_2026-08-14", today=TODAY)
    assert _iso(resolved) == ("2026-07-15", "2026-08-14", "2026-06-14", "2026-07-14")


@pytest.mark.unit
def test_explicit_range_ignores_the_clock():
    """An explicit range is absolute — the frozen clock must not leak into it."""
    a = resolve_time_period("2024-01-01_2024-03-31", today=TODAY)
    b = resolve_time_period("2024-01-01_2024-03-31", today=date(2031, 12, 31))
    assert _iso(a) == _iso(b) == ("2024-01-01", "2024-03-31", "2023-10-01", "2023-12-31")


# ---------------------------------------------------------------------------
# Fail closed — no silent 90-day fallback anywhere
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "garbage",
    [
        "bogus",
        "",
        "   ",
        "last_90_days",
        "2024Q3",
        "Q5_2026",
        "Q0_2026",
        "2026-Q0",
        "Q3 2026",
        "2026-07-01",
        "2026-07-01_2026-06-30",  # start after end
        "2026-02-30_2026-03-31",  # not a real date
        "2026-7-1_2026-8-30",  # not zero-padded: not the documented form
        "current-quarter",
    ],
)
def test_unknown_forms_raise_time_period_error_listing_the_accepted_forms(garbage):
    with pytest.raises(TimePeriodError) as excinfo:
        resolve_time_period(garbage, today=TODAY)
    message = str(excinfo.value)
    assert repr(garbage) in message or "empty" in message.lower()
    for form in ACCEPTED_FORMS:
        assert form in message, f"accepted form {form!r} missing from: {message}"


@pytest.mark.unit
def test_time_period_error_is_a_value_error():
    """Callers that already catch ValueError (pydantic validators, the node's broad
    except) must see the fail-closed signal without a new import."""
    assert issubclass(TimePeriodError, ValueError)


@pytest.mark.unit
def test_non_string_input_fails_closed():
    with pytest.raises(TimePeriodError):
        resolve_time_period(None, today=TODAY)  # type: ignore[arg-type]


@pytest.mark.unit
def test_whitespace_is_tolerated_around_a_valid_label():
    resolved = resolve_time_period("  current_quarter \n", today=TODAY)
    assert _iso(resolved)[:2] == ("2026-07-01", "2026-08-30")


# ---------------------------------------------------------------------------
# The resolved window is a plain, serialisable record
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_to_dict_is_iso_strings_keyed_for_state_and_response():
    resolved = resolve_time_period("current_quarter", today=TODAY)
    assert resolved.to_dict() == {
        "time_period": "current_quarter",
        "period_start": "2026-07-01",
        "period_end": "2026-08-30",
        "prior_start": "2026-04-01",
        "prior_end": "2026-06-30",
    }


@pytest.mark.unit
def test_production_default_clock_is_today():
    """Without ``today`` the grammar uses the real clock (no frozen leak into prod)."""
    resolved = resolve_time_period("MTD")
    real_today = date.today()
    assert resolved.period_end == real_today
    assert resolved.period_start == real_today.replace(day=1)


@pytest.mark.unit
def test_explicit_partial_month_range_starting_on_the_first_priors_the_whole_previous_month():
    """codex iter-1 LOW: intent pin. A range that starts on the 1st is month-grain
    even when it ends mid-month — Jul 1–15 compares against ALL of June (the one
    monthly row that exists), not against Jun 16–30 (zero rows)."""
    resolved = resolve_time_period("2026-07-01_2026-07-15", today=TODAY)
    assert _iso(resolved) == ("2026-07-01", "2026-07-15", "2026-06-01", "2026-06-30")
