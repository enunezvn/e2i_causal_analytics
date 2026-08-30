"""Shared ``time_period`` grammar for the gap analyzer (#1834).

ONE grammar, three consumers: the API request model (422 at the boundary), the
gap_detector node (resolves the window once, surfaces it in state, fails closed
on garbage) and ``SupabaseDataConnector`` (reads the current and prior windows
from it). Before #1834 the connector was the only parser, it recognised four
forms, and everything else — including the request DEFAULT ``current_quarter``
and the documented ``2024-Q3`` — fell through to a silent "last 90 days" window.

Accepted forms (case-insensitive keywords, surrounding whitespace ignored):

    current_quarter            quarter start → today
    previous_quarter           the preceding FULL calendar quarter
    last_quarter               alias of previous_quarter
    Q#_YYYY / YYYY-Q#          an explicit FULL calendar quarter
    YTD                        Jan 1 → today
    MTD                        1st of this month → today
    YYYY-MM-DD_YYYY-MM-DD      an explicit inclusive range (start <= end)

Prior window (what "temporal" gaps compare against):

* calendar-quarter forms (current/previous/explicit quarter): the preceding
  FULL calendar quarter. Rows in ``business_metrics`` sit on the 1st of each
  month (monthly grain); the old day-count shift turned a quarter-to-date
  window into a prior of May 2–Jun 30 with ONE monthly row instead of three.
  The connector pivots with ``mean`` per segment, so a 2-month quarter-to-date
  mean is compared against a 3-month full-quarter mean — like for like.
* YTD: the same Jan 1 → same-day span of the previous year (Feb 29 clamps to
  Feb 28).
* MTD: the preceding FULL calendar month (a day-shifted MTD prior of Jul 2–31
  would contain zero monthly rows).
* explicit ranges: the length-shift is kept, but ALIGNED TO THE DATA GRAIN —
  a range that starts on the 1st shifts back by the same number of whole
  months it touches (Feb 1–28 → Jan 1–31, not Jan 4–31; a partial month such
  as Jul 1–15 → ALL of June, because June's single monthly row sits on the
  1st and a day-shift to Jun 16–30 would contain no rows); any other range
  shifts back by its inclusive day count, ending the day before the range
  starts.

Anything else raises :class:`TimePeriodError` (a ``ValueError``) naming the
accepted forms. There is no fallback: an unparseable period is an error, not
a different analysis.

The clock is injected: ``resolve_time_period(label, today=...)``. Production
callers omit ``today`` and get ``date.today()`` via :func:`_today`, which is
the one seam tests may freeze when they cannot pass the parameter (the node
and connector read ``state["time_period"]`` and have no clock argument).

Clock semantics: ``date.today()`` is the SERVER's calendar date — the api
container runs with TZ unset (UTC), so relative forms (current_quarter,
previous_quarter, YTD, MTD) roll over at 00:00 UTC, not at a business-market
midnight (a request at 20:30 EDT on Aug 31 resolves MTD as Sep 1..Sep 1).
There is no business-timezone contract in this codebase to bind to; the
resolved dates are returned to the caller so the window is visible, and an
explicit ``YYYY-MM-DD_YYYY-MM-DD`` range is the escape hatch for an exact
as-of window. Binding relative forms to a market timezone is a product
decision, deliberately not taken here (see #1834 codex iter-1 MEDIUM-2).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Literal, Optional, Tuple

__all__ = [
    "ACCEPTED_FORMS",
    "ResolvedTimePeriod",
    "TimePeriodError",
    "accepted_forms_text",
    "resolve_time_period",
]

ACCEPTED_FORMS: Tuple[str, ...] = (
    "current_quarter",
    "previous_quarter",
    "last_quarter",
    "Q#_YYYY",
    "YYYY-Q#",
    "YTD",
    "MTD",
    "YYYY-MM-DD_YYYY-MM-DD",
)

PeriodKind = Literal[
    "quarter_to_date",
    "calendar_quarter",
    "year_to_date",
    "month_to_date",
    "explicit",
]


class TimePeriodError(ValueError):
    """An unparseable ``time_period``. Subclasses ``ValueError`` so pydantic
    validators and the gap_detector node's broad ``except`` see it unchanged."""


@dataclass(frozen=True)
class ResolvedTimePeriod:
    """The concrete inclusive windows an analysis compares."""

    time_period: str
    period_start: date
    period_end: date
    prior_start: date
    prior_end: date
    kind: PeriodKind

    def to_dict(self) -> Dict[str, str]:
        """ISO-8601 strings, keyed as the state channel and API field expect."""
        return {
            "time_period": self.time_period,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "prior_start": self.prior_start.isoformat(),
            "prior_end": self.prior_end.isoformat(),
        }


def accepted_forms_text() -> str:
    return ", ".join(ACCEPTED_FORMS)


def _today() -> date:
    """Production clock. Tests that cannot inject ``today`` freeze this seam."""
    return date.today()


_QUARTER_UNDERSCORE = re.compile(r"^Q([1-4])_(\d{4})$", re.IGNORECASE)
_QUARTER_ISO = re.compile(r"^(\d{4})-Q([1-4])$", re.IGNORECASE)
_EXPLICIT_RANGE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$")


def _unsupported(label: object) -> TimePeriodError:
    return TimePeriodError(
        f"Unsupported time_period {label!r}. Accepted forms: {accepted_forms_text()}."
    )


def _quarter_bounds(year: int, quarter: int) -> Tuple[date, date]:
    start_month = 3 * (quarter - 1) + 1
    end_month = start_month + 2
    return (
        date(year, start_month, 1),
        date(year, end_month, calendar.monthrange(year, end_month)[1]),
    )


def _previous_quarter(year: int, quarter: int) -> Tuple[int, int]:
    return (year - 1, 4) if quarter == 1 else (year, quarter - 1)


def _quarter_of(day: date) -> int:
    return (day.month - 1) // 3 + 1


def _first_of_month_shifted(day: date, months_back: int) -> date:
    """The 1st of the month ``months_back`` months before ``day``'s month."""
    index = day.year * 12 + (day.month - 1) - months_back
    return date(index // 12, index % 12 + 1, 1)


def _same_day_previous_year(day: date) -> date:
    if day.month == 2 and day.day == 29:
        return date(day.year - 1, 2, 28)
    return day.replace(year=day.year - 1)


def _explicit_prior(start: date, end: date) -> Tuple[date, date]:
    """Length-shift aligned to the monthly grain (see module docstring)."""
    prior_end = start - timedelta(days=1)
    if start.day == 1:
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
        return _first_of_month_shifted(start, months), prior_end
    inclusive_days = (end - start).days + 1
    return prior_end - timedelta(days=inclusive_days - 1), prior_end


def resolve_time_period(time_period: str, today: Optional[date] = None) -> ResolvedTimePeriod:
    """Resolve a ``time_period`` label to concrete current and prior windows.

    Args:
        time_period: one of the accepted forms (see module docstring).
        today: the clock; production callers omit it (``date.today()``).

    Raises:
        TimePeriodError: for anything outside the grammar — no fallback.
    """
    if not isinstance(time_period, str):
        raise TimePeriodError(
            f"time_period must be a string, got {type(time_period).__name__}. "
            f"Accepted forms: {accepted_forms_text()}."
        )
    label = time_period.strip()
    if not label:
        raise TimePeriodError(f"time_period is empty. Accepted forms: {accepted_forms_text()}.")

    clock = today if today is not None else _today()
    keyword = label.lower()

    if keyword == "current_quarter":
        year, quarter = clock.year, _quarter_of(clock)
        start, _ = _quarter_bounds(year, quarter)
        prior_start, prior_end = _quarter_bounds(*_previous_quarter(year, quarter))
        return ResolvedTimePeriod(label, start, clock, prior_start, prior_end, "quarter_to_date")

    if keyword in ("previous_quarter", "last_quarter"):
        year, quarter = _previous_quarter(clock.year, _quarter_of(clock))
        start, end = _quarter_bounds(year, quarter)
        prior_start, prior_end = _quarter_bounds(*_previous_quarter(year, quarter))
        return ResolvedTimePeriod(label, start, end, prior_start, prior_end, "calendar_quarter")

    match = _QUARTER_UNDERSCORE.match(label)
    if match:
        quarter, year = int(match.group(1)), int(match.group(2))
    else:
        match = _QUARTER_ISO.match(label)
        if match:
            year, quarter = int(match.group(1)), int(match.group(2))
    if match:
        start, end = _quarter_bounds(year, quarter)
        prior_start, prior_end = _quarter_bounds(*_previous_quarter(year, quarter))
        return ResolvedTimePeriod(label, start, end, prior_start, prior_end, "calendar_quarter")

    if keyword == "ytd":
        start = date(clock.year, 1, 1)
        prior_start = date(clock.year - 1, 1, 1)
        prior_end = _same_day_previous_year(clock)
        return ResolvedTimePeriod(label, start, clock, prior_start, prior_end, "year_to_date")

    if keyword == "mtd":
        start = clock.replace(day=1)
        prior_start = _first_of_month_shifted(clock, 1)
        prior_end = start - timedelta(days=1)
        return ResolvedTimePeriod(label, start, clock, prior_start, prior_end, "month_to_date")

    match = _EXPLICIT_RANGE.match(label)
    if match:
        try:
            start = date.fromisoformat(match.group(1))
            end = date.fromisoformat(match.group(2))
        except ValueError as exc:
            raise TimePeriodError(
                f"Unsupported time_period {label!r}: {exc}. "
                f"Accepted forms: {accepted_forms_text()}."
            ) from exc
        if start > end:
            raise TimePeriodError(
                f"Unsupported time_period {label!r}: start is after end. "
                f"Accepted forms: {accepted_forms_text()}."
            )
        prior_start, prior_end = _explicit_prior(start, end)
        return ResolvedTimePeriod(label, start, end, prior_start, prior_end, "explicit")

    raise _unsupported(label)
