# src/services/time_window.py
"""Parse user-requested time windows (rolling or absolute) into a normalized
half-open ``[start, end)`` UTC range for the KPI engine.

Rolling windows ("last 3 months") and calendar-aligned phrases ("this month",
"this quarter", "last quarter") are anchored to ``now``. Absolute windows
("Q1 2025", "2024", "Jan-Mar 2025", ISO dates) are fixed. Returns ``None`` for
no-window input; raises :class:`WindowParseError` on unparseable / invalid input
(never silently defaults)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from dateutil.relativedelta import relativedelta  # type: ignore[import-untyped]

_MONTH_ABBRS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTHS = {m: i for i, m in enumerate(_MONTH_ABBRS, start=1)}


class WindowParseError(ValueError):
    """Raised when a window string cannot be parsed or is invalid."""


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime
    kind: str  # "rolling" | "absolute"
    label: str

    @property
    def start_iso(self) -> str:
        return self.start.isoformat()

    @property
    def end_iso(self) -> str:
        return self.end.isoformat()

    def as_dict(self) -> dict[str, str]:
        return {"start": self.start_iso, "end": self.end_iso}


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _validate(start: datetime, end: datetime, kind: str, label: str) -> Window:
    if start >= end:
        raise WindowParseError(f"window start {start} is not before end {end}")
    return Window(start=start, end=end, kind=kind, label=label)


def parse_window(spec: Any, *, now: Optional[datetime] = None) -> Optional[Window]:
    now = _utc(now) if now is not None else datetime.now(timezone.utc)
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None

    if isinstance(spec, dict):
        try:
            start = _utc(datetime.fromisoformat(str(spec["start"])))
            end = _utc(datetime.fromisoformat(str(spec["end"])))
        except (KeyError, ValueError) as e:
            raise WindowParseError(f"invalid explicit window {spec!r}: {e}") from e
        return _validate(start, end, "absolute", f"{start.date()} to {end.date()}")

    if not isinstance(spec, str):
        raise WindowParseError(f"unsupported window type: {type(spec).__name__}")
    s = spec.strip().lower()

    # The count is optional: bare "last year" / "past month" means one unit.
    m = re.fullmatch(r"(?:last|past|trailing|previous)\s+(?:(\d+)\s+)?(day|week|month|year)s?", s)
    if m:
        n, unit = int(m.group(1) or 1), m.group(2)
        delta = {
            "day": relativedelta(days=n),
            "week": relativedelta(weeks=n),
            "month": relativedelta(months=n),
            "year": relativedelta(years=n),
        }[unit]
        return _validate(now - delta, now, "rolling", f"last {n} {unit}s")

    # Calendar-aligned phrases anchored to ``now`` (#1546): "this month" is the
    # full calendar month containing ``now``, "last quarter" the most recent
    # completed calendar quarter. The full period (not ``.. now``) keeps the
    # window valid even at the period's first instant; future dates simply hold
    # no data. "last month" / "last year" keep their rolling meaning — the
    # rolling branch above matches them first — so only "quarter", which has no
    # rolling unit, carries a last/previous calendar form here.
    m = re.fullmatch(
        r"(?:(?:this|current)\s+(week|month|quarter|year)|(?:last|previous)\s+(quarter))", s
    )
    if m:
        unit = m.group(1) or m.group(2)
        step = {
            "week": relativedelta(weeks=1),
            "month": relativedelta(months=1),
            "quarter": relativedelta(months=3),
            "year": relativedelta(years=1),
        }[unit]
        if unit == "week":
            start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - relativedelta(
                days=now.weekday()
            )
        elif unit == "month":
            start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        elif unit == "quarter":
            start = datetime(now.year, 3 * ((now.month - 1) // 3) + 1, 1, tzinfo=timezone.utc)
        else:  # year
            start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        if m.group(2):  # last/previous quarter
            start -= step
        label = {
            "week": f"week of {start.date()}",
            "month": f"{_MONTH_ABBRS[start.month - 1].capitalize()} {start.year}",
            "quarter": f"Q{(start.month - 1) // 3 + 1} {start.year}",
            "year": str(start.year),
        }[unit]
        return _validate(start, start + step, "absolute", label)

    m = re.fullmatch(r"q([1-4])\s+(\d{4})", s)
    if m:
        q, yr = int(m.group(1)), int(m.group(2))
        start = datetime(yr, 3 * (q - 1) + 1, 1, tzinfo=timezone.utc)
        return _validate(start, start + relativedelta(months=3), "absolute", f"Q{q} {yr}")

    m = re.fullmatch(r"([a-z]{3,})\s*(?:-|to|–)\s*([a-z]{3,})\s+(\d{4})", s)
    if m and m.group(1)[:3] in _MONTHS and m.group(2)[:3] in _MONTHS:
        a, b, yr = _MONTHS[m.group(1)[:3]], _MONTHS[m.group(2)[:3]], int(m.group(3))
        start = datetime(yr, a, 1, tzinfo=timezone.utc)
        end = datetime(yr, b, 1, tzinfo=timezone.utc) + relativedelta(months=1)
        return _validate(start, end, "absolute", f"{m.group(1)}-{m.group(2)} {yr}")

    m = re.fullmatch(r"([a-z]{3,})\s+(\d{4})", s)
    if m and m.group(1)[:3] in _MONTHS:
        mo, yr = _MONTHS[m.group(1)[:3]], int(m.group(2))
        start = datetime(yr, mo, 1, tzinfo=timezone.utc)
        return _validate(start, start + relativedelta(months=1), "absolute", f"{m.group(1)} {yr}")

    m = re.fullmatch(r"(\d{4})", s)
    if m:
        yr = int(m.group(1))
        start = datetime(yr, 1, 1, tzinfo=timezone.utc)
        return _validate(start, datetime(yr + 1, 1, 1, tzinfo=timezone.utc), "absolute", str(yr))

    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*(?:to|–|-)\s*(\d{4}-\d{2}-\d{2})", s)
    if m:
        start = _utc(datetime.fromisoformat(m.group(1)))
        end = _utc(datetime.fromisoformat(m.group(2)))
        return _validate(start, end, "absolute", f"{m.group(1)} to {m.group(2)}")

    raise WindowParseError(f"could not parse time window: {spec!r}")
