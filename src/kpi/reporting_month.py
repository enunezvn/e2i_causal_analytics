"""Frontier/period fields for the chat KPI payload (#2114 live certification, 2026-09-19).

The canonical WS3-BI-005..008 statements (migration 143) return the month a headline covers
(``data_month``) beside its last day (``data_through``). The chat payload used to copy only
``data_through``, so an answer read "the most recent complete calendar month, data through
2026-08-31" and never named August. Home labels the same period ``volume_period`` with the
same ``%B %Y`` format (``home_volume_summary``).

The label is attached ONLY to a default-window headline: a custom window sums several months
and its ``data_month`` is just the last of them, so a single-month label would misstate it.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional


def month_label(data_month: Any) -> Optional[str]:
    """``"August 2026"`` for a ``date`` or an ISO ``YYYY-MM[-DD]`` string; None otherwise."""
    if isinstance(data_month, date):
        return data_month.strftime("%B %Y")
    text = str(data_month or "")[:7]
    try:
        return date(int(text[:4]), int(text[5:7]), 1).strftime("%B %Y")
    except ValueError:
        return None


def frontier_fields(metadata: Mapping[str, Any], window_status: str, kpi_id: str) -> dict[str, Any]:
    """``data_through`` / ``reporting_month`` / ``reporting_window`` for one KPI payload.

    Each key is present only when its source is: honest absence, never a guessed period.
    ``reporting_window`` (the static default-window note) is emitted only for the default
    window, and carries the concrete month when one is known, because the model quotes
    that note verbatim.
    """
    from src.kpi.capability_policy import reporting_windows

    context: Mapping[str, Any] = metadata.get("context") or {}
    default_window_note = reporting_windows().get(kpi_id)
    fields: dict[str, Any] = {}
    if context.get("data_through") is not None:
        fields["data_through"] = context["data_through"]
    if window_status != "default":
        return fields
    label = month_label(context.get("data_month")) if context.get("data_month") else None
    if label:
        fields["reporting_month"] = label
    if default_window_note:
        fields["reporting_window"] = (
            f"{default_window_note} ({label})" if label else default_window_note
        )
    return fields
