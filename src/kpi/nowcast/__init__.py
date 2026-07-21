"""Completion-factor (chain-ladder) claims nowcast for the Rx-volume KPIs.

Backlog #45 PR-B. See :mod:`src.kpi.nowcast.completion_factor`.
"""

from src.kpi.nowcast.completion_factor import (
    MIN_MATURE_MONTHS,
    NOWCAST_KPI_QUERY_FAMILIES,
    MonthNowcast,
    NowcastConfig,
    NowcastResult,
    estimate_completion_from_rows,
    fetch_nowcast_rows,
)

__all__ = [
    "MIN_MATURE_MONTHS",
    "NOWCAST_KPI_QUERY_FAMILIES",
    "MonthNowcast",
    "NowcastConfig",
    "NowcastResult",
    "estimate_completion_from_rows",
    "fetch_nowcast_rows",
]
