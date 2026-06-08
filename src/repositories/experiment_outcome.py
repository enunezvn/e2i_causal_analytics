"""ExperimentOutcomeRepository — the real per-unit A/B outcome feed (R5).

Closes the long-standing #422 placeholder (``control_data = []``) that forced
``compute_experiment_results`` / ``scheduled_interim_analysis`` to bail with
``insufficient_data``. There is no dedicated per-unit observation table in the
schema; the REAL per-HCP outcome values live in ``business_metrics`` rows with
``metric_type='per_hcp_rollup'`` (typed columns ``trx_count``, ``nrx_count``,
``total_rx_count``, ``market_share``, ``conversion_rate``, ``engagement_score``,
``call_frequency``). This repository joins an experiment's assignments
(``ab_experiment_assignments.unit_id`` == ``business_metrics.hcp_id``) to those
real rows, collapses multiple ``metric_date`` rows per HCP to one scalar (SUM for
counts, MEAN for rates), and splits by assignment variant into the
(control, treatment) per-unit arrays that
``ResultsAnalysisService._compute_results`` consumes (a pooled two-sample test).

No fabrication: when an experiment has no assignments or no matching metric rows,
``load_arrays`` returns empty arrays and the caller keeps the honest
``insufficient_data`` bail (#422 NaN-safety preserved).

Follows the A/B-side repository convention (sync Supabase client via
``get_supabase_client``; sync ``.execute()``) — mirroring ``ABResultsRepository``
and ``ABExperimentRepository``, NOT the async Twin-side repos.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)

# primary_metric (normalized) -> (business_metrics column, reducer).
# Counts SUM across the window (total prescriptions over the period); rates MEAN
# (a per-period rate averaged over the window). Unknown metrics fail closed.
_COUNT_COLUMNS = {"trx_count", "nrx_count", "total_rx_count"}
_RATE_COLUMNS = {"market_share", "conversion_rate", "engagement_score", "call_frequency"}

METRIC_COLUMN_MAP: Dict[str, str] = {
    "trx": "trx_count",
    "trx_count": "trx_count",
    "nrx": "nrx_count",
    "nrx_count": "nrx_count",
    "total_rx": "total_rx_count",
    "total_rx_count": "total_rx_count",
    "rx": "total_rx_count",
    "market_share": "market_share",
    "conversion_rate": "conversion_rate",
    "conversion": "conversion_rate",
    "engagement_score": "engagement_score",
    "engagement": "engagement_score",
    "call_frequency": "call_frequency",
}


class ExperimentOutcomeRepository:
    """Loads real per-unit experiment outcomes from ``business_metrics``."""

    def __init__(self, supabase_client: Any = None) -> None:
        self.client = supabase_client
        self._ensure_client()

    def _ensure_client(self) -> None:
        """Lazily resolve the sync Supabase client (A/B-side convention)."""
        if self.client is None:
            try:
                from src.repositories import get_supabase_client

                self.client = get_supabase_client()
            except ImportError as exc:  # pragma: no cover - install-shape guard
                logger.warning(
                    "Supabase client not available for ExperimentOutcomeRepository "
                    "(ImportError: %s)",
                    exc,
                )

    # ------------------------------------------------------------------ pure
    # staticmethods so unit tests exercise the real aggregation logic WITHOUT
    # constructing a client-bearing repo (which would resolve a Supabase client
    # and fail in key-less CI). load_arrays calls them via self.
    @staticmethod
    def resolve_column(primary_metric: str) -> Tuple[str, str]:
        """Map a primary_metric to its (business_metrics column, reducer).

        Fail closed on an unknown metric rather than silently picking a column.
        """
        key = (primary_metric or "").strip().lower()
        column = METRIC_COLUMN_MAP.get(key)
        if column is None:
            raise ValueError(
                f"Unsupported primary_metric {primary_metric!r}: no business_metrics "
                f"per-HCP column maps to it. Known: {sorted(set(METRIC_COLUMN_MAP))}."
            )
        reducer = "sum" if column in _COUNT_COLUMNS else "mean"
        return column, reducer

    @staticmethod
    def aggregate_to_arrays(
        assignments: Sequence[Tuple[str, str]],
        rows: Sequence[Dict[str, Any]],
        *,
        column: str,
        reducer: str,
        control_label: str = "control",
        treatment_label: str = "treatment",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collapse per-HCP metric rows to one scalar each, split by variant.

        ``assignments``: (unit_id, variant) pairs. ``rows``: business_metrics
        dicts carrying ``hcp_id`` and ``column``. Units whose values are all NULL,
        or with no rows, are excluded (never NaN). Variants other than
        control/treatment are ignored. Returns per-unit (control, treatment)
        float arrays.
        """
        variant_by_unit: Dict[str, str] = {str(uid): str(var) for uid, var in assignments}

        # Gather non-null values per assigned HCP.
        values_by_unit: Dict[str, List[float]] = {}
        for row in rows:
            hcp = str(row.get("hcp_id")) if row.get("hcp_id") is not None else None
            if hcp is None or hcp not in variant_by_unit:
                continue
            raw = row.get(column)
            if raw is None:
                continue
            try:
                values_by_unit.setdefault(hcp, []).append(float(raw))
            except (TypeError, ValueError):
                continue

        control: List[float] = []
        treatment: List[float] = []
        for hcp, vals in values_by_unit.items():
            if not vals:
                continue
            scalar = float(np.sum(vals)) if reducer == "sum" else float(np.mean(vals))
            variant = variant_by_unit[hcp]
            if variant == control_label:
                control.append(scalar)
            elif variant == treatment_label:
                treatment.append(scalar)
            # other variants intentionally ignored for a 2-arm ATE

        return (
            np.asarray(control, dtype=float),
            np.asarray(treatment, dtype=float),
        )

    # ------------------------------------------------------------------- I/O
    async def load_arrays(
        self,
        experiment_id: UUID,
        primary_metric: str,
        *,
        brand: Optional[str] = None,
        window_days: Optional[int] = None,
        control_label: str = "control",
        treatment_label: str = "treatment",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Real outcome feed: assignments ⋈ business_metrics → (control, treatment).

        Returns empty arrays (caller bails ``insufficient_data``) when there are no
        assignments or no matching per-HCP metric rows. ``window_days`` is accepted
        for future post-assignment windowing; today the A/B schema carries no
        measurement-window columns, so when unset we read all available
        ``per_hcp_rollup`` rows for the brand (documented in the design brief).
        """
        column, reducer = self.resolve_column(primary_metric)

        if self.client is None:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        # 1) assignments for the experiment (unit_id, variant)
        assign_res = (
            self.client.table("ab_experiment_assignments")
            .select("unit_id,variant")
            .eq("experiment_id", str(experiment_id))
            .execute()
        )
        assignments = [
            (r["unit_id"], r["variant"]) for r in (assign_res.data or []) if r.get("unit_id")
        ]
        if not assignments:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        unit_ids = [uid for uid, _ in assignments]

        # 2) real per-HCP outcome rows from business_metrics
        query = (
            self.client.table("business_metrics")
            .select(f"hcp_id,{column},metric_date,brand")
            .eq("metric_type", "per_hcp_rollup")
            .in_("hcp_id", unit_ids)
        )
        if brand:
            query = query.eq("brand", brand)
        metric_res = query.execute()
        rows = metric_res.data or []

        # window filtering (optional; no A/B window in schema today)
        if window_days is not None:
            rows = self._filter_window(rows, window_days)

        return self.aggregate_to_arrays(
            assignments,
            rows,
            column=column,
            reducer=reducer,
            control_label=control_label,
            treatment_label=treatment_label,
        )

    @staticmethod
    def _filter_window(rows: List[Dict[str, Any]], window_days: int) -> List[Dict[str, Any]]:
        """Keep rows within the most-recent ``window_days`` of observed metric_dates.

        Conservative, schema-faithful default: anchor on the latest metric_date in
        the pulled set (no per-assignment start column exists). No-op if dates are
        absent.
        """
        from datetime import date, timedelta

        dates = []
        for r in rows:
            d = r.get("metric_date")
            if isinstance(d, str):
                try:
                    d = date.fromisoformat(d[:10])
                except ValueError:
                    d = None
            if isinstance(d, date):
                dates.append(d)
        if not dates:
            return rows
        cutoff = max(dates) - timedelta(days=window_days)

        def _in_window(r: Dict[str, Any]) -> bool:
            d = r.get("metric_date")
            if isinstance(d, str):
                try:
                    d = date.fromisoformat(d[:10])
                except ValueError:
                    return True
            return not isinstance(d, date) or d >= cutoff

        return [r for r in rows if _in_window(r)]
