"""ExperimentOutcomeRepository — the real per-unit A/B outcome feed (R5).

Closes the long-standing #422 placeholder (``control_data = []``) that forced
``compute_experiment_results`` / ``scheduled_interim_analysis`` to bail with
``insufficient_data``. TWO feeds, in precedence order:

1. ``ab_experiment_unit_outcomes`` (migration 155; option d1, owner decision
   2026-09-23, Part of #2207): ONE observed outcome per (experiment_id, unit_id,
   metric_name), time-indexed by ``observed_at``. Read FIRST for the experiment's
   ``prediction_target``; when >= 1 row exists the arrays come from it (MEAN per
   unit — one row per unit by the UNIQUE key). Synthetic experiments are fed by
   the generator from the SAME per-unit draw as their ``ab_experiment_results``
   row (measured 2026-09-23: the per-HCP outcome tables come from independent
   DGPs, so the join below yielded a structural-null ATE for every synthetic
   experiment; each HCP sits in ~12 experiments per brand, so the outcome must be
   keyed per experiment). No real writer exists yet — real experiments fall
   through to feed 2.
2. ``business_metrics`` rows with ``metric_type='per_hcp_rollup'`` (typed columns
   ``triggers_delivered_count``, ``triggers_accepted_count``,
   ``triggers_total_count`` (trigger funnel counts, migration 144),
   ``market_share``, ``conversion_rate``, ``engagement_score``,
   ``call_frequency``), joined on ``ab_experiment_assignments.unit_id`` ==
   ``business_metrics.hcp_id``, multiple ``metric_date`` rows per HCP collapsed
   to one scalar (SUM for counts, MEAN for rates). Unknown metrics fail closed
   here (``resolve_column``) exactly as before.

Both split by assignment variant into the (control, treatment) per-unit arrays
that ``ResultsAnalysisService._compute_results`` consumes (a pooled two-sample
test); nothing downstream knows which table fed it — the choice is logged at INFO
with the row counts so a worker log tells a reviewer which feed produced an ATE.

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
# Counts SUM across the window; rates MEAN. Unknown metrics fail closed.
# Canonical TRx lane / migration 144: the per-HCP counts are TRIGGER funnel
# counts. The prescription shorthands that used to map onto them are refused
# (see resolve_column) — no per-HCP prescription column exists.
_COUNT_COLUMNS = {"triggers_delivered_count", "triggers_accepted_count", "triggers_total_count"}
_RATE_COLUMNS = {"market_share", "conversion_rate", "engagement_score", "call_frequency"}
# The Digital Twin cohort OUTCOME (migration 147; src.data.per_hcp_cohort_columns
# .COHORT_OUTCOME_COLUMN). A draft experiment created from a twin proposal
# (#2206 item C) carries it as prediction_target so the final results measure the
# SAME quantity the twin predicted an effect on. Per-HCP numeric, not a count:
# the window collapses by MEAN.
_TWIN_OUTCOME_COLUMN = "cohort_conversion_outcome"
_PRESCRIPTION_SHORTHANDS = frozenset(
    {"trx", "nrx", "rx", "total_rx", "trx_count", "nrx_count", "total_rx_count"}
)

METRIC_COLUMN_MAP: Dict[str, str] = {
    "triggers_delivered": "triggers_delivered_count",
    "triggers_delivered_count": "triggers_delivered_count",
    "triggers_accepted": "triggers_accepted_count",
    "triggers_accepted_count": "triggers_accepted_count",
    "triggers_total": "triggers_total_count",
    "triggers_total_count": "triggers_total_count",
    "market_share": "market_share",
    "conversion_rate": "conversion_rate",
    "conversion": "conversion_rate",
    "engagement_score": "engagement_score",
    "engagement": "engagement_score",
    "call_frequency": "call_frequency",
    _TWIN_OUTCOME_COLUMN: _TWIN_OUTCOME_COLUMN,
}


#: Table of the per-experiment unit outcome feed (migration 155).
UNIT_OUTCOMES_TABLE = "ab_experiment_unit_outcomes"
#: PostgREST's default page / max-rows (1,000 on Supabase). BOTH reads below are
#: paged to exhaustion: an experiment can hold up to 1,400 units, and an un-ranged
#: read is capped at max-rows by the server — a truncated assignments read leaves
#: a 1,000-unit variant map that silently drops the other units' outcomes in
#: aggregate_to_arrays (codex r1 HIGH: measured ATE 0.0 vs the full 0.2857), a
#: truncated outcome page drops arms directly. Either is a plausible-wrong ATE.
_PAGE_SIZE = 1000
_MAX_PAGES = 1000


class ExperimentOutcomeRepository:
    """Loads real per-unit experiment outcomes: unit-outcome feed first, then
    the ``business_metrics`` per-HCP join (see the module docstring)."""

    def __init__(self, supabase_client: Any = None) -> None:
        self.client = supabase_client
        #: Which feed the last ``load_arrays`` call used ("unit_outcomes" /
        #: "business_metrics" / None before any call or on an early return).
        self.last_outcome_source: Optional[str] = None
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
        if key in _PRESCRIPTION_SHORTHANDS:
            raise ValueError(
                f"Unsupported primary_metric {primary_metric!r}: per-HCP business_metrics rows "
                "carry TRIGGER funnel counts (triggers_delivered_count / triggers_accepted_count / "
                "triggers_total_count, migration 144), not prescriptions. Name the trigger count "
                "you mean; prescription outcomes are not recorded per HCP."
            )
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
    @staticmethod
    def _page_to_exhaustion(
        build_query: Any, *, order_col: str, label: str
    ) -> List[Dict[str, Any]]:
        """Read every row of a filtered query with ``order_col``-ordered ``.range()``
        windows, advancing by the rows actually returned and stopping only on an
        EMPTY page (the ``fetch_synthetic_hcp_ids`` idiom in
        scripts/load_synthetic_data.py). ``build_query()`` must return a fresh
        filtered builder whose ``select`` asked for ``count="exact"``: the paged
        total is checked against the server's count and a short read fails LOUD
        (RuntimeError -> the Celery task reports ``failed``, never a truncated ATE).
        """
        rows: List[Dict[str, Any]] = []
        expected: Optional[int] = None
        offset = 0
        exhausted = False
        for _page in range(_MAX_PAGES):
            resp = build_query().order(order_col).range(offset, offset + _PAGE_SIZE - 1).execute()
            if expected is None:
                expected = getattr(resp, "count", None)
            page = resp.data or []
            if not page:
                exhausted = True
                break
            rows.extend(page)
            offset += len(page)
        if not exhausted:
            raise RuntimeError(
                f"{label}: paged read hit max_pages={_MAX_PAGES} before exhausting the rows"
            )
        if expected is not None and len(rows) != expected:
            raise RuntimeError(
                f"{label}: paged read returned {len(rows)} rows but the server reports "
                f"{expected} — refusing to feed a truncated arm to the pooled test"
            )
        return rows

    def load_unit_outcomes(
        self,
        experiment_id: UUID,
        metric_name: str,
        include_synthetic: bool = False,
    ) -> List[Dict[str, Any]]:
        """Read the experiment's rows from ``ab_experiment_unit_outcomes`` for
        ``metric_name`` (``unit_id, outcome_value, observed_at, is_synthetic``),
        paged to exhaustion with the exact-count check (see ``_page_to_exhaustion``).
        Provenance: the table is is_synthetic-tagged (migration 155) and the same
        ``include_synthetic`` opt-in as the assignments leg governs it.
        """
        if self.client is None:
            return []
        from src.repositories.provenance import apply_provenance_filter

        def _build() -> Any:
            q = (
                self.client.table(UNIT_OUTCOMES_TABLE)
                .select("unit_id,outcome_value,observed_at,is_synthetic", count="exact")
                .eq("experiment_id", str(experiment_id))
                .eq("metric_name", metric_name)
            )
            return apply_provenance_filter(q, include_synthetic)

        return self._page_to_exhaustion(
            _build, order_col="unit_id", label=f"{UNIT_OUTCOMES_TABLE}[{experiment_id}]"
        )

    async def load_arrays(
        self,
        experiment_id: UUID,
        primary_metric: str,
        *,
        brand: Optional[str] = None,
        window_days: Optional[int] = None,
        control_label: str = "control",
        treatment_label: str = "treatment",
        include_synthetic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Real outcome feed → (control, treatment) per-unit arrays.

        Precedence (option d1, 2026-09-23): after the (paged) assignments query,
        the per-experiment unit outcome feed (``ab_experiment_unit_outcomes``,
        keyed on the experiment and ``primary_metric`` as ``metric_name``) is read
        first; with >= 1 row the arrays come from it (MEAN per unit) and
        ``business_metrics`` is NOT queried. On this feed ``window_days`` is a
        PER-UNIT post-assignment window at timestamp precision: keep an outcome
        iff ``assigned_at <= observed_at <= assigned_at + window_days`` (the feed
        carries both timestamps; anchoring on the experiment's newest outcome
        would select recent ENROLLEES under rolling enrollment, not outcomes
        observed within each unit's window). With 0 rows the legacy path runs
        unchanged: ``resolve_column`` (fail closed on an unknown metric),
        assignments ⋈ ``business_metrics`` ``per_hcp_rollup`` (+ brand),
        ``window_days`` anchored on the newest ``metric_date`` (no per-assignment
        window is possible there). The feed used and the row counts are logged
        at INFO and recorded in ``self.last_outcome_source``.

        Completeness (codex r3 HIGH): the weekly ``--refresh-ab`` purges and
        reloads non-transactionally and the loader tolerates a failed batch, so a
        generator-written feed can be INCOMPLETE for an experiment while every
        row it does hold is fetched. The generator's contract is one row per
        assignment: when the feed for the experiment is synthetic
        (``is_synthetic`` rows) its unit set must EQUAL the assigned unit set or
        this raises ``RuntimeError`` (the task reports ``failed``) — never an ATE
        over a subset. A real feed (``is_synthetic=false``) may legitimately lack
        outcomes for some units (lost to follow-up); it proceeds on the observed
        units, as the legacy feed drops unit-less rows, with the coverage logged
        at WARNING.

        Returns empty arrays (caller bails ``insufficient_data``) when there are no
        assignments or no matching outcome rows on either feed.

        ``include_synthetic`` defaults to False so a real experiment's pooled test
        is never polluted by synthetic rows on either feed; validation runs opt in.
        """
        self.last_outcome_source = None
        if self.client is None:
            # Fail closed on an unmappable metric exactly as before (no feed can
            # be consulted without a client).
            self.resolve_column(primary_metric)
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        from src.repositories.provenance import apply_provenance_filter

        # 1) assignments for the experiment (unit_id, variant, assigned_at), paged
        # to exhaustion (up to 1,400 units; an un-ranged read is capped at 1,000
        # by PostgREST and silently truncated the variant map — codex r1 HIGH).
        # The table is is_synthetic-tagged (migration 063) — without the
        # predicate a real experiment's pooled test could ingest synthetic units
        # (#894); the same include_synthetic opt-in governs every leg of the join.
        def _build_assignments() -> Any:
            q = (
                self.client.table("ab_experiment_assignments")
                .select("unit_id,variant,assigned_at", count="exact")
                .eq("experiment_id", str(experiment_id))
            )
            return apply_provenance_filter(q, include_synthetic)

        assign_rows = self._page_to_exhaustion(
            _build_assignments,
            order_col="unit_id",
            label=f"ab_experiment_assignments[{experiment_id}]",
        )
        assignments = [(r["unit_id"], r["variant"]) for r in assign_rows if r.get("unit_id")]

        # 2a) the per-experiment unit outcome feed (migration 155) — FIRST.
        metric_name = (primary_metric or "").strip()
        unit_rows: List[Dict[str, Any]] = []
        if assignments and metric_name:
            unit_rows = self.load_unit_outcomes(
                experiment_id, metric_name, include_synthetic=include_synthetic
            )
        if unit_rows:
            self._check_feed_coverage(experiment_id, metric_name, assignments, unit_rows)
            # Present the rows in aggregate_to_arrays' shape (hcp_id / column /
            # date) so the ONE aggregation is reused, not forked. One row per
            # unit by the UNIQUE key, so MEAN is an identity; it stays the
            # documented reducer should a later writer add repeat observations.
            rows = [
                {
                    "hcp_id": r.get("unit_id"),
                    "outcome_value": r.get("outcome_value"),
                    "observed_at": r.get("observed_at"),
                }
                for r in unit_rows
            ]
            if window_days is not None:
                assigned_at_by_unit = {
                    str(r["unit_id"]): r.get("assigned_at") for r in assign_rows if r.get("unit_id")
                }
                rows = self._filter_post_assignment_window(rows, assigned_at_by_unit, window_days)
            self.last_outcome_source = "unit_outcomes"
            logger.info(
                "load_arrays(%s, %r): outcome feed = unit_outcomes "
                "(%d assignments, %d unit outcome rows%s)",
                experiment_id,
                metric_name,
                len(assignments),
                len(rows),
                f", window_days={window_days}" if window_days is not None else "",
            )
            return self.aggregate_to_arrays(
                assignments,
                rows,
                column="outcome_value",
                reducer="mean",
                control_label=control_label,
                treatment_label=treatment_label,
            )

        # 2b) legacy path, unchanged: business_metrics per_hcp_rollup join.
        column, reducer = self.resolve_column(primary_metric)
        if not assignments:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        unit_ids = [uid for uid, _ in assignments]

        query = (
            self.client.table("business_metrics")
            .select(f"hcp_id,{column},metric_date,brand")
            .eq("metric_type", "per_hcp_rollup")
            .in_("hcp_id", unit_ids)
        )
        query = apply_provenance_filter(query, include_synthetic)
        if brand:
            query = query.eq("brand", brand)
        metric_res = query.execute()
        rows = metric_res.data or []

        # window filtering (optional; no A/B window in schema today)
        if window_days is not None:
            rows = self._filter_window(rows, window_days)

        self.last_outcome_source = "business_metrics"
        logger.info(
            "load_arrays(%s, %r): outcome feed = business_metrics "
            "(no unit outcome rows; %d assignments, %d per_hcp_rollup rows, column=%s%s)",
            experiment_id,
            metric_name,
            len(assignments),
            len(rows),
            column,
            f", window_days={window_days}" if window_days is not None else "",
        )
        return self.aggregate_to_arrays(
            assignments,
            rows,
            column=column,
            reducer=reducer,
            control_label=control_label,
            treatment_label=treatment_label,
        )

    @staticmethod
    def _check_feed_coverage(
        experiment_id: UUID,
        metric_name: str,
        assignments: Sequence[Tuple[str, str]],
        unit_rows: Sequence[Dict[str, Any]],
    ) -> None:
        """Enforce the one-row-per-assignment contract on a synthetic feed; warn on
        an incomplete real feed (see ``load_arrays``)."""
        assigned = {str(uid) for uid, _ in assignments}
        observed = {str(r.get("unit_id")) for r in unit_rows if r.get("unit_id") is not None}
        if observed == assigned:
            return
        missing = assigned - observed
        extra = observed - assigned
        synthetic_feed = any(bool(r.get("is_synthetic")) for r in unit_rows)
        detail = (
            f"{UNIT_OUTCOMES_TABLE}[{experiment_id}, {metric_name!r}]: outcomes cover "
            f"{len(assigned) - len(missing)} of {len(assigned)} assigned units"
            f"{f', {len(extra)} outcome unit(s) not assigned' if extra else ''}"
        )
        if synthetic_feed:
            raise RuntimeError(
                f"{detail} — the generator writes one outcome per assignment, so this feed "
                "is incomplete (refresh in progress or a failed load batch); refusing to "
                "compute an ATE over a subset"
            )
        logger.warning("%s — real feed; proceeding on the observed units", detail)

    @staticmethod
    def _filter_post_assignment_window(
        rows: List[Dict[str, Any]],
        assigned_at_by_unit: Dict[str, Any],
        window_days: int,
    ) -> List[Dict[str, Any]]:
        """Unit-outcome feed window: keep a row iff
        ``assigned_at <= observed_at <= assigned_at + window_days`` for ITS unit,
        at timestamp precision (both ISO-8601 timestamps; naive values are read
        as UTC). Rows whose unit has no parseable ``assigned_at`` or whose
        ``observed_at`` is unparseable are dropped — a window that cannot be
        evaluated must not admit the row.
        """
        from datetime import datetime, timedelta, timezone

        def _ts(v: Any) -> Optional[datetime]:
            if isinstance(v, datetime):
                d = v
            elif isinstance(v, str):
                try:
                    d = datetime.fromisoformat(v.replace("Z", "+00:00"))
                except ValueError:
                    return None
            else:
                return None
            return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)

        span = timedelta(days=window_days)
        kept: List[Dict[str, Any]] = []
        for r in rows:
            start = _ts(assigned_at_by_unit.get(str(r.get("hcp_id"))))
            observed = _ts(r.get("observed_at"))
            if start is None or observed is None:
                continue
            if start <= observed <= start + span:
                kept.append(r)
        return kept

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
