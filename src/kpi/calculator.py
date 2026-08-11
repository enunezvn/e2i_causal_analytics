"""
KPI Calculator

Central calculation engine for on-demand KPI computation with
caching, causal library routing, and database integration.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

from src.kpi.cache import KPICache
from src.kpi.models import (
    CausalLibrary,
    KPIBatchResult,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)
from src.kpi.registry import KPIRegistry, get_registry
from src.kpi.router import CausalLibraryRouter
from src.kpi.synthetic_mode import kpi_include_synthetic


class KPICalculatorBase(ABC):
    """Abstract base class for KPI calculators."""

    @abstractmethod
    def calculate(
        self,
        kpi: KPIMetadata,
        context: dict[str, Any] | None = None,
    ) -> KPIResult:
        """Calculate a single KPI.

        Args:
            kpi: KPI metadata
            context: Optional calculation context (date range, brand, etc.)

        Returns:
            KPIResult with calculated value
        """
        ...

    @abstractmethod
    def supports(self, kpi: KPIMetadata) -> bool:
        """Check if this calculator supports the given KPI.

        Args:
            kpi: KPI metadata to check

        Returns:
            True if this calculator can handle the KPI
        """
        ...


class KPICalculator:
    """Main KPI calculation orchestrator.

    Coordinates calculation using registry, cache, and causal routing.
    """

    def __init__(
        self,
        registry: KPIRegistry | None = None,
        cache: KPICache | None = None,
        router: CausalLibraryRouter | None = None,
        db_connection: Any | None = None,
    ):
        """Initialize the KPI calculator.

        Args:
            registry: KPI registry (uses singleton if None)
            cache: KPI cache (creates new instance if None)
            router: Causal library router (creates new if None)
            db_connection: Database connection for SQL-based calculations

        F-007 NOTE (#421): the per-workstream calculators in
        `src/kpi/calculators/` are intentionally NOT auto-registered here.
        Auto-registration was reverted in codex iter-3 audit because 5 of the
        6 calculators still contain hardcoded `0.0`/`0.5`/`1.0` numeric
        defaults that swallow Supabase/MLflow failures (e.g.,
        `ModelPerformanceCalculator` returns `ROC-AUC=0.5` when MLflow is
        unreachable). Wiring them up here would make those latent
        placeholders user-visible.

        The hardening of those calculators is tracked in #439
        (F-007-PhaseB). Once each calculator has been audited and either (a)
        propagates errors via `KPIResult.error` or (b) returns honest "no
        data" (None + error) instead of fabricated numbers, auto-registration
        can be re-introduced in a follow-up PR.

        Until then: callers can register specific calculators via
        `register_calculator(workstream, instance)` (the existing API), and
        unregistered workstreams fall through to `_default_calculate` →
        `_calculate_from_view` (real Supabase query for view-backed KPIs) or
        `_calculate_from_tables` (raises `NotImplementedError`, surfaced via
        `KPIResult.error`). No silent placeholder zeros.
        """
        self._registry = registry or get_registry()
        self._cache = cache or KPICache()
        self._router = router or CausalLibraryRouter()
        self._db = db_connection
        self._calculators: dict[Workstream, KPICalculatorBase] = {}

        logger.info("KPI Calculator initialized")

    def register_calculator(self, workstream: Workstream, calculator: KPICalculatorBase) -> None:
        """Register a calculator for a workstream.

        Args:
            workstream: The workstream this calculator handles
            calculator: The calculator instance
        """
        self._calculators[workstream] = calculator
        logger.debug(f"Registered calculator for {workstream}")

    def calculate(
        self,
        kpi_id: str,
        use_cache: bool = True,
        force_refresh: bool = False,
        context: dict[str, Any] | None = None,
    ) -> KPIResult:
        """Calculate a single KPI on-demand.

        Args:
            kpi_id: The KPI identifier
            use_cache: Whether to check cache first
            force_refresh: Force recalculation even if cached
            context: Optional calculation context

        Returns:
            KPIResult with calculated value and status
        """
        # Private per-call copy: registered calculators stash per-KPI facts
        # into the context (data_through, funnel_stages,
        # temporal_variability_band) and embed it in metadata BY REFERENCE.
        # Sharing the caller's dict let one KPI's stash leak into every other
        # result computed with the same context (calculate_batch passes ONE
        # dict — even results created EARLIER changed retroactively) and into
        # subsequent cache keys (#1532 codex audit).
        context = dict(context) if context else {}

        # Get KPI metadata
        kpi = self._registry.get(kpi_id)
        if kpi is None:
            return KPIResult(
                kpi_id=kpi_id,
                value=None,
                status=KPIStatus.UNKNOWN,
                cached=False,
                error=f"KPI not found: {kpi_id}",
            )

        # Synthetic-visibility mode (E2I_KPI_INCLUDE_SYNTHETIC) changes the
        # underlying SQL (base vs _include_synthetic twin), so it MUST be part of
        # the cache key -- otherwise a synthetic value could be served after the
        # flag is unset (or vice versa), breaking the reversible gate. We add it
        # to the cache context only (not the calculator context, which the
        # calculators echo into metadata["context"]).
        include_synthetic = kpi_include_synthetic()
        # The requested window changes the underlying SQL (base vs _windowed
        # twin) and therefore the value, so it MUST be part of the cache key --
        # otherwise a default-window value could be served for an explicit
        # window (or two different windows could collide). Keyed by the
        # (start, end) pair only (not the whole window dict) so the key is a
        # stable, hashable scalar.
        window = context.get("window")
        cache_context = {
            **context,
            "_include_synthetic": include_synthetic,
            "_window": (window.get("start"), window.get("end")) if window else None,
        }

        # Check cache (unless force_refresh)
        if use_cache and not force_refresh and self._cache.enabled:
            cached = self._cache.get(kpi_id, **cache_context)
            # Region provenance is serialized with the entry (unlike window,
            # whose truth is registry-derivable and re-stamped below). An entry
            # that PREDATES #1538 deserializes to region_status="default" while
            # a region was requested — it cannot attest whether the region was
            # applied, so recompute rather than serve an unattested value for
            # one TTL cycle.
            if cached is not None and not (
                context.get("region") and cached.region_status == "default"
            ):
                # A cached entry was computed for THIS window (window is part of
                # the cache key), so reflect the requested window on the served
                # result rather than the cache's serialized "default".
                return self._stamp_window(cached, kpi, window)

        # Calculate the KPI
        result = self._calculate_kpi(kpi, context)
        # Provenance travels with the (cached) result so the API/FE can label a
        # synthetic-sourced figure honestly rather than passing it off as real.
        result.metadata["include_synthetic"] = include_synthetic
        # Window provenance is generic across ALL KPIs and both the
        # registered-calculator and default paths, so stamp it here (the single
        # place the successful result is produced for calculate() to return).
        result = self._stamp_window(result, kpi, window)
        # Region provenance (#1538): the routing seams mark the context when
        # they select a region-scoped variant; stamp the verdict here, BEFORE
        # caching, so the serialized entry attests it.
        result = self._stamp_region(result, context)

        # Cache the result
        if use_cache and result.error is None:
            ttl = self._get_cache_ttl(kpi)
            self._cache.set(result, ttl=ttl, **cache_context)

        return result

    @staticmethod
    def _stamp_window(
        result: KPIResult, kpi: KPIMetadata, window: dict[str, Any] | None
    ) -> KPIResult:
        """Stamp window provenance on a result from the KPI's windowable class.

        Generic across every KPI (registered-calculator path AND default path):

        * ``windowable in {"clean", "needs_care"}`` + a requested window ->
          ``window_status="applied"``; both requested and applied carry the
          window. The value itself is left as computed (the windowed SQL already
          time-bounded it).
        * ``windowable == "not_applicable"`` (or anything else) + a requested
          window -> ``window_status="not_applicable"``; the window is recorded as
          requested but NOT applied, and the value is kept (the KPI is a
          snapshot/ML metric with no claims time-dimension -- ignore the window
          honestly rather than erroring or fabricating).
        * No requested window -> the model defaults are left untouched
          (``window_status="default"``).
        """
        if not window:
            return result
        if kpi.windowable in ("clean", "needs_care"):
            result.window_requested = window
            result.window_applied = window
            result.window_status = "applied"
        else:
            result.window_requested = window
            result.window_applied = None
            result.window_status = "not_applicable"
        return result

    @staticmethod
    def _stamp_region(result: KPIResult, context: dict[str, Any]) -> KPIResult:
        """Stamp region provenance (#1538) from the routing seams' marker.

        Unlike the window stamp there is no registry attribute to consult:
        whether a region was applied is a ROUTING fact (a handful of
        calculators select ``*_region``/``*_brand_region`` variants, with
        per-method precedence rules that can silently drop the region). Each
        seam sets ``context["_region_routed"] = True`` at the exact decision
        point, so this stamp is truthful by construction:

        * no region requested          -> defaults untouched ("default")
        * region + marker              -> "applied" (requested == applied)
        * region + no marker           -> "not_applicable"; the value is kept
          but is global/portfolio-level — consumers must not caption it with
          the region. Errored results also land here (consumers read ``error``
          first; the requested region is still echoed for the record).
        """
        region = context.get("region")
        if not region:
            return result
        result.region_requested = region
        if context.get("_region_routed"):
            result.region_applied = region
            result.region_status = "applied"
        else:
            result.region_applied = None
            result.region_status = "not_applicable"
        return result

    def calculate_batch(
        self,
        kpi_ids: list[str] | None = None,
        workstream: Workstream | None = None,
        use_cache: bool = True,
        context: dict[str, Any] | None = None,
    ) -> KPIBatchResult:
        """Calculate multiple KPIs.

        Args:
            kpi_ids: List of KPI IDs to calculate (None for all)
            workstream: Calculate all KPIs for a workstream
            use_cache: Whether to use caching
            context: Calculation context

        Returns:
            KPIBatchResult with all results
        """
        batch = KPIBatchResult(workstream=workstream)

        # Determine which KPIs to calculate
        if workstream is not None:
            kpis = self._registry.get_by_workstream(workstream)
        elif kpi_ids is not None:
            kpis_raw = [self._registry.get(kpi_id) for kpi_id in kpi_ids]
            kpis = [k for k in kpis_raw if k is not None]
        else:
            kpis = self._registry.get_all()

        # Calculate each KPI
        for kpi in kpis:
            if kpi is not None:
                result = self.calculate(kpi.id, use_cache=use_cache, context=context)
                batch.add_result(result)

        return batch

    def _calculate_kpi(self, kpi: KPIMetadata, context: dict[str, Any]) -> KPIResult:
        """Internal KPI calculation logic.

        Args:
            kpi: KPI metadata
            context: Calculation context

        Returns:
            KPIResult with calculated value
        """
        try:
            # Get calculator for this workstream
            calculator = self._calculators.get(kpi.workstream)

            if calculator is not None and calculator.supports(kpi):
                return calculator.calculate(kpi, context)

            # Fallback to default calculation
            return self._default_calculate(kpi, context)

        except Exception as e:
            logger.error(f"KPI calculation failed for {kpi.id}: {e}")
            return KPIResult(
                kpi_id=kpi.id,
                value=None,
                status=KPIStatus.UNKNOWN,
                cached=False,
                error=str(e),
            )

    def _default_calculate(self, kpi: KPIMetadata, context: dict[str, Any]) -> KPIResult:
        """Default calculation using database views or direct SQL.

        Args:
            kpi: KPI metadata
            context: Calculation context

        Returns:
            KPIResult with calculated value
        """
        if self._db is None:
            return KPIResult(
                kpi_id=kpi.id,
                value=None,
                status=KPIStatus.UNKNOWN,
                cached=False,
                error="No database connection for default calculation",
            )

        try:
            value = None
            metadata: dict[str, Any] = {}

            # If KPI has a dedicated view, use it
            if kpi.view:
                value, metadata = self._calculate_from_view(kpi, context)
            else:
                # Calculate from tables directly
                value, metadata = self._calculate_from_tables(kpi, context)

            # Evaluate against thresholds
            status = KPIStatus.UNKNOWN
            if kpi.threshold and value is not None:
                lower_is_better = self._is_lower_better(kpi)
                status = kpi.threshold.evaluate(value, lower_is_better)

            # Determine causal library used
            causal_library = self._router.get_recommended_library(kpi)

            return KPIResult(
                kpi_id=kpi.id,
                value=value,
                status=status,
                cached=False,
                error=None,
                calculated_at=datetime.now(timezone.utc),
                metadata=metadata,
                causal_library_used=causal_library,
            )

        except Exception as e:
            logger.error(f"Default calculation failed for {kpi.id}: {e}")
            return KPIResult(
                kpi_id=kpi.id,
                value=None,
                status=KPIStatus.UNKNOWN,
                cached=False,
                error=str(e),
            )

    def _calculate_from_view(
        self, kpi: KPIMetadata, context: dict[str, Any]
    ) -> tuple[float | None, dict[str, Any]]:
        """Calculate KPI from a database view via Supabase.

        F-007 (issue #421): replaces the prior `return None` placeholder with a
        real Supabase query. The view name on the KPI metadata is queried via
        the PostgREST `client.table(view_name).select(...).execute()` chain;
        the first row's first numeric column is returned as the scalar value.

        Args:
            kpi: KPI metadata with view name
            context: Calculation context (unused in this minimal delegator;
                workstream-specific calculators registered via
                `register_calculator` apply context-aware queries).

        Returns:
            Tuple of (value, metadata) where `value` is the scalar drawn from
            the view's first row.

        Raises:
            RuntimeError: if `kpi.view` is unset, or the view returns no rows,
                or no numeric column is found in the first row. Caller
                (`_default_calculate`) catches and surfaces via
                `KPIResult.error` — no silent fallback to `None`.
        """
        view_name = kpi.view
        if not view_name:
            raise RuntimeError(f"KPI {kpi.id} has no `view` configured for view-based calculation")

        logger.debug(f"Calculating {kpi.id} from view {view_name}")
        rows = self._query_view_rows(view_name)
        if not rows:
            raise RuntimeError(f"View {view_name!r} returned no rows for KPI {kpi.id}")

        value = self._first_numeric_from_row(rows[0])
        if value is None:
            raise RuntimeError(
                f"View {view_name!r} returned a row with no numeric column "
                f"for KPI {kpi.id}: {rows[0]!r}"
            )
        return float(value), {
            "source": "view",
            "view_name": view_name,
            "row_count": len(rows),
        }

    def _calculate_from_tables(
        self, kpi: KPIMetadata, context: dict[str, Any]
    ) -> tuple[float | None, dict[str, Any]]:
        """Default fallback for table-derived KPIs without a registered calculator.

        F-007 (issue #421): table-derived KPIs require formula evaluation
        (e.g., `covered_patients / reference_patients`). The honest place for
        that logic is the per-workstream calculators in `src/kpi/calculators/`
        (e.g., `DataQualityCalculator._calc_source_coverage_patients` runs the
        actual joined SQL). Callers register those via the existing
        `register_calculator(workstream, instance)` API after auditing the
        specific calculator (see #439 / F-007-PhaseB for the hardening work).

        This fallback is hit when no workstream calculator is registered for
        the KPI's workstream. Rather than guess a generic "first numeric /
        first numeric" formula that mis-evaluates real KPIs (covered/reference
        is NOT row[0]/row[0] across two unrelated tables), this method raises
        `NotImplementedError` — surfaced via `KPIResult.error` by the caller.
        Silent fallback to `None` (or any fabricated default like 0.0) would
        re-introduce the placeholder pattern this PR is retiring.

        Args:
            kpi: KPI metadata with table/column info
            context: Calculation context

        Returns:
            Never returns — always raises.

        Raises:
            NotImplementedError: always. Register a per-workstream calculator
                or, for the immediate fix, surface the KPI via a Supabase view
                (`_calculate_from_view`).
        """
        raise NotImplementedError(
            f"Table-derived KPI {kpi.id} ({kpi.workstream}) has no registered "
            f"workstream calculator. Generic table-formula evaluation is not "
            f"implemented (see #421); register a per-workstream calculator in "
            f"`src/kpi/calculators/` or surface the KPI via a Supabase view."
        )

    def _query_view_rows(self, table_or_view_name: str) -> list[dict[str, Any]]:
        """Query a Supabase view or table and return rows as dicts.

        Uses the PostgREST chain `client.table(name).select('*').execute()`.

        Args:
            table_or_view_name: Name of the view or table to query.

        Returns:
            List of row dicts. Empty list if the source has no rows.

        Raises:
            RuntimeError: if no DB client is configured.
            Exception: if the underlying client raises (network, auth,
                missing-view, etc.) — propagated so the caller surfaces
                via `KPIResult.error`. No silent fallback to None.
        """
        if self._db is None:
            raise RuntimeError(
                f"No Supabase client configured; cannot query {table_or_view_name!r}"
            )
        response = self._db.table(table_or_view_name).select("*").limit(1).execute()
        data = getattr(response, "data", None)
        if data is None:
            return []
        return list(data)

    @staticmethod
    def _first_numeric_from_row(row: dict[str, Any]) -> float | None:
        """Return the canonical numeric value in a row dict, or None.

        Prefers canonical KPI column names in order: `value`, `kpi_value`,
        `score`, `rate`, `lift`, `match_rate`. Falls back to the first
        non-identifier numeric column only when no canonical name is present
        AND exactly one numeric column exists (to avoid ambiguous picks like
        "row has 3 numerics — which is the KPI?").

        F-007 (#421): the prior fallback ("first numeric of anything") was a
        wrong-but-passing heuristic for KPI views with multiple aggregate
        columns. The stricter rule: if a view has multiple numeric columns
        and none match a canonical name, return None — surfaced as an error
        rather than guessing.
        """
        if not row:
            return None

        # Tier 1: canonical KPI scalar column names.
        canonical_keys = (
            "value",
            "kpi_value",
            "score",
            "lift_score",
            "lift",
            "match_rate",
            "rate",
            "pass_rate",
            "consistency_rate",
            "avg_ttr_hours",  # #580: TTR registry row returns hours (was median_ttr_days)
            "median_lag_days",
        )
        for canonical in canonical_keys:
            if canonical in row:
                candidate = row[canonical]
                if isinstance(candidate, bool):
                    continue
                if isinstance(candidate, (int, float)):
                    return float(candidate)

        # Tier 2: non-identifier numeric column — only if exactly one exists.
        skip_substrings = ("id", "created_at", "updated_at", "timestamp", "uuid")
        numeric_candidates: list[float] = []
        for key, value in row.items():
            lower_key = key.lower()
            if any(skip in lower_key for skip in skip_substrings):
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric_candidates.append(float(value))
        if len(numeric_candidates) == 1:
            return numeric_candidates[0]
        # Ambiguous (multiple numerics, no canonical) → caller must surface error.
        return None

    def _get_cache_ttl(self, kpi: KPIMetadata) -> int:
        """Determine cache TTL based on KPI frequency.

        Args:
            kpi: KPI metadata

        Returns:
            TTL in seconds
        """
        frequency_ttl = {
            "realtime": 60,  # 1 minute
            "daily": 300,  # 5 minutes
            "weekly": 1800,  # 30 minutes
            "monthly": 3600,  # 1 hour
            "on_demand": 600,  # 10 minutes
        }
        return frequency_ttl.get(kpi.frequency, 300)

    def _is_lower_better(self, kpi: KPIMetadata) -> bool:
        """Determine if lower values are better for this KPI.

        Args:
            kpi: KPI metadata

        Returns:
            True if lower values are better
        """
        lower_better_patterns = [
            "error",
            "lag",
            "fail",
            "drift",
            "gap",
            "brier",
            "false",
        ]
        name_lower = kpi.name.lower()
        return any(pattern in name_lower for pattern in lower_better_patterns)

    def invalidate_cache(
        self,
        kpi_id: str | None = None,
        workstream: Workstream | None = None,
    ) -> int:
        """Invalidate cached KPI results.

        Args:
            kpi_id: Specific KPI to invalidate (None for workstream or all)
            workstream: Invalidate all KPIs for a workstream

        Returns:
            Number of cache entries invalidated
        """
        if kpi_id:
            self._cache.invalidate(kpi_id)
            return 1

        if workstream:
            kpis = self._registry.get_by_workstream(workstream)
            for kpi in kpis:
                self._cache.invalidate(kpi.id)
            return len(kpis)

        return self._cache.invalidate_all()

    def get_kpi_metadata(self, kpi_id: str) -> KPIMetadata | None:
        """Get metadata for a KPI.

        Args:
            kpi_id: KPI identifier

        Returns:
            KPIMetadata or None if not found
        """
        return self._registry.get(kpi_id)

    def list_kpis(
        self,
        workstream: Workstream | None = None,
        causal_library: CausalLibrary | None = None,
    ) -> list[KPIMetadata]:
        """List available KPIs.

        Args:
            workstream: Filter by workstream
            causal_library: Filter by causal library

        Returns:
            List of KPI metadata
        """
        if workstream:
            return self._registry.get_by_workstream(workstream)
        if causal_library:
            return self._registry.get_by_causal_library(causal_library)
        return self._registry.get_all()
