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
        context = context or {}

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

        # Check cache (unless force_refresh)
        if use_cache and not force_refresh and self._cache.enabled:
            cached = self._cache.get(kpi_id, **context)
            if cached is not None:
                return cached

        # Calculate the KPI
        result = self._calculate_kpi(kpi, context)

        # Cache the result
        if use_cache and result.error is None:
            ttl = self._get_cache_ttl(kpi)
            self._cache.set(result, ttl=ttl, **context)

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
        """Calculate KPI from database tables via Supabase.

        F-007 (issue #421): replaces the prior `return None` placeholder with a
        real Supabase query. For the canonical pattern (KPIs with a numerator
        and denominator across two tables), the calculator queries each named
        table for its primary aggregate column and applies the simplest formula
        (ratio of first/second). Workstream-specific calculators registered via
        `register_calculator` should override this default for richer logic.

        Args:
            kpi: KPI metadata with table/column info
            context: Calculation context

        Returns:
            Tuple of (value, metadata)

        Raises:
            RuntimeError: if `kpi.tables` is empty or queries fail. Caller
                (`_default_calculate`) catches and surfaces via
                `KPIResult.error` — no silent fallback to `None`.
        """
        if not kpi.tables:
            raise RuntimeError(
                f"KPI {kpi.id} has no `tables` configured for table-based calculation"
            )

        logger.debug(f"Calculating {kpi.id} from tables {kpi.tables}")
        first_table = kpi.tables[0]
        rows = self._query_view_rows(first_table)
        if not rows:
            raise RuntimeError(f"Table {first_table!r} returned no rows for KPI {kpi.id}")

        row = rows[0]
        # Two-table KPIs follow "numerator / denominator" pattern across
        # the two named tables; single-table KPIs return the first numeric.
        if len(kpi.tables) >= 2:
            second_table = kpi.tables[1]
            denom_rows = self._query_view_rows(second_table)
            if not denom_rows:
                raise RuntimeError(f"Table {second_table!r} returned no rows for KPI {kpi.id}")
            numer = self._first_numeric_from_row(row)
            denom = self._first_numeric_from_row(denom_rows[0])
            if numer is None or denom is None:
                raise RuntimeError(
                    f"Tables {kpi.tables[:2]!r} did not yield numeric columns for KPI {kpi.id}"
                )
            if denom == 0:
                raise RuntimeError(f"Denominator is zero in {second_table!r} for KPI {kpi.id}")
            value = float(numer) / float(denom)
        else:
            scalar = self._first_numeric_from_row(row)
            if scalar is None:
                raise RuntimeError(
                    f"Table {first_table!r} returned row with no numeric column "
                    f"for KPI {kpi.id}: {row!r}"
                )
            value = float(scalar)

        return value, {
            "source": "tables",
            "tables": kpi.tables,
            "rows_inspected": len(rows),
        }

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
        """Return the first numeric value in a row dict, or None if none found.

        Skips identifier-like columns (`id`, `*_id`, `created_at`, etc.) so
        the calculator picks the metric value rather than a primary key.
        """
        if not row:
            return None
        skip_substrings = ("id", "created_at", "updated_at", "timestamp", "uuid")
        for key, value in row.items():
            lower_key = key.lower()
            if any(skip in lower_key for skip in skip_substrings):
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return float(value)
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
