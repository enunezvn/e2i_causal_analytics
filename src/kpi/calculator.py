"""
KPI Calculator

Central calculation engine for on-demand KPI computation with
caching, causal library routing, and database integration.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from loguru import logger

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

    def register_calculator(
        self, workstream: Workstream, calculator: KPICalculatorBase
    ) -> None:
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
            kpis = [
                self._registry.get(kpi_id)
                for kpi_id in kpi_ids
                if self._registry.get(kpi_id) is not None
            ]
        else:
            kpis = self._registry.get_all()

        # Calculate each KPI
        for kpi in kpis:
            if kpi is not None:
                result = self.calculate(
                    kpi.id, use_cache=use_cache, context=context
                )
                batch.add_result(result)

        return batch

    def _calculate_kpi(
        self, kpi: KPIMetadata, context: dict[str, Any]
    ) -> KPIResult:
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
                error=str(e),
            )

    def _default_calculate(
        self, kpi: KPIMetadata, context: dict[str, Any]
    ) -> KPIResult:
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
                calculated_at=datetime.now(timezone.utc),
                metadata=metadata,
                causal_library_used=causal_library,
            )

        except Exception as e:
            logger.error(f"Default calculation failed for {kpi.id}: {e}")
            return KPIResult(
                kpi_id=kpi.id,
                error=str(e),
            )

    def _calculate_from_view(
        self, kpi: KPIMetadata, context: dict[str, Any]
    ) -> tuple[float | None, dict[str, Any]]:
        """Calculate KPI from a database view.

        Args:
            kpi: KPI metadata with view name
            context: Calculation context

        Returns:
            Tuple of (value, metadata)
        """
        # This is a placeholder - actual implementation will use Supabase client
        # Will be implemented when integrating with database
        logger.debug(f"Calculating {kpi.id} from view {kpi.view}")
        return None, {"source": "view", "view_name": kpi.view}

    def _calculate_from_tables(
        self, kpi: KPIMetadata, context: dict[str, Any]
    ) -> tuple[float | None, dict[str, Any]]:
        """Calculate KPI from database tables.

        Args:
            kpi: KPI metadata with table/column info
            context: Calculation context

        Returns:
            Tuple of (value, metadata)
        """
        # Placeholder - will be implemented for derived KPIs
        logger.debug(f"Calculating {kpi.id} from tables {kpi.tables}")
        return None, {"source": "tables", "tables": kpi.tables}

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
