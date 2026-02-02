"""Tests for KPI Calculator."""

from unittest.mock import Mock

import pytest

from src.kpi.calculator import KPICalculator, KPICalculatorBase
from src.kpi.models import (
    CalculationType,
    CausalLibrary,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    KPIThreshold,
    Workstream,
)
from src.kpi.registry import KPIRegistry


class MockCalculator(KPICalculatorBase):
    """Mock calculator for testing."""

    def __init__(self, value: float = 0.85):
        self.value = value
        self.calculate_called = False

    def calculate(self, kpi: KPIMetadata, context: dict | None = None) -> KPIResult:
        self.calculate_called = True
        return KPIResult(
            kpi_id=kpi.id,
            value=self.value,
            status=KPIStatus.GOOD,
        )

    def supports(self, kpi: KPIMetadata) -> bool:
        return True


class TestKPICalculator:
    """Tests for KPICalculator."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry."""
        registry = Mock(spec=KPIRegistry)
        kpi = KPIMetadata(
            id="WS1-DQ-001",
            name="Source Coverage - Patients",
            definition="Test KPI",
            formula="test",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS1_DATA_QUALITY,
            threshold=KPIThreshold(target=0.85, warning=0.70, critical=0.50),
        )
        registry.get.return_value = kpi
        registry.get_all.return_value = [kpi]
        registry.get_by_workstream.return_value = [kpi]
        return registry

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache."""
        cache = Mock()
        cache.enabled = False
        cache.get.return_value = None
        cache.set.return_value = True
        return cache

    @pytest.fixture
    def mock_router(self):
        """Create a mock router."""
        router = Mock()
        router.get_recommended_library.return_value = CausalLibrary.NONE
        return router

    @pytest.fixture
    def calculator(self, mock_registry, mock_cache, mock_router):
        """Create a calculator with mocks."""
        return KPICalculator(
            registry=mock_registry,
            cache=mock_cache,
            router=mock_router,
        )

    def test_calculate_kpi_not_found(self, calculator, mock_registry):
        """Test calculating non-existent KPI."""
        mock_registry.get.return_value = None

        result = calculator.calculate("NONEXISTENT")

        assert result.error == "KPI not found: NONEXISTENT"
        assert result.value is None

    def test_calculate_with_registered_calculator(self, calculator, mock_registry):
        """Test calculating with registered workstream calculator."""
        mock_calc = MockCalculator(value=0.90)
        calculator.register_calculator(Workstream.WS1_DATA_QUALITY, mock_calc)

        result = calculator.calculate("WS1-DQ-001")

        assert mock_calc.calculate_called
        assert result.value == 0.90
        assert result.status == KPIStatus.GOOD

    def test_calculate_uses_cache(self, calculator, mock_cache):
        """Test that calculation uses cache when available."""
        mock_cache.enabled = True
        cached_result = KPIResult(
            kpi_id="WS1-DQ-001",
            value=0.88,
            status=KPIStatus.GOOD,
            cached=True,
        )
        mock_cache.get.return_value = cached_result

        result = calculator.calculate("WS1-DQ-001")

        assert result.cached is True
        assert result.value == 0.88
        mock_cache.get.assert_called_once()

    def test_calculate_force_refresh_bypasses_cache(self, calculator, mock_cache):
        """Test force_refresh bypasses cache."""
        mock_cache.enabled = True
        mock_cache.get.return_value = None

        calculator.calculate("WS1-DQ-001", force_refresh=True)

        # Cache get should not be called with force_refresh
        mock_cache.get.assert_not_called()

    def test_calculate_batch(self, calculator, mock_registry):
        """Test batch calculation."""
        mock_calc = MockCalculator(value=0.85)
        calculator.register_calculator(Workstream.WS1_DATA_QUALITY, mock_calc)

        batch = calculator.calculate_batch(workstream=Workstream.WS1_DATA_QUALITY)

        assert batch.total_kpis == 1
        assert batch.successful == 1
        assert batch.failed == 0
        assert batch.workstream == Workstream.WS1_DATA_QUALITY

    def test_invalidate_cache(self, calculator, mock_cache):
        """Test cache invalidation."""
        mock_cache.enabled = True

        calculator.invalidate_cache("WS1-DQ-001")

        mock_cache.invalidate.assert_called_once_with("WS1-DQ-001")

    def test_get_kpi_metadata(self, calculator, mock_registry):
        """Test getting KPI metadata."""
        kpi = calculator.get_kpi_metadata("WS1-DQ-001")

        assert kpi is not None
        assert kpi.id == "WS1-DQ-001"
        mock_registry.get.assert_called_with("WS1-DQ-001")

    def test_list_kpis_by_workstream(self, calculator, mock_registry):
        """Test listing KPIs by workstream."""
        calculator.list_kpis(workstream=Workstream.WS1_DATA_QUALITY)

        mock_registry.get_by_workstream.assert_called_with(Workstream.WS1_DATA_QUALITY)

    def test_is_lower_better_detection(self, calculator):
        """Test detection of lower-is-better KPIs."""
        # Test KPIs where lower is better
        assert (
            calculator._is_lower_better(
                KPIMetadata(
                    id="test",
                    name="Brier Score",
                    definition="",
                    formula="",
                    calculation_type=CalculationType.DIRECT,
                    workstream=Workstream.WS1_MODEL_PERFORMANCE,
                )
            )
            is True
        )

        assert (
            calculator._is_lower_better(
                KPIMetadata(
                    id="test",
                    name="Feature Drift PSI",
                    definition="",
                    formula="",
                    calculation_type=CalculationType.DIRECT,
                    workstream=Workstream.WS1_MODEL_PERFORMANCE,
                )
            )
            is True
        )

        assert (
            calculator._is_lower_better(
                KPIMetadata(
                    id="test",
                    name="False Alert Rate",
                    definition="",
                    formula="",
                    calculation_type=CalculationType.DIRECT,
                    workstream=Workstream.WS2_TRIGGERS,
                )
            )
            is True
        )

        # Test KPIs where higher is better
        assert (
            calculator._is_lower_better(
                KPIMetadata(
                    id="test",
                    name="ROC-AUC",
                    definition="",
                    formula="",
                    calculation_type=CalculationType.DIRECT,
                    workstream=Workstream.WS1_MODEL_PERFORMANCE,
                )
            )
            is False
        )


class TestKPICalculatorCacheTTL:
    """Tests for cache TTL calculation."""

    @pytest.fixture
    def calculator(self):
        """Create calculator with minimal mocks."""
        return KPICalculator(
            registry=Mock(),
            cache=Mock(enabled=False),
            router=Mock(),
        )

    def test_cache_ttl_daily(self, calculator):
        """Test cache TTL for daily KPIs."""
        kpi = KPIMetadata(
            id="test",
            name="Test",
            definition="",
            formula="",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.WS1_DATA_QUALITY,
            frequency="daily",
        )
        assert calculator._get_cache_ttl(kpi) == 300  # 5 minutes

    def test_cache_ttl_weekly(self, calculator):
        """Test cache TTL for weekly KPIs."""
        kpi = KPIMetadata(
            id="test",
            name="Test",
            definition="",
            formula="",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.WS1_DATA_QUALITY,
            frequency="weekly",
        )
        assert calculator._get_cache_ttl(kpi) == 1800  # 30 minutes

    def test_cache_ttl_monthly(self, calculator):
        """Test cache TTL for monthly KPIs."""
        kpi = KPIMetadata(
            id="test",
            name="Test",
            definition="",
            formula="",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.WS1_DATA_QUALITY,
            frequency="monthly",
        )
        assert calculator._get_cache_ttl(kpi) == 3600  # 1 hour
