"""Tests for WS2, WS3, Brand-Specific, and Causal Metrics KPI Calculators."""

from unittest.mock import Mock

import pytest

from src.kpi.calculators.brand_specific import BrandSpecificCalculator
from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.calculators.causal_metrics import CausalMetricsCalculator
from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIStatus,
    KPIThreshold,
    Workstream,
)


class TestTriggerPerformanceCalculator:
    """Tests for TriggerPerformanceCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with mock db client."""
        mock_db = Mock()
        return TriggerPerformanceCalculator(db_client=mock_db)

    @pytest.fixture
    def precision_kpi(self):
        """Create a trigger precision KPI."""
        return KPIMetadata(
            id="WS2-TR-001",
            name="Trigger Precision",
            definition="Percentage of fired triggers resulting in positive outcome",
            formula="true_positives / (true_positives + false_positives)",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS2_TRIGGERS,
            threshold=KPIThreshold(target=0.70, warning=0.55, critical=0.40),
        )

    @pytest.fixture
    def lead_time_kpi(self):
        """Create a lead time KPI (lower is better)."""
        return KPIMetadata(
            id="WS2-TR-007",
            name="Lead Time",
            definition="Median days between trigger and outcome",
            formula="median(outcome_date - trigger_date)",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.WS2_TRIGGERS,
            threshold=KPIThreshold(target=14, warning=21, critical=30),
        )

    def test_supports_trigger_workstream(self, calculator, precision_kpi):
        """Test calculator supports WS2 Triggers KPIs."""
        assert calculator.supports(precision_kpi) is True

    def test_does_not_support_other_workstreams(self, calculator):
        """Test calculator doesn't support other workstreams."""
        kpi = KPIMetadata(
            id="WS1-DQ-001",
            name="Source Coverage",
            definition="Test",
            formula="test",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS1_DATA_QUALITY,
        )
        assert calculator.supports(kpi) is False

    def test_calculate_trigger_precision_good(self, calculator, precision_kpi):
        """Test trigger precision calculation with good result."""
        calculator._execute_query = Mock(return_value=[{"precision": 0.75}])

        result = calculator.calculate(precision_kpi)

        assert result.value == 0.75
        assert result.status == KPIStatus.GOOD
        assert result.error is None

    def test_calculate_trigger_precision_warning(self, calculator, precision_kpi):
        """Test trigger precision in warning zone."""
        calculator._execute_query = Mock(return_value=[{"precision": 0.58}])

        result = calculator.calculate(precision_kpi)

        assert result.value == 0.58
        assert result.status == KPIStatus.WARNING

    def test_calculate_lead_time_good(self, calculator, lead_time_kpi):
        """Test lead time with good result (lower is better)."""
        calculator._execute_query = Mock(return_value=[{"median_lead_time": 10}])

        result = calculator.calculate(lead_time_kpi)

        assert result.value == 10.0
        assert result.status == KPIStatus.GOOD
        assert result.metadata.get("lower_is_better") is True

    def test_calculate_lead_time_critical(self, calculator, lead_time_kpi):
        """Test lead time in critical zone (too high)."""
        calculator._execute_query = Mock(return_value=[{"median_lead_time": 35}])

        result = calculator.calculate(lead_time_kpi)

        assert result.value == 35.0
        assert result.status == KPIStatus.CRITICAL

    def test_calculate_unknown_kpi_returns_error(self, calculator):
        """Test calculator returns error for unknown KPI ID."""
        kpi = KPIMetadata(
            id="WS2-TR-999",
            name="Unknown KPI",
            definition="Test",
            formula="test",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS2_TRIGGERS,
        )
        result = calculator.calculate(kpi)
        assert result.error is not None
        assert "No calculator implemented" in result.error


class TestBusinessImpactCalculator:
    """Tests for BusinessImpactCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with mock db client."""
        mock_db = Mock()
        return BusinessImpactCalculator(db_client=mock_db)

    @pytest.fixture
    def mau_kpi(self):
        """Create a MAU KPI."""
        return KPIMetadata(
            id="WS3-BI-001",
            name="Monthly Active Users",
            definition="Unique users with at least one session in past 30 days",
            formula="count(distinct user_id)",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.WS3_BUSINESS,
            threshold=KPIThreshold(target=2000, warning=1500, critical=1000),
        )

    @pytest.fixture
    def trx_kpi(self):
        """Create a TRx KPI (no threshold - volume metric)."""
        return KPIMetadata(
            id="WS3-BI-005",
            name="Total Prescriptions (TRx)",
            definition="Total prescription volume",
            formula="count(prescriptions)",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS3_BUSINESS,
            threshold=None,
        )

    @pytest.fixture
    def conversion_kpi(self):
        """Create a conversion rate KPI."""
        return KPIMetadata(
            id="WS3-BI-009",
            name="Conversion Rate",
            definition="Percentage of triggers resulting in prescription",
            formula="prescriptions / triggers",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS3_BUSINESS,
            threshold=KPIThreshold(target=0.08, warning=0.05, critical=0.02),
        )

    def test_supports_business_workstream(self, calculator, mau_kpi):
        """Test calculator supports WS3 Business KPIs."""
        assert calculator.supports(mau_kpi) is True

    def test_calculate_mau_good(self, calculator, mau_kpi):
        """Test MAU calculation with good result."""
        calculator._execute_query = Mock(return_value=[{"mau": 2500}])

        result = calculator.calculate(mau_kpi)

        assert result.value == 2500.0
        assert result.status == KPIStatus.GOOD

    def test_calculate_mau_critical(self, calculator, mau_kpi):
        """Test MAU in critical zone."""
        calculator._execute_query = Mock(return_value=[{"mau": 800}])

        result = calculator.calculate(mau_kpi)

        assert result.value == 800.0
        assert result.status == KPIStatus.CRITICAL

    def test_calculate_trx_no_threshold(self, calculator, trx_kpi):
        """Test TRx returns UNKNOWN status (volume metric)."""
        calculator._execute_query = Mock(return_value=[{"trx": 15000}])

        result = calculator.calculate(trx_kpi)

        assert result.value == 15000.0
        assert result.status == KPIStatus.UNKNOWN

    def test_calculate_conversion_rate_good(self, calculator, conversion_kpi):
        """Test conversion rate with good result."""
        calculator._execute_query = Mock(return_value=[{"conversion_rate": 0.10}])

        result = calculator.calculate(conversion_kpi)

        assert result.value == 0.10
        assert result.status == KPIStatus.GOOD


class TestBrandSpecificCalculator:
    """Tests for BrandSpecificCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with mock db client."""
        mock_db = Mock()
        return BrandSpecificCalculator(db_client=mock_db)

    @pytest.fixture
    def remi_uncontrolled_kpi(self):
        """Create Remi AH Uncontrolled KPI (lower is better)."""
        return KPIMetadata(
            id="BR-001",
            name="Remi - AH Uncontrolled %",
            definition="Percentage of antihistamine patients with uncontrolled symptoms",
            formula="uncontrolled / ah_patients",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.BRAND_SPECIFIC,
            threshold=KPIThreshold(target=0.40, warning=0.50, critical=0.60),
        )

    @pytest.fixture
    def kisqali_adoption_kpi(self):
        """Create Kisqali Dx Adoption KPI (lower is better - days)."""
        return KPIMetadata(
            id="BR-004",
            name="Kisqali - Dx Adoption",
            definition="Median days from diagnosis to first Kisqali prescription",
            formula="median(first_rx - dx_date)",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.BRAND_SPECIFIC,
            threshold=KPIThreshold(target=30, warning=45, critical=60),
        )

    def test_supports_brand_specific_workstream(self, calculator, remi_uncontrolled_kpi):
        """Test calculator supports Brand-Specific KPIs."""
        assert calculator.supports(remi_uncontrolled_kpi) is True

    def test_calculate_remi_uncontrolled_good(self, calculator, remi_uncontrolled_kpi):
        """Test Remi uncontrolled with good result (lower is better)."""
        calculator._execute_query = Mock(return_value=[{"uncontrolled_pct": 0.35}])

        result = calculator.calculate(remi_uncontrolled_kpi)

        assert result.value == 0.35
        assert result.status == KPIStatus.GOOD
        assert result.metadata.get("lower_is_better") is True

    def test_calculate_remi_uncontrolled_critical(self, calculator, remi_uncontrolled_kpi):
        """Test Remi uncontrolled in critical zone (too high)."""
        calculator._execute_query = Mock(return_value=[{"uncontrolled_pct": 0.65}])

        result = calculator.calculate(remi_uncontrolled_kpi)

        assert result.value == 0.65
        assert result.status == KPIStatus.CRITICAL

    def test_calculate_kisqali_adoption_good(self, calculator, kisqali_adoption_kpi):
        """Test Kisqali adoption with good result (lower days is better)."""
        calculator._execute_query = Mock(return_value=[{"median_days": 25}])

        result = calculator.calculate(kisqali_adoption_kpi)

        assert result.value == 25.0
        assert result.status == KPIStatus.GOOD


class TestCausalMetricsCalculator:
    """Tests for CausalMetricsCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with mock db client."""
        mock_db = Mock()
        return CausalMetricsCalculator(db_client=mock_db)

    @pytest.fixture
    def ate_kpi(self):
        """Create ATE KPI."""
        return KPIMetadata(
            id="CM-001",
            name="Average Treatment Effect",
            definition="E[Y(1) - Y(0)]",
            formula="E[Y(1) - Y(0)]",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.CAUSAL_METRICS,
            threshold=None,
        )

    @pytest.fixture
    def cate_kpi(self):
        """Create CATE KPI."""
        return KPIMetadata(
            id="CM-002",
            name="Conditional ATE",
            definition="E[Y(1) - Y(0) | X=x]",
            formula="E[Y(1) - Y(0) | X=x]",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.CAUSAL_METRICS,
            threshold=None,
        )

    def test_supports_causal_metrics_workstream(self, calculator, ate_kpi):
        """Test calculator supports Causal Metrics KPIs."""
        assert calculator.supports(ate_kpi) is True

    def test_calculate_ate_from_predictions(self, calculator, ate_kpi):
        """Test ATE calculation from ml_predictions."""
        calculator._execute_query = Mock(
            return_value=[{"ate": 0.15, "ate_std": 0.05, "n_samples": 1000}]
        )

        result = calculator.calculate(ate_kpi)

        assert result.value == 0.15
        assert result.status == KPIStatus.UNKNOWN  # No threshold for causal
        assert result.metadata.get("ate_std") == 0.05
        assert "ci_lower" in result.metadata
        assert "ci_upper" in result.metadata

    def test_calculate_ate_returns_none_when_no_data(self, calculator, ate_kpi):
        """Test ATE returns None when no data available."""
        calculator._execute_query = Mock(return_value=[{"ate": None}])

        result = calculator.calculate(ate_kpi)

        assert result.value is None
        assert "error" in result.metadata

    def test_calculate_cate_with_segment_breakdown(self, calculator, cate_kpi):
        """Test CATE calculation returns segment breakdown."""
        calculator._execute_query = Mock(
            return_value=[
                {
                    "segment_assignment": "high_risk",
                    "cate": 0.25,
                    "cate_std": 0.08,
                    "n_samples": 300,
                },
                {
                    "segment_assignment": "medium_risk",
                    "cate": 0.12,
                    "cate_std": 0.05,
                    "n_samples": 500,
                },
                {
                    "segment_assignment": "low_risk",
                    "cate": 0.05,
                    "cate_std": 0.02,
                    "n_samples": 200,
                },
            ]
        )

        result = calculator.calculate(cate_kpi)

        assert result.value is not None
        assert "segment_breakdown" in result.metadata
        assert len(result.metadata["segment_breakdown"]) == 3

    def test_calculate_cate_for_specific_segment(self, calculator, cate_kpi):
        """Test CATE calculation for specific segment."""
        calculator._execute_query = Mock(
            return_value=[
                {
                    "segment_assignment": "high_risk",
                    "cate": 0.25,
                    "cate_std": 0.08,
                    "n_samples": 300,
                }
            ]
        )

        result = calculator.calculate(cate_kpi, context={"segment": "high_risk"})

        assert result.value == 0.25
        assert result.metadata.get("segment") == "high_risk"


class TestCalculatorIntegration:
    """Integration tests across all calculators."""

    def test_all_calculators_importable(self):
        """Test all calculators can be imported."""
        from src.kpi.calculators import (
            BrandSpecificCalculator,
            BusinessImpactCalculator,
            CausalMetricsCalculator,
            DataQualityCalculator,
            ModelPerformanceCalculator,
            TriggerPerformanceCalculator,
        )

        assert BrandSpecificCalculator is not None
        assert BusinessImpactCalculator is not None
        assert CausalMetricsCalculator is not None
        assert DataQualityCalculator is not None
        assert ModelPerformanceCalculator is not None
        assert TriggerPerformanceCalculator is not None

    def test_workstream_calculator_mapping(self):
        """Test each workstream has a corresponding calculator."""
        from src.kpi.calculators import (
            BrandSpecificCalculator,
            BusinessImpactCalculator,
            CausalMetricsCalculator,
            DataQualityCalculator,
            ModelPerformanceCalculator,
            TriggerPerformanceCalculator,
        )

        mock_db = Mock()

        # Create instances
        calculators = [
            DataQualityCalculator(db_client=mock_db),
            ModelPerformanceCalculator(db_client=mock_db),
            TriggerPerformanceCalculator(db_client=mock_db),
            BusinessImpactCalculator(db_client=mock_db),
            BrandSpecificCalculator(db_client=mock_db),
            CausalMetricsCalculator(db_client=mock_db),
        ]

        # Map workstreams
        workstream_map = {
            Workstream.WS1_DATA_QUALITY: False,
            Workstream.WS1_MODEL_PERFORMANCE: False,
            Workstream.WS2_TRIGGERS: False,
            Workstream.WS3_BUSINESS: False,
            Workstream.BRAND_SPECIFIC: False,
            Workstream.CAUSAL_METRICS: False,
        }

        for ws in workstream_map:
            test_kpi = KPIMetadata(
                id=f"TEST-{ws.value}",
                name="Test KPI",
                definition="Test",
                formula="test",
                calculation_type=CalculationType.DERIVED,
                workstream=ws,
            )
            for calc in calculators:
                if calc.supports(test_kpi):
                    workstream_map[ws] = True
                    break

        # All workstreams should have a calculator
        for ws, has_calculator in workstream_map.items():
            assert has_calculator, f"No calculator found for {ws}"
