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

    @pytest.fixture
    def action_rate_uplift_kpi(self):
        """Create an action rate uplift KPI (a relative-uplift fraction; higher is better)."""
        return KPIMetadata(
            id="WS2-TR-003",
            name="Action Rate Uplift",
            definition="Incremental action rate vs control group",
            formula="(action_rate_treatment - action_rate_control) / action_rate_control",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS2_TRIGGERS,
            threshold=KPIThreshold(target=0.15, warning=0.10, critical=0.05),
        )

    def test_calculate_action_rate_uplift_good(self, calculator, action_rate_uplift_kpi):
        """WS2-TR-003 returns the realized RELATIVE uplift as a bare fraction (NOT 100×ratio,
        NOT an absolute difference); the live ~0.2751 is GOOD against target 0.15, and the
        metric is HIGHER-is-better (not in the lower_is_better set)."""
        calculator._execute_query = Mock(return_value=[{"action_rate_uplift": 0.2751}])

        result = calculator.calculate(action_rate_uplift_kpi)

        assert result.value == 0.2751
        assert result.status == KPIStatus.GOOD
        assert result.metadata.get("lower_is_better") is False

    def test_action_rate_uplift_band_and_scale_guard(self, calculator, action_rate_uplift_kpi):
        """Higher-is-better bands (anti-mis-scale lock): >=0.15 GOOD; [0.05,0.15) WARNING;
        <0.05 CRITICAL — including a NEGATIVE uplift (treatment worse than control). Under a
        100×ratio mis-scale every value would be GOOD and a negative indistinguishable."""
        for value, expected in [
            (0.2751, KPIStatus.GOOD),
            (0.12, KPIStatus.WARNING),
            (0.04, KPIStatus.CRITICAL),
            (-0.05, KPIStatus.CRITICAL),
        ]:
            calculator._execute_query = Mock(return_value=[{"action_rate_uplift": value}])
            result = calculator.calculate(action_rate_uplift_kpi)
            assert result.value == value
            assert result.status == expected, f"{value} -> {result.status} (expected {expected})"

    def test_action_rate_uplift_fails_loud_on_empty_arm(self, calculator, action_rate_uplift_kpi):
        """An empty treatment/control arm (NULL uplift) -> KPIResult carries the error, value
        None (no fabricated 0.0)."""
        calculator._execute_query = Mock(return_value=[{"action_rate_uplift": None}])

        result = calculator.calculate(action_rate_uplift_kpi)

        assert result.value is None
        assert result.status == KPIStatus.UNKNOWN
        assert result.error is not None and "unavailable" in result.error

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

    @pytest.fixture
    def touch_rate_kpi(self):
        """Create a patient touch rate KPI (a [0,1] fraction; higher is better)."""
        return KPIMetadata(
            id="WS3-BI-003",
            name="Patient Touch Rate",
            definition="Fraction of eligible patients with a delivered trigger-driven touchpoint",
            formula="delivered_touched_eligible / eligible",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS3_BUSINESS,
            threshold=KPIThreshold(target=0.40, warning=0.30, critical=0.20),
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

    def test_calculate_patient_touch_rate_good(self, calculator, touch_rate_kpi):
        """WS3-BI-003 returns the FRACTION (sibling parity with conversion_rate); the live
        0.9074 is GOOD against target 0.40."""
        calculator._execute_query = Mock(return_value=[{"touch_rate": 0.9074}])

        result = calculator.calculate(touch_rate_kpi)

        assert result.value == 0.9074
        assert result.status == KPIStatus.GOOD

    def test_patient_touch_rate_scale_guard(self, calculator, touch_rate_kpi):
        """Anti-mis-scale regression-lock (#577): the value is a [0,1] FRACTION evaluated
        against the [0,1] threshold, NOT 100*ratio. Under the percentage-scale bug every
        value would be GOOD (90.74 >= 0.40) and 0.25 vs 0.15 would be indistinguishable —
        these band assertions would then fail. Higher-is-better bands: >=0.40 GOOD,
        0.20<=v<0.40 WARNING, <0.20 CRITICAL."""
        for value, expected in [
            (0.9074, KPIStatus.GOOD),
            (0.35, KPIStatus.WARNING),
            (0.25, KPIStatus.WARNING),
            (0.15, KPIStatus.CRITICAL),
        ]:
            calculator._execute_query = Mock(return_value=[{"touch_rate": value}])
            result = calculator.calculate(touch_rate_kpi)
            assert result.value == value
            assert result.status == expected, f"{value} -> {result.status} (expected {expected})"

    def test_patient_touch_rate_fails_loud_on_empty_cohort(self, calculator, touch_rate_kpi):
        """No eligible cohort (NULLIF -> NULL) -> KPIResult carries the error, value None
        (no fabricated 0.0)."""
        calculator._execute_query = Mock(return_value=[{"touch_rate": None}])

        result = calculator.calculate(touch_rate_kpi)

        assert result.value is None
        assert result.status == KPIStatus.UNKNOWN
        assert result.error is not None and "unavailable" in result.error


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

    def test_calculate_remi_uncontrolled_computes(self, calculator, remi_uncontrolled_kpi):
        """#577: BR-001 is now wired to the real CSU cohort (ATC R06A antihistamine events
        carrying a UAS7 reading). With data it computes the uncontrolled fraction; lower is
        better, so 0.30 (<= target 0.40) is GOOD."""
        calculator._execute_query = Mock(return_value=[{"uncontrolled_rate": 0.30}])

        result = calculator.calculate(remi_uncontrolled_kpi)

        assert result.value == 0.30
        assert result.status == KPIStatus.GOOD

    def test_calculate_remi_uncontrolled_fails_loud_on_empty(
        self, calculator, remi_uncontrolled_kpi
    ):
        """#577 (preserves the #574 discipline): an empty antihistamine cohort must FAIL
        LOUD (value=None + 'unavailable'), never a fabricated 0% 'fully controlled'."""
        calculator._execute_query = Mock(return_value=[{"uncontrolled_rate": None}])

        result = calculator.calculate(remi_uncontrolled_kpi)

        assert result.value is None
        assert result.error is not None
        assert "unavailable" in result.error.lower()

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

    # --- #577 PR1: CM-003 causal_impact (honest descriptive aggregate) ---------------

    @pytest.fixture
    def causal_impact_kpi(self):
        """Create CM-003 Causal Impact KPI."""
        return KPIMetadata(
            id="CM-003",
            name="Causal Impact",
            definition="Average strength of discovered causal effects",
            formula="AVG(causal_effect_size)",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.CAUSAL_METRICS,
            threshold=None,
        )

    def test_calculate_cm003_causal_impact_computes(self, calculator, causal_impact_kpi):
        """CM-003 = path-level mean causal_effect_size, with a start_node breakdown.

        The value is the path-weighted mean SUM(effect*n)/SUM(n), so a per-node
        breakdown reduces to the simple across-paths average.
        """
        calculator._execute_query = Mock(
            return_value=[
                {
                    "start_node": "Cost_Reduction",
                    "effect": 0.30,
                    "n_paths": 10,
                    "avg_confidence": 0.85,
                },
                {
                    "start_node": "Treatment_Response",
                    "effect": 0.20,
                    "n_paths": 10,
                    "avg_confidence": 0.80,
                },
            ]
        )

        result = calculator.calculate(causal_impact_kpi)

        assert result.value == pytest.approx(0.25)
        assert result.metadata.get("n_paths") == 20
        assert len(result.metadata["breakdown"]) == 2
        # Anti-relabel code-anchor (#574): start_node is a discovered path SOURCE, not a do()-target.
        assert "intervention target" in result.metadata.get("note", "").lower()

    def test_calculate_cm003_forwards_validation_status(self, calculator, causal_impact_kpi):
        """The optional validation_status context filter is forwarded to the query param."""
        calculator._execute_query = Mock(
            return_value=[
                {"start_node": "X", "effect": 0.21, "n_paths": 11, "avg_confidence": 0.83}
            ]
        )

        calculator.calculate(causal_impact_kpi, context={"validation_status": "validated"})

        calculator._execute_query.assert_called_once_with(
            "causal_metrics_causal_impact", ["validated"]
        )

    def test_calculate_cm003_returns_none_on_empty(self, calculator, causal_impact_kpi):
        """CM-003 fails loud (value None + error) when no causal paths exist — never fabricates 0."""
        calculator._execute_query = Mock(return_value=[])

        result = calculator.calculate(causal_impact_kpi)

        assert result.value is None
        assert "error" in result.metadata

    # --- #577 PR2: CM-004 counterfactual (coherent do-contrast) ----------------------

    @pytest.fixture
    def counterfactual_kpi(self):
        """Create CM-004 Counterfactual Outcome KPI."""
        return KPIMetadata(
            id="CM-004",
            name="Counterfactual Outcome",
            definition="E[Y(a') | do(A=a), X]",
            formula="mean(counterfactual_outcome)",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.CAUSAL_METRICS,
            threshold=None,
        )

    def test_calculate_cm004_counterfactual_computes(self, calculator, counterfactual_kpi):
        """CM-004 value = mean counterfactual outcome LEVEL; factual + realized contrast +
        nominal effect in metadata."""
        calculator._execute_query = Mock(
            return_value=[
                {
                    "mean_counterfactual": 0.34,
                    "mean_factual": 0.50,
                    "mean_realized_contrast": 0.16,
                    "mean_effect": 0.176,
                    "n": 626,
                }
            ]
        )

        result = calculator.calculate(counterfactual_kpi)

        assert result.value == pytest.approx(0.34)
        assert result.metadata.get("mean_factual") == 0.50
        assert result.metadata.get("mean_realized_contrast") == 0.16
        assert result.metadata.get("mean_effect") == 0.176
        assert result.metadata.get("n") == 626

    def test_calculate_cm004_forwards_prediction_type(self, calculator, counterfactual_kpi):
        """The optional prediction_type context filter is forwarded to the query param."""
        calculator._execute_query = Mock(
            return_value=[
                {
                    "mean_counterfactual": 0.30,
                    "mean_factual": 0.48,
                    "mean_realized_contrast": 0.18,
                    "mean_effect": 0.18,
                    "n": 116,
                }
            ]
        )

        calculator.calculate(counterfactual_kpi, context={"prediction_type": "churn"})

        calculator._execute_query.assert_called_once_with(
            "causal_metrics_counterfactual", ["churn"]
        )

    def test_calculate_cm004_returns_none_on_empty(self, calculator, counterfactual_kpi):
        """CM-004 fails loud (value None + error) when no counterfactual rows match — the
        AVG-over-empty NULL row must NOT become a fabricated 0.0."""
        calculator._execute_query = Mock(
            return_value=[
                {
                    "mean_counterfactual": None,
                    "mean_factual": None,
                    "mean_realized_contrast": None,
                    "mean_effect": None,
                    "n": 0,
                }
            ]
        )

        result = calculator.calculate(counterfactual_kpi)

        assert result.value is None
        assert "error" in result.metadata

    # --- #577 PR3: CM-005 mediation_effect (coherent decomposition) -------------------

    @pytest.fixture
    def mediation_kpi(self):
        """Create CM-005 Mediation Effect KPI."""
        return KPIMetadata(
            id="CM-005",
            name="Mediation Effect",
            definition="indirect_effect / total_effect",
            formula="AVG(indirect_effect / causal_effect_size)",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.CAUSAL_METRICS,
            threshold=None,
        )

    def test_calculate_cm005_mediation_computes(self, calculator, mediation_kpi):
        """CM-005 value = mean proportion mediated; indirect/direct means in metadata."""
        calculator._execute_query = Mock(
            return_value=[
                {
                    "proportion_mediated": 0.127,
                    "n_paths": 50,
                    "mean_indirect": 0.03,
                    "mean_direct": 0.20,
                }
            ]
        )

        result = calculator.calculate(mediation_kpi)

        assert result.value == pytest.approx(0.127)
        assert result.metadata.get("n_paths") == 50
        assert result.metadata.get("mean_indirect") == 0.03
        assert result.metadata.get("mean_direct") == 0.20

    def test_calculate_cm005_queries_mediation(self, calculator, mediation_kpi):
        """CM-005 reads the no-param mediation query (max_params 0)."""
        calculator._execute_query = Mock(
            return_value=[
                {
                    "proportion_mediated": 0.1,
                    "n_paths": 50,
                    "mean_indirect": 0.02,
                    "mean_direct": 0.2,
                }
            ]
        )

        calculator.calculate(mediation_kpi)

        calculator._execute_query.assert_called_once_with("causal_metrics_mediation", [])

    def test_calculate_cm005_returns_none_on_empty(self, calculator, mediation_kpi):
        """CM-005 fails loud (value None + error) when no paths exist — never fabricates 0."""
        calculator._execute_query = Mock(
            return_value=[
                {
                    "proportion_mediated": None,
                    "n_paths": 0,
                    "mean_indirect": None,
                    "mean_direct": None,
                }
            ]
        )

        result = calculator.calculate(mediation_kpi)

        assert result.value is None
        assert "error" in result.metadata


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
