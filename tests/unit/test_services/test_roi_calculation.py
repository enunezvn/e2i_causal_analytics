"""
Unit tests for ROI Calculation Service.

Tests the full E2I ROI methodology implementation including:
- 6 value driver calculations
- Bootstrap confidence intervals
- Causal attribution framework
- Risk adjustment
- Sensitivity analysis
- NPV calculations
"""

from datetime import datetime

import numpy as np
import pytest

from src.services.roi_calculation import (
    ActionRateCalculator,
    AttributionCalculator,
    AttributionLevel,
    BootstrapSimulator,
    CostCalculator,
    CostInput,
    DataQualityCalculator,
    DriftPreventionCalculator,
    IntentToPrescribeCalculator,
    NPVCalculator,
    PatientIdentificationCalculator,
    RiskAdjustmentCalculator,
    RiskAssessment,
    RiskLevel,
    # Main service
    ROICalculationService,
    TRxLiftCalculator,
    # Data classes
    ValueDriverInput,
    # Enums
    ValueDriverType,
)

# =============================================================================
# Value Driver Calculator Tests
# =============================================================================


class TestTRxLiftCalculator:
    """Tests for TRx lift value calculation."""

    def test_value_per_trx(self):
        """Verify correct value per TRx from methodology."""
        calc = TRxLiftCalculator()
        assert calc.VALUE_PER_TRX == 850.0

    def test_basic_calculation(self):
        """Test basic TRx lift calculation."""
        calc = TRxLiftCalculator()
        result = calc.calculate(100)
        assert result == 85_000.0  # 100 TRx * $850

    def test_zero_trx(self):
        """Test zero TRx returns zero value."""
        calc = TRxLiftCalculator()
        assert calc.calculate(0) == 0.0

    def test_fractional_trx(self):
        """Test fractional TRx calculation."""
        calc = TRxLiftCalculator()
        result = calc.calculate(10.5)
        assert result == pytest.approx(8_925.0)  # 10.5 * $850


class TestPatientIdentificationCalculator:
    """Tests for patient identification value calculation."""

    def test_value_per_patient(self):
        """Verify correct value per patient from methodology."""
        calc = PatientIdentificationCalculator()
        assert calc.VALUE_PER_PATIENT == 1200.0

    def test_basic_calculation(self):
        """Test basic patient identification calculation."""
        calc = PatientIdentificationCalculator()
        result = calc.calculate(50)
        assert result == 60_000.0  # 50 patients * $1,200

    def test_single_patient(self):
        """Test single patient value."""
        calc = PatientIdentificationCalculator()
        assert calc.calculate(1) == 1200.0


class TestActionRateCalculator:
    """Tests for HCP action rate improvement value calculation."""

    def test_value_per_pp(self):
        """Verify correct value per percentage point from methodology."""
        calc = ActionRateCalculator()
        assert calc.VALUE_PER_PP == 45.0

    def test_basic_calculation(self):
        """Test basic action rate calculation with 1pp and 1000 triggers."""
        calc = ActionRateCalculator()
        result = calc.calculate(1.0, 1000)
        # 1pp * $45 * 1000 triggers * 12 months = $540,000
        assert result == 540_000.0

    def test_different_trigger_volume(self):
        """Test with different trigger volumes."""
        calc = ActionRateCalculator()
        result = calc.calculate(2.0, 500)
        # 2pp * $45 * 500 triggers * 12 months = $540,000
        assert result == 540_000.0

    def test_default_trigger_count(self):
        """Test default trigger count of 1000."""
        calc = ActionRateCalculator()
        result = calc.calculate(1.0)  # Uses default 1000 triggers
        assert result == 540_000.0


class TestIntentToPrescribeCalculator:
    """Tests for Intent-to-Prescribe lift value calculation."""

    def test_value_per_hcp_pp(self):
        """Verify correct value per HCP per percentage point."""
        calc = IntentToPrescribeCalculator()
        assert calc.VALUE_PER_HCP_PP == 320.0

    def test_basic_calculation(self):
        """Test basic ITP lift calculation."""
        calc = IntentToPrescribeCalculator()
        result = calc.calculate(itp_improvement_pp=1.0, hcp_count=100)
        # 1pp * $320 * 100 HCPs = $32,000
        assert result == 32_000.0

    def test_multiple_pp_improvement(self):
        """Test with larger improvement."""
        calc = IntentToPrescribeCalculator()
        result = calc.calculate(itp_improvement_pp=5.0, hcp_count=200)
        # 5pp * $320 * 200 HCPs = $320,000
        assert result == 320_000.0


class TestDataQualityCalculator:
    """Tests for data quality improvement value calculation."""

    def test_values_per_error_type(self):
        """Verify correct values per error type from methodology."""
        calc = DataQualityCalculator()
        assert calc.VALUE_PER_FP_AVOIDED == 200.0
        assert calc.VALUE_PER_FN_AVOIDED == 650.0

    def test_fp_only(self):
        """Test false positive reduction only."""
        calc = DataQualityCalculator()
        result = calc.calculate(fp_reduction=10, fn_reduction=0)
        # 10 FP * $200 * 12 months = $24,000
        assert result == 24_000.0

    def test_fn_only(self):
        """Test false negative reduction only."""
        calc = DataQualityCalculator()
        result = calc.calculate(fp_reduction=0, fn_reduction=10)
        # 10 FN * $650 * 12 months = $78,000
        assert result == 78_000.0

    def test_combined(self):
        """Test combined FP and FN reduction."""
        calc = DataQualityCalculator()
        result = calc.calculate(fp_reduction=10, fn_reduction=10)
        # (10 * $200 + 10 * $650) * 12 = $102,000
        assert result == 102_000.0


class TestDriftPreventionCalculator:
    """Tests for drift prevention value calculation."""

    def test_multiplier(self):
        """Verify correct multiplier from methodology."""
        calc = DriftPreventionCalculator()
        assert calc.MULTIPLIER == 2.0

    def test_basic_calculation(self):
        """Test basic drift prevention calculation."""
        calc = DriftPreventionCalculator()
        result = calc.calculate(
            auc_drop_prevented=0.05,  # 5% AUC drop prevented
            baseline_model_value=1_000_000,  # $1M baseline model value
        )
        # 0.05 * $1M * 2x = $100,000
        assert result == 100_000.0

    def test_no_auc_drop(self):
        """Test zero AUC drop returns zero value."""
        calc = DriftPreventionCalculator()
        result = calc.calculate(auc_drop_prevented=0, baseline_model_value=1_000_000)
        assert result == 0.0


# =============================================================================
# Bootstrap Simulator Tests
# =============================================================================


class TestBootstrapSimulator:
    """Tests for Monte Carlo bootstrap simulation."""

    def test_default_simulations(self):
        """Test default simulation count."""
        sim = BootstrapSimulator()
        assert sim.n_simulations == 1000

    def test_custom_simulations(self):
        """Test custom simulation count."""
        sim = BootstrapSimulator(n_simulations=500)
        assert sim.n_simulations == 500

    def test_seed_reproducibility(self):
        """Test that seed produces reproducible results."""
        sim1 = BootstrapSimulator(n_simulations=100, seed=42)
        sim2 = BootstrapSimulator(n_simulations=100, seed=42)

        samples1 = sim1.simulate_value(1000, 0.15)
        samples2 = sim2.simulate_value(1000, 0.15)

        np.testing.assert_array_equal(samples1, samples2)

    def test_simulate_value_shape(self):
        """Test simulated value array shape."""
        sim = BootstrapSimulator(n_simulations=100, seed=42)
        samples = sim.simulate_value(1000, 0.15)
        assert samples.shape == (100,)

    def test_simulate_value_non_negative(self):
        """Test that simulated values are non-negative."""
        sim = BootstrapSimulator(n_simulations=1000, seed=42)
        samples = sim.simulate_value(100, 0.5)  # High uncertainty
        assert np.all(samples >= 0)

    def test_simulate_value_mean(self):
        """Test that simulated values have approximately correct mean."""
        sim = BootstrapSimulator(n_simulations=10000, seed=42)
        samples = sim.simulate_value(1000, 0.15)
        # Mean should be close to 1000 (within 5%)
        assert 950 < np.mean(samples) < 1050

    def test_simulate_cost_shape(self):
        """Test simulated cost array shape."""
        sim = BootstrapSimulator(n_simulations=100, seed=42)
        samples = sim.simulate_cost(50000)
        assert samples.shape == (100,)

    def test_simulate_cost_non_negative(self):
        """Test that simulated costs are non-negative."""
        sim = BootstrapSimulator(n_simulations=1000, seed=42)
        samples = sim.simulate_cost(50000)
        assert np.all(samples >= 0)

    def test_simulate_cost_zero_mean(self):
        """Test zero cost returns zeros."""
        sim = BootstrapSimulator(n_simulations=100)
        samples = sim.simulate_cost(0)
        assert np.all(samples == 0)

    def test_simulate_acceptance_rate_shape(self):
        """Test simulated acceptance rate array shape."""
        sim = BootstrapSimulator(n_simulations=100, seed=42)
        samples = sim.simulate_acceptance_rate(0.2, 5)
        assert samples.shape == (100,)

    def test_confidence_interval_bounds(self):
        """Test confidence interval bound ordering."""
        sim = BootstrapSimulator(n_simulations=1000, seed=42)
        roi_samples = np.random.default_rng(42).normal(5.0, 1.0, 1000)

        ci = sim.compute_confidence_interval(roi_samples, target_roi=5.0)

        assert ci.lower_bound <= ci.median <= ci.upper_bound

    def test_confidence_interval_probability_positive(self):
        """Test probability positive calculation."""
        sim = BootstrapSimulator(n_simulations=1000)
        # All positive ROIs
        roi_samples = np.array([2.0, 3.0, 4.0, 5.0] * 250)

        ci = sim.compute_confidence_interval(roi_samples)

        assert ci.probability_positive == 1.0  # All > 1x

    def test_confidence_interval_probability_target(self):
        """Test probability target calculation."""
        sim = BootstrapSimulator(n_simulations=1000)
        # Half above target
        roi_samples = np.array([4.0] * 500 + [6.0] * 500)

        ci = sim.compute_confidence_interval(roi_samples, target_roi=5.0)

        assert ci.probability_target == 0.5  # Half > 5x


# =============================================================================
# Attribution Calculator Tests
# =============================================================================


class TestAttributionCalculator:
    """Tests for causal attribution calculation."""

    def test_attribution_rates(self):
        """Verify attribution rates from methodology."""
        calc = AttributionCalculator()

        assert calc.get_attribution_rate(AttributionLevel.FULL) == 1.0
        assert calc.get_attribution_rate(AttributionLevel.PARTIAL) == 0.65
        assert calc.get_attribution_rate(AttributionLevel.SHARED) == 0.35
        assert calc.get_attribution_rate(AttributionLevel.MINIMAL) == 0.10

    def test_full_attribution(self):
        """Test full attribution (100%)."""
        calc = AttributionCalculator()
        value, rate = calc.apply_attribution(100_000, AttributionLevel.FULL)

        assert value == 100_000.0
        assert rate == 1.0

    def test_partial_attribution(self):
        """Test partial attribution (65%)."""
        calc = AttributionCalculator()
        value, rate = calc.apply_attribution(100_000, AttributionLevel.PARTIAL)

        assert value == 65_000.0
        assert rate == 0.65

    def test_shared_attribution(self):
        """Test shared attribution (35%)."""
        calc = AttributionCalculator()
        value, rate = calc.apply_attribution(100_000, AttributionLevel.SHARED)

        assert value == 35_000.0
        assert rate == 0.35

    def test_minimal_attribution(self):
        """Test minimal attribution (10%)."""
        calc = AttributionCalculator()
        value, rate = calc.apply_attribution(100_000, AttributionLevel.MINIMAL)

        assert value == 10_000.0
        assert rate == 0.10


# =============================================================================
# Risk Adjustment Calculator Tests
# =============================================================================


class TestRiskAdjustmentCalculator:
    """Tests for risk adjustment calculation."""

    def test_risk_factors(self):
        """Verify risk factor values from methodology."""
        calc = RiskAdjustmentCalculator()

        # Technical complexity
        assert calc.RISK_FACTORS["technical_complexity"][RiskLevel.LOW] == 0.0
        assert calc.RISK_FACTORS["technical_complexity"][RiskLevel.MEDIUM] == 0.15
        assert calc.RISK_FACTORS["technical_complexity"][RiskLevel.HIGH] == 0.30

        # Organizational change
        assert calc.RISK_FACTORS["organizational_change"][RiskLevel.LOW] == 0.0
        assert calc.RISK_FACTORS["organizational_change"][RiskLevel.MEDIUM] == 0.20
        assert calc.RISK_FACTORS["organizational_change"][RiskLevel.HIGH] == 0.40

        # Data dependencies
        assert calc.RISK_FACTORS["data_dependencies"][RiskLevel.LOW] == 0.0
        assert calc.RISK_FACTORS["data_dependencies"][RiskLevel.MEDIUM] == 0.25
        assert calc.RISK_FACTORS["data_dependencies"][RiskLevel.HIGH] == 0.50

        # Timeline uncertainty
        assert calc.RISK_FACTORS["timeline_uncertainty"][RiskLevel.LOW] == 0.0
        assert calc.RISK_FACTORS["timeline_uncertainty"][RiskLevel.MEDIUM] == 0.10
        assert calc.RISK_FACTORS["timeline_uncertainty"][RiskLevel.HIGH] == 0.25

    def test_all_low_risk(self):
        """Test all LOW risk factors = 0% adjustment."""
        calc = RiskAdjustmentCalculator()
        assessment = RiskAssessment()  # All LOW by default

        adjustment = calc.calculate_total_adjustment(assessment)

        assert adjustment == 0.0

    def test_all_medium_risk(self):
        """Test all MEDIUM risk factors."""
        calc = RiskAdjustmentCalculator()
        assessment = RiskAssessment(
            technical_complexity=RiskLevel.MEDIUM,
            organizational_change=RiskLevel.MEDIUM,
            data_dependencies=RiskLevel.MEDIUM,
            timeline_uncertainty=RiskLevel.MEDIUM,
        )

        adjustment = calc.calculate_total_adjustment(assessment)

        # Multiplicative: 1 - (0.85 * 0.80 * 0.75 * 0.90) = 1 - 0.459 = 0.541
        assert adjustment == pytest.approx(0.541, abs=0.001)

    def test_all_high_risk(self):
        """Test all HIGH risk factors."""
        calc = RiskAdjustmentCalculator()
        assessment = RiskAssessment(
            technical_complexity=RiskLevel.HIGH,
            organizational_change=RiskLevel.HIGH,
            data_dependencies=RiskLevel.HIGH,
            timeline_uncertainty=RiskLevel.HIGH,
        )

        adjustment = calc.calculate_total_adjustment(assessment)

        # Multiplicative: 1 - (0.70 * 0.60 * 0.50 * 0.75) = 1 - 0.1575 = 0.8425
        assert adjustment == pytest.approx(0.8425, abs=0.001)

    def test_apply_risk_adjustment(self):
        """Test applying risk adjustment to ROI."""
        calc = RiskAdjustmentCalculator()
        assessment = RiskAssessment()  # All LOW

        adjusted_roi, adjustment = calc.apply_risk_adjustment(5.0, assessment)

        assert adjusted_roi == 5.0  # No adjustment
        assert adjustment == 0.0

    def test_apply_risk_adjustment_with_risk(self):
        """Test applying risk adjustment with some risk."""
        calc = RiskAdjustmentCalculator()
        assessment = RiskAssessment(technical_complexity=RiskLevel.HIGH)  # 30%

        adjusted_roi, adjustment = calc.apply_risk_adjustment(5.0, assessment)

        assert adjustment == pytest.approx(0.30)
        assert adjusted_roi == pytest.approx(3.5)  # 5 * 0.70


# =============================================================================
# Cost Calculator Tests
# =============================================================================


class TestCostCalculator:
    """Tests for cost calculation."""

    def test_engineering_cost(self):
        """Test engineering cost calculation."""
        calc = CostCalculator()
        cost_input = CostInput(
            engineering_days=20,
            engineering_day_rate=2500,
        )

        total, breakdown = calc.calculate_total_cost(cost_input)

        assert breakdown["engineering"] == 50_000.0
        assert total == 50_000.0

    def test_data_acquisition_cost(self):
        """Test data acquisition cost calculation."""
        calc = CostCalculator()
        cost_input = CostInput(
            data_source_costs={"claims": 50000, "ehr": 30000},
            incremental_data_cost=10000,
        )

        total, breakdown = calc.calculate_total_cost(cost_input)

        assert breakdown["data_acquisition"] == 90_000.0

    def test_change_management_cost(self):
        """Test training and change management cost."""
        calc = CostCalculator()
        cost_input = CostInput(
            training_cost=15000,
            change_management_cost=25000,
        )

        total, breakdown = calc.calculate_total_cost(cost_input)

        assert breakdown["change_management"] == 40_000.0

    def test_infrastructure_cost(self):
        """Test infrastructure cost calculation."""
        calc = CostCalculator()
        cost_input = CostInput(
            monthly_infrastructure_cost=5000,
            infrastructure_months=12,
        )

        total, breakdown = calc.calculate_total_cost(cost_input)

        assert breakdown["infrastructure"] == 60_000.0

    def test_opportunity_cost(self):
        """Test opportunity cost calculation."""
        calc = CostCalculator()
        cost_input = CostInput(
            delayed_initiative_annual_value=120000,
            delay_months=3,
        )

        total, breakdown = calc.calculate_total_cost(cost_input)

        # (120000 / 12) * 3 = 30000
        assert breakdown["opportunity_cost"] == 30_000.0

    def test_total_cost(self):
        """Test complete cost breakdown."""
        calc = CostCalculator()
        cost_input = CostInput(
            engineering_days=20,
            engineering_day_rate=2500,
            data_source_costs={"claims": 50000},
            training_cost=15000,
            monthly_infrastructure_cost=5000,
            infrastructure_months=12,
        )

        total, breakdown = calc.calculate_total_cost(cost_input)

        expected_total = 50000 + 50000 + 15000 + 60000
        assert total == expected_total


# =============================================================================
# NPV Calculator Tests
# =============================================================================


class TestNPVCalculator:
    """Tests for Net Present Value calculation."""

    def test_discount_rate(self):
        """Verify correct corporate discount rate."""
        calc = NPVCalculator()
        assert calc.DISCOUNT_RATE == 0.10

    def test_single_year_npv(self):
        """Test NPV for single year."""
        calc = NPVCalculator()
        annual_values = [100_000]

        npv = calc.calculate_npv(annual_values)

        # 100000 / 1.10 = 90909.09
        assert npv == pytest.approx(90909.09, abs=1)

    def test_three_year_npv(self):
        """Test NPV for three years."""
        calc = NPVCalculator()
        annual_values = [100_000, 100_000, 100_000]

        npv = calc.calculate_npv(annual_values)

        # 100000/1.10 + 100000/1.21 + 100000/1.331 = 248685.20
        assert npv == pytest.approx(248685.20, abs=1)

    def test_custom_discount_rate(self):
        """Test NPV with custom discount rate."""
        calc = NPVCalculator()
        annual_values = [100_000]

        npv = calc.calculate_npv(annual_values, discount_rate=0.05)

        # 100000 / 1.05 = 95238.10
        assert npv == pytest.approx(95238.10, abs=1)

    def test_empty_values(self):
        """Test NPV with empty value list."""
        calc = NPVCalculator()
        annual_values = []

        npv = calc.calculate_npv(annual_values)

        assert npv == 0.0


# =============================================================================
# ROICalculationService Tests
# =============================================================================


class TestROICalculationService:
    """Tests for the main ROI calculation service."""

    def test_initialization(self):
        """Test service initialization."""
        service = ROICalculationService(n_simulations=100, seed=42)

        assert service.n_simulations == 100
        assert service.bootstrap.n_simulations == 100

    def test_calculate_value_driver_trx(self):
        """Test TRx lift value driver calculation."""
        service = ROICalculationService()
        driver = ValueDriverInput(
            driver_type=ValueDriverType.TRX_LIFT,
            quantity=100,
        )

        value = service.calculate_value_driver(driver)

        assert value == 85_000.0

    def test_calculate_value_driver_patient_id(self):
        """Test patient identification value driver."""
        service = ROICalculationService()
        driver = ValueDriverInput(
            driver_type=ValueDriverType.PATIENT_IDENTIFICATION,
            quantity=50,
        )

        value = service.calculate_value_driver(driver)

        assert value == 60_000.0

    def test_calculate_value_driver_action_rate(self):
        """Test action rate value driver."""
        service = ROICalculationService()
        driver = ValueDriverInput(
            driver_type=ValueDriverType.ACTION_RATE,
            quantity=1.0,  # 1pp improvement
            trigger_count=1000,
        )

        value = service.calculate_value_driver(driver)

        assert value == 540_000.0

    def test_calculate_value_driver_itp(self):
        """Test ITP value driver."""
        service = ROICalculationService()
        driver = ValueDriverInput(
            driver_type=ValueDriverType.INTENT_TO_PRESCRIBE,
            quantity=1.0,  # 1pp improvement
            hcp_count=100,
        )

        value = service.calculate_value_driver(driver)

        assert value == 32_000.0

    def test_calculate_value_driver_itp_missing_hcp_count(self):
        """Test ITP driver raises error when hcp_count missing."""
        service = ROICalculationService()
        driver = ValueDriverInput(
            driver_type=ValueDriverType.INTENT_TO_PRESCRIBE,
            quantity=1.0,
        )

        with pytest.raises(ValueError, match="hcp_count required"):
            service.calculate_value_driver(driver)

    def test_calculate_value_driver_data_quality(self):
        """Test data quality value driver."""
        service = ROICalculationService()
        driver = ValueDriverInput(
            driver_type=ValueDriverType.DATA_QUALITY,
            quantity=0,  # Not used
            fp_reduction=10,
            fn_reduction=10,
        )

        value = service.calculate_value_driver(driver)

        assert value == 102_000.0

    def test_calculate_value_driver_drift_prevention(self):
        """Test drift prevention value driver."""
        service = ROICalculationService()
        driver = ValueDriverInput(
            driver_type=ValueDriverType.DRIFT_PREVENTION,
            quantity=0,  # Not used
            auc_drop_prevented=0.05,
            baseline_model_value=1_000_000,
        )

        value = service.calculate_value_driver(driver)

        assert value == 100_000.0

    def test_calculate_roi_basic(self):
        """Test basic ROI calculation."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,  # $85,000 value
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,  # $25,000 cost
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
            attribution_level=AttributionLevel.FULL,
        )

        # ROI = (85000 - 25000) / 25000 = 2.4x
        assert result.incremental_value == 85_000.0
        assert result.implementation_cost == 25_000.0
        assert result.base_roi == pytest.approx(2.4)
        assert result.attribution_rate == 1.0
        assert result.attributed_value == 85_000.0

    def test_calculate_roi_with_attribution(self):
        """Test ROI with attribution adjustment."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,  # $85,000 value
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,  # $25,000 cost
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
            attribution_level=AttributionLevel.PARTIAL,  # 65%
        )

        # Attributed value = 85000 * 0.65 = 55250
        # ROI = (55250 - 25000) / 25000 = 1.21x
        assert result.attribution_rate == 0.65
        assert result.attributed_value == 55_250.0
        assert result.base_roi == pytest.approx(1.21)

    def test_calculate_roi_with_risk(self):
        """Test ROI with risk adjustment."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,
        )
        risk_assessment = RiskAssessment(
            technical_complexity=RiskLevel.HIGH,  # 30%
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
            attribution_level=AttributionLevel.FULL,
            risk_assessment=risk_assessment,
        )

        # Base ROI = 2.4x
        # Risk adjusted = 2.4 * 0.70 = 1.68x
        assert result.total_risk_adjustment == pytest.approx(0.30)
        assert result.risk_adjusted_roi == pytest.approx(1.68)

    def test_calculate_roi_confidence_interval(self):
        """Test that CI is computed."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
        )

        assert result.confidence_interval is not None
        assert result.confidence_interval.lower_bound < result.confidence_interval.median
        assert result.confidence_interval.median < result.confidence_interval.upper_bound
        assert 0 <= result.confidence_interval.probability_positive <= 1
        assert 0 <= result.confidence_interval.probability_target <= 1

    def test_calculate_roi_value_breakdown(self):
        """Test value breakdown by driver."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            ),
            ValueDriverInput(
                driver_type=ValueDriverType.PATIENT_IDENTIFICATION,
                quantity=50,
            ),
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
        )

        assert "trx_lift" in result.value_by_driver
        assert "patient_identification" in result.value_by_driver
        assert result.value_by_driver["trx_lift"] == 85_000.0
        assert result.value_by_driver["patient_identification"] == 60_000.0
        assert result.incremental_value == 145_000.0

    def test_calculate_roi_cost_breakdown(self):
        """Test cost breakdown."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,
            training_cost=5000,
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
        )

        assert "engineering" in result.cost_breakdown
        assert "change_management" in result.cost_breakdown
        assert result.cost_breakdown["engineering"] == 25_000.0
        assert result.cost_breakdown["change_management"] == 5_000.0

    def test_calculate_roi_zero_cost(self):
        """Test ROI with zero cost."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            )
        ]
        cost_input = CostInput()  # Zero cost

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
        )

        assert result.base_roi == float("inf")

    def test_calculate_roi_with_sensitivity(self):
        """Test ROI with sensitivity analysis."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
            run_sensitivity=True,
        )

        assert result.sensitivity_results is not None
        assert len(result.sensitivity_results) > 0
        # Should have sensitivity for value driver and cost
        params = [r.parameter for r in result.sensitivity_results]
        assert "trx_lift" in params
        assert "implementation_cost" in params

    def test_calculate_npv_roi(self):
        """Test ROI calculation with NPV."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,  # $85,000/year
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,  # $25,000 cost
        )

        result = service.calculate_npv_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
            years=3,
            attribution_level=AttributionLevel.FULL,
        )

        assert result.npv_value is not None
        assert result.npv_roi is not None
        # NPV of 3 years of $85k at 10% discount
        # = 85000/1.1 + 85000/1.21 + 85000/1.331 = 211,383
        assert result.npv_value == pytest.approx(211_383, abs=100)
        # NPV ROI = (211383 - 25000) / 25000 = 7.46x
        assert result.npv_roi == pytest.approx(7.45, abs=0.1)

    def test_format_roi_summary(self):
        """Test ROI summary formatting."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
            attribution_level=AttributionLevel.FULL,  # Explicit FULL for 2.4x ROI
        )

        summary = service.format_roi_summary(result)

        assert "summary" in summary
        assert "value" in summary
        assert "cost" in summary
        assert "risk" in summary
        assert "npv" in summary
        assert "methodology_version" in summary
        assert summary["summary"]["base_roi"] == "2.4x"

    def test_metadata(self):
        """Test result metadata."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
        )

        assert result.methodology_version == "1.0"
        assert isinstance(result.calculated_at, datetime)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_driver_type(self):
        """Test handling of invalid driver type."""
        service = ROICalculationService()

        # Create a driver with invalid type by monkeypatching
        driver = ValueDriverInput(
            driver_type=ValueDriverType.TRX_LIFT,
            quantity=100,
        )
        # Override to invalid value
        driver.driver_type = "invalid_type"  # type: ignore

        with pytest.raises(ValueError, match="Unknown driver type"):
            service.calculate_value_driver(driver)

    def test_negative_quantity(self):
        """Test handling of negative quantity (not blocked)."""
        service = ROICalculationService()
        driver = ValueDriverInput(
            driver_type=ValueDriverType.TRX_LIFT,
            quantity=-100,
        )

        # Currently not blocked - produces negative value
        value = service.calculate_value_driver(driver)
        assert value == -85_000.0

    def test_very_large_values(self):
        """Test handling of very large values."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=1_000_000,  # 1M TRx
            )
        ]
        cost_input = CostInput(
            engineering_days=100,
            engineering_day_rate=2500,
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
        )

        assert result.incremental_value == 850_000_000.0
        assert result.base_roi > 0

    def test_multiple_value_drivers(self):
        """Test combining multiple value drivers."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            ),
            ValueDriverInput(
                driver_type=ValueDriverType.PATIENT_IDENTIFICATION,
                quantity=50,
            ),
            ValueDriverInput(
                driver_type=ValueDriverType.DATA_QUALITY,
                quantity=0,
                fp_reduction=20,
                fn_reduction=10,
            ),
        ]
        cost_input = CostInput(
            engineering_days=20,
            engineering_day_rate=2500,
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
        )

        # 85000 + 60000 + (20*200 + 10*650)*12 = 85000 + 60000 + 126000 = 271000
        assert result.incremental_value == 271_000.0
        assert len(result.value_by_driver) == 3

    def test_all_risk_factors(self):
        """Test combining all risk factors."""
        service = ROICalculationService(n_simulations=100, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,
        )
        risk_assessment = RiskAssessment(
            technical_complexity=RiskLevel.HIGH,
            organizational_change=RiskLevel.MEDIUM,
            data_dependencies=RiskLevel.HIGH,
            timeline_uncertainty=RiskLevel.MEDIUM,
        )

        result = service.calculate_roi(
            value_drivers=value_drivers,
            cost_input=cost_input,
            risk_assessment=risk_assessment,
        )

        # Total adjustment = 1 - (0.70 * 0.80 * 0.50 * 0.90) = 1 - 0.252 = 0.748
        assert result.total_risk_adjustment == pytest.approx(0.748, abs=0.01)
        assert result.risk_adjusted_roi < result.base_roi

    def test_simulation_count_affects_ci_precision(self):
        """Test that more simulations give more stable CI."""
        # Low simulation count - more variance
        service_low = ROICalculationService(n_simulations=10, seed=42)
        # High simulation count - less variance
        service_high = ROICalculationService(n_simulations=1000, seed=42)

        value_drivers = [
            ValueDriverInput(
                driver_type=ValueDriverType.TRX_LIFT,
                quantity=100,
            )
        ]
        cost_input = CostInput(
            engineering_days=10,
            engineering_day_rate=2500,
        )

        result_low = service_low.calculate_roi(value_drivers, cost_input)
        result_high = service_high.calculate_roi(value_drivers, cost_input)

        # Both should have CI computed
        assert result_low.confidence_interval is not None
        assert result_high.confidence_interval is not None
        assert result_low.confidence_interval.simulation_count == 10
        assert result_high.confidence_interval.simulation_count == 1000
