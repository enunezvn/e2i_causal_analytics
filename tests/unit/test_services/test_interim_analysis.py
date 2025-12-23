"""
Unit tests for Interim Analysis Service.

Phase 15: A/B Testing Infrastructure

Tests for:
- O'Brien-Fleming alpha spending boundaries
- Pocock boundaries
- Haybittle-Peto boundaries
- Conditional power calculations
- Predictive probability calculations
- Interim analysis execution
- Stopping decisions (efficacy/futility)
- Analysis history retrieval
- Next analysis timing recommendations
"""

import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest
from scipy import stats

from src.services.interim_analysis import (
    InterimAnalysisConfig,
    InterimAnalysisResult,
    InterimAnalysisService,
    MetricData,
    SpendingFunction,
    StoppingDecision,
    get_interim_analysis_service,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config() -> InterimAnalysisConfig:
    """Default interim analysis configuration."""
    return InterimAnalysisConfig(
        total_alpha=0.05,
        spending_function=SpendingFunction.OBRIEN_FLEMING,
        num_planned_analyses=3,
        futility_threshold=0.2,
        enable_futility_stopping=True,
    )


@pytest.fixture
def pocock_config() -> InterimAnalysisConfig:
    """Pocock spending function configuration."""
    return InterimAnalysisConfig(
        total_alpha=0.05,
        spending_function=SpendingFunction.POCOCK,
        num_planned_analyses=3,
    )


@pytest.fixture
def haybittle_peto_config() -> InterimAnalysisConfig:
    """Haybittle-Peto spending function configuration."""
    return InterimAnalysisConfig(
        total_alpha=0.05,
        spending_function=SpendingFunction.HAYBITTLE_PETO,
        num_planned_analyses=3,
    )


@pytest.fixture
def custom_schedule_config() -> InterimAnalysisConfig:
    """Custom alpha schedule configuration."""
    return InterimAnalysisConfig(
        total_alpha=0.05,
        spending_function=SpendingFunction.CUSTOM,
        num_planned_analyses=3,
        custom_alpha_schedule=[0.001, 0.01, 0.05],
    )


@pytest.fixture
def service(default_config: InterimAnalysisConfig) -> InterimAnalysisService:
    """Default interim analysis service."""
    return InterimAnalysisService(config=default_config)


@pytest.fixture
def experiment_id() -> UUID:
    """Sample experiment ID."""
    return uuid4()


@pytest.fixture
def metric_data_significant() -> MetricData:
    """Metric data that should show significant effect."""
    np.random.seed(42)
    return MetricData(
        name="conversion_rate",
        control_values=np.random.normal(0.10, 0.02, 500),
        treatment_values=np.random.normal(0.15, 0.02, 500),  # 50% lift
    )


@pytest.fixture
def metric_data_not_significant() -> MetricData:
    """Metric data with no significant effect."""
    np.random.seed(42)
    return MetricData(
        name="conversion_rate",
        control_values=np.random.normal(0.10, 0.05, 100),
        treatment_values=np.random.normal(0.10, 0.05, 100),  # No lift
    )


@pytest.fixture
def metric_data_negative() -> MetricData:
    """Metric data with significant negative effect."""
    np.random.seed(42)
    return MetricData(
        name="conversion_rate",
        control_values=np.random.normal(0.15, 0.02, 500),
        treatment_values=np.random.normal(0.10, 0.02, 500),  # Negative effect
    )


# =============================================================================
# TEST ENUMS
# =============================================================================


class TestSpendingFunction:
    """Tests for SpendingFunction enum."""

    def test_spending_function_values(self):
        """Test all spending function values exist."""
        assert SpendingFunction.OBRIEN_FLEMING.value == "obrien_fleming"
        assert SpendingFunction.POCOCK.value == "pocock"
        assert SpendingFunction.HAYBITTLE_PETO.value == "haybittle_peto"
        assert SpendingFunction.CUSTOM.value == "custom"

    def test_spending_function_is_str_enum(self):
        """Test spending function is string enum."""
        assert isinstance(SpendingFunction.OBRIEN_FLEMING, str)
        assert SpendingFunction.OBRIEN_FLEMING == "obrien_fleming"


class TestStoppingDecision:
    """Tests for StoppingDecision enum."""

    def test_stopping_decision_values(self):
        """Test all stopping decision values exist."""
        assert StoppingDecision.CONTINUE.value == "continue"
        assert StoppingDecision.STOP_EFFICACY.value == "stop_efficacy"
        assert StoppingDecision.STOP_FUTILITY.value == "stop_futility"
        assert StoppingDecision.STOP_SAFETY.value == "stop_safety"
        assert StoppingDecision.MODIFY_SAMPLE.value == "modify_sample"

    def test_stopping_decision_is_str_enum(self):
        """Test stopping decision is string enum."""
        assert isinstance(StoppingDecision.CONTINUE, str)


# =============================================================================
# TEST DATACLASSES
# =============================================================================


class TestInterimAnalysisConfig:
    """Tests for InterimAnalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InterimAnalysisConfig()
        assert config.total_alpha == 0.05
        assert config.spending_function == SpendingFunction.OBRIEN_FLEMING
        assert config.num_planned_analyses == 3
        assert config.futility_threshold == 0.2
        assert config.enable_futility_stopping is True
        assert config.custom_alpha_schedule is None
        assert config.assumed_effect_size is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = InterimAnalysisConfig(
            total_alpha=0.10,
            spending_function=SpendingFunction.POCOCK,
            num_planned_analyses=5,
            futility_threshold=0.15,
            enable_futility_stopping=False,
            custom_alpha_schedule=[0.01, 0.02, 0.03, 0.04, 0.05],
            assumed_effect_size=0.3,
        )
        assert config.total_alpha == 0.10
        assert config.spending_function == SpendingFunction.POCOCK
        assert config.num_planned_analyses == 5
        assert len(config.custom_alpha_schedule) == 5


class TestInterimAnalysisResult:
    """Tests for InterimAnalysisResult dataclass."""

    def test_required_fields(self):
        """Test result with required fields."""
        exp_id = uuid4()
        now = datetime.now(timezone.utc)
        result = InterimAnalysisResult(
            experiment_id=exp_id,
            analysis_number=1,
            performed_at=now,
            information_fraction=0.33,
            sample_size_control=100,
            sample_size_treatment=100,
            total_sample_size=200,
        )
        assert result.experiment_id == exp_id
        assert result.analysis_number == 1
        assert result.information_fraction == 0.33

    def test_default_values(self):
        """Test result default values."""
        result = InterimAnalysisResult(
            experiment_id=uuid4(),
            analysis_number=1,
            performed_at=datetime.now(timezone.utc),
            information_fraction=0.5,
            sample_size_control=50,
            sample_size_treatment=50,
            total_sample_size=100,
        )
        assert result.target_sample_size is None
        assert result.control_mean == 0.0
        assert result.treatment_mean == 0.0
        assert result.effect_estimate == 0.0
        assert result.p_value == 1.0
        assert result.decision == StoppingDecision.CONTINUE
        assert result.metrics_snapshot == {}


class TestMetricData:
    """Tests for MetricData dataclass."""

    def test_metric_data_creation(self):
        """Test metric data creation."""
        control = np.array([1.0, 2.0, 3.0])
        treatment = np.array([2.0, 3.0, 4.0])
        data = MetricData(
            name="test_metric",
            control_values=control,
            treatment_values=treatment,
        )
        assert data.name == "test_metric"
        np.testing.assert_array_equal(data.control_values, control)
        np.testing.assert_array_equal(data.treatment_values, treatment)


# =============================================================================
# TEST O'BRIEN-FLEMING BOUNDARY
# =============================================================================


class TestOBrienFlemingBoundary:
    """Tests for O'Brien-Fleming alpha spending boundary calculation."""

    def test_boundary_at_info_fraction_1(self, service: InterimAnalysisService):
        """Test boundary at 100% information equals alpha."""
        boundary = service.calculate_obrien_fleming_boundary(
            information_fraction=1.0,
            total_alpha=0.05,
            num_analyses=3,
        )
        # At full information, boundary should be close to alpha
        assert 0.04 < boundary < 0.06

    def test_boundary_conservative_early(self, service: InterimAnalysisService):
        """Test O'Brien-Fleming is conservative early (small alpha early)."""
        boundary_early = service.calculate_obrien_fleming_boundary(
            information_fraction=0.33,
            total_alpha=0.05,
        )
        boundary_late = service.calculate_obrien_fleming_boundary(
            information_fraction=0.67,
            total_alpha=0.05,
        )
        # Early boundary should be much smaller (more conservative)
        assert boundary_early < boundary_late
        assert boundary_early < 0.01  # Very conservative at 33%

    def test_boundary_increases_with_information(self, service: InterimAnalysisService):
        """Test boundary increases as more information is collected."""
        boundaries = []
        for frac in [0.25, 0.50, 0.75, 1.0]:
            boundary = service.calculate_obrien_fleming_boundary(
                information_fraction=frac,
                total_alpha=0.05,
            )
            boundaries.append(boundary)
        # Boundaries should be monotonically increasing
        for i in range(len(boundaries) - 1):
            assert boundaries[i] < boundaries[i + 1]

    def test_invalid_information_fraction_zero(self, service: InterimAnalysisService):
        """Test invalid information fraction (zero)."""
        with pytest.raises(ValueError, match="Information fraction must be in"):
            service.calculate_obrien_fleming_boundary(
                information_fraction=0.0,
                total_alpha=0.05,
            )

    def test_invalid_information_fraction_negative(self, service: InterimAnalysisService):
        """Test invalid information fraction (negative)."""
        with pytest.raises(ValueError, match="Information fraction must be in"):
            service.calculate_obrien_fleming_boundary(
                information_fraction=-0.5,
                total_alpha=0.05,
            )

    def test_invalid_information_fraction_exceeds_one(self, service: InterimAnalysisService):
        """Test information fraction slightly over 1 is rejected."""
        with pytest.raises(ValueError, match="Information fraction must be in"):
            service.calculate_obrien_fleming_boundary(
                information_fraction=1.1,
                total_alpha=0.05,
            )


# =============================================================================
# TEST POCOCK BOUNDARY
# =============================================================================


class TestPocockBoundary:
    """Tests for Pocock alpha spending boundary calculation."""

    def test_pocock_constant_boundary(self):
        """Test Pocock uses approximately constant boundary."""
        config = InterimAnalysisConfig(
            spending_function=SpendingFunction.POCOCK,
            num_planned_analyses=3,
        )
        service = InterimAnalysisService(config)

        boundary_early = service.calculate_pocock_boundary(
            information_fraction=0.33,
            total_alpha=0.05,
            num_analyses=3,
        )
        boundary_late = service.calculate_pocock_boundary(
            information_fraction=0.67,
            total_alpha=0.05,
            num_analyses=3,
        )
        # Pocock boundaries should be similar (constant)
        assert abs(boundary_early - boundary_late) < 0.01

    def test_pocock_less_conservative_than_obrien_fleming(self):
        """Test Pocock is less conservative early than O'Brien-Fleming."""
        config_obf = InterimAnalysisConfig(spending_function=SpendingFunction.OBRIEN_FLEMING)
        config_poc = InterimAnalysisConfig(spending_function=SpendingFunction.POCOCK)

        service_obf = InterimAnalysisService(config_obf)
        service_poc = InterimAnalysisService(config_poc)

        boundary_obf = service_obf.calculate_obrien_fleming_boundary(0.33, 0.05, 3)
        boundary_poc = service_poc.calculate_pocock_boundary(0.33, 0.05, 3)

        # Pocock should have larger boundary early (less conservative)
        assert boundary_poc > boundary_obf


# =============================================================================
# TEST HAYBITTLE-PETO BOUNDARY
# =============================================================================


class TestHaybittlePetoBoundary:
    """Tests for Haybittle-Peto alpha spending boundary calculation."""

    def test_interim_boundary_very_stringent(self):
        """Test interim analysis boundary is very stringent (0.001)."""
        config = InterimAnalysisConfig(spending_function=SpendingFunction.HAYBITTLE_PETO)
        service = InterimAnalysisService(config)

        boundary = service.calculate_haybittle_peto_boundary(
            information_fraction=0.5,
            is_final_analysis=False,
        )
        assert boundary == 0.001

    def test_final_boundary_preserves_alpha(self):
        """Test final analysis preserves nearly all alpha."""
        config = InterimAnalysisConfig(spending_function=SpendingFunction.HAYBITTLE_PETO)
        service = InterimAnalysisService(config)

        boundary = service.calculate_haybittle_peto_boundary(
            information_fraction=1.0,
            is_final_analysis=True,
        )
        assert boundary == 0.05


# =============================================================================
# TEST GET ALPHA BOUNDARY
# =============================================================================


class TestGetAlphaBoundary:
    """Tests for get_alpha_boundary method."""

    def test_obrien_fleming_routing(self, default_config: InterimAnalysisConfig):
        """Test O'Brien-Fleming function is used correctly."""
        service = InterimAnalysisService(default_config)
        boundary = service.get_alpha_boundary(0.5, 2, is_final=False)
        expected = service.calculate_obrien_fleming_boundary(0.5, 0.05, 3)
        assert boundary == expected

    def test_pocock_routing(self, pocock_config: InterimAnalysisConfig):
        """Test Pocock function is used correctly."""
        service = InterimAnalysisService(pocock_config)
        boundary = service.get_alpha_boundary(0.5, 2, is_final=False)
        expected = service.calculate_pocock_boundary(0.5, 0.05, 3)
        assert boundary == expected

    def test_haybittle_peto_routing(self, haybittle_peto_config: InterimAnalysisConfig):
        """Test Haybittle-Peto function is used correctly."""
        service = InterimAnalysisService(haybittle_peto_config)
        boundary = service.get_alpha_boundary(0.5, 2, is_final=False)
        assert boundary == 0.001

    def test_haybittle_peto_final(self, haybittle_peto_config: InterimAnalysisConfig):
        """Test Haybittle-Peto final analysis."""
        service = InterimAnalysisService(haybittle_peto_config)
        boundary = service.get_alpha_boundary(1.0, 3, is_final=True)
        assert boundary == 0.05

    def test_custom_schedule_routing(self, custom_schedule_config: InterimAnalysisConfig):
        """Test custom schedule is used correctly."""
        service = InterimAnalysisService(custom_schedule_config)

        boundary1 = service.get_alpha_boundary(0.33, 1)
        boundary2 = service.get_alpha_boundary(0.67, 2)
        boundary3 = service.get_alpha_boundary(1.0, 3)

        assert boundary1 == 0.001
        assert boundary2 == 0.01
        assert boundary3 == 0.05

    def test_custom_schedule_beyond_planned(self, custom_schedule_config: InterimAnalysisConfig):
        """Test custom schedule falls back to total alpha when beyond planned."""
        service = InterimAnalysisService(custom_schedule_config)
        boundary = service.get_alpha_boundary(1.0, 5)  # Beyond planned
        assert boundary == 0.05


# =============================================================================
# TEST CONDITIONAL POWER
# =============================================================================


class TestConditionalPower:
    """Tests for conditional power calculation."""

    def test_conditional_power_high_effect(self, service: InterimAnalysisService):
        """Test conditional power with strong observed effect."""
        power = service.calculate_conditional_power(
            current_effect=0.5,
            current_variance=0.01,
            target_effect=0.3,
            current_n=100,
            target_n=200,
            alpha=0.05,
        )
        # Strong effect should give high conditional power
        assert power > 0.8

    def test_conditional_power_weak_effect(self, service: InterimAnalysisService):
        """Test conditional power with weak observed effect."""
        power = service.calculate_conditional_power(
            current_effect=0.01,
            current_variance=0.1,
            target_effect=0.3,
            current_n=100,
            target_n=200,
            alpha=0.05,
        )
        # Weak effect should give low conditional power
        assert power < 0.5

    def test_conditional_power_bounds(self, service: InterimAnalysisService):
        """Test conditional power is bounded between 0 and 1."""
        power = service.calculate_conditional_power(
            current_effect=10.0,  # Extreme effect
            current_variance=0.01,
            target_effect=0.3,
            current_n=100,
            target_n=200,
        )
        assert 0.0 <= power <= 1.0

    def test_conditional_power_at_target(self, service: InterimAnalysisService):
        """Test conditional power when at target sample size."""
        power = service.calculate_conditional_power(
            current_effect=0.3,
            current_variance=0.01,
            target_effect=0.3,
            current_n=200,
            target_n=200,  # At target
        )
        assert power == 0.0

    def test_conditional_power_beyond_target(self, service: InterimAnalysisService):
        """Test conditional power beyond target returns 0."""
        power = service.calculate_conditional_power(
            current_effect=0.3,
            current_variance=0.01,
            target_effect=0.3,
            current_n=250,  # Beyond target
            target_n=200,
        )
        assert power == 0.0

    def test_conditional_power_zero_variance(self, service: InterimAnalysisService):
        """Test conditional power with zero variance returns 0."""
        power = service.calculate_conditional_power(
            current_effect=0.3,
            current_variance=0.0,  # Zero variance
            target_effect=0.3,
            current_n=100,
            target_n=200,
        )
        assert power == 0.0


# =============================================================================
# TEST PREDICTIVE PROBABILITY
# =============================================================================


class TestPredictiveProbability:
    """Tests for predictive probability calculation."""

    def test_predictive_probability_strong_effect(self, service: InterimAnalysisService):
        """Test predictive probability with strong observed effect."""
        prob = service.calculate_predictive_probability(
            current_effect=0.5,
            current_se=0.05,
            target_effect=0.3,
            current_n=100,
            target_n=200,
            alpha=0.05,
        )
        # Strong effect should give high predictive probability
        assert prob > 0.5

    def test_predictive_probability_bounds(self, service: InterimAnalysisService):
        """Test predictive probability is bounded between 0 and 1."""
        prob = service.calculate_predictive_probability(
            current_effect=10.0,  # Extreme
            current_se=0.05,
            target_effect=0.3,
            current_n=100,
            target_n=200,
        )
        assert 0.0 <= prob <= 1.0

    def test_predictive_probability_at_target(self, service: InterimAnalysisService):
        """Test predictive probability at target returns 0."""
        prob = service.calculate_predictive_probability(
            current_effect=0.3,
            current_se=0.05,
            target_effect=0.3,
            current_n=200,
            target_n=200,
        )
        assert prob == 0.0

    def test_predictive_probability_zero_se(self, service: InterimAnalysisService):
        """Test predictive probability with zero SE returns 0."""
        prob = service.calculate_predictive_probability(
            current_effect=0.3,
            current_se=0.0,
            target_effect=0.3,
            current_n=100,
            target_n=200,
        )
        assert prob == 0.0


# =============================================================================
# TEST CUMULATIVE ALPHA CALCULATION
# =============================================================================


class TestCumulativeAlpha:
    """Tests for cumulative alpha spending calculation."""

    def test_cumulative_alpha_obrien_fleming(self, default_config: InterimAnalysisConfig):
        """Test cumulative alpha for O'Brien-Fleming."""
        service = InterimAnalysisService(default_config)

        cumulative_early = service._calculate_cumulative_alpha(0.33)
        cumulative_late = service._calculate_cumulative_alpha(0.67)
        cumulative_final = service._calculate_cumulative_alpha(1.0)

        # Should increase monotonically
        assert cumulative_early < cumulative_late < cumulative_final
        # Final should approach total alpha
        assert 0.04 < cumulative_final < 0.06

    def test_cumulative_alpha_pocock(self, pocock_config: InterimAnalysisConfig):
        """Test cumulative alpha for Pocock (linear)."""
        service = InterimAnalysisService(pocock_config)

        cumulative_50 = service._calculate_cumulative_alpha(0.5)
        # Pocock spends alpha linearly
        assert abs(cumulative_50 - 0.025) < 0.005  # ~50% of alpha at 50% info


# =============================================================================
# TEST STOPPING DECISION
# =============================================================================


class TestStoppingDecision:
    """Tests for stopping decision logic."""

    def test_stop_efficacy_positive_effect(self, service: InterimAnalysisService):
        """Test stopping for efficacy with positive significant effect."""
        decision, rationale = service._make_stopping_decision(
            p_value=0.001,
            alpha_boundary=0.01,
            conditional_power=0.9,
            effect_estimate=0.5,  # Positive effect
            is_final=False,
        )
        assert decision == StoppingDecision.STOP_EFFICACY
        assert "positive effect" in rationale.lower()

    def test_stop_efficacy_negative_effect(self, service: InterimAnalysisService):
        """Test stopping for efficacy with negative significant effect (harm)."""
        decision, rationale = service._make_stopping_decision(
            p_value=0.001,
            alpha_boundary=0.01,
            conditional_power=0.9,
            effect_estimate=-0.5,  # Negative effect
            is_final=False,
        )
        assert decision == StoppingDecision.STOP_EFFICACY
        assert "negative" in rationale.lower() or "harmful" in rationale.lower()

    def test_stop_futility_low_power(self, service: InterimAnalysisService):
        """Test stopping for futility with low conditional power."""
        decision, rationale = service._make_stopping_decision(
            p_value=0.3,
            alpha_boundary=0.01,
            conditional_power=0.1,  # Below threshold
            effect_estimate=0.01,
            is_final=False,
        )
        assert decision == StoppingDecision.STOP_FUTILITY
        assert "futility" in rationale.lower()

    def test_continue_high_power(self, service: InterimAnalysisService):
        """Test continue when conditional power is adequate."""
        decision, rationale = service._make_stopping_decision(
            p_value=0.3,
            alpha_boundary=0.01,
            conditional_power=0.8,  # Above threshold
            effect_estimate=0.1,
            is_final=False,
        )
        assert decision == StoppingDecision.CONTINUE
        assert "continue" in rationale.lower()

    def test_continue_final_not_significant(self, service: InterimAnalysisService):
        """Test continue at final if not significant."""
        decision, rationale = service._make_stopping_decision(
            p_value=0.1,
            alpha_boundary=0.05,
            conditional_power=None,
            effect_estimate=0.1,
            is_final=True,
        )
        assert decision == StoppingDecision.CONTINUE
        assert "final" in rationale.lower()

    def test_futility_disabled(self, default_config: InterimAnalysisConfig):
        """Test futility stopping is disabled."""
        default_config.enable_futility_stopping = False
        service = InterimAnalysisService(default_config)

        decision, rationale = service._make_stopping_decision(
            p_value=0.5,
            alpha_boundary=0.01,
            conditional_power=0.05,  # Very low
            effect_estimate=0.01,
            is_final=False,
        )
        # Should continue despite low power when futility disabled
        assert decision == StoppingDecision.CONTINUE

    def test_no_conditional_power_continues(self, service: InterimAnalysisService):
        """Test continues when conditional power is not available."""
        decision, rationale = service._make_stopping_decision(
            p_value=0.3,
            alpha_boundary=0.01,
            conditional_power=None,  # Not available
            effect_estimate=0.05,
            is_final=False,
        )
        assert decision == StoppingDecision.CONTINUE


# =============================================================================
# TEST PERFORM INTERIM ANALYSIS
# =============================================================================


class TestPerformInterimAnalysis:
    """Tests for perform_interim_analysis method."""

    @pytest.mark.asyncio
    async def test_analysis_with_significant_effect(
        self,
        service: InterimAnalysisService,
        experiment_id: UUID,
        metric_data_significant: MetricData,
    ):
        """Test interim analysis with significant effect."""
        with patch.object(service, "_persist_analysis", new_callable=AsyncMock):
            result = await service.perform_interim_analysis(
                experiment_id=experiment_id,
                analysis_number=2,
                metric_data=metric_data_significant,
                target_sample_size=2000,
                target_effect=0.05,
            )

            assert result.experiment_id == experiment_id
            assert result.analysis_number == 2
            assert result.effect_estimate > 0  # Treatment > control
            assert result.p_value < 0.05  # Should be significant

    @pytest.mark.asyncio
    async def test_analysis_with_not_significant_effect(
        self,
        service: InterimAnalysisService,
        experiment_id: UUID,
        metric_data_not_significant: MetricData,
    ):
        """Test interim analysis with no significant effect."""
        with patch.object(service, "_persist_analysis", new_callable=AsyncMock):
            result = await service.perform_interim_analysis(
                experiment_id=experiment_id,
                analysis_number=1,
                metric_data=metric_data_not_significant,
                target_sample_size=1000,
            )

            assert result.p_value > 0.05  # Should not be significant
            assert abs(result.effect_estimate) < 0.02  # Small effect

    @pytest.mark.asyncio
    async def test_analysis_calculates_statistics(
        self,
        service: InterimAnalysisService,
        experiment_id: UUID,
        metric_data_significant: MetricData,
    ):
        """Test interim analysis calculates all statistics."""
        with patch.object(service, "_persist_analysis", new_callable=AsyncMock):
            result = await service.perform_interim_analysis(
                experiment_id=experiment_id,
                analysis_number=1,
                metric_data=metric_data_significant,
                target_sample_size=2000,
            )

            # Check all statistics are computed
            assert result.control_mean != 0
            assert result.treatment_mean != 0
            assert result.standard_error > 0
            assert result.test_statistic != 0
            assert result.effect_ci_lower < result.effect_estimate < result.effect_ci_upper

    @pytest.mark.asyncio
    async def test_analysis_information_fraction(
        self,
        service: InterimAnalysisService,
        experiment_id: UUID,
        metric_data_significant: MetricData,
    ):
        """Test information fraction calculation."""
        with patch.object(service, "_persist_analysis", new_callable=AsyncMock):
            result = await service.perform_interim_analysis(
                experiment_id=experiment_id,
                analysis_number=1,
                metric_data=metric_data_significant,
                target_sample_size=2000,
            )

            expected_fraction = 1000 / 2000  # 500 control + 500 treatment
            assert abs(result.information_fraction - expected_fraction) < 0.01

    @pytest.mark.asyncio
    async def test_analysis_without_target_sample_size(
        self,
        service: InterimAnalysisService,
        experiment_id: UUID,
        metric_data_significant: MetricData,
    ):
        """Test analysis when target sample size not provided."""
        with patch.object(service, "_persist_analysis", new_callable=AsyncMock):
            result = await service.perform_interim_analysis(
                experiment_id=experiment_id,
                analysis_number=2,
                metric_data=metric_data_significant,
                target_sample_size=None,  # Not provided
            )

            # Info fraction should be based on analysis number
            expected_fraction = 2 / 3  # Analysis 2 of 3
            assert abs(result.information_fraction - expected_fraction) < 0.01
            assert result.conditional_power is None  # Can't compute without target

    @pytest.mark.asyncio
    async def test_analysis_metrics_snapshot(
        self,
        service: InterimAnalysisService,
        experiment_id: UUID,
        metric_data_significant: MetricData,
    ):
        """Test metrics snapshot is populated."""
        with patch.object(service, "_persist_analysis", new_callable=AsyncMock):
            result = await service.perform_interim_analysis(
                experiment_id=experiment_id,
                analysis_number=1,
                metric_data=metric_data_significant,
                target_sample_size=2000,
            )

            assert "primary_metric" in result.metrics_snapshot
            assert "sample_sizes" in result.metrics_snapshot
            assert result.metrics_snapshot["primary_metric"]["name"] == "conversion_rate"

    @pytest.mark.asyncio
    async def test_analysis_conditional_power_computed(
        self,
        service: InterimAnalysisService,
        experiment_id: UUID,
        metric_data_significant: MetricData,
    ):
        """Test conditional power is computed when possible."""
        with patch.object(service, "_persist_analysis", new_callable=AsyncMock):
            result = await service.perform_interim_analysis(
                experiment_id=experiment_id,
                analysis_number=1,
                metric_data=metric_data_significant,
                target_sample_size=2000,
                target_effect=0.05,
            )

            assert result.conditional_power is not None
            assert 0 <= result.conditional_power <= 1
            assert result.predictive_probability is not None


# =============================================================================
# TEST PERSIST ANALYSIS
# =============================================================================


class TestPersistAnalysis:
    """Tests for _persist_analysis method."""

    @pytest.mark.asyncio
    async def test_persist_calls_repository(self, service: InterimAnalysisService):
        """Test persist analysis calls repository correctly."""
        result = InterimAnalysisResult(
            experiment_id=uuid4(),
            analysis_number=1,
            performed_at=datetime.now(timezone.utc),
            information_fraction=0.5,
            sample_size_control=100,
            sample_size_treatment=100,
            total_sample_size=200,
            target_sample_size=400,
            effect_estimate=0.05,
            standard_error=0.02,
            test_statistic=2.5,
            p_value=0.01,
            effect_ci_lower=0.01,
            effect_ci_upper=0.09,
            alpha_boundary=0.025,
            alpha_spent=0.025,
            cumulative_alpha_spent=0.025,
            conditional_power=0.8,
            predictive_probability=0.75,
            decision=StoppingDecision.CONTINUE,
            decision_rationale="Test rationale",
            metrics_snapshot={"test": "data"},
        )

        with patch("src.repositories.ab_experiment.ABExperimentRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            await service._persist_analysis(result)

            mock_repo.record_interim_analysis.assert_called_once()
            call_kwargs = mock_repo.record_interim_analysis.call_args.kwargs
            assert call_kwargs["experiment_id"] == result.experiment_id
            assert call_kwargs["analysis_number"] == 1


# =============================================================================
# TEST GET ANALYSIS HISTORY
# =============================================================================


class TestGetAnalysisHistory:
    """Tests for get_analysis_history method."""

    @pytest.mark.asyncio
    async def test_get_history_returns_results(self, service: InterimAnalysisService):
        """Test get_analysis_history returns results."""
        experiment_id = uuid4()

        mock_analysis = MagicMock()
        mock_analysis.experiment_id = experiment_id
        mock_analysis.analysis_number = 1
        mock_analysis.performed_at = datetime.now(timezone.utc)
        mock_analysis.information_fraction = 0.5
        mock_analysis.sample_size_at_analysis = 200
        mock_analysis.target_sample_size = 400
        mock_analysis.effect_estimate = 0.05
        mock_analysis.standard_error = 0.02
        mock_analysis.test_statistic = 2.5
        mock_analysis.p_value = 0.01
        mock_analysis.effect_ci_lower = 0.01
        mock_analysis.effect_ci_upper = 0.09
        mock_analysis.adjusted_alpha = 0.025
        mock_analysis.alpha_spent = 0.025
        mock_analysis.cumulative_alpha_spent = 0.025
        mock_analysis.conditional_power = 0.8
        mock_analysis.predictive_probability = 0.75
        mock_analysis.decision = "continue"
        mock_analysis.decision_rationale = "Continue experiment"
        mock_analysis.metrics_snapshot = {}

        with patch("src.repositories.ab_experiment.ABExperimentRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo.get_interim_analyses.return_value = [mock_analysis]
            mock_repo_class.return_value = mock_repo

            results = await service.get_analysis_history(experiment_id)

            assert len(results) == 1
            assert results[0].experiment_id == experiment_id
            assert results[0].analysis_number == 1
            assert results[0].decision == StoppingDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_get_history_empty(self, service: InterimAnalysisService):
        """Test get_analysis_history with no history."""
        experiment_id = uuid4()

        with patch("src.repositories.ab_experiment.ABExperimentRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo.get_interim_analyses.return_value = []
            mock_repo_class.return_value = mock_repo

            results = await service.get_analysis_history(experiment_id)

            assert len(results) == 0


# =============================================================================
# TEST RECOMMEND NEXT ANALYSIS TIMING
# =============================================================================


class TestRecommendNextAnalysisTiming:
    """Tests for recommend_next_analysis_timing method."""

    def test_recommend_next_analysis(self, service: InterimAnalysisService):
        """Test recommendation for next interim analysis."""
        recommendation = service.recommend_next_analysis_timing(
            current_n=100,
            target_n=300,
            current_analysis_number=1,
        )

        assert recommendation["next_analysis_number"] == 2
        assert recommendation["target_information_fraction"] == 2/3
        assert recommendation["target_sample_size"] == 200
        assert recommendation["samples_needed"] == 100
        assert recommendation["current_progress"] == 100/300

    def test_recommend_no_more_analyses(self, service: InterimAnalysisService):
        """Test recommendation when all analyses complete."""
        recommendation = service.recommend_next_analysis_timing(
            current_n=300,
            target_n=300,
            current_analysis_number=3,  # All 3 planned analyses done
        )

        assert recommendation["recommendation"] == "no_more_analyses"
        assert "complete" in recommendation["reason"].lower()

    def test_recommend_samples_needed_zero(self, service: InterimAnalysisService):
        """Test samples needed is zero when already past target."""
        recommendation = service.recommend_next_analysis_timing(
            current_n=250,
            target_n=300,
            current_analysis_number=2,
        )

        # Already past target for analysis 3 (300 * 3/3 = 300)
        # But current is 250, target for next is 300, so need 50
        assert recommendation["samples_needed"] == 50

    def test_recommend_with_zero_target(self, service: InterimAnalysisService):
        """Test recommendation handles zero target gracefully."""
        recommendation = service.recommend_next_analysis_timing(
            current_n=100,
            target_n=0,  # Edge case
            current_analysis_number=1,
        )

        assert recommendation["current_progress"] == 0


# =============================================================================
# TEST FACTORY FUNCTION
# =============================================================================


class TestFactoryFunction:
    """Tests for get_interim_analysis_service factory."""

    def test_factory_default_config(self):
        """Test factory with default config."""
        service = get_interim_analysis_service()
        assert isinstance(service, InterimAnalysisService)
        assert service.config.spending_function == SpendingFunction.OBRIEN_FLEMING

    def test_factory_custom_config(self):
        """Test factory with custom config."""
        config = InterimAnalysisConfig(
            total_alpha=0.10,
            spending_function=SpendingFunction.POCOCK,
            num_planned_analyses=5,
        )
        service = get_interim_analysis_service(config)

        assert service.config.total_alpha == 0.10
        assert service.config.spending_function == SpendingFunction.POCOCK
        assert service.config.num_planned_analyses == 5


# =============================================================================
# TEST EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_init_without_config(self):
        """Test service initialization without config."""
        service = InterimAnalysisService()
        assert service.config is not None
        assert service.config.total_alpha == 0.05

    @pytest.mark.asyncio
    async def test_analysis_with_small_sample(self, service: InterimAnalysisService):
        """Test analysis with very small sample sizes."""
        metric_data = MetricData(
            name="test",
            control_values=np.array([1.0, 2.0]),
            treatment_values=np.array([3.0, 4.0]),
        )

        with patch.object(service, "_persist_analysis", new_callable=AsyncMock):
            result = await service.perform_interim_analysis(
                experiment_id=uuid4(),
                analysis_number=1,
                metric_data=metric_data,
                target_sample_size=100,
            )

            # Should still produce results
            assert result.total_sample_size == 4
            assert result.p_value >= 0  # Valid p-value

    @pytest.mark.asyncio
    async def test_analysis_with_identical_groups(self, service: InterimAnalysisService):
        """Test analysis when control and treatment are identical."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metric_data = MetricData(
            name="test",
            control_values=values,
            treatment_values=values.copy(),
        )

        with patch.object(service, "_persist_analysis", new_callable=AsyncMock):
            result = await service.perform_interim_analysis(
                experiment_id=uuid4(),
                analysis_number=1,
                metric_data=metric_data,
                target_sample_size=100,
            )

            assert result.effect_estimate == 0.0
            assert result.p_value == 1.0  # Not significant

    def test_conditional_power_negative_variance(self, service: InterimAnalysisService):
        """Test conditional power with edge case variance."""
        power = service.calculate_conditional_power(
            current_effect=0.1,
            current_variance=-0.01,  # Invalid
            target_effect=0.1,
            current_n=100,
            target_n=200,
        )
        assert power == 0.0

    @pytest.mark.asyncio
    async def test_analysis_preserves_experiment_id(
        self,
        service: InterimAnalysisService,
        metric_data_significant: MetricData,
    ):
        """Test experiment ID is preserved through analysis."""
        exp_id = uuid4()

        with patch.object(service, "_persist_analysis", new_callable=AsyncMock):
            result = await service.perform_interim_analysis(
                experiment_id=exp_id,
                analysis_number=1,
                metric_data=metric_data_significant,
            )

            assert result.experiment_id == exp_id

    def test_boundary_calculation_consistency(self, service: InterimAnalysisService):
        """Test boundary calculations are consistent across calls."""
        boundary1 = service.calculate_obrien_fleming_boundary(0.5, 0.05, 3)
        boundary2 = service.calculate_obrien_fleming_boundary(0.5, 0.05, 3)
        assert boundary1 == boundary2
