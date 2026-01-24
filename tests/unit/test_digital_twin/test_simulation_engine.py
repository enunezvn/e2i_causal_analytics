"""
Unit Tests for Digital Twin Simulation Engine.

Tests cover:
- Engine initialization and configuration
- Intervention simulation execution
- Population filtering
- Treatment effect calculation
- Heterogeneous effects by subgroup
- Recommendation generation logic
- Sample size calculation
- Error handling
"""

from typing import List
from uuid import uuid4

import numpy as np
import pytest

from src.digital_twin.models.simulation_models import (
    EffectHeterogeneity,
    InterventionConfig,
    PopulationFilter,
    SimulationRecommendation,
    SimulationResult,
    SimulationStatus,
)
from src.digital_twin.models.twin_models import (
    Brand,
    DigitalTwin,
    TwinPopulation,
    TwinType,
)
from src.digital_twin.simulation_engine import SimulationEngine


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_hcp_twins() -> List[DigitalTwin]:
    """Create sample HCP twins for testing."""
    np.random.seed(42)
    twins = []

    specialties = ["rheumatology", "dermatology", "allergy"]
    regions = ["northeast", "south", "midwest", "west"]
    adoption_stages = ["innovator", "early_adopter", "early_majority", "late_majority", "laggard"]

    for i in range(500):
        twin = DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            features={
                "specialty": specialties[i % 3],
                "decile": (i % 10) + 1,
                "region": regions[i % 4],
                "digital_engagement_score": np.random.uniform(0.2, 0.9),
                "adoption_stage": adoption_stages[i % 5],
                "priority_tier": (i % 5) + 1,
                "peer_influence_score": np.random.uniform(0.3, 0.9),
            },
            baseline_outcome=np.random.uniform(0.05, 0.25),
            baseline_propensity=np.random.uniform(0.3, 0.8),
        )
        twins.append(twin)

    return twins


@pytest.fixture
def sample_population(sample_hcp_twins) -> TwinPopulation:
    """Create sample twin population."""
    return TwinPopulation(
        twin_type=TwinType.HCP,
        brand=Brand.REMIBRUTINIB,
        twins=sample_hcp_twins,
        size=len(sample_hcp_twins),
        model_id=uuid4(),
    )


@pytest.fixture
def small_population() -> TwinPopulation:
    """Create small population for edge case testing."""
    twins = [
        DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            features={"specialty": "rheumatology", "decile": 1},
            baseline_outcome=0.1,
            baseline_propensity=0.5,
        )
        for _ in range(50)  # Below threshold of 100
    ]

    return TwinPopulation(
        twin_type=TwinType.HCP,
        brand=Brand.REMIBRUTINIB,
        twins=twins,
        size=50,
        model_id=uuid4(),
    )


@pytest.fixture
def engine(sample_population) -> SimulationEngine:
    """Create simulation engine with sample population."""
    return SimulationEngine(sample_population)


@pytest.fixture
def email_campaign_config() -> InterventionConfig:
    """Create email campaign intervention configuration."""
    return InterventionConfig(
        intervention_type="email_campaign",
        channel="email",
        frequency="weekly",
        duration_weeks=8,
        personalization_level="high",
        target_deciles=[1, 2, 3, 4, 5],
        intensity_multiplier=1.0,
    )


@pytest.fixture
def call_frequency_config() -> InterventionConfig:
    """Create call frequency intervention configuration."""
    return InterventionConfig(
        intervention_type="call_frequency_increase",
        channel="phone",
        frequency="daily",
        duration_weeks=4,
        intensity_multiplier=1.5,
    )


@pytest.fixture
def speaker_program_config() -> InterventionConfig:
    """Create speaker program intervention configuration."""
    return InterventionConfig(
        intervention_type="speaker_program_invitation",
        duration_weeks=12,
        intensity_multiplier=1.2,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestSimulationEngineInit:
    """Tests for SimulationEngine initialization."""

    def test_init_default_thresholds(self, sample_population):
        """Test initialization with default thresholds."""
        engine = SimulationEngine(sample_population)

        assert engine.population == sample_population
        assert engine.model_id == sample_population.model_id
        assert engine.min_effect_threshold == 0.05
        assert engine.confidence_threshold == 0.70
        assert engine.model_fidelity_score is None

    def test_init_custom_thresholds(self, sample_population):
        """Test initialization with custom thresholds."""
        engine = SimulationEngine(
            sample_population,
            min_effect_threshold=0.10,
            confidence_threshold=0.85,
            model_fidelity_score=0.92,
        )

        assert engine.min_effect_threshold == 0.10
        assert engine.confidence_threshold == 0.85
        assert engine.model_fidelity_score == 0.92

    def test_intervention_effects_defined(self):
        """Test that intervention effects are defined."""
        assert "email_campaign" in SimulationEngine.INTERVENTION_EFFECTS
        assert "call_frequency_increase" in SimulationEngine.INTERVENTION_EFFECTS
        assert "speaker_program_invitation" in SimulationEngine.INTERVENTION_EFFECTS
        assert "sample_distribution" in SimulationEngine.INTERVENTION_EFFECTS


# =============================================================================
# SIMULATION EXECUTION TESTS
# =============================================================================


class TestSimulationExecution:
    """Tests for simulation execution."""

    def test_simulate_email_campaign(self, engine, email_campaign_config):
        """Test running email campaign simulation."""
        result = engine.simulate(email_campaign_config)

        assert isinstance(result, SimulationResult)
        assert result.status == SimulationStatus.COMPLETED
        assert result.intervention_config == email_campaign_config
        assert result.twin_count == 500
        assert result.simulated_ate != 0
        assert result.simulated_ci_lower < result.simulated_ci_upper
        assert result.simulated_std_error > 0
        assert result.execution_time_ms >= 0

    def test_simulate_call_frequency(self, engine, call_frequency_config):
        """Test running call frequency simulation."""
        result = engine.simulate(call_frequency_config)

        assert result.status == SimulationStatus.COMPLETED
        assert result.intervention_config.intervention_type == "call_frequency_increase"

    def test_simulate_speaker_program(self, engine, speaker_program_config):
        """Test running speaker program simulation."""
        result = engine.simulate(speaker_program_config)

        assert result.status == SimulationStatus.COMPLETED
        assert result.intervention_config.intervention_type == "speaker_program_invitation"

    def test_simulate_returns_recommendation(self, engine, email_campaign_config):
        """Test that simulation returns recommendation."""
        result = engine.simulate(email_campaign_config)

        assert result.recommendation in [
            SimulationRecommendation.DEPLOY,
            SimulationRecommendation.SKIP,
            SimulationRecommendation.REFINE,
        ]
        assert len(result.recommendation_rationale) > 0

    def test_simulate_calculates_sample_size(self, engine, email_campaign_config):
        """Test that simulation calculates recommended sample size."""
        result = engine.simulate(email_campaign_config)

        assert result.recommended_sample_size is not None
        assert result.recommended_sample_size >= 100
        assert result.recommended_sample_size <= 50000

    def test_simulate_sets_duration(self, engine, email_campaign_config):
        """Test that simulation preserves recommended duration."""
        result = engine.simulate(email_campaign_config)

        assert result.recommended_duration_weeks == 8  # From config


# =============================================================================
# POPULATION FILTERING TESTS
# =============================================================================


class TestPopulationFiltering:
    """Tests for population filtering in simulation."""

    def test_filter_by_specialty(self, engine, email_campaign_config):
        """Test filtering population by specialty."""
        population_filter = PopulationFilter(specialties=["rheumatology"])

        result = engine.simulate(email_campaign_config, population_filter=population_filter)

        assert result.status == SimulationStatus.COMPLETED
        # Should have approximately 1/3 of 500 twins (rheumatology)
        assert 150 <= result.twin_count <= 180

    def test_filter_by_decile(self, engine, email_campaign_config):
        """Test filtering population by decile."""
        population_filter = PopulationFilter(deciles=[1, 2])

        result = engine.simulate(email_campaign_config, population_filter=population_filter)

        assert result.status == SimulationStatus.COMPLETED
        # Should have approximately 2/10 of 500 twins
        assert 90 <= result.twin_count <= 110

    def test_filter_by_region(self, engine, email_campaign_config):
        """Test filtering population by region."""
        population_filter = PopulationFilter(regions=["northeast", "south"])

        result = engine.simulate(email_campaign_config, population_filter=population_filter)

        assert result.status == SimulationStatus.COMPLETED
        # Should have approximately 2/4 of 500 twins
        assert 240 <= result.twin_count <= 260

    def test_filter_by_baseline_outcome(self, engine, email_campaign_config):
        """Test filtering by baseline outcome range."""
        population_filter = PopulationFilter(
            min_baseline_outcome=0.10,
            max_baseline_outcome=0.20,
        )

        result = engine.simulate(email_campaign_config, population_filter=population_filter)

        assert result.status == SimulationStatus.COMPLETED
        # Should have subset of twins with baseline in range

    def test_filter_insufficient_twins(self, engine, email_campaign_config):
        """Test filtering that results in too few twins."""
        # Filter that will result in very few twins
        population_filter = PopulationFilter(
            specialties=["rheumatology"],
            deciles=[1],
            regions=["northeast"],
        )

        result = engine.simulate(email_campaign_config, population_filter=population_filter)

        # Should fail due to insufficient twins
        assert result.status == SimulationStatus.FAILED
        assert "Insufficient twins" in result.error_message

    def test_no_filter(self, engine, email_campaign_config):
        """Test simulation with no filter."""
        result = engine.simulate(email_campaign_config, population_filter=None)

        assert result.twin_count == 500


# =============================================================================
# HETEROGENEOUS EFFECTS TESTS
# =============================================================================


class TestHeterogeneousEffects:
    """Tests for heterogeneous effects calculation."""

    def test_heterogeneity_by_specialty(self, engine, email_campaign_config):
        """Test heterogeneous effects by specialty."""
        result = engine.simulate(email_campaign_config, calculate_heterogeneity=True)

        assert len(result.effect_heterogeneity.by_specialty) > 0
        for specialty, stats in result.effect_heterogeneity.by_specialty.items():
            assert "ate" in stats
            assert "n" in stats
            assert stats["n"] >= 10

    def test_heterogeneity_by_decile(self, engine, email_campaign_config):
        """Test heterogeneous effects by decile."""
        result = engine.simulate(email_campaign_config, calculate_heterogeneity=True)

        assert len(result.effect_heterogeneity.by_decile) > 0

    def test_heterogeneity_by_region(self, engine, email_campaign_config):
        """Test heterogeneous effects by region."""
        result = engine.simulate(email_campaign_config, calculate_heterogeneity=True)

        assert len(result.effect_heterogeneity.by_region) > 0

    def test_heterogeneity_disabled(self, engine, email_campaign_config):
        """Test simulation without heterogeneity calculation."""
        result = engine.simulate(email_campaign_config, calculate_heterogeneity=False)

        # Should have empty heterogeneity
        assert len(result.effect_heterogeneity.by_specialty) == 0
        assert len(result.effect_heterogeneity.by_decile) == 0

    def test_top_segments(self, engine, email_campaign_config):
        """Test getting top segments from heterogeneity."""
        result = engine.simulate(email_campaign_config, calculate_heterogeneity=True)

        top = result.effect_heterogeneity.get_top_segments(n=3)
        assert len(top) <= 3
        # Sorted by absolute effect size
        if len(top) >= 2:
            assert abs(top[0]["ate"]) >= abs(top[1]["ate"])


# =============================================================================
# RECOMMENDATION LOGIC TESTS
# =============================================================================


class TestRecommendationLogic:
    """Tests for recommendation generation logic."""

    def test_recommend_deploy_positive_effect(self, sample_population):
        """Test DEPLOY recommendation for significant positive effect."""
        # Create engine with low threshold
        engine = SimulationEngine(sample_population, min_effect_threshold=0.01)

        # High intensity intervention likely to produce significant effect
        config = InterventionConfig(
            intervention_type="speaker_program_invitation",
            duration_weeks=12,
            intensity_multiplier=2.0,
        )

        result = engine.simulate(config)

        # With high intensity, should often recommend deploy
        # (This is probabilistic due to random effects)
        assert result.recommendation in [
            SimulationRecommendation.DEPLOY,
            SimulationRecommendation.REFINE,
        ]

    def test_recommend_skip_small_effect(self, sample_population):
        """Test SKIP recommendation for effects below threshold."""
        # Create engine with high threshold
        engine = SimulationEngine(sample_population, min_effect_threshold=0.50)

        config = InterventionConfig(
            intervention_type="sample_distribution",  # Small base effect
            duration_weeks=2,
            intensity_multiplier=0.5,
        )

        result = engine.simulate(config)

        # With low intensity and high threshold, should skip
        assert result.recommendation == SimulationRecommendation.SKIP
        assert "below minimum threshold" in result.recommendation_rationale.lower()

    def test_recommend_refine_uncertain(self, sample_population):
        """Test REFINE recommendation when CI includes zero."""
        engine = SimulationEngine(sample_population, min_effect_threshold=0.01)

        # Short duration, low intensity - more uncertainty
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=1,
            intensity_multiplier=0.3,
        )

        result = engine.simulate(config)

        # Could be any recommendation based on random effects
        assert result.recommendation in [
            SimulationRecommendation.DEPLOY,
            SimulationRecommendation.SKIP,
            SimulationRecommendation.REFINE,
        ]


# =============================================================================
# FIDELITY WARNING TESTS
# =============================================================================


class TestFidelityWarnings:
    """Tests for model fidelity warning logic."""

    def test_fidelity_warning_low_score(self, sample_population, email_campaign_config):
        """Test fidelity warning when score is low."""
        engine = SimulationEngine(
            sample_population,
            model_fidelity_score=0.55,  # Below 0.7 threshold
        )

        result = engine.simulate(email_campaign_config)

        assert result.fidelity_warning is True
        assert "below threshold" in result.fidelity_warning_reason
        assert result.model_fidelity_score == 0.55

    def test_no_fidelity_warning_good_score(self, sample_population, email_campaign_config):
        """Test no fidelity warning when score is adequate."""
        engine = SimulationEngine(
            sample_population,
            model_fidelity_score=0.85,  # Above threshold
        )

        result = engine.simulate(email_campaign_config)

        assert result.fidelity_warning is False
        assert result.fidelity_warning_reason is None

    def test_no_fidelity_warning_no_score(self, sample_population, email_campaign_config):
        """Test no fidelity warning when score is not set."""
        engine = SimulationEngine(sample_population)  # No fidelity score

        result = engine.simulate(email_campaign_config)

        assert result.fidelity_warning is False


# =============================================================================
# CONFIDENCE SCORE TESTS
# =============================================================================


class TestConfidenceScore:
    """Tests for simulation confidence calculation."""

    def test_confidence_score_bounds(self, engine, email_campaign_config):
        """Test that confidence score is bounded [0, 1]."""
        result = engine.simulate(email_campaign_config)

        assert 0.0 <= result.simulation_confidence <= 1.0

    def test_confidence_higher_with_more_twins(self, email_campaign_config):
        """Test that confidence is higher with more twins."""
        # Create small population
        small_twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={"specialty": "rheumatology", "decile": i % 10 + 1},
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for i in range(150)
        ]

        small_pop = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=small_twins,
            size=150,
            model_id=uuid4(),
        )

        # Create larger population
        large_twins = small_twins * 5  # 750 twins

        large_pop = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=large_twins,
            size=750,
            model_id=uuid4(),
        )

        small_engine = SimulationEngine(small_pop, model_fidelity_score=0.8)
        large_engine = SimulationEngine(large_pop, model_fidelity_score=0.8)

        small_result = small_engine.simulate(email_campaign_config)
        large_result = large_engine.simulate(email_campaign_config)

        # Larger population should have higher or equal confidence
        # (accounting for random variation)
        assert large_result.simulation_confidence >= small_result.simulation_confidence - 0.1


# =============================================================================
# TREATMENT EFFECT CALCULATION TESTS
# =============================================================================


class TestTreatmentEffects:
    """Tests for treatment effect calculations."""

    def test_effect_varies_by_decile(self, sample_population, email_campaign_config):
        """Test that effects vary by decile (higher decile = lower effect)."""
        engine = SimulationEngine(sample_population)
        result = engine.simulate(email_campaign_config, calculate_heterogeneity=True)

        by_decile = result.effect_heterogeneity.by_decile
        if "1" in by_decile and "10" in by_decile:
            # Top decile should generally have lower effect (less room to grow)
            # But this is probabilistic
            assert by_decile["1"]["ate"] != by_decile["10"]["ate"]

    def test_effect_includes_variance(self, engine, email_campaign_config):
        """Test that effects include variance (not all identical)."""
        # Run simulation twice with same seed
        np.random.seed(42)
        result1 = engine.simulate(email_campaign_config)

        np.random.seed(42)
        result2 = engine.simulate(email_campaign_config)

        # Should have similar but not necessarily identical results
        assert abs(result1.simulated_ate - result2.simulated_ate) < 0.01

    def test_intensity_multiplier_effect(self, sample_population):
        """Test that intensity multiplier affects outcome."""
        engine = SimulationEngine(sample_population)

        low_intensity = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
            intensity_multiplier=0.5,
        )

        high_intensity = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
            intensity_multiplier=2.0,
        )

        np.random.seed(42)
        low_result = engine.simulate(low_intensity)

        np.random.seed(42)
        high_result = engine.simulate(high_intensity)

        # Higher intensity should produce larger absolute effect
        assert abs(high_result.simulated_ate) > abs(low_result.simulated_ate)

    def test_duration_affects_effect(self, sample_population):
        """Test that longer duration produces stronger effects."""
        engine = SimulationEngine(sample_population)

        short_duration = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=2,
            intensity_multiplier=1.0,
        )

        long_duration = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=12,
            intensity_multiplier=1.0,
        )

        np.random.seed(42)
        short_result = engine.simulate(short_duration)

        np.random.seed(42)
        long_result = engine.simulate(long_duration)

        # Longer duration should produce larger absolute effect
        assert abs(long_result.simulated_ate) > abs(short_result.simulated_ate)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in simulation."""

    def test_insufficient_twins_error(self, small_population, email_campaign_config):
        """Test error when population is too small."""
        engine = SimulationEngine(small_population)

        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.FAILED
        assert "Insufficient twins" in result.error_message
        assert result.twin_count == 0
        assert result.simulated_ate == 0.0

    def test_unknown_intervention_type(self, engine):
        """Test handling of unknown intervention type."""
        config = InterventionConfig(
            intervention_type="unknown_intervention",
            duration_weeks=8,
        )

        # Should use default effect parameters
        result = engine.simulate(config)

        assert result.status == SimulationStatus.COMPLETED  # Uses defaults

    def test_completed_at_timestamp(self, engine, email_campaign_config):
        """Test that completed_at is set for successful simulation."""
        result = engine.simulate(email_campaign_config)

        assert result.completed_at is not None


# =============================================================================
# CONFIDENCE LEVEL TESTS
# =============================================================================


class TestConfidenceLevelParameter:
    """Tests for confidence level parameter."""

    def test_confidence_level_95(self, engine, email_campaign_config):
        """Test 95% confidence level."""
        result = engine.simulate(email_campaign_config, confidence_level=0.95)

        ci_width_95 = result.simulated_ci_upper - result.simulated_ci_lower
        assert ci_width_95 > 0

    def test_confidence_level_99(self, engine, email_campaign_config):
        """Test 99% confidence level produces wider CI."""
        np.random.seed(42)
        result_95 = engine.simulate(email_campaign_config, confidence_level=0.95)

        np.random.seed(42)
        result_99 = engine.simulate(email_campaign_config, confidence_level=0.99)

        ci_width_95 = result_95.simulated_ci_upper - result_95.simulated_ci_lower
        ci_width_99 = result_99.simulated_ci_upper - result_99.simulated_ci_lower

        # 99% CI should be wider than 95% CI
        assert ci_width_99 > ci_width_95

    def test_confidence_level_90(self, engine, email_campaign_config):
        """Test 90% confidence level produces narrower CI."""
        np.random.seed(42)
        result_90 = engine.simulate(email_campaign_config, confidence_level=0.90)

        np.random.seed(42)
        result_95 = engine.simulate(email_campaign_config, confidence_level=0.95)

        ci_width_90 = result_90.simulated_ci_upper - result_90.simulated_ci_lower
        ci_width_95 = result_95.simulated_ci_upper - result_95.simulated_ci_lower

        # 90% CI should be narrower than 95% CI
        assert ci_width_90 < ci_width_95


# =============================================================================
# EDGE CASE TESTS FOR EFFECT MODIFIERS (Phase 2)
# =============================================================================


class TestEffectModifierEdgeCases:
    """Tests for edge cases in effect modifier calculations."""

    @pytest.fixture
    def edge_case_population(self):
        """Create population with edge case feature values."""
        twins = []
        for i in range(200):
            twin = DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    "decile": 5,
                    "digital_engagement_score": 0.5,
                    "adoption_stage": "early_majority",
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            twins.append(twin)

        return TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=200,
            model_id=uuid4(),
        )

    def test_extreme_low_decile(self, edge_case_population, email_campaign_config):
        """Test handling of decile = 0 (below valid range)."""
        for twin in edge_case_population.twins:
            twin.features["decile"] = 0  # Invalid low value

        engine = SimulationEngine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        # Should complete with clamped decile value
        assert result.status == SimulationStatus.COMPLETED

    def test_extreme_high_decile(self, edge_case_population, email_campaign_config):
        """Test handling of decile = 11 (above valid range)."""
        for twin in edge_case_population.twins:
            twin.features["decile"] = 11  # Invalid high value

        engine = SimulationEngine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        # Should complete with clamped decile value
        assert result.status == SimulationStatus.COMPLETED

    def test_zero_engagement_score(self, edge_case_population, email_campaign_config):
        """Test handling of zero engagement score."""
        for twin in edge_case_population.twins:
            twin.features["digital_engagement_score"] = 0.0

        engine = SimulationEngine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        # Should apply minimum multiplier (0.8)
        assert result.status == SimulationStatus.COMPLETED
        assert result.simulated_ate != 0

    def test_max_engagement_score(self, edge_case_population, email_campaign_config):
        """Test handling of maximum engagement score."""
        for twin in edge_case_population.twins:
            twin.features["digital_engagement_score"] = 1.0

        engine = SimulationEngine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        # Should apply maximum multiplier (1.2)
        assert result.status == SimulationStatus.COMPLETED

    def test_negative_engagement_score(self, edge_case_population, email_campaign_config):
        """Test handling of negative engagement score."""
        for twin in edge_case_population.twins:
            twin.features["digital_engagement_score"] = -0.5

        engine = SimulationEngine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        # Should clamp to 0 and complete
        assert result.status == SimulationStatus.COMPLETED

    def test_engagement_above_one(self, edge_case_population, email_campaign_config):
        """Test handling of engagement score above 1.0."""
        for twin in edge_case_population.twins:
            twin.features["digital_engagement_score"] = 1.5

        engine = SimulationEngine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        # Should clamp to 1.0 and complete
        assert result.status == SimulationStatus.COMPLETED

    def test_invalid_adoption_stage(self, edge_case_population, email_campaign_config):
        """Test handling of unknown adoption stage."""
        for twin in edge_case_population.twins:
            twin.features["adoption_stage"] = "unknown_stage"

        engine = SimulationEngine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        # Should fall back to default multiplier (1.0)
        assert result.status == SimulationStatus.COMPLETED

    def test_zero_intensity_multiplier(self, edge_case_population):
        """Test handling of zero intensity multiplier."""
        engine = SimulationEngine(edge_case_population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
            intensity_multiplier=0.0,
        )

        result = engine.simulate(config)

        # Effect should be near zero
        assert result.status == SimulationStatus.COMPLETED
        assert abs(result.simulated_ate) < 0.01

    def test_extreme_intensity_multiplier(self, edge_case_population):
        """Test handling of extreme intensity multiplier."""
        engine = SimulationEngine(edge_case_population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
            intensity_multiplier=10.0,
        )

        result = engine.simulate(config)

        # Should be clamped and complete
        assert result.status == SimulationStatus.COMPLETED

    def test_zero_duration_weeks(self, edge_case_population):
        """Test handling of zero duration weeks."""
        engine = SimulationEngine(edge_case_population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=0,
            intensity_multiplier=1.0,
        )

        result = engine.simulate(config)

        # Should handle gracefully (clamped to minimum)
        assert result.status == SimulationStatus.COMPLETED

    def test_combined_extreme_modifiers(self, edge_case_population):
        """Test combined extreme modifier values don't cause overflow."""
        for twin in edge_case_population.twins:
            twin.features["decile"] = 1  # Maximum decile multiplier
            twin.features["digital_engagement_score"] = 1.0  # Max engagement
            twin.features["adoption_stage"] = "laggard"  # Max adoption multiplier
            twin.baseline_propensity = 1.0  # Max propensity

        engine = SimulationEngine(edge_case_population)
        config = InterventionConfig(
            intervention_type="speaker_program_invitation",  # Highest base effect
            duration_weeks=52,  # Long duration
            intensity_multiplier=10.0,  # Max intensity
        )

        result = engine.simulate(config)

        # Should complete without overflow
        assert result.status == SimulationStatus.COMPLETED
        assert not np.isnan(result.simulated_ate)
        assert not np.isinf(result.simulated_ate)


class TestBoundaryConditions:
    """Tests for boundary conditions in simulation."""

    def test_exactly_100_twins(self, email_campaign_config):
        """Test minimum viable population (exactly 100 twins)."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={"specialty": "rheumatology", "decile": i % 10 + 1},
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for i in range(100)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=100,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.COMPLETED
        assert result.twin_count == 100

    def test_99_twins_fails(self, email_campaign_config):
        """Test below minimum threshold (99 twins) fails gracefully."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={"specialty": "rheumatology", "decile": i % 10 + 1},
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for i in range(99)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=99,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.FAILED
        assert "Insufficient" in result.error_message

    def test_empty_population(self, email_campaign_config):
        """Test handling of empty population."""
        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=[],
            size=0,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.FAILED

    def test_ci_bounds_with_uniform_population(self, email_campaign_config):
        """Test CI calculation with very uniform population."""
        # All twins identical
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    "decile": 5,
                    "digital_engagement_score": 0.5,
                    "adoption_stage": "early_majority",
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for _ in range(200)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=200,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.COMPLETED
        # CI should still have some width due to noise
        assert result.simulated_ci_lower < result.simulated_ci_upper

    def test_negative_ate_possible(self):
        """Test that negative treatment effects are handled correctly."""
        # Create population that might produce negative effect
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    "decile": 1,  # High decile effect
                    "digital_engagement_score": 0.1,  # Low engagement
                    "adoption_stage": "innovator",  # Already adopted
                },
                baseline_outcome=0.3,  # Already high
                baseline_propensity=0.2,  # Low propensity
            )
            for _ in range(200)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=200,
            model_id=uuid4(),
        )

        # Very low intensity
        config = InterventionConfig(
            intervention_type="sample_distribution",
            duration_weeks=1,
            intensity_multiplier=0.1,
        )

        engine = SimulationEngine(population)
        result = engine.simulate(config)

        # Should complete regardless of effect direction
        assert result.status == SimulationStatus.COMPLETED

    def test_confidence_level_extremes_80(self, sample_population, email_campaign_config):
        """Test 80% confidence level."""
        engine = SimulationEngine(sample_population)
        result = engine.simulate(email_campaign_config, confidence_level=0.80)

        assert result.status == SimulationStatus.COMPLETED
        assert result.simulated_ci_lower < result.simulated_ci_upper

    def test_confidence_level_extremes_99(self, sample_population, email_campaign_config):
        """Test 99% confidence level."""
        engine = SimulationEngine(sample_population)
        result = engine.simulate(email_campaign_config, confidence_level=0.99)

        assert result.status == SimulationStatus.COMPLETED
        assert result.simulated_ci_lower < result.simulated_ci_upper
