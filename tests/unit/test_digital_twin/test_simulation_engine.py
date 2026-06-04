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
from pydantic import ValidationError

from src.digital_twin.effect.estimator import TwinEffectEstimator
from src.digital_twin.effect.provider import SyntheticEffectDataProvider
from src.digital_twin.models.simulation_models import (
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

# These tests now drive the real uplift effect engine (a fitted UpliftRandomForest
# per simulate() call), so they belong in the isolated heavy/slow CI lane.
pytestmark = pytest.mark.slow


def _fast_engine(
    population: TwinPopulation,
    *,
    true_ate: float = 0.15,
    **engine_kwargs,
) -> SimulationEngine:
    """Build a SimulationEngine wired to a small, fast, deterministic effect engine.

    A real fit on the default provider/estimator is ~30s; the small synthetic
    frame (n=300) + shallow forest (20 trees, depth 3) keeps each simulate() at
    ~1-2s while preserving the seeded, deterministic contract (random_state=42).
    """
    return SimulationEngine(
        population,
        effect_provider=SyntheticEffectDataProvider(n=300, true_ate=true_ate, seed=42),
        effect_estimator=TwinEffectEstimator(
            n_estimators=20, max_depth=3, min_training_samples=100
        ),
        **engine_kwargs,
    )


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
    """Create simulation engine with sample population (fast deterministic effect engine)."""
    return _fast_engine(sample_population)


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
        # Real effect engine stamps provenance; a fabricated ATE is never emitted.
        assert result.data_provenance == "synthetic_uplift_v1"

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
        """Test that simulation calculates a positive recommended sample size.

        The recommended sample size now comes from the policy's two-proportion
        power calculation (no longer the old heuristic [100, 50000] clamp), so a
        large estimated effect can legitimately require fewer than 100 per arm.
        We assert the structural property: a positive integer per-arm sample size.
        """
        result = engine.simulate(email_campaign_config)

        assert result.recommended_sample_size is not None
        assert result.recommended_sample_size > 0

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
        for _specialty, stats in result.effect_heterogeneity.by_specialty.items():
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
        """DEPLOY when the CI lower bound clears the min-effect threshold.

        Inject a known true_ate (0.2) well above the default min_effect (0.05);
        the CI-based policy then deterministically recommends DEPLOY.
        """
        engine = _fast_engine(sample_population, true_ate=0.2)

        config = InterventionConfig(
            intervention_type="speaker_program_invitation",
            duration_weeks=12,
        )

        result = engine.simulate(config)

        assert result.status == SimulationStatus.COMPLETED
        assert result.recommendation == SimulationRecommendation.DEPLOY
        assert result.simulated_ci_lower > engine.min_effect_threshold

    def test_recommend_skip_small_effect(self, sample_population):
        """SKIP when the CI upper bound is below the min-effect threshold.

        Inject true_ate=0.0 and raise the threshold above the (small, positive)
        estimate's CI so the policy deterministically recommends SKIP.
        """
        engine = _fast_engine(sample_population, true_ate=0.0, min_effect_threshold=0.2)

        config = InterventionConfig(
            intervention_type="sample_distribution",
            duration_weeks=2,
        )

        result = engine.simulate(config)

        assert result.recommendation == SimulationRecommendation.SKIP
        assert result.simulated_ci_upper < engine.min_effect_threshold
        assert "below min effect" in result.recommendation_rationale.lower()

    def test_recommend_refine_uncertain(self, sample_population):
        """REFINE when the CI straddles the min-effect threshold.

        Inject true_ate=0.2 and set the threshold (0.25) inside the resulting CI
        so the band straddles it; the policy then deterministically recommends
        REFINE (gather more data).
        """
        engine = _fast_engine(sample_population, true_ate=0.2, min_effect_threshold=0.25)

        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        assert result.recommendation == SimulationRecommendation.REFINE
        assert result.simulated_ci_lower < engine.min_effect_threshold < result.simulated_ci_upper


# =============================================================================
# FIDELITY WARNING TESTS
# =============================================================================


class TestFidelityWarnings:
    """Tests for model fidelity warning logic."""

    def test_fidelity_warning_low_score(self, sample_population, email_campaign_config):
        """Test fidelity warning when score is low."""
        engine = _fast_engine(
            sample_population,
            model_fidelity_score=0.55,  # Below 0.7 threshold
        )

        result = engine.simulate(email_campaign_config)

        assert result.fidelity_warning is True
        assert "below threshold" in result.fidelity_warning_reason
        assert result.model_fidelity_score == 0.55

    def test_no_fidelity_warning_good_score(self, sample_population, email_campaign_config):
        """Test no fidelity warning when score is adequate."""
        engine = _fast_engine(
            sample_population,
            model_fidelity_score=0.85,  # Above threshold
        )

        result = engine.simulate(email_campaign_config)

        assert result.fidelity_warning is False
        assert result.fidelity_warning_reason is None

    def test_no_fidelity_warning_no_score(self, sample_population, email_campaign_config):
        """Test no fidelity warning when score is not set."""
        engine = _fast_engine(sample_population)  # No fidelity score

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

    def test_confidence_bounded_regardless_of_twin_count(self, email_campaign_config):
        """Confidence stays within [0, 1] for both small and large populations.

        The new std_error/CI are training-evidence-bound (driven by the labeled
        frame's sample size, not the twin count), so confidence no longer rises
        with more twins. We therefore only assert the structural [0, 1] bound for
        both sizes rather than a (no-longer-true) size-monotonicity inequality.
        """
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

        small_engine = _fast_engine(small_pop, model_fidelity_score=0.8)
        large_engine = _fast_engine(large_pop, model_fidelity_score=0.8)

        small_result = small_engine.simulate(email_campaign_config)
        large_result = large_engine.simulate(email_campaign_config)

        assert 0.0 <= small_result.simulation_confidence <= 1.0
        assert 0.0 <= large_result.simulation_confidence <= 1.0


# =============================================================================
# TREATMENT EFFECT CALCULATION TESTS
# =============================================================================


class TestTreatmentEffects:
    """Tests for treatment effect calculations."""

    def test_effect_varies_by_decile(self, sample_population, email_campaign_config):
        """Decile-level heterogeneity is populated for sufficiently-sized buckets.

        The synthetic DGP effect is ~constant plus mild covariate noise, so the
        former decile-monotonic assertion no longer holds. We assert the
        structural property the engine now guarantees: per-decile uplift stats
        are computed (ate/std/n) for buckets meeting the min sample size.
        """
        engine = _fast_engine(sample_population)
        result = engine.simulate(email_campaign_config, calculate_heterogeneity=True)

        by_decile = result.effect_heterogeneity.by_decile
        assert len(by_decile) > 0
        for _decile, stats in by_decile.items():
            assert "ate" in stats
            assert "n" in stats
            assert stats["n"] >= 10

    def test_simulate_is_deterministic(self, engine, email_campaign_config):
        """The seeded estimator (random_state=42) makes simulate() deterministic.

        Replaces the former variance test: the effect engine is no longer
        stochastic per call, so two identical simulate() calls are bit-identical.
        """
        result1 = engine.simulate(email_campaign_config)
        result2 = engine.simulate(email_campaign_config)

        assert result1.simulated_ate == result2.simulated_ate
        assert result1.simulated_ci_lower == result2.simulated_ci_lower
        assert result1.simulated_ci_upper == result2.simulated_ci_upper

    def test_duration_flows_through_to_recommendation(self, sample_population):
        """Duration no longer drives the effect; it flows through as the recommended duration.

        The old heuristic scaled the effect by duration_weeks; the real effect
        engine does not. Duration is now a pass-through: recommended_duration_weeks
        equals the configured duration.
        """
        engine = _fast_engine(sample_population)

        for weeks in (2, 12):
            config = InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=weeks,
            )
            result = engine.simulate(config)
            assert result.status == SimulationStatus.COMPLETED
            assert result.recommended_duration_weeks == weeks


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in simulation."""

    def test_insufficient_twins_error(self, small_population, email_campaign_config):
        """Test error when population is too small."""
        engine = _fast_engine(small_population)

        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.FAILED
        assert "Insufficient twins" in result.error_message
        assert result.twin_count == 0
        assert result.simulated_ate == 0.0

    def test_unknown_intervention_type(self, engine):
        """Unknown intervention types now fail closed (no fabricated default effect).

        The provider only supports its 6 known intervention types; anything else
        raises EffectDataUnavailable, which the engine surfaces as a FAILED result
        with a zeroed ATE rather than a completed result with heuristic defaults.
        """
        config = InterventionConfig(
            intervention_type="unknown_intervention",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        assert result.status == SimulationStatus.FAILED
        assert result.simulated_ate == 0.0

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
        """Test 95% confidence level produces a positive-width CI."""
        result = engine.simulate(email_campaign_config, confidence_level=0.95)

        ci_width_95 = result.simulated_ci_upper - result.simulated_ci_lower
        assert ci_width_95 > 0

    def test_confidence_level_is_accepted_but_does_not_drive_ci_width(
        self, engine, email_campaign_config
    ):
        """confidence_level is accepted for API compatibility but no longer drives CI width.

        The CI is now the estimator's training-evidence interval (bounded by the
        labeled-frame sample size), not a width recomputed from confidence_level.
        The former 90% < 95% < 99% width-monotonicity tests asserted removed
        behavior; here we pin the new contract: the CI is identical regardless of
        the requested level, and remains positive-width.
        """
        result_90 = engine.simulate(email_campaign_config, confidence_level=0.90)
        result_99 = engine.simulate(email_campaign_config, confidence_level=0.99)

        width_90 = result_90.simulated_ci_upper - result_90.simulated_ci_lower
        width_99 = result_99.simulated_ci_upper - result_99.simulated_ci_lower

        assert width_90 > 0
        assert width_90 == width_99


# =============================================================================
# EDGE CASE TESTS FOR EFFECT MODIFIERS (Phase 2)
# =============================================================================


class TestEffectModifierEdgeCases:
    """Tests for edge cases in effect modifier calculations."""

    @pytest.fixture
    def edge_case_population(self):
        """Create population with edge case feature values."""
        twins = []
        for _i in range(200):
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
        """Out-of-range low decile values still yield a completed simulation."""
        for twin in edge_case_population.twins:
            twin.features["decile"] = 0  # Invalid low value

        engine = _fast_engine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        # The effect engine consumes decile as a numeric covariate; an out-of-range
        # value does not break the fit.
        assert result.status == SimulationStatus.COMPLETED

    def test_extreme_high_decile(self, edge_case_population, email_campaign_config):
        """Out-of-range high decile values still yield a completed simulation."""
        for twin in edge_case_population.twins:
            twin.features["decile"] = 11  # Invalid high value

        engine = _fast_engine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.COMPLETED

    def test_zero_engagement_score(self, edge_case_population, email_campaign_config):
        """Zero engagement score still yields a completed, non-zero-ATE simulation."""
        for twin in edge_case_population.twins:
            twin.features["digital_engagement_score"] = 0.0

        engine = _fast_engine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.COMPLETED
        assert result.simulated_ate != 0

    def test_max_engagement_score(self, edge_case_population, email_campaign_config):
        """Maximum engagement score still yields a completed simulation."""
        for twin in edge_case_population.twins:
            twin.features["digital_engagement_score"] = 1.0

        engine = _fast_engine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.COMPLETED

    def test_negative_engagement_score(self, edge_case_population, email_campaign_config):
        """Negative engagement score still yields a completed simulation."""
        for twin in edge_case_population.twins:
            twin.features["digital_engagement_score"] = -0.5

        engine = _fast_engine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.COMPLETED

    def test_engagement_above_one(self, edge_case_population, email_campaign_config):
        """Engagement score above 1.0 still yields a completed simulation."""
        for twin in edge_case_population.twins:
            twin.features["digital_engagement_score"] = 1.5

        engine = _fast_engine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.COMPLETED

    def test_invalid_adoption_stage(self, edge_case_population, email_campaign_config):
        """Unknown adoption stage (a string feature) is dropped; simulation completes."""
        for twin in edge_case_population.twins:
            twin.features["adoption_stage"] = "unknown_stage"

        engine = _fast_engine(edge_case_population)
        result = engine.simulate(email_campaign_config)

        # Non-numeric features are not used as covariates; an unknown value is inert.
        assert result.status == SimulationStatus.COMPLETED

    def test_zero_intensity_multiplier(self):
        """Test that zero intensity multiplier raises ValidationError.

        intensity_multiplier has ge=0.1 constraint.
        """
        with pytest.raises(ValidationError) as exc_info:
            InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=8,
                intensity_multiplier=0.0,  # Below min of 0.1
            )

        assert "intensity_multiplier" in str(exc_info.value)

    def test_extreme_intensity_multiplier(self, edge_case_population):
        """Extreme intensity_multiplier is inert (effect no longer depends on it); completes."""
        engine = _fast_engine(edge_case_population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
            intensity_multiplier=10.0,
        )

        result = engine.simulate(config)

        # intensity_multiplier was a heuristic scalar that no longer affects the
        # real effect engine; the simulation completes regardless of its value.
        assert result.status == SimulationStatus.COMPLETED

    def test_zero_duration_weeks(self):
        """Test that zero duration weeks raises ValidationError.

        duration_weeks has ge=1 constraint.
        """
        with pytest.raises(ValidationError) as exc_info:
            InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=0,  # Below min of 1
                intensity_multiplier=1.0,
            )

        assert "duration_weeks" in str(exc_info.value)

    def test_combined_extreme_modifiers(self, edge_case_population):
        """Test combined extreme modifier values don't cause overflow."""
        for twin in edge_case_population.twins:
            twin.features["decile"] = 1  # Maximum decile multiplier
            twin.features["digital_engagement_score"] = 1.0  # Max engagement
            twin.features["adoption_stage"] = "laggard"  # Max adoption multiplier
            twin.baseline_propensity = 1.0  # Max propensity

        engine = _fast_engine(edge_case_population)
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

        engine = _fast_engine(population)
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

        engine = _fast_engine(population)
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

        engine = _fast_engine(population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.FAILED

    def test_ci_bounds_with_uniform_population(self, email_campaign_config):
        """Test CI calculation completes with a degenerate, perfectly-uniform population."""
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

        engine = _fast_engine(population)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.COMPLETED
        # With perfectly-uniform covariates the uplift fit can collapse to a
        # zero-width training CI, so the only guaranteed structural property is a
        # non-negative CI width (lower <= upper); the simulation must not error.
        assert result.simulated_ci_lower <= result.simulated_ci_upper

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

        engine = _fast_engine(population)
        result = engine.simulate(config)

        # Should complete regardless of effect direction
        assert result.status == SimulationStatus.COMPLETED

    def test_confidence_level_extremes_80(self, sample_population, email_campaign_config):
        """Test simulation accepts an 80% confidence level and completes."""
        engine = _fast_engine(sample_population)
        result = engine.simulate(email_campaign_config, confidence_level=0.80)

        assert result.status == SimulationStatus.COMPLETED
        assert result.simulated_ci_lower < result.simulated_ci_upper

    def test_confidence_level_extremes_99(self, sample_population, email_campaign_config):
        """Test simulation accepts a 99% confidence level and completes."""
        engine = _fast_engine(sample_population)
        result = engine.simulate(email_campaign_config, confidence_level=0.99)

        assert result.status == SimulationStatus.COMPLETED
        assert result.simulated_ci_lower < result.simulated_ci_upper
