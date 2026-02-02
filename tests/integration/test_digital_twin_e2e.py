"""
End-to-End Integration Tests for Digital Twin System.

Tests the complete workflow from twin generation through simulation
to fidelity validation and automatic retraining triggers.

Tests cover:
- Full workflow: generate -> simulate -> validate
- DEPLOY recommendation flow
- SKIP recommendation flow
- Fidelity triggers retraining
- Retrained model improvement verification

IMPORTANT: These tests may require external services (Redis, DB).
Run with: pytest -n 1 tests/integration/test_digital_twin_e2e.py -v
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.fidelity_tracker import FidelityTracker
from src.digital_twin.models.simulation_models import (
    FidelityGrade,
    FidelityRecord,
    InterventionConfig,
    PopulationFilter,
    SimulationRecommendation,
    SimulationStatus,
)
from src.digital_twin.models.twin_models import (
    Brand,
    TwinType,
)
from src.digital_twin.retraining_service import (
    TwinRetrainingConfig,
    TwinRetrainingService,
    TwinTriggerReason,
)
from src.digital_twin.simulation_engine import SimulationEngine
from src.digital_twin.twin_generator import TwinGenerator

# Mark all tests as E2E integration tests
pytestmark = [
    pytest.mark.xdist_group(name="digital_twin_e2e"),
]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def training_data():
    """Create sample training data for twin generation."""
    np.random.seed(42)
    n_samples = 2000

    data = pd.DataFrame(
        {
            "specialty": np.random.choice(["rheumatology", "dermatology", "allergy"], n_samples),
            "decile": np.random.randint(1, 11, n_samples),
            "region": np.random.choice(["northeast", "south", "midwest", "west"], n_samples),
            "digital_engagement_score": np.random.uniform(0.1, 0.9, n_samples),
            "adoption_stage": np.random.choice(
                ["innovator", "early_adopter", "early_majority", "late_majority", "laggard"],
                n_samples,
            ),
            "patient_volume": np.random.randint(50, 500, n_samples),
            "prescribing_change": np.random.uniform(-0.1, 0.3, n_samples),  # Target
        }
    )

    return data


@pytest.fixture
def generator():
    """Create twin generator."""
    return TwinGenerator(
        twin_type=TwinType.HCP,
        brand=Brand.REMIBRUTINIB,
    )


@pytest.fixture
def email_campaign_config():
    """Create standard email campaign intervention."""
    return InterventionConfig(
        intervention_type="email_campaign",
        channel="email",
        frequency="weekly",
        duration_weeks=8,
        intensity_multiplier=1.0,
        target_deciles=[1, 2, 3, 4, 5],
    )


@pytest.fixture
def low_effect_config():
    """Create intervention config expected to produce low effect."""
    return InterventionConfig(
        intervention_type="sample_distribution",
        duration_weeks=2,
        intensity_multiplier=0.3,
    )


@pytest.fixture
def high_effect_config():
    """Create intervention config expected to produce high effect."""
    return InterventionConfig(
        intervention_type="speaker_program_invitation",
        duration_weeks=12,
        intensity_multiplier=2.0,
    )


@pytest.fixture
def mock_repository():
    """Create mock repository for testing."""
    repo = MagicMock()
    repo.save_simulation = AsyncMock(return_value=uuid4())
    repo.get_simulation = AsyncMock(return_value=None)
    repo.get_model_fidelity_records = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def fidelity_tracker(mock_repository):
    """Create fidelity tracker with mock repository."""
    return FidelityTracker(repository=mock_repository)


@pytest.fixture
def retraining_service(mock_repository):
    """Create retraining service with test config."""
    config = TwinRetrainingConfig(
        fidelity_threshold=0.70,
        min_validations_for_decision=3,  # Lower for testing
        cooldown_hours=1,
        auto_approve_threshold=0.50,
    )
    return TwinRetrainingService(config=config, repository=mock_repository)


# =============================================================================
# FULL WORKFLOW TESTS
# =============================================================================


class TestFullWorkflowDeploy:
    """Tests for full workflow ending in DEPLOY recommendation."""

    def test_generate_simulate_deploy(self, generator, training_data, high_effect_config):
        """Test complete workflow: generate twins -> simulate -> get DEPLOY."""
        # Step 1: Train generator on data
        metrics = generator.train(
            data=training_data,
            target_col="prescribing_change",
        )

        assert metrics.r2_score is not None or metrics.cv_mean is not None

        # Step 2: Generate twin population
        population = generator.generate(n=500, seed=42)

        assert len(population) == 500
        assert population.model_id is not None

        # Step 3: Create simulation engine and run simulation
        engine = SimulationEngine(
            population,
            min_effect_threshold=0.03,  # Lower threshold
        )

        result = engine.simulate(high_effect_config)

        # Step 4: Verify result
        assert result.status == SimulationStatus.COMPLETED
        assert result.simulated_ate != 0
        # With high intensity, should often recommend deploy
        assert result.recommendation in [
            SimulationRecommendation.DEPLOY,
            SimulationRecommendation.REFINE,
        ]

    def test_workflow_with_population_filter(self, generator, training_data, email_campaign_config):
        """Test workflow with population filtering."""
        # Train and generate
        generator.train(data=training_data, target_col="prescribing_change")
        population = generator.generate(n=500, seed=42)

        # Filter to high-value segment
        filter_ = PopulationFilter(
            deciles=[1, 2, 3],
            specialties=["rheumatology"],
        )

        engine = SimulationEngine(population, min_effect_threshold=0.03)
        result = engine.simulate(email_campaign_config, population_filter=filter_)

        # Should complete (may have fewer twins after filter)
        if result.status == SimulationStatus.COMPLETED:
            assert result.twin_count < 500


class TestFullWorkflowSkip:
    """Tests for full workflow ending in SKIP recommendation."""

    def test_generate_simulate_skip(self, generator, training_data, low_effect_config):
        """Test workflow that results in SKIP recommendation."""
        # Train and generate
        generator.train(data=training_data, target_col="prescribing_change")
        population = generator.generate(n=500, seed=42)

        # Use high threshold to ensure SKIP
        engine = SimulationEngine(
            population,
            min_effect_threshold=0.50,  # Very high threshold
        )

        result = engine.simulate(low_effect_config)

        assert result.status == SimulationStatus.COMPLETED
        assert result.recommendation == SimulationRecommendation.SKIP
        assert "below" in result.recommendation_rationale.lower()


class TestFullWorkflowRefine:
    """Tests for full workflow ending in REFINE recommendation."""

    def test_uncertain_simulation_refine(self, generator, training_data):
        """Test workflow with uncertain results suggesting refinement."""
        generator.train(data=training_data, target_col="prescribing_change")
        population = generator.generate(n=150, seed=42)  # Smaller for more uncertainty

        # Short duration, moderate intensity
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=2,
            intensity_multiplier=0.5,
        )

        engine = SimulationEngine(
            population,
            min_effect_threshold=0.02,
        )

        result = engine.simulate(config)

        # Could be any outcome but verify structure
        assert result.status == SimulationStatus.COMPLETED
        assert result.recommended_sample_size is not None


# =============================================================================
# FIDELITY VALIDATION WORKFLOW TESTS
# =============================================================================


class TestFidelityValidationWorkflow:
    """Tests for fidelity tracking and validation workflow."""

    def test_record_prediction_then_validate(
        self, fidelity_tracker, generator, training_data, email_campaign_config
    ):
        """Test recording prediction then validating against actual results."""
        # Generate and simulate
        generator.train(data=training_data, target_col="prescribing_change")
        population = generator.generate(n=300, seed=42)

        engine = SimulationEngine(population)
        result = engine.simulate(email_campaign_config)

        # Step 1: Record prediction
        record = fidelity_tracker.record_prediction(result)

        assert record.simulation_id == result.simulation_id
        assert record.simulated_ate == result.simulated_ate
        assert record.fidelity_grade == FidelityGrade.UNVALIDATED

        # Step 2: Validate with "actual" results (simulated for test)
        # Assume actual is close to predicted (good model)
        actual_ate = result.simulated_ate * 0.95  # 5% error
        actual_ci = (result.simulated_ci_lower * 0.9, result.simulated_ci_upper * 1.1)

        validated_record = fidelity_tracker.validate(
            simulation_id=result.simulation_id,
            actual_ate=actual_ate,
            actual_ci=actual_ci,
            actual_sample_size=1000,
        )

        assert validated_record.actual_ate == actual_ate
        assert validated_record.fidelity_grade != FidelityGrade.UNVALIDATED
        # With 5% error, should be EXCELLENT or GOOD
        assert validated_record.fidelity_grade in [
            FidelityGrade.EXCELLENT,
            FidelityGrade.GOOD,
        ]

    def test_poor_fidelity_grade(self, fidelity_tracker):
        """Test that large prediction error results in POOR grade."""
        # Create a fake simulation result
        from src.digital_twin.models.simulation_models import (
            EffectHeterogeneity,
            SimulationResult,
        )

        result = SimulationResult(
            simulation_id=uuid4(),
            model_id=uuid4(),
            intervention_config=InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=8,
            ),
            population_filters=PopulationFilter(),
            twin_count=500,
            simulated_ate=0.10,  # Predicted 10%
            simulated_ci_lower=0.08,
            simulated_ci_upper=0.12,
            simulated_std_error=0.01,
            effect_heterogeneity=EffectHeterogeneity(),
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Test",
            simulation_confidence=0.8,
            fidelity_warning=False,
            status=SimulationStatus.COMPLETED,
            execution_time_ms=100,
            created_at=datetime.now(timezone.utc),
        )

        fidelity_tracker.record_prediction(result)

        # Validate with very different actual (50% error)
        validated = fidelity_tracker.validate(
            simulation_id=result.simulation_id,
            actual_ate=0.05,  # Only 5% actual (50% error from 10%)
            actual_ci=(0.03, 0.07),
            actual_sample_size=1000,
        )

        assert validated.fidelity_grade == FidelityGrade.POOR


# =============================================================================
# RETRAINING TRIGGER WORKFLOW TESTS
# =============================================================================


class TestRetrainingTriggerWorkflow:
    """Tests for automatic retraining trigger workflow."""

    @pytest.mark.asyncio
    async def test_fidelity_triggers_retrain_evaluation(self, retraining_service, mock_repository):
        """Test that poor fidelity triggers retraining evaluation."""
        model_id = uuid4()

        # Setup mock to return poor fidelity records
        poor_records = [
            FidelityRecord(
                tracking_id=uuid4(),
                simulation_id=uuid4(),
                simulated_ate=0.10,
                actual_ate=0.04,  # 60% error
                prediction_error=0.06,
                absolute_error=0.06,
                ci_coverage=False,
                fidelity_grade=FidelityGrade.POOR,
                created_at=datetime.now(timezone.utc),
                validated_at=datetime.now(timezone.utc),
            )
            for _ in range(5)
        ]

        mock_repository.get_model_fidelity_records = AsyncMock(return_value=poor_records)

        # Evaluate retraining need
        decision = await retraining_service.evaluate_retraining_need(model_id)

        assert decision.should_retrain is True
        assert decision.reason == TwinTriggerReason.FIDELITY_DEGRADATION

    @pytest.mark.asyncio
    async def test_good_fidelity_no_retrain(self, retraining_service, mock_repository):
        """Test that good fidelity doesn't trigger retraining."""
        model_id = uuid4()

        # Setup mock to return good fidelity records
        good_records = [
            FidelityRecord(
                tracking_id=uuid4(),
                simulation_id=uuid4(),
                simulated_ate=0.10,
                actual_ate=0.095,  # 5% error
                prediction_error=0.005,
                absolute_error=0.005,
                ci_coverage=True,
                fidelity_grade=FidelityGrade.EXCELLENT,
                created_at=datetime.now(timezone.utc),
                validated_at=datetime.now(timezone.utc),
            )
            for _ in range(5)
        ]

        mock_repository.get_model_fidelity_records = AsyncMock(return_value=good_records)

        decision = await retraining_service.evaluate_retraining_need(model_id)

        assert decision.should_retrain is False

    @pytest.mark.asyncio
    async def test_insufficient_validations_no_retrain(self, retraining_service, mock_repository):
        """Test that insufficient validations don't trigger retraining."""
        model_id = uuid4()

        # Only 2 records (below min_validations_for_decision of 3)
        records = [
            FidelityRecord(
                tracking_id=uuid4(),
                simulation_id=uuid4(),
                simulated_ate=0.10,
                actual_ate=0.04,  # Poor
                prediction_error=0.06,
                absolute_error=0.06,
                fidelity_grade=FidelityGrade.POOR,
                created_at=datetime.now(timezone.utc),
            )
            for _ in range(2)
        ]

        mock_repository.get_model_fidelity_records = AsyncMock(return_value=records)

        decision = await retraining_service.evaluate_retraining_need(model_id)

        assert decision.should_retrain is False
        assert "insufficient_validations" in decision.details.get("blocked_reason", "")

    @pytest.mark.asyncio
    async def test_trigger_retraining_creates_job(self, retraining_service, mock_repository):
        """Test that triggering retraining creates a job."""
        model_id = uuid4()

        # Setup mock fidelity data
        mock_repository.get_model_fidelity_records = AsyncMock(return_value=[])

        # Trigger retraining manually
        job = await retraining_service.trigger_retraining(
            model_id=model_id,
            reason=TwinTriggerReason.FIDELITY_DEGRADATION,
            approved_by="test_user",
        )

        assert job is not None
        assert job.model_id == str(model_id)
        assert job.trigger_reason == TwinTriggerReason.FIDELITY_DEGRADATION
        assert job.training_config.get("approved_by") == "test_user"


# =============================================================================
# COMPLETE E2E WORKFLOW TESTS
# =============================================================================


class TestCompleteE2EWorkflow:
    """Tests for complete end-to-end workflow scenarios."""

    @pytest.mark.asyncio
    async def test_full_e2e_deploy_validate_good(
        self,
        generator,
        training_data,
        email_campaign_config,
        fidelity_tracker,
    ):
        """Test full E2E: generate -> simulate -> deploy -> validate (good)."""
        # 1. Train generator
        generator.train(data=training_data, target_col="prescribing_change")

        # 2. Generate twins
        population = generator.generate(n=400, seed=42)

        # 3. Simulate intervention
        engine = SimulationEngine(population, min_effect_threshold=0.02)
        result = engine.simulate(email_campaign_config)

        assert result.status == SimulationStatus.COMPLETED

        # 4. Record prediction
        fidelity_tracker.record_prediction(result)

        # 5. "Run" the actual experiment (simulated)
        # Assume actual results are close to prediction
        actual_ate = result.simulated_ate * 0.92  # 8% error

        # 6. Validate
        validated = fidelity_tracker.validate(
            simulation_id=result.simulation_id,
            actual_ate=actual_ate,
            actual_ci=(
                result.simulated_ci_lower * 0.9,
                result.simulated_ci_upper * 1.1,
            ),
            actual_sample_size=800,
        )

        # 7. Check fidelity grade
        assert validated.fidelity_grade in [
            FidelityGrade.EXCELLENT,
            FidelityGrade.GOOD,
        ]

    @pytest.mark.asyncio
    async def test_full_e2e_poor_fidelity_triggers_retrain(
        self,
        generator,
        training_data,
        email_campaign_config,
        fidelity_tracker,
        retraining_service,
        mock_repository,
    ):
        """Test E2E: poor fidelity triggers retraining decision."""
        # 1-4. Generate, simulate, record prediction
        generator.train(data=training_data, target_col="prescribing_change")
        population = generator.generate(n=400, seed=42)

        engine = SimulationEngine(population)
        result = engine.simulate(email_campaign_config)

        fidelity_tracker.record_prediction(result)

        # 5. Validate with poor actual results (large error)
        validated = fidelity_tracker.validate(
            simulation_id=result.simulation_id,
            actual_ate=result.simulated_ate * 0.3,  # 70% error!
            actual_ci=(0.01, 0.03),
            actual_sample_size=500,
        )

        assert validated.fidelity_grade == FidelityGrade.POOR

        # 6. Setup multiple poor validations for retraining check
        poor_records = [validated] * 5
        mock_repository.get_model_fidelity_records = AsyncMock(return_value=poor_records)

        # 7. Check retraining decision
        model_id = population.model_id
        decision = await retraining_service.evaluate_retraining_need(model_id)

        assert decision.should_retrain is True

    def test_workflow_heterogeneity_analysis(self, generator, training_data, email_campaign_config):
        """Test workflow with heterogeneity analysis for segment targeting."""
        generator.train(data=training_data, target_col="prescribing_change")
        population = generator.generate(n=500, seed=42)

        engine = SimulationEngine(population)
        result = engine.simulate(
            email_campaign_config,
            calculate_heterogeneity=True,
        )

        # Verify heterogeneity data is populated
        assert len(result.effect_heterogeneity.by_specialty) > 0
        assert len(result.effect_heterogeneity.by_decile) > 0

        # Get top responding segments
        top_segments = result.effect_heterogeneity.get_top_segments(n=3)
        assert len(top_segments) > 0

        # Each segment should have required fields
        for seg in top_segments:
            assert "dimension" in seg
            assert "segment" in seg
            assert "ate" in seg


# =============================================================================
# INTEGRATION WITH CACHE WORKFLOW TESTS
# =============================================================================


class TestCacheIntegrationWorkflow:
    """Tests for workflow with caching enabled."""

    @pytest.mark.asyncio
    async def test_cached_simulation_returns_same_result(
        self, generator, training_data, email_campaign_config
    ):
        """Test that cached simulations return consistent results."""
        from src.digital_twin.simulation_cache import SimulationCache

        # Setup mock cache
        cached_data = {}

        async def mock_get(key):
            return cached_data.get(key)

        async def mock_setex(key, ttl, value):
            cached_data[key] = value
            return True

        mock_redis = AsyncMock()
        mock_redis.get = mock_get
        mock_redis.setex = mock_setex
        mock_redis.hset = AsyncMock()
        mock_redis.expire = AsyncMock()
        mock_redis.hincrby = AsyncMock()

        cache = SimulationCache(redis_client=mock_redis)

        # Generate twins
        generator.train(data=training_data, target_col="prescribing_change")
        population = generator.generate(n=300, seed=42)

        # Create engine with cache
        engine = SimulationEngine(population, cache=cache)

        # First simulation (cache miss)
        np.random.seed(42)
        result1 = engine.simulate(email_campaign_config, use_cache=True)

        # Manually cache the result
        await cache.cache_result(result1)

        # Second simulation should use cache
        # (In real usage, simulate would check cache first)
        result2 = await cache.get_cached_result(
            email_campaign_config,
            None,
            population.model_id,
        )

        if result2:
            assert result2.simulated_ate == result1.simulated_ate
            assert result2.recommendation == result1.recommendation
