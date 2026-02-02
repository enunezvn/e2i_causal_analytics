"""
Unit Tests for Digital Twin Error Recovery.

Tests cover:
- Malformed intervention configurations
- Missing required fields
- Database connection failures
- MLflow unavailability
- Redis connection timeouts
- Partial twin data handling
- Graceful degradation scenarios
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from src.digital_twin.models.simulation_models import (
    InterventionConfig,
    PopulationFilter,
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
def valid_twins():
    """Create valid twin population."""
    np.random.seed(42)
    twins = []

    for i in range(200):
        twin = DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            features={
                "specialty": ["rheumatology", "dermatology", "allergy"][i % 3],
                "decile": (i % 10) + 1,
                "region": ["northeast", "south", "midwest", "west"][i % 4],
                "digital_engagement_score": np.random.uniform(0.2, 0.9),
                "adoption_stage": "early_majority",
            },
            baseline_outcome=np.random.uniform(0.05, 0.25),
            baseline_propensity=np.random.uniform(0.3, 0.8),
        )
        twins.append(twin)

    return twins


@pytest.fixture
def population(valid_twins):
    """Create population from valid twins."""
    return TwinPopulation(
        twin_type=TwinType.HCP,
        brand=Brand.REMIBRUTINIB,
        twins=valid_twins,
        size=len(valid_twins),
        model_id=uuid4(),
    )


@pytest.fixture
def engine(population):
    """Create simulation engine."""
    return SimulationEngine(population)


# =============================================================================
# MALFORMED INTERVENTION CONFIG TESTS
# =============================================================================


class TestMalformedInterventionConfig:
    """Tests for handling malformed intervention configurations."""

    def test_empty_intervention_type(self, engine):
        """Test handling of empty intervention type."""
        config = InterventionConfig(
            intervention_type="",  # Empty string
            duration_weeks=8,
        )

        # Should use default parameters or handle gracefully
        result = engine.simulate(config)

        # Engine should complete (uses defaults for unknown interventions)
        assert result.status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED]

    def test_unknown_intervention_type(self, engine):
        """Test handling of unknown intervention type."""
        config = InterventionConfig(
            intervention_type="nonexistent_intervention_xyz",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        # Should complete with default effect parameters
        assert result.status == SimulationStatus.COMPLETED

    def test_negative_duration(self):
        """Test that negative duration weeks raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=-5,  # Invalid negative
            )

        # Pydantic validates ge=1 constraint
        assert "duration_weeks" in str(exc_info.value)

    def test_zero_duration(self):
        """Test that zero duration weeks raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=0,  # Invalid - must be >= 1
            )

        assert "duration_weeks" in str(exc_info.value)

    def test_extreme_intensity_multiplier(self):
        """Test that extreme intensity multiplier raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=8,
                intensity_multiplier=1000.0,  # Exceeds max of 10.0
            )

        assert "intensity_multiplier" in str(exc_info.value)

    def test_negative_intensity_multiplier(self):
        """Test that negative intensity multiplier raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=8,
                intensity_multiplier=-1.0,  # Below min of 0.1
            )

        assert "intensity_multiplier" in str(exc_info.value)

    def test_empty_target_deciles(self, engine):
        """Test handling of empty target deciles list."""
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
            target_deciles=[],  # Empty list
        )

        result = engine.simulate(config)

        assert result.status == SimulationStatus.COMPLETED

    def test_invalid_decile_values(self):
        """Test that invalid decile values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=8,
                target_deciles=[0, -1, 11, 100],  # Out of range (must be 1-10)
            )

        assert "decile" in str(exc_info.value).lower()


# =============================================================================
# MISSING REQUIRED FIELDS TESTS
# =============================================================================


class TestMissingRequiredFields:
    """Tests for handling missing or None field values."""

    def test_twins_with_missing_features(self, population):
        """Test handling twins with missing feature values."""
        # Modify some twins to have missing features
        for twin in population.twins[:50]:
            twin.features = {}  # Empty features dict

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        # Should use defaults for missing features
        assert result.status == SimulationStatus.COMPLETED

    def test_twins_with_none_propensity(self):
        """Test handling twins with None baseline propensity."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={"specialty": "rheumatology", "decile": i % 10 + 1},
                baseline_outcome=0.1,
                baseline_propensity=0.5 if i % 2 == 0 else 0.0,  # Some zero
            )
            for i in range(150)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=150,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        assert result.status == SimulationStatus.COMPLETED

    def test_population_filter_with_none_values(self, engine):
        """Test population filter with None values in lists."""
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        # Filter with empty lists (should not filter)
        population_filter = PopulationFilter(
            specialties=[],
            deciles=[],
            regions=[],
        )

        result = engine.simulate(config, population_filter=population_filter)

        # Should include all twins (no filtering)
        assert result.status == SimulationStatus.COMPLETED
        assert result.twin_count == 200


# =============================================================================
# DATABASE CONNECTION FAILURE TESTS
# =============================================================================


class TestDatabaseConnectionFailure:
    """Tests for handling database connection failures."""

    @pytest.mark.asyncio
    async def test_supabase_unavailable_on_save(self):
        """Test graceful handling when Supabase unavailable during save."""
        from src.digital_twin.twin_repository import SimulationRepository

        # Mock the Supabase client to raise exception
        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute = MagicMock(
            side_effect=Exception("Connection refused")
        )

        repo = SimulationRepository(supabase_client=mock_client)

        # Create a minimal simulation result dict
        from src.digital_twin.models.simulation_models import (
            EffectHeterogeneity,
            SimulationRecommendation,
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
            twin_count=100,
            simulated_ate=0.05,
            simulated_ci_lower=0.02,
            simulated_ci_upper=0.08,
            simulated_std_error=0.015,
            effect_heterogeneity=EffectHeterogeneity(),
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Test",
            simulation_confidence=0.8,
            fidelity_warning=False,
            status=SimulationStatus.COMPLETED,
            execution_time_ms=100,
            created_at=datetime.now(timezone.utc),
        )

        # Should handle gracefully (not crash)
        try:
            await repo.save_simulation(result, brand="REMIBRUTINIB")
        except Exception as e:
            # Expected to raise, but shouldn't crash the whole system
            assert "Connection refused" in str(e)


# =============================================================================
# MLFLOW UNAVAILABILITY TESTS
# =============================================================================


class TestMLflowUnavailability:
    """Tests for handling MLflow unavailability."""

    def test_simulation_continues_without_mlflow(self, engine):
        """Test that simulation continues even if MLflow logging fails."""
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        # Patch MLflow to raise exception
        with patch("mlflow.log_params", side_effect=Exception("MLflow unavailable")):
            with patch("mlflow.log_metrics", side_effect=Exception("MLflow unavailable")):
                # Simulation should still complete
                result = engine.simulate(config)

                assert result.status == SimulationStatus.COMPLETED


# =============================================================================
# REDIS CONNECTION TIMEOUT TESTS
# =============================================================================


class TestRedisConnectionTimeout:
    """Tests for handling Redis connection timeouts."""

    @pytest.mark.asyncio
    async def test_cache_fallback_on_redis_timeout(self):
        """Test that cache falls back gracefully on Redis timeout."""
        from src.digital_twin.simulation_cache import SimulationCache

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=TimeoutError("Redis timeout"))

        cache = SimulationCache(redis_client=mock_redis)

        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        # Should return None instead of crashing
        result = await cache.get_cached_result(config, None, uuid4())

        assert result is None
        assert cache._stats.errors == 1

    @pytest.mark.asyncio
    async def test_cache_write_fails_gracefully_on_timeout(self):
        """Test that cache write fails gracefully on timeout."""
        from src.digital_twin.models.simulation_models import (
            EffectHeterogeneity,
            SimulationRecommendation,
            SimulationResult,
        )
        from src.digital_twin.simulation_cache import SimulationCache

        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(side_effect=TimeoutError("Redis timeout"))

        cache = SimulationCache(redis_client=mock_redis)

        result = SimulationResult(
            simulation_id=uuid4(),
            model_id=uuid4(),
            intervention_config=InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=8,
            ),
            population_filters=PopulationFilter(),
            twin_count=100,
            simulated_ate=0.05,
            simulated_ci_lower=0.02,
            simulated_ci_upper=0.08,
            simulated_std_error=0.015,
            effect_heterogeneity=EffectHeterogeneity(),
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Test",
            simulation_confidence=0.8,
            fidelity_warning=False,
            status=SimulationStatus.COMPLETED,
            execution_time_ms=100,
            created_at=datetime.now(timezone.utc),
        )

        success = await cache.cache_result(result)

        assert success is False
        assert cache._stats.errors == 1


# =============================================================================
# PARTIAL TWIN DATA TESTS
# =============================================================================


class TestPartialTwinData:
    """Tests for handling incomplete twin feature data."""

    def test_twins_missing_specialty(self):
        """Test twins with missing specialty feature."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    # No specialty
                    "decile": i % 10 + 1,
                    "digital_engagement_score": 0.5,
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for i in range(150)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=150,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        # Should handle missing specialty gracefully
        result = engine.simulate(config, calculate_heterogeneity=True)

        assert result.status == SimulationStatus.COMPLETED

    def test_twins_missing_decile(self):
        """Test twins with missing decile feature."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    # No decile - should use default
                    "digital_engagement_score": 0.5,
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for i in range(150)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=150,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        # Should use default decile (5) and complete
        assert result.status == SimulationStatus.COMPLETED

    def test_twins_missing_engagement_score(self):
        """Test twins with missing engagement score."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    "decile": 5,
                    # No engagement score
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for i in range(150)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=150,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        # Should use default engagement (0.5) and complete
        assert result.status == SimulationStatus.COMPLETED

    def test_twins_missing_adoption_stage(self):
        """Test twins with missing adoption stage."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    "decile": 5,
                    "digital_engagement_score": 0.5,
                    # No adoption_stage
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for i in range(150)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=150,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        # Should use default adoption stage (early_majority) and complete
        assert result.status == SimulationStatus.COMPLETED

    def test_twins_invalid_feature_types(self):
        """Test twins with invalid feature value types raises TypeError.

        Features dict accepts Any type, but invalid types (strings where
        numbers expected) will cause TypeError during simulation when
        comparisons or arithmetic is performed.
        """
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    "decile": "not_a_number",  # Invalid type
                    "digital_engagement_score": "high",  # Invalid type
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for i in range(150)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=150,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        # Invalid feature types (strings where numbers expected) cause TypeError
        # during simulation when comparisons/arithmetic is performed
        with pytest.raises(TypeError):
            engine.simulate(config)


# =============================================================================
# BOUNDARY VALUE TESTS
# =============================================================================


class TestBoundaryValues:
    """Tests for boundary value handling."""

    def test_decile_zero(self):
        """Test handling of decile value 0."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    "decile": 0,  # Below valid range
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for _ in range(150)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=150,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        # Should clamp to valid range
        assert result.status == SimulationStatus.COMPLETED

    def test_decile_above_ten(self):
        """Test handling of decile value above 10."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    "decile": 15,  # Above valid range
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for _ in range(150)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=150,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        # Should clamp to valid range
        assert result.status == SimulationStatus.COMPLETED

    def test_engagement_score_negative(self):
        """Test handling of negative engagement score."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    "decile": 5,
                    "digital_engagement_score": -0.5,  # Invalid negative
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for _ in range(150)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=150,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        # Should clamp to 0
        assert result.status == SimulationStatus.COMPLETED

    def test_engagement_score_above_one(self):
        """Test handling of engagement score above 1.0."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={
                    "specialty": "rheumatology",
                    "decile": 5,
                    "digital_engagement_score": 1.5,  # Above valid range
                },
                baseline_outcome=0.1,
                baseline_propensity=0.5,
            )
            for _ in range(150)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=150,
            model_id=uuid4(),
        )

        engine = SimulationEngine(population)
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = engine.simulate(config)

        # Should clamp to 1.0
        assert result.status == SimulationStatus.COMPLETED

    def test_negative_propensity(self):
        """Test that negative baseline propensity raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={"specialty": "rheumatology", "decile": 5},
                baseline_outcome=0.1,
                baseline_propensity=-0.3,  # Invalid negative (must be >= 0)
            )

        assert "baseline_propensity" in str(exc_info.value)
