"""
Unit tests for digital_twin/twin_repository.py

Tests cover:
- TwinModelRepository save, get, list, deactivate
- SimulationRepository save, get, list, update, link
- FidelityRepository save, update, get
- TwinRepository facade pattern
- Redis caching
- MLflow integration
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import UUID, uuid4

from src.digital_twin.twin_repository import (
    TwinModelRepository,
    SimulationRepository,
    FidelityRepository,
    TwinRepository,
)
from src.digital_twin.models.twin_models import (
    Brand,
    TwinModelConfig,
    TwinModelMetrics,
    TwinType,
)
from src.digital_twin.models.simulation_models import (
    FidelityGrade,
    FidelityRecord,
    SimulationResult,
    SimulationStatus,
    InterventionConfig,
    PopulationFilter,
    EffectHeterogeneity,
    SimulationRecommendation,
)


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    client = MagicMock()
    client.table = MagicMock(return_value=client)
    client.select = MagicMock(return_value=client)
    client.insert = MagicMock(return_value=client)
    client.update = MagicMock(return_value=client)
    client.eq = MagicMock(return_value=client)
    client.order = MagicMock(return_value=client)
    client.limit = MagicMock(return_value=client)
    not_mock = MagicMock()
    not_mock.is_ = MagicMock(return_value=client)
    client.not_ = not_mock
    client.is_ = MagicMock(return_value=client)
    client.execute = AsyncMock()
    return client


@pytest.fixture
def mock_mlflow():
    """Mock MLflow client."""
    return MagicMock()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = MagicMock()
    redis.setex = MagicMock()
    redis.get = MagicMock(return_value=None)
    redis.delete = MagicMock()
    return redis


@pytest.fixture
def twin_model_config():
    """Sample TwinModelConfig."""
    return TwinModelConfig(
        model_name="HCP Twin Model",
        model_description="Test twin model",
        twin_type=TwinType.HCP,
        brand=Brand.KISQALI,
        algorithm="random_forest",
        n_estimators=100,
        max_depth=10,
        training_samples=5000,
        validation_split=0.2,
        cv_folds=5,
        feature_columns=["decile", "specialty", "region"],
        target_column="prescribing_change",
        geographic_scope="US",
    )


@pytest.fixture
def twin_model_metrics():
    """Sample TwinModelMetrics."""
    return TwinModelMetrics(
        model_id=uuid4(),
        r2_score=0.85,
        rmse=0.12,
        mae=0.08,
        cv_scores=[0.82, 0.84, 0.86, 0.85, 0.83],
        cv_mean=0.84,
        cv_std=0.015,
        feature_importances={"decile": 0.45, "specialty": 0.30, "region": 0.25},
        top_features=["decile", "specialty", "region"],
        training_samples=5000,
        training_duration_seconds=120.5,
    )


class TestTwinModelRepository:
    """Tests for TwinModelRepository."""

    @pytest.mark.asyncio
    async def test_save_model(self, mock_supabase, mock_mlflow, mock_redis, twin_model_config, twin_model_metrics):
        """Test saving a twin model."""
        repo = TwinModelRepository(mock_supabase, mock_mlflow, mock_redis)
        mock_supabase.execute.return_value = MagicMock(data=[{"model_id": str(twin_model_metrics.model_id)}])

        model_id = await repo.save_model(twin_model_config, twin_model_metrics)

        assert model_id == twin_model_metrics.model_id
        mock_supabase.table.assert_called_once_with("digital_twin_models")
        mock_supabase.insert.assert_called_once()
        mock_supabase.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_model_no_client(self, twin_model_config, twin_model_metrics):
        """Test save model without client."""
        repo = TwinModelRepository(None, None, None)

        model_id = await repo.save_model(twin_model_config, twin_model_metrics)

        assert model_id == twin_model_metrics.model_id

    @pytest.mark.asyncio
    async def test_get_model_from_cache(self, mock_supabase, mock_redis):
        """Test getting model from Redis cache."""
        repo = TwinModelRepository(mock_supabase, None, mock_redis)
        model_id = uuid4()
        cached_data = '{"model_id": "' + str(model_id) + '", "model_name": "Test"}'
        mock_redis.get.return_value = cached_data

        result = await repo.get_model(model_id)

        assert result["model_id"] == str(model_id)
        mock_redis.get.assert_called_once()
        mock_supabase.table.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_model_from_database(self, mock_supabase):
        """Test getting model from database."""
        repo = TwinModelRepository(mock_supabase, None, None)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(data=[{
            "model_id": str(model_id),
            "model_name": "Test Model"
        }])

        result = await repo.get_model(model_id)

        assert result["model_id"] == str(model_id)
        assert result["model_name"] == "Test Model"
        mock_supabase.table.assert_called_with("digital_twin_models")
        mock_supabase.eq.assert_called_with("model_id", str(model_id))

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, mock_supabase):
        """Test getting non-existent model."""
        repo = TwinModelRepository(mock_supabase, None, None)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(data=[])

        result = await repo.get_model(model_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_list_active_models(self, mock_supabase):
        """Test listing active models."""
        repo = TwinModelRepository(mock_supabase, None, None)
        mock_supabase.execute.return_value = MagicMock(data=[
            {"model_id": str(uuid4()), "is_active": True},
            {"model_id": str(uuid4()), "is_active": True},
        ])

        result = await repo.list_active_models(twin_type=TwinType.HCP, brand="Kisqali", limit=10)

        assert len(result) == 2
        mock_supabase.eq.assert_any_call("is_active", True)
        mock_supabase.eq.assert_any_call("twin_type", TwinType.HCP.value)
        mock_supabase.eq.assert_any_call("brand", "Kisqali")

    @pytest.mark.asyncio
    async def test_deactivate_model(self, mock_supabase, mock_redis):
        """Test deactivating a model."""
        repo = TwinModelRepository(mock_supabase, None, mock_redis)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock()

        result = await repo.deactivate_model(model_id, "Testing")

        assert result is True
        mock_supabase.update.assert_called_once()
        mock_supabase.eq.assert_called_with("model_id", str(model_id))
        mock_redis.delete.assert_called_with(f"twin_model:{model_id}")

    @pytest.mark.asyncio
    async def test_update_fidelity_score(self, mock_supabase, mock_redis):
        """Test updating model fidelity score."""
        repo = TwinModelRepository(mock_supabase, None, mock_redis)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock()

        result = await repo.update_fidelity_score(model_id, 0.85, 100)

        assert result is True
        update_call = mock_supabase.update.call_args[0][0]
        assert update_call["fidelity_score"] == 0.85
        assert update_call["fidelity_sample_count"] == 100


class TestSimulationRepository:
    """Tests for SimulationRepository."""

    @pytest.fixture
    def simulation_result(self):
        """Sample SimulationResult."""
        return SimulationResult(
            model_id=uuid4(),
            intervention_config=InterventionConfig(
                intervention_type="email_campaign",
                channel="email",
                frequency="weekly",
                duration_weeks=8,
            ),
            population_filters=PopulationFilter(specialties=["cardiology"], deciles=[1, 2, 3]),
            twin_count=1000,
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            simulated_std_error=0.015,
            effect_heterogeneity=EffectHeterogeneity(),
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Positive effect detected",
            recommended_sample_size=500,
            recommended_duration_weeks=8,
            simulation_confidence=0.85,
            status=SimulationStatus.COMPLETED,
            execution_time_ms=1500,
        )

    @pytest.mark.asyncio
    async def test_save_simulation(self, mock_supabase, simulation_result):
        """Test saving simulation result."""
        repo = SimulationRepository(mock_supabase)
        mock_supabase.execute.return_value = MagicMock()

        result_id = await repo.save_simulation(simulation_result, "Kisqali")

        assert result_id == simulation_result.simulation_id
        mock_supabase.table.assert_called_with("twin_simulations")
        mock_supabase.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_simulation(self, mock_supabase):
        """Test getting simulation by ID."""
        repo = SimulationRepository(mock_supabase)
        sim_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(data=[{"simulation_id": str(sim_id)}])

        result = await repo.get_simulation(sim_id)

        assert result["simulation_id"] == str(sim_id)
        mock_supabase.eq.assert_called_with("simulation_id", str(sim_id))

    @pytest.mark.asyncio
    async def test_list_simulations(self, mock_supabase):
        """Test listing simulations with filters."""
        repo = SimulationRepository(mock_supabase)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(data=[
            {"simulation_id": str(uuid4())},
            {"simulation_id": str(uuid4())},
        ])

        result = await repo.list_simulations(
            model_id=model_id,
            brand="Kisqali",
            status=SimulationStatus.COMPLETED,
            limit=50
        )

        assert len(result) == 2
        mock_supabase.eq.assert_any_call("model_id", str(model_id))
        mock_supabase.eq.assert_any_call("brand", "Kisqali")
        mock_supabase.eq.assert_any_call("simulation_status", SimulationStatus.COMPLETED.value)

    @pytest.mark.asyncio
    async def test_update_status(self, mock_supabase):
        """Test updating simulation status."""
        repo = SimulationRepository(mock_supabase)
        sim_id = uuid4()
        mock_supabase.execute.return_value = MagicMock()

        result = await repo.update_status(sim_id, SimulationStatus.RUNNING)

        assert result is True
        update_call = mock_supabase.update.call_args[0][0]
        assert update_call["simulation_status"] == SimulationStatus.RUNNING.value
        assert "started_at" in update_call

    @pytest.mark.asyncio
    async def test_link_experiment(self, mock_supabase):
        """Test linking simulation to experiment."""
        repo = SimulationRepository(mock_supabase)
        sim_id = uuid4()
        exp_id = uuid4()
        mock_supabase.execute.return_value = MagicMock()

        result = await repo.link_experiment(sim_id, exp_id)

        assert result is True
        update_call = mock_supabase.update.call_args[0][0]
        assert update_call["experiment_design_id"] == str(exp_id)


class TestFidelityRepository:
    """Tests for FidelityRepository."""

    @pytest.fixture
    def fidelity_record(self):
        """Sample FidelityRecord."""
        return FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            actual_ate=0.09,
            actual_ci_lower=0.06,
            actual_ci_upper=0.12,
            actual_sample_size=450,
        )

    @pytest.mark.asyncio
    async def test_save_fidelity_record(self, mock_supabase, fidelity_record):
        """Test saving fidelity record."""
        repo = FidelityRepository(mock_supabase)
        mock_supabase.execute.return_value = MagicMock()

        result_id = await repo.save_fidelity_record(fidelity_record)

        assert result_id == fidelity_record.tracking_id
        mock_supabase.table.assert_called_with("twin_fidelity_tracking")
        mock_supabase.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_fidelity_validation(self, mock_supabase):
        """Test updating fidelity validation."""
        repo = FidelityRepository(mock_supabase)
        tracking_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(data=[{
            "tracking_id": str(tracking_id),
            "actual_ate": 0.09,
        }])

        result = await repo.update_fidelity_validation(
            tracking_id,
            actual_ate=0.09,
            actual_ci_lower=0.06,
            actual_ci_upper=0.12,
            validated_by="admin"
        )

        assert result is not None
        assert result["actual_ate"] == 0.09
        update_call = mock_supabase.update.call_args[0][0]
        assert update_call["actual_ate"] == 0.09
        assert update_call["validated_by"] == "admin"

    @pytest.mark.asyncio
    async def test_get_fidelity_by_simulation(self, mock_supabase):
        """Test getting fidelity record by simulation ID."""
        repo = FidelityRepository(mock_supabase)
        sim_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(data=[{
            "tracking_id": str(uuid4()),
            "simulation_id": str(sim_id),
            "simulated_ate": 0.08,
            "fidelity_grade": "good",
        }])

        result = await repo.get_fidelity_by_simulation(sim_id)

        assert result is not None
        assert result.simulation_id == sim_id
        assert result.simulated_ate == 0.08

    @pytest.mark.asyncio
    async def test_get_model_fidelity_records(self, mock_supabase):
        """Test getting fidelity records for a model."""
        repo = FidelityRepository(mock_supabase)
        model_id = uuid4()
        mock_supabase.execute.return_value = MagicMock(data=[
            {
                "tracking_id": str(uuid4()),
                "simulation_id": str(uuid4()),
                "simulated_ate": 0.08,
                "fidelity_grade": "good",
                "validated_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "tracking_id": str(uuid4()),
                "simulation_id": str(uuid4()),
                "simulated_ate": 0.10,
                "fidelity_grade": "excellent",
                "validated_at": datetime.now(timezone.utc).isoformat(),
            },
        ])

        result = await repo.get_model_fidelity_records(model_id, validated_only=True, limit=50)

        assert len(result) == 2
        assert all(isinstance(r, FidelityRecord) for r in result)


class TestTwinRepository:
    """Tests for unified TwinRepository facade."""

    def test_initialization(self, mock_supabase, mock_mlflow, mock_redis):
        """Test TwinRepository initialization."""
        repo = TwinRepository(mock_supabase, mock_mlflow, mock_redis)

        assert isinstance(repo.models, TwinModelRepository)
        assert isinstance(repo.simulations, SimulationRepository)
        assert isinstance(repo.fidelity, FidelityRepository)

    @pytest.mark.asyncio
    async def test_save_model_delegation(self, mock_supabase, twin_model_config, twin_model_metrics):
        """Test save_model delegates to models repository."""
        repo = TwinRepository(mock_supabase, None, None)
        mock_supabase.execute.return_value = MagicMock()

        with patch.object(repo.models, 'save_model', new_callable=AsyncMock) as mock_save:
            mock_save.return_value = twin_model_metrics.model_id
            result = await repo.save_model(twin_model_config, twin_model_metrics)

            mock_save.assert_called_once_with(twin_model_config, twin_model_metrics, None, None)
            assert result == twin_model_metrics.model_id

    @pytest.mark.asyncio
    async def test_get_model_delegation(self, mock_supabase):
        """Test get_model delegates to models repository."""
        repo = TwinRepository(mock_supabase, None, None)
        model_id = uuid4()

        with patch.object(repo.models, 'get_model', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"model_id": str(model_id)}
            result = await repo.get_model(model_id)

            mock_get.assert_called_once_with(model_id)
            assert result["model_id"] == str(model_id)

    @pytest.mark.asyncio
    async def test_list_active_models_delegation(self, mock_supabase):
        """Test list_active_models delegates to models repository."""
        repo = TwinRepository(mock_supabase, None, None)

        with patch.object(repo.models, 'list_active_models', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [{"model_id": str(uuid4())}]
            result = await repo.list_active_models(twin_type=TwinType.HCP, brand="Kisqali")

            mock_list.assert_called_once_with(TwinType.HCP, "Kisqali")
            assert len(result) == 1
