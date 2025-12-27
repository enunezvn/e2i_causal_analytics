"""
Unit tests for Twin Repository classes.

Tests TwinModelRepository, SimulationRepository, FidelityRepository, and TwinRepository.
Uses mocked database clients for unit testing.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.digital_twin.models.simulation_models import (
    EffectHeterogeneity,
    FidelityGrade,
    FidelityRecord,
    InterventionConfig,
    PopulationFilter,
    SimulationRecommendation,
    SimulationResult,
    SimulationStatus,
)
from src.digital_twin.models.twin_models import (
    Brand,
    TwinModelConfig,
    TwinModelMetrics,
    TwinType,
)
from src.digital_twin.twin_repository import (
    FidelityRepository,
    SimulationRepository,
    TwinModelRepository,
    TwinRepository,
)


class TestTwinModelRepositoryInit:
    """Tests for TwinModelRepository initialization."""

    def test_init_without_clients(self):
        """Test initialization without any clients."""
        repo = TwinModelRepository()

        assert repo.client is None
        assert repo.mlflow_client is None
        assert repo.redis_client is None

    def test_init_with_clients(self):
        """Test initialization with all clients."""
        mock_supabase = MagicMock()
        mock_mlflow = MagicMock()
        mock_redis = MagicMock()

        repo = TwinModelRepository(
            supabase_client=mock_supabase,
            mlflow_client=mock_mlflow,
            redis_client=mock_redis,
        )

        assert repo.client is mock_supabase
        assert repo.mlflow_client is mock_mlflow
        assert repo.redis_client is mock_redis

    def test_table_name(self):
        """Test that table name is set correctly."""
        repo = TwinModelRepository()

        assert repo.table_name == "digital_twin_models"


class TestTwinModelRepositorySaveModel:
    """Tests for TwinModelRepository.save_model method."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        # Configure table method to return a chainable mock
        mock_table = MagicMock()
        mock_insert = AsyncMock()
        mock_table.insert.return_value.execute = mock_insert
        client.table.return_value = mock_table
        return client

    @pytest.fixture
    def model_config(self):
        """Create a sample model config."""
        return TwinModelConfig(
            model_name="test_twin_model",
            model_description="Test description",
            twin_type=TwinType.HCP,
            algorithm="random_forest",
            n_estimators=100,
            max_depth=10,
            training_samples=5000,
            validation_split=0.2,
            cv_folds=5,
            feature_columns=["specialty", "decile", "region"],
            target_column="prescribing_change",
            brand=Brand.REMIBRUTINIB,
        )

    @pytest.fixture
    def model_metrics(self):
        """Create sample model metrics."""
        return TwinModelMetrics(
            model_id=uuid4(),
            r2_score=0.85,
            rmse=0.02,
            mae=0.1,
            cv_mean=0.82,
            cv_std=0.03,
            training_samples=5000,
            training_duration_seconds=120.5,
        )

    @pytest.mark.asyncio
    async def test_save_model_returns_model_id(self, mock_client, model_config, model_metrics):
        """Test that save_model returns model ID."""
        repo = TwinModelRepository(supabase_client=mock_client)

        result = await repo.save_model(model_config, model_metrics)

        assert result == model_metrics.model_id

    @pytest.mark.asyncio
    async def test_save_model_inserts_to_database(self, mock_client, model_config, model_metrics):
        """Test that save_model inserts row to database."""
        repo = TwinModelRepository(supabase_client=mock_client)

        await repo.save_model(model_config, model_metrics)

        mock_client.table.assert_called_with("digital_twin_models")
        mock_client.table.return_value.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_model_with_redis_caching(self, mock_client, model_config, model_metrics):
        """Test that save_model caches model info in Redis."""
        mock_redis = MagicMock()
        repo = TwinModelRepository(
            supabase_client=mock_client,
            redis_client=mock_redis,
        )

        await repo.save_model(model_config, model_metrics)

        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_model_without_client(self, model_config, model_metrics):
        """Test save_model without database client."""
        repo = TwinModelRepository()

        result = await repo.save_model(model_config, model_metrics)

        # Should still return model ID even without persistence
        assert result == model_metrics.model_id


class TestTwinModelRepositoryGetModel:
    """Tests for TwinModelRepository.get_model method."""

    @pytest.fixture
    def mock_client_with_data(self):
        """Create mock client that returns data."""
        client = MagicMock()
        mock_table = MagicMock()

        # Chain: table().select().eq().execute()
        mock_select = MagicMock()
        mock_eq = MagicMock()
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"model_id": "test-id", "model_name": "test"}]))

        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        mock_eq.execute = mock_execute

        client.table.return_value = mock_table
        return client

    @pytest.mark.asyncio
    async def test_get_model_from_cache(self):
        """Test get_model returns cached data when available."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({"model_id": "cached"})

        repo = TwinModelRepository(redis_client=mock_redis)
        model_id = uuid4()

        result = await repo.get_model(model_id)

        assert result == {"model_id": "cached"}
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_from_database(self, mock_client_with_data):
        """Test get_model queries database when not in cache."""
        repo = TwinModelRepository(supabase_client=mock_client_with_data)
        model_id = uuid4()

        result = await repo.get_model(model_id)

        assert result is not None
        mock_client_with_data.table.assert_called_with("digital_twin_models")

    @pytest.mark.asyncio
    async def test_get_model_not_found(self):
        """Test get_model returns None when model not found."""
        client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_eq = MagicMock()
        mock_eq.execute = AsyncMock(return_value=MagicMock(data=[]))

        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        client.table.return_value = mock_table

        repo = TwinModelRepository(supabase_client=client)

        result = await repo.get_model(uuid4())

        assert result is None


class TestTwinModelRepositoryListActiveModels:
    """Tests for TwinModelRepository.list_active_models method."""

    @pytest.mark.asyncio
    async def test_list_active_models_no_client(self):
        """Test list_active_models returns empty list without client."""
        repo = TwinModelRepository()

        result = await repo.list_active_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_active_models_with_filters(self):
        """Test list_active_models applies filters correctly."""
        client = MagicMock()
        mock_table = MagicMock()

        # Build query chain
        mock_chain = MagicMock()
        mock_chain.eq.return_value = mock_chain
        mock_chain.order.return_value = mock_chain
        mock_chain.limit.return_value = mock_chain
        mock_chain.execute = AsyncMock(return_value=MagicMock(data=[{"model_id": "1"}]))

        mock_table.select.return_value = mock_chain
        client.table.return_value = mock_table

        repo = TwinModelRepository(supabase_client=client)

        result = await repo.list_active_models(twin_type=TwinType.HCP, brand="Remibrutinib")

        assert len(result) == 1


class TestSimulationRepositoryInit:
    """Tests for SimulationRepository initialization."""

    def test_init(self):
        """Test initialization."""
        repo = SimulationRepository()

        assert repo.table_name == "twin_simulations"
        assert repo.client is None


class TestSimulationRepositorySaveSimulation:
    """Tests for SimulationRepository.save_simulation method."""

    @pytest.fixture
    def simulation_result(self):
        """Create a sample simulation result."""
        return SimulationResult(
            model_id=uuid4(),
            intervention_config=InterventionConfig(
                intervention_type="email_campaign",
                channel="email",
                frequency="weekly",
                duration_weeks=12,
            ),
            twin_count=1000,
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            simulated_std_error=0.015,
            status=SimulationStatus.COMPLETED,
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Effect is significant",
            simulation_confidence=0.85,
            execution_time_ms=150,
        )

    @pytest.mark.asyncio
    async def test_save_simulation_no_client(self, simulation_result):
        """Test save_simulation returns ID without client."""
        repo = SimulationRepository()

        result = await repo.save_simulation(simulation_result, brand="Remibrutinib")

        assert result == simulation_result.simulation_id

    @pytest.mark.asyncio
    async def test_save_simulation_with_client(self, simulation_result):
        """Test save_simulation inserts to database."""
        client = MagicMock()
        mock_table = MagicMock()
        mock_table.insert.return_value.execute = AsyncMock()
        client.table.return_value = mock_table

        repo = SimulationRepository(supabase_client=client)

        result = await repo.save_simulation(simulation_result, brand="Remibrutinib")

        client.table.assert_called_with("twin_simulations")
        assert result == simulation_result.simulation_id


class TestSimulationRepositoryGetSimulation:
    """Tests for SimulationRepository.get_simulation method."""

    @pytest.mark.asyncio
    async def test_get_simulation_no_client(self):
        """Test get_simulation returns None without client."""
        repo = SimulationRepository()

        result = await repo.get_simulation(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_simulation_found(self):
        """Test get_simulation returns data when found."""
        client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_eq = MagicMock()
        mock_eq.execute = AsyncMock(return_value=MagicMock(data=[{"simulation_id": "test"}]))

        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        client.table.return_value = mock_table

        repo = SimulationRepository(supabase_client=client)

        result = await repo.get_simulation(uuid4())

        assert result == {"simulation_id": "test"}


class TestSimulationRepositoryListSimulations:
    """Tests for SimulationRepository.list_simulations method."""

    @pytest.mark.asyncio
    async def test_list_simulations_no_client(self):
        """Test list_simulations returns empty list without client."""
        repo = SimulationRepository()

        result = await repo.list_simulations()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_simulations_with_filters(self):
        """Test list_simulations applies filters."""
        client = MagicMock()
        mock_table = MagicMock()

        mock_chain = MagicMock()
        mock_chain.eq.return_value = mock_chain
        mock_chain.order.return_value = mock_chain
        mock_chain.limit.return_value = mock_chain
        mock_chain.execute = AsyncMock(return_value=MagicMock(data=[]))

        mock_table.select.return_value = mock_chain
        client.table.return_value = mock_table

        repo = SimulationRepository(supabase_client=client)

        result = await repo.list_simulations(
            model_id=uuid4(),
            brand="Remibrutinib",
            status=SimulationStatus.COMPLETED,
        )

        assert result == []


class TestSimulationRepositoryUpdateStatus:
    """Tests for SimulationRepository.update_status method."""

    @pytest.mark.asyncio
    async def test_update_status_no_client(self):
        """Test update_status returns False without client."""
        repo = SimulationRepository()

        result = await repo.update_status(uuid4(), SimulationStatus.COMPLETED)

        assert result is False

    @pytest.mark.asyncio
    async def test_update_status_success(self):
        """Test update_status updates record."""
        client = MagicMock()
        mock_table = MagicMock()
        mock_update = MagicMock()
        mock_eq = MagicMock()
        mock_eq.execute = AsyncMock()

        mock_table.update.return_value = mock_update
        mock_update.eq.return_value = mock_eq
        client.table.return_value = mock_table

        repo = SimulationRepository(supabase_client=client)

        result = await repo.update_status(uuid4(), SimulationStatus.COMPLETED)

        assert result is True


class TestFidelityRepositoryInit:
    """Tests for FidelityRepository initialization."""

    def test_init(self):
        """Test initialization."""
        repo = FidelityRepository()

        assert repo.table_name == "twin_fidelity_tracking"
        assert repo.client is None


class TestFidelityRepositorySaveFidelityRecord:
    """Tests for FidelityRepository.save_fidelity_record method."""

    @pytest.fixture
    def fidelity_record(self):
        """Create a sample fidelity record."""
        return FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.10,
            simulated_ci_lower=0.06,
            simulated_ci_upper=0.14,
        )

    @pytest.mark.asyncio
    async def test_save_fidelity_record_no_client(self, fidelity_record):
        """Test save_fidelity_record returns ID without client."""
        repo = FidelityRepository()

        result = await repo.save_fidelity_record(fidelity_record)

        assert result == fidelity_record.tracking_id

    @pytest.mark.asyncio
    async def test_save_fidelity_record_with_client(self, fidelity_record):
        """Test save_fidelity_record inserts to database."""
        client = MagicMock()
        mock_table = MagicMock()
        mock_table.insert.return_value.execute = AsyncMock()
        client.table.return_value = mock_table

        repo = FidelityRepository(supabase_client=client)

        result = await repo.save_fidelity_record(fidelity_record)

        client.table.assert_called_with("twin_fidelity_tracking")
        assert result == fidelity_record.tracking_id


class TestFidelityRepositoryUpdateFidelityValidation:
    """Tests for FidelityRepository.update_fidelity_validation method."""

    @pytest.mark.asyncio
    async def test_update_fidelity_validation_no_client(self):
        """Test update_fidelity_validation returns None without client."""
        repo = FidelityRepository()

        result = await repo.update_fidelity_validation(
            tracking_id=uuid4(),
            actual_ate=0.09,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_update_fidelity_validation_success(self):
        """Test update_fidelity_validation updates record."""
        client = MagicMock()
        mock_table = MagicMock()
        mock_update = MagicMock()
        mock_eq = MagicMock()
        mock_select = MagicMock()
        mock_select.execute = AsyncMock(return_value=MagicMock(data=[{"tracking_id": "test"}]))

        mock_table.update.return_value = mock_update
        mock_update.eq.return_value = mock_eq
        mock_eq.select.return_value = mock_select
        client.table.return_value = mock_table

        repo = FidelityRepository(supabase_client=client)

        result = await repo.update_fidelity_validation(
            tracking_id=uuid4(),
            actual_ate=0.09,
            actual_ci_lower=0.05,
            actual_ci_upper=0.13,
        )

        assert result == {"tracking_id": "test"}


class TestFidelityRepositoryGetFidelityBySimulation:
    """Tests for FidelityRepository.get_fidelity_by_simulation method."""

    @pytest.mark.asyncio
    async def test_get_fidelity_by_simulation_no_client(self):
        """Test get_fidelity_by_simulation returns None without client."""
        repo = FidelityRepository()

        result = await repo.get_fidelity_by_simulation(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_fidelity_by_simulation_found(self):
        """Test get_fidelity_by_simulation returns record when found."""
        client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_eq = MagicMock()

        simulation_id = uuid4()
        tracking_id = uuid4()
        mock_eq.execute = AsyncMock(return_value=MagicMock(data=[{
            "tracking_id": str(tracking_id),
            "simulation_id": str(simulation_id),
            "simulated_ate": 0.10,
            "simulated_ci_lower": 0.06,
            "simulated_ci_upper": 0.14,
            "fidelity_grade": FidelityGrade.UNVALIDATED.value,
            "confounding_factors": [],
        }]))

        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        client.table.return_value = mock_table

        repo = FidelityRepository(supabase_client=client)

        result = await repo.get_fidelity_by_simulation(simulation_id)

        assert result is not None
        assert isinstance(result, FidelityRecord)


class TestFidelityRepositoryToFidelityRecord:
    """Tests for FidelityRepository._to_fidelity_record method."""

    def test_to_fidelity_record_minimal(self):
        """Test conversion with minimal data."""
        repo = FidelityRepository()
        simulation_id = uuid4()
        tracking_id = uuid4()

        row = {
            "tracking_id": str(tracking_id),
            "simulation_id": str(simulation_id),
            "simulated_ate": 0.10,
            "fidelity_grade": FidelityGrade.UNVALIDATED.value,
        }

        record = repo._to_fidelity_record(row)

        assert record.tracking_id == tracking_id
        assert record.simulation_id == simulation_id
        assert record.simulated_ate == 0.10

    def test_to_fidelity_record_full(self):
        """Test conversion with full data."""
        repo = FidelityRepository()
        simulation_id = uuid4()
        tracking_id = uuid4()
        experiment_id = uuid4()
        validated_at = datetime.now(timezone.utc)

        row = {
            "tracking_id": str(tracking_id),
            "simulation_id": str(simulation_id),
            "simulated_ate": 0.10,
            "simulated_ci_lower": 0.06,
            "simulated_ci_upper": 0.14,
            "actual_ate": 0.09,
            "actual_ci_lower": 0.05,
            "actual_ci_upper": 0.13,
            "actual_sample_size": 800,
            "actual_experiment_id": str(experiment_id),
            "prediction_error": -0.111,
            "absolute_error": 0.01,
            "ci_coverage": True,
            "fidelity_grade": FidelityGrade.GOOD.value,
            "validation_notes": "Test notes",
            "confounding_factors": ["seasonality"],
            "validated_by": "analyst",
            "validated_at": validated_at.isoformat(),
        }

        record = repo._to_fidelity_record(row)

        assert record.actual_ate == 0.09
        assert record.actual_experiment_id == experiment_id
        assert record.fidelity_grade == FidelityGrade.GOOD
        assert record.confounding_factors == ["seasonality"]


class TestTwinRepositoryFacade:
    """Tests for TwinRepository unified facade."""

    def test_init_creates_sub_repositories(self):
        """Test that initialization creates all sub-repositories."""
        repo = TwinRepository()

        assert isinstance(repo.models, TwinModelRepository)
        assert isinstance(repo.simulations, SimulationRepository)
        assert isinstance(repo.fidelity, FidelityRepository)

    def test_init_with_clients_passes_to_sub_repos(self):
        """Test that clients are passed to sub-repositories."""
        mock_supabase = MagicMock()
        mock_mlflow = MagicMock()
        mock_redis = MagicMock()

        repo = TwinRepository(
            supabase_client=mock_supabase,
            mlflow_client=mock_mlflow,
            redis_client=mock_redis,
        )

        assert repo.models.client is mock_supabase
        assert repo.models.mlflow_client is mock_mlflow
        assert repo.models.redis_client is mock_redis
        assert repo.simulations.client is mock_supabase
        assert repo.fidelity.client is mock_supabase

    @pytest.mark.asyncio
    async def test_convenience_method_save_model(self):
        """Test convenience method delegates to models repository."""
        repo = TwinRepository()
        repo.models.save_model = AsyncMock(return_value=uuid4())

        config = TwinModelConfig(
            model_name="test",
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            feature_columns=["specialty", "decile", "region"],
            target_column="prescribing_change",
        )
        metrics = TwinModelMetrics(
            model_id=uuid4(),
            training_samples=5000,
            training_duration_seconds=120.5,
        )

        await repo.save_model(config, metrics)

        repo.models.save_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_convenience_method_get_model(self):
        """Test convenience method delegates to models repository."""
        repo = TwinRepository()
        repo.models.get_model = AsyncMock(return_value={"model_id": "test"})

        model_id = uuid4()
        result = await repo.get_model(model_id)

        assert result == {"model_id": "test"}
        repo.models.get_model.assert_called_once_with(model_id)

    @pytest.mark.asyncio
    async def test_convenience_method_save_simulation(self):
        """Test convenience method delegates to simulations repository."""
        repo = TwinRepository()
        simulation_id = uuid4()
        repo.simulations.save_simulation = AsyncMock(return_value=simulation_id)

        result = SimulationResult(
            model_id=uuid4(),
            intervention_config=InterventionConfig(
                intervention_type="email_campaign",
                channel="email",
            ),
            twin_count=1000,
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            simulated_std_error=0.015,
            status=SimulationStatus.COMPLETED,
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Effect is significant",
            simulation_confidence=0.85,
            execution_time_ms=150,
        )

        returned_id = await repo.save_simulation(result, brand="Remibrutinib")

        assert returned_id == simulation_id
        repo.simulations.save_simulation.assert_called_once()

    @pytest.mark.asyncio
    async def test_convenience_method_save_fidelity_record(self):
        """Test convenience method delegates to fidelity repository."""
        repo = TwinRepository()
        tracking_id = uuid4()
        repo.fidelity.save_fidelity_record = AsyncMock(return_value=tracking_id)

        record = FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.10,
        )

        returned_id = await repo.save_fidelity_record(record)

        assert returned_id == tracking_id
        repo.fidelity.save_fidelity_record.assert_called_once_with(record)
