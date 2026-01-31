"""
Unit tests for ML Experiment Repositories.

Tests MLExperimentRepository, MLTrainingRunRepository, and MLModelRegistryRepository.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.repositories.ml_experiment import (
    MLExperiment,
    MLExperimentRepository,
    MLModelRegistry,
    MLModelRegistryRepository,
    MLTrainingRun,
    MLTrainingRunRepository,
    ModelStage,
    TrainingStatus,
)


@pytest.mark.unit
class TestMLExperimentDataClass:
    """Tests for MLExperiment data class."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all fields."""
        experiment = MLExperiment(
            id=uuid4(),
            experiment_name="test_experiment",
            mlflow_experiment_id="mlflow-123",
            prediction_target="churn",
            brand="Kisqali",
        )

        data = experiment.to_dict()

        assert "id" in data
        assert "experiment_name" in data
        assert "mlflow_experiment_id" in data
        assert "prediction_target" in data
        assert "brand" in data

    def test_from_dict_creates_instance(self):
        """Test that from_dict creates valid instance."""
        data = {
            "id": str(uuid4()),
            "experiment_name": "test_experiment",
            "prediction_target": "churn",
            "brand": "Kisqali",
        }

        experiment = MLExperiment.from_dict(data)

        assert experiment.experiment_name == "test_experiment"
        assert experiment.prediction_target == "churn"
        assert experiment.brand == "Kisqali"

    def test_from_dict_handles_missing_fields(self):
        """Test that from_dict handles missing optional fields."""
        data = {
            "experiment_name": "test_experiment",
            "prediction_target": "churn",
        }

        experiment = MLExperiment.from_dict(data)

        assert experiment.experiment_name == "test_experiment"
        assert experiment.brand is None


@pytest.mark.unit
class TestMLTrainingRunDataClass:
    """Tests for MLTrainingRun data class."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all fields."""
        run = MLTrainingRun(
            id=uuid4(),
            experiment_id=uuid4(),
            algorithm="RandomForest",
            training_samples=1000,
        )

        data = run.to_dict()

        assert "id" in data
        assert "experiment_id" in data
        assert "algorithm" in data
        assert "training_samples" in data

    def test_from_dict_creates_instance(self):
        """Test that from_dict creates valid instance."""
        data = {
            "id": str(uuid4()),
            "experiment_id": str(uuid4()),
            "algorithm": "RandomForest",
            "training_samples": 1000,
        }

        run = MLTrainingRun.from_dict(data)

        assert run.algorithm == "RandomForest"
        assert run.training_samples == 1000


@pytest.mark.unit
class TestMLModelRegistryDataClass:
    """Tests for MLModelRegistry data class."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all fields."""
        model = MLModelRegistry(
            id=uuid4(),
            experiment_id=uuid4(),
            model_name="churn_predictor",
            model_version="v1.0.0",
            algorithm="RandomForest",
        )

        data = model.to_dict()

        assert "id" in data
        assert "experiment_id" in data
        assert "model_name" in data
        assert "model_version" in data
        assert "algorithm" in data

    def test_from_dict_creates_instance(self):
        """Test that from_dict creates valid instance."""
        data = {
            "id": str(uuid4()),
            "experiment_id": str(uuid4()),
            "model_name": "churn_predictor",
            "model_version": "v1.0.0",
            "algorithm": "RandomForest",
        }

        model = MLModelRegistry.from_dict(data)

        assert model.model_name == "churn_predictor"
        assert model.model_version == "v1.0.0"
        assert model.algorithm == "RandomForest"


@pytest.mark.unit
class TestMLExperimentRepository:
    """Tests for MLExperimentRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return MLExperimentRepository(supabase_client=mock_client)

    @pytest.fixture
    def sample_experiment_data(self):
        """Sample experiment data."""
        return {
            "id": str(uuid4()),
            "experiment_name": "test_experiment",
            "mlflow_experiment_id": "mlflow-123",
            "prediction_target": "churn",
            "brand": "Kisqali",
            "region": "US",
            "minimum_auc": 0.75,
            "created_by": "test_user",
        }

    @pytest.mark.asyncio
    async def test_get_by_name_returns_experiment(self, repo, mock_client, sample_experiment_data):
        """Test that get_by_name returns experiment when found."""
        mock_result = MagicMock()
        mock_result.data = [sample_experiment_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_by_name("test_experiment")

        assert result is not None
        assert result.experiment_name == "test_experiment"
        mock_client.table.assert_called_with("ml_experiments")

    @pytest.mark.asyncio
    async def test_get_by_name_returns_none_when_not_found(self, repo, mock_client):
        """Test that get_by_name returns None when not found."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_by_name("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_mlflow_id_returns_experiment(self, repo, mock_client, sample_experiment_data):
        """Test that get_by_mlflow_id returns experiment when found."""
        mock_result = MagicMock()
        mock_result.data = [sample_experiment_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_by_mlflow_id("mlflow-123")

        assert result is not None
        assert result.mlflow_experiment_id == "mlflow-123"

    @pytest.mark.asyncio
    async def test_create_experiment_inserts_and_returns(self, repo, mock_client, sample_experiment_data):
        """Test that create_experiment inserts and returns experiment."""
        mock_result = MagicMock()
        mock_result.data = [sample_experiment_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        result = await repo.create_experiment(
            name="test_experiment",
            mlflow_experiment_id="mlflow-123",
            prediction_target="churn",
            brand="Kisqali",
            created_by="test_user",
            success_criteria={"minimum_auc": 0.75},
        )

        assert result is not None
        assert result.experiment_name == "test_experiment"
        assert result.minimum_auc == 0.75

    @pytest.mark.asyncio
    async def test_list_experiments_returns_all(self, repo, mock_client, sample_experiment_data):
        """Test that list_experiments returns all experiments."""
        mock_result = MagicMock()
        mock_result.data = [sample_experiment_data, {**sample_experiment_data, "id": str(uuid4())}]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.list_experiments()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_experiments_filters_by_brand(self, repo, mock_client, sample_experiment_data):
        """Test that list_experiments filters by brand."""
        mock_result = MagicMock()
        mock_result.data = [sample_experiment_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.list_experiments(brand="Kisqali")

        assert len(result) == 1


@pytest.mark.unit
class TestMLTrainingRunRepository:
    """Tests for MLTrainingRunRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return MLTrainingRunRepository(supabase_client=mock_client)

    @pytest.fixture
    def sample_run_data(self):
        """Sample training run data."""
        return {
            "id": str(uuid4()),
            "experiment_id": str(uuid4()),
            "run_name": "test_run",
            "mlflow_run_id": "run-123",
            "algorithm": "RandomForest",
            "hyperparameters": {"n_estimators": 100},
            "training_samples": 1000,
            "feature_names": ["feature1", "feature2"],
            "train_metrics": {"auc": 0.85},
            "validation_metrics": {"auc": 0.82},
            "status": "finished",
        }

    @pytest.mark.asyncio
    async def test_get_by_mlflow_run_id_returns_run(self, repo, mock_client, sample_run_data):
        """Test that get_by_mlflow_run_id returns run when found."""
        mock_result = MagicMock()
        mock_result.data = [sample_run_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_by_mlflow_run_id("run-123")

        assert result is not None
        assert result.mlflow_run_id == "run-123"

    @pytest.mark.asyncio
    async def test_create_run_inserts_and_returns(self, repo, mock_client, sample_run_data):
        """Test that create_run inserts and returns run."""
        mock_result = MagicMock()
        mock_result.data = [sample_run_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        experiment_id = UUID(sample_run_data["experiment_id"])
        result = await repo.create_run(
            experiment_id=experiment_id,
            run_name="test_run",
            mlflow_run_id="run-123",
            algorithm="RandomForest",
            hyperparameters={"n_estimators": 100},
            training_samples=1000,
            feature_names=["feature1", "feature2"],
        )

        assert result is not None
        assert result.run_name == "test_run"
        assert result.algorithm == "RandomForest"

    @pytest.mark.asyncio
    async def test_update_run_metrics_updates_successfully(self, repo, mock_client):
        """Test that update_run_metrics updates metrics."""
        mock_result = MagicMock()
        mock_result.data = [{"id": str(uuid4())}]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        run_id = uuid4()
        result = await repo.update_run_metrics(
            run_id=run_id,
            train_metrics={"auc": 0.85},
            validation_metrics={"auc": 0.82},
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_run_metrics_returns_false_without_client(self):
        """Test that update_run_metrics returns False without client."""
        repo = MLTrainingRunRepository(supabase_client=None)
        result = await repo.update_run_metrics(run_id=uuid4(), train_metrics={"auc": 0.85})
        assert result is False

    @pytest.mark.asyncio
    async def test_complete_run_updates_status(self, repo, mock_client, sample_run_data):
        """Test that complete_run updates status and completion time."""
        # Mock get_by_id to return a run with started_at
        mock_run = MLTrainingRun.from_dict(sample_run_data)
        mock_run.started_at = datetime.now(timezone.utc)

        with patch.object(repo, 'get_by_id', new=AsyncMock(return_value=mock_run)):
            mock_result = MagicMock()
            mock_result.data = [sample_run_data]
            mock_execute = AsyncMock(return_value=mock_result)
            mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

            run_id = UUID(sample_run_data["id"])
            result = await repo.complete_run(run_id=run_id, status="finished")

            assert result is True

    @pytest.mark.asyncio
    async def test_get_runs_for_experiment_returns_runs(self, repo, mock_client, sample_run_data):
        """Test that get_runs_for_experiment returns runs."""
        mock_result = MagicMock()
        mock_result.data = [sample_run_data, {**sample_run_data, "id": str(uuid4())}]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        experiment_id = UUID(sample_run_data["experiment_id"])
        result = await repo.get_runs_for_experiment(experiment_id=experiment_id)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_runs_for_experiment_filters_by_status(self, repo, mock_client, sample_run_data):
        """Test that get_runs_for_experiment filters by status."""
        mock_result = MagicMock()
        mock_result.data = [sample_run_data]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_status = MagicMock()
        mock_eq_status.limit.return_value = mock_limit
        mock_eq_exp = MagicMock()
        mock_eq_exp.eq.return_value = mock_eq_status
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_exp

        experiment_id = UUID(sample_run_data["experiment_id"])
        result = await repo.get_runs_for_experiment(experiment_id=experiment_id, status="finished")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_best_run_returns_highest_metric(self, repo, mock_client):
        """Test that get_best_run returns run with highest metric."""
        run1_data = {
            "id": str(uuid4()),
            "experiment_id": str(uuid4()),
            "status": "finished",
            "test_metrics": {"auc": 0.85},
        }
        run2_data = {
            "id": str(uuid4()),
            "experiment_id": run1_data["experiment_id"],
            "status": "finished",
            "test_metrics": {"auc": 0.90},
        }

        runs = [MLTrainingRun.from_dict(run1_data), MLTrainingRun.from_dict(run2_data)]

        with patch.object(repo, 'get_runs_for_experiment', new=AsyncMock(return_value=runs)):
            experiment_id = UUID(run1_data["experiment_id"])
            result = await repo.get_best_run(experiment_id=experiment_id, metric="auc")

            assert result is not None
            assert result.test_metrics["auc"] == 0.90

    @pytest.mark.asyncio
    async def test_get_best_run_returns_none_when_no_runs(self, repo, mock_client):
        """Test that get_best_run returns None when no runs exist."""
        with patch.object(repo, 'get_runs_for_experiment', new=AsyncMock(return_value=[])):
            result = await repo.get_best_run(experiment_id=uuid4(), metric="auc")
            assert result is None

    @pytest.mark.asyncio
    async def test_set_optuna_info_updates_run(self, repo, mock_client):
        """Test that set_optuna_info updates run with HPO information."""
        mock_result = MagicMock()
        mock_result.data = [{"id": str(uuid4())}]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        run_id = uuid4()
        result = await repo.set_optuna_info(
            run_id=run_id,
            optuna_study_name="study_123",
            optuna_trial_number=5,
            is_best_trial=True,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_create_run_with_hpo_includes_optuna_info(self, repo, mock_client, sample_run_data):
        """Test that create_run_with_hpo includes Optuna information."""
        # Add optuna fields to the mock data
        hpo_run_data = {
            **sample_run_data,
            "optuna_study_name": "study_123",
            "optuna_trial_number": 5,
            "is_best_trial": True,
        }
        mock_result = MagicMock()
        mock_result.data = [hpo_run_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        experiment_id = UUID(sample_run_data["experiment_id"])
        result = await repo.create_run_with_hpo(
            experiment_id=experiment_id,
            run_name="test_run",
            mlflow_run_id="run-123",
            algorithm="RandomForest",
            hyperparameters={"n_estimators": 100},
            training_samples=1000,
            feature_names=["feature1", "feature2"],
            optuna_study_name="study_123",
            optuna_trial_number=5,
            is_best_trial=True,
        )

        assert result is not None
        assert result.optuna_study_name == "study_123"
        assert result.optuna_trial_number == 5
        assert result.is_best_trial is True


@pytest.mark.unit
class TestMLModelRegistryRepository:
    """Tests for MLModelRegistryRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return MLModelRegistryRepository(supabase_client=mock_client)

    @pytest.fixture
    def sample_model_data(self):
        """Sample model registry data."""
        return {
            "id": str(uuid4()),
            "experiment_id": str(uuid4()),
            "model_name": "churn_predictor",
            "model_version": "v1.0.0",
            "mlflow_run_id": "run-123",
            "mlflow_model_uri": "models:/churn_predictor/v1.0.0",
            "algorithm": "RandomForest",
            "hyperparameters": {"n_estimators": 100},
            "auc": 0.85,
            "stage": "production",
            "is_champion": True,
        }

    @pytest.mark.asyncio
    async def test_get_by_name_version_returns_model(self, repo, mock_client, sample_model_data):
        """Test that get_by_name_version returns model when found."""
        mock_result = MagicMock()
        mock_result.data = [sample_model_data]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_eq_version = MagicMock()
        mock_eq_version.limit.return_value = mock_limit
        mock_eq_name = MagicMock()
        mock_eq_name.eq.return_value = mock_eq_version
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_name

        result = await repo.get_by_name_version("churn_predictor", "v1.0.0")

        assert result is not None
        assert result.model_name == "churn_predictor"
        assert result.model_version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_get_champion_model_returns_champion(self, repo, mock_client, sample_model_data):
        """Test that get_champion_model returns champion model."""
        mock_result = MagicMock()
        mock_result.data = [sample_model_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_champion_model()

        assert result is not None
        assert result.is_champion is True

    @pytest.mark.asyncio
    async def test_register_model_inserts_and_returns(self, repo, mock_client, sample_model_data):
        """Test that register_model inserts and returns model."""
        mock_result = MagicMock()
        mock_result.data = [sample_model_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        experiment_id = UUID(sample_model_data["experiment_id"])
        result = await repo.register_model(
            experiment_id=experiment_id,
            model_name="churn_predictor",
            model_version="v1.0.0",
            mlflow_run_id="run-123",
            mlflow_model_uri="models:/churn_predictor/v1.0.0",
            algorithm="RandomForest",
            hyperparameters={"n_estimators": 100},
            metrics={"auc": 0.85},
        )

        assert result is not None
        assert result.model_name == "churn_predictor"
        assert result.auc == 0.85

    @pytest.mark.asyncio
    async def test_transition_stage_updates_stage(self, repo, mock_client, sample_model_data):
        """Test that transition_stage updates model stage."""
        # Mock get_by_id to return a model
        mock_model = MLModelRegistry.from_dict(sample_model_data)

        with patch.object(repo, 'get_by_id', new=AsyncMock(return_value=mock_model)):
            mock_result = MagicMock()
            mock_result.data = [sample_model_data]
            mock_execute = AsyncMock(return_value=mock_result)
            mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

            # Mock the archive query
            mock_archive_result = MagicMock()
            mock_archive_result.data = []
            mock_archive_execute = AsyncMock(return_value=mock_archive_result)
            mock_client.table.return_value.update.return_value.eq.return_value.neq.return_value.execute = mock_archive_execute

            model_id = UUID(sample_model_data["id"])
            result = await repo.transition_stage(
                model_id=model_id,
                new_stage="production",
                archive_existing=True,
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_get_models_by_stage_returns_models(self, repo, mock_client, sample_model_data):
        """Test that get_models_by_stage returns models in stage."""
        mock_result = MagicMock()
        mock_result.data = [sample_model_data, {**sample_model_data, "id": str(uuid4())}]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_models_by_stage(stage="production")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_register_model_candidate_creates_candidate(self, repo, mock_client):
        """Test that register_model_candidate creates a candidate model."""
        mock_data = {
            "id": str(uuid4()),
            "experiment_id": str(uuid4()),
            "model_name": "RandomForestClassifier",
            "model_version": "candidate-20250130120000",
            "algorithm": "RandomForestClassifier",
            "stage": "candidate",
        }
        mock_result = MagicMock()
        mock_result.data = [mock_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        result = await repo.register_model_candidate(
            experiment_id=mock_data["experiment_id"],
            model_name="RandomForestClassifier",
            model_type="ensemble",
            model_class="sklearn.ensemble.RandomForestClassifier",
            hyperparameters={"n_estimators": 100},
            hyperparameter_search_space={"n_estimators": [50, 100, 200]},
            selection_score=0.95,
            selection_rationale="Best cross-validation score",
        )

        assert result is not None
        assert result.model_name == "RandomForestClassifier"
        assert result.stage == "candidate"

    @pytest.mark.asyncio
    async def test_register_model_candidate_returns_none_on_error(self, repo, mock_client):
        """Test that register_model_candidate returns None on error."""
        mock_client.table.return_value.insert.return_value.execute = AsyncMock(
            side_effect=Exception("Database error")
        )

        result = await repo.register_model_candidate(
            experiment_id=str(uuid4()),
            model_name="RandomForestClassifier",
            model_type="ensemble",
            model_class="sklearn.ensemble.RandomForestClassifier",
            hyperparameters={},
            hyperparameter_search_space={},
            selection_score=0.95,
            selection_rationale="Test",
        )

        assert result is None


@pytest.mark.unit
class TestEnums:
    """Tests for enum definitions."""

    def test_model_stage_enum_values(self):
        """Test that ModelStage has correct values."""
        assert ModelStage.DEVELOPMENT == "development"
        assert ModelStage.STAGING == "staging"
        assert ModelStage.PRODUCTION == "production"
        assert ModelStage.ARCHIVED == "archived"

    def test_training_status_enum_values(self):
        """Test that TrainingStatus has correct values."""
        assert TrainingStatus.RUNNING == "running"
        assert TrainingStatus.SCHEDULED == "scheduled"
        assert TrainingStatus.FINISHED == "finished"
        assert TrainingStatus.FAILED == "failed"
        assert TrainingStatus.KILLED == "killed"


@pytest.mark.unit
class TestRepositoryTableNames:
    """Tests for repository table names."""

    def test_experiment_repository_table_name(self, ):
        """Test that MLExperimentRepository has correct table name."""
        repo = MLExperimentRepository(supabase_client=None)
        assert repo.table_name == "ml_experiments"

    def test_training_run_repository_table_name(self):
        """Test that MLTrainingRunRepository has correct table name."""
        repo = MLTrainingRunRepository(supabase_client=None)
        assert repo.table_name == "ml_training_runs"

    def test_model_registry_repository_table_name(self):
        """Test that MLModelRegistryRepository has correct table name."""
        repo = MLModelRegistryRepository(supabase_client=None)
        assert repo.table_name == "ml_model_registry"
