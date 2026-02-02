"""
ML Experiment and Training Run repositories.

Provides database access for MLflow experiment tracking data,
enabling synchronization between MLflow and Supabase tables.

Tables:
- ml_experiments: MLflow experiment metadata
- ml_training_runs: Training job records
- ml_model_registry: Model versions and stages
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from .base import BaseRepository

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class ModelStage(str, Enum):
    """Model registry stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    SHADOW = "shadow"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class TrainingStatus(str, Enum):
    """Training run status."""

    RUNNING = "running"
    SCHEDULED = "scheduled"
    FINISHED = "finished"
    FAILED = "failed"
    KILLED = "killed"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class MLExperiment:
    """ML Experiment entity."""

    id: Optional[UUID] = None
    experiment_name: str = ""
    mlflow_experiment_id: Optional[str] = None
    description: Optional[str] = None

    # Scope definition
    prediction_target: str = ""
    target_population: Optional[str] = None
    observation_window_days: Optional[int] = None
    prediction_horizon_days: Optional[int] = None

    # Success criteria
    minimum_auc: Optional[float] = None
    minimum_precision_at_k: Optional[float] = None
    maximum_fpr: Optional[float] = None

    # Metadata
    brand: Optional[str] = None
    region: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    data_split: str = "unassigned"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": str(self.id) if self.id else None,
            "experiment_name": self.experiment_name,
            "mlflow_experiment_id": self.mlflow_experiment_id,
            "description": self.description,
            "prediction_target": self.prediction_target,
            "target_population": self.target_population,
            "observation_window_days": self.observation_window_days,
            "prediction_horizon_days": self.prediction_horizon_days,
            "minimum_auc": self.minimum_auc,
            "minimum_precision_at_k": self.minimum_precision_at_k,
            "maximum_fpr": self.maximum_fpr,
            "brand": self.brand,
            "region": self.region,
            "created_by": self.created_by,
            "data_split": self.data_split,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLExperiment":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else None,
            experiment_name=data.get("experiment_name", ""),
            mlflow_experiment_id=data.get("mlflow_experiment_id"),
            description=data.get("description"),
            prediction_target=data.get("prediction_target", ""),
            target_population=data.get("target_population"),
            observation_window_days=data.get("observation_window_days"),
            prediction_horizon_days=data.get("prediction_horizon_days"),
            minimum_auc=data.get("minimum_auc"),
            minimum_precision_at_k=data.get("minimum_precision_at_k"),
            maximum_fpr=data.get("maximum_fpr"),
            brand=data.get("brand"),
            region=data.get("region"),
            created_by=data.get("created_by"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            data_split=data.get("data_split", "unassigned"),
        )


@dataclass
class MLTrainingRun:
    """ML Training Run entity."""

    id: Optional[UUID] = None
    experiment_id: Optional[UUID] = None
    model_registry_id: Optional[UUID] = None

    # Run identification
    run_name: Optional[str] = None
    mlflow_run_id: Optional[str] = None

    # Configuration
    algorithm: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Data info
    training_samples: int = 0
    validation_samples: Optional[int] = None
    test_samples: Optional[int] = None
    feature_names: List[str] = field(default_factory=list)

    # Metrics per split
    train_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)

    # Run status
    status: str = "running"
    error_message: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

    # Resources
    compute_type: Optional[str] = None
    gpu_used: bool = False
    data_split: str = "train"

    # Optuna specific
    optuna_study_name: Optional[str] = None
    optuna_trial_number: Optional[int] = None
    is_best_trial: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": str(self.id) if self.id else None,
            "experiment_id": str(self.experiment_id) if self.experiment_id else None,
            "model_registry_id": str(self.model_registry_id) if self.model_registry_id else None,
            "run_name": self.run_name,
            "mlflow_run_id": self.mlflow_run_id,
            "algorithm": self.algorithm,
            "hyperparameters": self.hyperparameters,
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
            "feature_names": self.feature_names,
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "status": self.status,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "compute_type": self.compute_type,
            "gpu_used": self.gpu_used,
            "data_split": self.data_split,
            "optuna_study_name": self.optuna_study_name,
            "optuna_trial_number": self.optuna_trial_number,
            "is_best_trial": self.is_best_trial,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLTrainingRun":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else None,
            experiment_id=UUID(data["experiment_id"]) if data.get("experiment_id") else None,
            model_registry_id=UUID(data["model_registry_id"])
            if data.get("model_registry_id")
            else None,
            run_name=data.get("run_name"),
            mlflow_run_id=data.get("mlflow_run_id"),
            algorithm=data.get("algorithm", ""),
            hyperparameters=data.get("hyperparameters", {}),
            training_samples=data.get("training_samples", 0),
            validation_samples=data.get("validation_samples"),
            test_samples=data.get("test_samples"),
            feature_names=data.get("feature_names", []),
            train_metrics=data.get("train_metrics", {}),
            validation_metrics=data.get("validation_metrics", {}),
            test_metrics=data.get("test_metrics", {}),
            status=data.get("status", "running"),
            error_message=data.get("error_message"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            duration_seconds=data.get("duration_seconds"),
            compute_type=data.get("compute_type"),
            gpu_used=data.get("gpu_used", False),
            data_split=data.get("data_split", "train"),
            optuna_study_name=data.get("optuna_study_name"),
            optuna_trial_number=data.get("optuna_trial_number"),
            is_best_trial=data.get("is_best_trial", False),
        )


@dataclass
class MLModelRegistry:
    """ML Model Registry entity."""

    id: Optional[UUID] = None
    experiment_id: Optional[UUID] = None

    # Model identification
    model_name: str = ""
    model_version: str = ""
    mlflow_run_id: Optional[str] = None
    mlflow_model_uri: Optional[str] = None

    # Model metadata
    algorithm: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_count: Optional[int] = None
    training_samples: Optional[int] = None

    # Performance metrics
    auc: Optional[float] = None
    pr_auc: Optional[float] = None
    brier_score: Optional[float] = None
    calibration_slope: Optional[float] = None

    # Fairness metrics
    fairness_metrics: Dict[str, Any] = field(default_factory=dict)

    # Registry status
    stage: str = "development"
    is_champion: bool = False

    # Artifacts
    artifact_path: Optional[str] = None
    preprocessing_pipeline_path: Optional[str] = None

    # Timestamps
    trained_at: Optional[datetime] = None
    registered_at: Optional[datetime] = None
    promoted_at: Optional[datetime] = None
    data_split: str = "train"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": str(self.id) if self.id else None,
            "experiment_id": str(self.experiment_id) if self.experiment_id else None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "mlflow_run_id": self.mlflow_run_id,
            "mlflow_model_uri": self.mlflow_model_uri,
            "algorithm": self.algorithm,
            "hyperparameters": self.hyperparameters,
            "feature_count": self.feature_count,
            "training_samples": self.training_samples,
            "auc": self.auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "calibration_slope": self.calibration_slope,
            "fairness_metrics": self.fairness_metrics,
            "stage": self.stage,
            "is_champion": self.is_champion,
            "artifact_path": self.artifact_path,
            "preprocessing_pipeline_path": self.preprocessing_pipeline_path,
            "data_split": self.data_split,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLModelRegistry":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else None,
            experiment_id=UUID(data["experiment_id"]) if data.get("experiment_id") else None,
            model_name=data.get("model_name", ""),
            model_version=data.get("model_version", ""),
            mlflow_run_id=data.get("mlflow_run_id"),
            mlflow_model_uri=data.get("mlflow_model_uri"),
            algorithm=data.get("algorithm", ""),
            hyperparameters=data.get("hyperparameters", {}),
            feature_count=data.get("feature_count"),
            training_samples=data.get("training_samples"),
            auc=data.get("auc"),
            pr_auc=data.get("pr_auc"),
            brier_score=data.get("brier_score"),
            calibration_slope=data.get("calibration_slope"),
            fairness_metrics=data.get("fairness_metrics", {}),
            stage=data.get("stage", "development"),
            is_champion=data.get("is_champion", False),
            artifact_path=data.get("artifact_path"),
            preprocessing_pipeline_path=data.get("preprocessing_pipeline_path"),
            trained_at=data.get("trained_at"),
            registered_at=data.get("registered_at"),
            promoted_at=data.get("promoted_at"),
            data_split=data.get("data_split", "train"),
        )


# ============================================================================
# REPOSITORIES
# ============================================================================


class MLExperimentRepository(BaseRepository[MLExperiment]):
    """Repository for ML Experiments."""

    table_name = "ml_experiments"
    model_class = MLExperiment

    def _to_model(self, data: Dict[str, Any]) -> MLExperiment:
        """Convert database row to model."""
        return MLExperiment.from_dict(data)

    async def get_by_name(self, name: str) -> Optional[MLExperiment]:
        """Get experiment by name.

        Args:
            name: Experiment name

        Returns:
            MLExperiment or None
        """
        if not self.client:
            return None

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("experiment_name", name)
            .limit(1)
            .execute()
        )
        return self._to_model(result.data[0]) if result.data else None

    async def get_by_mlflow_id(self, mlflow_id: str) -> Optional[MLExperiment]:
        """Get experiment by MLflow ID.

        Args:
            mlflow_id: MLflow experiment ID

        Returns:
            MLExperiment or None
        """
        if not self.client:
            return None

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("mlflow_experiment_id", mlflow_id)
            .limit(1)
            .execute()
        )
        return self._to_model(result.data[0]) if result.data else None

    async def create_experiment(
        self,
        name: str,
        mlflow_experiment_id: str,
        prediction_target: str,
        description: Optional[str] = None,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        created_by: Optional[str] = None,
        success_criteria: Optional[Dict[str, float]] = None,
    ) -> MLExperiment:
        """Create a new experiment.

        Args:
            name: Experiment name
            mlflow_experiment_id: MLflow experiment ID
            prediction_target: Target variable/outcome
            description: Experiment description
            brand: Brand filter
            region: Region filter
            created_by: Creator username
            success_criteria: Dict with minimum_auc, etc.

        Returns:
            Created MLExperiment
        """
        experiment = MLExperiment(
            id=uuid4(),
            experiment_name=name,
            mlflow_experiment_id=mlflow_experiment_id,
            description=description,
            prediction_target=prediction_target,
            brand=brand,
            region=region,
            created_by=created_by,
            minimum_auc=success_criteria.get("minimum_auc") if success_criteria else None,
            minimum_precision_at_k=success_criteria.get("minimum_precision_at_k")
            if success_criteria
            else None,
            maximum_fpr=success_criteria.get("maximum_fpr") if success_criteria else None,
        )

        if self.client:
            data = experiment.to_dict()
            data.pop("id", None)  # Let DB generate ID
            result = await self.client.table(self.table_name).insert(data).execute()
            if result.data:
                return self._to_model(result.data[0])

        return experiment

    async def list_experiments(
        self,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        limit: int = 100,
    ) -> List[MLExperiment]:
        """List experiments with optional filters.

        Args:
            brand: Filter by brand
            region: Filter by region
            limit: Maximum number of results

        Returns:
            List of MLExperiment
        """
        filters = {}
        if brand:
            filters["brand"] = brand
        if region:
            filters["region"] = region

        return await self.get_many(filters=filters, limit=limit)


class MLTrainingRunRepository(BaseRepository[MLTrainingRun]):
    """Repository for ML Training Runs."""

    table_name = "ml_training_runs"
    model_class = MLTrainingRun

    def _to_model(self, data: Dict[str, Any]) -> MLTrainingRun:
        """Convert database row to model."""
        return MLTrainingRun.from_dict(data)

    async def get_by_mlflow_run_id(self, mlflow_run_id: str) -> Optional[MLTrainingRun]:
        """Get training run by MLflow run ID.

        Args:
            mlflow_run_id: MLflow run ID

        Returns:
            MLTrainingRun or None
        """
        if not self.client:
            return None

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("mlflow_run_id", mlflow_run_id)
            .limit(1)
            .execute()
        )
        return self._to_model(result.data[0]) if result.data else None

    async def create_run(
        self,
        experiment_id: UUID,
        run_name: str,
        mlflow_run_id: str,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        training_samples: int,
        feature_names: List[str],
    ) -> MLTrainingRun:
        """Create a new training run record.

        Args:
            experiment_id: Parent experiment ID
            run_name: Run name
            mlflow_run_id: MLflow run ID
            algorithm: Algorithm name
            hyperparameters: Hyperparameter dict
            training_samples: Number of training samples
            feature_names: List of feature names

        Returns:
            Created MLTrainingRun
        """
        run = MLTrainingRun(
            id=uuid4(),
            experiment_id=experiment_id,
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            training_samples=training_samples,
            feature_names=feature_names,
            started_at=datetime.now(timezone.utc),
            status="running",
        )

        if self.client:
            data = run.to_dict()
            data.pop("id", None)
            result = await self.client.table(self.table_name).insert(data).execute()
            if result.data:
                return self._to_model(result.data[0])

        return run

    async def update_run_metrics(
        self,
        run_id: UUID,
        train_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
        test_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Update run metrics.

        Args:
            run_id: Training run ID
            train_metrics: Training metrics
            validation_metrics: Validation metrics
            test_metrics: Test metrics

        Returns:
            True if successful
        """
        if not self.client:
            return False

        updates = {}
        if train_metrics:
            updates["train_metrics"] = train_metrics
        if validation_metrics:
            updates["validation_metrics"] = validation_metrics
        if test_metrics:
            updates["test_metrics"] = test_metrics

        if updates:
            await self.client.table(self.table_name).update(updates).eq("id", str(run_id)).execute()
            return True

        return False

    async def complete_run(
        self,
        run_id: UUID,
        status: str = "finished",
        error_message: Optional[str] = None,
    ) -> bool:
        """Mark a run as completed.

        Args:
            run_id: Training run ID
            status: Final status (finished, failed, killed)
            error_message: Error message if failed

        Returns:
            True if successful
        """
        if not self.client:
            return False

        now = datetime.now(timezone.utc)

        # Get current run to calculate duration
        current = await self.get_by_id(str(run_id))
        duration = None
        if current and current.started_at:
            duration = int((now - current.started_at).total_seconds())

        updates = {
            "status": status,
            "completed_at": now.isoformat(),
            "duration_seconds": duration,
        }
        if error_message:
            updates["error_message"] = error_message

        await self.client.table(self.table_name).update(updates).eq("id", str(run_id)).execute()
        return True

    async def get_runs_for_experiment(
        self,
        experiment_id: UUID,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[MLTrainingRun]:
        """Get all runs for an experiment.

        Args:
            experiment_id: Experiment ID
            status: Optional status filter
            limit: Maximum number of results

        Returns:
            List of MLTrainingRun
        """
        filters = {"experiment_id": str(experiment_id)}
        if status:
            filters["status"] = status

        return await self.get_many(filters=filters, limit=limit)

    async def get_best_run(
        self, experiment_id: UUID, metric: str = "auc"
    ) -> Optional[MLTrainingRun]:
        """Get the best run for an experiment based on a metric.

        Args:
            experiment_id: Experiment ID
            metric: Metric to optimize (from test_metrics)

        Returns:
            Best MLTrainingRun or None
        """
        runs = await self.get_runs_for_experiment(experiment_id, status="finished")
        if not runs:
            return None

        # Sort by metric (higher is better for AUC-like metrics)
        best = None
        best_value = -float("inf")
        for run in runs:
            value = run.test_metrics.get(metric, 0)
            if value > best_value:
                best_value = value
                best = run

        return best

    async def set_optuna_info(
        self,
        run_id: UUID,
        optuna_study_name: str,
        optuna_trial_number: Optional[int] = None,
        is_best_trial: bool = False,
    ) -> bool:
        """Set Optuna HPO information for a training run.

        This links the training run to its HPO study for traceability.

        Args:
            run_id: Training run ID
            optuna_study_name: Name of the Optuna study
            optuna_trial_number: Trial number if from HPO
            is_best_trial: Whether this was the best trial

        Returns:
            True if successful
        """
        if not self.client:
            return False

        updates = {
            "optuna_study_name": optuna_study_name,
        }
        if optuna_trial_number is not None:
            updates["optuna_trial_number"] = optuna_trial_number
        updates["is_best_trial"] = is_best_trial

        await self.client.table(self.table_name).update(updates).eq("id", str(run_id)).execute()
        return True

    async def create_run_with_hpo(
        self,
        experiment_id: UUID,
        run_name: str,
        mlflow_run_id: str,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        training_samples: int,
        feature_names: List[str],
        optuna_study_name: Optional[str] = None,
        optuna_trial_number: Optional[int] = None,
        is_best_trial: bool = False,
        validation_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
    ) -> MLTrainingRun:
        """Create a new training run record with HPO information.

        This is the preferred method when creating runs from model_trainer
        as it includes all HPO linkage information.

        Args:
            experiment_id: Parent experiment ID
            run_name: Run name
            mlflow_run_id: MLflow run ID
            algorithm: Algorithm name
            hyperparameters: Hyperparameter dict
            training_samples: Number of training samples
            feature_names: List of feature names
            optuna_study_name: Name of HPO study (if HPO was used)
            optuna_trial_number: Best trial number (if HPO was used)
            is_best_trial: Whether this represents the best HPO trial
            validation_samples: Number of validation samples
            test_samples: Number of test samples

        Returns:
            Created MLTrainingRun
        """
        run = MLTrainingRun(
            id=uuid4(),
            experiment_id=experiment_id,
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            training_samples=training_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            feature_names=feature_names,
            started_at=datetime.now(timezone.utc),
            status="running",
            optuna_study_name=optuna_study_name,
            optuna_trial_number=optuna_trial_number,
            is_best_trial=is_best_trial,
        )

        if self.client:
            data = run.to_dict()
            data.pop("id", None)
            result = await self.client.table(self.table_name).insert(data).execute()
            if result.data:
                return self._to_model(result.data[0])

        return run


class MLModelRegistryRepository(BaseRepository[MLModelRegistry]):
    """Repository for ML Model Registry."""

    table_name = "ml_model_registry"
    model_class = MLModelRegistry

    def _to_model(self, data: Dict[str, Any]) -> MLModelRegistry:
        """Convert database row to model."""
        return MLModelRegistry.from_dict(data)

    async def get_by_name_version(self, model_name: str, version: str) -> Optional[MLModelRegistry]:
        """Get model by name and version.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            MLModelRegistry or None
        """
        if not self.client:
            return None

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("model_name", model_name)
            .eq("model_version", version)
            .limit(1)
            .execute()
        )
        return self._to_model(result.data[0]) if result.data else None

    async def get_champion_model(
        self, experiment_id: Optional[UUID] = None
    ) -> Optional[MLModelRegistry]:
        """Get the current champion model.

        Args:
            experiment_id: Optional experiment filter

        Returns:
            Champion MLModelRegistry or None
        """
        if not self.client:
            return None

        query = self.client.table(self.table_name).select("*").eq("is_champion", True)

        if experiment_id:
            query = query.eq("experiment_id", str(experiment_id))

        result = await query.limit(1).execute()
        return self._to_model(result.data[0]) if result.data else None

    async def register_model(
        self,
        experiment_id: UUID,
        model_name: str,
        model_version: str,
        mlflow_run_id: str,
        mlflow_model_uri: str,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> MLModelRegistry:
        """Register a new model version.

        Args:
            experiment_id: Parent experiment ID
            model_name: Model name
            model_version: Version string
            mlflow_run_id: MLflow run ID
            mlflow_model_uri: MLflow model URI
            algorithm: Algorithm name
            hyperparameters: Hyperparameter dict
            metrics: Performance metrics

        Returns:
            Created MLModelRegistry
        """
        model = MLModelRegistry(
            id=uuid4(),
            experiment_id=experiment_id,
            model_name=model_name,
            model_version=model_version,
            mlflow_run_id=mlflow_run_id,
            mlflow_model_uri=mlflow_model_uri,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            auc=metrics.get("auc"),
            pr_auc=metrics.get("pr_auc"),
            brier_score=metrics.get("brier_score"),
            calibration_slope=metrics.get("calibration_slope"),
            stage="development",
            registered_at=datetime.now(timezone.utc),
        )

        if self.client:
            data = model.to_dict()
            data.pop("id", None)
            result = await self.client.table(self.table_name).insert(data).execute()
            if result.data:
                return self._to_model(result.data[0])

        return model

    async def transition_stage(
        self,
        model_id: UUID,
        new_stage: str,
        archive_existing: bool = True,
    ) -> bool:
        """Transition model to new stage.

        Args:
            model_id: Model registry ID
            new_stage: Target stage
            archive_existing: Archive existing models in target stage

        Returns:
            True if successful
        """
        if not self.client:
            return False

        # Get current model
        current = await self.get_by_id(str(model_id))
        if not current:
            return False

        # Archive existing models in production if needed
        if archive_existing and new_stage == "production":
            await (
                self.client.table(self.table_name)
                .update({"stage": "archived", "is_champion": False})
                .eq("stage", "production")
                .neq("id", str(model_id))
                .execute()
            )

        # Update stage
        updates = {
            "stage": new_stage,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        if new_stage == "production":
            updates["is_champion"] = True

        await self.client.table(self.table_name).update(updates).eq("id", str(model_id)).execute()

        return True

    async def get_models_by_stage(
        self,
        stage: str,
        experiment_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> List[MLModelRegistry]:
        """Get models by stage.

        Args:
            stage: Model stage
            experiment_id: Optional experiment filter
            limit: Maximum number of results

        Returns:
            List of MLModelRegistry
        """
        filters = {"stage": stage}
        if experiment_id:
            filters["experiment_id"] = str(experiment_id)

        return await self.get_many(filters=filters, limit=limit)

    async def register_model_candidate(
        self,
        experiment_id: str,
        model_name: str,
        model_type: str,
        model_class: str,
        hyperparameters: Dict[str, Any],
        hyperparameter_search_space: Dict[str, Any],
        selection_score: float,
        selection_rationale: str,
        stage: str = "candidate",
        created_by: str = "model_selector",
        mlflow_run_id: Optional[str] = None,
        mlflow_experiment_id: Optional[str] = None,
    ) -> Optional[MLModelRegistry]:
        """Register a model candidate from model_selector.

        This is called during model selection to store the selected
        algorithm configuration before training begins.

        Args:
            experiment_id: Parent experiment ID
            model_name: Algorithm name (e.g., "RandomForestClassifier")
            model_type: Algorithm family (e.g., "ensemble")
            model_class: Full class path (e.g., "sklearn.ensemble.RandomForestClassifier")
            hyperparameters: Default hyperparameters
            hyperparameter_search_space: HPO search space
            selection_score: Selection score from model_selector
            selection_rationale: Why this model was selected
            stage: Model stage (default: "candidate")
            created_by: Agent that created this entry
            mlflow_run_id: MLflow run ID for audit trail
            mlflow_experiment_id: MLflow experiment ID

        Returns:
            Created MLModelRegistry or None if creation failed
        """
        if not self.client:
            return None

        from uuid import uuid4

        model_id = uuid4()

        # Generate version from timestamp for candidates
        from datetime import datetime, timezone

        version = f"candidate-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        data = {
            "id": str(model_id),
            "experiment_id": experiment_id,
            "model_name": model_name,
            "model_version": version,
            "algorithm": model_name,  # For compatibility with existing schema
            "hyperparameters": hyperparameters,
            "metrics": {"selection_score": selection_score},
            "stage": stage,
            "is_champion": False,
            "description": selection_rationale,
            "mlflow_run_id": mlflow_run_id,
            "mlflow_model_uri": None,  # Not set until model is trained
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": created_by,
            # Store additional selection metadata
            "tags": {
                "model_type": model_type,
                "model_class": model_class,
                "hyperparameter_search_space": hyperparameter_search_space,
                "mlflow_experiment_id": mlflow_experiment_id,
            },
        }

        try:
            result = await self.client.table(self.table_name).insert(data).execute()
            if result.data:
                return self._to_model(result.data[0])
            return None
        except Exception:
            return None
