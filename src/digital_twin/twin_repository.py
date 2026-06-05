"""
Twin Repository
===============

Persistence layer for digital twin models, simulations, and fidelity tracking.
Integrates with:
    - Supabase (PostgreSQL) for structured data
    - MLflow for model artifacts and versioning
    - Redis for caching active models
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.repositories.base import BaseRepository

from .models.simulation_models import (
    FidelityGrade,
    FidelityRecord,
    SimulationResult,
    SimulationStatus,
)
from .models.twin_models import (
    TwinModelConfig,
    TwinModelMetrics,
    TwinType,
)

logger = logging.getLogger(__name__)


class TwinModelRepository(BaseRepository):
    """
    Repository for digital_twin_models table.

    Handles storage and retrieval of trained twin generator models.
    """

    table_name = "digital_twin_models"
    model_class = None  # Using raw dicts

    def __init__(self, supabase_client=None, mlflow_client=None, redis_client=None):
        """
        Initialize repository.

        Args:
            supabase_client: Supabase client for database operations
            mlflow_client: MLflow client for model registry
            redis_client: Redis client for caching
        """
        super().__init__(supabase_client)
        self.mlflow_client = mlflow_client
        self.redis_client = redis_client
        logger.info("Initialized TwinModelRepository")

    async def save_model(
        self,
        config: TwinModelConfig,
        metrics: TwinModelMetrics,
        model_artifact: Any = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_model_uri: Optional[str] = None,
    ) -> UUID:
        """
        Save a trained twin model's metadata row.

        MLflow artifact persistence is owned by
        :mod:`src.digital_twin.twin_persistence` (``save_twin_artifacts``), which
        the caller invokes BEFORE this method and whose real ``run_id`` /
        ``model_uri`` are passed in here. We store exactly what we are given —
        never a fabricated reference (#705 H4). The old internal stub returned a
        plausible-but-fake ``models:/twin_<type>_<brand>/latest`` URI that pointed
        nowhere; it has been removed.

        Args:
            config: Model configuration
            metrics: Training metrics
            model_artifact: Retained for backward compatibility; persistence is
                done by the caller via ``save_twin_artifacts`` (not used here).
            mlflow_run_id: Real MLflow run ID produced by ``save_twin_artifacts``.
            mlflow_model_uri: Real MLflow model URI produced by
                ``save_twin_artifacts`` (loadable via ``mlflow.sklearn.load_model``).

        Returns:
            UUID of saved model
        """
        model_id = metrics.model_id

        # Prepare database record
        row = {
            "model_id": str(model_id),
            "model_name": config.model_name,
            "model_description": config.model_description,
            "twin_type": config.twin_type.value,
            "model_version": "1.0",
            "mlflow_run_id": mlflow_run_id,
            "mlflow_model_uri": mlflow_model_uri,
            "training_config": {
                "algorithm": config.algorithm,
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "training_samples": config.training_samples,
                "validation_split": config.validation_split,
                "cv_folds": config.cv_folds,
            },
            "feature_columns": config.feature_columns,
            "target_columns": [config.target_column],
            "performance_metrics": metrics.to_dict(),
            "brand": config.brand.value,
            "geographic_scope": config.geographic_scope,
            "is_active": True,
            "activation_date": datetime.now(timezone.utc).isoformat(),
        }

        # Insert into database
        if self.client:
            try:
                await self.client.table(self.table_name).insert(row).execute()
                logger.info(f"Saved twin model {model_id} to database")
            except Exception as e:
                logger.error(f"Failed to save twin model {model_id}: {e}")
                raise

        # Cache active model info
        if self.redis_client:
            self._cache_model_info(model_id, config, metrics)

        return model_id

    async def get_model(self, model_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get model by ID.

        Args:
            model_id: Model UUID

        Returns:
            Model info dict or None
        """
        # Try cache first
        if self.redis_client:
            cached = self._get_cached_model(model_id)
            if cached:
                logger.debug(f"Cache hit for model {model_id}")
                return cached

        # Query database
        if not self.client:
            return None

        try:
            result = await (
                self.client.table(self.table_name)
                .select("*")
                .eq("model_id", str(model_id))
                .execute()
            )
            if result.data:
                model = result.data[0]
                # Cache for next time
                if self.redis_client:
                    self.redis_client.setex(
                        f"twin_model:{model_id}",
                        3600,
                        json.dumps(model, default=str),
                    )
                return model  # type: ignore[no-any-return]
            return None
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None

    async def list_active_models(
        self,
        twin_type: Optional[TwinType] = None,
        brand: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List active twin models.

        Args:
            twin_type: Optional filter by twin type
            brand: Optional filter by brand
            limit: Maximum records to return

        Returns:
            List of active model info dicts
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("is_active", True)
                .order("fidelity_score", desc=True, nullsfirst=False)
                .order("created_at", desc=True)
                .limit(limit)
            )

            if twin_type:
                query = query.eq("twin_type", twin_type.value)

            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to list active models: {e}")
            return []

    async def deactivate_model(
        self,
        model_id: UUID,
        reason: str,
    ) -> bool:
        """
        Deactivate a twin model.

        Args:
            model_id: Model to deactivate
            reason: Reason for deactivation

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            await (
                self.client.table(self.table_name)
                .update(
                    {
                        "is_active": False,
                        "deactivation_date": datetime.now(timezone.utc).isoformat(),
                        "deactivation_reason": reason,
                    }
                )
                .eq("model_id", str(model_id))
                .execute()
            )

            # Remove from cache
            if self.redis_client:
                self.redis_client.delete(f"twin_model:{model_id}")

            logger.info(f"Deactivated model {model_id}: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to deactivate model {model_id}: {e}")
            return False

    async def update_fidelity_score(
        self,
        model_id: UUID,
        fidelity_score: float,
        sample_count: int,
    ) -> bool:
        """
        Update model fidelity score.

        Args:
            model_id: Model to update
            fidelity_score: New fidelity score (0-1)
            sample_count: Number of validation samples

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            await (
                self.client.table(self.table_name)
                .update(
                    {
                        "fidelity_score": fidelity_score,
                        "fidelity_sample_count": sample_count,
                        "last_fidelity_update": datetime.now(timezone.utc).isoformat(),
                    }
                )
                .eq("model_id", str(model_id))
                .execute()
            )

            # Invalidate cache
            if self.redis_client:
                self.redis_client.delete(f"twin_model:{model_id}")

            logger.info(f"Updated fidelity for model {model_id}: {fidelity_score}")
            return True
        except Exception as e:
            logger.error(f"Failed to update fidelity for model {model_id}: {e}")
            return False

    def _cache_model_info(
        self,
        model_id: UUID,
        config: TwinModelConfig,
        metrics: TwinModelMetrics,
    ) -> None:
        """Cache model info in Redis."""
        if not self.redis_client:
            return

        cache_data = {
            "model_id": str(model_id),
            "twin_type": config.twin_type.value,
            "brand": config.brand.value,
            "fidelity_score": None,
            "r2_score": metrics.r2_score,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        self.redis_client.setex(
            f"twin_model:{model_id}",
            3600,  # 1 hour TTL
            json.dumps(cache_data),
        )

    def _get_cached_model(self, model_id: UUID) -> Optional[Dict[str, Any]]:
        """Get model info from Redis cache."""
        if not self.redis_client:
            return None

        data = self.redis_client.get(f"twin_model:{model_id}")
        if data:
            return json.loads(data)  # type: ignore[no-any-return]

        return None


class SimulationRepository(BaseRepository):
    """
    Repository for twin_simulations table.

    Handles storage and retrieval of simulation runs and results.
    """

    table_name = "twin_simulations"
    model_class = None

    async def save_simulation(self, result: SimulationResult, brand: str) -> UUID:
        """
        Save simulation result.

        Args:
            result: SimulationResult to save
            brand: Brand for the simulation

        Returns:
            Simulation UUID
        """
        if not self.client:
            logger.warning("No database client, simulation not persisted")
            return result.simulation_id

        row = {
            "simulation_id": str(result.simulation_id),
            "model_id": str(result.model_id),
            "intervention_type": result.intervention_config.intervention_type,
            "intervention_config": {
                "channel": result.intervention_config.channel,
                "frequency": result.intervention_config.frequency,
                "duration_weeks": result.intervention_config.duration_weeks,
                "content_type": result.intervention_config.content_type,
                "personalization_level": result.intervention_config.personalization_level,
                "target_segment": result.intervention_config.target_segment,
                "intensity_multiplier": result.intervention_config.intensity_multiplier,
            },
            "target_population": result.intervention_config.target_segment or "hcp",
            "population_filters": result.population_filters.to_dict(),
            "twin_count": result.twin_count,
            "simulated_ate": result.simulated_ate,
            "simulated_ci_lower": result.simulated_ci_lower,
            "simulated_ci_upper": result.simulated_ci_upper,
            "simulated_std_error": result.simulated_std_error,
            "effect_heterogeneity": result.effect_heterogeneity.model_dump(),
            "simulation_status": result.status.value,
            "recommendation": result.recommendation.value,
            "recommendation_rationale": result.recommendation_rationale,
            "recommended_sample_size": result.recommended_sample_size,
            "recommended_duration_weeks": result.recommended_duration_weeks,
            "simulation_confidence": result.simulation_confidence,
            "fidelity_warning": result.fidelity_warning,
            "fidelity_warning_reason": result.fidelity_warning_reason,
            "execution_time_ms": result.execution_time_ms,
            "memory_usage_mb": result.memory_usage_mb,
            "brand": brand,
            "created_at": result.created_at.isoformat() if result.created_at else None,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        }

        try:
            await self.client.table(self.table_name).insert(row).execute()
            logger.info(f"Saved simulation {result.simulation_id}")
            return result.simulation_id
        except Exception as e:
            logger.error(f"Failed to save simulation {result.simulation_id}: {e}")
            raise

    async def get_simulation(self, simulation_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get simulation by ID.

        Args:
            simulation_id: Simulation UUID

        Returns:
            Simulation record or None
        """
        if not self.client:
            return None

        try:
            result = await (
                self.client.table(self.table_name)
                .select("*")
                .eq("simulation_id", str(simulation_id))
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get simulation {simulation_id}: {e}")
            return None

    async def list_simulations(
        self,
        model_id: Optional[UUID] = None,
        brand: Optional[str] = None,
        status: Optional[SimulationStatus] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List simulations with optional filters.

        Args:
            model_id: Optional filter by model
            brand: Optional filter by brand
            status: Optional filter by status
            limit: Maximum records to return

        Returns:
            List of simulation records
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
            )

            if model_id:
                query = query.eq("model_id", str(model_id))
            if brand:
                query = query.eq("brand", brand)
            if status:
                query = query.eq("simulation_status", status.value)

            result = await query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to list simulations: {e}")
            return []

    async def update_status(
        self,
        simulation_id: UUID,
        status: SimulationStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update simulation status.

        Args:
            simulation_id: Simulation to update
            status: New status
            error_message: Optional error message for failed status

        Returns:
            True if successful
        """
        if not self.client:
            return False

        updates = {
            "simulation_status": status.value,
        }
        if status == SimulationStatus.COMPLETED:
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()
        if status == SimulationStatus.RUNNING:
            updates["started_at"] = datetime.now(timezone.utc).isoformat()
        if error_message:
            updates["error_message"] = error_message

        try:
            await (
                self.client.table(self.table_name)
                .update(updates)
                .eq("simulation_id", str(simulation_id))
                .execute()
            )
            logger.info(f"Updated simulation {simulation_id} status to {status.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to update simulation status: {e}")
            return False

    async def link_experiment(
        self,
        simulation_id: UUID,
        experiment_design_id: UUID,
    ) -> bool:
        """
        Link simulation to an experiment design.

        Args:
            simulation_id: Simulation to update
            experiment_design_id: Experiment design to link

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            await (
                self.client.table(self.table_name)
                .update({"experiment_design_id": str(experiment_design_id)})
                .eq("simulation_id", str(simulation_id))
                .execute()
            )
            logger.info(f"Linked simulation {simulation_id} to experiment {experiment_design_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to link simulation to experiment: {e}")
            return False


class FidelityRepository(BaseRepository):
    """
    Repository for twin_fidelity_tracking table.

    Handles storage and retrieval of fidelity validation records.
    """

    table_name = "twin_fidelity_tracking"
    model_class = None

    async def save_fidelity_record(self, record: FidelityRecord) -> UUID:
        """
        Save fidelity tracking record.

        Args:
            record: FidelityRecord to save

        Returns:
            Tracking UUID
        """
        if not self.client:
            logger.warning("No database client, fidelity record not persisted")
            return record.tracking_id

        row = {
            "tracking_id": str(record.tracking_id),
            "simulation_id": str(record.simulation_id),
            "simulated_ate": record.simulated_ate,
            "simulated_ci_lower": record.simulated_ci_lower,
            "simulated_ci_upper": record.simulated_ci_upper,
            "actual_ate": record.actual_ate,
            "actual_ci_lower": record.actual_ci_lower,
            "actual_ci_upper": record.actual_ci_upper,
            "actual_sample_size": record.actual_sample_size,
            "actual_experiment_id": str(record.actual_experiment_id)
            if record.actual_experiment_id
            else None,
            "prediction_error": record.prediction_error,
            "absolute_error": record.absolute_error,
            "ci_coverage": record.ci_coverage,
            "fidelity_grade": record.fidelity_grade.value,
            "validation_notes": record.validation_notes,
            "confounding_factors": record.confounding_factors,
            "validated_by": record.validated_by,
            "validated_at": record.validated_at.isoformat() if record.validated_at else None,
        }

        try:
            await self.client.table(self.table_name).insert(row).execute()
            logger.info(f"Saved fidelity record {record.tracking_id}")
            return record.tracking_id
        except Exception as e:
            logger.error(f"Failed to save fidelity record: {e}")
            raise

    async def update_fidelity_validation(
        self,
        tracking_id: UUID,
        actual_ate: float,
        actual_ci_lower: Optional[float] = None,
        actual_ci_upper: Optional[float] = None,
        actual_sample_size: Optional[int] = None,
        actual_experiment_id: Optional[UUID] = None,
        validation_notes: Optional[str] = None,
        validated_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update fidelity record with actual experiment results.

        Note: The database trigger will automatically calculate prediction_error,
        absolute_error, ci_coverage, and fidelity_grade.

        Args:
            tracking_id: Fidelity record to update
            actual_ate: Actual ATE from real experiment
            actual_ci_lower: Actual CI lower bound
            actual_ci_upper: Actual CI upper bound
            actual_sample_size: Sample size of actual experiment
            actual_experiment_id: UUID of actual experiment
            validation_notes: Optional notes
            validated_by: User who validated

        Returns:
            Updated record or None
        """
        if not self.client:
            return None

        updates = {
            "actual_ate": actual_ate,
            "actual_ci_lower": actual_ci_lower,
            "actual_ci_upper": actual_ci_upper,
            "actual_sample_size": actual_sample_size,
            "actual_experiment_id": str(actual_experiment_id) if actual_experiment_id else None,
            "validation_notes": validation_notes,
            "validated_by": validated_by,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            result = await (
                self.client.table(self.table_name)
                .update(updates)
                .eq("tracking_id", str(tracking_id))
                .select()
                .execute()
            )
            if result.data:
                logger.info(f"Updated fidelity validation for {tracking_id}")
                return result.data[0]  # type: ignore[no-any-return]
            return None
        except Exception as e:
            logger.error(f"Failed to update fidelity validation: {e}")
            return None

    async def get_fidelity_by_simulation(
        self,
        simulation_id: UUID,
    ) -> Optional[FidelityRecord]:
        """
        Get fidelity record by simulation ID.

        Args:
            simulation_id: Simulation UUID

        Returns:
            FidelityRecord or None
        """
        if not self.client:
            return None

        try:
            result = await (
                self.client.table(self.table_name)
                .select("*")
                .eq("simulation_id", str(simulation_id))
                .execute()
            )
            if result.data:
                return self._to_fidelity_record(result.data[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get fidelity by simulation {simulation_id}: {e}")
            return None

    async def get_model_fidelity_records(
        self,
        model_id: UUID,
        validated_only: bool = True,
        limit: int = 100,
    ) -> List[FidelityRecord]:
        """
        Get fidelity records for a model.

        Args:
            model_id: Model UUID
            validated_only: Only return validated records
            limit: Maximum records to return

        Returns:
            List of FidelityRecords
        """
        if not self.client:
            return []

        try:
            # Join with simulations to filter by model_id
            query = (
                self.client.table(self.table_name)
                .select("*, twin_simulations!inner(model_id)")
                .eq("twin_simulations.model_id", str(model_id))
                .order("validated_at", desc=True, nullsfirst=False)
                .limit(limit)
            )

            if validated_only:
                query = query.not_.is_("validated_at", "null")

            result = await query.execute()
            return [self._to_fidelity_record(row) for row in result.data or []]
        except Exception as e:
            logger.error(f"Failed to get model fidelity records for {model_id}: {e}")
            return []

    async def get_recent_fidelity_records(
        self,
        brand: Optional[str] = None,
        days: int = 30,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recent fidelity records.

        Args:
            brand: Optional filter by brand
            days: Number of days to look back
            limit: Maximum records

        Returns:
            List of fidelity records
        """
        if not self.client:
            return []

        try:
            datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            # Note: actual filtering by date range would need to be adjusted
            # based on Supabase's date handling

            query = (
                self.client.table(self.table_name)
                .select("*")
                .not_.is_("validated_at", "null")
                .order("validated_at", desc=True)
                .limit(limit)
            )

            result = await query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get recent fidelity records: {e}")
            return []

    def _to_fidelity_record(self, row: Dict[str, Any]) -> FidelityRecord:
        """Convert database row to FidelityRecord."""
        return FidelityRecord(
            tracking_id=UUID(row["tracking_id"]),
            simulation_id=UUID(row["simulation_id"]),
            simulated_ate=row["simulated_ate"],
            simulated_ci_lower=row.get("simulated_ci_lower"),
            simulated_ci_upper=row.get("simulated_ci_upper"),
            actual_ate=row.get("actual_ate"),
            actual_ci_lower=row.get("actual_ci_lower"),
            actual_ci_upper=row.get("actual_ci_upper"),
            actual_sample_size=row.get("actual_sample_size"),
            actual_experiment_id=UUID(row["actual_experiment_id"])
            if row.get("actual_experiment_id")
            else None,
            prediction_error=row.get("prediction_error"),
            absolute_error=row.get("absolute_error"),
            ci_coverage=row.get("ci_coverage"),
            fidelity_grade=FidelityGrade(row["fidelity_grade"]),
            validation_notes=row.get("validation_notes"),
            confounding_factors=row.get("confounding_factors", []),
            validated_by=row.get("validated_by"),
            validated_at=datetime.fromisoformat(row["validated_at"])
            if row.get("validated_at")
            else None,
        )


class TwinRepository:
    """
    Unified repository facade for all digital twin operations.

    Provides a single entry point for twin models, simulations, and fidelity tracking.

    Example:
        >>> repo = TwinRepository(supabase_client)
        >>> await repo.models.save_model(config, metrics)
        >>> await repo.simulations.save_simulation(result, brand="Kisqali")
        >>> await repo.fidelity.save_fidelity_record(record)
    """

    def __init__(
        self,
        supabase_client=None,
        mlflow_client=None,
        redis_client=None,
    ):
        """
        Initialize the unified repository.

        Args:
            supabase_client: Supabase client for database operations
            mlflow_client: MLflow client for model registry
            redis_client: Redis client for caching
        """
        self.models = TwinModelRepository(
            supabase_client=supabase_client,
            mlflow_client=mlflow_client,
            redis_client=redis_client,
        )
        self.simulations = SimulationRepository(supabase_client=supabase_client)
        self.fidelity = FidelityRepository(supabase_client=supabase_client)

        logger.info("Initialized unified TwinRepository")

    # Convenience methods that delegate to sub-repositories

    async def save_model(
        self,
        config: TwinModelConfig,
        metrics: TwinModelMetrics,
        model_artifact: Any = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_model_uri: Optional[str] = None,
    ) -> UUID:
        """Save a trained twin model."""
        return await self.models.save_model(  # type: ignore[no-any-return]
            config,
            metrics,
            model_artifact,
            mlflow_run_id,
            mlflow_model_uri,
        )

    async def get_model(self, model_id: UUID) -> Optional[Dict[str, Any]]:
        """Get model by ID."""
        return await self.models.get_model(model_id)  # type: ignore[no-any-return]

    async def list_active_models(
        self,
        twin_type: Optional[TwinType] = None,
        brand: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List active twin models."""
        return await self.models.list_active_models(  # type: ignore[no-any-return]
            twin_type, brand
        )

    async def save_simulation(self, result: SimulationResult, brand: str) -> UUID:
        """Save simulation result."""
        return await self.simulations.save_simulation(  # type: ignore[no-any-return]
            result, brand
        )

    async def get_simulation(self, simulation_id: UUID) -> Optional[Dict[str, Any]]:
        """Get simulation by ID."""
        return await self.simulations.get_simulation(  # type: ignore[no-any-return]
            simulation_id
        )

    async def save_fidelity_record(self, record: FidelityRecord) -> UUID:
        """Save fidelity tracking record."""
        return await self.fidelity.save_fidelity_record(  # type: ignore[no-any-return]
            record
        )

    async def update_fidelity_validation(
        self,
        tracking_id: UUID,
        actual_ate: float,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Update fidelity record with validation results."""
        return await self.fidelity.update_fidelity_validation(  # type: ignore[no-any-return]
            tracking_id, actual_ate, **kwargs
        )

    async def get_fidelity_by_simulation(self, simulation_id: UUID) -> Optional[FidelityRecord]:
        """Get fidelity tracking record by simulation ID."""
        return await self.fidelity.get_fidelity_by_simulation(  # type: ignore[no-any-return]
            simulation_id
        )


# --------------------------------------------------------------------------- #
# Twin retraining job persistence (#549)
#
# Before #549 a twin retraining job lived ONLY in the in-process
# ``TwinRetrainingService._pending_jobs`` dict, so a Celery worker (a separate
# process from the API that queued the job) had an empty store and
# ``complete_retraining`` returned ``None`` — the completion was never recorded
# and the path could not function end-to-end across the worker boundary.
#
# This is the twin analogue of ``RetrainingHistoryRepository``
# (``src/repositories/drift_monitoring.py``): a durable, Supabase-backed CRUD
# store so a job created in one process is found + updated in another and
# re-read by the first. Table: ``database/ml/029_twin_retraining_jobs.sql``.
# --------------------------------------------------------------------------- #
class TwinRetrainingJobRecord(BaseModel):
    """Row for the ``twin_retraining_jobs`` table (durable twin-retraining job).

    The DB record backing the in-memory ``TwinRetrainingJob`` dataclass
    (``retraining_service.py``). ``id`` is the job id (the service's
    ``job_id``); ``model_id`` is the twin model being retrained. ``fidelity_after``
    is the REAL held-out validation R² written ONLY on a certified completion —
    it stays ``None`` on failure (no fabricated 0.0), preserving the #548
    fail-closed invariant at the persistence layer.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    new_model_id: Optional[str] = None
    trigger_reason: str = "manual"
    status: str = "pending"
    fidelity_before: float = 0.0
    fidelity_after: Optional[float] = None
    training_config: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TwinRetrainingJobRepository(BaseRepository[TwinRetrainingJobRecord]):
    """Durable, shared CRUD store for twin retraining jobs.

    Mirrors ``RetrainingHistoryRepository``. Delegates reads/updates to
    ``BaseRepository`` (keyed on the ``id`` column), which ``await``s
    ``.execute()`` — so this needs the ASYNC Supabase client. When constructed
    with no explicit client it lazily binds the process-shared
    ``get_async_supabase_client()`` singleton on first async use, so a job
    created in the API process is found + updated by a Celery worker and re-read
    by the API (both processes resolve the SAME env-driven client). With no
    client configured it is inert (no-op writes, ``None`` reads) — never an error.
    """

    table_name = "twin_retraining_jobs"
    model_class = TwinRetrainingJobRecord

    def __init__(self, supabase_client: Any = None):
        super().__init__(supabase_client)
        # True once resolution has been attempted (an explicit client counts).
        self._client_resolved = supabase_client is not None

    async def _ensure_async_client(self) -> None:
        """Lazily bind the process-shared ASYNC Supabase client.

        ``BaseRepository`` ``await``s ``.execute()``, which requires the async
        client (``create_async_client`` via ``get_async_supabase_client()``), NOT
        the sync ``get_supabase()``. Resolved once and cached; stays inert
        (``client`` ``None``) when Supabase is unconfigured so reads/writes no-op
        rather than raise. Both the API and the Celery worker resolve the SAME
        env-driven cached client — that shared binding is what makes a job
        created in one process visible in the other.
        """
        if self._client_resolved:
            return
        self._client_resolved = True
        try:
            from src.memory.services.factories import get_async_supabase_client

            self.client = await get_async_supabase_client()
        except Exception:  # noqa: BLE001 — unconfigured/unavailable → stay inert
            logger.debug("Async Supabase client unavailable; TwinRetrainingJobRepository is inert")
            self.client = None

    @staticmethod
    def _serialize(record: TwinRetrainingJobRecord) -> Dict[str, Any]:
        """Model -> JSON-safe insert payload (datetimes as ISO-8601 strings)."""
        data = record.model_dump()
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    async def create_job(
        self,
        *,
        job_id: str,
        model_id: str,
        trigger_reason: str,
        status: str,
        fidelity_before: float,
        training_config: Dict[str, Any],
    ) -> TwinRetrainingJobRecord:
        """Persist a newly-triggered job. Returns the record even when inert."""
        await self._ensure_async_client()
        record = TwinRetrainingJobRecord(
            id=job_id,
            model_id=model_id,
            trigger_reason=trigger_reason,
            status=status,
            fidelity_before=fidelity_before,
            training_config=training_config,
        )
        if self.client:
            await self.client.table(self.table_name).insert(self._serialize(record)).execute()
            logger.info(f"Persisted twin retraining job {job_id} (model {model_id})")
        return record

    async def get_job(self, job_id: str) -> Optional[TwinRetrainingJobRecord]:
        """Fetch a job by id (``None`` if absent or no client)."""
        await self._ensure_async_client()
        return await self.get_by_id(job_id)

    async def complete_job(
        self,
        job_id: str,
        *,
        new_model_id: str,
        fidelity_after: float,
        success: bool = True,
    ) -> Optional[TwinRetrainingJobRecord]:
        """Record a certified completion (real metric) or a failed terminal state.

        The caller MUST have certified ``fidelity_after`` as a real finite
        validation metric before calling with ``success=True`` (the #548
        fail-closed contract lives in the task); this method only persists.
        ``new_model_id`` / ``fidelity_after`` are written ONLY on ``success=True``
        — a non-success completion records the failed status WITHOUT a metric,
        preserving the #548 invariant (no fabricated metric on a failed job).
        """
        await self._ensure_async_client()
        updates: Dict[str, Any] = {
            "status": "completed" if success else "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if success:
            updates["new_model_id"] = new_model_id
            updates["fidelity_after"] = fidelity_after
        return await self.update(job_id, updates)

    async def fail_job(
        self,
        job_id: str,
        *,
        error_message: str,
    ) -> Optional[TwinRetrainingJobRecord]:
        """Mark a job FAILED with the reason, writing NO fidelity metric (#548)."""
        await self._ensure_async_client()
        updates: Dict[str, Any] = {
            "status": "failed",
            "error_message": error_message,
            # Clear any stale metric (#548: a failed job carries NO metric) so a
            # complete->fail transition cannot leave a real-looking score behind.
            "new_model_id": None,
            "fidelity_after": None,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        return await self.update(job_id, updates)

    async def cancel_job(
        self,
        job_id: str,
        *,
        reason: str,
    ) -> Optional[TwinRetrainingJobRecord]:
        """Mark a job CANCELLED with the reason."""
        await self._ensure_async_client()
        updates: Dict[str, Any] = {
            "status": "cancelled",
            "error_message": f"Cancelled: {reason}",
        }
        return await self.update(job_id, updates)
