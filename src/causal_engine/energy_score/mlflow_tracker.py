"""
MLflow Integration for Energy Score Logging

Provides seamless integration with the existing MLflow experiment tracking
infrastructure, logging energy score metrics alongside standard model metrics.

Integration Points:
    - ml_experiments table (via MLflow tracking)
    - estimator_evaluations table (via direct Supabase logging)
    - Dashboard metrics (via MLflow queries)

Usage:
    tracker = EnergyScoreMLflowTracker()
    with tracker.start_selection_run(experiment_name="causal_analysis"):
        result = selector.select(treatment, outcome, covariates)
        tracker.log_selection_result(result)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from .estimator_selector import SelectionResult

logger = logging.getLogger(__name__)


@dataclass
class ExperimentContext:
    """Context for an MLflow experiment run."""

    experiment_id: str
    run_id: str
    experiment_name: str
    started_at: datetime

    # E2I specific
    brand: Optional[str] = None
    region: Optional[str] = None
    kpi_name: Optional[str] = None


class EnergyScoreMLflowTracker:
    """
    Tracks energy score metrics in MLflow.

    Integrates with your existing MLflow infrastructure to log:
    - Estimator evaluation results
    - Energy score components
    - Selection decisions
    - Performance comparisons

    Example:
        tracker = EnergyScoreMLflowTracker()

        with tracker.start_selection_run("trigger_effectiveness"):
            result = selector.select(T, Y, X)
            tracker.log_selection_result(result)

        # Later: query results
        comparison = tracker.get_selection_comparison(days=30)
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_prefix: str = "e2i_causal",
        enable_db_logging: bool = True,
        db_connection_string: Optional[str] = None,
    ):
        """
        Initialize the tracker.

        Args:
            tracking_uri: MLflow tracking server URI (default: from env)
            experiment_prefix: Prefix for experiment names
            enable_db_logging: Whether to also log to estimator_evaluations table
            db_connection_string: Supabase connection string for direct logging
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000"
        )
        self.experiment_prefix = experiment_prefix
        self.enable_db_logging = enable_db_logging
        self.db_connection_string = db_connection_string or os.getenv(
            "DATABASE_URL"
        )

        self._current_context: Optional[ExperimentContext] = None
        self._mlflow_available = self._check_mlflow()

    def _check_mlflow(self) -> bool:
        """Check if MLflow is available."""
        try:
            import mlflow
            os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
            return True
        except ImportError:
            logger.warning("MLflow not installed, metrics will only log to database")
            return False

    @contextmanager
    def start_selection_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        kpi_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ):
        """
        Start an MLflow run for estimator selection.

        Args:
            experiment_name: Name of the experiment (will be prefixed)
            run_name: Optional run name
            brand: E2I brand context
            region: E2I region context
            kpi_name: KPI being analyzed
            tags: Additional MLflow tags

        Yields:
            ExperimentContext for the run
        """
        full_experiment_name = f"{self.experiment_prefix}/{experiment_name}"

        if self._mlflow_available:
            import mlflow

            # Set or create experiment
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    full_experiment_name,
                    tags={"framework": "e2i_causal", "version": "4.2"}
                )
            else:
                experiment_id = experiment.experiment_id

            # Start run
            run_name = run_name or f"selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
            ) as run:
                # Log tags
                mlflow.set_tag("selection_strategy", "energy_score")
                mlflow.set_tag("framework_version", "4.2")

                if brand:
                    mlflow.set_tag("brand", brand)
                if region:
                    mlflow.set_tag("region", region)
                if kpi_name:
                    mlflow.set_tag("kpi_name", kpi_name)

                for key, value in (tags or {}).items():
                    mlflow.set_tag(key, value)

                self._current_context = ExperimentContext(
                    experiment_id=experiment_id,
                    run_id=run.info.run_id,
                    experiment_name=experiment_name,
                    started_at=datetime.now(),
                    brand=brand,
                    region=region,
                    kpi_name=kpi_name,
                )

                try:
                    yield self._current_context
                finally:
                    self._current_context = None
        else:
            # Fallback: create dummy context
            self._current_context = ExperimentContext(
                experiment_id=str(uuid4()),
                run_id=str(uuid4()),
                experiment_name=experiment_name,
                started_at=datetime.now(),
                brand=brand,
                region=region,
                kpi_name=kpi_name,
            )
            try:
                yield self._current_context
            finally:
                self._current_context = None

    def log_selection_result(
        self,
        result: "SelectionResult",
        additional_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a complete selection result to MLflow and database.

        Args:
            result: SelectionResult from EstimatorSelector
            additional_params: Additional parameters to log
        """
        if self._current_context is None:
            logger.warning("No active run context, creating standalone log")
            experiment_id = str(uuid4())
        else:
            experiment_id = self._current_context.experiment_id

        # Log to MLflow
        if self._mlflow_available:
            self._log_to_mlflow(result, additional_params)

        # Log to database
        if self.enable_db_logging:
            self._log_to_database(result, experiment_id)

    def _log_to_mlflow(
        self,
        result: "SelectionResult",
        additional_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log metrics to MLflow."""
        import mlflow

        # Log selection metrics
        mlflow.log_metric("selected_energy_score", result.selected.energy_score)
        if result.selected.ate is not None:
            mlflow.log_metric("selected_ate", result.selected.ate)
        if result.selected.ate_std is not None:
            mlflow.log_metric("selected_ate_std", result.selected.ate_std)

        # Log comparison metrics
        mlflow.log_metric("n_estimators_evaluated", len(result.all_results))
        mlflow.log_metric("n_estimators_succeeded",
                         sum(1 for r in result.all_results if r.success))
        mlflow.log_metric("energy_score_gap", result.energy_score_gap)
        mlflow.log_metric("total_selection_time_ms", result.total_time_ms)

        # Log energy scores for each estimator
        for est_type, score in result.energy_scores.items():
            mlflow.log_metric(f"energy_score_{est_type}", score)

        # Log parameters
        mlflow.log_param("selected_estimator", result.selected.estimator_type.value)
        mlflow.log_param("selection_strategy", result.selection_strategy.value)
        mlflow.log_param("selection_reason", result.selection_reason)

        # Log additional params
        for key, value in (additional_params or {}).items():
            mlflow.log_param(key, value)

        # Log detailed results as artifact
        results_dict = {
            "selection_result": result.to_dict(),
            "all_evaluations": [r.to_dict() for r in result.all_results],
            "timestamp": datetime.now().isoformat(),
        }

        # Write to temp file and log
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results_dict, f, indent=2, default=str)
            temp_path = f.name

        mlflow.log_artifact(temp_path, "selection_details")
        os.unlink(temp_path)

    def _log_to_database(
        self,
        result: "SelectionResult",
        experiment_id: str,
    ) -> None:
        """Log to estimator_evaluations table via Supabase."""
        if not self.db_connection_string:
            logger.warning("No database connection string, skipping DB logging")
            return

        try:
            import psycopg2
            from psycopg2.extras import Json

            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            for i, eval_result in enumerate(result.all_results):
                cur.execute("""
                    INSERT INTO estimator_evaluations (
                        experiment_id, estimator_type, estimator_priority,
                        success, ate, ate_std, ate_ci_lower, ate_ci_upper,
                        energy_score, treatment_balance_score, outcome_fit_score,
                        propensity_calibration, energy_ci_lower, energy_ci_upper,
                        energy_bootstrap_std, n_samples, n_treated, n_control,
                        estimation_time_ms, energy_computation_time_ms,
                        was_selected, selection_reason, error_message, error_type,
                        estimator_params, energy_details
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    experiment_id,
                    eval_result.estimator_type.value,
                    i + 1,  # priority
                    eval_result.success,
                    eval_result.ate,
                    eval_result.ate_std,
                    eval_result.ate_ci_lower,
                    eval_result.ate_ci_upper,
                    eval_result.energy_score if eval_result.success else None,
                    eval_result.energy_score_result.treatment_balance_score if eval_result.energy_score_result else None,
                    eval_result.energy_score_result.outcome_fit_score if eval_result.energy_score_result else None,
                    eval_result.energy_score_result.propensity_calibration if eval_result.energy_score_result else None,
                    eval_result.energy_score_result.ci_lower if eval_result.energy_score_result else None,
                    eval_result.energy_score_result.ci_upper if eval_result.energy_score_result else None,
                    eval_result.energy_score_result.bootstrap_std if eval_result.energy_score_result else None,
                    eval_result.energy_score_result.n_samples if eval_result.energy_score_result else None,
                    eval_result.energy_score_result.n_treated if eval_result.energy_score_result else None,
                    eval_result.energy_score_result.n_control if eval_result.energy_score_result else None,
                    eval_result.estimation_time_ms,
                    eval_result.energy_score_result.computation_time_ms if eval_result.energy_score_result else None,
                    eval_result.estimator_type == result.selected.estimator_type,
                    result.selection_reason if eval_result.estimator_type == result.selected.estimator_type else None,
                    eval_result.error_message,
                    eval_result.error_type,
                    Json({}),  # estimator_params
                    Json(eval_result.energy_score_result.details if eval_result.energy_score_result else {}),
                ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Logged {len(result.all_results)} evaluations to database")

        except Exception as e:
            logger.error(f"Failed to log to database: {e}")

    def get_selection_comparison(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get comparison of energy score vs legacy selection.

        Returns metrics showing how often energy score selection
        differs from legacy first-success, and the improvement gained.
        """
        if not self.db_connection_string:
            logger.warning("No database connection for comparison query")
            return {}

        try:
            import psycopg2

            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            cur.execute("""
                SELECT * FROM v_selection_comparison
            """)

            row = cur.fetchone()
            if row:
                result = {
                    "total_experiments": row[0],
                    "same_selection": row[1],
                    "different_selection": row[2],
                    "pct_improved": row[3],
                    "avg_energy_improvement": row[4],
                }
            else:
                result = {}

            cur.close()
            conn.close()

            return result

        except Exception as e:
            logger.error(f"Failed to get comparison: {e}")
            return {}

    def get_estimator_performance(self) -> list[dict[str, Any]]:
        """Get performance metrics for each estimator type."""
        if not self.db_connection_string:
            return []

        try:
            import psycopg2

            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            cur.execute("""
                SELECT * FROM v_estimator_performance
            """)

            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()

            return results

        except Exception as e:
            logger.error(f"Failed to get performance: {e}")
            return []


# Convenience function
def create_tracker(**kwargs) -> EnergyScoreMLflowTracker:
    """Create a tracker with environment-based configuration."""
    return EnergyScoreMLflowTracker(**kwargs)
