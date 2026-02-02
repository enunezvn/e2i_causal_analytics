"""
WS1 Model Performance KPI Calculators

Implements calculators for model performance metrics:
- ROC-AUC
- PR-AUC
- F1 Score
- Recall@Top-K
- Brier Score
- Calibration Slope
- SHAP Coverage
- Fairness Gap
- Feature Drift (PSI)
"""

from typing import Any

import numpy as np

from src.kpi.calculator import KPICalculatorBase
from src.kpi.models import (
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)


class ModelPerformanceCalculator(KPICalculatorBase):
    """Calculator for WS1 Model Performance KPIs."""

    def __init__(self, db_client: Any = None, mlflow_client: Any = None):
        """Initialize with database and MLflow clients.

        Args:
            db_client: Database client for executing queries.
            mlflow_client: MLflow client for retrieving model metrics.
        """
        self._db_client = db_client
        self._mlflow_client = mlflow_client

    @property
    def db_client(self) -> Any:
        """Get database client, lazily initializing if needed."""
        if self._db_client is None:
            from src.repositories import get_supabase_client

            self._db_client = get_supabase_client()
        return self._db_client

    @property
    def mlflow_client(self) -> Any:
        """Get MLflow client, lazily initializing if needed."""
        if self._mlflow_client is None:
            try:
                import mlflow

                self._mlflow_client = mlflow.tracking.MlflowClient()
            except ImportError:
                pass
        return self._mlflow_client

    def supports(self, kpi: KPIMetadata) -> bool:
        """Check if this calculator supports the given KPI."""
        return kpi.workstream == Workstream.WS1_MODEL_PERFORMANCE

    def calculate(self, kpi: KPIMetadata, context: dict[str, Any] | None = None) -> KPIResult:
        """Calculate a model performance KPI.

        Args:
            kpi: The KPI metadata defining what to calculate.
            context: Optional context with model_name, model_version, etc.

        Returns:
            KPIResult with calculated value and status.
        """
        context = context or {}

        # Route to specific calculator based on KPI ID
        calculator_map = {
            "WS1-MP-001": self._calc_roc_auc,
            "WS1-MP-002": self._calc_pr_auc,
            "WS1-MP-003": self._calc_f1_score,
            "WS1-MP-004": self._calc_recall_at_k,
            "WS1-MP-005": self._calc_brier_score,
            "WS1-MP-006": self._calc_calibration_slope,
            "WS1-MP-007": self._calc_shap_coverage,
            "WS1-MP-008": self._calc_fairness_gap,
            "WS1-MP-009": self._calc_feature_drift,
        }

        calc_func = calculator_map.get(kpi.id)
        if calc_func is None:
            return KPIResult(
                kpi_id=kpi.id,
                error=f"No calculator implemented for {kpi.id}",
            )

        try:
            value = calc_func(context)
            # Determine if lower is better
            lower_is_better = kpi.id in {"WS1-MP-005", "WS1-MP-008", "WS1-MP-009"}
            status = self._evaluate_status(kpi, value, lower_is_better)
            return KPIResult(
                kpi_id=kpi.id,
                value=value,
                status=status,
                metadata={"context": context, "lower_is_better": lower_is_better},
            )
        except Exception as e:
            return KPIResult(
                kpi_id=kpi.id,
                error=str(e),
            )

    def _evaluate_status(
        self, kpi: KPIMetadata, value: float | None, lower_is_better: bool = False
    ) -> KPIStatus:
        """Evaluate KPI value against thresholds."""
        if value is None or kpi.threshold is None:
            return KPIStatus.UNKNOWN
        return kpi.threshold.evaluate(value, lower_is_better=lower_is_better)

    def _calc_roc_auc(self, context: dict[str, Any]) -> float:
        """Calculate WS1-MP-001: ROC-AUC.

        Retrieves latest ROC-AUC from MLflow or model_metrics table.
        """
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "roc_auc", default=0.5)

    def _calc_pr_auc(self, context: dict[str, Any]) -> float:
        """Calculate WS1-MP-002: PR-AUC.

        Precision-Recall Area Under Curve.
        """
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "pr_auc", default=0.5)

    def _calc_f1_score(self, context: dict[str, Any]) -> float:
        """Calculate WS1-MP-003: F1 Score.

        Harmonic mean of precision and recall.
        """
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "f1_score", default=0.0)

    def _calc_recall_at_k(self, context: dict[str, Any]) -> float:
        """Calculate WS1-MP-004: Recall@Top-K.

        Recall among top K predictions (default K=100).
        """
        model_name = context.get("model_name", "default_model")
        k = context.get("k", 100)
        metric_name = f"recall_at_{k}"
        return self._get_metric_from_mlflow(model_name, metric_name, default=0.0)

    def _calc_brier_score(self, context: dict[str, Any]) -> float:
        """Calculate WS1-MP-005: Brier Score.

        Mean squared error of probabilistic predictions.
        Lower is better (0 = perfect calibration).
        """
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "brier_score", default=0.25)

    def _calc_calibration_slope(self, context: dict[str, Any]) -> float:
        """Calculate WS1-MP-006: Calibration Slope.

        Slope of reliability diagram (1.0 = perfectly calibrated).
        """
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "calibration_slope", default=1.0)

    def _calc_shap_coverage(self, context: dict[str, Any]) -> float:
        """Calculate WS1-MP-007: SHAP Coverage.

        Percentage of predictions with SHAP explanations generated.
        """
        query = """
            SELECT
                COUNT(CASE WHEN shap_values IS NOT NULL THEN 1 END)::float /
                NULLIF(COUNT(*), 0) as coverage
            FROM predictions p
            WHERE p.created_at >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
        if result and result[0]["coverage"] is not None:
            return result[0]["coverage"]
        return 0.0

    def _calc_fairness_gap(self, context: dict[str, Any]) -> float:
        """Calculate WS1-MP-008: Fairness Gap (ΔRecall).

        Max difference in recall across protected groups.
        Lower is better (0 = perfectly fair).
        """
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "fairness_gap", default=0.1)

    def _calc_feature_drift(self, context: dict[str, Any]) -> float:
        """Calculate WS1-MP-009: Feature Drift (PSI).

        Population Stability Index for feature distribution drift.
        Lower is better (< 0.1 = stable, 0.1-0.25 = moderate, > 0.25 = significant).
        """
        model_name = context.get("model_name", "default_model")

        # Query drift monitoring table
        query = """
            SELECT AVG(psi_value) as avg_psi
            FROM feature_drift_metrics
            WHERE model_name = $1
            AND measured_at >= NOW() - INTERVAL '7 days'
        """
        result = self._execute_query(query, [model_name])
        if result and result[0]["avg_psi"] is not None:
            return result[0]["avg_psi"]

        # Fall back to MLflow metric
        return self._get_metric_from_mlflow(model_name, "feature_drift_psi", default=0.0)

    def _get_metric_from_mlflow(
        self, model_name: str, metric_name: str, default: float = 0.0
    ) -> float:
        """Get a metric value from MLflow for the latest model version.

        Args:
            model_name: Name of the registered model.
            metric_name: Name of the metric to retrieve.
            default: Default value if metric not found.

        Returns:
            The metric value or default.
        """
        if self.mlflow_client is None:
            return default

        try:
            # Get latest production version
            versions = self.mlflow_client.get_latest_versions(
                model_name, stages=["Production", "Staging", "None"]
            )
            if not versions:
                return default

            # Get run metrics
            run_id = versions[0].run_id
            run = self.mlflow_client.get_run(run_id)
            return float(run.data.metrics.get(metric_name, default))
        except Exception:
            return default

    def _execute_query(self, query: str, params: list[Any]) -> list[dict[str, Any]] | None:
        """Execute a SQL query and return results."""
        try:
            response = self.db_client.rpc("execute_sql", {"query": query}).execute()
            return response.data
        except Exception:
            return None


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index.

    Args:
        expected: Expected/reference distribution.
        actual: Actual/current distribution.
        bins: Number of bins for histogram.

    Returns:
        PSI value (0 = no drift, higher = more drift).
    """
    # Bin the distributions
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions with smoothing
    expected_pct = (expected_counts + 1) / (len(expected) + bins)
    actual_pct = (actual_counts + 1) / (len(actual) + bins)

    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)
