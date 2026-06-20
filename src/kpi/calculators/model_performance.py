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

Unavailability discipline (#439, F-007-PhaseB)
----------------------------------------------
This calculator does NOT return plausible-fake defaults (0.0 / 0.5 / 1.0 /
0.25 / 0.1) when MLflow is unreachable, the model is missing, the metric
is absent, or a query fails. Every `_calc_*` method propagates an
explicit `KPIResult(value=None, error="<reason>")` shape so that the
existing `_evaluate_status:120-121` fail-close primitive routes the
result through `KPIStatus.UNKNOWN`. Downstream consumers (dashboards,
alerts) can then distinguish "MLflow unreachable" from "model is random",
and "no rows in window" from "SHAP coverage is 0%".

Unavailability reasons:
  - mlflow_client_unavailable    (no MLflow client wired in)
  - model_not_found:<name>       (registry returned no versions)
  - metric_not_found:<metric>    (run exists but metric key absent)
  - mlflow_exception:<Class>:<msg>  (any other MLflow-side failure)
  - db_query_failed:<reason>     (SQL execution raised)
  - db_query_returned_empty[:<note>]  (no rows / NULL value)
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
from src.kpi.synthetic_mode import resolve_kpi_query_id


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
            KPIResult with calculated value and status. On unavailability,
            `value=None`, `error=<reason>`, `status=KPIStatus.UNKNOWN`.
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
            return KPIResult(  # type: ignore[call-arg]
                kpi_id=kpi.id,
                error=f"No calculator implemented for {kpi.id}",
            )

        try:
            value, error = calc_func(context)
            # Determine if lower is better
            lower_is_better = kpi.id in {"WS1-MP-005", "WS1-MP-008", "WS1-MP-009"}
            status = self._evaluate_status(kpi, value, lower_is_better)
            return KPIResult(  # type: ignore[call-arg]
                kpi_id=kpi.id,
                value=value,
                status=status,
                error=error,
                metadata={"context": context, "lower_is_better": lower_is_better},
            )
        except Exception as e:
            return KPIResult(  # type: ignore[call-arg]
                kpi_id=kpi.id,
                error=str(e),
            )

    def _evaluate_status(
        self, kpi: KPIMetadata, value: float | None, lower_is_better: bool = False
    ) -> KPIStatus:
        """Evaluate KPI value against thresholds.

        The `value is None -> UNKNOWN` branch is the load-bearing fail-close
        primitive for #439. Every `_calc_*` method that hits unavailability
        returns `value=None` so that this method routes the result through
        `KPIStatus.UNKNOWN` instead of fabricating GOOD/WARNING/CRITICAL
        from a plausible default.
        """
        if value is None or kpi.threshold is None:
            return KPIStatus.UNKNOWN
        return kpi.threshold.evaluate(value, lower_is_better=lower_is_better)

    # ------------------------------------------------------------------ gold-standard helper

    def _goldstd_metric(self, context: dict[str, Any], metric_name: str) -> float | None:
        """Per-brand average of the gold-standard models' holdout ``metric_name``.

        Best-effort PRIMARY source for the dashboard: ``context['brand']`` scopes
        to that brand's ``*_goldstd_lr_v1`` staging models (absent/All -> all 12).
        Returns ``None`` (caller then falls back to the existing corpus/MLflow
        legs) when no gold-standard data is available or a read fails — never
        raises, never fabricates.
        """
        from src.kpi.goldstd_model_perf import summarize_sync

        try:
            summary = summarize_sync(self.db_client, context.get("brand"))
        except Exception:
            return None
        if not summary:
            return None
        val = summary.get(metric_name)
        return float(val) if val is not None else None

    # ------------------------------------------------------------------ MLflow-backed metrics

    def _calc_roc_auc(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-001: ROC-AUC.

        PRIMARY (gold-standard): per-brand average of the gold-standard models'
        holdout ``auc_roc`` (brand-reactive; fixes the corpus-wide invariant
        value the dashboard previously showed for every brand).

        FALLBACK 1 (SQL): the real ``ml_predictions.model_auc`` corpus mean via
        the ``kpi_query`` allowlist (registry id ``model_performance_roc_auc``),
        the source declared by ``config/kpi_definitions.yaml`` for WS1-MP-001.

        FALLBACK 2 (MLflow): the preserved fail-closed leg. Never fabricates a
        plausible default at any leg.
        """
        gs = self._goldstd_metric(context, "auc_roc")
        if gs is not None:
            return gs, None
        result, db_error = self._execute_query("model_performance_roc_auc", [])
        if db_error is None and result:
            roc_auc = result[0].get("roc_auc")
            if roc_auc is not None:
                return float(roc_auc), None
        # SQL leg unavailable (error, empty, or NULL avg) -> MLflow fail-closed leg.
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "roc_auc")

    def _calc_pr_auc(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-002: PR-AUC."""
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "pr_auc")

    def _calc_f1_score(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-003: F1 Score.

        PRIMARY: per-brand average of the gold-standard models' holdout ``f1``.
        FALLBACK: MLflow (fail-closed; no fabricated default).
        """
        gs = self._goldstd_metric(context, "f1")
        if gs is not None:
            return gs, None
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "f1_score")

    def _calc_recall_at_k(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-004: Recall@Top-K."""
        model_name = context.get("model_name", "default_model")
        k = context.get("k", 100)
        metric_name = f"recall_at_{k}"
        return self._get_metric_from_mlflow(model_name, metric_name)

    def _calc_brier_score(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-005: Brier Score (lower is better)."""
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "brier_score")

    def _calc_calibration_slope(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-006: Calibration Slope."""
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "calibration_slope")

    def _calc_fairness_gap(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-008: Fairness Gap (lower is better)."""
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "fairness_gap")

    # ------------------------------------------------------------------ SQL-backed / hybrid metrics

    def _calc_shap_coverage(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-007: SHAP Coverage.

        Percentage of predictions with SHAP explanations generated.
        Source is SQL only. Unavailability reasons:
          - db_query_failed: `_execute_query` returned None (exception raised
            during execution).
          - db_query_returned_empty: query returned no rows OR `coverage` was
            NULL (zero-denominator window).
        """
        result, db_error = self._execute_query("model_performance_shap_coverage", [])
        if db_error is not None:
            return None, f"db_query_failed:{db_error}"
        # `_execute_query` succeeded but may have returned [] or a row with
        # NULL coverage.
        if not result:
            return None, "db_query_returned_empty:no_rows_in_window"
        coverage = result[0].get("coverage")
        if coverage is None:
            return None, "db_query_returned_empty:null_coverage"
        return float(coverage), None

    def _calc_feature_drift(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-009: Feature Drift (PSI, lower is better).

        Two legs: SQL primary (drift monitoring table) + MLflow fallback.
        Returns the SQL value if it succeeds with a real (non-NULL) PSI.
        Otherwise consults MLflow. If both legs fail-close, returns
        `value=None` with a combined error.
        """
        model_name = context.get("model_name", "default_model")

        # #577 WS1-MP-009: the SQL leg is now REGISTERED + seeded (migration 053). It is a
        # CORPUS-level aggregate — `SELECT AVG(test_statistic) AS avg_psi FROM ml_drift_history
        # WHERE test_type='psi' AND drift_type='data'` — registered with max_params=0, so the
        # call binds NO params. We deliberately do NOT bind `model_name` here: the calculator's
        # model_name is a STRING but ml_drift_history keys on a UUID `model_id` (NULL in the
        # seed — ml_model_registry has 0 rows), so a filter would be a LABEL-not-functional
        # no-op falsely implying per-model scoping that does not exist (cf. the max_params=0
        # siblings model_performance_shap_coverage and data_quality_label_quality). A 1-element
        # `params` would make `kpi_query` RAISE "expects 0 param(s), got 1". A future PR that
        # seeds real ml_model_registry rows + per-model drift can promote this to arity 1.
        # `model_name` is still used by the MLflow fallback leg below. The two-leg fail-closed
        # contract is preserved: empty/unseeded table -> AVG over 0 rows is NULL -> MLflow ->
        # fail-closed UNKNOWN (never a fabricated PSI).
        sql_result, sql_error = self._execute_query("model_performance_feature_drift", [])

        # First leg: SQL succeeded with a real PSI.
        if sql_result:
            avg_psi = sql_result[0].get("avg_psi")
            if avg_psi is not None:
                return float(avg_psi), None
            sql_leg_error = "db_query_returned_empty:null_avg_psi"
        elif sql_error is not None:
            sql_leg_error = f"db_query_failed:{sql_error}"
        else:
            sql_leg_error = "db_query_returned_empty:no_rows_in_window"

        # Second leg: MLflow fallback. If it returns a real value, the SQL
        # leg's silence is acceptable (no rows yet in the table is not a
        # KPI failure as long as MLflow has the metric).
        mlflow_value, mlflow_error = self._get_metric_from_mlflow(model_name, "feature_drift_psi")
        if mlflow_value is not None:
            return mlflow_value, None

        # Both legs failed: surface combined unavailability honestly.
        combined_error = (
            f"feature_drift_psi unavailable: sql_leg={sql_leg_error}; mlflow_leg={mlflow_error}"
        )
        return None, combined_error

    # ------------------------------------------------------------------ infra

    def _get_metric_from_mlflow(
        self, model_name: str, metric_name: str
    ) -> tuple[float | None, str | None]:
        """Get a metric value from MLflow for the latest model version.

        Returns:
            A `(value, error)` tuple. Exactly one of the two is non-None:
              - `(<float>, None)` on success.
              - `(None, "mlflow_client_unavailable")` when no MLflow client
                is wired in (e.g., mlflow not installed in this environment).
              - `(None, "model_not_found:<name>")` when the registry has no
                versions for the model.
              - `(None, "metric_not_found:<metric>")` when the latest run
                does not have the requested metric key.
              - `(None, "mlflow_exception:<ExceptionClass>:<msg[:200]>")`
                on any other MLflow-side failure.

        This method NEVER returns a plausible-fake default like 0.5 or 1.0
        when the metric is unavailable — that was the #439 anti-pattern.
        """
        if self.mlflow_client is None:
            return None, "mlflow_client_unavailable"

        try:
            versions = self.mlflow_client.get_latest_versions(
                model_name, stages=["Production", "Staging", "None"]
            )
            if not versions:
                return None, f"model_not_found:{model_name}"

            run_id = versions[0].run_id
            run = self.mlflow_client.get_run(run_id)
            metrics = run.data.metrics
            if metric_name not in metrics:
                return None, f"metric_not_found:{metric_name}"
            return float(metrics[metric_name]), None
        except Exception as e:
            msg = str(e)[:200]
            return None, f"mlflow_exception:{type(e).__name__}:{msg}"

    def _execute_query(
        self, query_id: str, params: list[Any]
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Run a vetted read-only KPI statement via the kpi_query allowlist RPC (#574).

        `query_id` indexes a statement in `kpi_query_registry`; `params` bind its
        $1..$N placeholders. Returns `(rows, error)` — exactly one is non-None:
          - `(<rows>, None)` on success (rows may be `[]`).
          - `(None, "<ExceptionClass>:<msg[:200]>")` on any execution failure.
        """
        try:
            # Demo/review: swap to the _include_synthetic twin under the
            # E2I_KPI_INCLUDE_SYNTHETIC flag (no-op otherwise). See synthetic_mode.py.
            query_id = resolve_kpi_query_id(query_id)
            response = self.db_client.rpc(
                "kpi_query", {"query_id": query_id, "params": params}
            ).execute()
            return response.data, None  # type: ignore[no-any-return]
        except Exception as e:
            msg = str(e)[:200]
            return None, f"{type(e).__name__}:{msg}"


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
