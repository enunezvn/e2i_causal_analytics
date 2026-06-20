"""Fail-closed unavailability tests for ModelPerformanceCalculator (#439).

Each of the 9 `_calc_*` methods must propagate unavailability honestly via
`KPIResult(value=None, error="<reason>")` instead of returning fabricated
plausible defaults (0.0/0.5/1.0/0.25/0.1).

Coverage matrix (per KPI):
  1. MLflow client unavailable   -> error="mlflow_client_unavailable"
  2. MLflow returns no versions  -> error="model_not_found:<name>"
  3. MLflow run lacks metric     -> error="metric_not_found:<metric>"
  4. MLflow raises Exception     -> error="mlflow_exception:<Class>:<msg>"
  5. MLflow returns real value   -> value=<float>, status per threshold

Calculators using SQL (`_calc_shap_coverage`, `_calc_feature_drift`) get
additional `db_query_failed` / `db_query_returned_empty` coverage.

All paths route through the existing `_evaluate_status:120-121`
`value is None -> KPIStatus.UNKNOWN` primitive (UNCHANGED in this PR).
"""

from unittest.mock import Mock

import pytest

from src.kpi.calculators.model_performance import ModelPerformanceCalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIStatus,
    KPIThreshold,
    Workstream,
)

# ---------------------------------------------------------------------------
# Fixtures: one KPIMetadata per `_calc_*` method
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator_no_mlflow():
    """Calculator with MLflow client explicitly unset (client=None).

    Sets the private `_mlflow_client` to a sentinel to bypass the lazy
    initialization in the `mlflow_client` property. We use a Mock and then
    override `mlflow_client` to return None via property monkeypatch in tests
    that need MLflow-unavailable behavior.
    """
    mock_db = Mock()
    calc = ModelPerformanceCalculator(db_client=mock_db, mlflow_client=None)
    # Force the lazy property to return None too — patch the property at the
    # class level so that the `if self.mlflow_client is None` check fires.
    return calc


@pytest.fixture
def calculator_with_mlflow():
    """Calculator with a Mock MLflow client wired in."""
    mock_db = Mock()
    mock_mlflow = Mock()
    return ModelPerformanceCalculator(db_client=mock_db, mlflow_client=mock_mlflow)


@pytest.fixture
def roc_auc_kpi():
    return KPIMetadata(
        id="WS1-MP-001",
        name="ROC-AUC",
        definition="Area Under ROC Curve",
        formula="sklearn.metrics.roc_auc_score",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(target=0.80, warning=0.70, critical=0.60),
    )


@pytest.fixture
def pr_auc_kpi():
    return KPIMetadata(
        id="WS1-MP-002",
        name="PR-AUC",
        definition="Precision-Recall AUC",
        formula="sklearn.metrics.average_precision_score",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(target=0.75, warning=0.65, critical=0.50),
    )


@pytest.fixture
def f1_score_kpi():
    return KPIMetadata(
        id="WS1-MP-003",
        name="F1 Score",
        definition="F1 score",
        formula="sklearn.metrics.f1_score",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(target=0.70, warning=0.60, critical=0.50),
    )


@pytest.fixture
def recall_at_k_kpi():
    return KPIMetadata(
        id="WS1-MP-004",
        name="Recall@Top-K",
        definition="Recall among top K predictions",
        formula="custom",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(target=0.60, warning=0.50, critical=0.40),
    )


@pytest.fixture
def brier_score_kpi():
    """Lower-is-better metric."""
    return KPIMetadata(
        id="WS1-MP-005",
        name="Brier Score",
        definition="Brier score",
        formula="sklearn.metrics.brier_score_loss",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(target=0.10, warning=0.20, critical=0.30),
    )


@pytest.fixture
def calibration_slope_kpi():
    return KPIMetadata(
        id="WS1-MP-006",
        name="Calibration Slope",
        definition="Reliability diagram slope",
        formula="custom",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(target=1.0, warning=0.8, critical=0.6),
    )


@pytest.fixture
def shap_coverage_kpi():
    return KPIMetadata(
        id="WS1-MP-007",
        name="SHAP Coverage",
        definition="Pct of predictions with SHAP",
        formula="count(shap)/count(*)",
        calculation_type=CalculationType.DERIVED,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(target=0.90, warning=0.80, critical=0.50),
    )


@pytest.fixture
def feature_drift_kpi():
    """Lower-is-better metric (PSI)."""
    return KPIMetadata(
        id="WS1-MP-009",
        name="Feature Drift (PSI)",
        definition="Population stability index",
        formula="custom",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(target=0.10, warning=0.25, critical=0.50),
    )


# ---------------------------------------------------------------------------
# Helpers for MLflow stubbing
# ---------------------------------------------------------------------------


def _stub_mlflow_returns_metric(
    calculator: ModelPerformanceCalculator, metric_name: str, value: float
) -> None:
    """Configure the mock MLflow client to return a real metric value."""
    mock_version = Mock()
    mock_version.run_id = "test-run-id"
    calculator._mlflow_client.get_latest_versions.return_value = [mock_version]
    mock_run = Mock()
    mock_run.data.metrics = {metric_name: value}
    calculator._mlflow_client.get_run.return_value = mock_run


def _stub_mlflow_no_versions(calculator: ModelPerformanceCalculator) -> None:
    """Configure the mock MLflow client to return an empty version list."""
    calculator._mlflow_client.get_latest_versions.return_value = []


def _stub_mlflow_run_missing_metric(calculator: ModelPerformanceCalculator) -> None:
    """Configure the mock MLflow client to return a run whose metrics dict
    is empty (metric_name not present)."""
    mock_version = Mock()
    mock_version.run_id = "test-run-id"
    calculator._mlflow_client.get_latest_versions.return_value = [mock_version]
    mock_run = Mock()
    mock_run.data.metrics = {}  # the requested metric is not present
    calculator._mlflow_client.get_run.return_value = mock_run


def _stub_mlflow_raises(calculator: ModelPerformanceCalculator, exc: Exception) -> None:
    """Configure the mock MLflow client to raise on `get_latest_versions`."""
    calculator._mlflow_client.get_latest_versions.side_effect = exc


def _stub_db_query_empty(calculator: ModelPerformanceCalculator) -> None:
    """Make the SQL allowlist leg return no rows (so the MLflow fallback runs).

    WS1-MP-001 ROC-AUC now reads ml_predictions.model_auc via the kpi_query
    allowlist FIRST and falls back to MLflow only when the SQL leg is
    unavailable. To exercise the MLflow fail-closed paths, stub the db_client's
    rpc(...).execute().data to an empty list (no rows / NULL avg).
    """
    calculator._db_client.rpc.return_value.execute.return_value.data = []


def _stub_db_query_returns(calculator: ModelPerformanceCalculator, query_id_value: dict) -> None:
    """Make the SQL allowlist leg return one row with the given dict."""
    calculator._db_client.rpc.return_value.execute.return_value.data = [query_id_value]


# ===========================================================================
# Per-calculator unavailability matrices
# ===========================================================================


class TestRocAucUnavailability:
    """WS1-MP-001 ROC-AUC: SQL-primary (ml_predictions.model_auc) + MLflow fallback.

    The calculator reads ml_predictions.model_auc via the kpi_query allowlist
    FIRST, falling back to MLflow only when the SQL leg is genuinely unavailable
    (empty/NULL). These tests stub the SQL leg empty to exercise the preserved
    MLflow fail-closed paths, plus add a SQL-primary success path.
    """

    def test_sql_primary_returns_real_value(self, calculator_with_mlflow, roc_auc_kpi):
        # SQL leg returns the real corpus mean ROC-AUC (~0.80) — no MLflow needed.
        _stub_db_query_returns(calculator_with_mlflow, {"roc_auc": 0.7998})
        result = calculator_with_mlflow.calculate(roc_auc_kpi, {})
        assert result.value is not None
        assert abs(result.value - 0.7998) < 1e-6
        assert result.error is None

    def test_mlflow_client_unavailable(self, calculator_no_mlflow, roc_auc_kpi, monkeypatch):
        _stub_db_query_empty(calculator_no_mlflow)
        # Force the lazy property to return None (simulating no mlflow installed)
        monkeypatch.setattr(
            ModelPerformanceCalculator, "mlflow_client", property(lambda self: None)
        )
        result = calculator_no_mlflow.calculate(roc_auc_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error == "mlflow_client_unavailable"
        assert result.status == KPIStatus.UNKNOWN

    def test_model_not_found(self, calculator_with_mlflow, roc_auc_kpi):
        _stub_db_query_empty(calculator_with_mlflow)
        _stub_mlflow_no_versions(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(roc_auc_kpi, {"model_name": "missing_model"})
        assert result.value is None
        assert result.error == "model_not_found:missing_model"
        assert result.status == KPIStatus.UNKNOWN

    def test_metric_not_found(self, calculator_with_mlflow, roc_auc_kpi):
        _stub_db_query_empty(calculator_with_mlflow)
        _stub_mlflow_run_missing_metric(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(roc_auc_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error == "metric_not_found:roc_auc"
        assert result.status == KPIStatus.UNKNOWN

    def test_mlflow_exception(self, calculator_with_mlflow, roc_auc_kpi):
        _stub_db_query_empty(calculator_with_mlflow)
        _stub_mlflow_raises(calculator_with_mlflow, RuntimeError("connection refused"))
        result = calculator_with_mlflow.calculate(roc_auc_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error is not None
        assert result.error.startswith("mlflow_exception:RuntimeError")
        assert "connection refused" in result.error
        assert result.status == KPIStatus.UNKNOWN

    def test_real_value_returned(self, calculator_with_mlflow, roc_auc_kpi):
        # SQL leg empty -> MLflow fallback returns a real value.
        _stub_db_query_empty(calculator_with_mlflow)
        _stub_mlflow_returns_metric(calculator_with_mlflow, "roc_auc", 0.85)
        result = calculator_with_mlflow.calculate(roc_auc_kpi, {"model_name": "test"})
        assert result.value == 0.85
        assert result.error is None
        assert result.status == KPIStatus.GOOD


class TestPrAucUnavailability:
    """WS1-MP-002 PR-AUC: 5 unavailability paths."""

    def test_mlflow_client_unavailable(self, calculator_no_mlflow, pr_auc_kpi, monkeypatch):
        monkeypatch.setattr(
            ModelPerformanceCalculator, "mlflow_client", property(lambda self: None)
        )
        result = calculator_no_mlflow.calculate(pr_auc_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error == "mlflow_client_unavailable"
        assert result.status == KPIStatus.UNKNOWN

    def test_model_not_found(self, calculator_with_mlflow, pr_auc_kpi):
        _stub_mlflow_no_versions(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(pr_auc_kpi, {"model_name": "absent"})
        assert result.value is None
        assert result.error == "model_not_found:absent"
        assert result.status == KPIStatus.UNKNOWN

    def test_metric_not_found(self, calculator_with_mlflow, pr_auc_kpi):
        _stub_mlflow_run_missing_metric(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(pr_auc_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error == "metric_not_found:pr_auc"
        assert result.status == KPIStatus.UNKNOWN

    def test_mlflow_exception(self, calculator_with_mlflow, pr_auc_kpi):
        _stub_mlflow_raises(calculator_with_mlflow, ValueError("bad URI"))
        result = calculator_with_mlflow.calculate(pr_auc_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error is not None
        assert result.error.startswith("mlflow_exception:ValueError")
        assert result.status == KPIStatus.UNKNOWN

    def test_real_value_returned(self, calculator_with_mlflow, pr_auc_kpi):
        _stub_mlflow_returns_metric(calculator_with_mlflow, "pr_auc", 0.80)
        result = calculator_with_mlflow.calculate(pr_auc_kpi, {"model_name": "test"})
        assert result.value == 0.80
        assert result.error is None
        assert result.status == KPIStatus.GOOD


class TestF1ScoreUnavailability:
    """WS1-MP-003 F1: 5 unavailability paths."""

    def test_mlflow_client_unavailable(self, calculator_no_mlflow, f1_score_kpi, monkeypatch):
        monkeypatch.setattr(
            ModelPerformanceCalculator, "mlflow_client", property(lambda self: None)
        )
        result = calculator_no_mlflow.calculate(f1_score_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error == "mlflow_client_unavailable"
        assert result.status == KPIStatus.UNKNOWN

    def test_model_not_found(self, calculator_with_mlflow, f1_score_kpi):
        _stub_mlflow_no_versions(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(f1_score_kpi, {"model_name": "absent"})
        assert result.value is None
        assert result.error == "model_not_found:absent"
        assert result.status == KPIStatus.UNKNOWN

    def test_metric_not_found(self, calculator_with_mlflow, f1_score_kpi):
        _stub_mlflow_run_missing_metric(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(f1_score_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error == "metric_not_found:f1_score"
        assert result.status == KPIStatus.UNKNOWN

    def test_mlflow_exception(self, calculator_with_mlflow, f1_score_kpi):
        _stub_mlflow_raises(calculator_with_mlflow, ConnectionError("net down"))
        result = calculator_with_mlflow.calculate(f1_score_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error is not None
        assert result.error.startswith("mlflow_exception:ConnectionError")
        assert result.status == KPIStatus.UNKNOWN

    def test_real_value_returned(self, calculator_with_mlflow, f1_score_kpi):
        _stub_mlflow_returns_metric(calculator_with_mlflow, "f1_score", 0.75)
        result = calculator_with_mlflow.calculate(f1_score_kpi, {"model_name": "test"})
        assert result.value == 0.75
        assert result.error is None
        assert result.status == KPIStatus.GOOD


class TestRecallAtKUnavailability:
    """WS1-MP-004 Recall@Top-K: 5 unavailability paths."""

    def test_mlflow_client_unavailable(self, calculator_no_mlflow, recall_at_k_kpi, monkeypatch):
        monkeypatch.setattr(
            ModelPerformanceCalculator, "mlflow_client", property(lambda self: None)
        )
        result = calculator_no_mlflow.calculate(recall_at_k_kpi, {"model_name": "test", "k": 100})
        assert result.value is None
        assert result.error == "mlflow_client_unavailable"
        assert result.status == KPIStatus.UNKNOWN

    def test_model_not_found(self, calculator_with_mlflow, recall_at_k_kpi):
        _stub_mlflow_no_versions(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(
            recall_at_k_kpi, {"model_name": "absent", "k": 100}
        )
        assert result.value is None
        assert result.error == "model_not_found:absent"
        assert result.status == KPIStatus.UNKNOWN

    def test_metric_not_found(self, calculator_with_mlflow, recall_at_k_kpi):
        _stub_mlflow_run_missing_metric(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(recall_at_k_kpi, {"model_name": "test", "k": 100})
        assert result.value is None
        assert result.error == "metric_not_found:recall_at_100"
        assert result.status == KPIStatus.UNKNOWN

    def test_mlflow_exception(self, calculator_with_mlflow, recall_at_k_kpi):
        _stub_mlflow_raises(calculator_with_mlflow, KeyError("missing"))
        result = calculator_with_mlflow.calculate(recall_at_k_kpi, {"model_name": "test", "k": 100})
        assert result.value is None
        assert result.error is not None
        assert result.error.startswith("mlflow_exception:KeyError")
        assert result.status == KPIStatus.UNKNOWN

    def test_real_value_returned(self, calculator_with_mlflow, recall_at_k_kpi):
        _stub_mlflow_returns_metric(calculator_with_mlflow, "recall_at_100", 0.65)
        result = calculator_with_mlflow.calculate(recall_at_k_kpi, {"model_name": "test", "k": 100})
        assert result.value == 0.65
        assert result.error is None
        assert result.status == KPIStatus.GOOD


class TestBrierScoreUnavailability:
    """WS1-MP-005 Brier (lower-is-better): 5 unavailability paths."""

    def test_mlflow_client_unavailable(self, calculator_no_mlflow, brier_score_kpi, monkeypatch):
        monkeypatch.setattr(
            ModelPerformanceCalculator, "mlflow_client", property(lambda self: None)
        )
        result = calculator_no_mlflow.calculate(brier_score_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error == "mlflow_client_unavailable"
        assert result.status == KPIStatus.UNKNOWN

    def test_model_not_found(self, calculator_with_mlflow, brier_score_kpi):
        _stub_mlflow_no_versions(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(brier_score_kpi, {"model_name": "absent"})
        assert result.value is None
        assert result.error == "model_not_found:absent"
        assert result.status == KPIStatus.UNKNOWN

    def test_metric_not_found(self, calculator_with_mlflow, brier_score_kpi):
        _stub_mlflow_run_missing_metric(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(brier_score_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error == "metric_not_found:brier_score"
        assert result.status == KPIStatus.UNKNOWN

    def test_mlflow_exception(self, calculator_with_mlflow, brier_score_kpi):
        _stub_mlflow_raises(calculator_with_mlflow, RuntimeError("internal"))
        result = calculator_with_mlflow.calculate(brier_score_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error is not None
        assert result.error.startswith("mlflow_exception:RuntimeError")
        assert result.status == KPIStatus.UNKNOWN

    def test_real_value_returned(self, calculator_with_mlflow, brier_score_kpi):
        _stub_mlflow_returns_metric(calculator_with_mlflow, "brier_score", 0.08)
        result = calculator_with_mlflow.calculate(brier_score_kpi, {"model_name": "test"})
        assert result.value == 0.08
        assert result.error is None
        assert result.status == KPIStatus.GOOD


class TestCalibrationSlopeUnavailability:
    """WS1-MP-006 Calibration Slope: 5 unavailability paths."""

    def test_mlflow_client_unavailable(
        self, calculator_no_mlflow, calibration_slope_kpi, monkeypatch
    ):
        monkeypatch.setattr(
            ModelPerformanceCalculator, "mlflow_client", property(lambda self: None)
        )
        result = calculator_no_mlflow.calculate(calibration_slope_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error == "mlflow_client_unavailable"
        assert result.status == KPIStatus.UNKNOWN

    def test_model_not_found(self, calculator_with_mlflow, calibration_slope_kpi):
        _stub_mlflow_no_versions(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(calibration_slope_kpi, {"model_name": "absent"})
        assert result.value is None
        assert result.error == "model_not_found:absent"
        assert result.status == KPIStatus.UNKNOWN

    def test_metric_not_found(self, calculator_with_mlflow, calibration_slope_kpi):
        _stub_mlflow_run_missing_metric(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(calibration_slope_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error == "metric_not_found:calibration_slope"
        assert result.status == KPIStatus.UNKNOWN

    def test_mlflow_exception(self, calculator_with_mlflow, calibration_slope_kpi):
        _stub_mlflow_raises(calculator_with_mlflow, OSError("disk"))
        result = calculator_with_mlflow.calculate(calibration_slope_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error is not None
        assert result.error.startswith("mlflow_exception:OSError")
        assert result.status == KPIStatus.UNKNOWN

    def test_real_value_returned(self, calculator_with_mlflow, calibration_slope_kpi):
        _stub_mlflow_returns_metric(calculator_with_mlflow, "calibration_slope", 1.0)
        result = calculator_with_mlflow.calculate(calibration_slope_kpi, {"model_name": "test"})
        assert result.value == 1.0
        assert result.error is None
        assert result.status == KPIStatus.GOOD


class TestShapCoverageUnavailability:
    """WS1-MP-007 SHAP Coverage: DB-backed unavailability paths.

    Differs from MLflow-backed calculators:
      - Source is SQL via `_execute_query`, not MLflow.
      - Two unavailability reasons:
          * db_query_failed     (exception in query execution)
          * db_query_returned_empty (empty result list / coverage IS None)

    `_execute_query` returns a `(rows, error)` tuple. The mocks below
    follow that contract.
    """

    def test_db_query_failed(self, calculator_with_mlflow, shap_coverage_kpi):
        # `_execute_query` returns (None, "<ExceptionClass>:<msg>") on
        # internal exception.
        calculator_with_mlflow._execute_query = Mock(
            return_value=(None, "OperationalError:conn refused")
        )
        result = calculator_with_mlflow.calculate(shap_coverage_kpi)
        assert result.value is None
        assert result.error is not None
        assert result.error.startswith("db_query_failed")
        assert result.status == KPIStatus.UNKNOWN

    def test_db_query_returned_empty(self, calculator_with_mlflow, shap_coverage_kpi):
        # Empty row list — no rows in `predictions` window
        calculator_with_mlflow._execute_query = Mock(return_value=([], None))
        result = calculator_with_mlflow.calculate(shap_coverage_kpi)
        assert result.value is None
        assert result.error is not None
        assert result.error.startswith("db_query_returned_empty")
        assert result.status == KPIStatus.UNKNOWN

    def test_db_query_null_coverage(self, calculator_with_mlflow, shap_coverage_kpi):
        # Row present but coverage IS NULL (NULLIF zero-denominator path)
        calculator_with_mlflow._execute_query = Mock(return_value=([{"coverage": None}], None))
        result = calculator_with_mlflow.calculate(shap_coverage_kpi)
        assert result.value is None
        assert result.error is not None
        assert result.error.startswith("db_query_returned_empty")
        assert result.status == KPIStatus.UNKNOWN

    def test_real_value_returned(self, calculator_with_mlflow, shap_coverage_kpi):
        calculator_with_mlflow._execute_query = Mock(return_value=([{"coverage": 0.92}], None))
        result = calculator_with_mlflow.calculate(shap_coverage_kpi)
        assert result.value == 0.92
        assert result.error is None
        assert result.status == KPIStatus.GOOD


class TestFeatureDriftUnavailability:
    """WS1-MP-009 Feature Drift (PSI, lower-is-better): SQL primary +
    MLflow fallback chained-leg behavior."""

    def test_sql_succeeds_uses_db_value(self, calculator_with_mlflow, feature_drift_kpi):
        """If SQL returns real PSI, use it — do NOT consult MLflow."""
        calculator_with_mlflow._execute_query = Mock(return_value=([{"avg_psi": 0.05}], None))
        # MLflow should not be consulted; if it were, this would mistakenly succeed
        # with the wrong value.
        calculator_with_mlflow._mlflow_client.get_latest_versions.side_effect = AssertionError(
            "mlflow consulted despite SQL success"
        )
        result = calculator_with_mlflow.calculate(feature_drift_kpi, {"model_name": "test"})
        assert result.value == 0.05
        assert result.error is None
        assert result.status == KPIStatus.GOOD

    def test_sql_empty_mlflow_succeeds(self, calculator_with_mlflow, feature_drift_kpi):
        """If SQL returns empty but MLflow has the metric, use MLflow value."""
        calculator_with_mlflow._execute_query = Mock(return_value=([], None))
        _stub_mlflow_returns_metric(calculator_with_mlflow, "feature_drift_psi", 0.07)
        result = calculator_with_mlflow.calculate(feature_drift_kpi, {"model_name": "test"})
        assert result.value == 0.07
        assert result.error is None
        assert result.status == KPIStatus.GOOD

    def test_sql_null_mlflow_succeeds(self, calculator_with_mlflow, feature_drift_kpi):
        """SQL row with avg_psi IS NULL — fall through to MLflow leg."""
        calculator_with_mlflow._execute_query = Mock(return_value=([{"avg_psi": None}], None))
        _stub_mlflow_returns_metric(calculator_with_mlflow, "feature_drift_psi", 0.12)
        result = calculator_with_mlflow.calculate(feature_drift_kpi, {"model_name": "test"})
        assert result.value == 0.12
        assert result.error is None
        # 0.12 in lower-is-better (target=0.10, warning=0.25) -> WARNING
        assert result.status == KPIStatus.WARNING

    def test_both_legs_fail_db_error_propagates(self, calculator_with_mlflow, feature_drift_kpi):
        """SQL returns (None, error) (query failed) AND MLflow has no metric
        — chained unavailability must surface honestly."""
        calculator_with_mlflow._execute_query = Mock(
            return_value=(None, "OperationalError:conn refused")
        )
        _stub_mlflow_run_missing_metric(calculator_with_mlflow)
        result = calculator_with_mlflow.calculate(feature_drift_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error is not None
        # Must reflect that BOTH legs were unavailable
        assert "feature_drift_psi" in result.error or "db_query_failed" in result.error
        assert "sql_leg" in result.error
        assert "mlflow_leg" in result.error
        assert result.status == KPIStatus.UNKNOWN

    def test_both_legs_fail_mlflow_unavailable(
        self, calculator_no_mlflow, feature_drift_kpi, monkeypatch
    ):
        """SQL returns empty AND MLflow client unavailable."""
        monkeypatch.setattr(
            ModelPerformanceCalculator, "mlflow_client", property(lambda self: None)
        )
        calculator_no_mlflow._execute_query = Mock(return_value=([], None))
        result = calculator_no_mlflow.calculate(feature_drift_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error is not None
        assert result.status == KPIStatus.UNKNOWN

    def test_both_legs_fail_mlflow_exception(self, calculator_with_mlflow, feature_drift_kpi):
        """SQL empty AND MLflow raises — both legs fail-closed."""
        calculator_with_mlflow._execute_query = Mock(return_value=([], None))
        _stub_mlflow_raises(calculator_with_mlflow, RuntimeError("net"))
        result = calculator_with_mlflow.calculate(feature_drift_kpi, {"model_name": "test"})
        assert result.value is None
        assert result.error is not None
        assert result.status == KPIStatus.UNKNOWN


# ===========================================================================
# Guardrails: no plausible-default magic numbers remain in caller bodies
# ===========================================================================


class TestNoPlausibleDefaultLeakage:
    """Guard: when MLflow is unavailable, no calculator returns 0.0/0.5/1.0/0.25/0.1.

    This catches the LABEL-disguised-as-REWIRE anti-pattern where someone
    might shadow `default=0.5` with `value=0.5` somewhere else in the chain.
    """

    @pytest.mark.parametrize(
        "kpi_fixture",
        [
            "roc_auc_kpi",
            "pr_auc_kpi",
            "f1_score_kpi",
            "recall_at_k_kpi",
            "brier_score_kpi",
            "calibration_slope_kpi",
        ],
    )
    def test_mlflow_unavailable_never_returns_plausible_default(
        self, calculator_no_mlflow, kpi_fixture, request, monkeypatch
    ):
        """Across all MLflow-backed calculators, MLflow-unavailable must yield value=None."""
        monkeypatch.setattr(
            ModelPerformanceCalculator, "mlflow_client", property(lambda self: None)
        )
        kpi = request.getfixturevalue(kpi_fixture)
        result = calculator_no_mlflow.calculate(kpi, {"model_name": "test"})
        assert result.value is None, (
            f"Calculator for {kpi.id} leaked a plausible default ({result.value}) "
            f"when MLflow was unavailable — REWIRE must yield None, not 0.0/0.5/1.0."
        )
        assert result.error is not None
        assert result.status == KPIStatus.UNKNOWN


# ===========================================================================
# Guardrail: _evaluate_status (UNKNOWN flow) unchanged
# ===========================================================================


class TestEvaluateStatusUnchanged:
    """`_evaluate_status` is the existing fail-close primitive: value=None -> UNKNOWN.

    This PR must NOT change that behavior. Test it directly to lock the contract.
    """

    def test_none_value_returns_unknown(self, calculator_with_mlflow, roc_auc_kpi):
        status = calculator_with_mlflow._evaluate_status(roc_auc_kpi, None, lower_is_better=False)
        assert status == KPIStatus.UNKNOWN

    def test_none_threshold_returns_unknown(self, calculator_with_mlflow):
        kpi = KPIMetadata(
            id="WS1-MP-001",
            name="ROC-AUC",
            definition="x",
            formula="x",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.WS1_MODEL_PERFORMANCE,
            threshold=None,
        )
        status = calculator_with_mlflow._evaluate_status(kpi, 0.85, lower_is_better=False)
        assert status == KPIStatus.UNKNOWN

    def test_real_value_evaluated_against_threshold(self, calculator_with_mlflow, roc_auc_kpi):
        status = calculator_with_mlflow._evaluate_status(roc_auc_kpi, 0.85, lower_is_better=False)
        assert status == KPIStatus.GOOD
