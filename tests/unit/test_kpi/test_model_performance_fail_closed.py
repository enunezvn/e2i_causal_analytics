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

import os
import socket
import threading
import time
from typing import Any
from unittest.mock import Mock

import pytest

from src.kpi.calculators import model_performance as mp
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
        name="Calibration Slope Deviation",
        definition="Reliability diagram slope deviation fold",
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
    """WS1-MP-006 Calibration Slope Deviation: 5 unavailability paths."""

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
# #1650: the MLflow leg must fail CLOSED in seconds, not hang
# ===========================================================================

# Hard ceiling for the wall-clock probes below. The leg is exercised on a
# daemon thread and joined with this bound, so a REGRESSED (unbounded) leg
# fails the test at ~this mark instead of running until pytest-timeout kills
# the xdist worker — which is the #1648 "node down" shape that #1650 caused
# three times on PR #1643. Comfortably under the 30s global `timeout` in
# pyproject.toml, and comfortably over the ~6s worst case the fix allows.
_PROBE_CEILING_SECONDS = 20.0


def _dead_port() -> int:
    """A localhost port with nothing listening -> connection REFUSED.

    This is the faithful CI shape: the lanes export
    ``MLFLOW_TRACKING_URI: http://localhost:5000`` but only start a real
    tracking server in a DIFFERENT job, so the URI is populated and the
    endpoint is dead.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _silent_server_port() -> tuple[int, socket.socket]:
    """A socket that ACCEPTS and then never writes a byte -> read hangs.

    Distinct from `_dead_port`: connection-refused fails fast per attempt (so
    only the retry/backoff cap binds), whereas a silent peer hangs per attempt
    (so the connect/read timeout is the binding term). Both must be bounded.
    """
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    held: list[socket.socket] = []

    def _accept_forever() -> None:
        while True:
            try:
                conn, _ = srv.accept()
                held.append(conn)  # hold open, never respond
            except OSError:
                return

    threading.Thread(target=_accept_forever, daemon=True).start()
    return int(srv.getsockname()[1]), srv


def _time_the_leg(tracking_uri: str) -> tuple[float, Any]:
    """Call the real MLflow leg against `tracking_uri`, hard-bounded.

    Runs on a daemon thread joined at `_PROBE_CEILING_SECONDS`. Returns
    `(elapsed, (value, error))`, or raises AssertionError if the leg is still
    running at the ceiling — i.e. it is unbounded, which is the #1650 defect.
    """
    import mlflow

    calc = ModelPerformanceCalculator(db_client=Mock())
    calc._mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    out: dict[str, Any] = {}

    def _work() -> None:
        started = time.monotonic()
        out["result"] = calc._get_metric_from_mlflow("default_model", "roc_auc")
        out["elapsed"] = time.monotonic() - started

    worker = threading.Thread(target=_work, daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(_PROBE_CEILING_SECONDS)
    if worker.is_alive():
        raise AssertionError(
            f"#1650: the MLflow leg was STILL RUNNING after "
            f"{time.monotonic() - started:.1f}s against an unreachable tracking "
            f"server at {tracking_uri}. Fail-closed must take seconds, not "
            f"minutes — an unbounded leg is a hang, not a refusal."
        )
    return float(out["elapsed"]), out["result"]


class TestMlflowLegIsTimeBounded:
    """WS1-MP-001/002/003 (#1650): an unreachable tracking server must resolve
    to `value=None` + an `mlflow_exception:` reason in SECONDS.

    The leg already refuses to fabricate a value — that property is asserted
    here too and must never be traded away. What was broken is how LONG the
    refusal took to establish: mlflow's REST store retries every request with
    urllib3 exponential backoff (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=7`,
    `BACKOFF_FACTOR=2`, `TIMEOUT=120` by default), which is ~126s of sleep per
    endpoint. Measured on the pre-fix tree: still running at 75s with no
    result. None of these tests need a live MLflow server.
    """

    def test_budget_stays_small_enough_to_be_a_refusal(self):
        """Forcing function: nobody may quietly restore mlflow's own defaults.

        mlflow ships `max_retries=7, backoff_factor=2` precisely so a rate-
        limited backend is retried for ~4 minutes. That is the right default
        for the `mlflow_tracker` WRITE paths and the wrong one for a KPI READ
        that a user's question is waiting on.
        """
        assert mp.MLFLOW_LEG_MAX_RETRIES <= 2
        assert mp.MLFLOW_LEG_TIMEOUT_SECONDS <= 5
        assert (
            mp.MLFLOW_LEG_WORST_CASE_SECONDS
            == (mp.MLFLOW_LEG_MAX_RETRIES + 1) * mp.MLFLOW_LEG_TIMEOUT_SECONDS
        )
        assert mp.MLFLOW_LEG_WORST_CASE_SECONDS <= 10, (
            "The MLflow leg's worst case must stay well inside a request "
            "budget and inside the 30s pytest timeout (#1650/#1648)."
        )

    def test_refused_connection_fails_closed_fast(self):
        """Dead localhost port — the exact CI shape. Pre-fix: >75s."""
        elapsed, (value, error) = _time_the_leg(f"http://127.0.0.1:{_dead_port()}")
        assert value is None, "no-fabrication: an unreachable server must not yield a value"
        assert error is not None and error.startswith("mlflow_exception:"), error
        assert elapsed < mp.MLFLOW_LEG_WORST_CASE_SECONDS + 4.0, (
            f"refused-connection leg took {elapsed:.2f}s, over the "
            f"{mp.MLFLOW_LEG_WORST_CASE_SECONDS}s budget (#1650)"
        )

    def test_silent_server_fails_closed_fast(self):
        """Peer accepts but never replies — the connect/read timeout binds here.

        A retry cap alone does NOT bound this mode; without a short timeout
        each attempt sits for `MLFLOW_HTTP_REQUEST_TIMEOUT` (120s default).
        """
        port, srv = _silent_server_port()
        try:
            elapsed, (value, error) = _time_the_leg(f"http://127.0.0.1:{port}")
        finally:
            srv.close()
        assert value is None, "no-fabrication: a silent server must not yield a value"
        assert error is not None and error.startswith("mlflow_exception:"), error
        assert elapsed < mp.MLFLOW_LEG_WORST_CASE_SECONDS + 4.0, (
            f"silent-server leg took {elapsed:.2f}s, over the "
            f"{mp.MLFLOW_LEG_WORST_CASE_SECONDS}s budget (#1650)"
        )

    def test_bound_is_in_force_at_the_moment_mlflow_is_called(self):
        """The bound must be live WHEN we call in — not merely defined.

        Reads the knobs back through mlflow's own environment-variable
        accessors (the same objects `mlflow.utils.rest_utils.http_request`
        consults at call time), so this pins the real seam rather than our
        own bookkeeping. No sockets involved.
        """
        from mlflow.environment_variables import (
            MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
            MLFLOW_HTTP_REQUEST_MAX_RETRIES,
            MLFLOW_HTTP_REQUEST_TIMEOUT,
        )

        seen: dict[str, Any] = {}

        def _observe(*_args: Any, **_kwargs: Any):
            seen["max_retries"] = MLFLOW_HTTP_REQUEST_MAX_RETRIES.get()
            seen["timeout"] = MLFLOW_HTTP_REQUEST_TIMEOUT.get()
            seen["backoff_factor"] = MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR.get()
            return []

        calc = ModelPerformanceCalculator(db_client=Mock(), mlflow_client=Mock())
        calc._mlflow_client.get_latest_versions.side_effect = _observe

        value, error = calc._get_metric_from_mlflow("default_model", "roc_auc")

        assert value is None and error == "model_not_found:default_model"
        assert seen["max_retries"] == mp.MLFLOW_LEG_MAX_RETRIES
        assert seen["timeout"] == mp.MLFLOW_LEG_TIMEOUT_SECONDS
        assert seen["backoff_factor"] == 0, "exponential backoff must be off for this read"

    @pytest.mark.parametrize("raises", [False, True])
    def test_bound_does_not_leak_into_the_rest_of_the_process(self, raises, monkeypatch):
        """The knobs are process-global, so the scope must restore them.

        The `mlflow_tracker` WRITE paths (`src/agents/*/mlflow_tracker.py`)
        legitimately want mlflow's generous retries; this KPI read must not
        quietly shorten them. Checked on both the clean and the raising path.
        """
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "7")
        monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)

        calc = ModelPerformanceCalculator(db_client=Mock(), mlflow_client=Mock())
        if raises:
            calc._mlflow_client.get_latest_versions.side_effect = RuntimeError("boom")
        else:
            calc._mlflow_client.get_latest_versions.return_value = []

        calc._get_metric_from_mlflow("default_model", "roc_auc")

        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "7"
        assert "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ


# ===========================================================================
# #1658: a client that cannot be CONSTRUCTED must land in the taxonomy too
# ===========================================================================

# A tracking URI whose scheme is parseable but not registered with mlflow's
# store registry. This is a ONE-CHARACTER typo of the value prod actually ships
# (`docker/docker-compose.yml` x-common-env: `MLFLOW_TRACKING_URI:
# http://mlflow:5000`), which is why it is the faithful shape rather than an
# exotic one: measured, `mlflow.tracking.MlflowClient()` raises
# `UnsupportedModelRegistryStoreURIException` for it in ~0.00s, at CONSTRUCTION,
# before any request is made. `except ImportError` cannot see it.
_UNSUPPORTED_SCHEME_URI = "htp://mlflow:5000"

# Same family, different failure site: the scheme IS registered but the store
# cannot be opened. Raises `FileNotFoundError` out of mlflow's sqlite store.
_UNOPENABLE_STORE_URI = "sqlite:////proc/nope/cannot/exist.db"


class TestMlflowClientConstructionFailsClosed:
    """WS1-MP-001/002/003 (#1658): MISCONFIGURED must be classified, not raised.

    `mlflow_client` used to wrap two statements in one `try ... except
    ImportError`, but only `import mlflow` can raise `ImportError`. Every way
    `MlflowClient()` itself can fail — unsupported/typo'd `MLFLOW_TRACKING_URI`,
    an unopenable backing store, a bad credential — escaped a *property access*
    and bypassed the module's whole error taxonomy.

    ABSENT was handled (ImportError -> None -> `mlflow_client_unavailable`).
    MISCONFIGURED was not. These tests pin the gap shut using REAL
    configuration: the constructor is never patched to raise, the URI makes it
    raise on its own.
    """

    def test_property_returns_none_instead_of_raising(self, monkeypatch):
        """A property access must never explode. Pre-fix: raises."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", _UNSUPPORTED_SCHEME_URI)
        calc = ModelPerformanceCalculator(db_client=Mock())
        assert calc.mlflow_client is None, (
            "a construction failure must fail CLOSED to None, the same shape as "
            "the absent-mlflow case, not propagate out of attribute access"
        )

    def test_leg_honours_its_value_error_contract(self, monkeypatch):
        """`_get_metric_from_mlflow` documents a `(value, error)` return.

        Pre-fix it RAISED here instead, so the documented contract was a lie
        for exactly one input: a misconfigured tracking URI.
        """
        monkeypatch.setenv("MLFLOW_TRACKING_URI", _UNSUPPORTED_SCHEME_URI)
        calc = ModelPerformanceCalculator(db_client=Mock())
        value, error = calc._get_metric_from_mlflow("default_model", "pr_auc")
        assert value is None, "no-fabrication: a broken client must not yield a value"
        assert error is not None and error.startswith("mlflow_exception:"), error
        assert "UnsupportedModelRegistryStoreURIException" in error, error

    def test_unopenable_store_is_classified_too(self, monkeypatch):
        """Different failure site (store open, not scheme lookup), same taxonomy."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", _UNOPENABLE_STORE_URI)
        calc = ModelPerformanceCalculator(db_client=Mock())
        value, error = calc._get_metric_from_mlflow("default_model", "pr_auc")
        assert value is None
        assert error is not None and error.startswith("mlflow_exception:"), error
        assert "FileNotFoundError" in error, error

    def test_constructor_importerror_is_config_not_absence(self, monkeypatch):
        """The two legs are told apart by WHICH STATEMENT failed, not by type.

        `MlflowClient()` can itself raise an `ImportError` SUBCLASS: measured,
        `mssql://…` raises `ModuleNotFoundError: No module named 'pyodbc'` in
        ~0.03s. A missing DB driver is a configuration problem — mlflow is
        installed and fine. Collapsing the property's two `try`s into a single
        `except ImportError` would file it under `mlflow_client_unavailable`
        ("no MLflow here") and point the operator at the wrong thing. This test
        is the tripwire on that simplification.
        """
        try:
            import pyodbc  # noqa: F401
        except ImportError:
            pass
        else:  # pragma: no cover - no driver is pinned anywhere in this repo
            pytest.skip("pyodbc is installed; this URI would attempt a real connection")

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "mssql://u:p@127.0.0.1:1/db")
        calc = ModelPerformanceCalculator(db_client=Mock())
        value, error = calc._get_metric_from_mlflow("default_model", "pr_auc")
        assert value is None
        assert error != "mlflow_client_unavailable", (
            "a missing DB driver is MISCONFIGURED, not ABSENT — mlflow imported fine"
        )
        assert error is not None and error.startswith("mlflow_exception:ModuleNotFoundError"), error

    def test_misconfigured_is_distinguishable_from_absent(self, monkeypatch):
        """The point of the taxonomy is that these two need different fixes.

        `mlflow_client_unavailable` means "no MLflow here" — nothing to do.
        A construction failure means "your MLFLOW_TRACKING_URI is wrong" — an
        operator action. Collapsing the second into the first would stop the
        crash while destroying the diagnosis, which is a labelling fix, not a
        functional one.
        """
        monkeypatch.setenv("MLFLOW_TRACKING_URI", _UNSUPPORTED_SCHEME_URI)
        calc = ModelPerformanceCalculator(db_client=Mock())
        _, error = calc._get_metric_from_mlflow("default_model", "pr_auc")
        assert error != "mlflow_client_unavailable"
        assert _UNSUPPORTED_SCHEME_URI in (error or ""), (
            "the reason must carry enough detail to name the offending URI"
        )

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
    def test_calculate_returns_a_typed_unknown_not_a_raw_traceback(
        self, kpi_fixture, request, monkeypatch
    ):
        """End to end: every MLflow-backed KPI must reach UNKNOWN with a typed reason.

        `calculate()`'s blanket `except Exception` already stopped a raw
        exception reaching the caller, so `value=None`/UNKNOWN held even pre-fix
        — the no-fabrication property was never the thing at risk. What leaked
        was CLASSIFICATION: `error` was the bare exception message with no
        taxonomy prefix, indistinguishable to a consumer from any other crash.
        """
        monkeypatch.setenv("MLFLOW_TRACKING_URI", _UNSUPPORTED_SCHEME_URI)
        db = Mock()
        db.rpc.return_value.execute.return_value.data = []
        calc = ModelPerformanceCalculator(db_client=db)
        result = calc.calculate(request.getfixturevalue(kpi_fixture), {"model_name": "test"})
        assert result.value is None
        assert result.status == KPIStatus.UNKNOWN
        assert result.error is not None
        assert result.error.startswith("mlflow_exception:"), (
            f"construction failure escaped the taxonomy: {result.error!r}"
        )

    def test_feature_drift_keeps_its_two_leg_message(self, monkeypatch):
        """WS1-MP-009 composes both legs into one reason — pre-fix the raise
        unwound past that composition and the SQL leg's outcome was lost."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", _UNSUPPORTED_SCHEME_URI)
        calc = ModelPerformanceCalculator(db_client=Mock())
        calc._execute_query = Mock(return_value=([], None))
        kpi = KPIMetadata(
            id="WS1-MP-009",
            name="Feature Drift (PSI)",
            definition="Population stability index",
            formula="custom",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.WS1_MODEL_PERFORMANCE,
            threshold=KPIThreshold(target=0.10, warning=0.25, critical=0.50),
        )
        result = calc.calculate(kpi, {"model_name": "test"})
        assert result.value is None
        assert result.status == KPIStatus.UNKNOWN
        assert result.error is not None
        assert "sql_leg" in result.error and "mlflow_leg" in result.error, result.error
        assert "mlflow_exception:" in result.error, result.error

    def test_construction_is_attempted_once_per_calculator(self, monkeypatch):
        """The failure must be CACHED, or a bounded leg becomes an unbounded one.

        Construction cost is NOT uniform. Measured against mlflow 3.11.1:
          * unsupported scheme      -> raises in ~0.00s
          * unopenable sqlite store -> raises in ~0.00s (after import)
          * unreachable DB-backed tracking URI (`postgresql://...` on a dead
            port) -> raises after ~102s, because mlflow's
            `create_sqlalchemy_engine_with_retry` retries with 0.1*(2**n - 1)
            backoff and `MAX_RETRY_COUNT` is a module constant with no env knob.

        `_bounded_mlflow_http` does NOT cover that: it scopes
        `MLFLOW_HTTP_REQUEST_*`, which only the REST store reads. So a
        construction failure swallowed but NOT cached would be re-paid once per
        MLflow-backed KPI — 7 of them in a WS1 model-performance grid — turning
        one ~102s failure into ~12 minutes of "fail-closed". That is the #1650
        hang shape reintroduced through the door #1658 opens.

        Caching is safe precisely because the lifetime is short:
        `get_kpi_calculator()` in `src/api/routes/kpi.py` is a plain
        `Depends(...)` with no `lru_cache`, so a calculator lives for ONE
        request and a corrected config is picked up on the next one.

        The spy below only COUNTS; it delegates to the real constructor, and the
        real (broken) URI is what makes that constructor raise.
        """
        import mlflow

        monkeypatch.setenv("MLFLOW_TRACKING_URI", _UNSUPPORTED_SCHEME_URI)
        real_client_cls = mlflow.tracking.MlflowClient
        attempts = {"n": 0}

        def _counting_client(*args: Any, **kwargs: Any):
            attempts["n"] += 1
            return real_client_cls(*args, **kwargs)

        monkeypatch.setattr(mlflow.tracking, "MlflowClient", _counting_client)

        db = Mock()
        db.rpc.return_value.execute.return_value.data = []
        calc = ModelPerformanceCalculator(db_client=db)
        for metric in ("roc_auc", "pr_auc", "f1_score", "brier_score", "calibration_slope"):
            value, error = calc._get_metric_from_mlflow("default_model", metric)
            assert value is None
            assert error is not None and error.startswith("mlflow_exception:"), error

        assert attempts["n"] == 1, (
            f"MlflowClient() was constructed {attempts['n']}x across 5 KPI reads on "
            "one calculator; a failed construction must be remembered for the life "
            "of the instance (#1658/#1650)"
        )

    def test_reachable_but_dead_endpoint_still_takes_the_call_path(self):
        """Regression guard: do NOT collapse 'unreachable' into 'misconfigured'.

        Measured: `http://127.0.0.1:<dead port>` CONSTRUCTS fine — an HTTP
        tracking URI touches no network at construction time. It must keep
        failing where it always did, inside the #1650-bounded call, so the two
        remain separately diagnosable.
        """
        import mlflow

        uri = f"http://127.0.0.1:{_dead_port()}"
        calc = ModelPerformanceCalculator(db_client=Mock())
        calc._mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=uri)  # constructs fine
        value, error = calc._get_metric_from_mlflow("default_model", "roc_auc")
        assert value is None
        assert error is not None and error.startswith("mlflow_exception:MlflowException"), error

    def test_absent_mlflow_still_reads_as_client_unavailable(self, monkeypatch):
        """The ImportError leg keeps its own distinct reason (unchanged)."""
        monkeypatch.setattr(
            ModelPerformanceCalculator, "mlflow_client", property(lambda self: None)
        )
        calc = ModelPerformanceCalculator(db_client=Mock())
        value, error = calc._get_metric_from_mlflow("default_model", "roc_auc")
        assert value is None
        assert error == "mlflow_client_unavailable"


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

    The fail-closed lock is on the VALUE branch: a missing value must never
    fabricate a health status. A missing THRESHOLD with a real value is not a
    failure — it is a no-target-by-design KPI and reads INFORMATIONAL (still
    never GOOD/WARNING/CRITICAL, so the anti-fabrication contract holds).
    """

    def test_none_value_returns_unknown(self, calculator_with_mlflow, roc_auc_kpi):
        status = calculator_with_mlflow._evaluate_status(roc_auc_kpi, None, lower_is_better=False)
        assert status == KPIStatus.UNKNOWN

    def test_none_threshold_returns_informational(self, calculator_with_mlflow):
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
        assert status == KPIStatus.INFORMATIONAL

    def test_real_value_evaluated_against_threshold(self, calculator_with_mlflow, roc_auc_kpi):
        status = calculator_with_mlflow._evaluate_status(roc_auc_kpi, 0.85, lower_is_better=False)
        assert status == KPIStatus.GOOD
