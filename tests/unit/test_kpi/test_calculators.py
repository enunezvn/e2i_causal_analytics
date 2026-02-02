"""Tests for WS1 KPI Calculators."""

from unittest.mock import Mock

import numpy as np
import pytest

from src.kpi.calculators.data_quality import DataQualityCalculator
from src.kpi.calculators.model_performance import (
    ModelPerformanceCalculator,
    calculate_psi,
)
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIStatus,
    KPIThreshold,
    Workstream,
)


class TestDataQualityCalculator:
    """Tests for DataQualityCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with mock db client."""
        mock_db = Mock()
        return DataQualityCalculator(db_client=mock_db)

    @pytest.fixture
    def sample_kpi(self):
        """Create a sample data quality KPI."""
        return KPIMetadata(
            id="WS1-DQ-001",
            name="Source Coverage - Patients",
            definition="Percentage of eligible patients",
            formula="covered_patients / reference_patients",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS1_DATA_QUALITY,
            threshold=KPIThreshold(target=0.85, warning=0.70, critical=0.50),
        )

    def test_supports_data_quality_workstream(self, calculator, sample_kpi):
        """Test calculator supports WS1 Data Quality KPIs."""
        assert calculator.supports(sample_kpi) is True

    def test_does_not_support_other_workstreams(self, calculator):
        """Test calculator doesn't support other workstreams."""
        kpi = KPIMetadata(
            id="WS2-TR-001",
            name="Trigger Precision",
            definition="Test",
            formula="test",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS2_TRIGGERS,
        )
        assert calculator.supports(kpi) is False

    def test_calculate_returns_error_for_unknown_kpi(self, calculator):
        """Test calculator returns error for unknown KPI ID."""
        kpi = KPIMetadata(
            id="WS1-DQ-999",
            name="Unknown KPI",
            definition="Test",
            formula="test",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS1_DATA_QUALITY,
        )
        result = calculator.calculate(kpi)
        assert result.error is not None
        assert "No calculator implemented" in result.error

    def test_calculate_source_coverage_success(self, calculator, sample_kpi):
        """Test source coverage calculation with mock data."""
        # Mock the execute_query to return coverage data
        calculator._execute_query = Mock(return_value=[{"covered": 850, "total": 1000}])

        result = calculator.calculate(sample_kpi)

        assert result.value == 0.85
        assert result.status == KPIStatus.GOOD
        assert result.error is None

    def test_calculate_source_coverage_warning(self, calculator, sample_kpi):
        """Test source coverage in warning zone."""
        calculator._execute_query = Mock(return_value=[{"covered": 750, "total": 1000}])

        result = calculator.calculate(sample_kpi)

        assert result.value == 0.75
        assert result.status == KPIStatus.WARNING

    def test_calculate_source_coverage_critical(self, calculator, sample_kpi):
        """Test source coverage in critical zone."""
        calculator._execute_query = Mock(return_value=[{"covered": 400, "total": 1000}])

        result = calculator.calculate(sample_kpi)

        assert result.value == 0.4
        assert result.status == KPIStatus.CRITICAL

    def test_calculate_handles_empty_result(self, calculator, sample_kpi):
        """Test graceful handling of empty query results."""
        calculator._execute_query = Mock(return_value=[{"covered": 0, "total": 0}])

        result = calculator.calculate(sample_kpi)

        assert result.value == 0.0

    def test_calculate_handles_exception(self, calculator, sample_kpi):
        """Test graceful handling of calculation exceptions."""
        calculator._execute_query = Mock(side_effect=Exception("DB error"))

        result = calculator.calculate(sample_kpi)

        assert result.error is not None
        assert "DB error" in result.error


class TestModelPerformanceCalculator:
    """Tests for ModelPerformanceCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with mock clients."""
        mock_db = Mock()
        mock_mlflow = Mock()
        return ModelPerformanceCalculator(db_client=mock_db, mlflow_client=mock_mlflow)

    @pytest.fixture
    def roc_auc_kpi(self):
        """Create ROC-AUC KPI."""
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
    def brier_score_kpi(self):
        """Create Brier Score KPI (lower is better)."""
        return KPIMetadata(
            id="WS1-MP-005",
            name="Brier Score",
            definition="Mean squared error of probabilistic predictions",
            formula="sklearn.metrics.brier_score_loss",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.WS1_MODEL_PERFORMANCE,
            threshold=KPIThreshold(target=0.10, warning=0.20, critical=0.30),
        )

    def test_supports_model_performance_workstream(self, calculator, roc_auc_kpi):
        """Test calculator supports WS1 Model Performance KPIs."""
        assert calculator.supports(roc_auc_kpi) is True

    def test_does_not_support_other_workstreams(self, calculator):
        """Test calculator doesn't support other workstreams."""
        kpi = KPIMetadata(
            id="WS1-DQ-001",
            name="Source Coverage",
            definition="Test",
            formula="test",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS1_DATA_QUALITY,
        )
        assert calculator.supports(kpi) is False

    def test_calculate_roc_auc_from_mlflow(self, calculator, roc_auc_kpi):
        """Test ROC-AUC calculation from MLflow."""
        # Mock MLflow response
        mock_version = Mock()
        mock_version.run_id = "test-run-123"
        calculator._mlflow_client.get_latest_versions.return_value = [mock_version]

        mock_run = Mock()
        mock_run.data.metrics = {"roc_auc": 0.85}
        calculator._mlflow_client.get_run.return_value = mock_run

        result = calculator.calculate(roc_auc_kpi, {"model_name": "test_model"})

        assert result.value == 0.85
        assert result.status == KPIStatus.GOOD
        assert result.metadata.get("lower_is_better") is False

    def test_calculate_brier_score_lower_is_better(self, calculator, brier_score_kpi):
        """Test Brier Score with lower-is-better threshold evaluation."""
        # Mock MLflow response
        mock_version = Mock()
        mock_version.run_id = "test-run-123"
        calculator._mlflow_client.get_latest_versions.return_value = [mock_version]

        mock_run = Mock()
        mock_run.data.metrics = {"brier_score": 0.08}
        calculator._mlflow_client.get_run.return_value = mock_run

        result = calculator.calculate(brier_score_kpi, {"model_name": "test_model"})

        assert result.value == 0.08
        assert result.status == KPIStatus.GOOD
        assert result.metadata.get("lower_is_better") is True

    def test_calculate_brier_score_critical(self, calculator, brier_score_kpi):
        """Test Brier Score in critical zone (too high)."""
        mock_version = Mock()
        mock_version.run_id = "test-run-123"
        calculator._mlflow_client.get_latest_versions.return_value = [mock_version]

        mock_run = Mock()
        mock_run.data.metrics = {"brier_score": 0.35}
        calculator._mlflow_client.get_run.return_value = mock_run

        result = calculator.calculate(brier_score_kpi, {"model_name": "test_model"})

        assert result.value == 0.35
        assert result.status == KPIStatus.CRITICAL

    def test_calculate_returns_default_when_mlflow_unavailable(self, calculator, roc_auc_kpi):
        """Test calculation falls back to default when MLflow errors."""
        calculator._mlflow_client.get_latest_versions.side_effect = Exception("MLflow error")

        result = calculator.calculate(roc_auc_kpi)

        # Should return default value (0.5 for ROC-AUC)
        assert result.value == 0.5

    def test_calculate_shap_coverage_from_db(self, calculator):
        """Test SHAP coverage calculation from database."""
        kpi = KPIMetadata(
            id="WS1-MP-007",
            name="SHAP Coverage",
            definition="Percentage with SHAP explanations",
            formula="count(shap) / count(*)",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS1_MODEL_PERFORMANCE,
            threshold=KPIThreshold(target=0.90, warning=0.80, critical=0.50),
        )

        calculator._execute_query = Mock(return_value=[{"coverage": 0.92}])

        result = calculator.calculate(kpi)

        assert result.value == 0.92
        assert result.status == KPIStatus.GOOD


class TestPSICalculation:
    """Tests for PSI (Population Stability Index) calculation."""

    def test_psi_identical_distributions(self):
        """Test PSI is near zero for identical distributions."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)

        psi = calculate_psi(expected, actual)

        # Should be very low for similar distributions
        assert psi < 0.1

    def test_psi_shifted_distribution(self):
        """Test PSI detects shifted distributions."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)  # Mean shifted by 2

        psi = calculate_psi(expected, actual)

        # Should be high for significantly different distributions
        assert psi > 0.25

    def test_psi_scaled_distribution(self):
        """Test PSI detects variance changes."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 3, 1000)  # Higher variance

        psi = calculate_psi(expected, actual)

        # Should detect distribution change
        assert psi > 0.1

    def test_psi_returns_float(self):
        """Test PSI returns a float value."""
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        psi = calculate_psi(expected, actual)

        assert isinstance(psi, float)
