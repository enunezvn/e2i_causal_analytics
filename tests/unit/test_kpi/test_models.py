"""Tests for KPI models."""

import pytest
from datetime import datetime

from src.kpi.models import (
    CausalLibrary,
    CalculationType,
    KPIBatchResult,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    KPIThreshold,
    Workstream,
)


class TestKPIThreshold:
    """Tests for KPIThreshold evaluation."""

    def test_evaluate_good_higher_is_better(self):
        """Test threshold evaluation when higher values are better."""
        threshold = KPIThreshold(target=0.80, warning=0.70, critical=0.60)

        assert threshold.evaluate(0.85) == KPIStatus.GOOD
        assert threshold.evaluate(0.80) == KPIStatus.GOOD
        assert threshold.evaluate(0.75) == KPIStatus.WARNING
        assert threshold.evaluate(0.65) == KPIStatus.WARNING
        assert threshold.evaluate(0.50) == KPIStatus.CRITICAL

    def test_evaluate_good_lower_is_better(self):
        """Test threshold evaluation when lower values are better."""
        threshold = KPIThreshold(target=0.10, warning=0.20, critical=0.30)

        assert threshold.evaluate(0.05, lower_is_better=True) == KPIStatus.GOOD
        assert threshold.evaluate(0.10, lower_is_better=True) == KPIStatus.GOOD
        assert threshold.evaluate(0.15, lower_is_better=True) == KPIStatus.WARNING
        assert threshold.evaluate(0.25, lower_is_better=True) == KPIStatus.CRITICAL

    def test_evaluate_none_value(self):
        """Test threshold evaluation with None value."""
        threshold = KPIThreshold(target=0.80, warning=0.70, critical=0.60)
        assert threshold.evaluate(None) == KPIStatus.UNKNOWN

    def test_evaluate_no_target(self):
        """Test threshold evaluation with no target."""
        threshold = KPIThreshold()
        assert threshold.evaluate(0.85) == KPIStatus.UNKNOWN


class TestKPIMetadata:
    """Tests for KPIMetadata."""

    def test_create_kpi_metadata(self):
        """Test creating KPI metadata."""
        kpi = KPIMetadata(
            id="WS1-DQ-001",
            name="Source Coverage - Patients",
            definition="Percentage of eligible patients",
            formula="covered_patients / reference_patients",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS1_DATA_QUALITY,
            tables=["patient_journeys", "reference_universe"],
            threshold=KPIThreshold(target=0.85, warning=0.70, critical=0.50),
        )

        assert kpi.id == "WS1-DQ-001"
        assert kpi.name == "Source Coverage - Patients"
        assert kpi.calculation_type == CalculationType.DERIVED
        assert kpi.workstream == Workstream.WS1_DATA_QUALITY
        assert len(kpi.tables) == 2
        assert kpi.threshold is not None
        assert kpi.threshold.target == 0.85

    def test_kpi_metadata_defaults(self):
        """Test KPI metadata default values."""
        kpi = KPIMetadata(
            id="TEST-001",
            name="Test KPI",
            definition="Test definition",
            formula="test formula",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.WS1_DATA_QUALITY,
        )

        assert kpi.tables == []
        assert kpi.columns == []
        assert kpi.view is None
        assert kpi.frequency == "daily"
        assert kpi.primary_causal_library == CausalLibrary.NONE


class TestKPIResult:
    """Tests for KPIResult."""

    def test_create_kpi_result(self):
        """Test creating a KPI result."""
        result = KPIResult(
            kpi_id="WS1-DQ-001",
            value=0.87,
            status=KPIStatus.GOOD,
        )

        assert result.kpi_id == "WS1-DQ-001"
        assert result.value == 0.87
        assert result.status == KPIStatus.GOOD
        assert result.error is None
        assert result.cached is False

    def test_kpi_result_with_causal_info(self):
        """Test KPI result with causal analysis info."""
        result = KPIResult(
            kpi_id="CM-001",
            value=0.15,
            status=KPIStatus.GOOD,
            causal_library_used=CausalLibrary.DOWHY,
            confidence_interval=(0.10, 0.20),
            p_value=0.001,
            effect_size=0.15,
        )

        assert result.causal_library_used == CausalLibrary.DOWHY
        assert result.confidence_interval == (0.10, 0.20)
        assert result.p_value == 0.001
        assert result.effect_size == 0.15

    def test_kpi_result_with_error(self):
        """Test KPI result with error."""
        result = KPIResult(
            kpi_id="WS1-DQ-001",
            error="Database connection failed",
        )

        assert result.value is None
        assert result.status == KPIStatus.UNKNOWN
        assert result.error == "Database connection failed"


class TestKPIBatchResult:
    """Tests for KPIBatchResult."""

    def test_add_results(self):
        """Test adding results to batch."""
        batch = KPIBatchResult(workstream=Workstream.WS1_DATA_QUALITY)

        # Add successful result
        batch.add_result(KPIResult(kpi_id="WS1-DQ-001", value=0.85))

        # Add failed result
        batch.add_result(KPIResult(kpi_id="WS1-DQ-002", error="Failed"))

        assert batch.total_kpis == 2
        assert batch.successful == 1
        assert batch.failed == 1
        assert len(batch.results) == 2

    def test_empty_batch(self):
        """Test empty batch result."""
        batch = KPIBatchResult()

        assert batch.total_kpis == 0
        assert batch.successful == 0
        assert batch.failed == 0
        assert batch.workstream is None


class TestEnums:
    """Tests for enum values."""

    def test_causal_library_values(self):
        """Test CausalLibrary enum values."""
        assert CausalLibrary.DOWHY.value == "dowhy"
        assert CausalLibrary.ECONML.value == "econml"
        assert CausalLibrary.CAUSALML.value == "causalml"
        assert CausalLibrary.NETWORKX.value == "networkx"
        assert CausalLibrary.NONE.value == "none"

    def test_workstream_values(self):
        """Test Workstream enum values."""
        assert Workstream.WS1_DATA_QUALITY.value == "ws1_data_quality"
        assert Workstream.WS2_TRIGGERS.value == "ws2_triggers"
        assert Workstream.WS3_BUSINESS.value == "ws3_business"

    def test_kpi_status_values(self):
        """Test KPIStatus enum values."""
        assert KPIStatus.GOOD.value == "good"
        assert KPIStatus.WARNING.value == "warning"
        assert KPIStatus.CRITICAL.value == "critical"
        assert KPIStatus.UNKNOWN.value == "unknown"
