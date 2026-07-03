"""Tests for KPI models."""

import pytest
from pydantic import ValidationError

from src.kpi.models import (
    CalculationType,
    CausalLibrary,
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

    def test_lower_is_better_no_gap_between_warning_and_critical(self):
        """WS2-TR-007-shaped thresholds: (warning, critical] is CRITICAL, not a gap.

        Regression guard for #1126: the docs once left (21, 30] undefined for
        Lead Time (target=14, warning=21, critical=30). In lower-is-better mode
        the evaluator uses only ``target`` and ``warning`` — the configured
        ``critical`` is a declared ceiling, never a distinct band — so any value
        above the warning bound is CRITICAL.
        """
        threshold = KPIThreshold(target=14, warning=21, critical=30)

        assert threshold.evaluate(3, lower_is_better=True) == KPIStatus.GOOD
        assert threshold.evaluate(14, lower_is_better=True) == KPIStatus.GOOD
        assert threshold.evaluate(16, lower_is_better=True) == KPIStatus.WARNING
        assert threshold.evaluate(21, lower_is_better=True) == KPIStatus.WARNING
        # The once-undocumented (21, 30] region: CRITICAL, not undefined.
        assert threshold.evaluate(25, lower_is_better=True) == KPIStatus.CRITICAL
        assert threshold.evaluate(30, lower_is_better=True) == KPIStatus.CRITICAL
        assert threshold.evaluate(31, lower_is_better=True) == KPIStatus.CRITICAL

    def test_evaluate_none_value(self):
        """Test threshold evaluation with None value."""
        threshold = KPIThreshold(target=0.80, warning=0.70, critical=0.60)
        assert threshold.evaluate(None) == KPIStatus.UNKNOWN

    def test_evaluate_no_target(self):
        """No target defined -> INFORMATIONAL (tracked, no target by design)."""
        threshold = KPIThreshold()
        assert threshold.evaluate(0.85) == KPIStatus.INFORMATIONAL


class TestKPIThresholdBand:
    """Band mode (#1117): ideal value with symmetric tolerance bands.

    For deviation-from-ideal metrics (e.g. WS1-MP-006 calibration slope,
    ideal exactly 1.0) both directions away from ``ideal`` are worse, so
    status derives from ``abs(value - ideal)`` — never from direction.
    """

    def _band(self) -> KPIThreshold:
        return KPIThreshold(ideal=1.0, good_tolerance=0.05, warning_tolerance=0.15)

    def test_within_good_tolerance_is_good(self):
        band = self._band()
        assert band.evaluate(1.0) == KPIStatus.GOOD
        assert band.evaluate(0.9709) == KPIStatus.GOOD  # live WS1-MP-006 value
        assert band.evaluate(1.05) == KPIStatus.GOOD  # inclusive boundary
        assert band.evaluate(0.95) == KPIStatus.GOOD  # inclusive boundary (below)

    def test_within_warning_tolerance_is_warning(self):
        band = self._band()
        assert band.evaluate(1.12) == KPIStatus.WARNING
        assert band.evaluate(0.88) == KPIStatus.WARNING
        assert band.evaluate(1.15) == KPIStatus.WARNING  # inclusive boundary

    def test_beyond_warning_tolerance_is_critical_both_directions(self):
        band = self._band()
        assert band.evaluate(1.5) == KPIStatus.CRITICAL  # over-dispersed
        assert band.evaluate(0.5) == KPIStatus.CRITICAL  # over-confident
        assert band.evaluate(1.16) == KPIStatus.CRITICAL

    def test_none_value_is_unknown_fail_closed(self):
        """The #439/#1114 fail-closed primitive is preserved in band mode."""
        assert self._band().evaluate(None) == KPIStatus.UNKNOWN

    def test_lower_is_better_flag_ignored_in_band_mode(self):
        """The band is direction-symmetric; the monotone flag must not apply."""
        band = self._band()
        assert band.evaluate(0.9709, lower_is_better=True) == KPIStatus.GOOD
        assert band.evaluate(1.5, lower_is_better=True) == KPIStatus.CRITICAL

    def test_missing_warning_tolerance_caps_at_warning(self):
        """Mirrors monotone semantics: missing outer bound -> WARNING, never CRITICAL."""
        band = KPIThreshold(ideal=1.0, good_tolerance=0.05)
        assert band.evaluate(1.04) == KPIStatus.GOOD
        assert band.evaluate(2.0) == KPIStatus.WARNING

    # ---- config validation: a malformed band must fail loudly at load ----

    def test_ideal_requires_good_tolerance(self):
        with pytest.raises(ValidationError):
            KPIThreshold(ideal=1.0)

    def test_band_and_monotone_modes_are_mutually_exclusive(self):
        with pytest.raises(ValidationError):
            KPIThreshold(ideal=1.0, good_tolerance=0.05, target=1.0)

    def test_tolerances_require_ideal(self):
        with pytest.raises(ValidationError):
            KPIThreshold(good_tolerance=0.05)

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValidationError):
            KPIThreshold(ideal=1.0, good_tolerance=-0.05)

    def test_warning_tolerance_must_contain_good_tolerance(self):
        with pytest.raises(ValidationError):
            KPIThreshold(ideal=1.0, good_tolerance=0.15, warning_tolerance=0.05)


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
        assert KPIStatus.INFORMATIONAL.value == "informational"
