"""WS1-MP-006 band-threshold wiring (#1117).

Calibration slope is a deviation-from-1.0 metric: ideal is exactly 1.0 and
both directions away are worse. The old monotone higher-is-better threshold
(target=1.0) made the near-ideal live value 0.9709 read WARNING while a
badly over-dispersed 1.5 would read GOOD. These tests pin the band wiring
end-to-end: YAML -> registry -> KPIThreshold.evaluate, plus the two Python
call sites that evaluate statuses themselves (model-performance calculator
and the history backfill).
"""

from unittest.mock import Mock

import pytest

from src.kpi.calculators.model_performance import ModelPerformanceCalculator
from src.kpi.history_backfill import _status_for
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIStatus,
    KPIThreshold,
    Workstream,
)
from src.kpi.registry import KPIRegistry


def _band_kpi() -> KPIMetadata:
    return KPIMetadata(
        id="WS1-MP-006",
        name="Calibration Slope",
        definition="Reliability diagram slope",
        formula="logistic_regression(y ~ predicted_prob).slope",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_MODEL_PERFORMANCE,
        threshold=KPIThreshold(ideal=1.0, good_tolerance=0.05, warning_tolerance=0.15),
    )


class TestRegistryLoadsBandThreshold:
    """config/kpi_definitions.yaml expresses WS1-MP-006 as a band."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        KPIRegistry.reset()
        yield
        KPIRegistry.reset()

    def test_mp006_loaded_as_band(self):
        kpi = KPIRegistry().get("WS1-MP-006")
        assert kpi is not None
        assert kpi.threshold is not None
        assert kpi.threshold.ideal == 1.0
        assert kpi.threshold.good_tolerance == 0.05
        assert kpi.threshold.warning_tolerance == 0.15
        # Band mode replaces the monotone target entirely.
        assert kpi.threshold.target is None

    def test_mp006_end_to_end_evaluation(self):
        kpi = KPIRegistry().get("WS1-MP-006")
        assert kpi is not None and kpi.threshold is not None
        assert kpi.threshold.evaluate(0.9709) == KPIStatus.GOOD
        assert kpi.threshold.evaluate(1.12) == KPIStatus.WARNING
        assert kpi.threshold.evaluate(1.5) == KPIStatus.CRITICAL
        assert kpi.threshold.evaluate(None) == KPIStatus.UNKNOWN


class TestCalculatorBandStatus:
    """ModelPerformanceCalculator._evaluate_status routes bands correctly."""

    def test_band_statuses(self):
        calc = ModelPerformanceCalculator(db_client=Mock())
        kpi = _band_kpi()
        assert calc._evaluate_status(kpi, 0.9709) == KPIStatus.GOOD
        assert calc._evaluate_status(kpi, 1.12) == KPIStatus.WARNING
        assert calc._evaluate_status(kpi, 1.5) == KPIStatus.CRITICAL

    def test_band_value_none_stays_unknown(self):
        """The #439 fail-closed primitive is unchanged for band KPIs."""
        calc = ModelPerformanceCalculator(db_client=Mock())
        assert calc._evaluate_status(_band_kpi(), None) == KPIStatus.UNKNOWN


class TestHistoryBackfillBandStatus:
    """history_backfill._status_for delegates band evaluation consistently."""

    def test_band_statuses(self):
        kpi = _band_kpi()
        assert _status_for(kpi, 0.9709) == "good"
        assert _status_for(kpi, 1.12) == "warning"
        assert _status_for(kpi, 1.5) == "critical"
