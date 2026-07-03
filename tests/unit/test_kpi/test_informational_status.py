"""INFORMATIONAL status semantics (no target defined BY DESIGN).

Eight KPIs carry no threshold on purpose (docs/data/06-KPI-REFERENCE.md,
"Volume and Causal Metrics (No Thresholds)"): WS3-BI-005/006/007 volume
metrics and CM-001..CM-005 causal metrics. Rendering them as UNKNOWN
conflated "no target by design" with "evaluation failed". The contract:

- value is None            -> UNKNOWN       (could not evaluate; fail-closed)
- value ok, no threshold   -> INFORMATIONAL (tracked for trend/context only)
- value ok, target present -> GOOD/WARNING/CRITICAL bands (unchanged)
"""

from unittest.mock import Mock

import pytest

from src.kpi.calculators.brand_specific import BrandSpecificCalculator
from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.calculators.causal_metrics import CausalMetricsCalculator
from src.kpi.calculators.data_quality import DataQualityCalculator
from src.kpi.calculators.model_performance import ModelPerformanceCalculator
from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator
from src.kpi.history_backfill import _status_for
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIStatus,
    KPIThreshold,
    Workstream,
)

THRESHOLDED_CALCULATORS = [
    DataQualityCalculator,
    ModelPerformanceCalculator,
    TriggerPerformanceCalculator,
    BusinessImpactCalculator,
    BrandSpecificCalculator,
]


def _kpi(threshold: KPIThreshold | None) -> KPIMetadata:
    return KPIMetadata(
        id="WS3-BI-005",
        name="TRx Volume",
        definition="Total prescriptions",
        formula="count(prescriptions)",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS3_BUSINESS,
        threshold=threshold,
    )


class TestThresholdEvaluate:
    """KPIThreshold.evaluate distinguishes no-target from no-value."""

    def test_no_target_with_value_is_informational(self):
        assert KPIThreshold().evaluate(0.85) == KPIStatus.INFORMATIONAL

    def test_no_target_no_value_is_unknown(self):
        """value None wins: fail-closed UNKNOWN even without a target."""
        assert KPIThreshold().evaluate(None) == KPIStatus.UNKNOWN


class TestCalculatorEvaluateStatus:
    """Every calculator's _evaluate_status honours the 3-way contract."""

    @pytest.mark.parametrize("calc_cls", THRESHOLDED_CALCULATORS)
    def test_no_threshold_with_value_is_informational(self, calc_cls):
        calc = calc_cls(db_client=Mock())
        assert calc._evaluate_status(_kpi(None), 42.0) == KPIStatus.INFORMATIONAL

    @pytest.mark.parametrize("calc_cls", THRESHOLDED_CALCULATORS)
    def test_no_threshold_no_value_stays_unknown(self, calc_cls):
        """The #439 fail-closed primitive is unchanged."""
        calc = calc_cls(db_client=Mock())
        assert calc._evaluate_status(_kpi(None), None) == KPIStatus.UNKNOWN

    @pytest.mark.parametrize("calc_cls", THRESHOLDED_CALCULATORS)
    def test_target_present_still_banded(self, calc_cls):
        calc = calc_cls(db_client=Mock())
        threshold = KPIThreshold(target=0.8, warning=0.7, critical=0.6)
        assert calc._evaluate_status(_kpi(threshold), 0.85) == KPIStatus.GOOD


class TestCausalMetricsStatus:
    """CM-001..005 report INFORMATIONAL (was hardcoded UNKNOWN)."""

    @pytest.fixture
    def calculator(self):
        return CausalMetricsCalculator(db_client=Mock())

    @pytest.fixture
    def ate_kpi(self):
        return KPIMetadata(
            id="CM-001",
            name="Average Treatment Effect",
            definition="E[Y(1) - Y(0)]",
            formula="E[Y(1) - Y(0)]",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.CAUSAL_METRICS,
            threshold=None,
        )

    def test_ate_with_value_is_informational(self, calculator, ate_kpi):
        calculator._execute_query = Mock(
            return_value=[{"ate": 0.15, "ate_std": 0.05, "n_samples": 1000}]
        )
        result = calculator.calculate(ate_kpi)
        assert result.value == 0.15
        assert result.status == KPIStatus.INFORMATIONAL

    def test_ate_error_stays_unknown(self, calculator, ate_kpi):
        """Calculation failure keeps the fail-closed UNKNOWN default."""
        calculator._execute_query = Mock(side_effect=Exception("db down"))
        result = calculator.calculate(ate_kpi)
        assert result.value is None
        assert result.error is not None
        assert result.status == KPIStatus.UNKNOWN


class TestHistoryBackfillStatus:
    """_status_for mirrors the same semantics for persisted history points."""

    def test_no_threshold_is_informational(self):
        assert _status_for(_kpi(None), 42.0) == "informational"

    def test_target_present_still_banded(self):
        kpi = _kpi(KPIThreshold(target=0.8, warning=0.7, critical=0.6))
        assert _status_for(kpi, 0.85) == "good"
