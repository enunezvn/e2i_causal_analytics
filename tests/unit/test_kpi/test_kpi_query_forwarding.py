"""#574 hermetic contract tests (CI-runnable; the live e2e in
tests/integration/test_kpi_calculators_live.py is capability-gated and skips in CI):

1. Every calculator's `_execute_query` forwards to the `kpi_query` ALLOWLIST RPC with
   `{"query_id": ..., "params": ...}` — never the dead `execute_sql`.
2. Every MISSING-data metric FAILS LOUD (raises) rather than returning a fabricated value.
"""

from unittest.mock import MagicMock

import pytest

from src.kpi.calculators.brand_specific import BrandSpecificCalculator
from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.calculators.causal_metrics import CausalMetricsCalculator
from src.kpi.calculators.data_quality import DataQualityCalculator
from src.kpi.calculators.model_performance import ModelPerformanceCalculator
from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator

ALL_CALCULATORS = [
    CausalMetricsCalculator,
    BusinessImpactCalculator,
    BrandSpecificCalculator,
    TriggerPerformanceCalculator,
    DataQualityCalculator,
    ModelPerformanceCalculator,
]

# MISSING-data metrics that must fail loud (no real source in the schema — #574).
MISSING_METRICS = [
    (CausalMetricsCalculator, "_calc_causal_impact"),
    (CausalMetricsCalculator, "_calc_counterfactual"),
    (CausalMetricsCalculator, "_calc_mediation_effect"),
    (BusinessImpactCalculator, "_calc_patient_touch_rate"),
    (BrandSpecificCalculator, "_calc_remi_ah_uncontrolled"),
    (BrandSpecificCalculator, "_calc_fabhalta_pnh_tested"),
    (TriggerPerformanceCalculator, "_calc_action_rate_uplift"),
    (DataQualityCalculator, "_calc_source_coverage_hcps"),
    (DataQualityCalculator, "_calc_geographic_consistency"),
    (DataQualityCalculator, "_calc_label_quality"),
]


@pytest.mark.parametrize("calc_cls", ALL_CALCULATORS)
def test_execute_query_forwards_to_kpi_query_allowlist(calc_cls):
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[{"x": 1}])
    calc = calc_cls(db_client=client)

    calc._execute_query("some_query_id", ["p"])

    client.rpc.assert_called_once_with("kpi_query", {"query_id": "some_query_id", "params": ["p"]})
    # The dead execute_sql RPC must never be used.
    assert client.rpc.call_args.args[0] == "kpi_query"


@pytest.mark.parametrize(
    "calc_cls,method", MISSING_METRICS, ids=[f"{c.__name__}.{m}" for c, m in MISSING_METRICS]
)
def test_missing_metric_fails_loud(calc_cls, method):
    """No-source metrics must raise (fail loud), never return a fabricated 0.0/default."""
    calc = calc_cls(db_client=MagicMock())
    with pytest.raises(RuntimeError, match="unavailable"):
        getattr(calc, method)({"brand": "Fabhalta", "segment": None, "model_name": "m"})
