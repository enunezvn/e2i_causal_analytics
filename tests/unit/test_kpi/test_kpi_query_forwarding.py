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
# #577 Tier A: DQ-002 source_coverage_hcps + DQ-006 geographic_consistency are now WIRED
# to real data (reference_universe + hcp_profiles/patient_journeys) and so are no longer
# here — they move to the FIXABLE contract (see test_577_tier_a_* + the live e2e).
MISSING_METRICS = [
    (CausalMetricsCalculator, "_calc_causal_impact"),
    (CausalMetricsCalculator, "_calc_counterfactual"),
    (CausalMetricsCalculator, "_calc_mediation_effect"),
    (BusinessImpactCalculator, "_calc_patient_touch_rate"),
    (BrandSpecificCalculator, "_calc_remi_ah_uncontrolled"),
    (BrandSpecificCalculator, "_calc_fabhalta_pnh_tested"),
    (TriggerPerformanceCalculator, "_calc_action_rate_uplift"),
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


# --- #577 Tier A: DQ-002 + DQ-006 are now wired to real data (hermetic forwarding) -------


def _calc_returning(rows):
    """A DataQualityCalculator whose kpi_query RPC returns `rows`."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return DataQualityCalculator(db_client=client), client


def test_dq002_source_coverage_hcps_forwards_and_computes():
    """WS1-DQ-002 forwards to the allowlisted query_id (no params — global ratio)
    and computes covered/total."""
    calc, client = _calc_returning([{"covered": 546, "total": 21240}])
    val = calc._calc_source_coverage_hcps({"brand": None})
    client.rpc.assert_called_once_with(
        "kpi_query", {"query_id": "data_quality_source_coverage_hcps", "params": []}
    )
    assert abs(val - 546 / 21240) < 1e-9


def test_dq006_geographic_consistency_forwards_and_computes():
    """WS1-DQ-006 forwards to the allowlisted query_id and returns the max regional gap."""
    calc, client = _calc_returning([{"max_gap": 0.1049}])
    val = calc._calc_geographic_consistency({"brand": "Fabhalta"})
    client.rpc.assert_called_once_with(
        "kpi_query",
        {"query_id": "data_quality_geographic_consistency", "params": ["Fabhalta"]},
    )
    assert abs(val - 0.1049) < 1e-9


def test_dq006_status_is_lower_is_better():
    """WS1-DQ-006 is a GAP (lower is better): the status must invert the default
    higher-is-better evaluation (#577). Bands (target=0.05, warning=0.10): a small
    gap is GOOD, mid is WARNING, and a gap above the warning bound is CRITICAL — so
    the real all-brand gap (0.1049) is CRITICAL, not the GOOD the old code reported."""
    from src.kpi.models import (
        CalculationType,
        KPIMetadata,
        KPIStatus,
        KPIThreshold,
        Workstream,
    )

    calc = DataQualityCalculator(db_client=MagicMock())
    kpi = KPIMetadata(
        id="WS1-DQ-006",
        name="Geographic Consistency",
        definition="max gap",
        formula="max_region(|share_source - share_universe|)",
        calculation_type=CalculationType.DERIVED,
        workstream=Workstream.WS1_DATA_QUALITY,
        threshold=KPIThreshold(target=0.05, warning=0.10, critical=0.20),
    )
    assert calc._evaluate_status(kpi, 0.04) == KPIStatus.GOOD  # <= target
    assert calc._evaluate_status(kpi, 0.08) == KPIStatus.WARNING  # target < v <= warning
    assert calc._evaluate_status(kpi, 0.1049) == KPIStatus.CRITICAL  # > warning (real gap)
    # Guard the direction itself: under the (wrong) higher-is-better default, 0.04
    # would be CRITICAL — the fix flips it to GOOD.
    assert kpi.threshold.evaluate(0.04, lower_is_better=False) == KPIStatus.CRITICAL
