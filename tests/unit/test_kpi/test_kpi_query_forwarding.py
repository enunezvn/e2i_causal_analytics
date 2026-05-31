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
# #577 causal trio: all three CM metrics are now WIRED (meaning e2es in
# test_577_causal_metrics_live.py):
#   PR1 CM-003 _calc_causal_impact — honest descriptive aggregate (mean causal_effect_size).
#   PR2 CM-004 _calc_counterfactual — coherent do-contrast (factual − treatment effect, floored).
#   PR3 CM-005 _calc_mediation_effect — coherent decomposition (indirect_effect grounded in the
#       product of the causal_chain edge magnitudes; direct = total − indirect; proportion
#       mediated = indirect/total).
# #577 WS3-BI-003 _calc_patient_touch_rate is now WIRED (meaning e2e in
# test_577_patient_touch_live.py): fraction of code-anchored ELIGIBLE patients
# (primary_diagnosis_code in the brand's qualifying ICD-10 set, via v_patient_eligibility —
# NOT the absent is_eligible flag #574) with >=1 DELIVERED trigger (delivery_status IN
# ('delivered','viewed') — an actual touchpoint, NOT the degenerate any-trigger=99.5% relabel).
# #577 WS2-TR-003 _calc_action_rate_uplift is now WIRED (meaning e2e in
# test_577_action_rate_uplift_live.py): the REALIZED relative uplift
# (action_rate_treatment − action_rate_control)/action_rate_control over a randomized
# control_group_flag holdout, where "action" = action_taken IS NOT NULL (a rep BEHAVIOR
# measurable in BOTH arms — NOT acceptance_status, which is treatment-only).
MISSING_METRICS = [
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


# --- #577 Tier 2 (brand-specific): BR-001 + BR-003 wired to a real generated cohort ----


def _brand_calc_returning(rows):
    """A BrandSpecificCalculator whose kpi_query RPC returns `rows`."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return BrandSpecificCalculator(db_client=client), client


def test_br001_remi_ah_uncontrolled_forwards_and_computes():
    """BR-001 forwards the UAS7 cutoff and returns the uncontrolled rate from real rows."""
    calc, client = _brand_calc_returning([{"uncontrolled_rate": 0.45}])
    val = calc._calc_remi_ah_uncontrolled({"brand": "Remibrutinib"})
    # Passes the guideline UAS7>=7 cutoff (PMID 34536239) as the bound param.
    client.rpc.assert_called_once_with(
        "kpi_query", {"query_id": "brand_specific_remi_ah_uncontrolled", "params": [7]}
    )
    assert abs(val - 0.45) < 1e-9


def test_br001_fails_loud_on_empty_cohort():
    """No antihistamine-treated cohort -> fail loud (NOT a fabricated 0% 'controlled')."""
    calc, _ = _brand_calc_returning([{"uncontrolled_rate": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_remi_ah_uncontrolled({"brand": "Remibrutinib"})


def test_br003_fabhalta_pnh_tested_forwards_and_computes():
    """BR-003 forwards to the allowlist id and returns tested/eligible."""
    calc, client = _brand_calc_returning([{"tested_rate": 0.65}])
    val = calc._calc_fabhalta_pnh_tested({"brand": "Fabhalta"})
    client.rpc.assert_called_once_with(
        "kpi_query", {"query_id": "brand_specific_fabhalta_pnh_tested", "params": []}
    )
    assert abs(val - 0.65) < 1e-9


def test_br003_fails_loud_on_empty_eligible_cohort():
    """No D59.5-eligible cohort -> fail loud (NOT a fabricated rate)."""
    calc, _ = _brand_calc_returning([{"tested_rate": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_fabhalta_pnh_tested({"brand": "Fabhalta"})


# --- #577 WS3-BI-003: patient_touch_rate wired (code-anchored eligible + delivered touch) ----


def _bi_calc_returning(rows):
    """A BusinessImpactCalculator whose kpi_query RPC returns `rows`."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return BusinessImpactCalculator(db_client=client), client


def test_patient_touch_rate_forwards_and_computes():
    """WS3-BI-003 forwards the optional brand filter to the allowlist id and returns the
    touch-rate FRACTION (the division is done in SQL; sibling parity with conversion_rate)."""
    calc, client = _bi_calc_returning([{"touch_rate": 0.9074}])
    val = calc._calc_patient_touch_rate({"brand": "Fabhalta"})
    client.rpc.assert_called_once_with(
        "kpi_query",
        {"query_id": "business_impact_patient_touch_rate", "params": ["Fabhalta"]},
    )
    assert abs(val - 0.9074) < 1e-9


def test_patient_touch_rate_no_brand_binds_empty_sentinel():
    """No brand in context -> the empty-string sentinel (all brands). Locks the optional-param
    idiom AND the EXACT max_params=1 arity (always exactly one element, never [])."""
    calc, client = _bi_calc_returning([{"touch_rate": 0.9074}])
    calc._calc_patient_touch_rate({})
    client.rpc.assert_called_once_with(
        "kpi_query",
        {"query_id": "business_impact_patient_touch_rate", "params": [""]},
    )


def test_patient_touch_rate_fails_loud_on_empty_eligible_cohort():
    """No code-anchored eligible cohort (NULLIF -> NULL touch_rate) -> fail loud, NOT a
    fabricated 0.0."""
    calc, _ = _bi_calc_returning([{"touch_rate": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_patient_touch_rate({"brand": "Fabhalta"})


def test_patient_touch_rate_genuine_zero_is_returned_not_raised():
    """A genuine 0.0 (eligible cohort exists, but none delivered-touched) is a LEGITIMATE
    value and must be returned, never raised."""
    calc, _ = _bi_calc_returning([{"touch_rate": 0.0}])
    assert calc._calc_patient_touch_rate({"brand": "Fabhalta"}) == 0.0


# --- #577 WS2-TR-003: action_rate_uplift wired (randomized control arm + arm-conditioned action) ---


def _trigger_calc_returning(rows):
    """A TriggerPerformanceCalculator whose kpi_query RPC returns `rows`."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return TriggerPerformanceCalculator(db_client=client), client


def test_action_rate_uplift_forwards_and_computes():
    """WS2-TR-003 forwards to the allowlist id (no param — global treatment-vs-control ratio)
    and returns the realized RELATIVE uplift fraction (computed per-arm in SQL)."""
    calc, client = _trigger_calc_returning(
        [{"action_rate_uplift": 0.2751, "treatment_rate": 0.3861, "control_rate": 0.3028}]
    )
    val = calc._calc_action_rate_uplift({})
    client.rpc.assert_called_once_with(
        "kpi_query",
        {"query_id": "trigger_performance_action_rate_uplift", "params": []},
    )
    assert abs(val - 0.2751) < 1e-9


def test_action_rate_uplift_fails_loud_on_empty_arm():
    """Either arm empty -> NULL uplift (or no row) -> fail loud, NOT a fabricated 0.0."""
    calc, _ = _trigger_calc_returning([{"action_rate_uplift": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_action_rate_uplift({})


def test_action_rate_uplift_fails_loud_on_empty_result():
    """An empty CROSS JOIN (an arm has zero rows) returns [] -> the `not result` guard must
    fire and raise 'unavailable', never IndexError on result[0]."""
    calc, _ = _trigger_calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_action_rate_uplift({})


def test_action_rate_uplift_genuine_zero_and_negative_are_returned_not_raised():
    """A genuine 0.0 (both arms populated, equal action rates -> no lift) and a NEGATIVE
    uplift (treatment worse than control) are legitimate realized values — returned, not
    raised (the negative reads CRITICAL downstream via the higher-is-better bands)."""
    calc, _ = _trigger_calc_returning([{"action_rate_uplift": 0.0}])
    assert calc._calc_action_rate_uplift({}) == 0.0
    calc, _ = _trigger_calc_returning([{"action_rate_uplift": -0.05}])
    assert calc._calc_action_rate_uplift({}) == -0.05
