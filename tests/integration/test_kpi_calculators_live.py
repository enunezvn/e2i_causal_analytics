"""#574 faithful end-to-end: every rewired KPI calculator metric runs against the LIVE
DB through the kpi_query allowlist RPC.

- FIXABLE metrics must execute without raising (value may be 0.0/None for empty windows)
  — this catches column-alias mismatches between the registry SQL and the calculator's
  result handling, bad query_ids, and param-arity errors.
- MISSING metrics must FAIL LOUD (raise) — never a fabricated value.

CAPABILITY-GATED: skips unless SUPABASE_* is set AND the kpi_query RPC exists (migration
044 applied), e.g. CI without the migration skips.
"""

import os

import pytest

from src.kpi.calculators.brand_specific import BrandSpecificCalculator
from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.calculators.causal_metrics import CausalMetricsCalculator
from src.kpi.calculators.data_quality import DataQualityCalculator
from src.kpi.calculators.model_performance import ModelPerformanceCalculator
from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]

CTX = {"brand": "Fabhalta", "segment": None, "model_name": "default_model"}

# (calculator class, fixable _calc methods, missing _calc methods)
SPECS = [
    # #577 causal trio (PR1/PR2/PR3): CM-003 (_calc_causal_impact), CM-004
    # (_calc_counterfactual), and CM-005 (_calc_mediation_effect) are all WIRED — no causal
    # metric remains fail-loud. CM-003 = mean causal_effect_size; CM-004 = counterfactual
    # LEVEL after the do-contrast rework; CM-005 = proportion mediated from the coherent
    # direct/indirect decomposition. Their faithful e2es live in
    # test_577_causal_metrics_live.py — which capability-gates on their OWN query_ids
    # (migrations 047/048/049). Kept out of this shared list (which gates only on the
    # 044-era causal_metrics_ate) to avoid a FAIL (vs skip) on a 044-but-not-049 target.
    (
        CausalMetricsCalculator,
        ["_calc_ate", "_calc_cate"],
        [],
    ),
    # #577 WS3-BI-003: _calc_patient_touch_rate is wired to a code-anchored eligibility
    # view + delivered-touch over real data, but its faithful e2e lives in
    # test_577_patient_touch_live.py — which capability-gates on its OWN query_id
    # (migration 050). Keeping it out of this shared list (which gates only on the
    # 044-era causal_metrics_ate) avoids a FAIL (vs skip) on a 044-but-not-050 target.
    (
        BusinessImpactCalculator,
        [
            "_calc_mau",
            "_calc_wau",
            "_calc_hcp_coverage",
            "_calc_trx",
            "_calc_nrx",
            "_calc_nbrx",
            "_calc_trx_share",
            "_calc_conversion_rate",
            "_calc_roi",
        ],
        [],
    ),
    (
        BrandSpecificCalculator,
        ["_calc_remi_intent_delta", "_calc_kisqali_dx_adoption", "_calc_kisqali_oncologist_reach"],
        [],
    ),
    # #577 Tier 2: BR-001 (_calc_remi_ah_uncontrolled) + BR-003 (_calc_fabhalta_pnh_tested)
    # are wired to a real generated cohort, but their faithful e2e lives in
    # test_577_brand_specific_live.py — which capability-gates on their OWN query_ids
    # (migration 046). Keeping them out of this shared list (which gates only on the
    # 044-era causal_metrics_ate) avoids a FAIL (vs skip) on a 044-but-not-046 target.
    # #577 WS2-TR-003: _calc_action_rate_uplift is wired to a randomized control-arm holdout
    # + arm-conditioned action_taken, but its faithful e2e lives in
    # test_577_action_rate_uplift_live.py — which capability-gates on its OWN query_id
    # (migration 051). Keeping it out of this shared list (which gates only on the
    # 044-era causal_metrics_ate) avoids a FAIL (vs skip) on a 044-but-not-051 target.
    (
        TriggerPerformanceCalculator,
        [
            "_calc_trigger_precision",
            "_calc_trigger_recall",
            "_calc_acceptance_rate",
            "_calc_false_alert_rate",
            "_calc_override_rate",
            "_calc_lead_time",
            "_calc_change_fail_rate",
        ],
        [],
    ),
    (
        DataQualityCalculator,
        [
            "_calc_source_coverage_patients",
            "_calc_cross_source_match",
            "_calc_stacking_lift",
            "_calc_completeness_pass_rate",
            "_calc_data_lag",
            "_calc_time_to_release",
            # #577 Tier A: wired to real data (reference_universe + hcp_profiles/patient_journeys).
            "_calc_source_coverage_hcps",
            "_calc_geographic_consistency",
        ],
        # #577 WS1-DQ-008: _calc_label_quality is wired to the corpus-level generalized Fleiss κ
        # over the coherently-reseeded ml_annotations (latent-truth labels), but its faithful
        # e2e lives in test_577_label_quality_live.py — which capability-gates on its OWN
        # query_id (data_quality_label_quality, migration 052). Kept out of this shared list
        # (which gates only on the 044-era causal_metrics_ate) to avoid a FAIL (vs skip) on a
        # 044-but-not-052 target.
        [],
    ),
    (ModelPerformanceCalculator, ["_calc_shap_coverage"], []),
]


def _make(calc_cls):
    calc = calc_cls()
    if calc.db_client is None:
        pytest.skip("no Supabase client")
    try:
        calc.db_client.rpc("kpi_query", {"query_id": "causal_metrics_ate", "params": []}).execute()
    except Exception as e:
        pytest.skip(f"kpi_query RPC unavailable (migration 044 not applied?): {e}")
    return calc


_FIXABLE = [(c, m) for (c, fix, _miss) in SPECS for m in fix]
_MISSING = [(c, m) for (c, _fix, miss) in SPECS for m in miss]


@pytest.mark.parametrize(
    "calc_cls,method", _FIXABLE, ids=[f"{c.__name__}.{m}" for c, m in _FIXABLE]
)
def test_fixable_metric_runs(calc_cls, method):
    calc = _make(calc_cls)
    # Must not raise (column-alias / query_id / param errors would raise here).
    getattr(calc, method)(dict(CTX))


@pytest.mark.parametrize(
    "calc_cls,method", _MISSING, ids=[f"{c.__name__}.{m}" for c, m in _MISSING]
)
def test_missing_metric_fails_loud(calc_cls, method):
    calc = _make(calc_cls)
    with pytest.raises(Exception):
        getattr(calc, method)(dict(CTX))


def test_kpi_query_rejects_unknown_id():
    """The allowlist boundary: an unregistered query_id is rejected (no arbitrary run)."""
    calc = _make(BusinessImpactCalculator)
    with pytest.raises(Exception):
        calc.db_client.rpc(
            "kpi_query", {"query_id": "definitely_not_registered", "params": []}
        ).execute()


def test_kpi_query_rejects_wrong_arity():
    """Param count must match the registry's declared max_params."""
    calc = _make(BusinessImpactCalculator)
    with pytest.raises(Exception):
        calc.db_client.rpc(
            "kpi_query", {"query_id": "causal_metrics_ate", "params": ["unexpected"]}
        ).execute()
