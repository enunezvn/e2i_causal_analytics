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
# #577 WS1-DQ-008 _calc_label_quality is now WIRED (meaning e2e in
# test_577_label_quality_live.py): the corpus-level GENERALIZED Fleiss κ (Fleiss 1971,
# per-subject n_i) over a coherent LATENT-TRUTH reseed of ml_annotations (each iaa_group
# co-rates ONE subject; annotators agree with the latent truth ~92% of the time → a
# realistic substantial κ, NOT the prior independent-noise κ≈0).
# #577 WS1-MP-009 _calc_feature_drift is now WIRED (meaning e2e in
# test_577_feature_drift_live.py): the corpus-level AVG Population Stability Index over the
# coherently-seeded ml_drift_history (migration 053). It is tuple-returning (value, error) and
# fail-CLOSED (returns (None, error) -> KPIStatus.UNKNOWN), so it NEVER raises and is tracked
# separately from this raising-forwarding list — see the dedicated forwarding/arity lock below.
# => Every metric #574 left fail-loud is now wired; MISSING_METRICS is empty.
MISSING_METRICS: list[tuple[type, str]] = []


@pytest.mark.parametrize("calc_cls", ALL_CALCULATORS)
def test_execute_query_forwards_to_kpi_query_allowlist(calc_cls):
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[{"x": 1}])
    calc = calc_cls(db_client=client)

    calc._execute_query("some_query_id", ["p"])

    client.rpc.assert_called_once_with("kpi_query", {"query_id": "some_query_id", "params": ["p"]})
    # The dead execute_sql RPC must never be used.
    assert client.rpc.call_args.args[0] == "kpi_query"


@pytest.mark.skipif(not MISSING_METRICS, reason="all #574 fail-loud metrics now wired (#577)")
@pytest.mark.parametrize(
    "calc_cls,method",
    MISSING_METRICS or [(None, None)],
    ids=[f"{c.__name__}.{m}" for c, m in MISSING_METRICS] or ["none"],
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


# --- #577 WS1-DQ-008: label_quality wired (generalized Fleiss κ over coherent annotations) --


def _grp(p, n, u):
    """A per-iaa_group registry row (category counts + rater total)."""
    return {"n_positive": p, "n_negative": n, "n_uncertain": u, "n_raters": p + n + u}


def test_label_quality_forwards_to_allowlist():
    """WS1-DQ-008 forwards to the allowlisted query_id with no params (corpus-level pooled κ)."""
    calc, client = _calc_returning([_grp(3, 0, 0), _grp(2, 1, 0), _grp(0, 0, 3)])
    calc._calc_label_quality({})
    client.rpc.assert_called_once_with(
        "kpi_query", {"query_id": "data_quality_label_quality", "params": []}
    )


def test_label_quality_matches_statsmodels_on_fixed_n():
    """ANTI-FABRICATION PARITY: the hand-rolled generalized Fleiss κ MUST equal the vetted
    statsmodels.fleiss_kappa on a fixed-n subset (where classic Fleiss applies)."""
    sm = pytest.importorskip("statsmodels.stats.inter_rater")
    import numpy as np

    # 5 groups, all n_raters=4 (fixed-n => statsmodels applies), concentrated agreement.
    rows = [_grp(4, 0, 0), _grp(3, 1, 0), _grp(0, 4, 0), _grp(0, 0, 4), _grp(2, 1, 1)]
    calc, _ = _calc_returning(rows)
    val = calc._calc_label_quality({})
    table = np.array([[r["n_positive"], r["n_negative"], r["n_uncertain"]] for r in rows])
    assert abs(val - sm.fleiss_kappa(table, method="fleiss")) < 1e-9


def test_label_quality_substantial_when_concordant_near_zero_when_noise():
    """The metric DISCRIMINATES real agreement from chance: a concordant corpus yields a
    substantial κ (>0.6); a genuinely-random (independent) corpus yields κ ≈ 0 (|κ|<0.15).

    NOTE chance ≠ a fixed [1,1,1] split: an even split is *maximal* within-group
    disagreement (κ→−0.5). True chance is INDEPENDENT random labels averaged over many
    groups, which is what the live independent-noise data gave (κ=0.0174)."""
    import numpy as np

    concordant = [_grp(4, 0, 0)] * 8 + [_grp(0, 4, 0)] * 8 + [_grp(0, 0, 4)] * 8
    calc, _ = _calc_returning(concordant)
    assert calc._calc_label_quality({}) > 0.6
    # Each of 60 groups: 3 raters drawn INDEPENDENTLY at random over 3 categories => the
    # corpus agreement is at chance => κ ≈ 0 (the null the latent-truth rework rises above).
    rng = np.random.default_rng(0)
    noise = []
    for _ in range(60):
        labels = rng.integers(0, 3, size=3)
        noise.append(_grp(*(int((labels == c).sum()) for c in range(3))))
    calc, _ = _calc_returning(noise)
    assert abs(calc._calc_label_quality({})) < 0.15


def test_label_quality_fails_loud_on_no_groups():
    """No iaa_groups (empty result) -> fail loud BEFORE numpy (the `if not result` guard;
    np.array([]).sum(axis=1) would otherwise AxisError, not the 'unavailable' contract)."""
    calc, _ = _calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_label_quality({})


def test_label_quality_genuine_low_or_negative_returned_not_raised():
    """A real worse-than-chance corpus yields a NEGATIVE κ — a legitimate realized statistic,
    returned not raised (mirrors the patient_touch/action_uplift genuine-value precedent)."""
    # Maximal within-group disagreement at n=3 over 3 categories => κ < 0.
    calc, _ = _calc_returning([_grp(1, 1, 1)] * 6)
    val = calc._calc_label_quality({})
    assert isinstance(val, float)
    assert val < 0.05  # at/below chance, returned (not raised)


def test_label_quality_degenerate_single_category_is_one():
    """A corpus where every rating is the SAME single category (P_e==1.0, 0/0 undefined) is
    defined as perfect-but-degenerate agreement = 1.0, not NaN/raise."""
    calc, _ = _calc_returning([_grp(3, 0, 0), _grp(4, 0, 0), _grp(2, 0, 0)])
    assert calc._calc_label_quality({}) == 1.0


def test_label_quality_shuffle_disproof_collapses_kappa():
    """SHUFFLE DISPROOF (the decisive anti-fabrication coherence proof): a coherent corpus
    has substantial κ, but permuting the same labels across groups (destroying the latent-truth
    structure) collapses κ to ≈0 — so κ responds ONLY to real agreement, it is not a constant."""
    import numpy as np

    rng = np.random.default_rng(0)
    # 30 groups of 4 raters each, 90% concordant with a per-group latent truth.
    coherent_rows = []
    flat = []  # the full pool of individual labels, for the shuffle
    for _ in range(30):
        truth = int(rng.integers(0, 3))
        labels = [truth if rng.random() < 0.9 else int(rng.integers(0, 3)) for _ in range(4)]
        flat.extend(labels)
        counts = [labels.count(c) for c in range(3)]
        coherent_rows.append(_grp(*counts))
    calc, _ = _calc_returning(coherent_rows)
    k_coherent = calc._calc_label_quality({})
    assert k_coherent > 0.6, f"coherent κ should be substantial, got {k_coherent}"

    rng.shuffle(flat)
    shuffled_rows = []
    for i in range(30):
        labels = flat[i * 4 : (i + 1) * 4]
        counts = [labels.count(c) for c in range(3)]
        shuffled_rows.append(_grp(*counts))
    calc, _ = _calc_returning(shuffled_rows)
    k_shuffled = calc._calc_label_quality({})
    assert abs(k_shuffled) < 0.2, f"shuffled κ should collapse to ≈0, got {k_shuffled}"
    assert k_coherent - k_shuffled > 0.4


# --- #577 WS1-MP-009: feature_drift wired (corpus PSI over the seeded ml_drift_history) -----


def _mp_calc_returning(rows):
    """A ModelPerformanceCalculator whose kpi_query RPC returns `rows`."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return ModelPerformanceCalculator(db_client=client), client


def test_feature_drift_forwards_to_allowlist_with_no_params():
    """WS1-MP-009 forwards to the allowlisted query_id with NO params (max_params=0 corpus
    aggregate — model_id is NULL so there is no honest per-model band; same as
    model_performance_shap_coverage / data_quality_label_quality). This LOCKS the arity:
    the calculator passes a STRING model_name but ml_drift_history keys on a UUID model_id, so
    binding it would be a LABEL-not-functional no-op — the SQL leg must pass []. A 1-element
    params would make kpi_query RAISE 'expects 0 param(s), got 1' against migration 053."""
    calc, client = _mp_calc_returning([{"avg_psi": 0.094332}])
    value, error = calc._calc_feature_drift({"model_name": "tier0_df99c7ba"})
    client.rpc.assert_called_once_with(
        "kpi_query", {"query_id": "model_performance_feature_drift", "params": []}
    )
    # SQL leg succeeds -> tuple (value, None); MLflow is NOT consulted.
    assert error is None
    assert abs(value - 0.094332) < 1e-9


def test_feature_drift_sql_success_does_not_consult_mlflow():
    """When the SQL leg returns a real avg_psi, the tuple contract returns it and the MLflow
    fallback is never reached (preserves the existing test_sql_succeeds_uses_db_value contract
    now that the query_id is registered). Exercises the real RPC-forwarding SQL leg directly."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[{"avg_psi": 0.13}])
    # Poison MLflow (via the constructor param) so consulting it would surface — it must not be.
    poisoned = MagicMock()
    poisoned.get_latest_versions.side_effect = AssertionError("MLflow consulted on SQL success")
    calc = ModelPerformanceCalculator(db_client=client, mlflow_client=poisoned)
    value, error = calc._calc_feature_drift({"model_name": "m"})
    assert (value, error) == (0.13, None)


def test_feature_drift_null_avg_psi_falls_through_to_fail_loud():
    """Empty/unseeded ml_drift_history -> AVG over 0 rows is SQL NULL -> the SQL leg records
    null_avg_psi, the MLflow leg has no versions, and the calculator returns the tuple
    (None, combined_error) naming BOTH legs — fail-CLOSED, never a fabricated PSI.

    The mlflow_client is INJECTED (returning no versions) so the MLflow leg fails closed WITHOUT
    a real network call — otherwise the lazy `MlflowClient()` would connect to MLFLOW_TRACKING_URI
    (set in CI), and the thread-method pytest timeout cannot preempt that blocking socket -> hang.
    Mirrors the existing TestFeatureDriftUnavailability fixture's mlflow stubbing."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[{"avg_psi": None}])
    mlflow_client = MagicMock()
    mlflow_client.get_latest_versions.return_value = []  # no versions -> model_not_found, NO network
    calc = ModelPerformanceCalculator(db_client=client, mlflow_client=mlflow_client)
    value, error = calc._calc_feature_drift({"model_name": "m"})
    assert value is None
    assert "sql_leg=db_query_returned_empty:null_avg_psi" in error
    assert "mlflow_leg=model_not_found:m" in error


def test_dq008_status_is_higher_is_better():
    """WS1-DQ-008 (IAA κ) is HIGHER-is-better (the inverse of the DQ-006 gap): a high κ is
    GOOD, a low κ is CRITICAL. It must NOT be in _LOWER_IS_BETTER_IDS, else a strong agreement
    would be mis-scored CRITICAL. Bands target=0.85, warning=0.70, critical=0.60; the live
    0.7565 reads WARNING (substantial, below the high 0.85 target — honestly disclosed)."""
    from src.kpi.models import (
        CalculationType,
        KPIMetadata,
        KPIStatus,
        KPIThreshold,
        Workstream,
    )

    calc = DataQualityCalculator(db_client=MagicMock())
    kpi = KPIMetadata(
        id="WS1-DQ-008",
        name="Label Quality (IAA)",
        definition="inter-annotator agreement",
        formula="fleiss_kappa",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_DATA_QUALITY,
        threshold=KPIThreshold(target=0.85, warning=0.70, critical=0.60),
    )
    assert "WS1-DQ-008" not in DataQualityCalculator._LOWER_IS_BETTER_IDS
    assert calc._evaluate_status(kpi, 0.90) == KPIStatus.GOOD  # >= target
    assert calc._evaluate_status(kpi, 0.7565) == KPIStatus.WARNING  # critical <= v < target
    assert calc._evaluate_status(kpi, 0.50) == KPIStatus.CRITICAL  # < critical
    # Under the (wrong) lower-is-better direction, 0.90 would be CRITICAL — guard the flip.
    assert kpi.threshold.evaluate(0.90, lower_is_better=True) == KPIStatus.CRITICAL
