"""Task 03.5 — ATE/CATE recovery probe (gate 3 data layer).

The cheapest-disproof made permanent: generate one (cohort,brand) frame, recover
TRUE_ATE with LinearDML, recover per-segment CATE ordering with CausalForestDML,
and prove the propensity is estimable with both arms populated. In-process, no DB.
"""

import pytest

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.dgp.recovery_probe import recover_ate_and_cate
from src.ml.synthetic.dgp.treatment_arm import ARM_CONFOUNDERS
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

pytestmark = pytest.mark.heavy_ml  # groups econml import on one worker (pyproject:214)


# All 3 brands — Fabhalta INCLUDED. Its segment CATE ordering was historically
# the fragile case (high>med>low held only 2/3 before the T11 latent-CATE boost);
# leaving it out of the strict-ordering gate let that fragility regress silently.
# Measured (seed=21, n=3000): Fabhalta high 0.227 > med 0.128 > low 0.036, |ATE
# err| 0.025 — now locked.
@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_estimators_recover_true_ate_and_cate_ordering(brand):
    cfg = GeneratorConfig(seed=21, n_records=3000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    out = recover_ate_and_cate(df)

    # propensity estimable, both arms populated (overlap)
    assert out["propensity_auc"] > 0.5
    assert out["n_treated"] >= 30 and out["n_control"] >= 100

    # ATE recovery: LinearDML within tolerance of the realized TRUE_ATE.
    # Risk-difference scale tolerance is wider than the latent CATE (binary
    # outcome attenuates), so we assert SIGN + ORDERING, and ATE within 0.15.
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out

    # per-segment CATE ordering matches the DGP map high>medium>low
    cate = out["cate_by_segment_estimate"]
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"]


@pytest.mark.parametrize("outcome", ["adherent_180d", "low_gap_180d"])
@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_adherence_outcome_recoverable_on_existing_arm(brand, outcome):
    """Phase 0: treatment_arm -> {adherent_180d, low_gap_180d} must BOTH be
    recoverable (ATE within tolerance + CATE ordering), proving each curated
    binarized adherence outcome carries the planted effect and is recovery-gated
    (HIGH-2) BEFORE the allowlist exposes it."""
    cfg = GeneratorConfig(seed=21, n_records=3000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    truth = df.attrs["true_ate_by_arm"]["treatment_arm"][outcome]

    out = recover_ate_and_cate(
        df,
        treatment_col="treatment_arm",
        outcome_col=outcome,
        confounders=list(ARM_CONFOUNDERS),
        segment_col="segment_assignment",
        true_ate=truth["ate"],
        cate_map=truth["cate_by_segment"],
    )

    assert out["propensity_auc"] > 0.5
    assert out["n_treated"] >= 30 and out["n_control"] >= 100
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out
    cate = out["cate_by_segment_estimate"]
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"]


def test_biologic_experience_differential_recovers_for_remibrutinib():
    """Phase 3 (CLIN-SEG-P3): the planted biologic-experience differential CATE
    (experienced ~0.625x the naive effect — a mean-preserving 2x spread) must be
    RECOVERABLE by CausalForestDML for Remibrutinib, ON TOP of the severity CATE,
    while the severity ordering + ATE recovery are UNCHANGED by the mean-preserving
    construction.

    n=8000, not the n=3000 severity gate: the biologic gap is a SECOND, ~40/60 axis
    and needs more units to resolve (cheapest-disproof measured 5/5 seeds @ n>=8000;
    a spread below ~2x sign-flips). Remibrutinib ONLY — ``biologic_experienced`` is
    100% NULL for Kisqali/Fabhalta, so no biologic axis exists for them.
    """
    cfg = GeneratorConfig(
        seed=21, n_records=8000, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS
    )
    df = PatientGenerator(cfg).generate()
    assert df["biologic_experienced"].notna().all()  # Remibrutinib => populated

    # planted ground truth IS a real gap (naive effect > experienced effect)
    gt = df.attrs["cate_by_biologic"]
    assert gt["naive"] > gt["experienced"]

    out = recover_ate_and_cate(df, modifier_col="biologic_experienced")

    # RECOVERED biologic ordering: naive (0) recovered effect > experienced (1)
    bio = out["cate_by_modifier_estimate"]
    assert bio["0"] > bio["1"], out

    # severity ordering + ATE recovery survive the mean-preserving spread
    cate = out["cate_by_segment_estimate"]
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"], out
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out


def test_adherent_recoverable_under_causal_route_default_adjustment_set():
    """Guard for the overcontrol blocker: recovering treatment_arm -> adherent_180d
    using the causal page's DEFAULT adjustment set (the full covariate allowlist
    minus treatment/outcome) must stay within tolerance. If a post-treatment proxy
    is ever re-added to the covariate allowlist, this collapses and fails.

    Only Remibrutinib here (bound memory/time); the parametrized tests above cover
    all three brands with the hand-picked ARM_CONFOUNDERS.

    Phase 2 brand-gating: the causal route now applies ``_brand_scoped_covariates``
    to the allowlist before estimation, dropping indication-specific clinical
    covariates that are NULL off-brand (a Remibrutinib cohort has NULL egfr /
    proteinuria_g_day / ldh_ratio / ecog_performance_status after the DGP gating).
    Feeding those all-NULL columns to EconML raises ``Input contains NaN``. This test
    mirrors the production path exactly by brand-scoping the adjustment set to
    Remibrutinib — so it now also proves the real brand-aware adjustment set recovers
    the ATE on gated data.
    """
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS, _brand_scoped_covariates

    spec = _CAUSAL_DATASET_SPECS["patient_journeys"]
    default_adj = [c for c in spec["covariate"] if c not in ("treatment_arm", "adherent_180d")]

    # Mirror the causal route: brand-scope the adjustment set to the analyzed brand so
    # off-brand clinical columns (NULL under the Phase 2 gating) never reach EconML.
    default_adj = _brand_scoped_covariates(default_adj, "Remibrutinib")

    # geographic_region is categorical (string enum values) — the causal route
    # handles encoding separately from this probe. Filter it so recover_ate_and_cate
    # can call .to_numpy(dtype=float) on the adjustment columns without raising.
    default_adj = [c for c in default_adj if c != "geographic_region"]

    cfg = GeneratorConfig(
        seed=21, n_records=3000, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS
    )
    df = PatientGenerator(cfg).generate()
    truth = df.attrs["true_ate_by_arm"]["treatment_arm"]["adherent_180d"]
    out = recover_ate_and_cate(
        df,
        treatment_col="treatment_arm",
        outcome_col="adherent_180d",
        confounders=default_adj,
        segment_col="segment_assignment",
        true_ate=truth["ate"],
        cate_map=truth["cate_by_segment"],
    )
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out
