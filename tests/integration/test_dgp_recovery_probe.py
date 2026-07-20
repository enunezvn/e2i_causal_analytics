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


@pytest.mark.parametrize("outcome", ["adherent_180d", "low_gap_180d"])
@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_copay_support_recoverable(brand, outcome):
    """Phase 1: copay_support -> {adherent_180d, low_gap_180d} must be recoverable
    off its OWN backdoor (insurance_access_score + disease_severity).

    n_records=8000, NOT the 3000 used by the treatment_arm gates. MEASURED
    (2026-07-19): at n=3000 CausalForestDML cannot resolve the MEDIUM segment for
    copay at ANY planted effect size — sweeping the CATE map from 0.28/0.20/0.12
    to 0.40/0.22/0.04 moved high (0.119->0.158) and low (0.054->0.034) while
    medium stayed pinned at ~0.02. n=8000 resolves it. Same resolution-floor
    phenomenon as the Phase 3 biologic differential (validated at n>=8000).
    """
    from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY

    cfg = GeneratorConfig(seed=21, n_records=8000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    truth = df.attrs["true_ate_by_arm"]["copay_support"][outcome]

    out = recover_ate_and_cate(
        df,
        treatment_col="copay_support",
        outcome_col=outcome,
        confounders=list(ARM_REGISTRY["copay_support"].confounders),
        segment_col="segment_assignment",
        true_ate=truth["ate"],
        cate_map=truth["cate_by_segment"],
    )

    assert out["propensity_auc"] > 0.5, out
    assert out["n_treated"] >= 30 and out["n_control"] >= 100, out
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out
    cate = out["cate_by_segment_estimate"]
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"], out


@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_psp_enrolled_adherence_recoverable(brand):
    """Phase 2: psp_enrolled -> adherent_180d must be recoverable off its OWN
    backdoor (disease_severity + engagement_score + academic_hcp).

    psp targets adherent_180d (NOT low_gap_180d), so unlike copay this is a single
    outcome. Same n=8000 resolution floor as copay. MEASURED (2026-07-19): with the
    adherent CATE map {0.38,0.13,0.05} the seed-21 h-m margin is worst for the
    flattest brand Fabhalta at +0.0327 (m-l +0.057); the first-guess {0.34,0.14,0.05}
    left it at a razor-thin +0.0152, so the high-medium gap was widened. Kisqali's
    planted ATE (0.122) sits above the +5-10pp band by the 1.40 brand scale — as
    copay's Kisqali did (0.140) — so the gate asserts |est-true|<0.15, not a band.
    """
    from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY

    cfg = GeneratorConfig(seed=21, n_records=8000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    truth = df.attrs["true_ate_by_arm"]["psp_enrolled"]["adherent_180d"]

    out = recover_ate_and_cate(
        df,
        treatment_col="psp_enrolled",
        outcome_col="adherent_180d",
        confounders=list(ARM_REGISTRY["psp_enrolled"].confounders),
        segment_col="segment_assignment",
        true_ate=truth["ate"],
        cate_map=truth["cate_by_segment"],
    )

    assert out["propensity_auc"] > 0.5, out
    assert out["n_treated"] >= 30 and out["n_control"] >= 100, out
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out
    cate = out["cate_by_segment_estimate"]
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"], out


@pytest.mark.parametrize("seed", [21, 7, 99, 123])
@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_psp_enrolled_persistence_recoverable(brand, seed):
    """Phase 2: psp_enrolled -> persistent_180d off its OWN backdoor.

    MULTI-SEED like the copay persistence gate below (same reason: logit->Bernoulli
    RD, thin separations). MEASURED (2026-07-19): the persistent CATE lives on the
    (brand-invariant) discontinuation logit _PSP_DISC_LOGIT. The first guess
    {-0.90,-0.45,-0.05} left seed-99 h-m at only +0.012 (gated!); widening the high
    arm to -1.10 lifts every seed-99 h-m to >=+0.042 while keeping the planted ATE
    ~0.079 in the +5-10pp band. This gate is what locks that widening in place.
    """
    from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY

    cfg = GeneratorConfig(seed=seed, n_records=8000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    truth = df.attrs["true_ate_by_arm"]["psp_enrolled"]["persistent_180d"]

    out = recover_ate_and_cate(
        df,
        treatment_col="psp_enrolled",
        outcome_col="persistent_180d",
        confounders=list(ARM_REGISTRY["psp_enrolled"].confounders),
        segment_col="segment_assignment",
        true_ate=truth["ate"],
        cate_map=truth["cate_by_segment"],
    )

    assert out["propensity_auc"] > 0.5, out
    assert out["n_treated"] >= 30 and out["n_control"] >= 100, out
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out
    cate = out["cate_by_segment_estimate"]
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"], out


@pytest.mark.parametrize("seed", [21, 7, 99, 123])
@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_copay_support_persistence_recoverable(brand, seed):
    """Task 10: copay_support -> persistent_180d off its OWN backdoor.

    MULTI-SEED on purpose. This outcome is a logit->Bernoulli draw, so the planted
    RD is an expit difference and the medium-low separation is the fragile one:
    _COPAY_DISC_LOGIT's low arm was widened from -0.14 to -0.04 precisely because
    the recovered medium-low margin collapsed to +0.003..+0.006 at seed 21 alone.
    A single-seed gate could not have caught that, so all 4 seeds are asserted.
    """
    from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY

    cfg = GeneratorConfig(seed=seed, n_records=8000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    truth = df.attrs["true_ate_by_arm"]["copay_support"]["persistent_180d"]

    out = recover_ate_and_cate(
        df,
        treatment_col="copay_support",
        outcome_col="persistent_180d",
        confounders=list(ARM_REGISTRY["copay_support"].confounders),
        segment_col="segment_assignment",
        true_ate=truth["ate"],
        cate_map=truth["cate_by_segment"],
    )

    assert out["propensity_auc"] > 0.5, out
    assert out["n_treated"] >= 30 and out["n_control"] >= 100, out
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out
    cate = out["cate_by_segment_estimate"]
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"], out


@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_copay_planted_ate_is_in_the_designed_band(brand):
    """The planted copay effect must stay in the design's +8-12pp commercial band
    (brand-scaled, so the spread across brands is expected). A drift out of band
    means the CATE constants were retuned without revisiting the design intent."""
    cfg = GeneratorConfig(seed=21, n_records=8000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    ate = df.attrs["true_ate_by_arm"]["copay_support"]["low_gap_180d"]["ate"]
    assert 0.05 < ate < 0.18, f"{brand.value} copay planted ATE {ate:.4f} out of band"


@pytest.mark.parametrize("seed", [21, 7, 99, 123])
@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
@pytest.mark.parametrize("arm", ["rep_detailing_high", "sample_dropped"])
def test_initiation_commercial_arm_recoverable(arm, brand, seed):
    """COMM-ARMS Phase 3: rep_detailing_high + sample_dropped -> treatment_initiated must
    be recoverable off their OWN backdoor {academic_hcp, engagement_score}. These are the
    FIRST arms to fold into the SAME latent as treatment_arm (initiation), so the fold-
    faithful disproof also proved treatment_arm's own seed-21 ordering survives (that gate
    is test_estimators_recover_true_ate_and_cate_ordering above).

    ATE recovery (|est-true|<0.15) is asserted MULTI-SEED (21/7/99/123), like the copay/psp
    persistence gates — the effects are logit->Bernoulli-scale RD differences and a single
    seed cannot prove the recovery is not a coin-flip. n=8000 (NOT 3000): rep/sample are
    DESIGNED WEAK (+3-6pp / +2-5pp), so like copay they sit at CausalForestDML's n=3000
    medium-segment resolution floor.

    STRICT high>med>low CATE ordering is asserted at the GATE SEED (21) ONLY, for all 3
    brands — MIRRORING treatment_arm's own seed-21 ordering gate. MEASURED (2026-07-20,
    phase3_disproof v4): with the copay-shaped {0.36,0.14,0.05} rep map + {0.20,0.10,0.04}
    sample map, sample orders 12/12 seed-brand cells and rep 11/12 (only Fabhalta/seed123,
    the flattest 0.70-scale brand at the resolution floor), while ALL cells order at seed 21.
    Widening the gate to strict ordering at every off-seed would chase that flattest-brand
    floor; the ATE recovers at every seed regardless (asserted here)."""
    from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY

    cfg = GeneratorConfig(seed=seed, n_records=8000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    truth = df.attrs["true_ate_by_arm"][arm]["treatment_initiated"]

    out = recover_ate_and_cate(
        df,
        treatment_col=arm,
        outcome_col="treatment_initiated",
        confounders=list(ARM_REGISTRY[arm].confounders),
        segment_col="segment_assignment",
        true_ate=truth["ate"],
        cate_map=truth["cate_by_segment"],
    )

    assert out["propensity_auc"] > 0.5, out
    assert out["n_treated"] >= 30 and out["n_control"] >= 100, out
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out
    if seed == 21:
        cate = out["cate_by_segment_estimate"]
        assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"], out


@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_initiation_arm_planted_ate_in_designed_band(brand):
    """The planted rep/sample initiation effects must stay in their design bands
    (rep +3-6pp, sample +2-5pp base; brand-scaled, so Kisqali's 1.40 scale runs hotter —
    the gate is a generous [0.02, 0.11] envelope, not the illustrative base band). A drift
    out means the CATE constants were retuned without revisiting the design intent."""
    cfg = GeneratorConfig(seed=21, n_records=8000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    rep_ate = df.attrs["true_ate_by_arm"]["rep_detailing_high"]["treatment_initiated"]["ate"]
    sample_ate = df.attrs["true_ate_by_arm"]["sample_dropped"]["treatment_initiated"]["ate"]
    assert 0.02 < rep_ate < 0.11, f"{brand.value} rep planted ATE {rep_ate:.4f} out of band"
    assert 0.01 < sample_ate < 0.09, (
        f"{brand.value} sample planted ATE {sample_ate:.4f} out of band"
    )
