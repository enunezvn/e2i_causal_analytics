"""ArmSpec registry contract — every commercial arm is described by ONE frozen
spec, so a new arm cannot be added without declaring its confounders (which the
confounder-contract guard then forces into the analysis allowlist)."""

import numpy as np
import pytest

from src.ml.synthetic.dgp.treatment_arm import (
    ARM_CONFOUNDERS,
    ARM_REGISTRY,
    ArmSpec,
    assign_arm_from_spec,
    assign_treatment_arm,
    insurance_access_from_type,
)


@pytest.mark.unit
def test_registry_contains_the_existing_arm_and_commercial_arms():
    assert set(ARM_REGISTRY) == {
        "treatment_arm",
        "copay_support",
        "psp_enrolled",
        # COMM-ARMS Phase 3: two arms folding into the treatment_initiated latent.
        "rep_detailing_high",
        "sample_dropped",
    }


@pytest.mark.unit
def test_existing_arm_spec_matches_the_legacy_constants():
    """The registry entry for treatment_arm must encode EXACTLY the historical
    coefficients — this is what makes the delegation byte-identical."""
    spec = ARM_REGISTRY["treatment_arm"]
    assert isinstance(spec, ArmSpec)
    assert spec.name == "treatment_arm"
    assert spec.confounders == {"disease_severity": 0.30, "academic_hcp": 0.80}
    assert spec.intercept == -2.0
    assert spec.center == {"disease_severity": 5.0}
    # ARM_CONFOUNDERS stays the legacy tuple for treatment_arm's own contract.
    assert tuple(spec.confounders) == ARM_CONFOUNDERS


@pytest.mark.unit
def test_copay_spec_declares_its_backdoor_and_targets():
    spec = ARM_REGISTRY["copay_support"]
    assert spec.name == "copay_support"
    assert set(spec.confounders) == {"insurance_access_score", "disease_severity"}
    # LOWER access -> MORE copay support (the real-world skew).
    assert spec.confounders["insurance_access_score"] < 0
    assert spec.target_outcomes == ("adherent_180d", "low_gap_180d", "persistent_180d")


@pytest.mark.unit
def test_psp_spec_declares_its_backdoor_and_targets():
    spec = ARM_REGISTRY["psp_enrolled"]
    assert spec.name == "psp_enrolled"
    assert set(spec.confounders) == {"disease_severity", "engagement_score", "academic_hcp"}
    # psp skews to sicker + more-engaged + academic-HCP patients (all positive pulls).
    assert all(v > 0 for v in spec.confounders.values())
    # psp targets adherent + persistent (NOT low_gap — that is copay's, not psp's).
    assert spec.target_outcomes == ("adherent_180d", "persistent_180d")


@pytest.mark.unit
def test_psp_propensity_moves_with_each_declared_confounder():
    """Per-arm contract guard: perturbing ANY declared psp confounder must move the
    propensity, so the declared set cannot silently go stale."""
    n = 4000
    spec = ARM_REGISTRY["psp_enrolled"]
    base = {
        "disease_severity": np.full(n, 5.0),
        "engagement_score": np.full(n, 5.0),
        "academic_hcp": np.zeros(n),
    }
    _, p_base = assign_arm_from_spec(spec, base, np.random.default_rng(0))
    for cov in spec.confounders:
        bumped = {k: v.copy() for k, v in base.items()}
        bumped[cov] = bumped[cov] + 1.0
        _, p_bumped = assign_arm_from_spec(spec, bumped, np.random.default_rng(0))
        assert not np.allclose(p_base, p_bumped), f"propensity ignores {cov}"


@pytest.mark.unit
def test_psp_propensity_has_overlap():
    """Clipped to [0.01, 0.99] so both arms are populated and e(X) is estimable."""
    n = 5000
    rng = np.random.default_rng(7)
    covs = {
        "disease_severity": np.clip(rng.normal(5.0, 2.0, n), 0, 10),
        "engagement_score": np.clip(rng.normal(5.0, 2.0, n), 0, 10),
        "academic_hcp": (rng.random(n) < 0.30).astype(int),
    }
    arm, prop = assign_arm_from_spec(ARM_REGISTRY["psp_enrolled"], covs, rng)
    assert prop.min() >= 0.01 and prop.max() <= 0.99
    assert 0.20 < arm.mean() < 0.60, arm.mean()


@pytest.mark.unit
def test_arm_spec_is_frozen():
    spec = ARM_REGISTRY["copay_support"]
    with pytest.raises(Exception):
        spec.intercept = 0.5  # type: ignore[misc]


@pytest.mark.unit
def test_delegation_is_byte_identical_to_the_legacy_arm():
    """assign_treatment_arm must delegate to assign_arm_from_spec and produce
    BIT-IDENTICAL output — same values AND same RNG consumption. A drift here
    silently changes every shipped causal result on the next reseed."""
    rng_a = np.random.default_rng(21)
    rng_b = np.random.default_rng(21)
    n = 5000
    covs = {
        "disease_severity": np.clip(np.random.default_rng(1).normal(5.0, 2.0, n), 0, 10),
        "academic_hcp": (np.random.default_rng(2).random(n) < 0.30).astype(int),
    }
    arm_legacy, prop_legacy = assign_treatment_arm(covs, rng_a)
    arm_spec, prop_spec = assign_arm_from_spec(ARM_REGISTRY["treatment_arm"], covs, rng_b)

    np.testing.assert_array_equal(arm_legacy, arm_spec)
    np.testing.assert_array_equal(prop_legacy, prop_spec)
    # RNG streams must be at the SAME position afterwards (equal consumption).
    assert rng_a.random() == rng_b.random()


@pytest.mark.unit
def test_copay_propensity_moves_with_each_declared_confounder():
    """Per-arm contract guard: perturbing ANY declared confounder must move the
    propensity, so the declared set cannot silently go stale."""
    n = 4000
    spec = ARM_REGISTRY["copay_support"]
    base = {
        "insurance_access_score": np.full(n, 0.10),
        "disease_severity": np.full(n, 5.0),
    }
    _, p_base = assign_arm_from_spec(spec, base, np.random.default_rng(0))
    for cov in spec.confounders:
        bumped = {k: v.copy() for k, v in base.items()}
        bumped[cov] = bumped[cov] + 1.0
        _, p_bumped = assign_arm_from_spec(spec, bumped, np.random.default_rng(0))
        assert not np.allclose(p_base, p_bumped), f"propensity ignores {cov}"


@pytest.mark.unit
def test_copay_propensity_has_overlap():
    """Clipped to [0.01, 0.99] so both arms are populated and e(X) is estimable."""
    n = 5000
    rng = np.random.default_rng(7)
    covs = {
        "insurance_access_score": rng.choice([0.45, 0.10, -0.35, -0.55], n),
        "disease_severity": np.clip(rng.normal(5.0, 2.0, n), 0, 10),
    }
    arm, prop = assign_arm_from_spec(ARM_REGISTRY["copay_support"], covs, rng)
    assert prop.min() >= 0.01 and prop.max() <= 0.99
    assert 0.20 < arm.mean() < 0.60, arm.mean()


@pytest.mark.unit
def test_insurance_access_score_is_the_documented_gradient():
    """commercial > medicare > medicaid > uninsured, sourced from the _INIT_INS_ACCESS
    SSOT so it can never drift from the initiation prognostic offset."""
    ins = np.array(["commercial", "medicare", "medicaid", "uninsured"])
    score = insurance_access_from_type(ins)
    assert score[0] > score[1] > score[2] > score[3]
    np.testing.assert_allclose(score, [0.45, 0.10, -0.35, -0.55])


@pytest.mark.unit
def test_insurance_access_score_handles_unknown_category():
    """An unseen insurance category must degrade to the neutral 0.0, never KeyError
    (the generator must not crash on a future enum addition)."""
    np.testing.assert_allclose(insurance_access_from_type(np.array(["medicare_advantage"])), [0.0])


@pytest.mark.unit
def test_assign_arm_from_spec_rejects_a_missing_declared_confounder():
    """A caller that forgets a declared confounder must FAIL LOUD, not silently
    estimate a propensity that omits a backdoor path.

    Matches the INTENTIONAL guard's wording, not just the column name. The bare
    name matched the incidental `KeyError` the old `covariates[cov]` lookup raised
    anyway, so this test passed BEFORE the hardening existed and would keep passing
    if the guard were reverted -- it locked nothing. Matching "declares confounder"
    ties it to the explicit check, which is the behavior worth protecting: the
    incidental KeyError only fires because the loop happens to touch every
    confounder, which is exactly the implementation detail this guard removes
    reliance on.
    """
    import numpy as np
    import pytest as _pytest

    from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY, assign_arm_from_spec

    spec = ARM_REGISTRY["copay_support"]
    with _pytest.raises(KeyError, match="declares confounder"):
        assign_arm_from_spec(
            spec,
            {"disease_severity": np.zeros(10)},  # insurance_access_score MISSING
            np.random.default_rng(0),
        )


@pytest.mark.unit
def test_assign_arm_from_spec_rejects_ragged_covariates():
    """Mismatched lengths must raise, not broadcast into a wrong-length arm."""
    import numpy as np
    import pytest as _pytest

    from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY, assign_arm_from_spec

    spec = ARM_REGISTRY["copay_support"]
    with _pytest.raises(ValueError, match="length"):
        assign_arm_from_spec(
            spec,
            {"insurance_access_score": np.zeros(10), "disease_severity": np.zeros(7)},
            np.random.default_rng(0),
        )
