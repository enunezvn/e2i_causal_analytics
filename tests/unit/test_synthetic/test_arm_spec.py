"""ArmSpec registry contract — every commercial arm is described by ONE frozen
spec, so a new arm cannot be added without declaring its confounders (which the
confounder-contract guard then forces into the analysis allowlist)."""

import pytest

from src.ml.synthetic.dgp.treatment_arm import ARM_CONFOUNDERS, ARM_REGISTRY, ArmSpec


@pytest.mark.unit
def test_registry_contains_the_existing_arm_and_copay():
    assert set(ARM_REGISTRY) == {"treatment_arm", "copay_support"}


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
def test_arm_spec_is_frozen():
    spec = ARM_REGISTRY["copay_support"]
    with pytest.raises(Exception):
        spec.intercept = 0.5  # type: ignore[misc]
