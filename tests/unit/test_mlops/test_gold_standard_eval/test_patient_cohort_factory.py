"""Tests for the per-brand patient cohort factory (P3-T1)."""

import pytest

from src.mlops.gold_standard_eval.cohort_spec import (
    BRANDS,
    PATIENT_COHORTS,
    goldstd_experiment_name,
    goldstd_model_name,
    make_patient_spec,
)

_THREE = ("disease_severity", "academic_hcp", "geographic_region")
# T9: persistence/discontinuation depend on 7 leakage-safe covariates.
# T11 (2026-06-22): initiation ALSO uses the 7-covariate set — the same 4 prognostic
# drivers were added to the treatment_initiated outcome eqn (⊥ treatment_arm).
_SEVEN = _THREE + (
    "insurance_type",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
)
# COMM-ARMS Phase 1 (2026-07-19): persistence/discontinuation carry an 8th covariate,
# copay_support, because copay enters the discontinuation logit and is therefore real
# outcome signal the model should see. initiation stays at 7 — copay is absent from the
# treatment_initiated equation, so fetching it there would be an unused feature and a
# gratuitous widening of that cohort's serving contract.
_EIGHT = _SEVEN + ("copay_support",)
# COMM-ARMS Phase 2 (2026-07-19): persistence/discontinuation gain a 9th covariate,
# psp_enrolled, which also enters the discontinuation logit. initiation stays at 7.
_NINE = _EIGHT + ("psp_enrolled",)
_EXPECTED_COVARIATES = {
    "initiation": _SEVEN,
    "persistence": _NINE,
    "discontinuation": _NINE,
}


def test_factory_covers_9_patient_slots():
    seen = set()
    for cohort in PATIENT_COHORTS:
        for brand in BRANDS:
            s = make_patient_spec(cohort, brand)
            assert s.brand == brand
            assert s.target == f"{cohort}_{brand.lower()}"
            # T11 + COMM-ARMS Phase 1: the three cohorts are no longer identical —
            # assert each against its OWN expected set rather than a shared one, so a
            # cohort that silently gains or loses a covariate still fails here.
            assert s.base_covariates == _EXPECTED_COVARIATES[cohort], f"{cohort}/{brand}"
            seen.add((s.target, s.label_column))
    assert len(seen) == 9
    # labels correct
    assert make_patient_spec("initiation", "Kisqali").label_column == "treatment_initiated"
    assert make_patient_spec("persistence", "Fabhalta").label_column == "persistent_180d"
    assert make_patient_spec("discontinuation", "Remibrutinib").label_column == "discontinued_180d"


def test_persistence_cohorts_use_nine_covariates():
    """COMM-ARMS Phase 1/2: persistence + discontinuation additionally observe
    copay_support + psp_enrolled. Ordering matters as well as membership — the serving
    bundle builds its vector positionally from base_covariates, so this asserts the
    exact tuple."""
    for brand in ("Remibrutinib", "Fabhalta", "Kisqali"):
        for cohort in ("persistence", "discontinuation"):
            assert make_patient_spec(cohort, brand).base_covariates == _NINE, f"{cohort}/{brand}"


def test_initiation_stays_at_seven_covariates():
    """initiation must NOT pick up copay_support: copay does not enter the
    treatment_initiated equation (verified byte-identical when the arm was wired), so
    adding it would widen the serving contract for a feature carrying no signal."""
    assert make_patient_spec("initiation", "Remibrutinib").base_covariates == _SEVEN
    assert "copay_support" not in make_patient_spec("initiation", "Remibrutinib").base_covariates


def test_name_helpers():
    assert goldstd_model_name("persistence", "Kisqali") == "persistence_kisqali_goldstd_lr_v1"
    assert (
        goldstd_experiment_name("initiation", "Fabhalta") == "initiation_fabhalta_goldstd_eval_v1"
    )


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        make_patient_spec("hcp_adoption", "Kisqali")  # not a patient cohort
    with pytest.raises(ValueError):
        make_patient_spec("initiation", "Tasigna")  # unknown brand
