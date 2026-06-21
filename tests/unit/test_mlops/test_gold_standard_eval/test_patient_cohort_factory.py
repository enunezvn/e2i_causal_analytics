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
# T9: persistence/discontinuation now depend on 7 leakage-safe covariates.
_SEVEN = _THREE + (
    "insurance_type",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
)


def test_factory_covers_9_patient_slots():
    seen = set()
    for cohort in PATIENT_COHORTS:
        for brand in BRANDS:
            s = make_patient_spec(cohort, brand)
            assert s.brand == brand
            assert s.target == f"{cohort}_{brand.lower()}"
            expected = _SEVEN if cohort in ("persistence", "discontinuation") else _THREE
            assert s.base_covariates == expected, f"{cohort}/{brand}"
            seen.add((s.target, s.label_column))
    assert len(seen) == 9
    # labels correct
    assert make_patient_spec("initiation", "Kisqali").label_column == "treatment_initiated"
    assert make_patient_spec("persistence", "Fabhalta").label_column == "persistent_180d"
    assert make_patient_spec("discontinuation", "Remibrutinib").label_column == "discontinued_180d"


def test_persistence_cohorts_use_seven_covariates():
    for brand in ("Remibrutinib", "Fabhalta", "Kisqali"):
        for cohort in ("persistence", "discontinuation"):
            assert make_patient_spec(cohort, brand).base_covariates == _SEVEN, f"{cohort}/{brand}"


def test_initiation_stays_three_covariates():
    assert make_patient_spec("initiation", "Remibrutinib").base_covariates == _THREE


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
