"""Tests for the per-brand patient cohort factory (P3-T1)."""

import pytest

from src.mlops.gold_standard_eval.cohort_spec import (
    BRANDS,
    PATIENT_COHORTS,
    goldstd_experiment_name,
    goldstd_model_name,
    make_patient_spec,
)


def test_factory_covers_9_patient_slots():
    seen = set()
    for cohort in PATIENT_COHORTS:
        for brand in BRANDS:
            s = make_patient_spec(cohort, brand)
            assert s.brand == brand
            assert s.target == f"{cohort}_{brand.lower()}"
            assert s.base_covariates == ("disease_severity", "academic_hcp", "geographic_region")
            seen.add((s.target, s.label_column))
    assert len(seen) == 9
    # labels correct
    assert make_patient_spec("initiation", "Kisqali").label_column == "treatment_initiated"
    assert make_patient_spec("persistence", "Fabhalta").label_column == "persistent_180d"
    assert make_patient_spec("discontinuation", "Remibrutinib").label_column == "discontinued_180d"


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
