"""Tests for CohortSpec — grounded in _PJ_COHORTS['initiation'] in cohort_resolution.py."""

from src.mlops.gold_standard_eval.cohort_spec import INITIATION


def test_initiation_spec_matches_codebase_intent():
    assert INITIATION.target == "csu_treatment_initiation"
    assert INITIATION.brand == "Remibrutinib"
    assert INITIATION.label_column == "treatment_initiated"
    assert INITIATION.grain == "patient"
    assert INITIATION.base_covariates
    assert INITIATION.label_column not in INITIATION.base_covariates
    for leak in ("days_to_treatment", "discontinued_180d", "persistent_180d", "adherence_rate"):
        assert leak not in INITIATION.base_covariates
