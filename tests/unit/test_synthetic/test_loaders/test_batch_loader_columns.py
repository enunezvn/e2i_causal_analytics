"""Loader column-registration guards.

An unregistered generator column is SILENTLY dropped at load
(batch_loader._load_table's TABLE_COLUMNS gate), so weekly frontier-append
cohorts carry NULL for it while the base substrate (backfilled out-of-band)
does not. That drift broke cohort scoring on 2026-07-14: the T9 covariates
comorbidity_burden/prior_therapy_lines were never registered, appended holdout
rows arrived as NaN, and httpx's ``allow_nan=False`` JSON encoding crashed the
job. These guards pin the failure class, not just the two columns.
"""

import pytest

from src.ml.synthetic.loaders import batch_loader

# Generator-emitted patient_journeys columns that are INTENTIONALLY unregistered:
# no such DB column exists (registering them would 42703 the insert). Adding a
# column to the generator without either registering it here or listing it below
# is exactly the silent-NULL drift this module guards against.
_INTENTIONALLY_UNREGISTERED_PJ = {"treatment_effect_estimate"}


@pytest.mark.unit
def test_goldstd_patient_covariates_survive_the_loader():
    """Every gold-standard model covariate must be loader-registered — a dropped
    covariate means NULL features on appended rows and NaN at scoring time."""
    from src.mlops.gold_standard_eval import cohort_spec

    registered = set(batch_loader.TABLE_COLUMNS["patient_journeys"])
    for cohort, covariates in cohort_spec._PATIENT_COVARIATES.items():
        missing = set(covariates) - registered
        assert not missing, (
            f"{cohort} covariates {sorted(missing)} not registered in "
            "TABLE_COLUMNS['patient_journeys'] -> frontier-append rows will "
            "carry NULL for them and cohort scoring will fail on NaN"
        )


@pytest.mark.unit
def test_patient_generator_columns_registered_or_intentionally_excluded():
    """Emitted-vs-registered sync guard: any NEW PatientGenerator column must be
    registered (so the loader carries it) or explicitly listed as intentional."""
    from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

    df = PatientGenerator(GeneratorConfig(n_records=40, seed=7)).generate()
    registered = set(batch_loader.TABLE_COLUMNS["patient_journeys"])
    drift = set(df.columns) - registered - _INTENTIONALLY_UNREGISTERED_PJ
    assert not drift, (
        f"PatientGenerator emits {sorted(drift)} but the loader does not register "
        "them: appended rows would silently carry NULL. Register the column(s) in "
        "TABLE_COLUMNS['patient_journeys'] or add to _INTENTIONALLY_UNREGISTERED_PJ "
        "with a reason."
    )


@pytest.mark.unit
def test_treatment_events_registers_hcp_id():
    """business_impact_hcp_reach counts DISTINCT hcp_id on treatment_events; the
    generator draws it from the loaded HCP universe (FK-safe)."""
    assert "hcp_id" in set(batch_loader.TABLE_COLUMNS["treatment_events"])


@pytest.mark.unit
def test_patient_journeys_registers_commercial_arms_columns():
    registered = set(batch_loader.TABLE_COLUMNS["patient_journeys"])
    for col in (
        "adherent_180d",
        "low_gap_180d",
        "adherence_rate",
        "gap_days",
        "copay_support",
        "psp_enrolled",
        "rep_detailing_high",
        "sample_dropped",
        "copay_support_propensity",
        "psp_enrolled_propensity",
        "rep_detailing_high_propensity",
        "sample_dropped_propensity",
        "insurance_access_score",
    ):
        assert col in registered, f"{col} not registered -> loader will drop it"
