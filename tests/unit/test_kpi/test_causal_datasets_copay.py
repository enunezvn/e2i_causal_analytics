"""The /segments/datasets and /causal/variables surfaces are data-driven from
_CAUSAL_DATASET_SPECS, so exposing a new arm is an allowlist change with no
per-surface code. These assertions lock that the arm actually reaches the menus
with a human label (the frontend has no humanizer).

Lives in tests/unit/ deliberately: tests/api/ is NOT collected by CI.
"""

import pytest


@pytest.mark.unit
def test_copay_is_offered_as_a_treatment_with_a_label():
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS, _COLUMN_LABELS

    spec = _CAUSAL_DATASET_SPECS["patient_journeys"]
    assert "copay_support" in spec["treatment"]
    assert _COLUMN_LABELS["copay_support"] == "Copay support"


@pytest.mark.unit
def test_insurance_access_label_marks_itself_as_derived():
    """insurance_access_score is a deterministic index computed from insurance_type
    (approx -0.35..+0.45), NOT a measured payer metric. The label must say so, for
    the same reason disease_severity's label says "cross-indication": an analyst
    reading a raw name off a menu has no other cue that the number is synthetic.

    Asserting the "derived" substring rather than the exact string keeps this a
    guard against the label silently reverting to a bare metric name, without
    pinning harmless copy edits.
    """
    from src.api.routes.causal import _COLUMN_LABELS

    label = _COLUMN_LABELS["insurance_access_score"]
    assert "derived" in label.lower(), (
        f"insurance_access_score label {label!r} no longer marks itself as derived; "
        "a synthetic index presented as a measured payer metric misleads the analyst."
    )


@pytest.mark.unit
def test_copay_curated_outcomes_are_offered():
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS

    outcomes = set(_CAUSAL_DATASET_SPECS["patient_journeys"]["outcome"])
    assert {"adherent_180d", "low_gap_180d", "persistent_180d"} <= outcomes


@pytest.mark.unit
def test_new_columns_are_float_coerced():
    """Un-coerced columns reach the executors as strings/None and drop rows."""
    from src.api.routes.causal import _CAUSAL_NUMERIC_COLUMNS

    numeric = _CAUSAL_NUMERIC_COLUMNS["patient_journeys"]
    assert {"copay_support", "insurance_access_score"} <= numeric


@pytest.mark.unit
def test_post_treatment_proxies_stay_out_of_the_adjustment_set():
    """Regression guard for the 2026-06-29 adversarial correction: adjusting on
    adherence_rate/gap_days overcontrols and collapses the effect to a fake ~0."""
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS

    covariates = set(_CAUSAL_DATASET_SPECS["patient_journeys"]["covariate"])
    assert "adherence_rate" not in covariates
    assert "gap_days" not in covariates
    assert "insurance_type" not in covariates  # categorical stays a cohort filter
