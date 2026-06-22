"""Contract guard — production patient-grain estimators MUST adjust for every
covariate the synthetic DGP confounds the treatment arm on.

WHY THIS TEST EXISTS
--------------------
``src/ml/synthetic/dgp/treatment_arm.py::assign_treatment_arm`` assigns the
binary ``treatment_arm`` via a covariate PROPENSITY on ``ARM_CONFOUNDERS``
(``disease_severity``, ``academic_hcp``): treated patients are systematically
sicker and more often seen by academic HCPs, and those same covariates also
drive the outcome. That confounding is INTENTIONAL — it is what makes the
gold-standard cohort a realistic *observational* dataset that the causal engine
has to adjust. The naive difference-in-means is biased UPWARD (~0.28 vs the
designed ~0.18 true effect); only an estimator that adjusts for
``ARM_CONFOUNDERS`` recovers the honest effect.

If a future refactor drops ``disease_severity`` or ``academic_hcp`` from a
production patient-grain adjustment set, that estimator would silently report
the confounded ~0.28 as "the treatment effect" — a plausible-but-wrong value
(exactly the anti-mocking harm). These tests lock the contract so such a
regression fails CI instead of shipping. They are the durable counterpart to
the naive-vs-adjusted surfacing (Option D), which makes the *removed* bias
visible to the analyst.
"""

import numpy as np
import pytest

from src.ml.synthetic.dgp.treatment_arm import ARM_CONFOUNDERS, assign_treatment_arm


@pytest.mark.unit
def test_causal_route_patient_journeys_adjusts_for_arm_confounders():
    """The causal-discovery patient_journeys adjustment set must offer every
    DGP arm confounder as an available covariate."""
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS

    covariates = set(_CAUSAL_DATASET_SPECS["patient_journeys"]["covariate"])
    missing = set(ARM_CONFOUNDERS) - covariates
    assert not missing, (
        "patient_journeys causal adjustment set is missing DGP arm confounder(s) "
        f"{sorted(missing)}; a causal estimate on treatment_arm would be confounded "
        "(naive diff-in-means biased upward)."
    )


@pytest.mark.unit
def test_treatment_effects_route_adjusts_for_arm_confounders():
    """The GET /causal/treatment-effects cohort×brand ATE surface estimates
    treatment_arm on the patient cohorts using _TE_PATIENT_CONFOUNDERS — that
    adjustment set must also contain every DGP arm confounder (it may be a
    superset; supersets over-adjust harmlessly, a missing confounder does not)."""
    from src.api.routes.causal import _TE_PATIENT_CONFOUNDERS, _TE_TREATMENT_VAR

    assert _TE_TREATMENT_VAR == "treatment_arm"
    confounders = set(_TE_PATIENT_CONFOUNDERS)
    missing = set(ARM_CONFOUNDERS) - confounders
    assert not missing, (
        "GET /causal/treatment-effects patient confounders are missing DGP arm "
        f"confounder(s) {sorted(missing)}; the cohort×brand ATE would be confounded."
    )


@pytest.mark.unit
def test_recovery_probe_adjusts_for_arm_confounders():
    """The DGP recovery/calibration probe (LinearDML on treatment_arm) must adjust
    for every arm confounder, else the acceptance gate validates recovery against
    a stale adjustment set and false-greens. _COVARS is sourced from
    ARM_CONFOUNDERS, so this also pins that wiring."""
    from src.ml.synthetic.dgp.recovery_probe import _COVARS

    missing = set(ARM_CONFOUNDERS) - set(_COVARS)
    assert not missing, (
        f"recovery_probe._COVARS is missing DGP arm confounder(s) {sorted(missing)}; "
        "the recovery gate would validate against a stale adjustment set."
    )


@pytest.mark.unit
def test_segment_hte_adjusts_for_arm_confounders():
    """The segment-HTE effect-modifier set (X — conditioned on by the DML
    heterogeneity model) must contain every DGP arm confounder, since the
    segment HTE default treatment IS treatment_arm."""
    from src.api.routes.segments import (
        _SEGMENT_HTE_DEFAULT_TREATMENT,
        _SEGMENT_HTE_EFFECT_MODIFIERS,
    )

    # Guard only applies while the default treatment is the confounded arm.
    assert _SEGMENT_HTE_DEFAULT_TREATMENT == "treatment_arm"
    modifiers = set(_SEGMENT_HTE_EFFECT_MODIFIERS)
    missing = set(ARM_CONFOUNDERS) - modifiers
    assert not missing, (
        "segment-HTE effect-modifier/adjustment set is missing DGP arm confounder(s) "
        f"{sorted(missing)}; segment CATE on treatment_arm would be confounded."
    )


@pytest.mark.unit
def test_arm_confounders_constant_is_load_bearing():
    """``ARM_CONFOUNDERS`` must stay in lock-step with the columns
    ``assign_treatment_arm`` actually uses — perturbing each one must move the
    propensity. Guards against the constant drifting away from the code (a stale
    constant would make the contract tests above assert the wrong column set)."""
    n = 5000
    base = {
        "disease_severity": np.full(n, 5.0),
        "academic_hcp": np.zeros(n),
    }
    _, prop_base = assign_treatment_arm(base, np.random.default_rng(1))
    for cov in ARM_CONFOUNDERS:
        perturbed = {k: v.copy() for k, v in base.items()}
        perturbed[cov] = perturbed[cov] + 1.0
        _, prop_pert = assign_treatment_arm(perturbed, np.random.default_rng(1))
        assert not np.allclose(prop_base, prop_pert), (
            f"assign_treatment_arm propensity does not depend on {cov!r}, but it is "
            "listed in ARM_CONFOUNDERS — the constant has drifted from the DGP."
        )
