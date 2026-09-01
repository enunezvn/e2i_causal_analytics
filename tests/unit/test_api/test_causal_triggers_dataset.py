# tests/unit/test_api/test_causal_triggers_dataset.py
"""Unit coverage for the nba_triggers dataset spec registration and coercion.

Locks the SSOT maps (brand column, numeric columns, derivation fns, fill-zero
outcomes, physical-table mapping) that teach the loaders how to read the
triggers grain without DB access.
"""

import pytest

from src.api.routes.causal import (
    _CAUSAL_BRAND_COLUMN,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_FILL_ZERO_OUTCOMES,
    _CAUSAL_NUMERIC_COLUMNS,
    _CAUSAL_NUMERIC_DERIVATIONS,
)


@pytest.mark.unit
def test_nba_triggers_spec_registered_with_rct_and_modifier_questions():
    spec = _CAUSAL_DATASET_SPECS["nba_triggers"]
    # RCT: control_group_flag -> action_taken; modifier: acceptance_status -> conversion_flag.
    assert "control_group_flag" in spec["treatment"]
    assert "acceptance_status" in spec["treatment"]
    assert "action_taken" in spec["outcome"]
    assert "conversion_flag" in spec["outcome"]


@pytest.mark.unit
def test_nba_triggers_numeric_and_derivation_and_fill_registered():
    numeric = _CAUSAL_NUMERIC_COLUMNS["nba_triggers"]
    # All four question columns coerce to numeric 0/1.
    assert {"control_group_flag", "action_taken", "conversion_flag", "acceptance_status"} <= numeric
    deriv = _CAUSAL_NUMERIC_DERIVATIONS["nba_triggers"]
    # acceptance_status derives to the "is accepted" indicator; action_taken to presence.
    assert deriv["acceptance_status"]("accepted") == 1.0
    assert deriv["acceptance_status"]("rejected") == 0.0
    assert deriv["acceptance_status"](None) == 0.0
    assert deriv["action_taken"]("called_patient") == 1.0
    assert deriv["action_taken"](None) == 0.0
    # Designed-NULL outcomes fill to 0 instead of dropping the row.
    assert {"action_taken", "conversion_flag"} <= _CAUSAL_FILL_ZERO_OUTCOMES["nba_triggers"]


@pytest.mark.unit
def test_nba_triggers_brand_column_is_brand_id():
    # triggers has NO `brand` column — the filter resolves against brand_id.
    assert _CAUSAL_BRAND_COLUMN.get("nba_triggers") == "brand_id"
    # patient_journeys keeps the default `brand` column.
    assert _CAUSAL_BRAND_COLUMN.get("patient_journeys", "brand") == "brand"


@pytest.mark.unit
def test_nba_triggers_physical_table_is_triggers():
    from src.api.routes.causal import _CAUSAL_PHYSICAL_TABLE

    assert _CAUSAL_PHYSICAL_TABLE["nba_triggers"] == "triggers"


# ---------------------------------------------------------------------------
# _coerce_estimation_row — trigger semantics locked via the SSOT maps
# ---------------------------------------------------------------------------

from src.api.routes.causal import (  # noqa: E402 — after registration tests
    _coerce_estimation_row,
)


def _trig_kw() -> dict:
    """Keyword args for _coerce_estimation_row that express the trigger grain."""
    return {
        "numeric_cols": _CAUSAL_NUMERIC_COLUMNS["nba_triggers"],
        "derivations": _CAUSAL_NUMERIC_DERIVATIONS["nba_triggers"],
        "fill_zero": frozenset(_CAUSAL_FILL_ZERO_OUTCOMES["nba_triggers"]),
    }


@pytest.mark.unit
def test_coerce_row_derives_bool_text_and_fills_designed_null_zero():
    # RCT row: control arm (control_group_flag True), no action taken (NULL).
    rec = _coerce_estimation_row(
        {"control_group_flag": True, "action_taken": None},
        select_cols=["control_group_flag", "action_taken"],
        treatment_var="control_group_flag",
        outcome_var="action_taken",
        **_trig_kw(),
    )
    assert rec == {"control_group_flag": 1.0, "action_taken": 0.0}  # NULL outcome -> 0, NOT dropped


@pytest.mark.unit
def test_coerce_row_modifier_question_accepted_and_converted():
    rec = _coerce_estimation_row(
        {"acceptance_status": "accepted", "conversion_flag": True},
        select_cols=["acceptance_status", "conversion_flag"],
        treatment_var="acceptance_status",
        outcome_var="conversion_flag",
        **_trig_kw(),
    )
    assert rec == {"acceptance_status": 1.0, "conversion_flag": 1.0}
    rec2 = _coerce_estimation_row(
        {"acceptance_status": "rejected", "conversion_flag": None},
        select_cols=["acceptance_status", "conversion_flag"],
        treatment_var="acceptance_status",
        outcome_var="conversion_flag",
        **_trig_kw(),
    )
    assert rec2 == {"acceptance_status": 0.0, "conversion_flag": 0.0}


@pytest.mark.unit
def test_coerce_row_patient_outcome_null_still_drops():
    # patient_journeys is NOT in _CAUSAL_FILL_ZERO_OUTCOMES: a NULL outcome still
    # drops the row (returns None) -> the existing gate is unchanged.
    rec = _coerce_estimation_row(
        {"treatment_arm": 1, "persistent_180d": None},
        select_cols=["treatment_arm", "persistent_180d"],
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        numeric_cols=_CAUSAL_NUMERIC_COLUMNS["patient_journeys"],
    )
    assert rec is None


# ---------------------------------------------------------------------------
# #1188: curated BASELINE covariates (RCT efficiency / ANCOVA), distinct from
# the de-confounding covariate set (#1872: non-empty — the OBSERVATIONAL
# acceptance_status -> conversion_flag edge carries a real backdoor).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_nba_triggers_baseline_covariates_registered():
    spec = _CAUSAL_DATASET_SPECS["nba_triggers"]
    baselines = spec["baseline_covariate"]
    # Pre-treatment prognostic covariates joined from patient_journeys.
    assert "disease_severity" in baselines
    assert "age_at_diagnosis" in baselines
    # Balanced strata offered per the issue (no prognostic signal expected).
    assert "academic_hcp" in baselines
    assert "geographic_region" in baselines


@pytest.mark.unit
def test_nba_triggers_baselines_exclude_post_treatment_descendants():
    """Post-treatment descendants would overcontrol (2026-06-29 lesson) and
    engagement_score accumulates over the journey (post-trigger contamination)
    — none may EVER be offered as an RCT baseline."""
    baselines = set(_CAUSAL_DATASET_SPECS["nba_triggers"]["baseline_covariate"])
    assert not {"adherence_rate", "gap_days", "engagement_score"} & baselines


@pytest.mark.unit
def test_nba_triggers_covariates_match_ssot_acceptance_backdoor():
    """#1872: the dataset offers exactly the SSOT backdoor set of the
    OBSERVATIONAL acceptance_status -> conversion_flag edge (COMM-ARMS Phase 4:
    the trigger_accepted arm is confounded on disease_severity +
    engagement_score, treatment_arm.ARM_REGISTRY). Locked to the causal_paths
    generator SSOT so the two can never drift apart again — the pre-fix
    `covariate: []` silently emptied every registry-derived adjustment set and
    shipped the naive difference (+0.0145 measured confounding bias on Kisqali).
    The RCT edge keeps its empty default via the randomized-treatment guard,
    not via an empty offer."""
    from src.ml.synthetic.generators.causal_paths_generator import _TRIGGER_EDGES

    ssot = {(t, o): confounders for t, o, confounders in _TRIGGER_EDGES}
    expected = ssot[("acceptance_status", "conversion_flag")]
    assert expected == ["disease_severity", "engagement_score"]
    assert _CAUSAL_DATASET_SPECS["nba_triggers"]["covariate"] == expected
    # The RCT edge's registry backdoor is empty — the covariate OFFER above must
    # never leak into it (guarded per-treatment at the submit endpoint).
    assert ssot[("control_group_flag", "action_taken")] == []


@pytest.mark.unit
def test_nba_triggers_confounders_registered_numeric():
    """#1872: the discovery-path intersection (spec covariate ∩ numeric∪categorical)
    must KEEP the two patient-joined confounders — absent registration they were
    silently dropped from every leaderboard adjustment set."""
    numeric = _CAUSAL_NUMERIC_COLUMNS["nba_triggers"]
    assert {"disease_severity", "engagement_score"} <= numeric


@pytest.mark.unit
def test_other_datasets_have_no_baseline_role_or_empty():
    """baseline_covariate is an RCT-only role today; observational grains must
    not silently gain one."""
    for ds in ("patient_journeys", "hcp_adoption"):
        assert not _CAUSAL_DATASET_SPECS[ds].get("baseline_covariate", [])
