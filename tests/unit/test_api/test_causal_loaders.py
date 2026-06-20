"""Unit tests for the shared _coerce_estimation_row helper in causal.py.

These tests verify the helper's exact semantics in isolation. The existing
_load_agent_estimation_frame tests (test_causal_brands.py etc.) continue to
pass unchanged, proving P1 behavior is byte-for-byte preserved after the
inline loop is replaced by a call to this helper.
"""

from src.api.routes.causal import _coerce_estimation_row


def _row(**kw):
    return kw


def test_numeric_coercion_and_passthrough():
    rec = _coerce_estimation_row(
        _row(treatment_arm="1", persistent_180d="0", disease_severity="2.5", note="x"),
        select_cols=["treatment_arm", "persistent_180d", "disease_severity", "note"],
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        numeric_cols={"treatment_arm", "persistent_180d", "disease_severity"},
    )
    assert rec == {
        "treatment_arm": 1.0,
        "persistent_180d": 0.0,
        "disease_severity": 2.5,
        "note": "x",
    }


def test_missing_treatment_or_outcome_drops_row():
    assert (
        _coerce_estimation_row(
            _row(treatment_arm=None, persistent_180d="0"),
            select_cols=["treatment_arm", "persistent_180d"],
            treatment_var="treatment_arm",
            outcome_var="persistent_180d",
            numeric_cols={"treatment_arm", "persistent_180d"},
        )
        is None
    )


def test_non_coercible_numeric_becomes_none_and_may_drop():
    # non-coercible treatment -> None -> row dropped
    assert (
        _coerce_estimation_row(
            _row(treatment_arm="abc", persistent_180d="1"),
            select_cols=["treatment_arm", "persistent_180d"],
            treatment_var="treatment_arm",
            outcome_var="persistent_180d",
            numeric_cols={"treatment_arm", "persistent_180d"},
        )
        is None
    )


def test_categorical_passthrough_not_floatcoerced():
    rec = _coerce_estimation_row(
        _row(
            treatment_arm="1",
            persistent_180d="1",
            geographic_region="midwest",
        ),
        select_cols=["treatment_arm", "persistent_180d", "geographic_region"],
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        numeric_cols={"treatment_arm", "persistent_180d", "geographic_region"},
        categorical_cols=frozenset({"geographic_region"}),
    )
    assert rec["geographic_region"] == "midwest"  # NOT float-coerced


def test_derivations_then_fill_zero():
    rec = _coerce_estimation_row(
        _row(control_group_flag=None, action_taken="accepted"),
        select_cols=["control_group_flag", "action_taken"],
        treatment_var="control_group_flag",
        outcome_var="action_taken",
        numeric_cols={"control_group_flag", "action_taken"},
        derivations={"action_taken": lambda v: 1.0 if v == "accepted" else 0.0},
        fill_zero=frozenset({"control_group_flag"}),
    )
    assert rec == {"control_group_flag": 0.0, "action_taken": 1.0}
