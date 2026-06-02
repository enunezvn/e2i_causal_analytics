"""Tests for success criteria validation."""

import os
from unittest.mock import patch

import pytest

from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
    _define_baseline_model,
    _define_classification_criteria,
    _define_regression_criteria,
    _validate_criteria,
    define_success_criteria,
)


@pytest.mark.asyncio
async def test_define_classification_criteria():
    """Test success criteria for classification problems."""
    state = {
        "inferred_problem_type": "binary_classification",
        "experiment_id": "exp_test_123",
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should have classification metrics
    assert "minimum_auc" in criteria
    assert "minimum_precision" in criteria
    assert "minimum_recall" in criteria
    assert "minimum_f1" in criteria

    # Should NOT have regression metrics
    assert criteria["minimum_rmse"] is None
    assert criteria["minimum_r2"] is None

    # Should have baseline. Section B (pre_phase2_unblockers) corrected the
    # metadata label to match what evaluator._compute_baseline_test_metrics
    # actually computes (a stratified DummyClassifier).
    assert criteria["baseline_model"] == "stratified_dummy"

    # Section B: minimum_lift_over_baseline must be injected by default so
    # the evaluator gets a chance to compute and compare the metric.
    assert criteria["minimum_lift_over_baseline"] == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_define_regression_criteria():
    """Test success criteria for regression problems."""
    state = {
        "inferred_problem_type": "regression",
        "experiment_id": "exp_test_123",
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should have regression metrics
    assert "minimum_rmse" in criteria
    assert "minimum_r2" in criteria
    assert "minimum_mape" in criteria

    # Should NOT have classification metrics
    assert criteria["minimum_auc"] is None
    assert criteria["minimum_precision"] is None

    # Should have baseline
    assert criteria["baseline_model"] == "linear_regression_baseline"


@pytest.mark.asyncio
async def test_define_causal_criteria():
    """Test success criteria for causal inference problems."""
    state = {
        "inferred_problem_type": "causal_inference",
        "experiment_id": "exp_test_123",
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should use RMSE for ATE standard error
    assert "minimum_rmse" in criteria
    assert "minimum_r2" in criteria

    # Should have causal baseline
    assert criteria["baseline_model"] == "ols_baseline"


@pytest.mark.asyncio
async def test_performance_requirements_override_defaults():
    """Test that explicit performance requirements override defaults."""
    state = {
        "inferred_problem_type": "binary_classification",
        "experiment_id": "exp_test_123",
        "performance_requirements": {
            "min_auc": 0.85,
            "min_precision": 0.80,
            "min_recall": 0.75,
            "min_f1": 0.78,
        },
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should use custom thresholds
    assert criteria["minimum_auc"] == 0.85
    assert criteria["minimum_precision"] == 0.80
    assert criteria["minimum_recall"] == 0.75
    assert criteria["minimum_f1"] == 0.78


@pytest.mark.asyncio
async def test_minimum_lift_over_baseline():
    """Test that minimum lift over baseline is set."""
    state = {
        "inferred_problem_type": "binary_classification",
        "experiment_id": "exp_test_123",
        "performance_requirements": {"min_lift": 0.15},
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should have 15% lift requirement
    assert criteria["minimum_lift_over_baseline"] == 0.15


@pytest.mark.asyncio
async def test_default_minimum_lift():
    """Test default minimum lift is 10%."""
    state = {
        "inferred_problem_type": "binary_classification",
        "experiment_id": "exp_test_123",
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should default to 10% lift
    assert criteria["minimum_lift_over_baseline"] == 0.10


def test_classification_criteria_defaults():
    """Test default thresholds for classification."""
    criteria = _define_classification_criteria({})

    # Check reasonable defaults
    assert 0.5 <= criteria["minimum_auc"] <= 0.9
    assert 0.5 <= criteria["minimum_precision"] <= 0.9
    assert 0.5 <= criteria["minimum_recall"] <= 0.9
    assert 0.5 <= criteria["minimum_f1"] <= 0.9


def test_regression_criteria_defaults():
    """Test default thresholds for regression."""
    criteria = _define_regression_criteria({})

    # Check reasonable defaults
    assert criteria["minimum_r2"] >= 0.5
    assert criteria["minimum_rmse"] is not None
    assert criteria["minimum_mape"] is not None


def test_baseline_model_selection():
    """Test baseline model selection for different problem types."""
    # Classification
    # Section B (pre_phase2_unblockers): aligned the binary/multiclass
    # baseline label with the stratified-dummy actually used by the
    # evaluator's _compute_baseline_test_metrics helper.
    assert _define_baseline_model("binary_classification") == "stratified_dummy"
    assert _define_baseline_model("multiclass_classification") == "stratified_dummy"

    # Regression
    assert _define_baseline_model("regression") == "linear_regression_baseline"

    # Causal
    assert _define_baseline_model("causal_inference") == "ols_baseline"

    # Time series
    assert _define_baseline_model("time_series") == "arima_baseline"


def test_validate_criteria_warns_on_low_samples():
    """Test validation warns when minimum samples is too low."""
    criteria = {"experiment_id": "exp_test_123", "baseline_model": "test"}

    state = {"scope_spec": {"minimum_samples": 50}}  # Very low

    result = _validate_criteria(criteria, state)

    # Should pass but warn
    assert result["passed"] is True
    assert len(result["warnings"]) > 0
    assert any("sample" in w.lower() for w in result["warnings"])


def test_validate_criteria_warns_on_high_auc():
    """Test validation warns when AUC threshold is unrealistic."""
    criteria = {
        "experiment_id": "exp_test_123",
        "baseline_model": "test",
        "minimum_auc": 0.98,  # Very high
    }

    state = {"scope_spec": {"minimum_samples": 1000}}

    result = _validate_criteria(criteria, state)

    # Should pass but warn
    assert result["passed"] is True
    assert len(result["warnings"]) > 0
    assert any("auc" in w.lower() for w in result["warnings"])


def test_validate_criteria_warns_on_high_r2():
    """Test validation warns when R² threshold is unrealistic."""
    criteria = {
        "experiment_id": "exp_test_123",
        "baseline_model": "test",
        "minimum_r2": 0.95,  # Very high
    }

    state = {"scope_spec": {"minimum_samples": 1000}}

    result = _validate_criteria(criteria, state)

    # Should pass but warn
    assert result["passed"] is True
    assert len(result["warnings"]) > 0
    assert any("r²" in w.lower() or "r2" in w.lower() for w in result["warnings"])


def test_validate_criteria_warns_on_low_time_budget():
    """Test validation warns when time budget is too low."""
    criteria = {
        "experiment_id": "exp_test_123",
        "baseline_model": "test",
    }

    state = {
        "scope_spec": {"minimum_samples": 1000},
        "time_budget_hours": 0.5,  # 30 minutes - very low
    }

    result = _validate_criteria(criteria, state)

    # Should pass but warn
    assert result["passed"] is True
    assert len(result["warnings"]) > 0
    assert any("time" in w.lower() for w in result["warnings"])


def test_validate_criteria_fails_on_missing_experiment_id():
    """Test validation fails when experiment_id is missing."""
    criteria = {
        "baseline_model": "test",
        # Missing experiment_id
    }

    state = {"scope_spec": {"minimum_samples": 1000}}

    result = _validate_criteria(criteria, state)

    # Should fail
    assert result["passed"] is False
    assert len(result["errors"]) > 0
    assert any("experiment_id" in e.lower() for e in result["errors"])


def test_validate_criteria_fails_on_missing_baseline():
    """Test validation fails when baseline_model is missing."""
    criteria = {
        "experiment_id": "exp_test_123",
        # Missing baseline_model
    }

    state = {"scope_spec": {"minimum_samples": 1000}}

    result = _validate_criteria(criteria, state)

    # Should fail
    assert result["passed"] is False
    assert len(result["errors"]) > 0
    assert any("baseline" in e.lower() for e in result["errors"])


@pytest.mark.asyncio
async def test_validation_passed_flag_set():
    """Test that validation_passed flag is correctly set."""
    # Valid state
    state = {
        "inferred_problem_type": "binary_classification",
        "experiment_id": "exp_test_123",
        "scope_spec": {"minimum_samples": 1000},
    }

    result = await define_success_criteria(state)

    # Should pass validation
    assert result["validation_passed"] is True
    assert len(result["validation_errors"]) == 0


@pytest.mark.asyncio
async def test_validation_errors_populated_on_failure():
    """Test that validation_errors are populated when validation fails."""
    # Missing experiment_id
    state = {
        "inferred_problem_type": "binary_classification",
        # No experiment_id
        "scope_spec": {"minimum_samples": 1000},
    }

    result = await define_success_criteria(state)

    # Should fail validation
    assert result["validation_passed"] is False
    assert len(result["validation_errors"]) > 0


# ---------------------------------------------------------------------------
# ADAPTIVE_CRITERIA flag + criteria_source audit field (task 02 of
# .claude/plans/adaptive_success_criteria/)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_criteria_source_defaults_to_adaptive_when_flag_unset() -> None:
    """PRODUCTION DEFAULT (2026-06-02): ADAPTIVE_CRITERIA defaults to true,
    so an UNSET flag now routes binary-classification scopes to the v3
    adaptive engine. With NO pre-eval inputs on state the validator cannot
    compute or stash adaptive thresholds, so it falls back to fixed
    thresholds and tags the audit value ``adaptive_fallback_to_fixed`` (the
    gap is observable). Pre-flip this test asserted ``"fixed"``; the flip
    changes the DEFAULT, not the threshold values when state is incomplete.
    """
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ADAPTIVE_CRITERIA", None)
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-default-on",
        }
        result = await define_success_criteria(state)
    sc = result["success_criteria"]
    assert sc["criteria_source"] == "adaptive_fallback_to_fixed"
    # Fixed thresholds still computed (the flip changes routing, not the
    # fallback values) — the precision/F1 gates are NOT dropped on the
    # fallback path so Apr-26 reproducibility holds when state is incomplete.
    assert sc["minimum_auc"] == 0.75
    assert sc["minimum_precision"] == 0.70
    assert sc["minimum_f1"] == 0.70


@pytest.mark.asyncio
async def test_criteria_source_is_fixed_when_flag_explicitly_off() -> None:
    """The explicit opt-OUT (ADAPTIVE_CRITERIA=false) still produces the
    fixed scheme tagged ``"fixed"`` — this is the ROLLBACK switch."""
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": "false"}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-optout",
        }
        result = await define_success_criteria(state)
    assert result["success_criteria"]["criteria_source"] == "fixed"


@pytest.mark.asyncio
async def test_criteria_source_is_adaptive_when_flag_true() -> None:
    """ADAPTIVE_CRITERIA=true + complete pre-eval state ⇒ tagged 'adaptive'.

    The fixture includes ALL five pre-eval inputs (n_samples / prevalence /
    baseline_auc / feature_count / regime) so this test stays green after
    task 03 wires the three-valued ``criteria_source`` logic — task 03
    emits ``"adaptive_fallback_to_fixed"`` when the flag is on but state
    is incomplete.
    """
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": "true"}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-adapt",
            "n_samples": 900,
            "prevalence": 0.30,
            "baseline_auc": 0.50,
            "feature_count": 14,
            "regime": "default",
        }
        result = await define_success_criteria(state)
    assert result["success_criteria"]["criteria_source"] == "adaptive"


@pytest.mark.parametrize("falsy", ["false", "0", "no", "", "FALSE"])
@pytest.mark.asyncio
async def test_criteria_source_falsy_values_keep_fixed(falsy: str) -> None:
    """Conventional falsy strings keep the flag off ⇒ criteria_source=='fixed'."""
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": falsy}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-fixed-falsy",
            "n_samples": 900,
            "prevalence": 0.30,
            "feature_count": 14,
            "regime": "default",
        }
        result = await define_success_criteria(state)
    assert result["success_criteria"]["criteria_source"] == "fixed"


# ---------------------------------------------------------------------------
# Validator branch — adaptive path wiring (task 03 of
# .claude/plans/adaptive_success_criteria/)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adaptive_path_used_when_flag_on_with_full_state() -> None:
    """Flag on AND state has dataset characteristics ⇒ adaptive thresholds.

    v3 (Option C): precision/F1 are dropped entirely — they are NOT in
    success_criteria when adaptive succeeds. Skipped criteria (e.g.,
    lift at adverse) are absent from the dict and recorded on
    ``_adaptive_skipped``. ``_adaptive_p_t`` carries the regime's
    threshold probability.
    """
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": "true"}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-adapt-full",
            "n_samples": 900,
            "prevalence": 0.02,
            "baseline_auc": 0.50,
            "feature_count": 14,
            "regime": "adverse",
        }
        result = await define_success_criteria(state)

    sc = result["success_criteria"]
    assert sc["criteria_source"] == "adaptive"
    # v3 drops precision/F1 entirely (Van Calster 2025).
    assert "minimum_precision" not in sc
    assert "minimum_f1" not in sc
    # Adverse regime: lift skipped because n_pos=18 ⇒ 2*SE > 0.10.
    assert "minimum_lift_over_baseline" not in sc
    # Skipped names recorded; v3 only skips lift at adverse N=900 prev=0.02.
    assert sc["_adaptive_skipped"] == ["minimum_lift_over_baseline"]
    # Adverse-keyed thresholds fire.
    assert sc["minimum_recall"] == pytest.approx(0.50, abs=1e-6)
    assert sc["minimum_auc"] == pytest.approx(0.70, abs=1e-6)
    assert sc["minimum_mcc"] == pytest.approx(0.20, abs=1e-6)
    assert sc["minimum_net_benefit_at_p_t"] == pytest.approx(0.0, abs=1e-6)
    assert sc["maximum_calibration_slope_deviation"] == pytest.approx(0.15, abs=1e-6)
    assert sc["maximum_calibration_intercept_magnitude"] == pytest.approx(0.30, abs=1e-6)
    # v3 audit field for the regime-keyed threshold probability.
    assert sc["_adaptive_p_t"] == pytest.approx(0.05, abs=1e-6)


@pytest.mark.asyncio
async def test_adaptive_fallback_when_state_incomplete() -> None:
    """Flag on but state missing dataset characteristics ⇒ fall back to
    fixed defaults. The audit value is the v2 third option
    ``"adaptive_fallback_to_fixed"`` so the gap is loud in audit logs.
    """
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": "true"}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-adapt-incomplete",
            # Note: no n_samples / prevalence / baseline_auc / feature_count
        }
        result = await define_success_criteria(state)

    sc = result["success_criteria"]
    # Fall back to fixed thresholds (precision/F1 retained per B2 fix
    # so flag-OFF / Apr-26 baseline reproduces).
    assert sc["minimum_auc"] == 0.75
    assert sc["minimum_precision"] == 0.70
    assert sc["minimum_recall"] == 0.65
    assert sc["minimum_f1"] == 0.70
    # ...and the audit tag is the v2 third value (NOT "adaptive").
    assert sc["criteria_source"] == "adaptive_fallback_to_fixed"
    # Adaptive-only audit fields absent in fallback.
    assert "_adaptive_skipped" not in sc
    assert "_adaptive_p_t" not in sc


@pytest.mark.asyncio
async def test_flag_off_reproduces_fixed_thresholds_exactly() -> None:
    """The Apr-26 baseline guarantee: flag OFF ⇒ exactly the historical
    fixed dict, regardless of state['regime'] / state['n_samples'].
    """
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": "false"}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-fixed-with-state",
            "n_samples": 900,
            "prevalence": 0.02,
            "baseline_auc": 0.50,
            "feature_count": 14,
            "regime": "adverse",
        }
        result = await define_success_criteria(state)

    sc = result["success_criteria"]
    assert sc["criteria_source"] == "fixed"
    assert sc["minimum_auc"] == 0.75  # not the adaptive 0.70
    assert sc["minimum_precision"] == 0.70  # NOT dropped under fixed mode
    assert sc["minimum_recall"] == 0.65
    assert sc["minimum_f1"] == 0.70  # NOT dropped under fixed mode
    assert sc["minimum_lift_over_baseline"] == 0.10
    # v3 active gates absent under fixed (only adaptive populates them).
    assert "minimum_net_benefit_at_p_t" not in sc
    assert "minimum_mcc" not in sc
    assert "maximum_calibration_slope_deviation" not in sc
    assert "maximum_calibration_intercept_magnitude" not in sc
    assert "maximum_calibration_error" not in sc
    assert "maximum_train_val_delta" not in sc
    assert "_adaptive_skipped" not in sc
    assert "_adaptive_p_t" not in sc


@pytest.mark.parametrize(
    "regime,expected_p_t",
    [
        ("adverse", 0.05),
        ("default", 0.20),
        ("clean", 0.30),
        (None, 0.30),  # RWD: regime=None ⇒ treated as clean
    ],
)
@pytest.mark.asyncio
async def test_adaptive_p_t_audit_value_per_regime(regime: object, expected_p_t: float) -> None:
    """v3 audit field ``_adaptive_p_t`` carries the regime-keyed threshold
    probability used for the NB > 0 gate. Vickers 2019 cost-ratio defaults.
    """
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": "true"}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-pt-audit",
            "n_samples": 900,
            "prevalence": 0.30,  # nonzero so adverse path picks p_t=0.05 only via regime
            "baseline_auc": 0.50,
            "feature_count": 14,
            "regime": regime,
        }
        result = await define_success_criteria(state)

    sc = result["success_criteria"]
    assert sc["criteria_source"] == "adaptive"
    assert sc["_adaptive_p_t"] == pytest.approx(expected_p_t, abs=1e-6)


@pytest.mark.asyncio
async def test_adaptive_default_regime_drops_auc_from_success_criteria() -> None:
    """Default-regime adaptive: ``minimum_auc`` is popped from
    success_criteria (skipped via the explicit set), not retained from the
    fixed dict.
    """
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": "true"}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-default-adapt",
            "n_samples": 900,
            "prevalence": 0.30,
            "baseline_auc": 0.50,
            "feature_count": 14,
            "regime": "default",
        }
        result = await define_success_criteria(state)

    sc = result["success_criteria"]
    assert sc["criteria_source"] == "adaptive"
    assert "minimum_auc" not in sc
    assert "minimum_auc" in sc["_adaptive_skipped"]


# ---------------------------------------------------------------------------
# Config-typo defense (S4 fix) — `_define_classification_criteria`
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Two-phase stash path (task 05 of adaptive_success_criteria/) — the
# production case where state has the 3 pre-eval inputs but NOT
# baseline_auc, so the validator stashes inputs for the evaluator overlay.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adaptive_inputs_stashed_when_pre_eval_inputs_present_no_baseline_auc() -> None:
    """v3 production path: state has n_samples / prevalence / feature_count
    (and regime) but NOT baseline_auc. The validator stashes the inputs on
    ``success_criteria['_adaptive_inputs']`` for the evaluator overlay to
    pick up later. ``criteria_source`` tags ``"adaptive"`` (the overlay
    will fill in actual thresholds and ``_adaptive_p_t`` at eval time).
    """
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": "true"}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-stash",
            "n_samples": 900,
            "prevalence": 0.30,
            "feature_count": 14,
            "regime": "default",
            # baseline_auc DELIBERATELY ABSENT — production case
        }
        result = await define_success_criteria(state)

    sc = result["success_criteria"]
    assert sc["criteria_source"] == "adaptive"
    assert "_adaptive_inputs" in sc
    inputs = sc["_adaptive_inputs"]
    assert inputs["n_samples"] == 900
    assert inputs["prevalence"] == pytest.approx(0.30, abs=1e-9)
    assert inputs["feature_count"] == 14
    assert inputs["regime"] == "default"
    # No _adaptive_skipped or _adaptive_p_t yet — overlay sets them later.
    assert "_adaptive_skipped" not in sc
    assert "_adaptive_p_t" not in sc
    # Fixed thresholds remain in place pending overlay.
    assert sc["minimum_auc"] == 0.75
    assert sc["minimum_precision"] == 0.70
    assert sc["minimum_f1"] == 0.70


@pytest.mark.asyncio
async def test_adaptive_inputs_absent_when_flag_off() -> None:
    """Flag off ⇒ no ``_adaptive_inputs`` sneaks into the dict."""
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": "false"}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-no-inputs",
            "n_samples": 900,
            "prevalence": 0.30,
            "feature_count": 14,
            "regime": "default",
        }
        result = await define_success_criteria(state)

    assert "_adaptive_inputs" not in result["success_criteria"]


@pytest.mark.asyncio
async def test_pre_eval_inputs_incomplete_falls_back() -> None:
    """If pre-eval inputs are incomplete (missing one of n_samples /
    prevalence / feature_count), fall back to fixed with the third audit
    value.
    """
    with patch.dict(os.environ, {"ADAPTIVE_CRITERIA": "true"}, clear=False):
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {},
            "experiment_id": "exp-incomplete-stash",
            "n_samples": 900,
            "prevalence": 0.30,
            # feature_count and regime missing
        }
        result = await define_success_criteria(state)

    sc = result["success_criteria"]
    assert sc["criteria_source"] == "adaptive_fallback_to_fixed"
    assert "_adaptive_inputs" not in sc
    assert sc["minimum_auc"] == 0.75


@pytest.mark.asyncio
async def test_none_threshold_in_performance_requirements_falls_back_to_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A misconfigured ``min_auc: null`` upstream must NOT silently bypass
    the AUC gate. The validator detects None thresholds for binary-
    classification expected criteria, warns, and falls back to the safe
    default.
    """
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ADAPTIVE_CRITERIA", None)
        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {"min_auc": None, "min_recall": 0.65},
            "experiment_id": "exp-typo",
        }
        with caplog.at_level("WARNING"):
            result = await define_success_criteria(state)

    sc = result["success_criteria"]
    # Safe default applied, not None.
    assert sc["minimum_auc"] == 0.75
    # Warning emitted (substring match — the message content is the
    # operator-visible part).
    assert any(
        "min_auc=None" in rec.getMessage() and "config typo" in rec.getMessage()
        for rec in caplog.records
    )
