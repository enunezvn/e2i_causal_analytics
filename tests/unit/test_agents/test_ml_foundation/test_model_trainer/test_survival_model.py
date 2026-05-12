"""Unit tests for v5 Gate B2 survival model helper + LangGraph node.

Pre-spec: docs/specs/v5_b2_survival_modeling_prespec_2026-05-12.md.
Implementation: src/agents/ml_foundation/model_trainer/nodes/survival_model.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.model_trainer.nodes.survival_model import (
    _derive_admin_censored_target,
    _derive_csu_survival_target,
    derive_survival_target,
    fit_cox,
    fit_rsf,
    survival_concordance,
    survival_model_node,
)


# === Synthetic fixtures ============================================


@pytest.fixture
def synthetic_csu_journey():
    """Patient journeys frame mirroring CSU schema for survival derivation."""
    return pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(10)],
            "treatment_initiated": [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
            "journey_duration_days": [120, 300, 50, 400, 180, 365, 90, 200, 250, 365],
        }
    )


@pytest.fixture
def synthetic_csu_events():
    """Treatment events with mix of rx (positive cases) + lab tests + pre-index rx."""
    return pd.DataFrame(
        {
            "patient_id": ["P0", "P2", "P4", "P6", "P7", "P0", "P2", "P8"],
            "event_type": [
                "prescription",
                "prescription",
                "prescription",
                "prescription",
                "prescription",
                "lab_test",  # ignored
                "prescription",  # pre-index, ignored for time
                "prescription",  # P8 is not in positives -> shouldn't affect
            ],
            "days_from_diagnosis": [60, 30, 90, 45, 150, 10, -20, 200],
        }
    )


@pytest.fixture
def synthetic_optum_journey():
    """Constant-time admin-censored optum cohort."""
    return pd.DataFrame(
        {
            "patient_id": [f"O{i}" for i in range(10)],
            "treatment_initiated": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        }
    )


# === Derivation tests ==============================================


def test_csu_derivation_uses_post_index_rx_for_events(synthetic_csu_journey, synthetic_csu_events):
    t, e = _derive_csu_survival_target(synthetic_csu_journey, synthetic_csu_events)
    assert t.shape == (10,)
    assert e.shape == (10,)
    # P0 (event=1, rx at 60d) -> time=60
    assert t[0] == 60.0
    # P2 (event=1, rx at 30d) -> time=30
    assert t[2] == 30.0
    # P4 (event=1, rx at 90d) -> time=90
    assert t[4] == 90.0
    # P6 (event=1, rx at 45d) -> time=45
    assert t[6] == 45.0
    # P7 (event=1, rx at 150d) -> time=150
    assert t[7] == 150.0
    # P1 (event=0, jd=300) -> time=300 (within cap of 365)
    assert t[1] == 300.0
    # Event array correct
    np.testing.assert_array_equal(e.astype(int), [1, 0, 1, 0, 1, 0, 1, 1, 0, 0])


def test_csu_derivation_caps_times_at_365(synthetic_csu_journey, synthetic_csu_events):
    # P3 (event=0, jd=400) -> capped at 365
    t, _ = _derive_csu_survival_target(synthetic_csu_journey, synthetic_csu_events)
    assert t[3] == 365.0


def test_csu_derivation_ignores_pre_index_rx(synthetic_csu_journey, synthetic_csu_events):
    """P2's pre-index rx (-20d) must NOT be used as the event time."""
    t, _ = _derive_csu_survival_target(synthetic_csu_journey, synthetic_csu_events)
    # P2 should use its 30d post-index rx, not its -20d pre-index rx.
    assert t[2] == 30.0


def test_csu_derivation_event_without_rx_falls_back_to_journey_duration():
    """Edge case: event=1 patient with NO post-index rx falls back to censoring time."""
    pj = pd.DataFrame(
        {
            "patient_id": ["P0"],
            "treatment_initiated": [1],
            "journey_duration_days": [100],
        }
    )
    ev = pd.DataFrame({"patient_id": [], "event_type": [], "days_from_diagnosis": []})
    t, e = _derive_csu_survival_target(pj, ev)
    # No rx event found; falls back to journey_duration_days=100.
    assert t[0] == 100.0
    assert e[0] == True  # noqa: E712


def test_csu_derivation_handles_missing_events_arg():
    """When treatment_events is None, falls back to journey_duration censoring only."""
    pj = pd.DataFrame(
        {
            "patient_id": ["P0", "P1"],
            "treatment_initiated": [1, 0],
            "journey_duration_days": [120, 200],
        }
    )
    t, e = _derive_csu_survival_target(pj, treatment_events=None)
    assert t[0] == 120.0
    assert t[1] == 200.0


def test_csu_derivation_zero_or_negative_time_clamped():
    """times of 0 or negative must clamp to 1 (sksurv requires strictly positive)."""
    pj = pd.DataFrame(
        {
            "patient_id": ["P0", "P1"],
            "treatment_initiated": [1, 0],
            "journey_duration_days": [0, -5],
        }
    )
    ev = pd.DataFrame({"patient_id": [], "event_type": [], "days_from_diagnosis": []})
    t, _ = _derive_csu_survival_target(pj, ev)
    assert (t > 0).all()


def test_optum_derivation_constant_180d(synthetic_optum_journey):
    t, e = _derive_admin_censored_target(synthetic_optum_journey)
    assert t.shape == (10,)
    # All times must be the default 180d.
    assert np.all(t == 180.0)
    np.testing.assert_array_equal(e.astype(int), [1, 0, 0, 1, 0, 0, 0, 1, 0, 0])


def test_optum_derivation_admin_horizon_configurable(synthetic_optum_journey):
    t, _ = _derive_admin_censored_target(synthetic_optum_journey, admin_censoring_days=365)
    assert np.all(t == 365.0)


def test_dispatch_csu(synthetic_csu_journey, synthetic_csu_events):
    t, e = derive_survival_target(synthetic_csu_journey, "csu", treatment_events=synthetic_csu_events)
    assert t.shape == (10,)
    assert e.shape == (10,)


def test_dispatch_optum(synthetic_optum_journey):
    t, e = derive_survival_target(synthetic_optum_journey, "optum")
    assert np.all(t == 180.0)


def test_dispatch_unknown_manifest_raises():
    pj = pd.DataFrame({"patient_id": ["P0"], "treatment_initiated": [1]})
    with pytest.raises(ValueError, match="unknown manifest_source"):
        derive_survival_target(pj, "unknown")


def test_dispatch_object_dtype_days_from_diagnosis():
    """Loaders that return days_from_diagnosis as object dtype must coerce."""
    pj = pd.DataFrame(
        {
            "patient_id": ["P0"],
            "treatment_initiated": [1],
            "journey_duration_days": [100],
        }
    )
    ev = pd.DataFrame(
        {
            "patient_id": ["P0"],
            "event_type": ["prescription"],
            "days_from_diagnosis": ["50"],  # string
        }
    )
    t, _ = derive_survival_target(pj, "csu", treatment_events=ev)
    assert t[0] == 50.0


# === Model fitting tests ===========================================


@pytest.fixture
def synthetic_survival_data():
    """Generate a small survival dataset where x0 is informative."""
    rng = np.random.default_rng(42)
    n = 200
    # x0 = risk factor; higher x0 -> earlier event time.
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)  # noise
    # Latent time scales inversely with risk.
    latent_t = rng.exponential(scale=np.exp(-0.5 * x0))
    # Administrative censoring at t=1.0.
    time = np.minimum(latent_t, 1.0)
    event = latent_t <= 1.0
    X = pd.DataFrame({"x0": x0, "x1": x1})
    return X, time, event


def test_fit_cox_returns_concordance_above_chance(synthetic_survival_data):
    X, t, e = synthetic_survival_data
    cox = fit_cox(X, t, e)
    c = survival_concordance(cox, X, t, e)
    # In-sample c on an informative feature must beat random (0.5).
    assert c > 0.6


def test_fit_rsf_returns_concordance_above_chance(synthetic_survival_data):
    X, t, e = synthetic_survival_data
    rsf = fit_rsf(X, t, e, n_estimators=50)
    c = survival_concordance(rsf, X, t, e)
    assert c > 0.55


def test_fit_cox_alpha_regularizes_collinear_features():
    """With strongly collinear features, alpha=1e-3 should still converge."""
    rng = np.random.default_rng(42)
    n = 100
    x0 = rng.normal(size=n)
    x0_copy = x0 + 1e-6 * rng.normal(size=n)  # near-duplicate
    X = pd.DataFrame({"x0": x0, "x0_copy": x0_copy})
    latent = rng.exponential(scale=np.exp(-0.5 * x0))
    time = np.minimum(latent, 1.0)
    event = latent <= 1.0
    # Should fit without ConvergenceError thanks to alpha.
    cox = fit_cox(X, time, event, alpha=1e-3)
    assert cox is not None


def test_concordance_is_in_zero_one(synthetic_survival_data):
    X, t, e = synthetic_survival_data
    cox = fit_cox(X, t, e)
    c = survival_concordance(cox, X, t, e)
    assert 0.0 <= c <= 1.0


def test_fit_rsf_raises_on_constant_time():
    """RSF predict path is degenerate when all training times are equal.

    Pre-spec §4 (Optum): administrative censoring at constant 180d
    triggers this. Cleaner to raise an informative ValueError than to
    let sksurv crash with an obscure IndexError downstream in predict.
    """
    rng = np.random.default_rng(42)
    n = 40
    X = pd.DataFrame({"x0": rng.normal(size=n)})
    t = np.full(n, 180.0)  # CONSTANT
    e = rng.random(n) < 0.3
    with pytest.raises(ValueError, match="constant-time"):
        fit_rsf(X, t, e, n_estimators=10)


def test_cross_val_skips_single_class_validation_folds():
    """L2 codex pass-1: zero-event val folds must be skipped, not crashed on.

    Validates the skip-degenerate-fold guard in
    scripts/measure_b2_cindex_contrast.py._cross_val. The contrast
    script has no unit-test entry point (it's a script), so we test
    the guard indirectly by constructing a tiny cohort where some
    folds will have single-class val sets.
    """
    # Note: this is an integration-style smoke test using the script's
    # helper directly. Stratified KFold should distribute events across
    # folds, but with only 2 events out of 100 patients some folds will
    # have 0 val events (single-class). The _cross_val guard must skip
    # those folds and report n_folds < 5.
    from scripts.measure_b2_cindex_contrast import _cross_val

    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    y = np.zeros(n, dtype=int)
    y[:2] = 1  # Only 2 positives.
    time = np.where(y == 1, 50.0, 100.0)
    event = y.astype(bool)
    metrics = _cross_val(X, y, time, event, seed=42)
    # With 2 positives spread across 5 folds, at most 2 folds can have
    # a positive val sample. Other folds skip.
    # The skip-guard fires when y_va_bin has only one class.
    assert metrics["binary_auc"]["n_folds"] <= 2


def test_fit_rsf_works_with_two_unique_times():
    """RSF should work the moment there is variation in time."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({"x0": rng.normal(size=n)})
    t = np.where(rng.random(n) < 0.5, 100.0, 200.0)
    e = rng.random(n) < 0.5
    rsf = fit_rsf(X, t, e, n_estimators=10)
    assert rsf is not None


# === LangGraph node tests ==========================================


@pytest.mark.asyncio
async def test_node_no_op_when_gate_disabled():
    """Default enable_survival_modeling=False yields empty patch."""
    state = {"enable_survival_modeling": False}
    patch = await survival_model_node(state)
    assert patch == {}


@pytest.mark.asyncio
async def test_node_returns_patch_for_csu(synthetic_csu_journey, synthetic_csu_events):
    state = {
        "enable_survival_modeling": True,
        "manifest_source": "csu",
        "patient_journeys_df": synthetic_csu_journey,
        "treatment_events_df": synthetic_csu_events,
    }
    patch = await survival_model_node(state)
    assert "survival_time_days" in patch
    assert "survival_event" in patch
    assert patch["survival_manifest_source"] == "csu"
    assert patch["survival_time_days"].shape == (10,)


@pytest.mark.asyncio
async def test_node_returns_patch_for_optum(synthetic_optum_journey):
    state = {
        "enable_survival_modeling": True,
        "manifest_source": "optum",
        "patient_journeys_df": synthetic_optum_journey,
        "treatment_events_df": None,
    }
    patch = await survival_model_node(state)
    assert "survival_time_days" in patch
    assert np.all(patch["survival_time_days"] == 180.0)


@pytest.mark.asyncio
async def test_node_does_not_mutate_state_in_place(synthetic_csu_journey, synthetic_csu_events):
    """B3 H3 lesson: node must return a patch, not mutate state.

    M2 codex pass-1 strengthening: also verify the input DataFrames
    themselves are not mutated (object identity preserved + column
    set + row count + dtypes), so an in-place sort/append regression
    in a future refactor would be caught.
    """
    state = {
        "enable_survival_modeling": True,
        "manifest_source": "csu",
        "patient_journeys_df": synthetic_csu_journey,
        "treatment_events_df": synthetic_csu_events,
    }
    state_before_keys = set(state.keys())
    pj_id_before = id(state["patient_journeys_df"])
    ev_id_before = id(state["treatment_events_df"])
    pj_cols_before = set(state["patient_journeys_df"].columns)
    ev_cols_before = set(state["treatment_events_df"].columns)
    pj_rows_before = len(state["patient_journeys_df"])
    ev_rows_before = len(state["treatment_events_df"])
    pj_dtypes_before = dict(state["patient_journeys_df"].dtypes)
    ev_dtypes_before = dict(state["treatment_events_df"].dtypes)

    patch = await survival_model_node(state)

    # State key-set unchanged.
    assert set(state.keys()) == state_before_keys
    # Frames not replaced.
    assert id(state["patient_journeys_df"]) == pj_id_before
    assert id(state["treatment_events_df"]) == ev_id_before
    # Frame columns / rows / dtypes preserved.
    assert set(state["patient_journeys_df"].columns) == pj_cols_before
    assert set(state["treatment_events_df"].columns) == ev_cols_before
    assert len(state["patient_journeys_df"]) == pj_rows_before
    assert len(state["treatment_events_df"]) == ev_rows_before
    assert dict(state["patient_journeys_df"].dtypes) == pj_dtypes_before
    assert dict(state["treatment_events_df"].dtypes) == ev_dtypes_before
    # Patch carries the survival fields.
    assert "survival_time_days" in patch


@pytest.mark.asyncio
async def test_node_unknown_manifest_returns_empty_patch():
    state = {
        "enable_survival_modeling": True,
        "manifest_source": "garbage",
        "patient_journeys_df": pd.DataFrame({"patient_id": ["P0"], "treatment_initiated": [1]}),
    }
    patch = await survival_model_node(state)
    assert patch == {}


@pytest.mark.asyncio
async def test_node_empty_journey_returns_empty_patch():
    state = {
        "enable_survival_modeling": True,
        "manifest_source": "csu",
        "patient_journeys_df": pd.DataFrame(),
    }
    patch = await survival_model_node(state)
    assert patch == {}


@pytest.mark.asyncio
async def test_node_surfaces_derivation_error_in_patch():
    """If derivation raises unexpectedly, the node surfaces the error in the patch."""
    # Patient journey missing required treatment_initiated column.
    state = {
        "enable_survival_modeling": True,
        "manifest_source": "csu",
        "patient_journeys_df": pd.DataFrame({"patient_id": ["P0"]}),
    }
    patch = await survival_model_node(state)
    assert "survival_target_error" in patch
