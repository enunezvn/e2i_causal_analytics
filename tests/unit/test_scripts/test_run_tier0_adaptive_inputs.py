"""Unit tests for adaptive-criteria input injection in scripts/run_tier0_test.py.

The pipeline runner is responsible for computing the four pre-eval inputs
(n_samples, prevalence, feature_count, regime) from the synthetic data
and injecting them into the scope_definer initial state. Without this
injection, the criteria_validator's ADAPTIVE_CRITERIA path falls back to
fixed thresholds.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest


def test_compute_adaptive_inputs_from_dataframe() -> None:
    """The helper reads patient_df + feature columns + target + regime and
    emits the four-tuple in the exact shape ScopeDefinerState expects.
    """
    runner = importlib.import_module("scripts.run_tier0_test")

    df = pd.DataFrame(
        {
            "patient_id": list(range(900)),
            "feat_a": [0.1] * 900,
            "feat_b": [0.2] * 900,
            "discontinuation_flag": [1] * 270 + [0] * 630,  # prev=0.30
        }
    )
    feature_columns = ["feat_a", "feat_b"]

    inputs = runner._compute_adaptive_state_inputs(
        df=df,
        feature_columns=feature_columns,
        target_col="discontinuation_flag",
        regime="default",
    )

    assert inputs["n_samples"] == 900
    assert inputs["prevalence"] == pytest.approx(0.30, abs=1e-9)
    assert inputs["feature_count"] == 2
    assert inputs["regime"] == "default"
    # Deployment intent defaults to clinical (never silently loosened).
    assert inputs["deployment_intent"] == "clinical"


def test_compute_adaptive_inputs_threads_commercial_intent() -> None:
    """The runner threads --deployment-intent into the scope_definer state so
    the commercial use-case bar (AUC 0.65) is applied end-to-end."""
    runner = importlib.import_module("scripts.run_tier0_test")
    df = pd.DataFrame({"feat_a": [0.0] * 100, "discontinuation_flag": [1] * 11 + [0] * 89})
    inputs = runner._compute_adaptive_state_inputs(
        df=df,
        feature_columns=["feat_a"],
        target_col="discontinuation_flag",
        regime="default",
        deployment_intent="commercial",
    )
    assert inputs["deployment_intent"] == "commercial"


def test_compute_adaptive_inputs_invalid_intent_falls_back_to_clinical() -> None:
    runner = importlib.import_module("scripts.run_tier0_test")
    df = pd.DataFrame({"feat_a": [0.0] * 100, "discontinuation_flag": [1] * 11 + [0] * 89})
    inputs = runner._compute_adaptive_state_inputs(
        df=df,
        feature_columns=["feat_a"],
        target_col="discontinuation_flag",
        regime="default",
        deployment_intent="garbage",
    )
    assert inputs["deployment_intent"] == "clinical"


def test_compute_adaptive_inputs_returns_none_regime_for_unknown_label() -> None:
    """Non-synthetic / unknown regime label ⇒ ``regime=None`` (RWD path)."""
    runner = importlib.import_module("scripts.run_tier0_test")
    df = pd.DataFrame(
        {
            "feat_a": [0.0] * 100,
            "discontinuation_flag": [1] * 50 + [0] * 50,
        }
    )

    inputs = runner._compute_adaptive_state_inputs(
        df=df,
        feature_columns=["feat_a"],
        target_col="discontinuation_flag",
        regime="unknown_label",
    )
    assert inputs["regime"] is None
    assert inputs["n_samples"] == 100
    assert inputs["prevalence"] == pytest.approx(0.50, abs=1e-9)
    assert inputs["feature_count"] == 1


def test_compute_adaptive_inputs_handles_each_valid_regime() -> None:
    """All three valid regime labels round-trip through the helper."""
    runner = importlib.import_module("scripts.run_tier0_test")
    df = pd.DataFrame({"x": [0.0] * 10, "discontinuation_flag": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})

    for regime in ("default", "clean", "adverse"):
        inputs = runner._compute_adaptive_state_inputs(
            df=df,
            feature_columns=["x"],
            target_col="discontinuation_flag",
            regime=regime,
        )
        assert inputs["regime"] == regime, regime
