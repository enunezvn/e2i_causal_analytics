"""Tests for the adaptive_validity_check node — Layer 5 pipeline integration.

This node runs Layers 1+3 (and optionally Layer 4 when LM configured) on the
data_preparer's transformed features, emitting a structured LeakageVerdict
per feature. It augments the existing detect_leakage findings without
replacing them, providing the audit trail required by acceptance criterion
#4 of adaptive_temporal_validity_redesign.md.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd
import pytest


def _make_state(train_df: pd.DataFrame, target: str, **overrides: Any) -> dict:
    """Build a minimal DataPreparerState dict for the node."""
    state: dict = {
        "experiment_id": "test-exp",
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": target,
            "required_features": [c for c in train_df.columns if c != target],
            "excluded_features": [],
        },
        "leakage_findings": [],
        "leaked_features": [],
    }
    state.update(overrides)
    return state


def _run(state: dict) -> dict:
    """Run the node synchronously via asyncio."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )

    return asyncio.run(adaptive_validity_check(state))


def test_node_returns_required_keys():
    """Node returns the documented state-update keys."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "noise_a": rng.standard_normal(200),
            "noise_b": rng.standard_normal(200),
            "y": rng.integers(0, 2, 200),
        }
    )
    state = _make_state(df, "y")
    result = _run(state)
    assert "adaptive_verdicts" in result
    assert "adaptive_flagged_features" in result
    assert isinstance(result["adaptive_verdicts"], list)
    assert isinstance(result["adaptive_flagged_features"], list)


def test_node_flags_obvious_leak_via_layer_3():
    """A feature perfectly correlated with target should be flagged HIGH."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "leak_perfect": y.astype(float) + rng.normal(0, 0.01, n),
            "noise": rng.standard_normal(n),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    result = _run(state)
    flagged = set(result["adaptive_flagged_features"])
    assert "leak_perfect" in flagged
    assert "noise" not in flagged

    # Each verdict must have the documented schema
    for v in result["adaptive_verdicts"]:
        assert "feature" in v
        assert "layer" in v
        assert "z_score" in v or v["layer"] == "1"
        assert "remediation" in v
        assert "severity" in v


def test_node_does_not_flag_pure_noise():
    """Pure-noise features should NOT be flagged (no false positives)."""
    rng = np.random.default_rng(7)
    n = 500
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "noise_1": rng.standard_normal(n),
            "noise_2": rng.standard_normal(n),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    result = _run(state)
    flagged = set(result["adaptive_flagged_features"])
    assert flagged == set(), f"Pure noise falsely flagged: {flagged}"


def test_node_merges_with_existing_leaked_features():
    """If detect_leakage already found some leaks, the node augments rather than
    replaces."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "obvious_leak": y.astype(float) + rng.normal(0, 0.01, n),
            "other_leak_already_known": rng.standard_normal(n),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    state["leaked_features"] = ["other_leak_already_known"]
    result = _run(state)
    flagged = set(result["adaptive_flagged_features"])
    assert "obvious_leak" in flagged

    # The merged set passed forward in leakage state must contain BOTH
    merged = set(result.get("leaked_features") or [])
    assert "obvious_leak" in merged
    assert "other_leak_already_known" in merged


def test_node_skips_when_target_missing():
    """If no prediction_target is configured, node returns empty verdicts gracefully."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    state: dict = {
        "experiment_id": "test",
        "train_df": df,
        "scope_spec": {},
        "leakage_findings": [],
        "leaked_features": [],
    }
    result = _run(state)
    assert result["adaptive_verdicts"] == []
    assert result["adaptive_flagged_features"] == []


def test_node_escalates_severity_when_layer_3_catches_what_legacy_missed():
    """Layer 3 catching a leak the legacy detector marked as 'none' should
    escalate severity to 'high' so the routing layer triggers remediation."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "leak_perfect": y.astype(float) + rng.normal(0, 0.01, n),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    state["leakage_severity"] = "none"
    state["leakage_detected"] = False
    result = _run(state)
    assert result.get("leakage_severity") == "high"
    assert result.get("leakage_detected") is True


def test_node_does_not_downgrade_existing_severity():
    """Adaptive must never downgrade severity — only escalate."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "noise": rng.standard_normal(n),
            "y": rng.integers(0, 2, n),
        }
    )
    state = _make_state(df, "y")
    state["leakage_severity"] = "critical"
    result = _run(state)
    # critical > anything adaptive can return → severity should NOT change
    assert "leakage_severity" not in result or result["leakage_severity"] == "critical"


def test_node_skips_features_in_excluded_list():
    """excluded_features (PII, leakage already declared) should be skipped."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "obvious_leak": y.astype(float) + rng.normal(0, 0.01, n),
            "noise": rng.standard_normal(n),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    state["scope_spec"]["excluded_features"] = ["obvious_leak"]
    result = _run(state)
    # The leak is excluded → should NOT appear in adaptive verdicts
    feature_names_seen = {v["feature"] for v in result["adaptive_verdicts"]}
    assert "obvious_leak" not in feature_names_seen
