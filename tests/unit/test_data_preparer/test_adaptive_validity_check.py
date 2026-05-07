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


def test_sidecar_writer_skips_when_env_unset(tmp_path, monkeypatch):
    """No env var → no write (unit-test default)."""
    from src.agents.ml_foundation.data_preparer.graph import write_adaptive_verdicts_sidecar

    monkeypatch.delenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", raising=False)
    state = {"experiment_id": "x", "adaptive_verdicts": [{"feature": "a"}]}
    out = write_adaptive_verdicts_sidecar(state)
    assert out is None


def test_sidecar_writer_skips_when_no_verdicts(tmp_path, monkeypatch):
    """Empty verdicts list → no write even with env set."""
    from src.agents.ml_foundation.data_preparer.graph import write_adaptive_verdicts_sidecar

    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    state = {"experiment_id": "x", "adaptive_verdicts": []}
    out = write_adaptive_verdicts_sidecar(state)
    assert out is None
    assert list(tmp_path.iterdir()) == []


def test_sidecar_writer_persists_verdicts_to_json(tmp_path, monkeypatch):
    """With env set + verdicts present, JSON sidecar lands at the configured path."""
    import json as _json

    from src.agents.ml_foundation.data_preparer.graph import write_adaptive_verdicts_sidecar

    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    state = {
        "experiment_id": "exp-42",
        "data_source": "csu",
        "leakage_severity": "high",
        "leaked_features": ["leak_perfect"],
        "adaptive_flagged_features": ["leak_perfect"],
        "adaptive_verdicts": [
            {
                "feature": "leak_perfect",
                "layer": "3",
                "z_score": 9.4,
                "severity": "high",
                "remediation": "drop",
                "evidence": "z=9.4σ above null",
            }
        ],
    }
    sidecar_path = write_adaptive_verdicts_sidecar(state)
    assert sidecar_path is not None
    assert sidecar_path.exists()
    payload = _json.loads(sidecar_path.read_text())
    assert payload["experiment_id"] == "exp-42"
    assert payload["leakage_severity"] == "high"
    assert payload["adaptive_flagged_features"] == ["leak_perfect"]
    assert len(payload["adaptive_verdicts"]) == 1
    assert payload["adaptive_verdicts"][0]["feature"] == "leak_perfect"


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


def test_layer_1_catches_forbidden_csu_feature_via_manifest():
    """When a column from the CSU manifest is post-index (e.g.
    journey_duration_days), Layer 5 should emit a layer="1" verdict that
    drops it WITHOUT needing the adversarial discriminator to fire. The
    contract alone is sufficient evidence.
    """
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            # `journey_duration_days` is forbidden by the CSU manifest.
            "journey_duration_days": rng.normal(180, 60, n),
            "noise": rng.standard_normal(n),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    result = _run(state)

    flagged = set(result["adaptive_flagged_features"])
    assert "journey_duration_days" in flagged

    # The verdict should attribute the call to Layer 1, not Layer 3
    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}
    jd = verdicts_by_feat["journey_duration_days"]
    assert jd["layer"] == "1", f"Expected layer=1; got layer={jd['layer']}"
    assert jd["severity"] == "high"
    assert jd["remediation"] == "drop"
    assert "post_index" in jd["evidence"] or "contract" in jd["evidence"].lower()


def test_layer_1_does_not_flag_legitimate_csu_demographic():
    """`age_continuous` is on the CSU SAFE list (knowable_at=enrollment).
    The manifest must NOT cause it to be flagged."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "age_continuous": rng.normal(45, 15, n),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    result = _run(state)
    flagged = set(result["adaptive_flagged_features"])
    assert "age_continuous" not in flagged


def test_layer_3_runs_for_unknown_features():
    """A feature with no manifest contract should still be evaluated by
    Layer 3 (the existing behavior is preserved for unknown columns)."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            # Made-up name not in any manifest
            "synthetic_unique_zzz": y.astype(float) + rng.normal(0, 0.01, n),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    result = _run(state)
    flagged = set(result["adaptive_flagged_features"])
    assert "synthetic_unique_zzz" in flagged

    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}
    v = verdicts_by_feat["synthetic_unique_zzz"]
    # Unknown features go through Layer 3 (z-score path)
    assert v["layer"] == "3"


def test_layer_1_verdict_includes_contract_metadata():
    """When Layer 1 catches a forbidden feature, the verdict's evidence
    should reference the contract's knowable_at (so a reviewer can trace
    the decision back to a declared rule)."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "journey_status": rng.choice([0, 1], size=n).astype(float),
            "y": rng.integers(0, 2, n),
        }
    )
    state = _make_state(df, "y")
    result = _run(state)
    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}
    js = verdicts_by_feat["journey_status"]
    assert js["layer"] == "1"
    assert js["severity"] == "high"


# --- Codex-rescue audit follow-ups (PR #84) ---------------------------------


@pytest.mark.parametrize("ext_dtype", ["Int64", "Float64", "boolean"])
def test_node_handles_pandas_extension_dtypes(ext_dtype: str):
    """Pandas extension dtypes (Int64/Float64/boolean) must NOT crash the node.

    Regression for codex-rescue Critical #1: ``np.issubdtype`` raises
    ``TypeError: Cannot interpret 'Int64Dtype()' as a data type`` when given a
    pandas extension dtype. Anything ingested from Supabase/SQLAlchemy with a
    nullable-int schema would otherwise crash Layer 5. The fix uses
    ``pd.api.types.is_numeric_dtype`` which handles extension dtypes correctly.
    """
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    if ext_dtype == "Int64":
        col = pd.array(rng.integers(-100, 100, n).astype("int64"), dtype="Int64")
    elif ext_dtype == "Float64":
        col = pd.array(rng.standard_normal(n), dtype="Float64")
    else:  # boolean
        col = pd.array(rng.choice([True, False], size=n), dtype="boolean")

    df = pd.DataFrame(
        {
            "ext_feature": col,
            "noise_native": rng.standard_normal(n),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    # Must not raise — the dtype check itself was the bug.
    result = _run(state)
    feature_names_seen = {v["feature"] for v in result["adaptive_verdicts"]}
    assert "ext_feature" in feature_names_seen, (
        f"{ext_dtype} column was not evaluated by Layer 3 — likely "
        "filtered out by the dtype check."
    )


def test_explicit_seed_zero_is_honored(monkeypatch):
    """Passing ``adaptive_seed=0`` must NOT be replaced by the default seed=7.

    Regression for codex-rescue High #2: ``int(state.get(...) or 7)`` evaluates
    to 7 for state value 0 because 0 is falsy in Python. Same falsy-zero pattern
    applied to ``adaptive_n_permutations``. The fix uses explicit
    ``is not None`` checks.
    """
    captured: dict = {}

    def _fake_score(values, target, *, n_permutations, seed, z_threshold):
        captured["seed"] = seed
        captured["n_permutations"] = n_permutations
        return {
            "actual_auc": 0.5,
            "z_score": 0.0,
            "null_mean": 0.5,
            "null_std": 0.05,
            "p_value": 0.5,
            "n_permutations": n_permutations,
        }

    import importlib

    avc_mod = importlib.import_module(
        "src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check"
    )
    monkeypatch.setattr(avc_mod, "compute_adversarial_score", _fake_score)

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "feature_a": rng.standard_normal(n),
            "y": rng.integers(0, 2, n),
        }
    )
    state = _make_state(df, "y", adaptive_seed=0, adaptive_n_permutations=0)
    _ = _run(state)
    assert captured.get("seed") == 0, (
        f"Expected seed=0 to be honored; got {captured.get('seed')!r} "
        "(falsy-zero bug regression)"
    )
    assert captured.get("n_permutations") == 0, (
        f"Expected n_permutations=0 to be honored; got "
        f"{captured.get('n_permutations')!r}"
    )


def test_default_seed_when_state_omits_field(monkeypatch):
    """When state does NOT include ``adaptive_seed``, the default (7) is used."""
    captured: dict = {}

    def _fake_score(values, target, *, n_permutations, seed, z_threshold):
        captured["seed"] = seed
        captured["n_permutations"] = n_permutations
        return {
            "actual_auc": 0.5,
            "z_score": 0.0,
            "null_mean": 0.5,
            "null_std": 0.05,
            "p_value": 0.5,
            "n_permutations": n_permutations,
        }

    import importlib

    avc_mod = importlib.import_module(
        "src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check"
    )
    monkeypatch.setattr(avc_mod, "compute_adversarial_score", _fake_score)

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "feature_a": rng.standard_normal(n),
            "y": rng.integers(0, 2, n),
        }
    )
    # No adaptive_seed / adaptive_n_permutations in state → defaults apply.
    state = _make_state(df, "y")
    _ = _run(state)
    assert captured.get("seed") == 7
    assert captured.get("n_permutations") == 200
