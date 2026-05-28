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


def _make_state(
    train_df: pd.DataFrame,
    target: str,
    feature_manifest_source: str | None = "csu",
    **overrides: Any,
) -> dict:
    """Build a minimal DataPreparerState dict for the node.

    Defaults to ``feature_manifest_source="csu"`` because the bulk of the
    pre-existing tests in this file exercise Layer 1 against CSU-canonical
    column names (``journey_duration_days``, ``brand``, etc.). Tests that
    exercise the synthetic / no-manifest code path should pass
    ``feature_manifest_source=None`` explicitly.
    """
    state: dict = {
        "experiment_id": "test-exp",
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": target,
            "required_features": [c for c in train_df.columns if c != target],
            "excluded_features": [],
            "feature_manifest_source": feature_manifest_source,
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
        f"{ext_dtype} column was not evaluated by Layer 3 — likely filtered out by the dtype check."
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
    # FDR off: this guards the σ-band path's explicit-zero READ (the falsy-zero
    # regression). The Phase-1 FDR pre-pass deliberately RAISES n_permutations=0
    # to the BH feasibility floor — correct budget-sizing, but it would mask the
    # raw read this test pins. The falsy-zero `is not None` read is unchanged by
    # Phase 1; FDR's budget-raising is exercised in test_adaptive_validity_check_fdr.
    state = _make_state(
        df, "y", adaptive_seed=0, adaptive_n_permutations=0, adaptive_fdr_enabled=False
    )
    _ = _run(state)
    assert captured.get("seed") == 0, (
        f"Expected seed=0 to be honored; got {captured.get('seed')!r} (falsy-zero bug regression)"
    )
    assert captured.get("n_permutations") == 0, (
        f"Expected n_permutations=0 to be honored; got {captured.get('n_permutations')!r}"
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


def test_integer_target_with_sentinel_value_does_not_silence_layer_3():
    """A leaky feature must still be flagged HIGH when the target contains
    integer sentinel values like -1 (unknown outcome).

    Regression for codex-rescue High #3: ``pd.isna(-1) is False`` (integers
    cannot be NaN), so the sentinel rows pass the 2-class check, reach
    ``roc_auc_score`` as a 3-class input, raise ``ValueError``, get caught,
    and silently produce ``severity=info, remediation=keep`` for every
    numeric feature. The fix excludes non-binary target rows from the per-
    feature mask so Layer 3 sees only {0, 1}.
    """
    rng = np.random.default_rng(0)
    n = 600
    # 1/3 of rows have target=-1 (sentinel for "unknown outcome")
    base_y = rng.integers(0, 2, n)
    sentinel_mask = rng.random(n) < 0.33
    y_with_sentinels = np.where(sentinel_mask, -1, base_y).astype(int)
    df = pd.DataFrame(
        {
            "leak_perfect": base_y.astype(float) + rng.normal(0, 0.01, n),
            "noise": rng.standard_normal(n),
            "y": y_with_sentinels,
        }
    )
    state = _make_state(df, "y")
    result = _run(state)
    flagged = set(result["adaptive_flagged_features"])
    assert "leak_perfect" in flagged, (
        "Sentinel -1 values in the target masked the leak entirely. "
        f"Flagged set: {flagged}. Verdicts: {result['adaptive_verdicts']}"
    )


def test_verdict_schema_is_uniform_across_layer_1_and_layer_3():
    """Every verdict (Layer 1, Layer 3 normal, Layer 3 short-circuit) must
    share the same key set so JSON-sidecar consumers can rely on it.

    Regression for codex-rescue Medium #6: ``_build_verdict`` was missing
    ``contract_source``/``contract_window_days``, and the inline too-few-rows
    + exception branches were missing several Layer-3 score fields. Sidecar
    consumers had to special-case which fields might be present. The fix
    routes every Layer 3 short-circuit through ``_short_circuit_verdict`` and
    adds the contract-metadata keys to ``_build_verdict``.

    Phase 2.9 Stage 1 (2026-05-08) extended the schema with three new
    optional audit fields: ``decided_by``, ``disagreements``, ``kg_signal``.
    These are populated by every verdict path (legacy + ensemble) so
    sidecar consumers continue to see a uniform schema.
    """
    rng = np.random.default_rng(0)
    n = 250
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            # Layer 1: manifest catches this CSU post-index column
            "journey_status": rng.choice([0, 1], size=n).astype(float),
            # Layer 3 (normal scoring path)
            "synthetic_unique_zzz": y.astype(float) + rng.normal(0, 0.01, n),
            # Layer 3 (short-circuit: only 5 non-null rows of 250 → <30)
            "tiny_feature": [1.0] * 5 + [None] * (n - 5),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    result = _run(state)

    canonical_keys = {
        "feature",
        "layer",
        "z_score",
        "actual_auc",
        "null_mean",
        "null_std",
        "p_value",
        "n_permutations",
        "severity",
        "remediation",
        "evidence",
        "contract_source",
        "contract_window_days",
        # Phase 2.9 Stage 1 audit fields:
        "decided_by",
        "disagreements",
        "kg_signal",
        # Phase 2.9 Stage 3 audit fields (issue #193 codex pass-3 LOW):
        # surface LLM verdict role + remediation even when the
        # deterministic veto path wins on severity, so audit-cost is
        # observable per-feature.
        "llm_role",
        "llm_remediation",
        # Issue #194 joint-check audit fields (codex pass-1 LOW-1).
        "delta_auc",
        "delta_auc_floor",
        "delta_auc_below_floor",
        # Issue #196 Phase 3.3 — Layer-3 ablation audit fields.
        "ablation_z_score",
        "ablation_delta_auc",
        "ablation_null_mean",
        "ablation_null_std",
        "ablation_severity",
        # Issue #212 — z-only severity before joint-check clamp.
        # Audit field; lets downstream consumers distinguish "Layer 4
        # fired because pre-joint-check was moderate, joint-clamped to
        # info" from inconsistent layer routing.
        "severity_pre_joint_check",
        # Layer-4 evaluator audit-only fields (Plan
        # .claude/plans/layer4_evaluator_audit_signal.md). All five
        # are None when the evaluator is disabled / failed / had no
        # LLM verdict to read.
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
        # Issue #240 Stage 2/3 soft-gate keys — additive; merged into every
        # verdict by #240 but this assertion set was not updated then (the test
        # was red on main). Repaired here alongside the #501 addition below.
        "would_promote_severity",
        "would_flag_for_review",
        "gate_rule_fired",
        "worker_severity_pre_gate",
        "rationale_incomplete_flag",
        "evaluator_latency_ms",
        "evaluator_input_tokens",
        "evaluator_output_tokens",
        "evaluator_cost_usd",
        # Issue #501 — leakage × role shadow cross-check key:
        "would_flag_role_leak_disagreement",
        # Issue #501 — M-structure structural-remediation gate shadow keys:
        "structural_role",
        "structural_llm_disagreement",
        "structural_remediation_override",
        "structural_gate_fired",
        # Plan v4 Layer B / Phase 2 — structural decider unclassifiable flag:
        "structural_unclassifiable",
    }
    for v in result["adaptive_verdicts"]:
        assert set(v.keys()) == canonical_keys, (
            f"Verdict for {v.get('feature')!r} has non-uniform keys: "
            f"missing={canonical_keys - set(v.keys())}, "
            f"extra={set(v.keys()) - canonical_keys}"
        )


# --- Cross-cohort manifest false-positive regression (item M) ---------------


def test_synthetic_run_with_no_manifest_source_does_not_layer_1_flag_brand():
    """Scenario_a / synthetic regimes leave ``feature_manifest_source`` unset.
    A constant column named ``brand`` (e.g. ``df["brand"] = "Kisqali"`` from
    run_tier0_test.py) must NOT match the CSU manifest's post-index ``brand``
    contract. Otherwise scenario_a halts before training and the
    test_synthetic_e2e_scenario_a_pins_7dim_baseline regression test trips.

    Regression guard for the cross-cohort false-positive bug introduced by
    PR #84 commit 33fd376 (manifest wiring) and fixed in this commit by
    making Layer 1 opt-in via scope_spec.feature_manifest_source.
    """
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            # CSU-manifest-canonical column name, but as a constant (RCT brand
            # assignment) rather than the post-index event the CSU contract
            # describes. Without the opt-in guard, the manifest matches by
            # name alone and flags this as a Layer 1 leak.
            "brand": ["Kisqali"] * n,
            "noise": rng.standard_normal(n),
            "y": y,
        }
    )
    # Explicitly simulate a synthetic run by passing manifest_source=None.
    state = _make_state(df, "y", feature_manifest_source=None)
    result = _run(state)

    flagged = set(result["adaptive_flagged_features"])
    assert "brand" not in flagged, (
        f"Layer 1 manifest fired on a constant `brand` column in a synthetic "
        f"context (no manifest source set). flagged={flagged}. "
        f"This is the PR #84 cross-cohort regression."
    )
    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}
    if "brand" in verdicts_by_feat:
        # If brand got a verdict at all (Layer 3 short-circuit since constant),
        # it must be Layer 3, not Layer 1.
        assert verdicts_by_feat["brand"]["layer"] == "3", (
            f"`brand` got a Layer 1 verdict in a synthetic context: {verdicts_by_feat['brand']!r}"
        )


def test_csu_manifest_source_still_catches_brand():
    """The CSU opt-in (``feature_manifest_source="csu"``) must continue to
    catch the post-index ``brand`` column. Anti-regression for the fix in
    the test above — ensure we didn't accidentally disable the CSU path.
    """
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            # CSU contract: brand="competitor" if treatment_initiated else None
            "brand": rng.choice(["competitor", "novartis"], size=n),
            "noise": rng.standard_normal(n),
            "y": y,
        }
    )
    state = _make_state(df, "y", feature_manifest_source="csu")
    result = _run(state)

    flagged = set(result["adaptive_flagged_features"])
    assert "brand" in flagged, (
        f"CSU manifest opt-in failed to catch `brand` post-index column. flagged={flagged}"
    )
    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}
    assert verdicts_by_feat["brand"]["layer"] == "1"


def test_unknown_manifest_source_falls_through_to_layer_3():
    """An unrecognized ``feature_manifest_source`` (typo, future cohort, etc.)
    must NOT silently apply CSU or Optum contracts. Layer 1 should be a
    no-op; Layer 3 still runs on numeric features.
    """
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            # CSU-canonical name; would be Layer-1-flagged under "csu" but the
            # cohort here is a hypothetical "future_indication" not in
            # MANIFEST_SOURCES.
            "journey_duration_days": rng.normal(180, 60, n),
            "y": y,
        }
    )
    state = _make_state(df, "y", feature_manifest_source="future_indication")
    result = _run(state)

    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}
    if "journey_duration_days" in verdicts_by_feat:
        assert verdicts_by_feat["journey_duration_days"]["layer"] == "3", (
            f"Unknown manifest source incorrectly routed through CSU manifest. "
            f"Verdict: {verdicts_by_feat['journey_duration_days']!r}"
        )


# --- Issue #356: _resolve_manifest_features synthetic-key coverage ----------


def test_resolve_manifest_features_csu_returns_non_empty_list():
    """Regression: ``_resolve_manifest_features("csu")`` returns the
    full CSU FeatureContract registry as a list.

    Pre-existing behaviour; this test pins it so issue #356's "add
    synthetic" change cannot silently regress the csu/optum branches.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _resolve_manifest_features,
    )
    from src.data.feature_contract import FeatureContract
    from src.data.manifests import CSU_FEATURES

    result = _resolve_manifest_features("csu")
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == len(CSU_FEATURES)
    for contract in result:
        assert isinstance(contract, FeatureContract)


def test_resolve_manifest_features_optum_returns_non_empty_list():
    """Regression: ``_resolve_manifest_features("optum")`` returns the
    full Optum FeatureContract registry as a list.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _resolve_manifest_features,
    )
    from src.data.feature_contract import FeatureContract
    from src.data.manifests import OPTUM_FEATURES

    result = _resolve_manifest_features("optum")
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == len(OPTUM_FEATURES)
    for contract in result:
        assert isinstance(contract, FeatureContract)


def test_resolve_manifest_features_synthetic_returns_non_empty_list():
    """Issue #356: ``_resolve_manifest_features("synthetic")`` MUST return
    a list of FeatureContract entries — not ``None``.

    Before the fix this returned ``None`` because ``"synthetic"`` was
    missing from the resolver dict, even though the synthetic manifest is
    a registered first-class data source (see
    ``src.data.manifests.MANIFEST_SOURCES``). Callers (KG cache
    validation, role-attribution derivation) treated the ``None`` return
    as "bypass / unknown source" rather than the real manifest registry,
    silently dropping Layer 1 fingerprint validation and contract-keyed
    role attribution for synthetic runs (which is the third major regime
    in the codebase per ``data/synthetic/`` and the README).

    Acceptance per issue #356:
      * Returns a ``list`` (possibly empty, but not ``None``).
      * Every entry is a ``FeatureContract``.
      * The list matches ``SYNTHETIC_FEATURES`` registry membership
        (so the resolver stays in lockstep with the manifest module).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _resolve_manifest_features,
    )
    from src.data.feature_contract import FeatureContract
    from src.data.manifests import SYNTHETIC_FEATURES

    result = _resolve_manifest_features("synthetic")
    # Issue #356 acceptance criterion: returns a list, not None.
    assert result is not None, (
        "_resolve_manifest_features('synthetic') returned None — this is "
        "the issue #356 bug. The resolver dict is missing the 'synthetic' "
        "key. Synthetic is a registered MANIFEST_SOURCES regime; the "
        "resolver must include it so callers (KG cache validation, "
        "role-attribution derivation) get a real registry instead of "
        "the legacy-bypass fallback."
    )
    assert isinstance(result, list)
    # Lockstep with the manifest module: the resolver list must match the
    # SYNTHETIC_FEATURES dict's value set. If a future PR adds a contract
    # to ``synthetic_feature_manifest.py`` it MUST also be reachable via
    # this resolver — otherwise the two diverge and Layer 5 silently
    # under-validates the new feature.
    assert len(result) == len(SYNTHETIC_FEATURES)
    by_name = {c.name for c in result}
    assert by_name == set(SYNTHETIC_FEATURES.keys())
    for contract in result:
        assert isinstance(contract, FeatureContract)


def test_resolve_manifest_features_unknown_still_returns_none():
    """Regression: an unknown manifest source (typo, future cohort) still
    returns ``None`` so the caller falls through to the legacy "trusted
    upstream" branch with a warning. This pins the only-csu/optum/synthetic
    contract — adding ``"synthetic"`` to the resolver must NOT make it
    permissive for arbitrary strings.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _resolve_manifest_features,
    )

    assert _resolve_manifest_features("future_indication") is None
    assert _resolve_manifest_features("") is None
    # Trailing whitespace / case typos must NOT pattern-match either.
    assert _resolve_manifest_features("Synthetic") is None
    assert _resolve_manifest_features("synthetic ") is None


def test_resolve_manifest_features_lockstep_with_manifest_sources_registry():
    """The resolver dict in ``adaptive_validity_check`` must cover every
    key in ``src.data.manifests.MANIFEST_SOURCES``. Drift in either
    direction (a manifest registered but not resolvable, or a resolvable
    key with no manifest) breaks the Layer 1 / Layer 5 contract.

    Drift-detection guard: future PRs that add a new cohort manifest
    (e.g., Phase-5 cohort onboarding) must also update this resolver, or
    this test trips with a clear "lockstep broken" message.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _resolve_manifest_features,
    )
    from src.data.manifests import MANIFEST_SOURCES

    for source in MANIFEST_SOURCES.keys():
        result = _resolve_manifest_features(source)
        assert result is not None, (
            f"manifest source {source!r} is in MANIFEST_SOURCES but "
            f"_resolve_manifest_features returned None — the resolver "
            f"and the manifest registry have drifted. Add {source!r} to "
            f"the registries dict in _resolve_manifest_features."
        )
        assert isinstance(result, list)


# =============================================================================
# Phase 2.9 Stage 1 — EnsembleVoter wiring (2026-05-08)
# =============================================================================


def test_phase29_layer_1_verdict_carries_decided_by_layer_1():
    """Manifest-caught features expose ``decided_by="layer_1"`` so the
    audit trail names the deciding source."""
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
    assert "journey_status" in verdicts_by_feat
    v = verdicts_by_feat["journey_status"]
    assert v["layer"] == "1"
    assert v["decided_by"] == "layer_1"
    assert v["severity"] == "high"
    assert v["remediation"] == "drop"
    assert v["disagreements"] == []  # no other source spoke
    assert v["kg_signal"] == "no_signal"  # KG inputs are None in Stage 1


def test_phase29_layer_3_high_verdict_carries_decided_by_adversarial():
    """Adversarial high (z>5σ) verdicts expose ``decided_by="adversarial"``."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "leak_feature": y.astype(float) + rng.normal(0, 0.01, n),
            "y": y,
        }
    )
    state = _make_state(df, "y", feature_manifest_source=None)
    result = _run(state)

    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}
    assert "leak_feature" in verdicts_by_feat
    v = verdicts_by_feat["leak_feature"]
    assert v["layer"] == "3"
    assert v["decided_by"] == "adversarial"
    assert v["severity"] == "high"


def test_phase29_layer_3_info_verdict_bypasses_voter_keeps_legacy_severity():
    """Pure-noise features still get ``severity=info, remediation=keep``
    even though the voter would abstain on adv=info-alone signals.

    The bypass path in ``_compose_legacy_verdict`` preserves the legacy
    contract that downstream consumers rely on.
    """
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame(
        {
            "noise": rng.standard_normal(n),
            "y": rng.integers(0, 2, n),
        }
    )
    state = _make_state(df, "y", feature_manifest_source=None)
    result = _run(state)

    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}
    v = verdicts_by_feat["noise"]
    assert v["layer"] == "3"
    assert v["severity"] == "info"
    assert v["remediation"] == "keep"
    assert v["decided_by"] == "adversarial"  # not "abstain"
    assert v["kg_signal"] == "no_signal"


def test_phase29_short_circuit_verdict_carries_audit_fields():
    """Short-circuit verdicts (too-few-rows) still carry the new audit fields."""
    rng = np.random.default_rng(0)
    n = 250
    df = pd.DataFrame(
        {
            "tiny_feature": [1.0] * 5 + [None] * (n - 5),
            "y": rng.integers(0, 2, n),
        }
    )
    state = _make_state(df, "y", feature_manifest_source=None)
    result = _run(state)

    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}
    v = verdicts_by_feat["tiny_feature"]
    assert v["layer"] == "3"
    assert v["severity"] == "info"
    assert v["decided_by"] == "adversarial"
    assert v["disagreements"] == []
    assert v["kg_signal"] == "no_signal"
    assert v["z_score"] is None  # short-circuit: probe didn't run
    assert "Skipped" in v["evidence"]


def test_phase29_disagreements_field_defaults_to_empty_list():
    """Until Stage 2 (KG) and Stage 3 (LLM) are wired, no source can
    contradict another; ``disagreements`` is always empty."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "noise": rng.standard_normal(n),
            "y": rng.integers(0, 2, n),
        }
    )
    state = _make_state(df, "y", feature_manifest_source=None)
    result = _run(state)

    for v in result["adaptive_verdicts"]:
        assert v["disagreements"] == []


def test_phase29_ensemble_to_legacy_dict_helper_directly():
    """Unit-test the adapter helper without running the full node."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _ensemble_to_legacy_dict,
    )
    from src.data.kg.types import EnsembleVerdict

    verdict = EnsembleVerdict(
        feature_name="test_feat",
        severity="high",
        remediation="drop",
        decided_by="layer_1",
        final_role="descendant",
        confidence=1.0,
        evidence=("Manifest contract: post_index",),
        layer_1_input={
            "feature": "test_feat",
            "severity": "high",
            "contract_source": "csu",
            "contract_window_days": None,
        },
    )
    legacy = _ensemble_to_legacy_dict(verdict, adversarial_input=None)
    assert legacy["feature"] == "test_feat"
    assert legacy["layer"] == "1"
    assert legacy["severity"] == "high"
    assert legacy["remediation"] == "drop"
    assert legacy["decided_by"] == "layer_1"
    assert legacy["contract_source"] == "csu"
    assert legacy["contract_window_days"] is None
    assert legacy["evidence"] == "Manifest contract: post_index"
    assert legacy["disagreements"] == []
    assert legacy["kg_signal"] == "no_signal"


def test_phase29_decided_by_to_layer_mapping_covers_all_cases():
    """Pin the decided_by → layer mapping so future EnsembleVoter
    additions (kg, llm) get caught if not wired here."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _DECIDED_BY_TO_LAYER,
    )

    assert _DECIDED_BY_TO_LAYER == {
        "layer_1": "1",
        "adversarial": "3",
        # Issue #196 Phase 3.3 — ``adversarial_ablation`` tag set when
        # the Layer-3 ablation MAX-rule escalates severity beyond what
        # the permutation pass alone produced. Maps back to layer "3"
        # because ablation IS a Layer 3 sub-test.
        "adversarial_ablation": "3",
        "kg": "2",
        "llm": "4",
        # Plan v4 Layer B / Phase 2 — structural decider maps to Layer 4.
        "structural": "4",
        "abstain": "abstain",
    }


def test_phase29_mixed_severities_in_single_run_no_voter_cross_feature_state():
    """Codex review LOW (L9-1, 2026-05-08): the shared ``EnsembleVoter``
    instance is reused across all features in one node invocation.
    Verify it has no cross-feature state by running a mix of severities
    (Layer 1 high, Layer 3 high, moderate, info, short-circuit) and
    asserting each verdict matches its own input only.
    """
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            # Layer 1 high (CSU-forbidden post_index column)
            "journey_status": rng.choice([0, 1], size=n).astype(float),
            # Layer 3 high (perfect leak — z >> 5σ)
            "leak": y.astype(float) + rng.normal(0, 0.01, n),
            # Layer 3 moderate (weak signal — z ≈ 4σ)
            # Constructed to be statistically present but below the high threshold
            "moderate_signal": (y * 0.3 + rng.standard_normal(n)).astype(float),
            # Layer 3 info (pure noise)
            "noise": rng.standard_normal(n),
            # Layer 3 short-circuit (only 5 non-null rows)
            "tiny": [1.0] * 5 + [None] * (n - 5),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    result = _run(state)

    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}

    # Layer 1 high
    assert verdicts_by_feat["journey_status"]["decided_by"] == "layer_1"
    assert verdicts_by_feat["journey_status"]["severity"] == "high"

    # Layer 3 high
    assert verdicts_by_feat["leak"]["decided_by"] == "adversarial"
    assert verdicts_by_feat["leak"]["severity"] == "high"
    assert verdicts_by_feat["leak"]["remediation"] == "drop"

    # Layer 3 noise
    assert verdicts_by_feat["noise"]["decided_by"] == "adversarial"
    assert verdicts_by_feat["noise"]["severity"] == "info"
    assert verdicts_by_feat["noise"]["remediation"] == "keep"

    # Layer 3 short-circuit
    assert verdicts_by_feat["tiny"]["decided_by"] == "adversarial"
    assert verdicts_by_feat["tiny"]["severity"] == "info"
    assert verdicts_by_feat["tiny"]["z_score"] is None
    assert "Skipped" in verdicts_by_feat["tiny"]["evidence"]


def test_phase29_compose_legacy_verdict_all_none_signals_returns_abstain():
    """Codex review LOW (L9-4, 2026-05-08): when ``_compose_legacy_verdict``
    is called with no signals at all (caller misuse), it falls through
    to ``voter.vote()`` with all-None inputs → voter returns abstain
    → adapter produces the abstain dict shape.

    Pin this behaviour so caller-misuse paths produce well-formed
    audit records (not crashes or silent skips).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter

    voter = EnsembleVoter()
    verdict = _compose_legacy_verdict("orphan_feat", voter=voter)
    assert verdict["feature"] == "orphan_feat"
    assert verdict["decided_by"] == "abstain"
    assert verdict["layer"] == "abstain"
    assert verdict["severity"] == "abstain"
    assert verdict["remediation"] == "review"
    # Schema invariant: all 27 canonical fields present
    # (16 Phase 2.9 Stage 1 + 2 Phase 2.9 Stage 3 LLM audit fields from issue #193
    # + 3 issue #194 joint-check audit fields from codex pass-1 LOW-1
    # + 5 issue #196 Phase 3.3 ablation audit fields
    # + 1 issue #212 severity_pre_joint_check audit field).
    expected_keys = {
        "feature",
        "layer",
        "z_score",
        "actual_auc",
        "null_mean",
        "null_std",
        "p_value",
        "n_permutations",
        "severity",
        "remediation",
        "evidence",
        "contract_source",
        "contract_window_days",
        "decided_by",
        "disagreements",
        "kg_signal",
        # Phase 2.9 Stage 3 audit fields (issue #193 codex pass-3 LOW):
        "llm_role",
        "llm_remediation",
        # Issue #194 joint-check audit fields (codex pass-1 LOW-1):
        "delta_auc",
        "delta_auc_floor",
        "delta_auc_below_floor",
        # Issue #196 Phase 3.3 — Layer-3 ablation audit fields:
        "ablation_z_score",
        "ablation_delta_auc",
        "ablation_null_mean",
        "ablation_null_std",
        "ablation_severity",
        # Issue #212 — pre-joint-check severity audit field:
        "severity_pre_joint_check",
        # Layer-4 evaluator audit-only fields (Plan
        # .claude/plans/layer4_evaluator_audit_signal.md):
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
        # Issue #240 Stage 2/3 soft-gate keys — additive; merged into every
        # verdict by #240 but this assertion set was not updated then (the test
        # was red on main). Repaired here alongside the #501 addition below.
        "would_promote_severity",
        "would_flag_for_review",
        "gate_rule_fired",
        "worker_severity_pre_gate",
        "rationale_incomplete_flag",
        "evaluator_latency_ms",
        "evaluator_input_tokens",
        "evaluator_output_tokens",
        "evaluator_cost_usd",
        # Issue #501 — leakage × role shadow cross-check key:
        "would_flag_role_leak_disagreement",
        # Issue #501 — M-structure structural-remediation gate shadow keys:
        "structural_role",
        "structural_llm_disagreement",
        "structural_remediation_override",
        "structural_gate_fired",
        # Plan v4 Layer B / Phase 2 — structural decider unclassifiable flag:
        "structural_unclassifiable",
    }
    assert set(verdict.keys()) == expected_keys


def test_phase29_m2_layer_1_verdict_wrapper_routes_through_voter():
    """Codex review MEDIUM (M2, 2026-05-08): the legacy ``_layer_1_verdict``
    wrapper used to bypass the voter, missing M4's malformed-contract
    guard. The fix routes the wrapper through ``_compose_legacy_verdict``
    so all Layer 1 verdict construction sites apply the same guards.

    Pin: a properly-constructed FeatureContract still produces the
    expected severity=high/drop verdict.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _layer_1_verdict,
    )
    from src.data.feature_contract import FeatureContract, KnowableAt

    contract = FeatureContract(
        name="some_post_index_field",
        knowable_at=KnowableAt(reference="post_index"),
        source="csu",
        derivation_inputs=("foo",),
    )
    verdict = _layer_1_verdict("some_post_index_field", contract)
    assert verdict["layer"] == "1"
    assert verdict["severity"] == "high"
    assert verdict["remediation"] == "drop"
    assert verdict["decided_by"] == "layer_1"
    assert verdict["contract_source"] == "csu"


def test_phase29_h5_moderate_adversarial_alone_keeps_ambiguous_remediation():
    """Codex review HIGH (H5, 2026-05-08): adv-moderate-alone used to
    flow through the voter which rewrote remediation from the legacy
    ``ambiguous`` to ``review``. Downstream consumers branching on
    remediation strings would see inconsistent values for equivalent
    risk levels depending on whether Layer 1 happened to fire.

    The fix bypasses the voter for adv-alone of ANY severity so the
    legacy remediation (info→keep, moderate→ambiguous, high→drop) is
    preserved.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _build_verdict,
    )

    verdict = _build_verdict(
        "feat_x",
        {
            "z_score": 4.0,  # > 3σ but < 5σ → moderate
            "actual_auc": 0.75,
            "null_mean": 0.5,
            "null_std": 0.05,
            "p_value": 0.01,
            "n_permutations": 200,
        },
    )
    assert verdict["severity"] == "moderate"
    assert verdict["remediation"] == "ambiguous"  # NOT "review"
    assert verdict["decided_by"] == "adversarial"


def test_phase29_h5_high_adversarial_alone_keeps_legacy_drop():
    """Sanity: adv-high-alone still produces drop via the bypass path."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _build_verdict,
    )

    verdict = _build_verdict(
        "feat_x",
        {
            "z_score": 7.5,
            "actual_auc": 0.95,
            "null_mean": 0.5,
            "null_std": 0.05,
            "p_value": 0.0,
            "n_permutations": 200,
        },
    )
    assert verdict["severity"] == "high"
    assert verdict["remediation"] == "drop"
    assert verdict["decided_by"] == "adversarial"


def test_phase29_h3_adversarial_input_handles_explicit_none_z_score():
    """Codex review HIGH (H3, 2026-05-08): an explicit ``z_score=None``
    (e.g., from a custom scorer or malformed payload) used to crash
    ``_adversarial_input`` with a TypeError on the ``z > HIGH_Z``
    comparison. ``dict.get(default=NaN)`` only catches the *missing*
    case; a None VALUE bypasses the default. The fix treats any
    non-numeric z as degenerate.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _build_verdict,
    )

    # Should not raise — emits the legacy degenerate-info verdict
    verdict = _build_verdict(
        "feat_x",
        {
            "z_score": None,
            "actual_auc": 0.5,
            "null_mean": 0.5,
            "null_std": 0.0,
            "p_value": None,
            "n_permutations": 10,
        },
    )
    assert verdict["severity"] == "info"
    assert verdict["remediation"] == "keep"
    assert verdict["z_score"] is None
    assert verdict["decided_by"] == "adversarial"


def test_phase29_h3_adversarial_input_handles_string_z_score():
    """A string value in ``z_score`` is also malformed; should not crash."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _build_verdict,
    )

    verdict = _build_verdict(
        "feat_x",
        {"z_score": "not_a_number", "actual_auc": 0.5, "null_mean": 0.5, "null_std": 0.0},
    )
    assert verdict["severity"] == "info"
    assert verdict["remediation"] == "keep"


def test_phase29_h3_adversarial_input_handles_bool_z_score():
    """A bool value (Python's ``True`` would otherwise pass ``isinstance(int)``)
    is also malformed; reject."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _build_verdict,
    )

    verdict = _build_verdict(
        "feat_x",
        {"z_score": True, "actual_auc": 0.5, "null_mean": 0.5, "null_std": 0.0},
    )
    assert verdict["severity"] == "info"


def test_phase29_voter_decides_when_layer_1_and_adversarial_both_high():
    """When Layer 1 contract AND adversarial both fire high, the voter
    rule (Layer 1 > adversarial) wins. ``decided_by="layer_1"``."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    # journey_duration_days is CSU-forbidden + leaky-correlated
    df = pd.DataFrame(
        {
            "journey_duration_days": y.astype(float) * 100 + rng.normal(0, 1, n),
            "y": y,
        }
    )
    state = _make_state(df, "y")
    result = _run(state)

    verdicts_by_feat = {v["feature"]: v for v in result["adaptive_verdicts"]}
    v = verdicts_by_feat["journey_duration_days"]
    assert v["decided_by"] == "layer_1"  # Layer 1 wins per voter precedence
    assert v["layer"] == "1"


# ---------------------------------------------------------------------------
# Manifest enforcement at _select_features (defense-in-depth)
# ---------------------------------------------------------------------------


def test_select_features_excludes_csu_forbidden_when_manifest_csu():
    """`_select_features` consults the manifest's FORBIDDEN list and drops
    target-coupled / post-index columns proactively.

    Defense-in-depth: complements the Layer 1 contract audit downstream
    (which would catch these columns in the verdict pass and route them
    through leakage_remediation). With the proactive exclusion, forbidden
    columns never reach Layer 3 scoring at all — saving compute AND
    closing the gap if a Layer 1 manifest lookup ever silently failed.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _select_features,
    )
    from src.data.manifests import CSU_FORBIDDEN_AS_FEATURES

    df = pd.DataFrame(
        {
            "age_continuous": [30, 40, 50, 60],
            "journey_duration_days": [10, 20, 30, 40],  # CSU forbidden
            "brand": [1.0, 2.0, 3.0, 4.0],  # CSU forbidden (numeric encoded)
            "y": [0, 1, 0, 1],
        }
    )
    cols = _select_features(df, target="y", excluded=[], manifest_source="csu")
    assert "age_continuous" in cols
    assert "journey_duration_days" not in cols
    assert "brand" not in cols
    # Sanity: the excluded names must be in the manifest's forbidden set.
    assert "journey_duration_days" in CSU_FORBIDDEN_AS_FEATURES
    assert "brand" in CSU_FORBIDDEN_AS_FEATURES


def test_select_features_excludes_optum_forbidden_when_manifest_optum():
    """Same defense-in-depth check for the Optum manifest."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _select_features,
    )
    from src.data.manifests import OPTUM_FORBIDDEN_AS_FEATURES

    if not OPTUM_FORBIDDEN_AS_FEATURES:
        pytest.skip("OPTUM_FORBIDDEN_AS_FEATURES is empty")

    forbidden_sample = OPTUM_FORBIDDEN_AS_FEATURES[0]
    df = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            forbidden_sample: [1.0, 2.0, 3.0, 4.0],
            "y": [0, 1, 0, 1],
        }
    )
    cols = _select_features(df, target="y", excluded=[], manifest_source="optum")
    assert "age" in cols
    assert forbidden_sample not in cols


def test_select_features_no_manifest_source_falls_through():
    """Synthetic regimes (and any cohort that doesn't opt into a manifest)
    must NOT have CSU / Optum forbidden lists applied — that would
    cross-cohort-pollute (e.g., synthetic's `brand` column would be
    falsely excluded under the CSU manifest).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _select_features,
    )

    df = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "journey_duration_days": [10, 20, 30, 40],
            "y": [0, 1, 0, 1],
        }
    )
    cols = _select_features(df, target="y", excluded=[], manifest_source=None)
    assert "journey_duration_days" in cols  # not excluded without manifest


def test_select_features_unknown_manifest_source_falls_through(caplog):
    """An unknown ``manifest_source`` value (typo, future cohort not yet
    registered) must NOT raise and must NOT apply any forbidden list —
    fail open for unknown manifests, not closed.

    Codex M1 (PR #92): the function emits a logger.warning so an operator
    who typoed ``feature_manifest_source`` in scope_spec can spot that
    the proactive defense-in-depth pass was skipped. Confirm the warning
    fires and names the unknown source.
    """
    import logging

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _select_features,
    )

    df = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "journey_duration_days": [10, 20, 30, 40],
            "y": [0, 1, 0, 1],
        }
    )
    with caplog.at_level(logging.WARNING):
        cols = _select_features(
            df, target="y", excluded=[], manifest_source="cohort_does_not_exist"
        )
    assert "journey_duration_days" in cols
    # Operator-facing warning fired with the unknown source name.
    assert any(
        "cohort_does_not_exist" in rec.message and "unknown manifest_source" in rec.message
        for rec in caplog.records
    )


def test_select_features_empty_string_manifest_source_falls_through(caplog):
    """Codex N1 (PR #92): an empty-string ``manifest_source`` enters the
    ``is not None`` branch but produces no forbidden list — must fall
    through harmlessly. Useful documentation test for the contract.
    """
    import logging

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _select_features,
    )

    df = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "journey_duration_days": [10, 20, 30, 40],
            "y": [0, 1, 0, 1],
        }
    )
    with caplog.at_level(logging.WARNING):
        cols = _select_features(df, target="y", excluded=[], manifest_source="")
    assert "journey_duration_days" in cols
    # Warning fires for empty-string the same way it does for any
    # other unknown value.
    assert any("unknown manifest_source" in rec.message for rec in caplog.records)


def test_select_features_manifest_layered_with_scope_excluded():
    """``excluded`` (scope_spec) and the manifest forbidden list compose
    additively — both apply.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _select_features,
    )

    df = pd.DataFrame(
        {
            "age_continuous": [30, 40, 50, 60],
            "patient_id_hash": [1.1, 2.2, 3.3, 4.4],  # scope-excluded (PII)
            "journey_duration_days": [10, 20, 30, 40],  # manifest-excluded
            "y": [0, 1, 0, 1],
        }
    )
    cols = _select_features(df, target="y", excluded=["patient_id_hash"], manifest_source="csu")
    assert cols == ["age_continuous"]


def test_select_features_manifest_does_not_drop_target_or_non_numeric():
    """Manifest-excluded columns are dropped, but the existing target /
    non-numeric exclusions are preserved. Order of exclusion shouldn't
    matter for the final result.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _select_features,
    )

    df = pd.DataFrame(
        {
            "age_continuous": [30, 40, 50, 60],
            "journey_status": [
                "active",
                "active",
                "complete",
                "complete",
            ],  # non-numeric AND CSU-forbidden
            "journey_duration_days": [10, 20, 30, 40],  # CSU-forbidden
            "y": [0, 1, 0, 1],
        }
    )
    cols = _select_features(df, target="y", excluded=[], manifest_source="csu")
    # journey_status: non-numeric → out (would be out anyway)
    # journey_duration_days: manifest-forbidden → out
    # y: target → out
    # age_continuous: kept
    assert cols == ["age_continuous"]


# =============================================================================
# Stage 2 PR-D: KG edges plumbed through _compose_legacy_verdict
# =============================================================================


def test_phase29_stage2_compose_legacy_verdict_passes_kg_edges_to_voter():
    """Stage 2 wiring: kg_edges + entity_ids flow into voter.vote() and the
    voter returns a kg-decided verdict (decided_by='kg', layer='2',
    kg_signal='leak_drug_treats_disease'). With no Layer 1 contract and no
    adversarial verdict, the kg edge is the sole signal.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import KGEdge

    voter = EnsembleVoter()
    edge = KGEdge(
        subject_id="CHEMBL1234",
        predicate="treats",
        object_id="EFO_0000270",
        evidence_source="open_targets",
        score=0.85,
    )
    verdict = _compose_legacy_verdict(
        "x",
        voter=voter,
        layer_1_input=None,
        adversarial_input=None,
        kg_edges=(edge,),
        feature_entity_ids=("CHEMBL1234",),
        target_entity_ids=("EFO_0000270",),
    )

    assert verdict["decided_by"] == "kg"
    assert verdict["kg_signal"] == "leak_drug_treats_disease"
    assert verdict["layer"] == "2"


def test_phase29_stage2_compose_legacy_verdict_empty_kg_preserves_stage1_behavior():
    """Empty kg_edges preserves Stage 1 behavior — no regression.

    Layer 1 high contract still produces decided_by='layer_1' / layer='1'
    when kg_edges is empty (the default).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter

    voter = EnsembleVoter()
    layer_1 = {
        "feature": "x",
        "layer": "1",
        "severity": "high",
        "remediation": "drop",
        "evidence": "post_index",
        "contract_source": "csu",
        "contract_window_days": None,
    }
    verdict = _compose_legacy_verdict(
        "x",
        voter=voter,
        layer_1_input=layer_1,
        adversarial_input=None,
    )
    assert verdict["decided_by"] == "layer_1"
    assert verdict["layer"] == "1"


def test_phase29_stage2_load_kg_cache_returns_none_without_path(tmp_path):
    """No scope_spec['kg_cache_path'] → loader returns None (Stage 1 path)."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )

    assert _load_kg_cache({}) is None
    assert _load_kg_cache({"kg_cache_path": ""}) is None


def test_phase29_stage2_load_kg_cache_warns_and_returns_none_when_missing(tmp_path):
    """Configured path that doesn't exist → warn + None (shadow-mode-friendly)."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )

    missing = tmp_path / "nope.json"
    result = _load_kg_cache({"kg_cache_path": str(missing)})
    assert result is None


def test_phase29_stage2_load_kg_cache_loads_records_to_dict(tmp_path):
    """Cache file → dict feature_name -> list of KGEdge."""
    from datetime import datetime, timezone

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )
    from src.data.kg.cache import CacheRecord, save_cache
    from src.data.kg.types import KGEdge

    edge = KGEdge(
        subject_id="CHEMBL2107858",
        predicate="treats",
        object_id="MONDO_0011918",
        evidence_source="open_targets",
        score=0.9,
    )
    record = CacheRecord(
        feature_name="primary_diagnosis_code",
        manifest_fingerprint_sha8="a",
        target_codes_fingerprint_sha8="b",
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(("ICD10CM", "L50.9"),),
        target_entity_codes=(("RXNORM", "479158"),),
        sources_attempted=("umls_uts", "open_targets"),
        status="ok",
        edges=(edge,),
        errors=(),
    )
    cache_path = tmp_path / "cache.json"
    save_cache([record], cache_path)

    # kg_mode must be set to load the cache (PR-E gate: default 'off'
    # preserves Stage 1 behavior). Either 'shadow' or 'promoted' loads.
    loaded = _load_kg_cache({"kg_cache_path": str(cache_path), "kg_mode": "shadow"})
    assert loaded is not None
    assert "primary_diagnosis_code" in loaded
    assert len(loaded["primary_diagnosis_code"]) == 1
    assert loaded["primary_diagnosis_code"][0].subject_id == "CHEMBL2107858"


# ---------------------------------------------------------------------------
# Plan task #6 — KG cache fingerprint validation (D2)
# ---------------------------------------------------------------------------


def _build_cache_with_fingerprints(
    tmp_path,
    *,
    manifest_fp: str,
    target_fp: str,
    feature_name: str = "primary_diagnosis_code",
):
    """Helper: write a single-record cache file with the given fingerprints."""
    from datetime import datetime, timezone

    from src.data.kg.cache import CacheRecord, save_cache
    from src.data.kg.types import KGEdge

    edge = KGEdge(
        subject_id="C0042109",
        predicate="isa",
        object_id="C0011615",
        evidence_source="umls_relations",
    )
    record = CacheRecord(
        feature_name=feature_name,
        manifest_fingerprint_sha8=manifest_fp,
        target_codes_fingerprint_sha8=target_fp,
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(("ICD10CM", "L50.9"), ("UMLS", "C0042109")),
        target_entity_codes=(("RXNORM", "479158"),),
        sources_attempted=("umls_uts",),
        status="ok",
        edges=(edge,),
        errors=(),
    )
    cache_path = tmp_path / "cache.json"
    save_cache([record], cache_path)
    return cache_path


def _current_csu_fingerprints(target_codes):
    """Compute the fingerprints the cache reader will recompute for CSU."""
    from src.data.kg.cache import (
        compute_manifest_fingerprint,
        compute_target_codes_fingerprint,
    )
    from src.data.manifests import CSU_FEATURES

    # Hashes the FULL CSU_FEATURES list (not filtered to entity-bearing).
    # Matches the writer at ``scripts/build_kg_cache.py:
    # build_cache_for_manifest`` which computes
    # ``compute_manifest_fingerprint(features)`` over the unfiltered
    # iterable. Codex HIGH-1 review of D2 PR caught a reader/writer
    # asymmetry — the fix aligned both sides on the unfiltered hash.
    return (
        compute_manifest_fingerprint(CSU_FEATURES),
        compute_target_codes_fingerprint(target_codes),
    )


def test_phase29_stage2_load_kg_cache_validates_fingerprints_happy_path(tmp_path):
    """When ``feature_manifest_source`` is set AND the cache's fingerprints
    match the current manifest + target_entity_codes, ``_load_kg_cache``
    returns the dict (validation passes silently)."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )

    target_codes = [("RXNORM", "479158")]
    manifest_fp, target_fp = _current_csu_fingerprints(target_codes)
    cache_path = _build_cache_with_fingerprints(
        tmp_path, manifest_fp=manifest_fp, target_fp=target_fp
    )

    loaded = _load_kg_cache(
        {
            "kg_cache_path": str(cache_path),
            "kg_mode": "shadow",
            "feature_manifest_source": "csu",
            "target_entity_codes": target_codes,
        }
    )
    assert loaded is not None
    assert "primary_diagnosis_code" in loaded


def test_phase29_stage2_load_kg_cache_shadow_warns_and_returns_none_on_mismatch(tmp_path, caplog):
    """In ``kg_mode=shadow`` a stale fingerprint logs a warning naming
    the offending feature(s) and returns None — the run proceeds without
    KG verdicts. Audit-only path tolerates staleness."""
    import logging

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )

    target_codes = [("RXNORM", "479158")]
    cache_path = _build_cache_with_fingerprints(
        tmp_path,
        manifest_fp="deadbeef",  # intentionally wrong
        target_fp="cafebabe",  # intentionally wrong
        feature_name="primary_diagnosis_code",
    )

    with caplog.at_level(
        logging.WARNING,
        logger="src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check",
    ):
        result = _load_kg_cache(
            {
                "kg_cache_path": str(cache_path),
                "kg_mode": "shadow",
                "feature_manifest_source": "csu",
                "target_entity_codes": target_codes,
            }
        )
    assert result is None
    assert any(
        "fingerprint mismatch" in rec.message
        and "primary_diagnosis_code" in rec.message
        and "scripts/build_kg_cache.py" in rec.message
        for rec in caplog.records
    ), f"Expected fingerprint-mismatch warning; got {[r.message for r in caplog.records]}"


def test_phase29_stage2_load_kg_cache_promoted_raises_on_mismatch(tmp_path):
    """In ``kg_mode=promoted`` a stale fingerprint MUST raise
    ``KGCacheStaleError`` — promoted mode lets KG verdicts DROP features,
    and a silent mismatch could drop the wrong ones. The pipeline halts
    until the operator rebuilds the cache."""
    import pytest as _pytest

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )
    from src.data.kg.cache import KGCacheStaleError

    target_codes = [("RXNORM", "479158")]
    cache_path = _build_cache_with_fingerprints(
        tmp_path,
        manifest_fp="deadbeef",
        target_fp="cafebabe",
    )

    with _pytest.raises(KGCacheStaleError, match="fingerprint mismatch"):
        _load_kg_cache(
            {
                "kg_cache_path": str(cache_path),
                "kg_mode": "promoted",
                "feature_manifest_source": "csu",
                "target_entity_codes": target_codes,
            }
        )


def test_phase29_stage2_load_kg_cache_bypasses_validation_when_manifest_source_unset(
    tmp_path,
):
    """Legacy compatibility: when ``feature_manifest_source`` is not set
    in scope_spec (synthetic regimes, custom-runner paths), fingerprint
    validation is skipped entirely. Cache loads even with arbitrary
    fingerprint strings — the surrounding Layer 1 manifest contracts
    also no-op without a manifest source, so cache-vs-manifest
    consistency is moot."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )

    cache_path = _build_cache_with_fingerprints(
        tmp_path,
        manifest_fp="placeholder",
        target_fp="placeholder",
    )

    loaded = _load_kg_cache(
        {
            "kg_cache_path": str(cache_path),
            "kg_mode": "shadow",
            # NO feature_manifest_source key
        }
    )
    assert loaded is not None
    assert "primary_diagnosis_code" in loaded


def test_phase29_stage2_load_kg_cache_bypasses_validation_for_unknown_manifest_source(
    tmp_path,
):
    """Unknown manifest source → bypass validation (forward-compat for
    custom manifests not yet in MANIFEST_SOURCES). The cache reader has
    no way to resolve the manifest and treats the cache as
    'trusted upstream'."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )

    cache_path = _build_cache_with_fingerprints(
        tmp_path,
        manifest_fp="placeholder",
        target_fp="placeholder",
    )

    loaded = _load_kg_cache(
        {
            "kg_cache_path": str(cache_path),
            "kg_mode": "shadow",
            "feature_manifest_source": "future_v3_cohort",  # unknown
        }
    )
    assert loaded is not None
    assert "primary_diagnosis_code" in loaded


def test_phase29_stage2_load_kg_cache_partial_mismatch_in_promoted_raises_naming_features(
    tmp_path,
):
    """Even ONE mismatched record triggers fail-fast in promoted mode;
    the error message names the offending feature(s) so the operator
    can identify which manifest change caused the staleness."""
    from datetime import datetime, timezone

    import pytest as _pytest

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )
    from src.data.kg.cache import (
        CacheRecord,
        KGCacheStaleError,
        save_cache,
    )

    target_codes = [("RXNORM", "479158")]
    manifest_fp_good, target_fp_good = _current_csu_fingerprints(target_codes)

    rec_ok = CacheRecord(
        feature_name="rec_ok_feature",
        manifest_fingerprint_sha8=manifest_fp_good,
        target_codes_fingerprint_sha8=target_fp_good,
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(("ICD10CM", "L50.9"),),
        target_entity_codes=(("RXNORM", "479158"),),
        sources_attempted=("umls_uts",),
        status="queried_no_edges",
        edges=(),
        errors=(),
    )
    rec_stale = CacheRecord(
        feature_name="rec_stale_feature",
        manifest_fingerprint_sha8="deadbeef",
        target_codes_fingerprint_sha8=target_fp_good,
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(("ICD10CM", "L50.9"),),
        target_entity_codes=(("RXNORM", "479158"),),
        sources_attempted=("umls_uts",),
        status="queried_no_edges",
        edges=(),
        errors=(),
    )
    cache_path = tmp_path / "mixed_cache.json"
    save_cache([rec_ok, rec_stale], cache_path)

    with _pytest.raises(KGCacheStaleError) as excinfo:
        _load_kg_cache(
            {
                "kg_cache_path": str(cache_path),
                "kg_mode": "promoted",
                "feature_manifest_source": "csu",
                "target_entity_codes": target_codes,
            }
        )
    assert "rec_stale_feature" in str(excinfo.value)
    # The good record should NOT appear in the mismatch preview.
    assert "rec_ok_feature" not in str(excinfo.value)


def test_phase29_stage2_load_kg_cache_validates_target_codes_fingerprint(tmp_path):
    """A cache built against one set of target_entity_codes must trip
    fingerprint validation when reloaded under a different set —
    promotion across cohorts with divergent target sets is unsafe."""
    import pytest as _pytest

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )
    from src.data.kg.cache import KGCacheStaleError

    # Cache was written with target=(RXNORM, 479158)
    original_target_codes = [("RXNORM", "479158")]
    manifest_fp, target_fp_original = _current_csu_fingerprints(original_target_codes)
    cache_path = _build_cache_with_fingerprints(
        tmp_path, manifest_fp=manifest_fp, target_fp=target_fp_original
    )

    # Reader supplies a DIFFERENT target set → mismatch.
    different_target_codes = [("RXNORM", "1011295")]
    with _pytest.raises(KGCacheStaleError, match="target_fp"):
        _load_kg_cache(
            {
                "kg_cache_path": str(cache_path),
                "kg_mode": "promoted",
                "feature_manifest_source": "csu",
                "target_entity_codes": different_target_codes,
            }
        )


def test_phase29_stage2_load_kg_cache_writer_reader_symmetry_on_real_csu(tmp_path):
    """Codex HIGH-1 recommended fix: prove a writer-produced cache passes
    the reader's fingerprint validation end-to-end. This is the load-
    bearing invariant: if writer and reader compute the same fingerprint
    over the SAME input set, every legitimately-built cache loads
    cleanly. A regression in either side immediately trips this test."""
    from scripts.build_kg_cache import build_cache_for_manifest
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )
    from src.data.manifests import CSU_FEATURES

    target_codes = [("RXNORM", "479158"), ("RXNORM", "1011295")]
    out_dir = tmp_path / "kg_cache_writer"
    cache_path = build_cache_for_manifest(
        features=CSU_FEATURES,
        target_entity_codes=target_codes,
        out_dir=out_dir,
    )

    # Writer-produced cache MUST load via the reader without tripping
    # any fingerprint mismatch.
    loaded = _load_kg_cache(
        {
            "kg_cache_path": str(cache_path),
            "kg_mode": "promoted",  # strict mode — would raise on mismatch
            "feature_manifest_source": "csu",
            "target_entity_codes": target_codes,
        }
    )
    assert loaded is not None, (
        "Writer-produced cache failed to load under reader's fingerprint "
        "validation — writer/reader fingerprint asymmetry regression."
    )


def test_phase29_stage2_load_kg_cache_unknown_manifest_source_warns(tmp_path, caplog):
    """Codex MEDIUM-3 follow-up: a typo in ``feature_manifest_source``
    ("cs" vs "csu") would silently bypass validation without an
    operator-visible signal. The reader logs a WARNING naming the
    unknown source and the registered alternatives so a typo at
    orchestration time produces one log line to grep for."""
    import logging

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )

    cache_path = _build_cache_with_fingerprints(
        tmp_path,
        manifest_fp="placeholder",
        target_fp="placeholder",
    )

    with caplog.at_level(
        logging.WARNING,
        logger="src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check",
    ):
        _load_kg_cache(
            {
                "kg_cache_path": str(cache_path),
                "kg_mode": "shadow",
                "feature_manifest_source": "cs",  # typo
                "target_entity_codes": [("RXNORM", "479158")],
            }
        )
    assert any(
        "feature_manifest_source='cs' is not in the registered manifests" in rec.message
        for rec in caplog.records
    ), f"Expected unknown-manifest-source warning; got {[r.message for r in caplog.records]}"


def test_phase29_stage2_load_kg_cache_empty_records_skips_validation(tmp_path):
    """An empty cache file (zero records) has nothing to validate.
    Callers who configured ``kg_cache_path`` to an empty cache are
    treated as a no-op (legacy preservation; previously this loaded as
    an empty dict, and the validation gate must NOT raise on it)."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )
    from src.data.kg.cache import save_cache

    cache_path = tmp_path / "empty_cache.json"
    save_cache([], cache_path)

    loaded = _load_kg_cache(
        {
            "kg_cache_path": str(cache_path),
            "kg_mode": "promoted",  # even strict mode shouldn't trip
            "feature_manifest_source": "csu",
            "target_entity_codes": [("RXNORM", "479158")],
        }
    )
    assert loaded == {}


def test_phase29_stage2_adv_moderate_alone_with_unrelated_kg_edges_preserves_legacy_contract():
    """Bug guard: kg_edges that produce no_signal must NOT change adv-moderate-alone
    remediation. Voter rule #6 emits remediation='review'; Stage 1 contract is
    'ambiguous'. The bypass is preserved when classify_kg_signal returns no_signal.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import KGEdge

    voter = EnsembleVoter()
    # KG edge connecting OTHER drug to OTHER disease — no relation to
    # this feature's entities or the target's.
    unrelated_edge = KGEdge(
        subject_id="CHEMBL999",
        predicate="treats",
        object_id="EFO_999999",
        evidence_source="open_targets",
        score=0.85,
    )
    adv = {
        "feature": "x",
        "layer": "3",
        "severity": "moderate",
        "remediation": "ambiguous",  # legacy contract
        "evidence": "z=4.0",
        "z_score": 4.0,
        "actual_auc": 0.65,
        "null_mean": 0.50,
        "null_std": 0.04,
        "p_value": 0.001,
        "n_permutations": 200,
        # codex MED-5: hand-rolled adv dicts in tests need to opt into the
        # HBLP-classified tag (production code always builds via
        # _adversarial_input which sets this).
        "_hblp_classified": True,
    }
    verdict = _compose_legacy_verdict(
        "x",
        voter=voter,
        layer_1_input=None,
        adversarial_input=adv,
        kg_edges=(unrelated_edge,),
        feature_entity_ids=("CHEMBL_FEATURE",),
        target_entity_ids=("EFO_TARGET",),
    )
    # Bypass triggered → legacy contract preserved (severity=moderate, remediation=ambiguous)
    assert verdict["severity"] == "moderate"
    assert verdict["remediation"] == "ambiguous"


def test_phase29_stage2_parse_target_entity_codes_skips_malformed():
    """Codex H5: malformed scope_spec target_entity_codes warn-and-skip,
    not crash.

    Bare ``code for _, code in target_codes`` would raise ValueError on
    1- or 3-element entries, killing the pipeline. Helper guards with a
    log warning and accepts only well-formed 2-tuples.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _parse_target_entity_codes,
    )

    # Empty / None
    assert _parse_target_entity_codes(None) == ()
    assert _parse_target_entity_codes([]) == ()

    # Well-formed mixed with malformed → only well-formed survive
    raw = [
        ("RXNORM", "479158"),
        ["RXNORM"],  # 1-element — would crash
        ["RXNORM", "1011295"],  # 2-element list (JSON-deserialized)
        ["RXNORM", "extra", "third"],  # 3-element — would crash
    ]
    out = _parse_target_entity_codes(raw)
    assert out == ("479158", "1011295")


@pytest.mark.asyncio
async def test_phase29_stage2_e2e_main_loop_with_populated_cache(tmp_path, monkeypatch):
    """Codex M6a: end-to-end test of the async main loop with a populated
    KG cache pointing at a real file.

    Constructs a minimal state: train_df with one numeric feature and a
    binary target; scope_spec with kg_cache_path. Asserts a verdict is
    emitted and (where edges connect) routes through the voter.
    """
    from datetime import datetime, timezone

    import pandas as pd

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        adaptive_validity_check,
    )
    from src.data.kg.cache import CacheRecord, save_cache
    from src.data.kg.types import KGEdge

    # Build a cache with one record for the feature we'll check
    edge = KGEdge(
        subject_id="CHEMBL2107858",
        predicate="treats",
        object_id="MONDO_0011918",
        evidence_source="open_targets",
        score=0.9,
    )
    record = CacheRecord(
        feature_name="age",
        manifest_fingerprint_sha8="aaaaaaaa",
        target_codes_fingerprint_sha8="bbbbbbbb",
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(("UMLS", "C0001779"),),
        target_entity_codes=(("UMLS", "C0042109"),),
        sources_attempted=("umls_uts", "open_targets"),
        status="ok",
        edges=(edge,),
        errors=(),
    )
    cache_path = tmp_path / "cache.json"
    save_cache([record], cache_path)

    # Train_df must have ≥30 non-null rows for Layer 3 to run
    df = pd.DataFrame(
        {
            "age": list(range(60)),
            "y": [0, 1] * 30,
        }
    )
    state = {
        "train_df": df,
        "scope_spec": {
            "prediction_target": "y",
            "kg_cache_path": str(cache_path),
            # No target_entity_codes — KG signal won't fire (no_signal),
            # but the loader and the main-loop wiring still execute.
        },
    }
    result = await adaptive_validity_check(state)

    # Main loop ran end-to-end — at least one verdict emitted for the
    # numeric "age" column.
    assert "adaptive_verdicts" in result
    assert len(result["adaptive_verdicts"]) >= 1
    # Verdict carries the canonical 27-field shape (regression guard).
    # Issue #193 added llm_role + llm_remediation. Issue #194 added
    # 3 joint-check audit fields. Issue #196 added 5 ablation audit fields.
    # Issue #212 added severity_pre_joint_check.
    verdict = next(v for v in result["adaptive_verdicts"] if v["feature"] == "age")
    expected_keys = {
        "feature",
        "layer",
        "z_score",
        "actual_auc",
        "null_mean",
        "null_std",
        "p_value",
        "n_permutations",
        "severity",
        "remediation",
        "evidence",
        "contract_source",
        "contract_window_days",
        "decided_by",
        "disagreements",
        "kg_signal",
        # Phase 2.9 Stage 3 audit fields (issue #193 codex pass-3 LOW):
        "llm_role",
        "llm_remediation",
        # Issue #194 joint-check audit fields (codex pass-1 LOW-1):
        "delta_auc",
        "delta_auc_floor",
        "delta_auc_below_floor",
        # Issue #196 Phase 3.3 — Layer-3 ablation audit fields.
        "ablation_z_score",
        "ablation_delta_auc",
        "ablation_null_mean",
        "ablation_null_std",
        "ablation_severity",
        # Issue #212 — pre-joint-check severity audit field:
        "severity_pre_joint_check",
        # Layer-4 evaluator audit-only fields (Plan
        # .claude/plans/layer4_evaluator_audit_signal.md):
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
        # Issue #240 Stage 2/3 soft-gate keys — additive; merged into every
        # verdict by #240 but this assertion set was not updated then (the test
        # was red on main). Repaired here alongside the #501 addition below.
        "would_promote_severity",
        "would_flag_for_review",
        "gate_rule_fired",
        "worker_severity_pre_gate",
        "rationale_incomplete_flag",
        "evaluator_latency_ms",
        "evaluator_input_tokens",
        "evaluator_output_tokens",
        "evaluator_cost_usd",
        # Issue #501 — leakage × role shadow cross-check key:
        "would_flag_role_leak_disagreement",
        # Issue #501 — M-structure structural-remediation gate shadow keys:
        "structural_role",
        "structural_llm_disagreement",
        "structural_remediation_override",
        "structural_gate_fired",
        # Plan v4 Layer B / Phase 2 — structural decider unclassifiable flag:
        "structural_unclassifiable",
    }
    assert set(verdict.keys()) == expected_keys


# =============================================================================
# Stage 2 PR-E: shadow-mode promotion gate
# =============================================================================


def test_phase29_stage2_pre_kg_mode_off_skips_cache_load(tmp_path):
    """kg_mode='off' preserves Stage 1: cache not loaded even when
    kg_cache_path is configured AND the file exists.
    """
    from datetime import datetime, timezone

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )
    from src.data.kg.cache import CacheRecord, save_cache

    # File EXISTS — make sure the kg_mode check fires before file IO.
    record = CacheRecord(
        feature_name="x",
        manifest_fingerprint_sha8="a",
        target_codes_fingerprint_sha8="b",
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(),
        target_entity_codes=(),
        sources_attempted=(),
        status="ok",
        edges=(),
        errors=(),
    )
    cache_path = tmp_path / "cache.json"
    save_cache([record], cache_path)

    result = _load_kg_cache({"kg_cache_path": str(cache_path), "kg_mode": "off"})
    assert result is None


def test_phase29_stage2_pre_kg_mode_shadow_loads_cache(tmp_path):
    """kg_mode='shadow' loads the cache like 'promoted'; severity cap
    happens at compose time, not load time.
    """
    from datetime import datetime, timezone

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
    )
    from src.data.kg.cache import CacheRecord, save_cache

    record = CacheRecord(
        feature_name="x",
        manifest_fingerprint_sha8="a",
        target_codes_fingerprint_sha8="b",
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(),
        target_entity_codes=(),
        sources_attempted=(),
        status="ok",
        edges=(),
        errors=(),
    )
    cache_path = tmp_path / "cache.json"
    save_cache([record], cache_path)

    loaded = _load_kg_cache({"kg_cache_path": str(cache_path), "kg_mode": "shadow"})
    assert loaded is not None
    assert "x" in loaded


def test_phase29_stage2_pre_shadow_mode_caps_kg_severity_to_info():
    """Shadow mode: a KG-decided 'high' severity verdict is capped to 'info'
    so leakage_remediation cannot drop the feature on KG signal alone.
    Evidence string carries a shadow-mode annotation so audit readers
    aren't confused by voter.evidence saying "drop" while severity says
    "info" / remediation says "keep" (codex M1).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import KGEdge

    voter = EnsembleVoter()
    edge = KGEdge(
        subject_id="CHEMBL1234",
        predicate="treats",
        object_id="EFO_0000270",
        evidence_source="open_targets",
        score=0.85,
    )
    verdict = _compose_legacy_verdict(
        "x",
        voter=voter,
        kg_edges=(edge,),
        feature_entity_ids=("CHEMBL1234",),
        target_entity_ids=("EFO_0000270",),
        kg_mode="shadow",
    )
    # decided_by + kg_signal still recorded (audit)
    assert verdict["decided_by"] == "kg"
    assert verdict["kg_signal"] == "leak_drug_treats_disease"
    # severity capped to info (cannot drop the feature)
    assert verdict["severity"] == "info"
    # remediation softened to keep — nothing should be dropped on KG alone in shadow
    assert verdict["remediation"] == "keep"
    # evidence carries the shadow-mode annotation
    assert "[shadow-mode" in verdict["evidence"]


def test_phase29_stage2_pre_promoted_mode_kg_drives_high_severity():
    """Promoted mode: KG-decided verdict keeps voter's full severity output
    (severity='high', remediation='drop' for leak_drug_treats_disease).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import KGEdge

    voter = EnsembleVoter()
    edge = KGEdge(
        subject_id="CHEMBL1234",
        predicate="treats",
        object_id="EFO_0000270",
        evidence_source="open_targets",
        score=0.85,
    )
    verdict = _compose_legacy_verdict(
        "x",
        voter=voter,
        kg_edges=(edge,),
        feature_entity_ids=("CHEMBL1234",),
        target_entity_ids=("EFO_0000270",),
        kg_mode="promoted",
    )
    assert verdict["decided_by"] == "kg"
    assert verdict["kg_signal"] == "leak_drug_treats_disease"
    assert verdict["severity"] == "high"
    assert verdict["remediation"] == "drop"


def test_phase29_stage2_pre_compute_promotion_eligibility_passes_threshold():
    """Promotion eligibility metrics: 95% non-abstain AND ≤5% disagreement
    → passes=True (assuming n_patients ≥ 200 too).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        compute_promotion_eligibility,
    )

    # 100 verdicts, 97 non-abstain, 3 with adv vs kg disagreement
    verdicts = (
        [{"decided_by": "kg", "disagreements": []}] * 80
        + [{"decided_by": "adversarial", "disagreements": []}] * 17
        + [{"decided_by": "abstain", "disagreements": []}] * 3
    )
    metrics = compute_promotion_eligibility(verdicts, n_patients=500)
    assert metrics["n_features"] == 100
    assert metrics["n_patients"] == 500
    assert metrics["non_abstain_pct"] == pytest.approx(0.97)
    assert metrics["patient_count_pass"] is True
    assert metrics["passes"] is True


def test_phase29_stage2_pre_compute_promotion_eligibility_fails_low_coverage():
    """When non-abstain < 95%, passes=False."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        compute_promotion_eligibility,
    )

    verdicts = [{"decided_by": "kg", "disagreements": []}] * 50 + [
        {"decided_by": "abstain", "disagreements": []}
    ] * 50
    metrics = compute_promotion_eligibility(verdicts, n_patients=500)
    assert metrics["non_abstain_pct"] == pytest.approx(0.50)
    assert metrics["passes"] is False


def test_phase29_stage2_pre_compute_promotion_eligibility_fails_high_disagreement():
    """When cross-source disagreement > 5%, passes=False."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        compute_promotion_eligibility,
    )

    verdicts = [{"decided_by": "kg", "disagreements": ["adversarial=high but kg=accept"]}] * 10 + [
        {"decided_by": "kg", "disagreements": []}
    ] * 90
    metrics = compute_promotion_eligibility(verdicts, n_patients=500)
    assert metrics["cross_source_disagreement_rate"] == pytest.approx(0.10)
    assert metrics["passes"] is False


def test_phase29_stage2_pre_compute_promotion_eligibility_zero_features():
    """Empty verdict list: well-formed metrics, passes=False (no evidence).

    n_patients still required to surface the patient-count guard alongside
    the no-evidence guard.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        compute_promotion_eligibility,
    )

    metrics = compute_promotion_eligibility([], n_patients=500)
    assert metrics["n_features"] == 0
    assert metrics["n_patients"] == 500
    assert metrics["non_abstain_pct"] == 0.0
    assert metrics["kg_decided_count"] == 0
    assert metrics["cross_source_disagreement_rate"] == 0.0
    # Patient-count guard still measured + reported even when no verdicts:
    assert metrics["patient_count_pass"] is True
    # Overall passes still False because no verdicts:
    assert metrics["passes"] is False


def test_phase29_stage2_pre_compute_promotion_eligibility_fails_no_kg_decided():
    """Codex H2: passes=False when KG never fired (all decided by adversarial),
    even with high non_abstain_pct and zero disagreement.

    Promoting in this state is meaningless — there is no KG signal to
    promote.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        compute_promotion_eligibility,
    )

    verdicts = [{"decided_by": "adversarial", "disagreements": []}] * 100
    metrics = compute_promotion_eligibility(verdicts, n_patients=500)
    assert metrics["n_features"] == 100
    assert metrics["non_abstain_pct"] == pytest.approx(1.0)
    assert metrics["kg_decided_count"] == 0
    assert metrics["passes"] is False


# ---------------------------------------------------------------------------
# Backlog #14: patient-count minimum (N≥200) enforcement
# ---------------------------------------------------------------------------


def _well_formed_verdicts(count: int = 100) -> list[dict[str, Any]]:
    """``count`` verdicts where 97% are non-abstain, ≥1 KG-decided, 0 disagreements.

    Mirrors the passing-threshold case so each backlog #14 test isolates
    on the patient-count guard alone — no other gate fires.

    Codex pass-1 LOW Q5: assert ``count >= 97`` so a future test author who
    reuses this helper with a small count gets a fail-loud error instead of
    silent abstain-padding underflow.
    """
    if count < 97:
        raise ValueError(
            f"_well_formed_verdicts requires count>=97 to keep the abstain padding "
            f"non-negative and 97% non-abstain shape; got {count}."
        )
    return (
        [{"decided_by": "kg", "disagreements": []}] * 80
        + [{"decided_by": "adversarial", "disagreements": []}] * 17
        + [{"decided_by": "abstain", "disagreements": []}] * (count - 97)
    )


def test_backlog_14_patient_count_at_threshold_passes():
    """Boundary case: ``n_patients == min_n_patients`` (default 200) passes."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        compute_promotion_eligibility,
    )

    metrics = compute_promotion_eligibility(_well_formed_verdicts(), n_patients=200)
    assert metrics["n_patients"] == 200
    assert metrics["patient_count_pass"] is True
    assert metrics["passes"] is True


def test_backlog_14_patient_count_below_threshold_fails():
    """``n_patients < min_n_patients`` → ``passes=False`` even when verdict gates pass.

    This is the load-bearing assertion for backlog #14: an under-powered
    cohort cannot promote on a verdict-only signal.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        compute_promotion_eligibility,
    )

    metrics = compute_promotion_eligibility(_well_formed_verdicts(), n_patients=199)
    # Verdict gates all pass:
    assert metrics["non_abstain_pct"] == pytest.approx(0.97)
    assert metrics["kg_decided_count"] == 80
    assert metrics["cross_source_disagreement_rate"] == pytest.approx(0.0)
    # Patient guard fails → overall passes False:
    assert metrics["patient_count_pass"] is False
    assert metrics["passes"] is False


def test_backlog_14_custom_min_n_patients_threshold_honored():
    """``min_n_patients`` override raises the bar (e.g., 1000 for stricter cohorts)."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        compute_promotion_eligibility,
    )

    # 500 patients passes default 200 but fails custom 1000:
    metrics_default = compute_promotion_eligibility(_well_formed_verdicts(), n_patients=500)
    assert metrics_default["passes"] is True

    metrics_strict = compute_promotion_eligibility(
        _well_formed_verdicts(), n_patients=500, min_n_patients=1000
    )
    assert metrics_strict["patient_count_pass"] is False
    assert metrics_strict["passes"] is False


def test_backlog_14_negative_n_patients_raises():
    """Negative cohort sizes are nonsensical → fail-loud at the boundary."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        compute_promotion_eligibility,
    )

    with pytest.raises(ValueError, match="n_patients must be non-negative"):
        compute_promotion_eligibility(_well_formed_verdicts(), n_patients=-1)


def test_backlog_14_n_patients_required_param():
    """Calling without ``n_patients`` raises TypeError (param is keyword-only required)."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        compute_promotion_eligibility,
    )

    with pytest.raises(TypeError, match="n_patients"):
        compute_promotion_eligibility(_well_formed_verdicts())  # type: ignore[call-arg]


def test_phase29_stage2_pre_shadow_mode_does_not_cap_adversarial_high():
    """Codex M2: shadow mode must NOT suppress adversarial-driven high severity.

    When adversarial z>5σ wins precedence (decided_by='adversarial'),
    shadow's decided_by=='kg' check doesn't fire → severity stays 'high'.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import KGEdge

    voter = EnsembleVoter()
    edge = KGEdge(
        subject_id="CHEMBL1234",
        predicate="treats",
        object_id="EFO_0000270",
        evidence_source="open_targets",
        score=0.85,
    )
    adv_input = {
        "feature": "drug_exposure_days",
        "layer": "3",
        "severity": "high",
        "z_score": 6.0,
        "actual_auc": 0.82,
        "null_mean": 0.51,
        "null_std": 0.03,
        "p_value": 0.0,
        "n_permutations": 200,
        "remediation": "drop",
        "evidence": "z=6.0 > 5σ",
        # codex MED-5: hand-rolled adv dicts in tests need to opt into the
        # HBLP-classified tag (production code always builds via
        # _adversarial_input which sets this).
        "_hblp_classified": True,
    }
    verdict = _compose_legacy_verdict(
        "drug_exposure_days",
        voter=voter,
        adversarial_input=adv_input,
        kg_edges=(edge,),
        feature_entity_ids=("CHEMBL1234",),
        target_entity_ids=("EFO_0000270",),
        kg_mode="shadow",
    )
    # Adversarial wins precedence; shadow cap MUST NOT fire
    assert verdict["decided_by"] == "adversarial"
    assert verdict["severity"] == "high"
    assert verdict["remediation"] == "drop"


def test_phase29_stage2_pre_shadow_mode_does_not_cap_layer_1_high():
    """Codex M3: shadow mode must NOT suppress Layer 1-driven high severity.

    When Layer 1 contract fires (decided_by='layer_1'), shadow's
    decided_by=='kg' check doesn't fire → severity stays 'high'.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import KGEdge

    voter = EnsembleVoter()
    edge = KGEdge(
        subject_id="CHEMBL1234",
        predicate="treats",
        object_id="EFO_0000270",
        evidence_source="open_targets",
        score=0.85,
    )
    layer_1_input = {
        "feature": "drug_exposure_days",
        "layer": "1",
        "severity": "high",
        "remediation": "drop",
        "evidence": "post_index feature",
        "contract_source": "Optum",
        "contract_window_days": 30,
    }
    verdict = _compose_legacy_verdict(
        "drug_exposure_days",
        voter=voter,
        layer_1_input=layer_1_input,
        kg_edges=(edge,),
        feature_entity_ids=("CHEMBL1234",),
        target_entity_ids=("EFO_0000270",),
        kg_mode="shadow",
    )
    # Layer 1 wins precedence; shadow cap MUST NOT fire
    assert verdict["decided_by"] == "layer_1"
    assert verdict["severity"] == "high"
    assert verdict["remediation"] == "drop"


def test_phase29_stage2_pre_kg_mode_unknown_value_defaults_to_off_with_warning(caplog):
    """Codex L1/L2: a typo like 'shadowmode' or 'Shadow' falls back to 'off'
    with a warning log so misconfiguration surfaces clearly.
    """
    import logging

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _resolve_kg_mode,
    )

    with caplog.at_level(logging.WARNING):
        assert _resolve_kg_mode("shadowmode") == "off"
        assert _resolve_kg_mode("Shadow") == "off"
    # Both invalid values triggered warnings
    assert sum("kg_mode=" in record.message for record in caplog.records) >= 2


def test_phase29_stage2_pre_kg_mode_none_defaults_to_off_silently(caplog):
    """None → 'off' without a warning (default-unset, not misconfiguration)."""
    import logging

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _resolve_kg_mode,
    )

    with caplog.at_level(logging.WARNING):
        assert _resolve_kg_mode(None) == "off"
    assert all("kg_mode=" not in record.message for record in caplog.records)


def test_phase29_stage2_pre_kg_mode_empty_string_defaults_to_off_silently(caplog):
    """Empty string (likely YAML default) → 'off' without a warning."""
    import logging

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _resolve_kg_mode,
    )

    with caplog.at_level(logging.WARNING):
        assert _resolve_kg_mode("") == "off"
    assert all("kg_mode=" not in record.message for record in caplog.records)


def test_phase29_stage2_pre_kg_mode_valid_modes_pass_through():
    """Valid modes: off, shadow, promoted — all pass through unchanged."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _resolve_kg_mode,
    )

    assert _resolve_kg_mode("off") == "off"
    assert _resolve_kg_mode("shadow") == "shadow"
    assert _resolve_kg_mode("promoted") == "promoted"


# =============================================================================
# Issue #194 — Joint (z, |delta_AUC|) threshold for Layer 5 severity.
#
# These tests pin the joint-check semantics calibrated 2026-05-14:
#
#     severity ∈ {moderate, high}  ⇔  (z > k) AND (|delta_AUC| > epsilon)
#
# with ``k = HIGH_Z = 5.0`` and ``epsilon = LAYER5_DELTA_AUC_FLOOR_DEFAULT
# = 0.10``. The floor is enforced INSIDE ``hblp_classify`` when its
# optional ``delta_auc`` kwarg is supplied; legacy callers that don't
# thread ``delta_auc`` see legacy z-only behaviour. See the long-form
# comment in ``adaptive_validity_check.py`` for the calibration sweep.
#
# Coverage:
#   1. The floor downgrades severity=high → info when |delta_AUC| ≤ floor.
#   2. The floor downgrades severity=moderate → info when |delta_AUC| ≤ floor.
#   3. The floor does NOT downgrade when |delta_AUC| > floor (real leak).
#   4. ``delta_auc=None`` (legacy z-only) preserves legacy severity.
#   5. Non-finite ``delta_auc`` (NaN) preserves legacy z-only behaviour.
#   6. ``_adversarial_input`` threads delta_AUC from the score dict
#      (integration with hblp_classify).
#   7. End-to-end node behaviour at n=10000 with a benign weak-correlation
#      feature: FPR ≤ 1% on benign features (the issue body's acceptance
#      criterion). Uses a constructed cohort where the benign feature
#      delta_AUC is below the floor but z is above HIGH_Z.
#   8. End-to-end node behaviour preserves TPR on injected leaks at n=2000
#      (small-n behaviour unchanged for real leakage signals).
# =============================================================================


def test_issue_194_hblp_floor_downgrades_high_when_below_floor():
    """Joint check fires: z > HIGH_Z but |delta_AUC| ≤ floor → severity=info."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        LAYER5_DELTA_AUC_FLOOR_DEFAULT,
        hblp_classify,
    )

    # z=8σ (well above 5σ), delta_AUC=0.05 (well below the 0.10 floor).
    cls = hblp_classify(
        z_score=8.0,
        n_positives=500,
        layer_1_declared_safe=False,
        delta_auc=0.05,
    )
    assert cls["severity"] == "info", (
        f"Joint check should force info; got {cls['severity']}, rationale={cls['rationale']}"
    )
    assert cls["delta_auc_below_floor"] is True
    assert cls["delta_auc"] == 0.05
    assert cls["delta_auc_floor"] == LAYER5_DELTA_AUC_FLOOR_DEFAULT
    # The rationale must explicitly mention the joint-check reason so
    # audit-trail readers see WHY the high-z feature was kept.
    assert "joint check" in cls["rationale"].lower() or "#194" in cls["rationale"]


def test_issue_194_hblp_floor_downgrades_moderate_when_below_floor():
    """z in moderate band but |delta_AUC| ≤ floor → severity=info."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        hblp_classify,
    )

    # z=4σ (between MODERATE_Z=3σ and HIGH_Z=5σ), delta_AUC=0.05 below floor.
    cls = hblp_classify(
        z_score=4.0,
        n_positives=500,
        layer_1_declared_safe=False,
        delta_auc=0.05,
    )
    assert cls["severity"] == "info"
    assert cls["delta_auc_below_floor"] is True


def test_issue_194_hblp_floor_does_not_downgrade_real_leak():
    """When |delta_AUC| > floor, severity=high is preserved (no false negative)."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        hblp_classify,
    )

    # Real leak: z=40σ AND delta_AUC=0.4 (above floor).
    cls = hblp_classify(
        z_score=40.0,
        n_positives=500,
        layer_1_declared_safe=False,
        delta_auc=0.4,
    )
    assert cls["severity"] == "high", f"Real leak must stay high; got {cls['severity']}"
    assert cls["delta_auc_below_floor"] is False
    assert cls["delta_auc"] == 0.4


def test_issue_194_hblp_floor_inactive_when_delta_auc_none():
    """``delta_auc=None`` (legacy z-only call) preserves z-only behaviour."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        hblp_classify,
    )

    # z=8σ, no delta_auc supplied → legacy z-only branch.
    cls = hblp_classify(
        z_score=8.0,
        n_positives=500,
        layer_1_declared_safe=False,
    )
    assert cls["severity"] == "high"
    # Audit fields are populated even when the floor was inactive.
    assert cls["delta_auc"] is None
    assert cls["delta_auc_below_floor"] is False


def test_issue_194_hblp_floor_z_positive_inf_strong_effect_keeps_high():
    """Codex pass-1 MEDIUM-1: zero-variance permutation null produces
    ``z=+inf`` when ``actual_auc > null_mean``. Pre-fix the non-finite-z
    guard in ``hblp_classify`` silently dropped these to severity=info,
    creating a false-negative on deterministic high-effect signals.
    Post-fix the joint check provides a principled escape: when
    ``z=+inf`` AND ``|delta_auc| > floor``, severity stays ``high``.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        hblp_classify,
    )

    cls = hblp_classify(
        z_score=float("inf"),
        n_positives=500,
        layer_1_declared_safe=False,
        delta_auc=0.5,
    )
    assert cls["severity"] == "high"
    # Rationale should record the degenerate-null + strong-effect path
    # so audit readers see WHY a non-finite z reached severity=high.
    assert "degenerate" in cls["rationale"].lower() or "194 codex pass-1 MED-1" in cls["rationale"]


def test_issue_194_hblp_floor_z_positive_inf_weak_effect_stays_info():
    """Codex pass-1 MEDIUM-1 (negative complement): when ``z=+inf`` but
    ``|delta_auc| <= floor``, severity is still info — the joint check
    fires on a degenerate null with a weak absolute effect just like
    the finite-z path.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        hblp_classify,
    )

    cls = hblp_classify(
        z_score=float("inf"),
        n_positives=500,
        layer_1_declared_safe=False,
        delta_auc=0.05,
    )
    assert cls["severity"] == "info"


def test_issue_194_hblp_floor_z_negative_inf_stays_info():
    """``z=-inf`` (custom scorer or anti-correlation past the fold).
    Preserve the legacy non-finite-z severity=info fallback for
    negative inf regardless of |delta_auc| — the MED-1 escape only
    fires on positive inf (strong-effect signal).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        hblp_classify,
    )

    cls = hblp_classify(
        z_score=float("-inf"),
        n_positives=500,
        layer_1_declared_safe=False,
        delta_auc=0.5,
    )
    assert cls["severity"] == "info"


def test_issue_194_audit_fields_propagated_through_adversarial_input():
    """Codex pass-1 LOW-1: the three audit fields
    (``delta_auc``, ``delta_auc_floor``, ``delta_auc_below_floor``)
    must be present on ``_adversarial_input``'s output dict, so
    structured-sidecar consumers can branch on them without parsing
    the human-readable evidence string.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        LAYER5_DELTA_AUC_FLOOR_DEFAULT,
        _adversarial_input,
    )

    # Joint check fires: z=8σ, |delta_AUC|=0.05 < 0.10 floor → info.
    score = {
        "z_score": 8.0,
        "actual_auc": 0.55,
        "null_mean": 0.50,
        "null_std": 0.00625,
        "p_value": 0.005,
        "n_permutations": 200,
    }
    ad = _adversarial_input(score, n_train_pos=500, layer_1_declared_safe=False)
    assert "delta_auc" in ad
    assert ad["delta_auc"] == pytest.approx(0.05)
    assert "delta_auc_floor" in ad
    assert ad["delta_auc_floor"] == LAYER5_DELTA_AUC_FLOOR_DEFAULT
    assert "delta_auc_below_floor" in ad
    assert ad["delta_auc_below_floor"] is True

    # Real leak: |delta_AUC|=0.4 > floor → joint check does NOT fire.
    score2 = {
        "z_score": 40.0,
        "actual_auc": 0.95,
        "null_mean": 0.55,
        "null_std": 0.01,
        "p_value": 0.0,
        "n_permutations": 200,
    }
    ad2 = _adversarial_input(score2, n_train_pos=500, layer_1_declared_safe=False)
    assert ad2["delta_auc"] == pytest.approx(0.4)
    assert ad2["delta_auc_below_floor"] is False


def test_issue_194_voter_honors_z_inf_strong_effect_through_kg_path():
    """Codex pass-2 MED-1: when adversarial verdict carries severity=high
    with z=+inf BUT delta_auc_below_floor=False (joint check
    corroborated), the EnsembleVoter must NOT downgrade to no-signal.
    Pre-fix the M3 non-finite-z guard would reject these even though
    they're legitimate deterministic high-effect signals — re-opening
    the false-negative path in KG-active shadow interactions.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _adversarial_input,
        _get_ensemble_voter_class,
    )
    from src.data.kg.types import KGEdge

    # Adversarial score with z=+inf (degenerate null) + strong delta_AUC
    score = {
        "z_score": float("inf"),
        "actual_auc": 0.95,
        "null_mean": 0.50,
        "null_std": 0.0,  # degenerate null → z=inf
        "p_value": 0.0,
        "n_permutations": 200,
    }
    ad = _adversarial_input(score, n_train_pos=500, layer_1_declared_safe=False)
    assert ad["severity"] == "high", (
        f"Issue #194 codex pass-1 MED-1 invariant: _adversarial_input must "
        f"emit severity=high for z=+inf + strong delta_AUC; got {ad['severity']}"
    )
    assert ad["delta_auc_below_floor"] is False

    # Codex pass-3 LOW-1 fix: call voter.vote() DIRECTLY rather than
    # routing through _compose_legacy_verdict with a disjoint KG edge.
    # The disjoint KG edge falls through to the adversarial-alone
    # bypass at adaptive_validity_check.py:1189, which preserves
    # severity without invoking the voter. Direct .vote() guarantees
    # the voter's M3 guard predicate is exercised.
    voter = _get_ensemble_voter_class()()
    kg_edge = KGEdge(
        subject_id="X",
        predicate="treats",
        object_id="Y",
        evidence_source="test",
        score=0.9,
    )
    # Connect feature_entity_ids to target_entity_ids THROUGH the edge
    # so kg_signal is NOT no_signal → voter is fully exercised.
    verdict_obj = voter.vote(
        "test_feat",
        adversarial_verdict=ad,
        kg_edges=(kg_edge,),
        feature_entity_ids=("X",),
        target_entity_ids=("Y",),
    )
    assert verdict_obj.severity == "high", (
        f"Issue #194 codex pass-2 MED-1: voter.vote() must accept z=+inf "
        f"when joint check corroborated; got {verdict_obj.severity}, "
        f"evidence={verdict_obj.evidence}"
    )
    assert verdict_obj.remediation == "drop"
    # Pass-3 MED-2: evidence text must carry the delta_AUC justification.
    evidence_str = "; ".join(verdict_obj.evidence)
    assert "degenerate" in evidence_str.lower() or "194" in evidence_str, (
        f"Pass-3 MED-2: voter evidence must record the joint-check "
        f"corroboration; got {evidence_str!r}"
    )


def test_issue_194_voter_rejects_z_inf_without_joint_check_corroboration():
    """Codex pass-2 MED-1 (complement): legacy non-finite-z guard MUST
    still reject severity=high when delta_auc corroboration is
    UNAVAILABLE (a hand-crafted malformed adversarial input that
    skipped _adversarial_input). Directly invokes voter.vote() to
    exercise the guard.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _get_ensemble_voter_class,
    )

    # Hand-crafted malformed input: severity=high + z=inf + NO
    # delta_auc fields populated. Simulates a stale producer.
    malformed_ad = {
        "layer": "3",
        "severity": "high",
        "remediation": "drop",
        "evidence": "malformed test fixture",
        "z_score": float("inf"),
        "actual_auc": None,
        "null_mean": None,
        "null_std": None,
        "p_value": None,
        "n_permutations": None,
        "_hblp_classified": True,
        # delta_auc missing entirely
    }
    voter = _get_ensemble_voter_class()()
    verdict = voter.vote(
        "test_feat",
        adversarial_verdict=malformed_ad,
    )
    # M3 guard downgrades the malformed high to no-signal. The voter
    # then falls through to abstain (no Layer 1, no KG, no LLM).
    assert verdict.severity != "high", (
        f"Issue #194 codex pass-2 MED-1: malformed adversarial input "
        f"without delta_auc corroboration must NOT drive a high veto; "
        f"got {verdict.severity}"
    )


def test_issue_194_voter_rejects_z_inf_without_hblp_classified_tag():
    """Codex pass-4 LOW-1: a valid-looking z=+inf dict that is OTHERWISE
    well-formed (severity=high + finite delta_auc + finite floor + abs
    above floor) BUT lacks the ``_hblp_classified=True`` producer tag
    is rejected by the voter. The producer tag is the audit-integrity
    anchor — it confirms the dict came from production ``_adversarial_
    input``, not from a hand-built fixture / stale producer.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        LAYER5_DELTA_AUC_FLOOR_DEFAULT,
        _get_ensemble_voter_class,
    )

    # Valid-looking input EXCEPT the _hblp_classified tag is missing.
    untagged_ad = {
        "layer": "3",
        "severity": "high",
        "remediation": "drop",
        "evidence": "untagged test fixture",
        "z_score": float("inf"),
        "actual_auc": 0.95,
        "null_mean": 0.50,
        "null_std": 0.0,
        "p_value": 0.0,
        "n_permutations": 200,
        "delta_auc": 0.45,  # above floor 0.10
        "delta_auc_floor": LAYER5_DELTA_AUC_FLOOR_DEFAULT,
        "delta_auc_below_floor": False,
        # _hblp_classified intentionally missing — producer didn't tag.
    }
    voter = _get_ensemble_voter_class()()
    verdict = voter.vote("test_feat", adversarial_verdict=untagged_ad)
    assert verdict.severity != "high", (
        f"Issue #194 codex pass-4 LOW-1: untagged adversarial input must "
        f"be rejected even when joint-check fields look valid; "
        f"got {verdict.severity}"
    )


def test_issue_194_audit_fields_propagated_through_short_circuit_verdict():
    """Codex pass-1 LOW-1: short-circuit path (too-few-rows / scoring-error)
    must populate the 3 audit fields too — schema uniformity for
    downstream sidecar consumers. Joint check never ran (no score
    computed) so all 3 default sensibly.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        LAYER5_DELTA_AUC_FLOOR_DEFAULT,
        _legacy_short_circuit_verdict,
    )

    v = _legacy_short_circuit_verdict("dummy", evidence="too few rows")
    assert "delta_auc" in v
    assert v["delta_auc"] is None
    assert v["delta_auc_floor"] == LAYER5_DELTA_AUC_FLOOR_DEFAULT
    assert v["delta_auc_below_floor"] is False


def test_issue_194_hblp_floor_inactive_when_delta_auc_nan():
    """``delta_auc=nan`` (degenerate score) preserves z-only behaviour."""
    import math

    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        hblp_classify,
    )

    cls = hblp_classify(
        z_score=8.0,
        n_positives=500,
        layer_1_declared_safe=False,
        delta_auc=math.nan,
    )
    assert cls["severity"] == "high"
    assert cls["delta_auc"] is None
    assert cls["delta_auc_below_floor"] is False


def test_issue_194_adversarial_input_threads_delta_auc_from_score():
    """``_adversarial_input`` computes delta_AUC and threads it to hblp_classify."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _adversarial_input,
    )

    # Construct a score dict where z is well above HIGH_Z but
    # |delta_AUC| = 0.05 < floor=0.10. Without the joint check this
    # would emit severity=high; with the joint check it emits info.
    score = {
        "z_score": 8.0,
        "actual_auc": 0.55,
        "null_mean": 0.50,
        "null_std": 0.00625,  # 0.05 / 8
        "p_value": 0.005,
        "n_permutations": 200,
    }
    ad = _adversarial_input(score, n_train_pos=500, layer_1_declared_safe=False)
    assert ad["severity"] == "info", (
        f"Joint check must propagate via _adversarial_input; got {ad['severity']}, "
        f"evidence={ad['evidence']}"
    )
    assert ad["remediation"] == "keep"
    # The evidence text records the joint-check footnote.
    assert "194" in ad["evidence"] or "joint check" in ad["evidence"].lower()


def test_issue_194_adversarial_input_real_leak_stays_high():
    """``_adversarial_input`` retains severity=high for real leaks."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _adversarial_input,
    )

    score = {
        "z_score": 40.0,
        "actual_auc": 0.95,
        "null_mean": 0.55,
        "null_std": 0.01,
        "p_value": 0.0,
        "n_permutations": 200,
    }
    ad = _adversarial_input(score, n_train_pos=500, layer_1_declared_safe=False)
    assert ad["severity"] == "high"
    assert ad["remediation"] == "drop"


def test_issue_194_layer5_benign_fpr_at_n_10k_under_one_percent():
    """Layer 5 acceptance criterion: FPR ≤ 1% on benign weak predictors at n=10000.

    Calls the production severity classifier
    (``_adversarial_input`` → ``hblp_classify``) over 10 i.i.d. synthetic
    cohorts at n=10000 from ``synthetic_rwd_realistic`` (signal_scale=1.0)
    on the benign 'age' feature, and asserts that severity=high fires
    on 0 of 10 cohorts (matches the calibration sweep result; the larger
    50-replicate sweep at
    ``scripts/calibration/run_layer5_joint_threshold_sweep.py`` confirmed
    0 flags out of 50 runs at the calibrated (k=5.0, ε=0.10)).

    Test-cost note: a 30-replicate version (matching the calibration
    sweep's seed count) crashed pytest-xdist workers (16-worker OOM at
    30 × 10k DataFrames × 200 permutations). 10 replicates × 200
    permutations × n=10000 is enough to detect a regression that
    re-introduces large-n FPR (pre-fix flag rate was ~40%, so even a
    single flag in 10 trials would be 17x the new joint-check FPR).

    'age' is a legitimate weak predictor in ``_generate_target`` (per
    the regime's calibration) — its single-feature AUC is ~0.54, with
    delta_AUC empirically < 0.10 (the calibrated floor). Pre-issue-#194
    the legacy 5σ threshold flagged it ~40% of the time at n=10k.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _adversarial_input,
    )
    from src.data.adversarial_leakage import compute_adversarial_score
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    n_replicates = 10
    n_flagged = 0
    n_run = 0
    for trial in range(n_replicates):
        cohort = generate_rwd_realistic(
            RwdRealisticConfig(
                n_patients=10000,
                prevalence=0.024,
                missing_demo_rate=0.0,
                signal_scale=1.0,
                seed=42 + trial,
            )
        )
        target = cohort["treatment_initiated"].to_numpy(dtype=int)
        if len(np.unique(target)) < 2:
            continue
        age = cohort["age"].to_numpy(dtype=float)
        score = compute_adversarial_score(age, target, n_permutations=200, seed=7)
        ad = _adversarial_input(
            score, n_train_pos=int(np.sum(target == 1)), layer_1_declared_safe=False
        )
        n_run += 1
        if ad["severity"] == "high":
            n_flagged += 1
    fpr = n_flagged / n_run if n_run > 0 else float("nan")
    assert n_flagged == 0, (
        f"Issue #194 acceptance criterion: severity=high on benign 'age' must "
        f"be 0 at n=10000 (FPR ≤ 1% target); observed {n_flagged}/{n_run} "
        f"(empirical FPR={fpr:.2%})"
    )


def test_issue_194_layer5_tpr_preserved_on_injected_leaks_at_n_2000():
    """End-to-end Layer 5 acceptance criterion (companion to FPR test):
    TPR on injected leak patterns at n=2000 stays at 100% — joint check
    does NOT degrade small-n leakage detection.

    Runs the full node ONCE per leak pattern at n=2000 (cheap; 8k rows
    per cohort × 1 replicate = ~40MB transient). Asserts each of the 4
    injection patterns is flagged. Same pattern as the pre-existing
    ``test_post_index_aggregation_leak_is_caught_by_layer_3`` at the
    score-only level; here we verify it propagates through the full
    node post-issue-#194 joint check.

    The calibration sweep at
    ``scripts/calibration/run_layer5_joint_threshold_sweep.py`` showed
    minimum injected-leak |delta_AUC|=0.354 (treatment_leaked_code at
    n=2000), well above the 0.10 floor — joint check fires on z AND
    delta_AUC, so leak detection is unchanged.
    """
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    leak_patterns_and_cols = [
        ("post_index_aggregation", "post_index_med_count_LEAK"),
        ("post_hoc_termination", "months_remaining_eligibility_LEAK"),
        ("treatment_leaked_code", "has_z79_long_term_drug_LEAK"),
        ("spurious_correlation", "spurious_score_LEAK"),
    ]
    for pattern, leak_col in leak_patterns_and_cols:
        cohort = generate_rwd_realistic(
            RwdRealisticConfig(
                n_patients=2000,
                prevalence=0.024,
                missing_demo_rate=0.0,
                signal_scale=1.0,
                leakage_pattern=pattern,
                seed=42,
            )
        )
        train_df = pd.DataFrame(
            {
                leak_col: cohort[leak_col].to_numpy(dtype=float),
                "age": cohort["age"].to_numpy(dtype=float),
                "treatment_initiated": cohort["treatment_initiated"].to_numpy(dtype=int),
            }
        )
        state = _make_state(train_df, "treatment_initiated", feature_manifest_source=None)
        result = _run(state)
        flagged = set(result.get("adaptive_flagged_features") or [])
        assert leak_col in flagged, (
            f"Issue #194 acceptance criterion: joint check must preserve "
            f"TPR=100% on injected leak pattern {pattern!r}; "
            f"got flagged={flagged}"
        )


def test_issue_194_floor_constant_pinned_to_calibrated_value():
    """Pin the calibrated floor at 0.10. Future changes to this value
    require a new sweep — codex pass-1 question (a) preservation pin.

    The value 0.10 is the upper-rounded p99 of legitimate weak demo
    features at n=10000 from the calibration sweep (run via
    ``scripts/calibration/run_layer5_joint_threshold_sweep.py``). Lower
    values fail the FPR ≤ 1% criterion; higher values are safe but
    cost detection precision on borderline leaks.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        LAYER5_DELTA_AUC_FLOOR_DEFAULT,
    )

    assert LAYER5_DELTA_AUC_FLOOR_DEFAULT == 0.10, (
        "Calibrated value pinned at 0.10. Any change must re-run "
        "scripts/calibration/run_layer5_joint_threshold_sweep.py and "
        "update both this assertion and the long-form code comment "
        "in adaptive_validity_check.py."
    )


@pytest.mark.parametrize("n", [1000, 5000, 10000, 50000])
def test_issue_194_joint_check_holds_across_cohort_sizes(n: int):
    """Codex pass-1 question (a): the joint check must hold the FPR
    contract across n ∈ {1k, 5k, 10k, 50k}.

    This is a single-cohort smoke at each n (not a full sweep — that's
    in ``scripts/calibration/run_layer5_joint_threshold_sweep.py``). We
    verify that for the canonical benign 'age' feature, ``_adversarial_
    input`` does NOT emit severity=high when delta_AUC stays below the
    calibrated floor.

    Three-cohort replication per n to reduce single-seed flake risk
    while keeping the test fast (≤ ~30s total across all n).
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _adversarial_input,
    )
    from src.data.adversarial_leakage import compute_adversarial_score
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    n_high = 0
    n_run = 0
    for trial in range(3):
        cohort = generate_rwd_realistic(
            RwdRealisticConfig(
                n_patients=n,
                prevalence=0.024,
                missing_demo_rate=0.0,
                signal_scale=1.0,
                seed=42 + trial,
            )
        )
        target = cohort["treatment_initiated"].to_numpy(dtype=int)
        if len(np.unique(target)) < 2:
            continue
        age = cohort["age"].to_numpy(dtype=float)
        score = compute_adversarial_score(age, target, n_permutations=200, seed=7)
        n_run += 1
        ad = _adversarial_input(score, n_train_pos=int(np.sum(target == 1)))
        if ad["severity"] == "high":
            n_high += 1
    # Three replicates × FPR ≤ 1% target → expect 0 false positives at
    # any n. Anchored at 0/3 not "≤ 1" to keep the regression strict.
    assert n_high == 0, (
        f"Joint check failed at n={n}: severity=high on {n_high}/{n_run} "
        f"benign 'age' cohorts; expected 0 false positives."
    )


# =============================================================================
# Issue #196 Phase 3.3 — codex pass-1 regression pins.
# =============================================================================


class TestClassifyAblationSeverityCodexPass1:
    """Pin codex pass-1 MED-1 + MED-2 fixes for ``_classify_ablation_severity``.

    MED-1: NaN z-score must NOT escape via the strong-effect path. The null
    distribution being undefined means the ablation sub-test has no
    statistical anchor; the verdict must fall back to permutation-only.

    MED-2: Negative ``delta_auc`` must NOT escape via the strong-effect path.
    ``delta_auc = full_auc - ablated_auc`` > 0 means the feature ADDS to
    joint AUC (the leak-carrier case); a NEGATIVE delta means removing the
    feature IMPROVES the joint model — model-instability noise from
    multicollinearity, NOT a leak.
    """

    def test_nan_z_blocks_strong_effect_escape(self) -> None:
        """codex MED-1: NaN z with finite delta > 0.30 must NOT classify high."""
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _classify_ablation_severity,
        )

        row = {
            "feature": "feat",
            "full_auc": 0.95,
            "ablated_auc": 0.55,
            "delta_auc": 0.40,  # well above 0.30 strong-effect threshold
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "z_score": float("nan"),  # null undefined
            "suspicious": False,
        }
        # Pre-MED-1 fix this returned "high" (false positive on undefined null).
        # Post-fix: NaN z degrades to info per the documented contract.
        assert _classify_ablation_severity(row) == "info"

    def test_negative_delta_blocks_strong_effect_escape(self) -> None:
        """codex MED-2: large NEGATIVE delta_auc must NOT classify high.

        delta_auc < 0 means model improves when the feature is dropped.
        This is multicollinearity / nuisance-variable behaviour, not a leak.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _classify_ablation_severity,
        )

        row = {
            "feature": "noisy_nuisance",
            "full_auc": 0.75,
            "ablated_auc": 0.95,  # ablated model IMPROVES
            "delta_auc": -0.20,  # |delta| = 0.20 below floor; but even at -0.40 below should hold
            "null_mean": -0.20,
            "null_std": 0.01,
            "z_score": 0.0,
            "suspicious": False,
        }
        assert _classify_ablation_severity(row) == "info"

        # Magnitude > 0.30 with negative sign: pre-MED-2 fix this returned
        # "high" via abs(delta_f) > 0.30; post-fix: signed escape requires
        # positive delta.
        row["delta_auc"] = -0.40
        assert _classify_ablation_severity(row) == "info"

    def test_positive_strong_delta_with_finite_z_still_classifies_high(self) -> None:
        """Sanity: positive delta > 0.30 AND finite z DOES classify high.

        Pin the contract MED-1/MED-2 must not over-tighten — legitimate
        strong-effect leak-carriers (the intended positive case) still
        escalate.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _classify_ablation_severity,
        )

        row = {
            "feature": "leak_carrier",
            "full_auc": 0.95,
            "ablated_auc": 0.55,
            "delta_auc": 0.40,
            "null_mean": 0.35,
            "null_std": 0.10,
            "z_score": 0.5,  # below permutation HIGH_Z=5.0 but strong-effect escape fires
            "suspicious": False,
        }
        assert _classify_ablation_severity(row) == "high"

    def test_positive_inf_z_with_above_floor_delta_classifies_high(self) -> None:
        """Mirror of hblp_classify's issue #194 MED-1 escape: z=+inf
        (degenerate null with zero variance, not undefined null) plus
        |delta| above the issue #194 floor must classify high.

        This is distinct from NaN z (codex MED-1) — +inf z means the null
        is DEFINED but degenerate; NaN z means the null is UNDEFINED.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _classify_ablation_severity,
        )

        row = {
            "feature": "degenerate_null_leak",
            "full_auc": 0.95,
            "ablated_auc": 0.80,
            "delta_auc": 0.15,  # below strong-effect 0.30, above floor 0.10
            "null_mean": 0.05,
            "null_std": 0.0,  # zero variance → z=+inf via compute_feature_ablation
            "z_score": float("inf"),
            "suspicious": True,
        }
        assert _classify_ablation_severity(row) == "high"

    def test_nan_delta_classifies_info(self) -> None:
        """NaN delta_auc → info regardless of z. Degradation contract."""
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _classify_ablation_severity,
        )

        row = {
            "feature": "broken_feat",
            "full_auc": 0.95,
            "ablated_auc": float("nan"),
            "delta_auc": float("nan"),
            "null_mean": 0.0,
            "null_std": 0.01,
            "z_score": 10.0,  # would classify high if delta were finite
            "suspicious": False,
        }
        assert _classify_ablation_severity(row) == "info"

    def test_negative_delta_with_high_z_classifies_info(self) -> None:
        """codex pass-2 MED-1: extend MED-2 signed-delta rule to the z-band
        ladder. ``delta_auc=-0.20, z=6.0`` must NOT classify high via the
        ``z > z_threshold`` branch — negative delta = nuisance behaviour.

        Pre-pass-2 the z-band ladder used ``abs(delta_f) > floor``, which
        would classify this row as high. Post-fix the ladder uses signed
        ``delta_f > floor`` symmetrically with the strong-effect escape.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _classify_ablation_severity,
        )

        # Negative delta, |delta| > floor (0.10), z > HIGH_Z (5.0)
        # Strong-effect escape doesn't fire (delta is negative).
        # Pre-fix: z-band ladder fired high via abs(); post-fix: info.
        row = {
            "feature": "noisy_nuisance_high_z",
            "full_auc": 0.75,
            "ablated_auc": 0.95,
            "delta_auc": -0.20,
            "null_mean": -0.20,
            "null_std": 0.01,
            "z_score": 6.0,
            "suspicious": False,
        }
        assert _classify_ablation_severity(row) == "info"

        # Same setup with moderate z. Pre-fix: 'moderate'; post-fix: 'info'.
        row["z_score"] = 4.0
        assert _classify_ablation_severity(row) == "info"


# ============================================================================
# Issue #212 — Layer 4 fires on pre-joint-check severity.
#
# Pre-#212: when the issue #194 joint check clamped severity to ``info``
# (z passes the band but |delta_AUC| ≤ floor), the orchestrator's Layer 4
# trigger never fired — it read the joint-clamped ``severity`` field
# instead of the underlying z-only band. That starved the LLM-verdict
# audit channel for exactly the class of features the joint check was
# designed to protect (legitimate weak signals).
#
# Post-#212: ``hblp_classify`` publishes ``severity_pre_joint_check``
# (the z-only band before the joint-clamp downgrade), and the
# orchestrator's Layer 4 trigger reads THAT field. The final
# ``severity`` is unchanged (still joint-clamped — issue #194's bar is
# preserved). The pre-joint-check severity is a parallel audit channel,
# not a relaxation of the joint check.
# ============================================================================


class TestIssue212PreJointCheckSeverity:
    """Pin the issue #212 contract on ``severity_pre_joint_check``."""

    def test_hblp_classify_publishes_severity_pre_joint_check_high_z_weak_delta(
        self,
    ) -> None:
        """When the joint check clamps a high-z signal to ``info`` because
        ``|delta_AUC| ≤ floor``, ``severity_pre_joint_check`` MUST preserve
        the ``high`` band that z alone would have selected.

        Setup: z = 6.0σ (above HIGH_Z=5.0, n_pos=200 + declared_safe=False
        so no HBLP relaxation), delta_auc = 0.05 (below 0.10 floor).
        Pre-#212 the joint check downgraded severity to ``info`` and the
        old z-band was unrecoverable. Post-#212 the new field carries it.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            hblp_classify,
        )

        result = hblp_classify(
            z_score=6.0,
            n_positives=200,
            layer_1_declared_safe=False,
            delta_auc=0.05,  # below floor 0.10
        )
        assert result["severity"] == "info"  # joint check clamps
        assert result["severity_pre_joint_check"] == "high"
        assert result["delta_auc_below_floor"] is True

    def test_hblp_classify_publishes_severity_pre_joint_check_moderate_z_weak_delta(
        self,
    ) -> None:
        """Moderate-z signal clamped by joint check: severity_pre_joint_check
        preserves the moderate band so Layer 4 can still see the signal.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            hblp_classify,
        )

        result = hblp_classify(
            z_score=4.0,  # in moderate band (3.0 < z ≤ 5.0)
            n_positives=200,
            layer_1_declared_safe=False,
            delta_auc=0.03,  # below floor
        )
        assert result["severity"] == "info"  # joint check clamps
        assert result["severity_pre_joint_check"] == "moderate"

    def test_hblp_classify_severity_pre_matches_final_severity_when_joint_inactive(
        self,
    ) -> None:
        """When the joint check does NOT fire (|delta_AUC| > floor),
        ``severity_pre_joint_check`` MUST match the final ``severity``.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            hblp_classify,
        )

        result = hblp_classify(
            z_score=6.0,
            n_positives=200,
            layer_1_declared_safe=False,
            delta_auc=0.25,  # well above floor
        )
        assert result["severity"] == "high"
        assert result["severity_pre_joint_check"] == "high"
        assert result["delta_auc_below_floor"] is False

    def test_hblp_classify_severity_pre_matches_when_no_delta_auc_supplied(self) -> None:
        """Legacy callers that omit ``delta_auc`` MUST see
        ``severity_pre_joint_check == severity`` (no clamp to consider).
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            hblp_classify,
        )

        # Legacy call shape — no delta_auc.
        result = hblp_classify(
            z_score=6.0,
            n_positives=200,
            layer_1_declared_safe=False,
        )
        assert result["severity"] == "high"
        assert result["severity_pre_joint_check"] == "high"

        # Moderate z with no delta.
        result2 = hblp_classify(
            z_score=4.0,
            n_positives=200,
            layer_1_declared_safe=False,
        )
        assert result2["severity"] == "moderate"
        assert result2["severity_pre_joint_check"] == "moderate"

        # Info z with no delta.
        result3 = hblp_classify(
            z_score=2.0,
            n_positives=200,
            layer_1_declared_safe=False,
        )
        assert result3["severity"] == "info"
        assert result3["severity_pre_joint_check"] == "info"

    def test_hblp_classify_severity_pre_high_on_z_positive_inf_strong_effect(
        self,
    ) -> None:
        """The z=+inf strong-effect escape (issue #194 codex pass-1 MED-1)
        sets final severity=high; severity_pre_joint_check MUST match.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            hblp_classify,
        )

        result = hblp_classify(
            z_score=float("inf"),
            n_positives=200,
            layer_1_declared_safe=False,
            delta_auc=0.50,  # well above floor
        )
        assert result["severity"] == "high"
        assert result["severity_pre_joint_check"] == "high"

    def test_adversarial_input_propagates_severity_pre_joint_check(self) -> None:
        """``_adversarial_input`` MUST surface ``severity_pre_joint_check``
        from the underlying ``hblp_classify`` result so the orchestrator
        can read it for the Layer 4 trigger.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _adversarial_input,
        )

        # Weak-effect signal that the joint check clamps to info.
        # z = 6.0, actual_auc=0.55, null_mean=0.50 → delta_auc=0.05 < floor 0.10
        score = {
            "z_score": 6.0,
            "actual_auc": 0.55,
            "null_mean": 0.50,
            "null_std": 0.0083,
            "p_value": 0.001,
            "n_permutations": 200,
        }
        adv_input = _adversarial_input(
            score,
            n_train_pos=200,
            layer_1_declared_safe=False,
        )
        assert adv_input["severity"] == "info"  # joint clamp
        assert adv_input["severity_pre_joint_check"] == "high"
        assert adv_input["delta_auc_below_floor"] is True
        assert adv_input["_hblp_classified"] is True

    def test_adversarial_input_degenerate_z_publishes_severity_pre_info(self) -> None:
        """Degenerate z (NaN, None) → ``severity_pre_joint_check == 'info'``
        matching the final ``severity``. No z-band to recover.
        """
        import math

        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _adversarial_input,
        )

        score = {
            "z_score": float("nan"),
            "actual_auc": float("nan"),
            "null_mean": float("nan"),
            "null_std": 0.0,
            "p_value": 1.0,
            "n_permutations": 0,
        }
        adv_input = _adversarial_input(
            score,
            n_train_pos=200,
            layer_1_declared_safe=False,
        )
        assert adv_input["severity"] == "info"
        assert adv_input["severity_pre_joint_check"] == "info"

        # Also verify with z_score=None.
        score2 = {**score, "z_score": None}
        adv_input2 = _adversarial_input(
            score2,
            n_train_pos=200,
            layer_1_declared_safe=False,
        )
        assert adv_input2["severity_pre_joint_check"] == "info"
        # Quiet the unused-var lint on math import for static analyzers.
        _ = math.nan

    def test_adversarial_input_no_delta_auc_pre_matches_post(self) -> None:
        """When ``compute_adversarial_score`` doesn't supply finite
        actual_auc/null_mean (custom scorer), ``delta_auc`` is None and
        joint check can't fire. ``severity_pre_joint_check`` MUST equal
        the final ``severity``.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _adversarial_input,
        )

        score = {
            "z_score": 4.0,
            "actual_auc": float("nan"),  # missing; joint check skipped
            "null_mean": float("nan"),
            "null_std": 0.01,
            "p_value": 0.0001,
            "n_permutations": 200,
        }
        adv_input = _adversarial_input(
            score,
            n_train_pos=200,
            layer_1_declared_safe=False,
        )
        # Note: actual_auc / null_mean are NaN so _adversarial_input
        # gates them out; the verdict's "actual_auc" field becomes None.
        # The joint check can't fire without delta_auc, so severity ==
        # severity_pre_joint_check.
        assert adv_input["severity"] == adv_input["severity_pre_joint_check"]
        assert adv_input["severity"] == "moderate"  # z=4 in moderate band

    def test_combine_ablation_escalates_severity_pre_joint_check(self) -> None:
        """Issue #212 codex pass-2 MED-1: ``_combine_ablation_with_permutation``
        MUST escalate ``severity_pre_joint_check`` symmetrically with the
        final ``severity``. Otherwise the orchestrator's Layer 4 trigger
        (which reads ``severity_pre_joint_check``) skips ablation-only
        signals — losing the LLM review #196's ablation pass added to
        catch interaction-only leaks.
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _combine_ablation_with_permutation,
        )

        # Permutation path produces info severity (z below noise band).
        perm_input = {
            "layer": "3",
            "severity": "info",
            "severity_pre_joint_check": "info",
            "remediation": "keep",
            "evidence": "permutation info",
            "z_score": 2.0,
            "actual_auc": 0.52,
            "null_mean": 0.50,
            "null_std": 0.01,
            "p_value": 0.5,
            "n_permutations": 200,
            "delta_auc": 0.02,
            "delta_auc_floor": 0.10,
            "delta_auc_below_floor": True,
            "_hblp_classified": True,
        }
        # Ablation row escalates to moderate (above floor + above
        # moderate z-band but below high z-band).
        ablation_row = {
            "feature": "feat_x",
            "full_auc": 0.85,
            "ablated_auc": 0.70,
            "delta_auc": 0.15,  # above floor 0.10 (interaction leak signature)
            "null_mean": 0.0,
            "null_std": 0.03,
            "z_score": 4.0,  # above MODERATE_Z=3.0, below HIGH_Z=5.0
            "suspicious": True,
        }
        combined = _combine_ablation_with_permutation(perm_input, ablation_row)
        # Final severity escalated by MAX rule.
        assert combined["severity"] == "moderate"
        # Issue #212 pass-2 MED-1: severity_pre_joint_check MUST also
        # be escalated so the orchestrator's Layer 4 trigger fires.
        assert combined["severity_pre_joint_check"] == "moderate", (
            f"Issue #212 pass-2 MED-1: severity_pre_joint_check must "
            f"be escalated symmetrically with severity when ablation "
            f"escalates. Got severity_pre_joint_check="
            f"{combined['severity_pre_joint_check']!r}, "
            f"severity={combined['severity']!r}"
        )

    def test_combine_ablation_no_escalation_preserves_severity_pre(self) -> None:
        """Symmetric pin: when ablation does NOT escalate (e.g.
        ablation severity = info), ``severity_pre_joint_check`` MUST
        remain at the permutation path's value (no spurious
        downgrade).
        """
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            _combine_ablation_with_permutation,
        )

        # Permutation at moderate (pre-joint-check), joint-clamped to
        # info on final severity. severity_pre_joint_check='moderate'.
        perm_input = {
            "layer": "3",
            "severity": "info",  # joint-clamped
            "severity_pre_joint_check": "moderate",
            "remediation": "keep",
            "evidence": "permutation moderate joint-clamped",
            "z_score": 4.0,
            "actual_auc": 0.55,
            "null_mean": 0.50,
            "null_std": 0.0125,
            "p_value": 0.001,
            "n_permutations": 200,
            "delta_auc": 0.05,
            "delta_auc_floor": 0.10,
            "delta_auc_below_floor": True,
            "_hblp_classified": True,
        }
        # Ablation row that produces severity='info' (no escalation).
        ablation_row = {
            "feature": "feat_y",
            "full_auc": 0.55,
            "ablated_auc": 0.54,
            "delta_auc": 0.01,
            "null_mean": 0.01,
            "null_std": 0.005,
            "z_score": 1.0,  # below MODERATE_Z
            "suspicious": False,
        }
        combined = _combine_ablation_with_permutation(perm_input, ablation_row)
        # No escalation: severity stays at joint-clamped 'info'.
        assert combined["severity"] == "info"
        # severity_pre_joint_check preserved at 'moderate' (NOT
        # downgraded to 'info' by ablation's info-severity).
        assert combined["severity_pre_joint_check"] == "moderate"
