"""Phase 1 — FDR confident set as the firing/severity driver (Layer-4 redesign).

The dynamic Benjamini-Hochberg confident set replaces the static z>5σ band as
the auto-fire (severity=high) driver. A feature is a *confident leak* iff it is
BH-rejected at FDR ``q`` AND its absolute effect exceeds the issue-#194 floor.

This file pins the Phase-1 wiring:
  * ``_apply_fdr_firing_override`` — confident-set membership drives the HIGH
    tier (promote where FDR confidently flags; demote a σ-band high that FDR is
    NOT confident about to moderate/review — suspicious, not auto-dropped).
  * the ``adaptive_validity_check`` node with FDR enabled (default-on): a real
    confident leak → high/drop; the σ-band fallback when a cohort is too wide
    for BH at the permutation cap.
"""

from __future__ import annotations

from typing import Any

MOD = "src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check"


def _adv_input(**overrides: Any) -> dict[str, Any]:
    """A representative ``_adversarial_input`` output dict for override tests."""
    base: dict[str, Any] = {
        "layer": "3",
        "severity": "info",
        "severity_pre_joint_check": "info",
        "remediation": "keep",
        "evidence": "Layer 3 adversarial discriminator: z=1.20σ (below noise floor)",
        "z_score": 1.2,
        "actual_auc": 0.55,
        "null_mean": 0.50,
        "null_std": 0.04,
        "p_value": 0.30,
        "n_permutations": 1000,
        "delta_auc": 0.05,
        "delta_auc_floor": 0.10,
        "delta_auc_below_floor": True,
        "_hblp_classified": True,
    }
    base.update(overrides)
    return base


def test_fdr_override_confident_promotes_to_high_drop():
    """A feature the FDR confident set flags fires HIGH/drop even if the static
    σ-band only saw 'info' — the adaptive benefit of replacing the fixed
    threshold with a cohort-relative FDR decision."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(
        severity="info", remediation="keep", delta_auc=0.40, delta_auc_below_floor=False
    )
    out = mod._apply_fdr_firing_override(adv, is_confident=True, fdr_q=0.10)
    assert out["severity"] == "high"
    assert out["remediation"] == "drop"
    assert out["fdr_confident"] is True


def test_fdr_override_not_confident_demotes_sigma_high_to_review():
    """A σ-band 'high' (z>5σ) the FDR set does NOT confidently confirm is demoted
    to moderate/review — suspicious but not auto-dropped (FDR is now the
    auto-fire authority)."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(
        severity="high",
        severity_pre_joint_check="high",
        remediation="drop",
        z_score=6.0,
        actual_auc=0.70,
        delta_auc=0.20,
        delta_auc_below_floor=False,
    )
    out = mod._apply_fdr_firing_override(adv, is_confident=False, fdr_q=0.10)
    assert out["severity"] == "moderate"
    assert out["remediation"] == "ambiguous"
    assert out["fdr_confident"] is False


def test_fdr_override_not_confident_leaves_moderate_and_info_unchanged():
    """When FDR is not confident, a σ-band moderate stays moderate (review) and
    an info stays info — only the high→moderate demotion changes the verdict."""
    import importlib

    mod = importlib.import_module(MOD)
    for sev, rem in [("moderate", "ambiguous"), ("info", "keep")]:
        adv = _adv_input(severity=sev, severity_pre_joint_check=sev, remediation=rem)
        out = mod._apply_fdr_firing_override(adv, is_confident=False, fdr_q=0.10)
        assert out["severity"] == sev
        assert out["remediation"] == rem
        assert out["fdr_confident"] is False


def test_fdr_override_does_not_mutate_input():
    """The override returns a new dict; the caller's adv_input is untouched."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(severity="high", remediation="drop")
    snapshot = dict(adv)
    _ = mod._apply_fdr_firing_override(adv, is_confident=False, fdr_q=0.10)
    assert adv == snapshot


# ---------------------------------------------------------------------------
# Node-level FDR integration: default-on firing, σ-band fallback, off-switch.
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _make_state(train_df: pd.DataFrame, target: str, **overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "experiment_id": "test-fdr",
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {
            "prediction_target": target,
            "required_features": [c for c in train_df.columns if c != target],
            "excluded_features": [],
            "feature_manifest_source": None,  # no Layer 1 → exercise the Layer-3 FDR path
        },
        "leakage_findings": [],
        "leaked_features": [],
    }
    state.update(overrides)
    return state


def _run(state: dict[str, Any]) -> dict[str, Any]:
    import importlib

    mod = importlib.import_module(MOD)
    return asyncio.run(mod.adaptive_validity_check(state))


def _leak_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "leak_perfect": y.astype(float) + rng.normal(0, 0.01, n),  # near-perfect leak
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "y": y,
        }
    )


def test_node_fdr_active_drives_high_on_leak():
    """FDR is the default firing driver: a confident leak → flagged HIGH, the
    node reports leakage_fdr.active=True, and the leak's verdict records
    fdr_confident=True (proving FDR — not the σ-band — drove the decision)."""
    result = _run(_make_state(_leak_df(), "y"))
    fdr = result.get("leakage_fdr")
    assert fdr is not None and fdr["active"] is True
    assert fdr["n_confident"] >= 1
    flagged = set(result["adaptive_flagged_features"])
    assert "leak_perfect" in flagged
    assert "noise_a" not in flagged and "noise_b" not in flagged
    # the confident-set summary records WHICH features FDR confidently flagged
    assert "leak_perfect" in fdr["confident_features"]


def test_node_fdr_sigma_fallback_when_cohort_exceeds_perm_cap():
    """When the BH feasibility floor exceeds the permutation cap, FDR is
    infeasible: the node falls back to the σ-band (leakage_fdr.active=False) and
    the obvious leak is STILL caught by the static threshold."""
    state = _make_state(_leak_df(), "y", adaptive_fdr_max_permutations=5)
    result = _run(state)
    fdr = result.get("leakage_fdr")
    assert fdr is not None and fdr["active"] is False
    assert "fallback" in (fdr.get("reason") or "").lower()
    assert "leak_perfect" in set(result["adaptive_flagged_features"])


def test_node_fdr_disabled_uses_sigma_band():
    """adaptive_fdr_enabled=False forces the legacy σ-band path; the leak is
    still flagged and leakage_fdr.active=False."""
    state = _make_state(_leak_df(), "y", adaptive_fdr_enabled=False)
    result = _run(state)
    fdr = result.get("leakage_fdr")
    assert fdr is not None and fdr["active"] is False
    assert "leak_perfect" in set(result["adaptive_flagged_features"])


def test_node_adaptive_escalates_regardless_of_skip_leakage_check():
    """#533 (Option 2): ``skip_leakage_check`` gates ONLY the legacy name-based
    detect_leakage node — the data-driven adaptive/FDR layer always runs as the
    safety net. The leak is flagged even when skip_leakage_check=True."""
    state = _make_state(_leak_df(), "y", skip_leakage_check=True)
    result = _run(state)
    assert "leak_perfect" in set(result["adaptive_flagged_features"]), (
        "adaptive_validity_check must escalate the leak regardless of "
        "skip_leakage_check (which gates only the legacy name-based detector)"
    )
