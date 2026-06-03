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
# Layer-1 declared-safe prior must govern the FDR auto-fire too (root-cause fix
# for the dx_l50_1_count false positive, faithful Optum D4-run 2026-05-28).
#
# The adversarial discriminator cannot distinguish a strong PRE-INDEX predictor
# from a leak — both predict the target. ``hblp_classify`` compensates with a
# 1.5x declared-safe threshold inflation, so a manifest-cleared pre-index
# feature lands at info/moderate. But ``_apply_fdr_firing_override`` historically
# escalated ANY BH-confident feature to high/drop UNCONDITIONALLY, bypassing that
# prior and silently dropping legitimate pre-index confounders (e.g. CSU
# dx_l50_1_count: knowable_at<=index, σ-band info, yet auto-dropped). The fix:
# honor declared-safe in the FDR tier as well — route the FDR-vs-manifest
# disagreement to review instead of auto-dropping, UNLESS the σ-band (already
# inflated) independently reached high (overwhelming evidence still fires).
# ---------------------------------------------------------------------------


def test_fdr_override_confident_declared_safe_info_routes_to_review_not_drop():
    """A Layer-1 declared-safe (pre-index) feature the FDR set flags confident,
    whose inflated σ-band severity is NOT high, must NOT be auto-dropped — route
    to review (moderate/ambiguous) so the structural decider / Layer-4 / operator
    adjudicates. This is the dx_l50_1_count case (σ-band info, BH-confident)."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(
        severity="info",
        severity_pre_joint_check="info",
        remediation="keep",
        z_score=5.80,
        actual_auc=0.70,
        delta_auc=0.16,
        delta_auc_below_floor=False,
    )
    out = mod._apply_fdr_firing_override(
        adv, is_confident=True, fdr_q=0.10, layer_1_declared_safe=True
    )
    assert out["severity"] == "moderate"
    assert out["remediation"] == "ambiguous"
    assert out["fdr_confident"] is True
    assert "declared-safe" in out["evidence"].lower()


def test_fdr_override_confident_declared_safe_moderate_stays_review():
    """A declared-safe feature whose σ-band is moderate and which FDR flags
    confident is also NOT force-dropped — it stays at review, not high/drop."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(
        severity="moderate",
        severity_pre_joint_check="moderate",
        remediation="ambiguous",
        delta_auc=0.16,
        delta_auc_below_floor=False,
    )
    out = mod._apply_fdr_firing_override(
        adv, is_confident=True, fdr_q=0.10, layer_1_declared_safe=True
    )
    assert out["severity"] == "moderate"
    assert out["remediation"] == "ambiguous"


def test_fdr_override_confident_declared_safe_but_sigma_high_still_drops():
    """Overwhelming evidence: a declared-safe feature whose (already 1.5x-inflated)
    σ-band severity ALREADY reached high still fires high/drop. Declared-safe
    raises the bar; it does not grant immunity when the bar is cleared anyway."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(
        severity="high",
        severity_pre_joint_check="high",
        remediation="drop",
        z_score=15.0,
        actual_auc=0.95,
        delta_auc=0.45,
        delta_auc_below_floor=False,
    )
    out = mod._apply_fdr_firing_override(
        adv, is_confident=True, fdr_q=0.10, layer_1_declared_safe=True
    )
    assert out["severity"] == "high"
    assert out["remediation"] == "drop"


def test_fdr_override_confident_not_declared_safe_still_drops():
    """Scope guard: a NOT-declared-safe feature (post-index / no manifest
    clearance) the FDR set flags confident still fires high/drop. The fix is
    scoped to pre-index manifest-cleared features; real leaks are unaffected."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(
        severity="info", remediation="keep", delta_auc=0.40, delta_auc_below_floor=False
    )
    out = mod._apply_fdr_firing_override(
        adv, is_confident=True, fdr_q=0.10, layer_1_declared_safe=False
    )
    assert out["severity"] == "high"
    assert out["remediation"] == "drop"


def test_fdr_override_declared_safe_param_defaults_false():
    """Backward compatibility: existing callers omit ``layer_1_declared_safe``;
    the default (False) preserves the legacy unconditional high/drop promotion."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(
        severity="info", remediation="keep", delta_auc=0.40, delta_auc_below_floor=False
    )
    out = mod._apply_fdr_firing_override(adv, is_confident=True, fdr_q=0.10)
    assert out["severity"] == "high"
    assert out["remediation"] == "drop"


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


def test_fdr_confident_features_errored_feature_does_not_shrink_bh_family():
    """codex iter-0 HIGH: a scoring EXCEPTION must remain in the BH family as a
    non-rejected NaN — NOT be dropped. Dropping it shrinks m and LOOSENS the BH
    threshold q/m, which can falsely promote a borderline feature to confident.

    Family {leak, borderline, errored}, q=0.10:
      * keep errored as NaN (m=3): borderline p=0.08 vs rank-2 threshold
        2*0.10/3≈0.067 → NOT rejected → confident set = {leak}.
      * drop errored (m=2): borderline p=0.08 vs rank-2 threshold 0.10 → rejected
        → confident set = {leak, borderline} (the bug).
    """
    import importlib

    mod = importlib.import_module(MOD)
    leak = {"p_value": 0.005, "actual_auc": 0.99, "null_mean": 0.50}  # effect 0.49 > floor
    borderline = {"p_value": 0.08, "actual_auc": 0.99, "null_mean": 0.50}  # effect 0.49 > floor
    scores: dict[str, Any] = {
        "leak": leak,
        "borderline": borderline,
        "errored": RuntimeError("scoring blew up"),
    }
    confident = mod._fdr_confident_features(
        ["leak", "borderline", "errored"],
        scores,
        q=0.10,
        n_permutations=1000,
        effect_floor=0.1,
    )
    assert confident == {"leak"}


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


# ---------------------------------------------------------------------------
# #604: declared-safe FULL immunity (scoped to manifest-trusted synthetic
# fixtures). The σ-band-high case the existing carve-out abdicates on — a
# DESIGNED-legit synthetic predictor whose 1.5x-inflated σ-band still reaches
# high — must be routed to review (NOT auto-dropped) when the run grants full
# immunity, because the synthetic manifest is leak-free BY CONSTRUCTION. Real
# runs (immunity off) keep the "overwhelming evidence still drops" behavior.
# ---------------------------------------------------------------------------


def test_fdr_override_declared_safe_full_immunity_routes_sigma_high_to_review():
    """With full immunity, a declared-safe + FDR-confident feature whose σ-band
    ALREADY reached high is routed to review (moderate/ambiguous), NOT dropped —
    the synthetic manifest's temporal precedence overrides statistical strength."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(
        severity="high",
        severity_pre_joint_check="high",
        remediation="drop",
        z_score=15.0,
        actual_auc=0.95,
        delta_auc=0.45,
        delta_auc_below_floor=False,
    )
    out = mod._apply_fdr_firing_override(
        adv,
        is_confident=True,
        fdr_q=0.10,
        layer_1_declared_safe=True,
        declared_safe_full_immunity=True,
    )
    assert out["severity"] == "moderate"
    assert out["remediation"] == "ambiguous"
    assert out["fdr_confident"] is True
    assert "immunity" in out["evidence"].lower()


def test_fdr_override_full_immunity_defaults_off_preserves_sigma_high_drop():
    """Backward compatibility / real-cohort safety: WITHOUT the immunity flag, a
    declared-safe feature whose σ-band reached high STILL drops (the 61f500ca
    'overwhelming evidence' defensive contract is preserved by default)."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(
        severity="high",
        severity_pre_joint_check="high",
        remediation="drop",
        z_score=15.0,
        actual_auc=0.95,
        delta_auc=0.45,
        delta_auc_below_floor=False,
    )
    out = mod._apply_fdr_firing_override(
        adv, is_confident=True, fdr_q=0.10, layer_1_declared_safe=True
    )
    assert out["severity"] == "high"
    assert out["remediation"] == "drop"


def test_fdr_override_full_immunity_not_declared_safe_still_drops():
    """Scope guard: full immunity applies ONLY to declared-safe features. A
    NOT-declared-safe feature (genuine leak / no manifest clearance) still fires
    high/drop even when the run grants immunity — detection is not blinded."""
    import importlib

    mod = importlib.import_module(MOD)
    adv = _adv_input(
        severity="high",
        severity_pre_joint_check="high",
        remediation="drop",
        z_score=15.0,
        actual_auc=0.95,
        delta_auc=0.45,
        delta_auc_below_floor=False,
    )
    out = mod._apply_fdr_firing_override(
        adv,
        is_confident=True,
        fdr_q=0.10,
        layer_1_declared_safe=False,
        declared_safe_full_immunity=True,
    )
    assert out["severity"] == "high"
    assert out["remediation"] == "drop"


def _legit_and_leak_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A strong DESIGNED-legit predictor (σ-band high, manifest-declarable) plus a
    genuine undeclared leak and noise. ``days_on_therapy`` is a registered
    synthetic-manifest column; ``leak_x`` is a near-perfect outcome proxy that is
    NOT in any manifest."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            # strong legit predictor: ~3σ class separation → σ-band high, BH-confident
            "days_on_therapy": y * 3.0 + rng.normal(0, 1.0, n),
            # near-perfect genuine leak (post-outcome proxy), NOT manifest-declared
            "leak_x": y.astype(float) + rng.normal(0, 0.01, n),
            "noise_a": rng.standard_normal(n),
            "y": y,
        }
    )


def test_node_synthetic_immunity_keeps_strong_legit_drops_genuine_leak():
    """#604 core assertion (faithful node-level): under FDR-on + the synthetic
    manifest + full immunity, a σ-band-high DESIGNED-legit predictor is BH-confident
    yet survives (routed to review), while a genuine undeclared leak still drops."""
    state = _make_state(_legit_and_leak_df(), "y")
    state["scope_spec"]["feature_manifest_source"] = "synthetic"
    state["adaptive_fdr_enabled"] = True
    state["adaptive_declared_safe_full_immunity"] = True
    result = _run(state)
    fdr = result.get("leakage_fdr")
    assert fdr is not None and fdr["active"] is True
    flagged = set(result["adaptive_flagged_features"])
    # the legit strong predictor IS FDR-confident (immunity does not change confidence)
    assert "days_on_therapy" in fdr["confident_features"]
    # but immunity routes it to review → it does NOT enter the dropped set
    assert "days_on_therapy" not in flagged
    # the genuine undeclared leak still drops (detection not blinded)
    assert "leak_x" in flagged


def test_node_synthetic_without_immunity_strong_legit_still_drops():
    """FIX 2 (2026-06-03) supersedes the #604 per-verdict immunity flag for the
    synthetic manifest case: the new post-aggregation manifest immunity exempts
    manifest-declared pre-index features from leakage unconditionally, regardless of
    whether ``adaptive_declared_safe_full_immunity`` is set. ``days_on_therapy`` is
    declared pre-index in the synthetic manifest (knowable_at=index_date), so FIX 2
    strips it from ``adaptive_flagged_features`` even without the old flag.
    The #604 flag is now redundant for this scenario (though harmless if set).
    The genuine post-index leak (leak_x) is still correctly flagged and dropped.
    """
    state = _make_state(_legit_and_leak_df(), "y")
    state["scope_spec"]["feature_manifest_source"] = "synthetic"
    state["adaptive_fdr_enabled"] = True
    # no adaptive_declared_safe_full_immunity flag -> FIX 2 post-aggregation
    # immunity still protects days_on_therapy (supersedes old per-verdict flag).
    result = _run(state)
    flagged = set(result["adaptive_flagged_features"])
    # FIX 2: declared-safe pre-index features are exempt from leakage even
    # without adaptive_declared_safe_full_immunity=True.
    assert "days_on_therapy" not in flagged, (
        "FIX 2 post-aggregation immunity must protect manifest-declared pre-index "
        "features (days_on_therapy, knowable_at=index_date in synthetic manifest) "
        "from leakage regardless of the adaptive_declared_safe_full_immunity flag"
    )
    # Genuine post-index leak is still correctly flagged.
    assert "leak_x" in flagged
