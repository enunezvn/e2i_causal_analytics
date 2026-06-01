"""#604: the tier0 runner re-enables FDR for the legacy synthetic fixtures and
wires the synthetic declared-safe carve-out for them.

The #594 mitigation disabled the FDR firing driver wholesale for ALL synthetic
fixtures (legacy + scenario). #604 replaces that for the LEGACY ml_patients
fixtures (default/adverse/clean) with the real fix: FDR stays ON, the legit
pre-index predictors are protected by the synthetic manifest declared-safe
carve-out granted FULL immunity (manifest correct by construction). The scenario_*
family (synthetic_v2, different columns, not in the must-pass CI lane) keeps the
FDR-disable mitigation. Real ``--data-dir`` runs are unchanged (FDR on, no
immunity, no synthetic manifest injection).

These unit-test the three pure resolvers that are the single points of behavior
change — faithful without running the full pipeline.
"""

from __future__ import annotations

from scripts.run_tier0_test import (
    _resolve_adaptive_fdr_enabled,
    _resolve_declared_safe_full_immunity,
    _resolve_synthetic_manifest_source,
)

# ── FDR re-enabled for legacy fixtures; still off for scenario; on for real ──


def test_fdr_enabled_for_legacy_fixtures() -> None:
    """#604: FDR firing is now ON for the legacy ml_patients fixtures (the over-drop
    set is protected by the manifest carve-out, not by disabling the detector)."""
    for regime in ("default", "adverse", "clean"):
        assert _resolve_adaptive_fdr_enabled(regime, None) is True, (
            f"legacy fixture {regime!r} must keep FDR ON (#604 manifest carve-out)"
        )


def test_fdr_still_disabled_for_scenario_fixtures() -> None:
    """scenario_* (synthetic_v2) is NOT manifest-wired and not in must-pass CI →
    retain the #594 FDR-disable mitigation there."""
    assert _resolve_adaptive_fdr_enabled("scenario_a", None) is False
    assert _resolve_adaptive_fdr_enabled("scenario_b", None) is False


def test_fdr_enabled_for_real_runs() -> None:
    """#594 production-safety guard preserved: a real --data-dir run keeps FDR ON
    regardless of the (ignored) regime name."""
    assert _resolve_adaptive_fdr_enabled("default", "/some/real/cohort") is True
    assert _resolve_adaptive_fdr_enabled("clean", "/some/real/cohort") is True


# ── declared-safe FULL immunity: legacy fixtures only ──


def test_declared_safe_full_immunity_on_for_legacy_fixtures() -> None:
    """Legacy synthetic fixtures get full declared-safe immunity (manifest is
    leak-free by construction) so a σ-band-high legit predictor is routed to
    review instead of auto-dropped."""
    for regime in ("default", "adverse", "clean"):
        assert _resolve_declared_safe_full_immunity(regime, None) is True


def test_declared_safe_full_immunity_off_for_scenario_and_real() -> None:
    """Immunity must NOT apply to scenario_* (FDR off there anyway) nor to real
    cohorts (the real-cohort defensive 'overwhelming evidence still drops'
    behavior is preserved)."""
    assert _resolve_declared_safe_full_immunity("scenario_a", None) is False
    assert _resolve_declared_safe_full_immunity("default", "/some/real/cohort") is False
    assert _resolve_declared_safe_full_immunity("clean", "/some/real/cohort") is False


# ── synthetic manifest source threaded for legacy fixtures only ──


def test_synthetic_manifest_source_for_legacy_fixtures() -> None:
    """A legacy synthetic fixture run with no explicit manifest override resolves
    to the 'synthetic' manifest so lookup_feature_contract clears the legit
    pre-index columns."""
    assert _resolve_synthetic_manifest_source("default", None, None) == "synthetic"
    assert _resolve_synthetic_manifest_source("clean", None, None) == "synthetic"
    assert _resolve_synthetic_manifest_source("adverse", None, None) == "synthetic"


def test_synthetic_manifest_source_respects_override() -> None:
    """An explicit manifest override always wins (never silently overwritten)."""
    assert _resolve_synthetic_manifest_source("default", None, "optum") == "optum"
    assert _resolve_synthetic_manifest_source("clean", None, "csu") == "csu"


def test_synthetic_manifest_source_none_for_scenario_and_real() -> None:
    """No synthetic-manifest injection for scenario_* (FDR off) nor real runs
    (which resolve csu/optum via the RWD path)."""
    assert _resolve_synthetic_manifest_source("scenario_a", None, None) is None
    assert _resolve_synthetic_manifest_source("default", "/some/real/cohort", None) is None
