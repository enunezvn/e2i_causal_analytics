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
change, plus two integration tests that spy on step_2_data_preparer to prove the
``run_pipeline`` wiring (the partial ``--step 2`` path and the immunity/effective-
manifest coupling) — faithful without running the heavy downstream pipeline.
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


def test_declared_safe_full_immunity_requires_synthetic_manifest() -> None:
    """#604 + codex round-2: full immunity is granted IFF the run is a real-cohort-
    free legacy fixture AND its EFFECTIVE feature manifest is the leak-free-by-
    construction 'synthetic' one — never merely because the regime is a legacy name.

    This closes two holes: (a) an operator `--feature-manifest-source optum/csu`
    override on a legacy regime (no --data-dir) makes the node consult a fallible
    real-cohort manifest → immunity must be withheld; (b) an explicit `--feature-
    manifest-source synthetic` on a REAL `--data-dir` run must still be denied
    immunity (real data, fallible)."""
    # legacy fixture + synthetic manifest → immunity ON
    for regime in ("default", "adverse", "clean"):
        assert _resolve_declared_safe_full_immunity(regime, None, "synthetic") is True
    # legacy fixture + operator override to a real manifest → immunity OFF
    assert _resolve_declared_safe_full_immunity("clean", None, "optum") is False
    assert _resolve_declared_safe_full_immunity("clean", None, "csu") is False
    # real --data-dir run, even with a 'synthetic' override → immunity OFF
    assert _resolve_declared_safe_full_immunity("clean", "/some/real/cohort", "synthetic") is False
    # scenario_* / no manifest → immunity OFF
    assert _resolve_declared_safe_full_immunity("scenario_a", None, None) is False
    assert _resolve_declared_safe_full_immunity("clean", None, None) is False


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


# ── partial-run (--step 2) wiring: the manifest source must reach Step 2 even
#    when the Step-1 block (which normally injects it) is skipped (codex MEDIUM) ──


def test_step2_only_run_wires_synthetic_manifest_into_scope(monkeypatch) -> None:
    """#604 regression: a partial ``--step 2`` legacy-fixture run must still pass
    ``feature_manifest_source='synthetic'`` to step_2_data_preparer. Otherwise FDR
    is ON and immunity is granted but the manifest is absent → layer_1_declared_safe
    stays False → the legit columns are over-dropped (immunity ineffective).

    Spies on step_2_data_preparer to capture the scope_spec it receives, then halts
    the pipeline (gate_passed=False) so no heavier downstream step runs.
    """
    import asyncio

    import scripts.run_tier0_test as rt

    captured: dict = {}

    async def _spy(experiment_id, scope_spec, sample_df, **kwargs):
        captured["scope_spec"] = scope_spec
        captured["adaptive_fdr_enabled"] = kwargs.get("adaptive_fdr_enabled")
        captured["adaptive_declared_safe_full_immunity"] = kwargs.get(
            "adaptive_declared_safe_full_immunity"
        )
        # Halt: a blocked QC gate sets pipeline_halted, skipping all later work.
        return {"gate_passed": False, "qc_report": {"gate_passed": False}}

    monkeypatch.setattr(rt, "step_2_data_preparer", _spy)
    asyncio.run(rt.run_pipeline(step=2, regime="clean", n_total=60, seed=42))

    # The legacy fixture (FDR on + immunity on) MUST also receive the synthetic
    # manifest, or the immunity has nothing to protect.
    assert captured["adaptive_fdr_enabled"] is True
    assert captured["adaptive_declared_safe_full_immunity"] is True
    assert captured["scope_spec"].get("feature_manifest_source") == "synthetic"


def test_step2_override_to_real_manifest_withholds_immunity(monkeypatch) -> None:
    """#604 codex round-2: a legacy regime with an explicit --feature-manifest-source
    override (no --data-dir) must pass the OVERRIDE manifest to the node AND withhold
    full immunity — the override manifest is a fallible real-cohort attestation, not
    the leak-free synthetic one, so the σ!=high "overwhelming evidence" backstop must
    stay in force."""
    import asyncio

    import scripts.run_tier0_test as rt

    captured: dict = {}

    async def _spy(experiment_id, scope_spec, sample_df, **kwargs):
        captured["scope_spec"] = scope_spec
        captured["immunity"] = kwargs.get("adaptive_declared_safe_full_immunity")
        return {"gate_passed": False, "qc_report": {"gate_passed": False}}

    monkeypatch.setattr(rt, "step_2_data_preparer", _spy)
    asyncio.run(
        rt.run_pipeline(
            step=2,
            regime="clean",
            data_dir=None,
            feature_manifest_source="optum",
            n_total=60,
            seed=42,
        )
    )
    # the operator override reaches the node...
    assert captured["scope_spec"].get("feature_manifest_source") == "optum"
    # ...but immunity is withheld because the effective manifest is not 'synthetic'.
    assert captured["immunity"] is False
