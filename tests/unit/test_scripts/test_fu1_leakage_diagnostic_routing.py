"""FU1 / #528: scenario-regime leakage routing must not block deploy or remediate.

The tier0 runner has two PRE-TRAINING heuristic leakage writers — Step-2's graph
(``adaptive_validity_check`` escalation, which is NOT ``skip_leakage_check``-gated)
and Step-5's structural checks on the real feature matrix — that can false-positive
on clinically-grounded synthetic fixtures (the documented ``journey_status`` trap).
On scenario regimes their output is routed to a diagnostic-only field so it neither
trips the Step-5a LLM remediator (``leakage_severity in {critical,high}``) nor blocks
deployment. The POST-training EMPIRICAL signal (``leakage_suspected`` /
``suspicion_level`` from ``check_imbalance_aware_suspicion``) is the genuine gate and
stays live on EVERY regime — including scenario, where a model that genuinely leaks
MUST still block.

These unit-test the two pure functions that are the single points of behavior change
(``_route_leakage_outputs`` + ``_deploy_blocked_by_leakage``), asserting the exact
state fields the inline runner gates read (``run_tier0_test.py`` Step-5a ``:5628`` and
the deploy gate ``:6538``) — faithful without running the full pipeline.
"""

from __future__ import annotations

from scripts.run_tier0_test import _deploy_blocked_by_leakage, _route_leakage_outputs


def test_p2_scenario_step5_does_not_block_or_remediate():
    """Step-5 structural CRITICAL finding on a scenario regime → diagnostic-only."""
    state: dict = {}
    _route_leakage_outputs(
        state,
        severity="critical",
        leaked=["journey_status"],
        findings=[{"check_name": "logical_dependency", "severity": "critical"}],
        source="step5_structural",
        is_scenario_regime=True,
    )
    # The field the Step-5a remediation gate AND the deploy gate read stays "none".
    assert state.get("leakage_severity", "none") == "none"
    assert state.get("leakage_severity", "none") not in ("critical", "high")  # no remediation
    assert not _deploy_blocked_by_leakage(state)  # no deploy block
    # ...but the finding is recorded for transparency.
    diag = state["leakage_diagnostics"]["step5_structural"]
    assert diag["severity"] == "critical"
    assert "journey_status" in diag["leaked_features"]


def test_p1_scenario_adaptive_escalation_does_not_block_or_remediate():
    """Step-2 graph (adaptive_validity_check) HIGH escalation on scenario → diagnostic-only."""
    state: dict = {}
    _route_leakage_outputs(
        state,
        severity="high",
        leaked=["esr1_status"],
        findings=[{"check_name": "adaptive_discriminator", "severity": "high"}],
        source="step2_graph",
        is_scenario_regime=True,
    )
    assert state.get("leakage_severity", "none") == "none"
    assert not _deploy_blocked_by_leakage(state)
    assert state["leakage_diagnostics"]["step2_graph"]["severity"] == "high"


def test_p3_empirical_signal_still_blocks_on_scenario_regime():
    """The post-training EMPIRICAL signal stays live and still blocks on a scenario regime.

    Routing governs only the pre-training severity/leaked/findings; it must not touch
    ``leakage_suspected``/``suspicion_level``, so a genuinely-leaking trained model
    still blocks deploy even on a scenario regime.
    """
    state: dict = {}
    _route_leakage_outputs(
        state,
        severity="critical",
        leaked=["x"],
        findings=[{}],
        source="step5_structural",
        is_scenario_regime=True,
    )
    assert not _deploy_blocked_by_leakage(state)  # pre-training heuristic alone: no block
    # P3 fires (post-training empirical) → MUST block.
    state["leakage_suspected"] = True
    assert _deploy_blocked_by_leakage(state)
    # And routing never writes/clears the P3 fields itself.
    state2: dict = {"leakage_suspected": True}
    _route_leakage_outputs(
        state2,
        severity="critical",
        leaked=["x"],
        findings=[{}],
        source="step5_structural",
        is_scenario_regime=True,
    )
    assert state2["leakage_suspected"] is True
    # suspicion_level path also still blocks.
    assert _deploy_blocked_by_leakage({"suspicion_level": "high"})


def test_rwd_regime_still_blocks_and_remediates():
    """Non-scenario (RWD) regime: a pre-training CRITICAL finding writes live → blocks + remediates."""
    state: dict = {}
    _route_leakage_outputs(
        state,
        severity="critical",
        leaked=["leaked_col"],
        findings=[{"check_name": "perfect_separation", "severity": "critical"}],
        source="step5_structural",
        is_scenario_regime=False,
    )
    assert state["leakage_severity"] == "critical"
    assert state["leakage_severity"] in ("critical", "high")  # Step-5a remediation gate fires
    assert _deploy_blocked_by_leakage(state)  # deploy gate blocks
    assert "leakage_diagnostics" not in state  # NOT diverted to diagnostics on RWD


def test_rwd_clean_run_writes_live_none_and_no_diagnostics():
    """Sanity: RWD with no findings writes live 'none' and never creates the diagnostic field."""
    state: dict = {}
    _route_leakage_outputs(
        state,
        severity="none",
        leaked=[],
        findings=[],
        source="step2_graph",
        is_scenario_regime=False,
    )
    assert state.get("leakage_severity") == "none"
    assert not _deploy_blocked_by_leakage(state)
    assert "leakage_diagnostics" not in state


# ── #594: legacy synthetic regimes share the scenario-regime fixture contract ──


def test_594_legacy_synthetic_regimes_are_fixture_regimes():
    """#594: the legacy ``ml_patients()`` regimes (default/adverse/clean) are
    clinically-grounded synthetic fixtures with no real leakage by construction —
    the same designed-signal property the FU1/#528 scenario regimes have.

    The Layer-3 FDR confident-set firing driver (#538, default-on) false-positively
    escalates ``leakage_severity=high`` and auto-drops legitimately-predictive
    features (e.g. ``days_on_therapy``, ``prior_treatments``) on these fixtures,
    degrading the clean-regime val_AUC below band and (post #556 fail-closed Feast)
    halting at ``qc_gate_blocked`` → empty ``validation_metrics``.

    The tier0 runner therefore classifies these as fixture regimes and disables the
    FDR firing for them (``adaptive_fdr_enabled=False`` → static σ-band fallback,
    which still catches genuine leaks like ``journey_status`` WITHOUT over-dropping).
    ``rwd_realistic`` and real ``--data-source`` runs must NOT be classified as
    fixtures — the FDR driver stays ON there (validated on the Optum cohort).

    This unit-tests the single point of the classification decision
    (``_is_synthetic_fixture_regime``); the end-to-end FDR-disable is exercised by
    the synthetic-regime e2e tests in the slow-tests lane.
    """
    from scripts.run_tier0_test import (  # noqa: PLC0415
        _LEGACY_REGIMES,
        _SCENARIO_REGIME_TO_NAME,
        _is_synthetic_fixture_regime,
    )

    # Legacy synthetic regimes: fixtures → FDR firing disabled.
    for regime in _LEGACY_REGIMES:
        assert _is_synthetic_fixture_regime(regime), (
            f"legacy synthetic regime {regime!r} must be a fixture regime so the FDR "
            "firing driver is disabled (no real leakage by construction)"
        )

    # Scenario regimes stay fixtures (unchanged FU1/#528 behavior).
    for regime in _SCENARIO_REGIME_TO_NAME:
        assert _is_synthetic_fixture_regime(regime)

    # Real / RWD regimes are NOT fixtures — the FDR driver stays ON.
    assert not _is_synthetic_fixture_regime("rwd_realistic")
    assert not _is_synthetic_fixture_regime("optum_csu_real")


def test_594_fdr_stays_on_for_real_data_runs_even_with_fixture_regime_name():
    """#594 production-safety guard: FDR firing must be disabled ONLY for
    SYNTHETIC fixture GENERATION (no real --data-dir).

    ``--regime`` defaults to ``"default"`` (a legacy fixture regime), but when
    ``--data-dir`` is supplied run_pipeline loads a REAL cohort and IGNORES the
    regime for data generation. So a real ``--data-dir`` run with the default
    regime must NOT silently disable the FDR leakage detector — the fixture
    classification has to be conjoined with "no real data supplied".
    """
    from scripts.run_tier0_test import _resolve_adaptive_fdr_enabled  # noqa: PLC0415

    # Synthetic fixture GENERATION (no data_dir) → FDR OFF (σ-band fallback).
    assert _resolve_adaptive_fdr_enabled("clean", None) is False
    assert _resolve_adaptive_fdr_enabled("scenario_b", None) is False

    # Real cohort run (data_dir supplied) → FDR ON regardless of the (ignored)
    # regime name — the regression Codex flagged on `--data-dir` + default regime.
    assert _resolve_adaptive_fdr_enabled("default", "/some/real/cohort") is True
    assert _resolve_adaptive_fdr_enabled("clean", "/some/real/cohort") is True

    # Non-fixture synthetic regime → FDR ON.
    assert _resolve_adaptive_fdr_enabled("rwd_realistic", None) is True


def test_594_every_valid_regime_is_classified_as_fixture():
    """Predicate-completeness guard (gap G5): EVERY regime in the argparse
    source-of-truth ``_VALID_REGIMES`` must be classified as a synthetic fixture
    by ``_is_synthetic_fixture_regime`` — otherwise it silently keeps the #538
    FDR firing driver ON and re-fires the #594 over-drop (val_AUC below band).

    This is NON-tautological (unlike the loops in
    ``test_594_legacy_synthetic_regimes_are_fixture_regimes``, which iterate the
    classifier's own source sets): it cross-checks the SEPARATE
    ``_VALID_REGIMES`` tuple (consumed by argparse) against
    ``_LEGACY_REGIMES`` + ``_SCENARIO_REGIME_TO_NAME``. Adding a new regime to
    ``_VALID_REGIMES`` without registering it in one of those sets trips this
    test — closing the maintainer footgun the code comment at ~line 4279 warns
    about.
    """
    from scripts.run_tier0_test import (  # noqa: PLC0415
        _VALID_REGIMES,
        _is_synthetic_fixture_regime,
    )

    for regime in _VALID_REGIMES:
        assert _is_synthetic_fixture_regime(regime), (
            f"regime {regime!r} is in _VALID_REGIMES but is NOT classified a "
            "synthetic fixture — the #538 FDR firing driver would stay ON and "
            "re-fire the #594 over-drop. Register it in _LEGACY_REGIMES or "
            "_SCENARIO_REGIME_TO_NAME (or exclude it from _VALID_REGIMES if it "
            "is a real-data regime)."
        )

    # Teeth: an unregistered fixture-looking name is NOT classified — proving a
    # new _VALID_REGIMES entry missing from the source sets WOULD trip the loop.
    assert not _is_synthetic_fixture_regime("clean2_unregistered")
    assert not _is_synthetic_fixture_regime("scenario_zzz")
