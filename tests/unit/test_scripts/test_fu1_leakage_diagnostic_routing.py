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
