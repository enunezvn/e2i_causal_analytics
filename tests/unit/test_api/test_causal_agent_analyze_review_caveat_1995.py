"""#1995 -- the agent's ``review_caveat`` leaves state and reaches the API record.

``RefutationNode._review_fields`` (src/agents/causal_impact/nodes/refutation.py)
writes ``review_caveat`` -- the band sentence plus the HITL sentence naming the
approval (reviewer, validity window) or rejection (reviewer, reason), or the
queued / blocked / unavailable variant -- on every REVIEW/BLOCK gate consult and
on a PROCEED-band rejection. Until this change it never left state: the halt
message embeds it verbatim, but the halt is raised only on REJECTED (REVIEW and
PROCEED bands) and on the REVIEW band with the approval switch on -- never on
the BLOCK band, where the statistical gate already withheld the estimate. So a
BLOCK-band run carried no prose naming its adjudication in the API record.

``_agent_state_to_response`` now surfaces it twice: as
``refutation.review_caveat`` (structured) and, when the run was NOT halted, as
a ``warnings`` entry (the drill-down's only prose channel). On a halt run the
halt line already carries the caveat verbatim, so it must appear exactly once.
"""

from __future__ import annotations

import pytest

from src.api.schemas.causal import AgentCausalAnalysisRequest

GATE_BLOCKED = "Refutation gate BLOCKED — the estimate did not survive robustness checks."

APPROVAL_CAVEAT = (
    "Refutation gate is BLOCK (failed robustness, confidence=0.41). This estimate did "
    "not pass and has been routed to expert review for adjudication. The DAG structure "
    "was expert-approved by admin@e2i.local (valid until 2027-03-09); that approval "
    "covers the DAG structure, not this estimate's statistical robustness."
)

REJECTION_CAVEAT = (
    "Refutation gate is BLOCK (failed robustness, confidence=0.41). This estimate did "
    "not pass and has been routed to expert review for adjudication. The DAG structure "
    "was REJECTED by expert review by Dr. No (review rev-rejected): collider. A "
    "rejected structure is not re-queued; revise the DAG (a changed structure gets its "
    "own review) or ask an operator to queue a new review for this hash."
)


def _req() -> AgentCausalAnalysisRequest:
    return AgentCausalAnalysisRequest(treatment_var="treatment_arm", outcome_var="persistent_180d")


def _state(**overrides):
    state = {
        "causal_graph": {
            "nodes": ["treatment_arm", "persistent_180d"],
            "edges": [("treatment_arm", "persistent_180d")],
            "treatment_nodes": ["treatment_arm"],
            "outcome_nodes": ["persistent_180d"],
            "adjustment_sets": [[]],
            "dag_dot": "digraph {}",
        },
        "estimation_result": {
            "ate": 0.12,
            "ate_ci_lower": 0.05,
            "ate_ci_upper": 0.19,
            "statistical_significance": True,
            "method": "CausalForestDML",
        },
        "refutation_results": {"gate_decision": "proceed", "tests_passed": 3, "total_tests": 3},
        "interpretation": {},
        "warnings": [],
    }
    state.update(overrides)
    return state


def _block_state(**overrides):
    return _state(
        refutation_results={"gate_decision": "block", "tests_passed": 1, "total_tests": 3},
        **overrides,
    )


def _response(state):
    from src.api.routes.causal import _agent_state_to_response

    return _agent_state_to_response(
        analysis_id="a-1995",
        request=_req(),
        data_source="database",
        n_rows=100,
        final_state=state,
        latency_ms=10,
    )


@pytest.mark.unit
def test_block_band_approval_caveat_reaches_both_surfaces():
    """BLOCK gate on an expert-approved structure: the approval sentence (reviewer +
    validity window) is in the structured field AND in warnings; the statistical
    verdict is untouched -- status stays failed and the BLOCKED warning stays."""
    resp = _response(
        _block_state(
            expert_review_decision="proceed",
            expert_review_id="rev-approved",
            review_caveat=APPROVAL_CAVEAT,
        )
    )
    assert resp.refutation.review_caveat == APPROVAL_CAVEAT
    assert APPROVAL_CAVEAT in resp.warnings
    assert resp.warnings.count(APPROVAL_CAVEAT) == 1
    assert resp.status == "failed"
    assert GATE_BLOCKED in resp.warnings
    assert resp.refutation.expert_review_decision == "proceed"
    assert resp.refutation.expert_review_id == "rev-approved"


@pytest.mark.unit
def test_block_band_rejection_caveat_reaches_both_surfaces():
    """BLOCK gate on a REJECTED structure. The BLOCK branch never raises the
    expert-review halt (refutation.py execute: halt_reason is computed only on
    the REVIEW and PROCEED branches), so without this change the run's record
    carried no sentence naming the reviewer or the reason."""
    resp = _response(
        _block_state(
            expert_review_decision="rejected",
            expert_review_id="rev-rejected",
            review_caveat=REJECTION_CAVEAT,
        )
    )
    assert resp.refutation.review_caveat == REJECTION_CAVEAT
    assert REJECTION_CAVEAT in resp.warnings
    assert resp.warnings.count(REJECTION_CAVEAT) == 1
    assert resp.status == "failed"
    assert GATE_BLOCKED in resp.warnings
    assert resp.refutation.expert_review_decision == "rejected"


@pytest.mark.unit
def test_halt_run_carries_the_caveat_exactly_once_inside_the_halt_line():
    """A halted run: the halt message embeds the caveat verbatim (refutation.py
    _expert_review_halt_reason), so the route must NOT append it a second time.
    The structured field still carries it on its own."""
    caveat = (
        "Refutation gate is PROCEED (robust, confidence=0.91). The DAG structure was "
        "REJECTED by expert review by Dr. No (review rev-rejected): collider. A rejected "
        "structure is not re-queued; revise the DAG (a changed structure gets its own "
        "review) or ask an operator to queue a new review for this hash."
    )
    halt = (
        "Estimate withheld: a domain expert REJECTED this DAG structure, and an estimate "
        "built on a rejected structure is not a valid causal estimate whatever its "
        "refutation verdict. " + caveat
    )
    resp = _response(
        _state(
            expert_review_halt=True,
            expert_review_decision="rejected",
            expert_review_id="rev-rejected",
            review_caveat=caveat,
            error_message=halt,
            status="failed",
        )
    )
    assert resp.status == "failed"
    assert resp.refutation.review_caveat == caveat
    assert halt in resp.warnings
    # Exactly one warning mentions the caveat text -- the halt line -- and it is
    # not also present as a standalone entry.
    assert sum(1 for w in resp.warnings if caveat in w) == 1
    assert caveat not in resp.warnings
    # Positive control for the counting assertion above: the text IS present.
    assert any(caveat in w for w in resp.warnings)


@pytest.mark.unit
def test_proceed_band_without_a_consult_has_no_caveat_and_unchanged_warnings():
    """Positive control: a PROCEED gate that never consulted the review gate
    (no ``review_caveat`` in state) is byte-for-byte what it was before."""
    resp = _response(_state(warnings=["pre-existing warning"]))
    assert resp.status == "completed"
    assert resp.refutation.review_caveat is None
    assert resp.warnings == ["pre-existing warning"]


@pytest.mark.unit
def test_blank_caveat_is_normalised_to_none_and_not_appended():
    """A whitespace-only caveat (a defensive corner: ``_review_note`` can return
    the empty string, and the band sentence is never blank, but the mapping must
    not depend on that) reads as absent on both surfaces."""
    resp = _response(_block_state(expert_review_decision="pending_review", review_caveat="   "))
    assert resp.refutation.review_caveat is None
    assert resp.warnings == [GATE_BLOCKED]
