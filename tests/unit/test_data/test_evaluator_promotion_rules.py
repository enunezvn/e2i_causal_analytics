"""Stage-1 shadow-mode unit tests for :mod:`src.data.evaluator_promotion_rules`.

Design reference: ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 1
and §4 rules R1/R2/R3. Tests assert each rule's trigger condition exactly,
including the explicit ``None`` (no-fire) behaviour that the shadow column
relies on to remain NULL.
"""

from __future__ import annotations

from src.data.evaluator_promotion_rules import (
    PROMOTION_RULES,
    evaluate_r1,
    evaluate_r2,
    evaluate_r3,
)
from src.data.kg.types import LLMEvaluatorAudit


def _audit(
    *,
    satisfied: bool = True,
    rationale_complete: bool = True,
    missed: tuple[str, ...] = (),
    notes: str = "",
) -> LLMEvaluatorAudit:
    return LLMEvaluatorAudit(
        satisfied=satisfied,
        rationale_complete=rationale_complete,
        missed_considerations=missed,
        notes=notes,
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )


# ---------------------------------------------------------------------------
# R1 — info→moderate escalation when evaluator dissatisfied AND ≥1 missed.
#
# Reframed (2026-05-25) from the original moderate→high after the deep-research
# finding that (ensemble severity == "moderate" AND evaluator_audit present) is
# structurally unreachable in production: the audit only ever rides a valid-role
# worker verdict, whose ensemble severity is high/info — never moderate. The
# reachable, intent-preserving transition is "accept-role (info) verdict whose
# reasoning the evaluator distrusts → escalate disposition info→moderate".
# See docs/plans/240-r1-reachability-investigation.md.
# ---------------------------------------------------------------------------


def test_r1_fires_returns_moderate_when_all_conditions_met():
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    assert evaluate_r1("info", audit) == "moderate"


def test_r1_does_not_fire_when_severity_is_high():
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    assert evaluate_r1("high", audit) is None


def test_r1_does_not_fire_when_severity_is_moderate():
    # The pre-reframe precondition; now a no-fire (moderate has no audit in prod
    # and is no longer the trigger severity).
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    assert evaluate_r1("moderate", audit) is None


def test_r1_does_not_fire_when_severity_is_abstain():
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    assert evaluate_r1("abstain", audit) is None


def test_r1_does_not_fire_when_evaluator_satisfied():
    audit = _audit(satisfied=True, missed=())
    assert evaluate_r1("info", audit) is None


def test_r1_does_not_fire_when_no_missed_considerations():
    audit = _audit(satisfied=False, missed=())
    assert evaluate_r1("info", audit) is None


def test_r1_does_not_fire_when_audit_is_none():
    assert evaluate_r1("info", None) is None


# ---------------------------------------------------------------------------
# R2 — flag for review when ≥2 missed considerations
# ---------------------------------------------------------------------------


def test_r2_fires_when_dissatisfied_with_two_or_more_missed():
    audit = _audit(satisfied=False, missed=("temporal_filter", "pearl_arrows"))
    assert evaluate_r2("moderate", audit) is True


def test_r2_fires_regardless_of_worker_severity():
    audit = _audit(satisfied=False, missed=("temporal_filter", "pearl_arrows"))
    assert evaluate_r2("high", audit) is True
    assert evaluate_r2("info", audit) is True


def test_r2_does_not_fire_when_only_one_missed():
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    assert evaluate_r2("moderate", audit) is None


def test_r2_does_not_fire_when_satisfied():
    audit = _audit(satisfied=True, missed=())
    assert evaluate_r2("moderate", audit) is None


def test_r2_does_not_fire_when_audit_is_none():
    assert evaluate_r2("moderate", None) is None


# ---------------------------------------------------------------------------
# R3 — rationale-incomplete soft flag
# ---------------------------------------------------------------------------


def test_r3_fires_when_rationale_not_complete():
    audit = _audit(rationale_complete=False)
    assert evaluate_r3("moderate", audit) is True


def test_r3_fires_independent_of_satisfied():
    # rationale_complete is the only condition R3 reads
    audit_a = _audit(satisfied=True, rationale_complete=False)
    audit_b = _audit(satisfied=False, rationale_complete=False)
    assert evaluate_r3("moderate", audit_a) is True
    assert evaluate_r3("moderate", audit_b) is True


def test_r3_does_not_fire_when_rationale_complete():
    audit = _audit(rationale_complete=True)
    assert evaluate_r3("moderate", audit) is None


def test_r3_does_not_fire_when_audit_is_none():
    assert evaluate_r3("moderate", None) is None


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


def test_promotion_rules_registry_has_three_entries():
    assert len(PROMOTION_RULES) == 3
    rule_ids = [rid for rid, _fn in PROMOTION_RULES]
    assert rule_ids == ["R1", "R2", "R3"]


def test_promotion_rules_registry_callables_match_named_functions():
    by_id = dict(PROMOTION_RULES)
    assert by_id["R1"] is evaluate_r1
    assert by_id["R2"] is evaluate_r2
    assert by_id["R3"] is evaluate_r3
