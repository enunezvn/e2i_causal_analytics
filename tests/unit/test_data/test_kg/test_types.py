"""Tests for src/data/kg/types.py — focused on the new
LLMEvaluatorAudit dataclass and LLMVerdict.evaluator_audit field
(Plan .claude/plans/layer4_evaluator_audit_signal.md)."""

from __future__ import annotations

import dataclasses

import pytest


def test_llm_evaluator_audit_construction_and_frozen():
    from src.data.kg.types import LLMEvaluatorAudit

    audit = LLMEvaluatorAudit(
        satisfied=True,
        rationale_complete=True,
        missed_considerations=("temporal_filter",),
        notes="rationale references prefix-censoring",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    assert audit.satisfied is True
    assert audit.missed_considerations == ("temporal_filter",)
    # Frozen invariant
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        audit.satisfied = False  # type: ignore[misc]


def test_llm_verdict_evaluator_audit_default_none():
    from src.data.kg.types import LLMVerdict

    v = LLMVerdict(
        causal_role="confounder",
        mechanism="m",
        recommended_remediation="keep_with_caveat",
    )
    assert v.evaluator_audit is None


def test_llm_verdict_with_evaluator_audit():
    from src.data.kg.types import LLMEvaluatorAudit, LLMVerdict

    audit = LLMEvaluatorAudit(
        satisfied=False,
        rationale_complete=False,
        missed_considerations=("pearl_arrows", "remediation_mapping"),
        notes="weak rationale",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    v = LLMVerdict(
        causal_role="confounder",
        mechanism="m",
        recommended_remediation="keep_with_caveat",
        evaluator_audit=audit,
    )
    assert v.evaluator_audit is audit


# ---------------------------------------------------------------------------
# Issue #240 Stage 3 — EnsembleDecidedBy widened + EnsembleVerdict gate field.
# ---------------------------------------------------------------------------


def test_ensemble_verdict_accepts_evaluator_gate_decided_by():
    """The Stage-3 schema prerequisite: ``decided_by="evaluator_gate"`` must
    construct without error (the typed dataclass would otherwise reject it at
    construction). Also pins ``gate_rule_fired`` default + set."""
    from src.data.kg.types import EnsembleVerdict

    # Default gate_rule_fired is None (pre-Stage-3 / gate disabled).
    v0 = EnsembleVerdict(
        feature_name="feat_x",
        severity="moderate",
        remediation="review",
        decided_by="adversarial",
        confidence=0.6,
    )
    assert v0.gate_rule_fired is None

    # Gate-flipped verdict: decided_by="evaluator_gate" + gate_rule_fired="R1".
    v1 = EnsembleVerdict(
        feature_name="feat_x",
        severity="high",
        remediation="drop",
        decided_by="evaluator_gate",
        confidence=0.6,
        gate_rule_fired="R1",
    )
    assert v1.decided_by == "evaluator_gate"
    assert v1.gate_rule_fired == "R1"


def test_ensemble_decided_by_literal_includes_evaluator_gate():
    """Static-typing contract surfaced at runtime via typing.get_args."""
    import typing

    from src.data.kg.types import EnsembleDecidedBy

    assert "evaluator_gate" in typing.get_args(EnsembleDecidedBy)


def test_ensemble_decided_by_literal_includes_structural():
    """Plan v4 Layer B / Phase 2: the structural decider tags its verdicts
    ``decided_by="structural"`` — the literal must admit it."""
    import typing

    from src.data.kg.types import EnsembleDecidedBy

    assert "structural" in typing.get_args(EnsembleDecidedBy)
