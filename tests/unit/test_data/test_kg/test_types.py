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
