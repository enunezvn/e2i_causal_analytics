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


# ---------------------------------------------------------------------------
# Issue #242 — multi-model ensemble types (EnsembleModelVote / EnsembleClassification)
# ---------------------------------------------------------------------------


def test_ensemble_model_vote_construction_and_frozen():
    from src.data.kg.types import EnsembleModelVote

    vote = EnsembleModelVote(
        model="openai/gpt-5",
        causal_role="confounder",
        mechanism="pre-index comorbidity measured before prediction time",
        recommended_remediation="keep_with_caveat",
        latency_ms=812.4,
        input_tokens=410,
        output_tokens=55,
        cost_usd=0.0123,
    )
    assert vote.model == "openai/gpt-5"
    assert vote.causal_role == "confounder"
    assert vote.error is None
    # Frozen invariant
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        vote.causal_role = "mediator"  # type: ignore[misc]


def test_ensemble_model_vote_error_path_defaults():
    """A model that errored is a non-vote: role None, error label set,
    telemetry fields default to None."""
    from src.data.kg.types import EnsembleModelVote

    vote = EnsembleModelVote(model="anthropic/claude-opus-4-7", causal_role=None, error="timeout")
    assert vote.causal_role is None
    assert vote.error == "timeout"
    assert vote.mechanism == ""
    assert vote.recommended_remediation is None
    assert vote.latency_ms is None
    assert vote.cost_usd is None


def test_ensemble_classification_construction_and_frozen():
    from src.data.kg.types import EnsembleClassification, EnsembleModelVote

    votes = (
        EnsembleModelVote(model="anthropic/claude-sonnet-4-6", causal_role="confounder"),
        EnsembleModelVote(model="anthropic/claude-opus-4-7", causal_role="confounder"),
        EnsembleModelVote(model="openai/gpt-5", causal_role="confounder"),
    )
    clf = EnsembleClassification(
        feature_name="comorbidity_count_preindex",
        agreement="full",
        fused_role="confounder",
        fused_mechanism="pre-index comorbidity burden",
        fused_remediation="keep_with_caveat",
        votes=votes,
        healthy_votes=3,
        total_cost_usd=0.03,
        max_latency_ms=900.0,
    )
    assert clf.agreement == "full"
    assert clf.fused_role == "confounder"
    assert len(clf.votes) == 3
    assert clf.healthy_votes == 3
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        clf.agreement = "split"  # type: ignore[misc]


def test_ensemble_agreement_literal_values():
    import typing

    from src.data.kg.types import EnsembleAgreement

    assert set(typing.get_args(EnsembleAgreement)) == {"full", "majority", "split"}
