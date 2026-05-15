"""Tests for the 5 evaluator-audit keys threaded through the
verdict-composition sites in adaptive_validity_check.py
(Plan .claude/plans/layer4_evaluator_audit_signal.md)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    _compose_legacy_verdict,
    _ensemble_to_legacy_dict,
    _legacy_adversarial_alone_verdict,
    _legacy_info_verdict,
    _legacy_short_circuit_verdict,
)
from src.data.kg.types import (
    EnsembleVerdict,
    LLMEvaluatorAudit,
    LLMVerdict,
)


def _make_ensemble_verdict_with_llm(llm: LLMVerdict) -> EnsembleVerdict:
    """Construct a minimal valid EnsembleVerdict with the given LLMVerdict."""
    return EnsembleVerdict(
        feature_name="f",
        severity="moderate",
        remediation="keep_with_caveat",
        decided_by="llm",
        confidence=0.8,
        final_role="confounder",
        evidence=("layer-4 llm",),
        disagreements=(),
        llm_input=llm,
    )


def test_ensemble_to_legacy_dict_threads_evaluator_audit_fields():
    audit = LLMEvaluatorAudit(
        satisfied=True,
        rationale_complete=True,
        missed_considerations=("pearl_arrows",),
        notes="rationale cites prefix-censoring",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    llm = LLMVerdict(
        causal_role="confounder",
        mechanism="m",
        recommended_remediation="keep_with_caveat",
        evaluator_audit=audit,
    )
    verdict = _make_ensemble_verdict_with_llm(llm)
    out = _ensemble_to_legacy_dict(
        verdict,
        adversarial_input={
            "feature": "f",
            "severity_pre_joint_check": "moderate",
            "z_score": 4.2,
            "delta_auc": 0.12,
            "delta_auc_below_floor": False,
            "_hblp_classified": True,
        },
    )
    assert out["evaluator_satisfied"] is True
    assert out["evaluator_rationale_complete"] is True
    assert out["evaluator_missed_considerations"] == ("pearl_arrows",)
    assert out["evaluator_notes"] == "rationale cites prefix-censoring"
    assert out["evaluator_model"] == "anthropic/claude-haiku-4-5-20251001"
    # Existing LLM audit fields unchanged.
    assert out["llm_role"] == "confounder"
    assert out["llm_remediation"] == "keep_with_caveat"


def test_ensemble_to_legacy_dict_evaluator_fields_none_when_audit_absent():
    llm = LLMVerdict(
        causal_role="confounder",
        mechanism="m",
        recommended_remediation="keep_with_caveat",
    )
    verdict = _make_ensemble_verdict_with_llm(llm)
    out = _ensemble_to_legacy_dict(
        verdict,
        adversarial_input={
            "feature": "f",
            "severity_pre_joint_check": "moderate",
            "z_score": 4.2,
            "delta_auc": 0.12,
            "delta_auc_below_floor": False,
            "_hblp_classified": True,
        },
    )
    assert out["evaluator_satisfied"] is None
    assert out["evaluator_rationale_complete"] is None
    assert out["evaluator_missed_considerations"] is None
    assert out["evaluator_notes"] is None
    assert out["evaluator_model"] is None


def test_ensemble_to_legacy_dict_evaluator_fields_none_when_llm_input_absent():
    # When the voter records no LLM input (e.g., Layer 1 / adversarial-only
    # decisions routed through the voter), the 5 new keys must still
    # appear, all None.
    verdict = EnsembleVerdict(
        feature_name="f",
        severity="moderate",
        remediation="keep_with_caveat",
        decided_by="adversarial",
        confidence=0.95,
        final_role="descendant",
        evidence=("z_score=4.2",),
        disagreements=(),
        llm_input=None,
    )
    out = _ensemble_to_legacy_dict(
        verdict,
        adversarial_input={
            "feature": "f",
            "severity_pre_joint_check": "moderate",
            "z_score": 4.2,
            "delta_auc": 0.12,
            "delta_auc_below_floor": False,
            "_hblp_classified": True,
        },
    )
    for key in (
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
    ):
        assert key in out
        assert out[key] is None


def test_legacy_adversarial_alone_verdict_stamps_evaluator_fields_none():
    out = _legacy_adversarial_alone_verdict(
        "f",
        {
            "feature": "f",
            "severity": "info",
            "remediation": "keep",
            "evidence": "z=1.0",
            "severity_pre_joint_check": "info",
            "z_score": 1.0,
            "delta_auc": 0.0,
            "delta_auc_below_floor": True,
            "_hblp_classified": True,
        },
    )
    for key in (
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
    ):
        assert key in out, f"{key} missing from _legacy_adversarial_alone_verdict"
        assert out[key] is None


def test_legacy_info_verdict_stamps_evaluator_fields_none():
    out = _legacy_info_verdict(
        "f",
        adversarial_input={
            "feature": "f",
            "severity": "info",
            "remediation": "keep",
            "evidence": "Adversarial score undefined",
            "_hblp_classified": True,
        },
        evidence="Adversarial score undefined",
    )
    for key in (
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
    ):
        assert key in out, f"{key} missing from _legacy_info_verdict"
        assert out[key] is None


def test_legacy_short_circuit_verdict_stamps_evaluator_fields_none():
    out = _legacy_short_circuit_verdict("f", evidence="too few rows")
    for key in (
        "evaluator_satisfied",
        "evaluator_rationale_complete",
        "evaluator_missed_considerations",
        "evaluator_notes",
        "evaluator_model",
    ):
        assert key in out, f"{key} missing from _legacy_short_circuit_verdict"
        assert out[key] is None


def test_issue_212_cap_preserves_evaluator_audit_fields():
    """Cap predicate (delta_auc_below_floor=True AND decided_by='llm' AND
    not ablation_corroborated) clamps severity/remediation but must NOT
    touch any of the 5 evaluator audit keys."""
    audit = LLMEvaluatorAudit(
        satisfied=False,
        rationale_complete=False,
        missed_considerations=("temporal_filter",),
        notes="thin rationale",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    llm = LLMVerdict(
        causal_role="confounder",
        mechanism="m",
        recommended_remediation="drop",
        evaluator_audit=audit,
    )
    voter = MagicMock()
    voter.vote.return_value = EnsembleVerdict(
        feature_name="f",
        severity="high",  # LLM says leak → pre-cap severity high
        remediation="drop",  # LLM says drop → pre-cap remediation drop
        decided_by="llm",
        confidence=0.7,
        final_role="confounder",
        evidence=("layer-4 llm",),
        disagreements=(),
        llm_input=llm,
    )
    adv_input = {
        "feature": "f",
        "severity": "info",  # joint-clamped → info
        "remediation": "keep",  # joint-clamped → keep
        "severity_pre_joint_check": "moderate",
        "z_score": 5.5,
        "delta_auc": 0.05,
        "delta_auc_floor": 0.10,
        "delta_auc_below_floor": True,  # cap predicate trigger
        "_hblp_classified": True,
        # No ablation corroboration so the cap fires.
        "ablation_severity": None,
    }
    # llm_verdict= must be passed so the adversarial-alone bypass
    # (`:1581-1587`) does NOT fire — only the voter path reaches the
    # issue-#212 cap.
    out = _compose_legacy_verdict(
        "f",
        voter=voter,
        adversarial_input=adv_input,
        llm_verdict=llm,
    )
    # The cap fires: severity/remediation clamped to joint-clamped values.
    assert out["severity"] == "info"
    assert out["remediation"] == "keep"
    # Audit-only keys must survive unchanged.
    assert out["evaluator_satisfied"] is False
    assert out["evaluator_rationale_complete"] is False
    assert out["evaluator_missed_considerations"] == ("temporal_filter",)
    assert out["evaluator_notes"] == "thin rationale"
    assert out["evaluator_model"] == "anthropic/claude-haiku-4-5-20251001"
    # LLM audit fields also preserved by the cap.
    assert out["llm_role"] == "confounder"
    assert out["llm_remediation"] == "drop"
