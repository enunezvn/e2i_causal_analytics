"""Unit tests for the Haiku audit evaluator (Layer-4 sidecar)."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_evaluator_signature_field_names_pinned():
    # Field names are part of the audit contract; pin them so a rename
    # surfaces in CI.
    from src.data.causal_role_evaluator import CausalRoleEvaluatorSignature

    inputs = set(CausalRoleEvaluatorSignature.input_fields)
    outputs = set(CausalRoleEvaluatorSignature.output_fields)
    assert inputs == {
        "feature_name",
        "derivation_pseudocode",
        "dataset_context",
        "worker_causal_role",
        "worker_mechanism",
        "worker_recommended_remediation",
        "criteria",
    }
    assert outputs == {
        "satisfied",
        "rationale_complete",
        "missed_considerations",
        "notes",
    }


def test_evaluator_criteria_text_is_load_bearing():
    from src.data.causal_role_evaluator import EVALUATOR_CRITERIA

    # The criteria text encodes the audit contract. If it shrinks below
    # the documented coverage, the audit becomes vacuous.
    assert "temporal" in EVALUATOR_CRITERIA.lower()
    assert "pearl" in EVALUATOR_CRITERIA.lower() or "arrow" in EVALUATOR_CRITERIA.lower()
    assert "remediation" in EVALUATOR_CRITERIA.lower()
    # Must explicitly state this is audit-only, not a gate.
    assert "audit" in EVALUATOR_CRITERIA.lower()


def test_evaluator_returns_llm_evaluator_audit_via_stub():
    from src.data.causal_role_evaluator import CausalRoleEvaluator
    from src.data.kg.types import LLMEvaluatorAudit, LLMVerdict

    stub_module = MagicMock()
    stub_module.return_value = MagicMock(
        satisfied=True,
        rationale_complete=True,
        missed_considerations="temporal_filter, pearl_arrows",
        notes="rationale cites the prefix-censoring window",
    )
    evaluator = CausalRoleEvaluator(module=stub_module)

    audit = evaluator.evaluate(
        feature_name="ondansetron_fills_180d",
        derivation_pseudocode="count fills in [anchor-180, anchor]",
        dataset_context="CSU target ON_180",
        worker_verdict=LLMVerdict(
            causal_role="confounder",
            mechanism="prefix-censored count",
            recommended_remediation="keep_with_caveat",
        ),
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    assert isinstance(audit, LLMEvaluatorAudit)
    assert audit.satisfied is True
    # missed_considerations must arrive as a tuple even if the LM returns
    # a comma-separated string.
    assert audit.missed_considerations == ("temporal_filter", "pearl_arrows")
    assert audit.evaluator_model == "anthropic/claude-haiku-4-5-20251001"


def test_evaluator_truncates_notes_to_500_chars():
    from src.data.causal_role_evaluator import CausalRoleEvaluator
    from src.data.kg.types import LLMVerdict

    stub_module = MagicMock()
    long_notes = "x" * 1000
    stub_module.return_value = MagicMock(
        satisfied=False,
        rationale_complete=False,
        missed_considerations="",
        notes=long_notes,
    )
    evaluator = CausalRoleEvaluator(module=stub_module)
    audit = evaluator.evaluate(
        feature_name="f",
        derivation_pseudocode="d",
        dataset_context="c",
        worker_verdict=LLMVerdict(
            causal_role="confounder",
            mechanism="m",
            recommended_remediation="keep_with_caveat",
        ),
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    assert len(audit.notes) == 500
    assert audit.missed_considerations == ()


def test_evaluator_accepts_list_missed_considerations():
    # Codex final-review MEDIUM: some provider-side structured parsing
    # paths return a list/tuple for a str-annotated field. Accept both
    # shapes so the audit signal is not silently lost on provider drift.
    from src.data.causal_role_evaluator import CausalRoleEvaluator
    from src.data.kg.types import LLMVerdict

    stub_module = MagicMock()
    stub_module.return_value = MagicMock(
        satisfied=False,
        rationale_complete=False,
        missed_considerations=["temporal_filter", "pearl_arrows"],
        notes="list-shaped output",
    )
    evaluator = CausalRoleEvaluator(module=stub_module)
    audit = evaluator.evaluate(
        feature_name="f",
        derivation_pseudocode="d",
        dataset_context="c",
        worker_verdict=LLMVerdict(
            causal_role="confounder",
            mechanism="m",
            recommended_remediation="keep_with_caveat",
        ),
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    assert audit.missed_considerations == ("temporal_filter", "pearl_arrows")


def test_evaluator_coerces_non_bool_satisfied_to_false():
    from src.data.causal_role_evaluator import CausalRoleEvaluator
    from src.data.kg.types import LLMVerdict

    # The evaluator must not crash on a malformed LM output. Coerce
    # missing / non-bool satisfied to False (conservative — "not
    # affirmatively satisfied").
    stub_module = MagicMock()
    stub_module.return_value = MagicMock(
        satisfied="maybe",
        rationale_complete=None,
        missed_considerations=None,
        notes=None,
    )
    evaluator = CausalRoleEvaluator(module=stub_module)
    audit = evaluator.evaluate(
        feature_name="f",
        derivation_pseudocode="d",
        dataset_context="c",
        worker_verdict=LLMVerdict(
            causal_role="confounder",
            mechanism="m",
            recommended_remediation="keep_with_caveat",
        ),
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    assert audit.satisfied is False
    assert audit.rationale_complete is False
    assert audit.missed_considerations == ()
    assert audit.notes == ""


def test_evaluator_disabled_when_env_flag_unset(monkeypatch):
    from src.data.causal_role_evaluator import evaluator_is_enabled

    monkeypatch.delenv("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", raising=False)
    assert evaluator_is_enabled() is False


def test_evaluator_disabled_when_env_flag_zero(monkeypatch):
    from src.data.causal_role_evaluator import evaluator_is_enabled

    monkeypatch.setenv("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", "0")
    assert evaluator_is_enabled() is False


def test_evaluator_enabled_when_env_flag_one(monkeypatch):
    from src.data.causal_role_evaluator import evaluator_is_enabled

    monkeypatch.setenv("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", "1")
    assert evaluator_is_enabled() is True


def test_evaluator_lm_unconfigured_when_anthropic_key_missing(monkeypatch):
    from src.data.causal_role_evaluator import _evaluator_lm_is_configured

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _evaluator_lm_is_configured() is False


def test_evaluator_lm_configured_when_anthropic_key_present(monkeypatch):
    from src.data.causal_role_evaluator import _evaluator_lm_is_configured

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-redacted")
    assert _evaluator_lm_is_configured() is True


def test_evaluator_model_default():
    from src.data.causal_role_evaluator import DEFAULT_EVALUATOR_MODEL

    assert DEFAULT_EVALUATOR_MODEL == "anthropic/claude-haiku-4-5-20251001"
