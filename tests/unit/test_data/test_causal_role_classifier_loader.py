"""Tests for src/data/causal_role_classifier_loader.py — focused on the
evaluator wiring added per Plan
.claude/plans/layer4_evaluator_audit_signal.md."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_classify_feature_returns_evaluator_audit_when_enabled(monkeypatch):
    """When the env flag is set and the evaluator returns an audit, the
    returned LLMVerdict carries it."""
    from src.data import causal_role_classifier_loader as loader
    from src.data.kg.types import LLMEvaluatorAudit, LLMVerdict

    monkeypatch.setenv("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    stub_worker = MagicMock()
    stub_worker.return_value = MagicMock(
        causal_role="confounder",
        mechanism="prefix-censored count of fills",
        recommended_remediation="keep_with_caveat",
    )

    canned_audit = LLMEvaluatorAudit(
        satisfied=True,
        rationale_complete=True,
        missed_considerations=(),
        notes="ok",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    stub_eval = MagicMock()
    stub_eval.evaluate.return_value = canned_audit
    monkeypatch.setattr(loader, "_build_evaluator", lambda: stub_eval)
    monkeypatch.setattr(loader, "_lm_is_configured", lambda: True)

    verdict = loader.classify_feature(
        feature_name="ondansetron_fills_180d",
        derivation_pseudocode="count fills in [anchor-180, anchor]",
        dataset_context="CSU target ON_180",
        classifier=stub_worker,
    )
    assert isinstance(verdict, LLMVerdict)
    assert verdict.evaluator_audit is canned_audit


def test_classify_feature_evaluator_audit_none_when_flag_off(monkeypatch):
    from src.data import causal_role_classifier_loader as loader

    monkeypatch.delenv("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", raising=False)
    monkeypatch.setattr(loader, "_lm_is_configured", lambda: True)

    stub_worker = MagicMock()
    stub_worker.return_value = MagicMock(
        causal_role="confounder",
        mechanism="m",
        recommended_remediation="keep_with_caveat",
    )

    verdict = loader.classify_feature(
        feature_name="f",
        derivation_pseudocode="d",
        dataset_context="c",
        classifier=stub_worker,
    )
    assert verdict is not None
    assert verdict.evaluator_audit is None


def test_classify_feature_evaluator_raise_degrades_gracefully(monkeypatch, caplog):
    from src.data import causal_role_classifier_loader as loader

    monkeypatch.setenv("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(loader, "_lm_is_configured", lambda: True)

    stub_worker = MagicMock()
    stub_worker.return_value = MagicMock(
        causal_role="confounder",
        mechanism="m",
        recommended_remediation="keep_with_caveat",
    )
    stub_eval = MagicMock()
    stub_eval.evaluate.side_effect = RuntimeError("haiku-rate-limit")
    monkeypatch.setattr(loader, "_build_evaluator", lambda: stub_eval)

    with caplog.at_level("WARNING"):
        verdict = loader.classify_feature(
            feature_name="f",
            derivation_pseudocode="d",
            dataset_context="c",
            classifier=stub_worker,
        )
    assert verdict is not None
    assert verdict.evaluator_audit is None
    assert any("evaluator" in rec.message.lower() for rec in caplog.records)


def test_classify_feature_evaluator_not_invoked_when_worker_returns_none(monkeypatch):
    from src.data import causal_role_classifier_loader as loader

    monkeypatch.setenv("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(loader, "_lm_is_configured", lambda: True)

    # Worker returns a malformed role → classify_feature returns None
    stub_worker = MagicMock()
    stub_worker.return_value = MagicMock(
        causal_role="not-a-real-role",
        mechanism="m",
        recommended_remediation="keep_with_caveat",
    )
    stub_eval = MagicMock()
    monkeypatch.setattr(loader, "_build_evaluator", lambda: stub_eval)

    verdict = loader.classify_feature(
        feature_name="f",
        derivation_pseudocode="d",
        dataset_context="c",
        classifier=stub_worker,
    )
    assert verdict is None
    stub_eval.evaluate.assert_not_called()
