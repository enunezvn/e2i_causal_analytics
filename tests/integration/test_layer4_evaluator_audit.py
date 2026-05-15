"""End-to-end integration test for the Layer-4 evaluator audit signal.

Stubs both the DSPy worker module and the Haiku evaluator module so the
test is deterministic and does not require live API keys.

Plan: .claude/plans/layer4_evaluator_audit_signal.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data import causal_role_classifier_loader as loader
from src.data.causal_role_evaluator import CausalRoleEvaluator
from src.data.kg.types import LLMEvaluatorAudit


@pytest.fixture
def enable_evaluator(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(loader, "_lm_is_configured", lambda: True)


def _make_worker_stub(*, role, mechanism, remediation):
    stub = MagicMock()
    stub.return_value = MagicMock(
        causal_role=role,
        mechanism=mechanism,
        recommended_remediation=remediation,
    )
    return stub


def _make_evaluator_stub(*, satisfied, rationale_complete, missed, notes):
    eval_module = MagicMock()
    eval_module.return_value = MagicMock(
        satisfied=satisfied,
        rationale_complete=rationale_complete,
        missed_considerations=missed,
        notes=notes,
    )
    return CausalRoleEvaluator(module=eval_module)


def test_evaluator_agrees_with_strong_worker_rationale(monkeypatch, enable_evaluator):
    worker = _make_worker_stub(
        role="confounder",
        mechanism=(
            "Counts of ondansetron fills in the prefix-censored window "
            "[anchor-180, anchor]. Acts as a confounder fork — common "
            "cause of both prior exposure and outcome severity. "
            "Remediation: keep_with_caveat per role-to-remediation map."
        ),
        remediation="keep_with_caveat",
    )
    evaluator = _make_evaluator_stub(
        satisfied=True,
        rationale_complete=True,
        missed="",
        notes="rationale cites the temporal window and the fork structure",
    )
    monkeypatch.setattr(loader, "_build_evaluator", lambda: evaluator)

    verdict = loader.classify_feature(
        feature_name="ondansetron_fills_180d",
        derivation_pseudocode="count fills in [anchor-180, anchor]",
        dataset_context="CSU target ON_180",
        classifier=worker,
    )
    assert verdict is not None
    audit = verdict.evaluator_audit
    assert isinstance(audit, LLMEvaluatorAudit)
    assert audit.satisfied is True
    assert audit.rationale_complete is True
    assert audit.missed_considerations == ()


def test_evaluator_disagrees_with_thin_worker_rationale(monkeypatch, enable_evaluator):
    worker = _make_worker_stub(
        role="confounder",
        mechanism="seems like a confounder",  # thin rationale
        remediation="keep_with_caveat",
    )
    evaluator = _make_evaluator_stub(
        satisfied=False,
        rationale_complete=False,
        missed="temporal_filter, pearl_arrows, remediation_mapping",
        notes="rationale does not cite temporal window or Pearl arrows",
    )
    monkeypatch.setattr(loader, "_build_evaluator", lambda: evaluator)

    verdict = loader.classify_feature(
        feature_name="ondansetron_fills_180d",
        derivation_pseudocode="count fills in [anchor-180, anchor]",
        dataset_context="CSU target ON_180",
        classifier=worker,
    )
    assert verdict is not None
    audit = verdict.evaluator_audit
    assert audit.satisfied is False
    assert audit.rationale_complete is False
    assert set(audit.missed_considerations) == {
        "temporal_filter",
        "pearl_arrows",
        "remediation_mapping",
    }
    # Critical: even with satisfied=False, the worker's verdict is UNCHANGED.
    assert verdict.causal_role == "confounder"
    assert verdict.recommended_remediation == "keep_with_caveat"


def test_evaluator_failure_preserves_worker_verdict(monkeypatch, enable_evaluator):
    worker = _make_worker_stub(
        role="confounder",
        mechanism="m",
        remediation="keep_with_caveat",
    )
    failing_evaluator = MagicMock()
    failing_evaluator.evaluate.side_effect = RuntimeError("haiku-down")
    monkeypatch.setattr(loader, "_build_evaluator", lambda: failing_evaluator)

    verdict = loader.classify_feature(
        feature_name="f",
        derivation_pseudocode="d",
        dataset_context="c",
        classifier=worker,
    )
    assert verdict is not None
    assert verdict.evaluator_audit is None
    assert verdict.causal_role == "confounder"  # worker verdict intact
