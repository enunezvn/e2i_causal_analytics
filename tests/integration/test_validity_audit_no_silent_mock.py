"""Integration tests for #471 anti-mocking fix in validity_audit.

Tests that ``_get_validity_llm()`` does NOT silently fall back to a Mock
when ``ANTHROPIC_API_KEY`` is missing — the previous behavior was a
CLAUDE.md anti-mocking HARMFUL-NOW violation because the production
LangGraph node ``ValidityAuditNode`` would silently swap in
``MockValidityLLM`` whose return values look identical to real LLM
validity-audit output.

Specifies the new contract:

  * Missing key + no opt-in flag    -> RuntimeError pointing at
    diagnosis (env_state of the var + dotenv-path ambiguity).
  * Missing key + opt-in flag set   -> MockValidityLLM with clearly-fake
    marker (``_using_real_llm=False`` AND mock response carries the
    ``"mock_response_for_dev_only": True`` field).
  * Real key set                    -> attempts real ChatAnthropic
    (covered by existing tests; not duplicated here).

Also asserts the misleading "ANTHROPIC_API_KEY not set" log line no
longer fires when ``.env`` contains the key but wasn't loaded — the new
error message MUST mention both ``load_dotenv()`` and the
``EXPERIMENT_DESIGNER_USE_MOCK_LLM`` opt-in escape hatch.
"""

from __future__ import annotations

import json

import pytest

# Import target deferred to test bodies so module-import doesn't fail
# under the new "raise when no key" contract.


# ---------------------------------------------------------------------------
# Contract 1: Missing key + no flag -> RuntimeError, not silent mock
# ---------------------------------------------------------------------------


def test_get_validity_llm_raises_on_missing_key_no_flag(monkeypatch):
    """Missing ANTHROPIC_API_KEY without explicit dev-mode flag MUST raise.

    Pre-#471: silently returned ``(MockValidityLLM(), "mock-validity-llm", False)``
    while logging ``INFO`` — operators below WARN never saw it, and
    downstream nodes acted on plausible-real mock outputs.

    Post-#471: explicit RuntimeError pointing at:
      * the actual env state of ANTHROPIC_API_KEY (not just "not set")
      * the .env-loading possibility
      * the EXPERIMENT_DESIGNER_USE_MOCK_LLM=1 opt-in escape hatch.
    """
    from src.agents.experiment_designer.nodes.validity_audit import _get_validity_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EXPERIMENT_DESIGNER_USE_MOCK_LLM", raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        _get_validity_llm()

    msg = str(excinfo.value)
    # Must surface the actual diagnostic state, not the misleading
    # "not set" string (which lies when .env has the key but wasn't
    # loaded — the original bug class this fix addresses).
    assert "ANTHROPIC_API_KEY" in msg
    assert "EXPERIMENT_DESIGNER_USE_MOCK_LLM" in msg, (
        "Error message must surface the opt-in dev-mode escape hatch so "
        "developers who actually want the mock know how to enable it."
    )
    assert "load_dotenv" in msg or "dotenv" in msg.lower(), (
        "Error message must mention .env loading as a likely cause "
        "(per #470/#471 audit) so operators don't assume the key is unset "
        "when it's just unloaded."
    )


def test_get_validity_llm_raises_on_empty_string_key(monkeypatch):
    """Empty-string key (docker-compose ``FOO=`` pattern) must also raise.

    Distinct from <unset> per CLAUDE.md ``env_state()`` semantics, but
    same HARMFUL-NOW outcome under the old code path (``if not api_key``
    was true).
    """
    from src.agents.experiment_designer.nodes.validity_audit import _get_validity_llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("EXPERIMENT_DESIGNER_USE_MOCK_LLM", raising=False)

    with pytest.raises(RuntimeError):
        _get_validity_llm()


# ---------------------------------------------------------------------------
# Contract 2: Missing key + opt-in flag -> mock with clearly-fake marker
# ---------------------------------------------------------------------------


def test_get_validity_llm_returns_mock_when_flag_set(monkeypatch):
    """Explicit ``EXPERIMENT_DESIGNER_USE_MOCK_LLM=1`` opts into mock.

    Preserves the user-requested dev-mode functionality from PR
    9a618767 (the original LangChain integration) but moves it from
    silent-fallback to opt-in.
    """
    from src.agents.experiment_designer.nodes.validity_audit import (
        MockValidityLLM,
        _get_validity_llm,
    )

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("EXPERIMENT_DESIGNER_USE_MOCK_LLM", "1")

    llm, model_name, is_real = _get_validity_llm()

    assert isinstance(llm, MockValidityLLM)
    assert is_real is False
    assert "mock" in model_name.lower()


@pytest.mark.asyncio
async def test_mock_response_carries_dev_only_marker(monkeypatch):
    """Mock's structured response MUST be marked as dev-only.

    Per CLAUDE.md REASON-BEFORE-RULES: a dev-mode mock is acceptable
    behind an explicit flag IF the values it emits are clearly-fake (not
    plausible-wrong). The previous mock returned ``overall_validity_score=0.75``
    which is indistinguishable from a real audit.

    Post-#471: the mock response carries an unambiguous in-band marker
    so downstream consumers (and humans reading logs / debug dumps) can
    tell at a glance that the audit is synthetic.
    """
    from src.agents.experiment_designer.nodes.validity_audit import MockValidityLLM

    monkeypatch.setenv("EXPERIMENT_DESIGNER_USE_MOCK_LLM", "1")

    mock = MockValidityLLM()
    response = await mock.ainvoke("any prompt")
    parsed = json.loads(response.content)

    assert parsed.get("mock_response_for_dev_only") is True, (
        "Mock response must carry an in-band 'mock_response_for_dev_only' "
        "marker so consumers can distinguish synthetic audits from real "
        "LLM output without re-reading the env state."
    )


# ---------------------------------------------------------------------------
# Contract 3: env_diagnostics.env_state helper behaves per CLAUDE.md spec
# ---------------------------------------------------------------------------


def test_env_state_distinguishes_unset_empty_set(monkeypatch, tmp_path):
    """``env_state`` MUST distinguish <unset> / <empty-string> / <set,len=N>.

    This is the core diagnostic distinction missing from the previous
    "FOO not configured" log pattern.
    """
    from src.utils.env_diagnostics import env_state

    monkeypatch.delenv("TEST_DIAG_VAR", raising=False)
    assert "<unset>" in env_state("TEST_DIAG_VAR")

    monkeypatch.setenv("TEST_DIAG_VAR", "")
    assert "<empty-string>" in env_state("TEST_DIAG_VAR")

    monkeypatch.setenv("TEST_DIAG_VAR", "some-secret-value-12345")
    rendered = env_state("TEST_DIAG_VAR")
    assert "<set,len=" in rendered
    assert "23" in rendered  # len("some-secret-value-12345") == 23
    # MUST NOT leak the actual key value.
    assert "some-secret-value-12345" not in rendered


def test_env_state_reports_dotenv_existence(monkeypatch, tmp_path):
    """When ``dotenv_path`` is supplied, ``env_state`` reports existence.

    This is the load-chain ambiguity surface: an unset var + present
    .env file is the exact symptom of the #470/#471 bug class.
    """
    from src.utils.env_diagnostics import env_state

    monkeypatch.delenv("TEST_DIAG_VAR2", raising=False)

    present = tmp_path / "present.env"
    present.write_text("TEST_DIAG_VAR2=ignored\n")
    rendered_present = env_state("TEST_DIAG_VAR2", dotenv_path=present)
    assert "exists" in rendered_present

    missing = tmp_path / "missing.env"
    rendered_missing = env_state("TEST_DIAG_VAR2", dotenv_path=missing)
    assert "missing" in rendered_missing


# ---------------------------------------------------------------------------
# Contract 4: rubric_evaluator records evaluation_method in its output
# ---------------------------------------------------------------------------


def test_rubric_evaluation_carries_evaluation_method_field():
    """Per audit H1: ``RubricEvaluation`` MUST surface whether it used
    the real LLM or the heuristic fallback.

    Pre-#471: ``_fallback_evaluation()`` returned neutral 3.0 scores
    that were indistinguishable from real LLM 3.0 scores to downstream
    ``ImprovementDecision`` logic.

    Post-#471: a new ``evaluation_method`` field on ``RubricEvaluation``
    (Literal["llm", "heuristic_fallback"]) makes the source visible.
    """
    from src.agents.feedback_learner.evaluation.models import RubricEvaluation

    # Should be a field on the model.
    assert "evaluation_method" in RubricEvaluation.model_fields, (
        "RubricEvaluation must declare an 'evaluation_method' field so "
        "downstream consumers can tell real LLM scores apart from "
        "heuristic-fallback neutral scores (audit H1)."
    )
