"""Shared fixtures for heterogeneous_optimizer unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_llm_narrative(monkeypatch):
    """Unit tests must NEVER make a real LLM call.

    The profile_generator's agentic explanation (the agent's own
    ``CATEInterpretationSignature`` run via DSPy + OpenAI) is disabled by default for
    every test in this package, so the deterministic factual FALLBACK is exercised
    (and no network/OpenAI call is made even when a local ``OPENAI_API_KEY`` is
    present). LLM-path tests re-patch ``generate_cate_interpretation`` explicitly in
    the test body to assert the LLM output is wired through.
    """
    monkeypatch.setattr(
        "src.agents.heterogeneous_optimizer.dspy_integration.generate_cate_interpretation",
        lambda **kwargs: None,
        raising=False,
    )
