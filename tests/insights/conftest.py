"""Test isolation for insight unit tests.

These tests assert the deterministic, grounded FALLBACK path (no live LLM). We force
that path by making ``ensure_dspy_configured`` return False, regardless of a locally
present OPENAI_API_KEY — this mirrors CI (no key) and keeps the suite offline and
deterministic. It is NOT mocking insight values: the fallback computes real grounded
text from the real inputs. The live LLM path is verified manually (plan Task 12),
outside pytest.
"""
import pytest


@pytest.fixture(autouse=True)
def _force_factual_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # run_signature does a call-time `from src.optimization.dspy_lm import
    # ensure_dspy_configured`, so patching the module attribute is picked up.
    monkeypatch.setattr(
        "src.optimization.dspy_lm.ensure_dspy_configured",
        lambda *args, **kwargs: False,
    )
