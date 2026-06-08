"""Shard 05: optimizer save fix + orchestrator wiring."""

from __future__ import annotations

import inspect
import os

import pytest

from src.agents.feedback_learner import dspy_integration as fdi

dspy = pytest.importorskip("dspy")


def test_gepa_no_broken_await_save():
    """The broken `await save_optimized_module(...)` call must be gone (sync fn)."""
    src = inspect.getsource(fdi.FeedbackLearnerOptimizer._optimize_with_gepa)
    assert "await save_optimized_module" not in src
    assert "optimized_module=" not in src  # the wrong kwarg name


def test_gepa_passes_auto_not_budget_kwarg():
    """create_gepa_optimizer maps budget->auto; passing budget= would TypeError GEPA()."""
    src = inspect.getsource(fdi.FeedbackLearnerOptimizer._optimize_with_gepa)
    assert "budget=budget" not in src
    assert "auto=budget" in src


def test_save_load_roundtrip_offline(tmp_path):
    """A real ChainOfThought saves and loads on dspy 3.1.0 without an LLM call."""
    from src.agents.feedback_learner.dspy_integration import PatternDetectionSignature
    from src.optimization.gepa import load_optimized_module, save_optimized_module

    module = dspy.ChainOfThought(PatternDetectionSignature)
    info = save_optimized_module(
        module,
        agent_name="feedback_learner_pattern",
        output_dir=str(tmp_path),
        metadata={"phase": "pattern", "budget": "light"},
    )
    assert "version_id" in info and "path" in info
    # load_optimized_module calls module_cls() (versioning.py:160), so pass a
    # zero-arg FACTORY that supplies the signature — NOT the bare class.
    loaded, meta = load_optimized_module(
        lambda: dspy.ChainOfThought(PatternDetectionSignature),
        agent_name="feedback_learner_pattern",
        version_id=info["version_id"],
        input_dir=str(tmp_path),
    )
    # Loaded object is a usable dspy module.
    assert hasattr(loaded, "predictors") or hasattr(loaded, "forward")


@pytest.mark.asyncio
async def test_orchestrator_skips_when_insufficient_signals():
    from src.agents.feedback_learner.optimization_runner import (
        run_feedback_learner_optimization,
    )

    class _EmptyClient:
        def table(self, *_):
            return self

        def select(self, *_):
            return self

        def eq(self, *_):
            return self

        def gte(self, *_):
            return self

        def limit(self, *_):
            return self

        def execute(self):
            return type("R", (), {"data": []})()

    result = await run_feedback_learner_optimization(client=_EmptyClient())
    assert result["status"] == "skipped_insufficient_signals"
    assert result["signals_used"] == 0
