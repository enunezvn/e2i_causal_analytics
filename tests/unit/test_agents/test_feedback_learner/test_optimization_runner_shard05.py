"""Shard 05: optimizer save fix + orchestrator wiring."""

from __future__ import annotations

import inspect

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


@pytest.mark.asyncio
async def test_orchestrator_discards_run_on_pre_compile_skip(monkeypatch):
    """optimize() returns None ONLY from pre-compile guards (both optimizer
    paths: dspy/GEPA unavailable, <5 examples, unavailable phase — compile
    failures raise). No budget was spent, so the provisional run row must be
    DISCARDED, not persisted as a failed run. codex iter-1 HIGH finding."""
    import src.repositories.prompt_optimization as rec
    from src.agents.feedback_learner.optimization_runner import (
        run_feedback_learner_optimization,
    )

    calls = {"started": 0, "failed": 0, "discarded": []}

    async def fake_started(**kwargs):
        calls["started"] += 1
        return "run-prov-1"

    async def fake_failed(run_id, *args, **kwargs):
        calls["failed"] += 1
        return True

    async def fake_discarded(run_id, client=None):
        calls["discarded"].append(run_id)
        return True

    monkeypatch.setattr(rec, "record_run_started", fake_started)
    monkeypatch.setattr(rec, "record_run_failed", fake_failed)
    monkeypatch.setattr(rec, "record_run_discarded", fake_discarded, raising=False)

    async def fake_signals(client=None, min_reward=0.5):
        # 6 signals pass the runner's MIN_SIGNALS=5 guard, yet GEPA's own
        # trainset guard (int(6 * 0.8) = 4 < 5) skips pre-compile — the
        # common real shape of this bug.
        return [{"source_agent": "feedback_learner", "reward": 0.9}] * 6

    monkeypatch.setattr(
        "src.agents.feedback_learner.signal_store.get_feedback_learner_training_signals",
        fake_signals,
    )
    monkeypatch.setattr("src.optimization.dspy_lm.ensure_dspy_configured", lambda: True)

    class _SkippingOptimizer:
        optimizer_type = "gepa"

        def __init__(self, optimizer_type="gepa"):
            pass

        async def optimize(self, phase, signals, budget="light"):
            return None

    monkeypatch.setattr(fdi, "FeedbackLearnerOptimizer", _SkippingOptimizer)

    result = await run_feedback_learner_optimization(phases=("pattern",))

    assert result["phases"]["pattern"]["status"] == "no_module"
    assert calls["started"] == 1
    assert calls["discarded"] == ["run-prov-1"]
    assert calls["failed"] == 0
