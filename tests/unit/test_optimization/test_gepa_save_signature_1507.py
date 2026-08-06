"""#1507: the GEPA optimization-run save step must call the real save API.

Four sites invoked ``save_optimized_module`` with kwargs the function does not
have (``optimized_module=``, ``budget=``, ``score=``) and ``await``-ed it
although it is sync. Both failures are measured, not inferred::

    save_optimized_module(agent_name=..., optimized_module=..., budget=..., score=...)
    -> TypeError: unexpected keyword argument 'optimized_module'
    await save_optimized_module(module, agent_name=...)
    -> TypeError: object dict can't be used in 'await' expression

The project already resolved this exact class once — shard 05 of the
dspy-feedback-loop-closure plan rewrote the feedback-learner caller against the
sync function and pinned it in
``tests/unit/test_agents/test_feedback_learner/test_optimization_runner_shard05.py``.
These tests pin the same resolution for the remaining sites, so the fix cannot
regress back to an imagined async wrapper.

The cognitive-RAG site is absent from ``SAVE_CALL_SITES`` on purpose: it lived
inside ``CognitiveRAGOptimizer.optimize_phase``, the legacy GEPA entry point
superseded by the nightly cycle in ``src/tasks/dspy_optimization_tasks.py``
(#1486). That path is removed rather than repaired — pinned separately below.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Every site that saves a GEPA-optimized module, minus the removed legacy path.
SAVE_CALL_SITES = (
    "src/api/routes/chatbot_dspy.py",
    "scripts/gepa_pilot.py",
    "scripts/gepa_phase2_hybrid.py",
)

# Files that must not reference the saver at all after the legacy removal.
REMOVED_SAVE_SITES = ("src/rag/cognitive_rag_dspy.py",)


def _saver_calls(relative_path: str) -> list[ast.Call]:
    """Every ``save_optimized_module(...)`` call in the file, via AST.

    Substring matching is not enough: the phantom kwargs sit on continuation
    lines, so a line-scoped grep for the call reports a clean file while
    ``optimized_module=`` is two lines below it.
    """
    tree = ast.parse((REPO_ROOT / relative_path).read_text())
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "save_optimized_module":
            calls.append(node)
    return calls


def _awaited_saver_calls(relative_path: str) -> list[ast.Call]:
    tree = ast.parse((REPO_ROOT / relative_path).read_text())
    awaited = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "save_optimized_module":
            awaited.append(node.value)
    return awaited


@pytest.mark.parametrize("relative_path", SAVE_CALL_SITES)
def test_save_site_does_not_await_the_sync_saver(relative_path: str) -> None:
    """``save_optimized_module`` returns a dict; awaiting it raises TypeError."""
    awaited = _awaited_saver_calls(relative_path)
    assert not awaited, f"{relative_path}: awaited at line(s) {[c.lineno for c in awaited]}"


@pytest.mark.parametrize("relative_path", SAVE_CALL_SITES)
def test_save_site_uses_real_kwargs(relative_path: str) -> None:
    """The real signature is (module, agent_name, version_id, output_dir, metadata)."""
    calls = _saver_calls(relative_path)
    assert calls, f"{relative_path} no longer calls save_optimized_module"

    accepted = {"module", "agent_name", "version_id", "output_dir", "metadata"}
    for call in calls:
        passed = {kw.arg for kw in call.keywords if kw.arg}
        phantom = passed - accepted
        assert not phantom, f"{relative_path}:{call.lineno} passes unknown kwargs {sorted(phantom)}"


@pytest.mark.parametrize("relative_path", SAVE_CALL_SITES)
def test_save_site_has_no_type_ignore_suppression(relative_path: str) -> None:
    """The suppressions hid the mismatch from mypy; the fix must not need them."""
    lines = (REPO_ROOT / relative_path).read_text().splitlines()
    for call in _saver_calls(relative_path):
        span = lines[call.lineno - 1 : (call.end_lineno or call.lineno)]
        for offset, line in enumerate(span):
            assert "type: ignore" not in line, (
                f"{relative_path}:{call.lineno + offset}: suppression survives: {line.strip()}"
            )


@pytest.mark.parametrize("relative_path", REMOVED_SAVE_SITES)
def test_legacy_path_no_longer_saves(relative_path: str) -> None:
    """optimize_phase's saver call disappears with the path, not fixed in place."""
    source = (REPO_ROOT / relative_path).read_text()
    assert "save_optimized_module" not in source


def test_cognitive_rag_optimizer_drops_the_legacy_entry_point() -> None:
    """#1486 deferral 3: the unwired GEPA/MIPROv2 entry point is removed.

    The RAGAS-driven cycle it scaffolded now runs for real from
    ``src.tasks.dspy_optimization_tasks.run_dspy_rag_optimization``; keeping a
    second, never-triggered entry point that TypeErrors on both its optimizer
    construction and its save was the maintenance hazard #1486 called out.
    """
    pytest.importorskip("dspy")
    from src.rag.cognitive_rag_dspy import CognitiveRAGOptimizer

    for removed in ("optimize_phase", "_optimize_with_gepa", "_optimize_with_miprov2"):
        assert not hasattr(CognitiveRAGOptimizer, removed), f"{removed} still present"

    # The phase quality metrics are the class's surviving purpose.
    for kept in ("summarizer_metric", "investigator_metric", "agent_metric"):
        assert hasattr(CognitiveRAGOptimizer, kept)


def test_gepa_probe_still_binds_fallback_names_when_ragas_is_missing(monkeypatch) -> None:
    """Removing the legacy path must not shrink the module's import surface.

    ``tests/integration/test_gepa_integration.py::test_cognitive_rag_optimizer_gepa_imports``
    imports ``create_ragas_metric`` from this module *unconditionally* and only
    asserts it is non-None when ``GEPA_AVAILABLE`` — i.e. it relies on the name
    being bound to None in the unavailable case. Dropping the ``= None``
    fallbacks turns that import into an ImportError under exactly the partial
    install (GEPA present, ragas missing) the probe exists to detect.
    """
    import importlib
    import sys

    pytest.importorskip("dspy")

    blocked = "src.optimization.gepa.integration.ragas_feedback"
    target = "src.rag.cognitive_rag_dspy"

    class _BlockRagasFeedback:
        def find_spec(self, name, path=None, target=None):
            if name == blocked:
                raise ImportError(f"blocked for test: {name}")
            return None

    monkeypatch.delitem(sys.modules, blocked, raising=False)
    monkeypatch.delitem(sys.modules, target, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockRagasFeedback(), *sys.meta_path])

    module = importlib.import_module(target)

    assert module.GEPA_AVAILABLE is False
    # Bound, not merely absent — `from ... import create_ragas_metric` must work.
    assert module.create_ragas_metric is None
    assert module.create_gepa_optimizer is None


def test_create_optimizer_for_agent_honours_a_caller_budget() -> None:
    """``--budget`` must reach GEPA, so the phase-2 script can reach its save.

    Measured before the fix: ``budget=`` lands in ``**kwargs`` and reaches
    ``GEPA.__init__`` (no such parameter, no ``**kwargs``) -> TypeError; ``auto=``
    collides with the registry default -> "multiple values for keyword argument
    'auto'". Neither spelling worked, so the flag was unusable.
    """
    pytest.importorskip("dspy")
    from src.optimization.gepa.optimizer_setup import AGENT_BUDGETS, create_optimizer_for_agent

    assert AGENT_BUDGETS.get("causal_impact") == "medium", "fixture assumes a non-light default"

    overridden = create_optimizer_for_agent(agent_name="causal_impact", trainset=[], auto="light")
    assert overridden.auto == "light"

    default = create_optimizer_for_agent(agent_name="causal_impact", trainset=[])
    assert default.auto == "medium"


@pytest.mark.asyncio
async def test_chatbot_optimization_run_writes_a_real_artifact(tmp_path, monkeypatch) -> None:
    """The production save step, exercised end-to-end against the real saver.

    Only the two external boundaries are substituted — the Supabase signal read
    and the GEPA optimizer (an LLM-driven compile). Everything from
    ``ChatbotOptimizer.optimize_module``'s save call through
    ``save_optimized_module`` runs for real and must leave a loadable artifact
    on disk carrying the run's budget and score.
    """
    dspy = pytest.importorskip("dspy")
    from src.api.routes import chatbot_dspy as cd

    if not cd.GEPA_AVAILABLE:
        pytest.skip("GEPA extras not installed")

    optimizer = cd.ChatbotOptimizer(optimizer_type="gepa")

    signals = [
        {
            "query": f"what is TRx for brand {i}",
            "predicted_intent": "kpi_query",
            "brand_context": "Fabhalta",
            "intent_confidence": 0.9,
        }
        for i in range(20)
    ]

    async def _training_signals(phase, min_reward=0.5, limit=100):
        return signals

    monkeypatch.setattr(optimizer, "get_training_signals", _training_signals)
    # Skip the LM configuration side effect; dspy 3.1.0 binds the configuring
    # thread permanently and module construction needs no LM.
    monkeypatch.setattr(cd, "_dspy_lm_configured", True)

    compiled = dspy.ChainOfThought(cd.ChatbotIntentClassificationSignature)

    class _StubGEPA:
        best_score = 0.77

        def compile(self, module, trainset=None, valset=None):
            return compiled

    monkeypatch.setattr(cd, "create_gepa_optimizer", lambda **kwargs: _StubGEPA())
    # save_optimized_module's default output_dir is relative, so cwd is the seam
    # that keeps the artifact out of the repo without a test-only parameter.
    monkeypatch.chdir(tmp_path)

    result = await optimizer.optimize_module("intent_classifier", budget="light")

    assert result.get("success") is True, result

    artifacts = sorted(
        (tmp_path / "optimized_modules" / "chatbot_intent_classifier").glob("gepa_*.json")
    )
    assert len(artifacts) == 1, artifacts

    payload = json.loads(artifacts[0].read_text())
    assert payload["agent_name"] == "chatbot_intent_classifier"
    assert payload["metadata"]["budget"] == "light"
    assert payload["metadata"]["score"] == pytest.approx(0.77)
    assert result["version_id"] == payload["version_id"]
