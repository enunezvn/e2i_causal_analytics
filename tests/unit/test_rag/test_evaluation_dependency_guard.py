"""Regression guard for issue #491.

A broken RAGAS dependency tree must fail **loud**, never silently degrade to
heuristic fallback scores that masquerade as a real RAG-quality regression.

Background
----------
RAGAS pulls ``langchain-community`` transitively. When that drifts past the
version where ``langchain_community.chat_models.vertexai`` exists,
``from ragas import ...`` (or a lazy import during ``evaluate()``) raises
``ModuleNotFoundError``. The previous code caught it in a broad
``except Exception`` and returned heuristic fallback scores — which look like
real (but failing) RAG metrics. That cost 5 days of silently-red CI plus
fake-scoring confusion, because a dependency breakage was indistinguishable
from a genuine quality regression.

These tests pin the contract:

* ``ImportError``/``ModuleNotFoundError`` from the RAGAS import path  →  raise
  ``RagasDependencyError`` (loud, with the original cause preserved).
* A genuine *runtime* failure inside ``evaluate()`` (e.g. a 401 from the LLM)
  →  still degrade to the heuristic fallback (existing graceful-degradation
  contract is preserved; the guard is surgical, not a blanket re-raise).
"""

import builtins
import sys
import types
from unittest.mock import MagicMock

import pytest

from src.rag.evaluation import (
    EvaluationConfig,
    EvaluationSample,
    RagasDependencyError,
    RAGASEvaluator,
    _ensure_ragas_vertexai_compat,
)


def _make_ragas_path_evaluator() -> RAGASEvaluator:
    """Build an evaluator wired to take the real RAGAS path (not the
    'ragas absent' or 'no LLM key' heuristic shortcuts)."""
    evaluator = RAGASEvaluator(
        config=EvaluationConfig(log_to_mlflow=False),
        llm_provider="openai",
        enable_opik_tracing=False,
    )
    evaluator._ragas_available = True
    evaluator._llm_configured = True
    return evaluator


def _sample() -> EvaluationSample:
    return EvaluationSample(
        query="What are the TRx trends for Kisqali?",
        ground_truth="Kisqali TRx grew 15% in Q4.",
        answer="Kisqali TRx grew 15% in Q4, reaching 45,000 units.",
        retrieved_contexts=["Kisqali Q4 TRx report: prescriptions up 15%."],
    )


@pytest.mark.asyncio
async def test_broken_ragas_dependency_raises_loud_not_silent_fallback(monkeypatch):
    """ModuleNotFoundError from the RAGAS import path → RagasDependencyError,
    NOT a silent heuristic-fallback EvaluationResult."""
    evaluator = _make_ragas_path_evaluator()

    # Stub the importable deps so RAGAS is unambiguously the failure point,
    # then make any ``import ragas[...]`` raise the exact #491 breakage.
    monkeypatch.setitem(sys.modules, "openai", types.ModuleType("openai"))
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = type("Dataset", (), {})
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "ragas" or name.startswith("ragas."):
            raise ModuleNotFoundError("No module named 'langchain_community.chat_models.vertexai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.raises(RagasDependencyError) as exc_info:
        await evaluator.evaluate_sample(_sample())

    # Original cause must be preserved so operators see the real breakage.
    assert isinstance(exc_info.value.__cause__, ImportError)
    assert "vertexai" in str(exc_info.value.__cause__)


@pytest.mark.asyncio
async def test_runtime_eval_error_still_falls_back_to_heuristic(monkeypatch):
    """Non-import runtime failures inside RAGAS (e.g. a 401) must STILL degrade
    to the heuristic fallback — the guard only makes *dependency* failures
    loud, it does not break the existing graceful-degradation contract."""
    evaluator = _make_ragas_path_evaluator()

    # A fake RAGAS stack that imports cleanly but whose evaluate() raises a
    # genuine runtime error (not an ImportError).
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda *a, **k: MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = MagicMock()
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_ragas = types.ModuleType("ragas")

    def _evaluate(*a, **k):
        raise RuntimeError("openai.AuthenticationError: 401 Unauthorized")

    fake_ragas.evaluate = _evaluate
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)

    emb = types.ModuleType("ragas.embeddings")
    emb.OpenAIEmbeddings = lambda **k: MagicMock()
    monkeypatch.setitem(sys.modules, "ragas.embeddings", emb)

    llms = types.ModuleType("ragas.llms")
    llms.llm_factory = lambda *a, **k: MagicMock()
    monkeypatch.setitem(sys.modules, "ragas.llms", llms)

    metrics = types.ModuleType("ragas.metrics")
    for _name in ("answer_relevancy", "context_precision", "context_recall", "faithfulness"):
        setattr(metrics, _name, MagicMock())
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics)

    result = await evaluator.evaluate_sample(_sample())

    # Degraded to heuristic fallback — did NOT raise.
    assert result.metadata.get("evaluation_method") == "fallback_heuristic"


@pytest.mark.asyncio
async def test_lazy_import_break_during_evaluate_raises_loud(monkeypatch):
    """A dependency break that surfaces as ImportError DURING evaluate() (a
    lazy import inside ragas, not at initial import) must ALSO fail loud, not
    silently fall back. Guards the runtime-block half of issue #491."""
    evaluator = _make_ragas_path_evaluator()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda *a, **k: MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = MagicMock()
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_ragas = types.ModuleType("ragas")

    def _evaluate(*a, **k):
        # ragas lazily imports a now-removed langchain symbol at call time.
        raise ModuleNotFoundError("No module named 'langchain_community.chat_models.vertexai'")

    fake_ragas.evaluate = _evaluate
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)

    emb = types.ModuleType("ragas.embeddings")
    emb.OpenAIEmbeddings = lambda **k: MagicMock()
    monkeypatch.setitem(sys.modules, "ragas.embeddings", emb)

    llms = types.ModuleType("ragas.llms")
    llms.llm_factory = lambda *a, **k: MagicMock()
    monkeypatch.setitem(sys.modules, "ragas.llms", llms)

    metrics = types.ModuleType("ragas.metrics")
    for _name in ("answer_relevancy", "context_precision", "context_recall", "faithfulness"):
        setattr(metrics, _name, MagicMock())
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics)

    with pytest.raises(RagasDependencyError):
        await evaluator.evaluate_sample(_sample())


def test_vertexai_compat_shim_makes_import_available(monkeypatch):
    """The compat shim registers Vertex stubs so ragas 0.4.x can import even
    though modern langchain-community removed langchain_community.chat_models.
    vertexai and langchain_community.llms.VertexAI (#491).

    Simulated deterministically by faking a langchain-community install that
    is present but LACKS the Vertex integrations — exactly the #491 state.
    """
    lc = types.ModuleType("langchain_community")
    lc.__path__ = []  # mark as a package with no search locations
    chat_models = types.ModuleType("langchain_community.chat_models")
    chat_models.__path__ = []
    llms = types.ModuleType("langchain_community.llms")
    llms.__path__ = []
    monkeypatch.setitem(sys.modules, "langchain_community", lc)
    monkeypatch.setitem(sys.modules, "langchain_community.chat_models", chat_models)
    monkeypatch.setitem(sys.modules, "langchain_community.llms", llms)
    monkeypatch.delitem(sys.modules, "langchain_community.chat_models.vertexai", raising=False)

    # Before the shim, ragas's offending imports are unsatisfiable.
    with pytest.raises(ImportError):
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401

    _ensure_ragas_vertexai_compat()

    # After the shim, the exact imports ragas/llms/base.py performs succeed.
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
    from langchain_community.llms import VertexAI  # noqa: F401

    assert ChatVertexAI is not None
    assert VertexAI is not None
