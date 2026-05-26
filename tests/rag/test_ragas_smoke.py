"""Smoke tests for the RAGAS dependency stack (verifier logic).

Context (#504 / #491): the full gpt-4o RAGAS eval is now manual-only because it
is bound by the CI OpenAI key's throughput (#504). That removed the only
automatic signal that the RAGAS dependency tree still *imports* — the exact
thing that silently broke for 5 days in #491 (ragas 0.4.x importing a Vertex
symbol that modern langchain-community dropped, with the evaluator degrading to
plausible-looking heuristic fallback scores).

These tests cover the verifier *logic* and the #491 loud-failure contract. They
are ragas-independent — they monkeypatch the import boundary
(``_import_ragas_components``) — so they run in the normal backend-tests
environment where the pinned ragas stack is deliberately absent.

The REAL import path (does ``verify_ragas_dependencies`` actually pass on the
pinned ``requirements-ragas.txt`` stack?) is guarded in CI by the ``ragas-smoke``
workflow, which runs ``scripts/run_ragas_eval.py --smoke`` in the minimal ragas
environment. It is not exercised via pytest here: that minimal env lacks pytest
plugins (pytest-asyncio) and the test conftest's deps (fastapi), while this env
lacks ragas — so no single environment can host both. The contract for that
workflow is pinned in ``tests/unit/test_ragas_smoke_workflow.py``.
"""

import pytest

from src.rag.evaluation import (
    EvaluationConfig,
    EvaluationSample,
    RagasDependencyError,
    RAGASEvaluator,
    RagasSmokeResult,
    verify_ragas_dependencies,
)

_EVAL = "src.rag.evaluation"


class _FakeDataset:
    """Stand-in for ``datasets.Dataset`` whose ``from_dict`` always succeeds."""

    @staticmethod
    def from_dict(mapping):  # noqa: ANN001, ANN205
        return mapping


def _ok_components() -> dict:
    """Imported-components dict stub good enough for the dataset-build check."""
    return {"evaluate": lambda **_: None, "Dataset": _FakeDataset}


class TestVerifyRagasDependencies:
    """``verify_ragas_dependencies`` returns a structured pass/fail result."""

    def test_reports_failure_on_broken_import(self, monkeypatch):
        """A broken RAGAS import must make the smoke REPORT failure.

        This is the #491 silent-fallback class: the guard must surface the
        broken dependency tree (ok=False, imports check False) instead of
        passing as if nothing were wrong.
        """

        def _boom() -> dict:
            raise RagasDependencyError("simulated #491 break: vertexai import gone")

        monkeypatch.setattr(f"{_EVAL}._import_ragas_components", _boom)

        result = verify_ragas_dependencies()

        assert isinstance(result, RagasSmokeResult)
        assert result.ok is False
        assert result.checks.get("imports") is False
        assert any(
            "import" in f.lower() or "#491" in f or "ragas" in f.lower() for f in result.failures
        ), result.failures

    def test_reports_failure_on_degenerate_golden_set(self, monkeypatch):
        """A truncated/empty golden set must fail the smoke (catches corruption)."""
        monkeypatch.setattr(f"{_EVAL}._import_ragas_components", _ok_components)
        monkeypatch.setattr(f"{_EVAL}.get_default_evaluation_dataset", lambda: [])

        result = verify_ragas_dependencies(min_samples=30)

        assert result.ok is False
        assert result.checks.get("golden_set") is False
        assert any("golden" in f.lower() or "sample" in f.lower() for f in result.failures), (
            result.failures
        )

    def test_reports_failure_on_golden_sample_missing_fields(self, monkeypatch):
        """Samples missing fields the evaluator needs must fail the smoke."""
        monkeypatch.setattr(f"{_EVAL}._import_ragas_components", _ok_components)
        broken = [
            EvaluationSample(query="q", ground_truth="", answer="a", retrieved_contexts=["c"])
        ] * 30
        monkeypatch.setattr(f"{_EVAL}.get_default_evaluation_dataset", lambda: broken)

        result = verify_ragas_dependencies(min_samples=30)

        assert result.ok is False
        assert result.checks.get("golden_set") is False

    def test_passes_when_imports_ok_and_golden_set_healthy(self, monkeypatch):
        """Imports stubbed OK + the real 30-sample golden set => all checks pass."""
        monkeypatch.setattr(f"{_EVAL}._import_ragas_components", _ok_components)

        result = verify_ragas_dependencies(min_samples=30)

        assert result.checks.get("imports") is True
        assert result.checks.get("golden_set") is True
        assert result.checks.get("dataset_build") is True
        assert result.ok is True, result.failures


class TestLoudFailureContract:
    """The #491 contract: a broken import fails LOUD, never silent fallback."""

    @pytest.mark.asyncio
    async def test_evaluate_with_ragas_raises_loudly_on_broken_import(self, monkeypatch):
        """``_evaluate_with_ragas`` must propagate ``RagasDependencyError``.

        It must NOT be swallowed by the broad ``except Exception`` that falls
        back to heuristic scores — those look like a real (failing) RAG
        regression and masked #491 for 5 days.
        """

        def _boom() -> dict:
            raise RagasDependencyError("broken dependency tree")

        monkeypatch.setattr(f"{_EVAL}._import_ragas_components", _boom)

        evaluator = RAGASEvaluator(config=EvaluationConfig(log_to_mlflow=False))
        sample = EvaluationSample(
            query="q", ground_truth="gt", answer="a", retrieved_contexts=["c"]
        )

        with pytest.raises(RagasDependencyError):
            await evaluator._evaluate_with_ragas(sample, "sid")
