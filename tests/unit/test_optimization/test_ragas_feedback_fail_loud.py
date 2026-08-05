"""Fail-loud guards for ``RAGASFeedbackProvider`` (issue #1488).

``src/rag/evaluation.py`` went to deliberate trouble in #491 to fail loud rather
than emit heuristic scores: ``RagasDependencyError`` is raised and kept *outside*
the broad ``except`` in ``_evaluate_with_ragas`` because those fallback values
"look like real (failing) RAG metrics and masquerade as a quality regression".

``RAGASFeedbackProvider`` — the wrapper that turns those scores into GEPA
optimization feedback — used to do the opposite on three paths:

1. evaluator import failure → ``_ragas_evaluator = None`` → ``_mock_evaluate``
   heuristics clamped into ``[0.3, 0.9]``, unmarked;
2. any exception (incl. ``RagasDependencyError``) → ``{"score": 0.0}``, which
   GEPA cannot tell apart from a genuinely terrible candidate;
3. the evaluator's *own* stamped heuristic fallback
   (``metadata={"evaluation_method": "fallback_heuristic"}``) consumed with the
   stamp stripped.

These tests pin all three shut. See also
``tests/unit/test_rag/test_evaluation_dependency_guard.py`` for the #491 guards
on the evaluator itself.
"""

import sys
from typing import Any
from unittest.mock import patch

import pytest

from src.optimization.gepa.integration.ragas_feedback import (
    RAGASFeedbackProvider,
    RAGASFeedbackUnavailableError,
    create_ragas_metric,
)
from src.rag.evaluation import EvaluationResult, RagasDependencyError

# DSPy import safety — keep this module on one xdist worker.
pytestmark = pytest.mark.xdist_group(name="gepa_metrics")


def _result(**overrides: Any) -> EvaluationResult:
    """A judged EvaluationResult, as the real evaluator would return it."""
    fields: dict[str, Any] = {
        "sample_id": "s1",
        "query": "What caused the TRx drop?",
        "faithfulness": 0.8,
        "answer_relevancy": 0.8,
        "context_precision": 0.8,
        "context_recall": 0.8,
        "overall_score": 0.8,
        "metadata": {},
    }
    fields.update(overrides)
    return EvaluationResult(**fields)


class _StubEvaluator:
    """Stands in for ``RAGASEvaluator`` at the provider's seam.

    Test-only stub at the evaluator boundary — the production path is never
    mocked. Either returns ``result`` or raises ``raises``.
    """

    def __init__(self, result: Any = None, raises: BaseException | None = None):
        self._result = result
        self._raises = raises

    async def evaluate_sample(self, sample: Any, run_id: Any = None) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._result


def _provider_with(evaluator: _StubEvaluator) -> RAGASFeedbackProvider:
    """Build a provider, then swap in the stub evaluator."""
    provider = RAGASFeedbackProvider()
    provider._ragas_evaluator = evaluator
    return provider


class TestImportFailureFailsLoud:
    """Path 1: a missing evaluator must abort, not silently go heuristic."""

    def test_evaluator_import_failure_raises(self):
        """``__post_init__`` must raise, not log-and-degrade to mock scoring.

        ``sys.modules[name] = None`` makes ``import name`` raise ImportError —
        the cheapest faithful simulation of the broken-dependency case.
        """
        with patch.dict(sys.modules, {"src.rag.evaluation": None}):
            with pytest.raises(RAGASFeedbackUnavailableError) as exc_info:
                RAGASFeedbackProvider()

        # The original cause must survive for diagnosis (#491 discipline).
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ImportError)

    def test_metric_factory_propagates_import_failure(self):
        """``create_ragas_metric`` builds a provider — it must not swallow either.

        This is the seam that actually aborts a GEPA run: it runs *outside*
        DSPy's evaluator, so the raise is not converted to ``failure_score``.
        """
        with patch.dict(sys.modules, {"src.rag.evaluation": None}):
            with pytest.raises(RAGASFeedbackUnavailableError):
                create_ragas_metric(agent_name="cognitive_rag")

    def test_mock_evaluate_is_gone(self):
        """The unmarked heuristic scorer must not exist at all.

        Its only role was standing in before the real wiring landed
        (0e7ae110, 2026-01-24). Keeping it as an import-failure fallback is the
        silent substitution #1488 removes.
        """
        assert not hasattr(RAGASFeedbackProvider, "_mock_evaluate")


class TestDependencyErrorPropagates:
    """Path 2: a broken judge must not read as 'candidate scored 0.0'."""

    async def test_evaluate_propagates_ragas_dependency_error(self):
        """The #491 error must escape ``evaluate()`` intact, not become 0.0."""
        provider = _provider_with(
            _StubEvaluator(raises=RagasDependencyError("simulated #491 break"))
        )

        with pytest.raises(RagasDependencyError):
            await provider.evaluate(question="q", answer="a", contexts=["c1"])

    async def test_metric_propagates_ragas_dependency_error(self):
        """Same at the GEPA-facing boundary produced by ``create_ragas_metric``."""
        provider = _provider_with(
            _StubEvaluator(raises=RagasDependencyError("simulated #491 break"))
        )
        metric = create_ragas_metric(provider=provider, agent_name="cognitive_rag")

        with pytest.raises(RagasDependencyError):
            await metric({"question": "q"}, {"answer": "a", "contexts": ["c1"]})

    async def test_evaluate_propagates_unexpected_exception(self):
        """No blanket ``except Exception`` may launder a failure into a score."""
        provider = _provider_with(_StubEvaluator(raises=RuntimeError("judge exploded")))

        with pytest.raises(RuntimeError, match="judge exploded"):
            await provider.evaluate(question="q", answer="a", contexts=["c1"])

    async def test_metric_propagates_unexpected_exception(self):
        """Same for the metric wrapper's own handler."""
        provider = _provider_with(_StubEvaluator(raises=RuntimeError("judge exploded")))
        metric = create_ragas_metric(provider=provider, agent_name="cognitive_rag")

        with pytest.raises(RuntimeError, match="judge exploded"):
            await metric({"question": "q"}, {"answer": "a", "contexts": ["c1"]})


class TestHeuristicFallbackRefused:
    """Path 3: the evaluator's stamped fallback must not lose its stamp."""

    async def test_fallback_heuristic_result_is_refused(self):
        """Heuristic scores are not optimization signal — refuse them loudly.

        ``RAGASEvaluator._evaluate_with_fallback`` stamps
        ``metadata={"evaluation_method": "fallback_heuristic"}`` precisely so
        downstream consumers can distinguish synthetic from judged scores. The
        provider read only the score fields and dropped the stamp.
        """
        provider = _provider_with(
            _StubEvaluator(result=_result(metadata={"evaluation_method": "fallback_heuristic"}))
        )

        with pytest.raises(RAGASFeedbackUnavailableError, match="fallback_heuristic"):
            await provider.evaluate(question="q", answer="a", contexts=["c1"])


class TestJudgedScoresStillFlow:
    """Regression guard: hardening must not break the real path."""

    async def test_judged_result_produces_score_and_feedback(self):
        provider = _provider_with(_StubEvaluator(result=_result()))

        out = await provider.evaluate(question="q", answer="a", contexts=["c1"])

        assert out["score"] == pytest.approx(0.8)
        assert "RAG Quality Assessment" in str(out["feedback"])

    async def test_genuine_zero_still_scores_zero(self):
        """A real judged 0.0 must still reach GEPA as 0.0.

        After this change a 0.0 from the provider means exactly one thing:
        the judge ran and the candidate scored zero.
        """
        provider = _provider_with(
            _StubEvaluator(
                result=_result(
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                    overall_score=0.0,
                )
            )
        )

        out = await provider.evaluate(question="q", answer="a", contexts=["c1"])

        assert out["score"] == pytest.approx(0.0)
