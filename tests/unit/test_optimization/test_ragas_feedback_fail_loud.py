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

import logging
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
    mocked. Implements the judged-path surface the provider checks at
    construction, so these tests do not depend on ambient API keys.
    """

    def __init__(
        self,
        result: Any = None,
        raises: BaseException | None = None,
        blockers: tuple[str, ...] = (),
        dependency_error: BaseException | None = None,
    ):
        self._result = result
        self._raises = raises
        self._blockers = blockers
        self._dependency_error = dependency_error

    @property
    def judged_path_blockers(self) -> tuple[str, ...]:
        return self._blockers

    @property
    def can_judge(self) -> bool:
        return not self._blockers

    def verify_dependencies(self) -> None:
        if self._dependency_error is not None:
            raise self._dependency_error

    async def evaluate_sample(self, sample: Any, run_id: Any = None) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._result


def _provider_with(evaluator: _StubEvaluator) -> RAGASFeedbackProvider:
    """Build a provider whose evaluator is the stub, from construction on.

    Patches the factory rather than swapping the attribute afterwards, so the
    construction-time judged-path checks see the stub too.
    """
    with patch("src.rag.evaluation.get_ragas_evaluator", return_value=evaluator):
        return RAGASFeedbackProvider()


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


class TestConstructionVerifiesJudgedPath:
    """Importability is not runnability (codex iter-1, HIGH).

    ``RAGASEvaluator`` constructs fine with no LLM key — ``__init__`` only sets
    flags, and ``_detect_llm_provider`` merely warns and returns "none".
    ``evaluate_sample`` then routes every sample to the stamped heuristic
    fallback. Without a construction-time check the keyless nightly env would
    start GEPA, hit the per-example refusal on every example, and have dspy
    convert each raise to ``failure_score`` 0.0 — the masquerade again.
    """

    def test_construction_refuses_when_llm_key_missing(self):
        evaluator = _StubEvaluator(blockers=("no LLM API key configured",))

        with pytest.raises(RAGASFeedbackUnavailableError, match="no LLM API key"):
            _provider_with(evaluator)

    def test_construction_refuses_when_ragas_missing(self):
        evaluator = _StubEvaluator(blockers=("ragas package is not importable",))

        with pytest.raises(RAGASFeedbackUnavailableError, match="ragas package"):
            _provider_with(evaluator)

    def test_construction_names_every_failing_precondition(self):
        evaluator = _StubEvaluator(
            blockers=("ragas package is not importable", "no LLM API key configured")
        )

        with pytest.raises(RAGASFeedbackUnavailableError) as exc_info:
            _provider_with(evaluator)

        message = str(exc_info.value)
        assert "ragas package" in message
        assert "no LLM API key" in message

    def test_construction_propagates_dependency_break(self):
        """#491 class: find_spec proves presence, not importability.

        ``_check_ragas`` only calls ``importlib.util.find_spec("ragas")``, so a
        broken import tree leaves ``_ragas_available`` True. The construction
        check must run the real import sequence too.
        """
        evaluator = _StubEvaluator(dependency_error=RagasDependencyError("simulated #491 break"))

        with pytest.raises(RagasDependencyError, match="simulated #491 break"):
            _provider_with(evaluator)

    def test_construction_succeeds_when_judge_is_available(self):
        provider = _provider_with(_StubEvaluator(result=_result()))

        assert provider.enabled is True


@pytest.fixture
def restore_evaluator_singleton():
    """Rebuild the module-global evaluator under the restored env after a test."""
    yield
    from src.rag.evaluation import get_ragas_evaluator

    get_ragas_evaluator(reset=True)


class TestConstructionUsesCurrentEnv:
    """The availability check must read CURRENT env, not a cached singleton.

    ``get_ragas_evaluator`` memoizes a module-global instance whose
    ``_ragas_available`` / ``_llm_configured`` / ``llm_provider`` are frozen at
    first construction. Reusing it would let a stale "judgeable" verdict —
    cached when a key happened to be set — wave through a provider that then
    degrades on every example (codex iter-2, HIGH).
    """

    def test_refuses_when_key_removed_after_singleton_cached(
        self, monkeypatch, restore_evaluator_singleton
    ):
        from src.rag.evaluation import get_ragas_evaluator

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        cached = get_ragas_evaluator(reset=True)
        assert cached.can_judge is True

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # The cached instance still reports the stale verdict — that is the trap.
        assert get_ragas_evaluator().can_judge is True

        with pytest.raises(RAGASFeedbackUnavailableError, match="no LLM API key"):
            RAGASFeedbackProvider()

    def test_constructs_when_key_added_after_singleton_cached(
        self, monkeypatch, restore_evaluator_singleton
    ):
        """The other direction: a stale 'blocked' verdict must not veto a good env."""
        from src.rag.evaluation import get_ragas_evaluator

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cached = get_ragas_evaluator(reset=True)
        assert cached.can_judge is False

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        provider = RAGASFeedbackProvider()
        assert provider.enabled is True


class TestUnjudgedMetricsAreRefused:
    """All-``None`` metrics are 'not measured', not 'measured zero'.

    ``evaluate_sample``'s no-answer early return yields every metric ``None``
    with ``metadata={"error": "No answer provided"}`` and NO
    ``fallback_heuristic`` stamp, so ``or 0.0`` used to manufacture a 0.0 with
    no judge having run (codex iter-2, HIGH). The candidate still ends up at
    dspy's ``failure_score`` 0.0, but the invariant and the log line survive.
    """

    async def test_error_metadata_is_refused(self):
        provider = _provider_with(
            _StubEvaluator(
                result=_result(
                    faithfulness=None,
                    answer_relevancy=None,
                    context_precision=None,
                    context_recall=None,
                    overall_score=None,
                    metadata={"error": "No answer provided"},
                )
            )
        )

        with pytest.raises(RAGASFeedbackUnavailableError, match="No answer provided"):
            await provider.evaluate(question="q", answer="", contexts=["c1"])

    async def test_all_none_metrics_refused_even_without_error_metadata(self):
        provider = _provider_with(
            _StubEvaluator(
                result=_result(
                    faithfulness=None,
                    answer_relevancy=None,
                    context_precision=None,
                    context_recall=None,
                    overall_score=None,
                )
            )
        )

        with pytest.raises(RAGASFeedbackUnavailableError):
            await provider.evaluate(question="q", answer="a", contexts=["c1"])

    async def test_partial_none_metric_is_refused_and_named(self):
        """One unmeasured metric silently dragged the weighted score down."""
        provider = _provider_with(_StubEvaluator(result=_result(context_recall=None)))

        with pytest.raises(RAGASFeedbackUnavailableError, match="context_recall"):
            await provider.evaluate(question="q", answer="a", contexts=["c1"])

    async def test_ragas_nan_reaches_the_provider_as_a_refusal(self):
        """End-to-end through the REAL evaluator with only ragas stubbed.

        ragas emits NaN per metric on metric-level failures. The judged path
        used to coerce that to 0.0 upstream of every refusal here, so the
        provider scored a fabricated 0.0 and never saw a problem.
        """
        from tests.unit.test_rag.test_evaluation_unmeasured_metrics import (
            ALL_JUDGED,
            _components,
        )

        row = {**ALL_JUDGED, "faithfulness": float("nan")}
        with patch("src.rag.evaluation._import_ragas_components", return_value=_components(row)):
            provider = RAGASFeedbackProvider()
            with pytest.raises(RAGASFeedbackUnavailableError, match="faithfulness"):
                await provider.evaluate(
                    question="What caused the TRx drop?",
                    answer="Payer mix shifted.",
                    contexts=["Q4 report: payer mix shifted."],
                )


class TestEvaluatorJudgedPathProperties:
    """The public seam on RAGASEvaluator the provider checks (#1488).

    Lives with the #1488 tests because the properties exist for this consumer;
    see also tests/unit/test_rag/test_evaluation_dependency_guard.py.
    """

    def _evaluator(self, ragas: bool, llm: bool):
        from src.rag.evaluation import RAGASEvaluator

        evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
        evaluator._ragas_available = ragas
        evaluator._llm_configured = llm
        evaluator.llm_provider = "openai" if llm else "none"
        return evaluator

    def test_can_judge_true_only_when_both_preconditions_hold(self):
        assert self._evaluator(ragas=True, llm=True).can_judge is True
        assert self._evaluator(ragas=False, llm=True).can_judge is False
        assert self._evaluator(ragas=True, llm=False).can_judge is False

    def test_blockers_empty_when_judgeable(self):
        assert self._evaluator(ragas=True, llm=True).judged_path_blockers == ()

    def test_blockers_name_the_missing_precondition(self):
        assert any(
            "ragas" in b for b in self._evaluator(ragas=False, llm=True).judged_path_blockers
        )
        assert any("key" in b for b in self._evaluator(ragas=True, llm=False).judged_path_blockers)


class TestFailuresAreLogged:
    """Removing the blanket handlers removed the module's only failure log.

    Under dspy the propagated exception is swallowed into ``failure_score``, so
    without a module-owned log line the operator sees nothing at all
    (codex iter-1, MED). Log and re-raise — never substitute a score.
    """

    async def test_fallback_heuristic_refusal_is_logged(self, caplog):
        provider = _provider_with(
            _StubEvaluator(result=_result(metadata={"evaluation_method": "fallback_heuristic"}))
        )

        with caplog.at_level(
            logging.ERROR, logger="src.optimization.gepa.integration.ragas_feedback"
        ):
            with pytest.raises(RAGASFeedbackUnavailableError):
                await provider.evaluate(question="q", answer="a", contexts=["c1"])

        assert any("fallback_heuristic" in r.getMessage() for r in caplog.records)

    async def test_unexpected_exception_is_logged_and_reraised(self, caplog):
        provider = _provider_with(_StubEvaluator(raises=RuntimeError("judge exploded")))

        with caplog.at_level(
            logging.ERROR, logger="src.optimization.gepa.integration.ragas_feedback"
        ):
            with pytest.raises(RuntimeError, match="judge exploded"):
                await provider.evaluate(question="q", answer="a", contexts=["c1"])

        assert caplog.records, "evaluation failure must leave a module-owned log record"

    async def test_log_record_names_the_cause_and_carries_traceback(self, caplog):
        """The record must be diagnosable on its own.

        DSPy discards the exception, so this log line is all the operator gets:
        it has to name the cause and carry the traceback.
        """
        provider = _provider_with(
            _StubEvaluator(raises=RagasDependencyError("simulated #491 break"))
        )

        with caplog.at_level(
            logging.ERROR, logger="src.optimization.gepa.integration.ragas_feedback"
        ):
            with pytest.raises(RagasDependencyError):
                await provider.evaluate(question="q", answer="a", contexts=["c1"])

        named = [r for r in caplog.records if "simulated #491 break" in r.getMessage()]
        assert named, "log record must name the cause, not just say 'failed'"
        assert named[0].exc_info is not None, "logger.exception must attach the traceback"


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
