"""RAGAS-backed GEPA metric for RAG-shaped agents (issue #1486).

Why this metric exists as a *class* in the registry rather than reusing
``create_ragas_metric`` directly: three properties were measured against the
installed dspy 3.1.0, and each one rules the older factory out.

1. ``create_ragas_metric`` returns an ``async`` function. GEPA calls its metric
   synchronously, so the return value would be a coroutine object, never a
   score.
2. A metric returning a plain ``{"score": ..., "feedback": ...}`` dict makes
   ``dspy.Evaluate`` die with ``TypeError: unsupported operand type(s) for +:
   'int' and 'dict'``. Only a ``dspy.Prediction`` survives GEPA's
   ``s["score"] if hasattr(s, "score") else s`` coercion.
3. A metric that *raises* does NOT abort the run: dspy's parallelizer swallows
   it and records ``failure_score`` (0.0) for that example. So "refuse by
   raising" silently becomes "this candidate scored zero" — the exact
   fabrication #1488 set out to prevent, reappearing one layer up. An unmeasured
   RAGAS metric therefore has to be *excluded and reweighted*, never raised past.

Probe output backing all three is quoted in the issue #1486 lane report.
"""

from typing import Any, Optional

import pytest

# DSPy import safety — keep this module on one xdist worker (repo convention).
pytestmark = pytest.mark.xdist_group(name="gepa_metrics")


class _StubEvaluator:
    """Stands in for ``RAGASEvaluator`` at its public seam.

    Test-only stub at the evaluator boundary, mirroring
    ``tests/unit/test_optimization/test_ragas_feedback_fail_loud.py`` (#1488):
    the production path is never mocked, and these tests must not depend on an
    ambient OPENAI_API_KEY.
    """

    def __init__(
        self,
        result: Any = None,
        blockers: tuple[str, ...] = (),
        raises: Optional[BaseException] = None,
    ) -> None:
        self._result = result
        self._blockers = blockers
        self._raises = raises
        self.samples: list[Any] = []

    @property
    def judged_path_blockers(self) -> tuple[str, ...]:
        return self._blockers

    @property
    def can_judge(self) -> bool:
        return not self._blockers

    def verify_dependencies(self) -> None:
        return None

    async def evaluate_sample(self, sample: Any, run_id: Any = None) -> Any:
        self.samples.append(sample)
        if self._raises is not None:
            raise self._raises
        return self._result


def _result(**overrides: Any) -> Any:
    """A judged EvaluationResult, as the real evaluator would return it."""
    from src.rag.evaluation import EvaluationResult

    fields: dict[str, Any] = {
        "sample_id": "s1",
        "query": "What drove the Kisqali TRx drop in the Northeast?",
        "faithfulness": 0.8,
        "answer_relevancy": 0.4,
        "context_precision": 0.6,
        "context_recall": 0.7,
        "overall_score": 0.625,
        "metadata": {},
    }
    fields.update(overrides)
    return EvaluationResult(**fields)


def _metric(monkeypatch: pytest.MonkeyPatch, evaluator: _StubEvaluator) -> Any:
    """Build a RagasGEPAMetric bound to a stub evaluator."""
    import src.rag.evaluation as evaluation_module
    from src.optimization.gepa.metrics.ragas_metric import RagasGEPAMetric

    monkeypatch.setattr(
        evaluation_module, "get_ragas_evaluator", lambda *a, **k: evaluator, raising=True
    )
    return RagasGEPAMetric()


def _example(**kwargs: Any) -> Any:
    import dspy

    return dspy.Example(**kwargs)


def _prediction(**kwargs: Any) -> Any:
    import dspy

    return dspy.Prediction(**kwargs)


class TestReturnShape:
    """GEPA only understands dspy.Prediction(score, feedback)."""

    def test_returns_dspy_prediction_not_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import dspy

        metric = _metric(monkeypatch, _StubEvaluator(result=_result()))
        out = metric(
            _example(user_query="why did TRx drop?", evidence_board=["ctx a", "ctx b"]),
            _prediction(synthesis="TRx fell 12% on payer mix."),
            None,
            None,
            None,
        )

        assert isinstance(out, dspy.Prediction), f"GEPA cannot consume {type(out).__name__}"
        assert not isinstance(out, dict)
        assert isinstance(out.score, float)
        assert isinstance(out.feedback, str)
        assert 0.0 <= out.score <= 1.0

    def test_survives_gepas_score_coercion_and_sum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression guard for the measured `int + dict` crash in dspy.Evaluate.

        Replays the exact coercion dspy/teleprompt/gepa/gepa_utils.py applies to
        every metric return, then the sum that Evaluate performs over them.
        """
        metric = _metric(monkeypatch, _StubEvaluator(result=_result()))
        returns = [
            metric(
                _example(user_query="q", evidence_board=["ctx"]),
                _prediction(synthesis="a grounded answer"),
                None,
                None,
                None,
            )
            for _ in range(3)
        ]

        coerced = [s["score"] if hasattr(s, "score") else s for s in returns]
        assert all(isinstance(c, float) for c in coerced), coerced
        assert isinstance(sum(coerced), float)


class TestRunsInsideRealDspyEvaluate:
    """The contract that matters is GEPA's, so exercise the real evaluator.

    ``dspy.Evaluate`` with ``failure_score``/``max_errors`` is precisely how
    ``dspy/teleprompt/gepa/gepa_utils.py`` scores a candidate.
    """

    def test_metric_scores_a_candidate_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import dspy
        from dspy.utils.dummies import DummyLM

        class Synthesise(dspy.Signature):
            user_query: str = dspy.InputField()
            synthesis: str = dspy.OutputField()

        metric = _metric(monkeypatch, _StubEvaluator(result=_result()))
        dspy.configure(lm=DummyLM([{"synthesis": "TRx fell 12% on payer mix."}] * 6))

        devset = [
            dspy.Example(
                user_query=f"why did TRx drop in region {i}?",
                evidence_board=[f"Q4 report for region {i}"],
            ).with_inputs("user_query")
            for i in range(3)
        ]

        result = dspy.Evaluate(
            devset=devset,
            metric=metric,
            num_threads=1,
            return_all_scores=True,
            failure_score=0.0,
            max_errors=len(devset) * 100,
            display_progress=False,
        )(dspy.Predict(Synthesise))

        scores = [s["score"] if hasattr(s, "score") else s for s in (r[2] for r in result.results)]
        assert scores == [pytest.approx(0.625)] * 3, (
            f"expected the judged composite on every example, got {scores}"
        )
        # 0.0 here would mean dspy swallowed an exception into failure_score.
        assert result.score == pytest.approx(62.5)


class TestUnmeasuredMetricsAreExcludedNotZeroed:
    """An unmeasured RAGAS metric must never be scored as 0.0."""

    def test_unmeasured_metric_is_excluded_and_weights_renormalise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """faithfulness=None must not drag the composite toward zero.

        With equal default weights and the other three at 0.8, the honest score
        over the *measured* metrics is 0.8. Coercing the unmeasured one to 0.0
        would give 0.6 — a quality regression that no judge ever observed.
        """
        evaluator = _StubEvaluator(
            result=_result(
                faithfulness=None,
                answer_relevancy=0.8,
                context_precision=0.8,
                context_recall=0.8,
                overall_score=None,
                metadata={"unmeasured_metrics": ["faithfulness"]},
            )
        )
        metric = _metric(monkeypatch, evaluator)

        out = metric(
            _example(user_query="q", evidence_board=["ctx"]),
            _prediction(synthesis="an answer"),
            None,
            None,
            None,
        )

        assert out.score == pytest.approx(0.8), (
            "unmeasured faithfulness was folded in as 0.0 instead of excluded"
        )
        assert "faithfulness" in out.feedback
        assert "unmeasured" in out.feedback.lower()

    def test_all_metrics_unmeasured_never_returns_a_plausible_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing judged => there is no honest number to return."""
        from src.optimization.gepa.metrics.ragas_metric import RagasUnjudgeableExampleError

        evaluator = _StubEvaluator(
            result=_result(
                faithfulness=None,
                answer_relevancy=None,
                context_precision=None,
                context_recall=None,
                overall_score=None,
                metadata={
                    "unmeasured_metrics": [
                        "answer_relevancy",
                        "context_precision",
                        "context_recall",
                        "faithfulness",
                    ]
                },
            )
        )
        metric = _metric(monkeypatch, evaluator)

        with pytest.raises(RagasUnjudgeableExampleError):
            metric(
                _example(user_query="q", evidence_board=["ctx"]),
                _prediction(synthesis="an answer"),
                None,
                None,
                None,
            )

    def test_heuristic_stamped_result_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The evaluator's own heuristic fallback is not optimization signal."""
        from src.optimization.gepa.metrics.ragas_metric import RagasUnjudgeableExampleError

        evaluator = _StubEvaluator(
            result=_result(metadata={"evaluation_method": "fallback_heuristic"})
        )
        metric = _metric(monkeypatch, evaluator)

        with pytest.raises(RagasUnjudgeableExampleError):
            metric(
                _example(user_query="q", evidence_board=["ctx"]),
                _prediction(synthesis="an answer"),
                None,
                None,
                None,
            )


class TestJudgeDegradationIsCounted:
    """Separate environmental judge failure from candidate-caused unjudgeability.

    Codex iter-1 F2. The raise->failure_score 0.0 inversion is accepted for a
    candidate that produced an unjudgeable answer — that is signal. It is NOT
    acceptable for a judge that timed out or fell back to heuristics mid-run:
    that zeroes a possibly-good candidate, which is noise, and the run would
    still be saved as a success. Counting degradation lets the caller refuse to
    persist a run selected on noise.
    """

    def test_heuristic_stamp_counts_as_degradation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.optimization.gepa.metrics.ragas_metric import RagasUnjudgeableExampleError

        metric = _metric(
            monkeypatch,
            _StubEvaluator(result=_result(metadata={"evaluation_method": "fallback_heuristic"})),
        )
        assert metric.degraded_examples == 0

        with pytest.raises(RagasUnjudgeableExampleError):
            metric(
                _example(user_query="q", evidence_board=["ctx"]),
                _prediction(synthesis="an answer"),
                None,
                None,
                None,
            )

        assert metric.degraded_examples == 1

    def test_judge_exception_counts_as_degradation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A timeout or rate-limit is the environment failing, not the candidate."""
        metric = _metric(monkeypatch, _StubEvaluator(raises=TimeoutError("judge timed out")))

        with pytest.raises(TimeoutError):
            metric(
                _example(user_query="q", evidence_board=["ctx"]),
                _prediction(synthesis="an answer"),
                None,
                None,
                None,
            )

        assert metric.degraded_examples == 1

    def test_candidate_caused_unjudgeability_is_not_degradation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-RAG-shaped example is a dataset defect, not a judge outage.

        Counting it would make every run look degraded and suppress every save.
        """
        from src.optimization.gepa.metrics.ragas_metric import RagasUnjudgeableExampleError

        metric = _metric(monkeypatch, _StubEvaluator(result=_result()))

        with pytest.raises(RagasUnjudgeableExampleError):
            metric(
                _example(analysis_results="no retrieval here"),
                _prediction(executive_summary="a summary"),
                None,
                None,
                None,
            )

        assert metric.degraded_examples == 0

    def test_a_clean_run_reports_no_degradation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        metric = _metric(monkeypatch, _StubEvaluator(result=_result()))
        metric(
            _example(user_query="q", evidence_board=["ctx"]),
            _prediction(synthesis="an answer"),
            None,
            None,
            None,
        )
        assert metric.degraded_examples == 0


class TestFixtureContaminationGuard:
    """Fixture-derived contexts make precision/recall 1.0 by construction."""

    def test_tautological_contexts_exclude_precision_and_recall(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """retrieved == reference => those two metrics carry zero retrieval signal.

        Scoring them would let a 1.0-by-construction pair inflate the composite:
        with faithfulness 0.4 / relevancy 0.4 and precision/recall 1.0, the naive
        mean is 0.7 while the honest score over the metrics that measured
        anything is 0.4.
        """
        shared = ["Q4 report: Northeast TRx fell 12% on payer mix."]
        evaluator = _StubEvaluator(
            result=_result(
                faithfulness=0.4,
                answer_relevancy=0.4,
                context_precision=1.0,
                context_recall=1.0,
            )
        )
        metric = _metric(monkeypatch, evaluator)

        out = metric(
            _example(user_query="q", evidence_board=shared, contexts=shared),
            _prediction(synthesis="an answer"),
            None,
            None,
            None,
        )

        assert out.score == pytest.approx(0.4), (
            "tautological context_precision/context_recall inflated the composite"
        )
        assert "context_precision" in out.feedback
        assert "tautolog" in out.feedback.lower()

    def test_distinct_contexts_keep_precision_and_recall(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real pipeline retrieves something different from the reference."""
        evaluator = _StubEvaluator(
            result=_result(
                faithfulness=0.4,
                answer_relevancy=0.4,
                context_precision=1.0,
                context_recall=1.0,
            )
        )
        metric = _metric(monkeypatch, evaluator)

        out = metric(
            _example(
                user_query="q",
                evidence_board=["retrieved passage from the live index"],
                contexts=["the curated reference passage"],
            ),
            _prediction(synthesis="an answer"),
            None,
            None,
            None,
        )

        assert out.score == pytest.approx(0.7)
        assert "tautolog" not in out.feedback.lower()


class TestExampleShapeExtraction:
    """The metric must read the field names this repo actually emits."""

    def test_reads_evidence_synthesis_signature_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EvidenceSynthesisSignature is the RAG-shaped signature in this codebase.

        Its inputs are user_query + evidence_board and its output is synthesis —
        none of which are the question/answer/contexts names the older
        create_ragas_metric reads.
        """
        evaluator = _StubEvaluator(result=_result())
        metric = _metric(monkeypatch, evaluator)

        metric(
            _example(user_query="why did TRx drop?", evidence_board=["ctx a", "ctx b"]),
            _prediction(synthesis="TRx fell 12%."),
            None,
            None,
            None,
        )

        assert len(evaluator.samples) == 1
        sample = evaluator.samples[0]
        assert sample.query == "why did TRx drop?"
        assert sample.answer == "TRx fell 12%."
        assert sample.retrieved_contexts == ["ctx a", "ctx b"]

    def test_refuses_an_example_with_no_retrieved_contexts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrieval quality cannot be judged where no retrieval happened.

        Guards the case that made `explainer` the wrong registry key: its
        signatures carry no retrieved contexts at all.
        """
        from src.optimization.gepa.metrics.ragas_metric import RagasUnjudgeableExampleError

        evaluator = _StubEvaluator(result=_result())
        metric = _metric(monkeypatch, evaluator)

        with pytest.raises(RagasUnjudgeableExampleError):
            metric(
                _example(analysis_results="agent output", user_expertise="executive"),
                _prediction(executive_summary="a summary with no retrieval behind it"),
                None,
                None,
                None,
            )

        assert evaluator.samples == [], "judge was billed for an unjudgeable example"


class TestFailClosedConstruction:
    """The availability gate has to sit at construction, not per example."""

    def test_construction_refuses_when_judge_cannot_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A per-example refusal would be swallowed into failure_score 0.0.

        Measured on dspy 3.1.0: a raising metric yields 0.0 for every example and
        the run completes. So a judge that cannot run must stop the run *before*
        it starts.
        """
        import src.rag.evaluation as evaluation_module
        from src.optimization.gepa.metrics.ragas_metric import RagasGEPAMetric

        evaluator = _StubEvaluator(
            result=_result(), blockers=("no LLM API key configured for provider 'openai'",)
        )
        monkeypatch.setattr(
            evaluation_module, "get_ragas_evaluator", lambda *a, **k: evaluator, raising=True
        )

        with pytest.raises(Exception) as excinfo:
            RagasGEPAMetric()

        assert "no LLM API key" in str(excinfo.value) or "unavailable" in str(excinfo.value).lower()


class TestRegistryWiring:
    """AGENT_METRICS is the documented factory seam (issue #1486 item 2)."""

    def test_cognitive_rag_resolves_to_the_ragas_metric(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.rag.evaluation as evaluation_module
        from src.optimization.gepa.metrics import get_metric_for_agent
        from src.optimization.gepa.metrics.ragas_metric import RagasGEPAMetric

        monkeypatch.setattr(
            evaluation_module,
            "get_ragas_evaluator",
            lambda *a, **k: _StubEvaluator(result=_result()),
            raising=True,
        )

        assert isinstance(get_metric_for_agent("cognitive_rag"), RagasGEPAMetric)

    def test_explainer_is_not_remapped_to_ragas(self) -> None:
        """#1486 proposed `explainer` too; its signatures carry no retrieval.

        ExplanationSynthesisSignature takes analysis_results / user_expertise /
        focus_areas / output_format — there is nothing retrieved to score, so a
        RAGAS metric here would grade retrieval that never happened.
        """
        from src.optimization.gepa.metrics import AGENT_METRICS
        from src.optimization.gepa.metrics.feedback_learner_metric import (
            FeedbackLearnerGEPAMetric,
        )

        assert AGENT_METRICS["explainer"] is FeedbackLearnerGEPAMetric

    def test_unknown_agents_still_resolve_without_a_judge(self) -> None:
        """Registering a fail-closed entry must not make the factory total-fail."""
        from src.optimization.gepa.metrics import get_metric_for_agent
        from src.optimization.gepa.metrics.standard_agent_metric import StandardAgentGEPAMetric

        assert isinstance(get_metric_for_agent("scope_definer"), StandardAgentGEPAMetric)


class TestNoImportTimeDspyConfigure:
    """dspy 3.1.0 binds the first configure() to its owner thread permanently."""

    def test_module_does_not_configure_dspy_at_import(self) -> None:
        import pathlib

        import src.optimization.gepa.metrics.ragas_metric as module

        source = pathlib.Path(module.__file__).read_text()
        assert "dspy.configure(" not in source
        assert "dspy.settings.configure(" not in source
