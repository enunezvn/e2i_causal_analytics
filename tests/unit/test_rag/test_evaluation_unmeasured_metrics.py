"""A NaN'd RAGAS metric is "not measured", never "measured 0.0" (issue #1488).

``_evaluate_with_ragas`` used to run every extracted metric through a nested
``safe_score(value, default=0.0)`` that turned ``None``/``NaN`` into ``0.0``.
ragas emits ``NaN`` per metric on metric-level failures (an exception inside a
metric, empty statement/claim extraction), so the result carried four plausible
floats with no marker at all — the fabricated-score class this issue removes,
one layer below ``RAGASFeedbackProvider``'s refusals, and indistinguishable
after the fact from a genuinely judged 0.0.

These tests pin the judged path to preserving ``None`` and naming what was not
measured, and pin ``run_evaluation``'s aggregation to tolerating it. See also
tests/unit/test_optimization/test_ragas_feedback_fail_loud.py for the consumer
that refuses on those signals.
"""

import math
from typing import Any
from unittest.mock import patch

import pytest

from src.rag.evaluation import (
    EvaluationResult,
    EvaluationSample,
    RAGASEvaluator,
)

SAMPLE = EvaluationSample(
    query="What caused the TRx drop?",
    ground_truth="Payer mix shifted.",
    answer="The drop was caused by payer mix.",
    retrieved_contexts=["Q4 report: payer mix shifted."],
)


class _Frame:
    """Minimal stand-in for the ragas result's pandas frame."""

    def __init__(self, row: dict[str, Any]):
        self._row = row

    def to_pandas(self) -> "_Frame":
        return self

    @property
    def iloc(self) -> "_Frame":
        return self

    def __getitem__(self, _index: int) -> "_Frame":
        return self

    def to_dict(self) -> dict[str, Any]:
        return self._row


def _components(row: dict[str, Any]) -> dict[str, Any]:
    """The component bundle `_evaluate_with_ragas` imports, with ragas stubbed.

    Test-only stub at the ragas boundary — everything the evaluator does with
    the returned scores is the real production code.
    """

    class _Dataset:
        @staticmethod
        def from_dict(_data: dict[str, Any]) -> object:
            return object()

    class _Embeddings:
        def __init__(self, *a: Any, **k: Any) -> None: ...

    class _OpenAI:
        def __init__(self, *a: Any, **k: Any) -> None: ...

    class _Metric:
        """The evaluator assigns .llm / .embeddings onto each metric object."""

    return {
        "openai": type("openai", (), {"OpenAI": _OpenAI}),
        "Dataset": _Dataset,
        "evaluate": lambda **_kwargs: _Frame(row),
        "OpenAIEmbeddings": _Embeddings,
        "llm_factory": lambda *a, **k: _Metric(),
        "faithfulness": _Metric(),
        "answer_relevancy": _Metric(),
        "context_precision": _Metric(),
        "context_recall": _Metric(),
    }


async def _judge(row: dict[str, Any]) -> EvaluationResult:
    evaluator = RAGASEvaluator(enable_opik_tracing=False)
    with patch("src.rag.evaluation._import_ragas_components", return_value=_components(row)):
        return await evaluator._evaluate_with_ragas(SAMPLE, "sid")


ALL_JUDGED = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.7,
    "context_precision": 0.9,
    "context_recall": 0.6,
}


class TestNaNIsNotZero:
    async def test_nan_metric_is_preserved_as_none(self):
        result = await _judge({**ALL_JUDGED, "faithfulness": float("nan")})

        assert result.faithfulness is None, "NaN must not become a real-looking 0.0"

    async def test_metrics_that_were_judged_survive(self):
        """A partial failure must not discard the judgements that did happen."""
        result = await _judge({**ALL_JUDGED, "faithfulness": float("nan")})

        assert result.answer_relevancy == pytest.approx(0.7)
        assert result.context_precision == pytest.approx(0.9)
        assert result.context_recall == pytest.approx(0.6)

    async def test_nan_does_not_degrade_to_heuristic_fallback(self):
        """The judge ran — routing to heuristics would be a different lie."""
        result = await _judge({**ALL_JUDGED, "context_recall": float("nan")})

        assert result.metadata.get("evaluation_method") != "fallback_heuristic"

    async def test_unmeasured_metrics_are_named(self):
        result = await _judge(
            {**ALL_JUDGED, "faithfulness": float("nan"), "context_recall": float("nan")}
        )

        assert result.metadata.get("unmeasured_metrics") == ["context_recall", "faithfulness"]

    async def test_overall_score_is_none_when_a_metric_is_unmeasured(self):
        """A mean over four metrics is undefined when one never happened."""
        result = await _judge({**ALL_JUDGED, "context_recall": float("nan")})

        assert result.overall_score is None

    async def test_thresholds_cannot_pass_with_an_unmeasured_metric(self):
        result = await _judge(dict.fromkeys(ALL_JUDGED, 1.0) | {"context_recall": float("nan")})

        assert result.passed_thresholds is False

    async def test_fully_judged_sample_is_unchanged(self):
        result = await _judge(ALL_JUDGED)

        assert result.faithfulness == pytest.approx(0.8)
        assert result.overall_score == pytest.approx((0.8 + 0.7 + 0.9 + 0.6) / 4)
        assert result.metadata.get("unmeasured_metrics") is None

    async def test_judged_zero_survives(self):
        """0.0 from the judge is a real score and must not be confused with NaN."""
        result = await _judge({**ALL_JUDGED, "faithfulness": 0.0})

        assert result.faithfulness == 0.0
        assert result.metadata.get("unmeasured_metrics") is None

    async def test_none_metric_is_treated_like_nan(self):
        result = await _judge({**ALL_JUDGED, "answer_relevancy": None})

        assert result.answer_relevancy is None
        assert result.metadata.get("unmeasured_metrics") == ["answer_relevancy"]

    async def test_missing_metric_key_is_treated_like_nan(self):
        row = {k: v for k, v in ALL_JUDGED.items() if k != "context_precision"}
        result = await _judge(row)

        assert result.context_precision is None


def _result(**overrides: Any) -> EvaluationResult:
    fields: dict[str, Any] = {
        "sample_id": "s",
        "query": "q",
        "faithfulness": 0.8,
        "answer_relevancy": 0.8,
        "context_precision": 0.8,
        "context_recall": 0.8,
        "overall_score": 0.8,
    }
    fields.update(overrides)
    return EvaluationResult(**fields)


class TestAggregationToleratesUnmeasured:
    """run_evaluation filtered on faithfulness alone, then summed the rest.

    A sample with a judged faithfulness but one NaN'd metric therefore passed
    the filter and blew up the whole run on a TypeError.
    """

    async def _report(self, results: list[EvaluationResult]):
        from src.rag.evaluation import RAGEvaluationPipeline

        with patch("src.rag.evaluation.get_default_evaluation_dataset", return_value=[SAMPLE]):
            pipeline = RAGEvaluationPipeline(enable_opik_tracing=False)

        async def _batch(*_a: Any, **_k: Any) -> list[EvaluationResult]:
            return results

        with patch.object(pipeline.evaluator, "evaluate_batch", _batch):
            return await pipeline.run_evaluation()

    async def test_partially_unmeasured_result_does_not_abort_the_run(self):
        report = await self._report([_result(), _result(context_recall=None, overall_score=None)])

        assert report is not None
        assert report.avg_faithfulness == pytest.approx(0.8)

    async def test_average_skips_only_the_unmeasured_metric(self):
        report = await self._report(
            [_result(context_recall=0.4), _result(context_recall=None, overall_score=None)]
        )

        # context_recall averages over the one sample that measured it.
        assert report.avg_context_recall == pytest.approx(0.4)
        # the others average over both.
        assert report.avg_answer_relevancy == pytest.approx(0.8)

    async def test_metric_unmeasured_everywhere_reports_none(self):
        report = await self._report([_result(context_recall=None, overall_score=None)])

        assert report.avg_context_recall is None

    async def test_all_judged_report_is_unchanged(self):
        report = await self._report([_result(), _result()])

        assert report.avg_context_recall == pytest.approx(0.8)
        assert report.overall_score == pytest.approx(0.8)


class TestCheckThresholdsFailsClosed:
    """An unmeasured aggregate must not silently pass the quality gate.

    ``check_thresholds`` guards every metric with ``if report.avg_X is not
    None``. Those guards were dead code while ``safe_score`` guaranteed floats;
    making unmeasured metrics reachable turned the latent fail-open live, and
    for THIS consumer it is a behaviour regression: a NaN'd metric used to
    become a fabricated 0.0 that failed the gate (wrong, but closed), and would
    now be None and skipped (silently open). ``scripts/run_ragas_eval.py``
    exits 0 under ``--fail-on-threshold`` on that path, while the very same
    report carries ``passed_thresholds=False`` per sample.
    """

    def _pipeline(self):
        from src.rag.evaluation import RAGEvaluationPipeline

        with patch("src.rag.evaluation.get_default_evaluation_dataset", return_value=[SAMPLE]):
            return RAGEvaluationPipeline(enable_opik_tracing=False)

    def _report(self, **overrides: Any):
        from src.rag.evaluation import EvaluationReport

        fields: dict[str, Any] = {
            "run_id": "r",
            "timestamp": "2026-08-05T00:00:00",
            "total_samples": 1,
            "passed_samples": 1,
            "failed_samples": 0,
            "avg_faithfulness": 0.9,
            "avg_answer_relevancy": 0.9,
            "avg_context_precision": 0.9,
            "avg_context_recall": 0.9,
            "overall_score": 0.9,
            "evaluation_time_seconds": 1.0,
        }
        fields.update(overrides)
        return EvaluationReport(**fields)

    def test_all_measured_and_passing_still_passes(self):
        passed, failures = self._pipeline().check_thresholds(self._report())

        assert (passed, failures) == (True, [])

    def test_all_measured_below_threshold_still_fails_with_the_comparison(self):
        passed, failures = self._pipeline().check_thresholds(self._report(avg_context_recall=0.10))

        assert passed is False
        assert any("Context Recall 0.100 <" in f for f in failures)

    def test_unmeasured_metric_fails_closed(self):
        passed, failures = self._pipeline().check_thresholds(
            self._report(avg_context_recall=None, overall_score=None)
        )

        assert passed is False, "an unmeasured metric must not pass the gate"
        assert any("Context Recall" in f for f in failures)

    def test_unmeasured_message_does_not_fabricate_a_comparison(self):
        """Report it as unverifiable — never as a number that lost to a threshold."""
        _passed, failures = self._pipeline().check_thresholds(
            self._report(avg_faithfulness=None, overall_score=None)
        )

        message = next(f for f in failures if "Faithfulness" in f)
        assert "unmeasured" in message.lower()
        assert "<" not in message, "no comparison happened; do not imply one"

    def test_every_unmeasured_metric_is_named(self):
        _passed, failures = self._pipeline().check_thresholds(
            self._report(
                avg_faithfulness=None,
                avg_answer_relevancy=None,
                avg_context_precision=None,
                avg_context_recall=None,
                overall_score=None,
            )
        )

        joined = " | ".join(failures)
        for name in ("Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"):
            assert name in joined

    def test_unmeasured_overall_score_fails_closed(self):
        """Catches "no sample was fully judged" even when each metric has data.

        Two half-judged samples can leave all four averages populated (each from
        the sample that measured it) while no sample produced an overall_score.
        """
        passed, failures = self._pipeline().check_thresholds(self._report(overall_score=None))

        assert passed is False
        assert any("Overall" in f and "unmeasured" in f.lower() for f in failures)


def test_safe_score_helper_is_gone():
    """The NaN->0.0 coercion must not survive anywhere in the judged path."""
    import inspect

    source = inspect.getsource(RAGASEvaluator._evaluate_with_ragas)

    assert "safe_score" not in source
    assert math  # imported for the NaN literals above
