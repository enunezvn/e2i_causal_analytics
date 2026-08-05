"""RAGAS-backed GEPA metric for RAG-shaped agents (issue #1486).

RAGAS scores retrieval and grounding quality; this metric is how those scores
reach a GEPA optimizer. It is registered in
:data:`src.optimization.gepa.metrics.AGENT_METRICS` under ``cognitive_rag``.

WHAT IT OPTIMIZES
    Prompts, not retrieval configuration (#1486 item 4). GEPA evolves the
    instruction text of a DSPy signature; ``retrieval_configurations`` in
    ``database/ml/022_self_improvement_tables.sql`` is a different optimizer
    with a different search space and is deliberately out of scope here.

WHY THIS IS A CLASS AND NOT ``create_ragas_metric``
    ``src.optimization.gepa.integration.ragas_feedback.create_ragas_metric``
    predates this module and cannot be handed to GEPA as-is. Three properties,
    each measured against the installed dspy 3.1.0:

    1. It returns an ``async`` function. GEPA calls its metric synchronously, so
       the return value is a coroutine object — never a score.
    2. It returns a plain ``dict``. ``dspy.Evaluate`` sums metric returns and
       dies with ``TypeError: unsupported operand type(s) for +: 'int' and
       'dict'``; GEPA's own coercion is ``s["score"] if hasattr(s, "score")
       else s``, which a dict fails. Only ``dspy.Prediction`` survives.
    3. It reads ``example.question`` / ``pred.answer`` / ``pred.contexts``,
       none of which any signature in this repository emits. The RAG-shaped
       signature here is ``EvidenceSynthesisSignature`` (``user_query`` +
       ``evidence_board`` -> ``synthesis``).

WHY AN UNMEASURED METRIC IS EXCLUDED RATHER THAN REFUSED
    #1488 made ``RAGASFeedbackProvider.evaluate`` *raise* when any RAGAS metric
    came back unmeasured, so that "not measured" could never be reported as
    "measured 0.0". Inside GEPA that guarantee inverts: dspy's parallelizer
    catches a raising metric and records ``failure_score`` (0.0) for the
    example, then completes the run. Measured on dspy 3.1.0 — a metric that
    raises on every example produces ``Average Metric: 0.0 / 3 (0.0%)``, not an
    abort. Refusing per example therefore *creates* the fabricated
    bad-quality signal it was meant to prevent.

    So this metric consumes the evaluator directly, where per-metric ``None``
    is still visible, and drops unmeasured metrics from the weighted average
    (renormalising over what was actually judged) while naming them in the
    feedback text. That reuses #1488's real contribution — ``None`` means
    unmeasured — at the layer that can act on it.

    Two cases genuinely have no honest number and DO raise
    (:class:`RagasUnjudgeableExampleError`): nothing was judged at all, and the
    example is not RAG-shaped. Both are dataset/wiring defects rather than
    candidate quality, and both are logged at ERROR. They still land as 0.0
    inside GEPA, but a run where they fire en masse collapses to an obviously
    broken aggregate instead of a plausible mid-range score — a loud failure,
    which is the point.

WHY THE AVAILABILITY GATE IS AT CONSTRUCTION
    Same measurement, opposite conclusion: because a per-example refusal is
    swallowed, a judge that cannot run at all has to stop the run *before* it
    starts. Constructing this metric verifies the judged path and raises
    otherwise, so ``get_metric_for_agent("cognitive_rag")`` fails loudly in a
    keyless environment rather than handing back something that would score
    every candidate 0.0.
"""

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dspy import Example, Prediction

from src.optimization.gepa.integration.ragas_feedback import RAGASFeedbackConfig
from src.optimization.gepa.metrics.base import DSPyTrace

logger = logging.getLogger(__name__)

# Field names this repository actually uses, most specific first. The older
# create_ragas_metric read question/answer/contexts, which no signature here
# emits; EvidenceSynthesisSignature is the RAG-shaped one (user_query +
# evidence_board -> synthesis).
_QUESTION_FIELDS = ("user_query", "question", "query", "original_query", "rewritten_query")
_ANSWER_FIELDS = ("synthesis", "answer", "response", "generated")
_RETRIEVED_CONTEXT_FIELDS = ("retrieved_contexts", "evidence_board", "contexts", "evidence")
_REFERENCE_CONTEXT_FIELDS = ("reference_contexts", "contexts", "ground_truth_contexts")

_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# context_precision and context_recall both compare retrieved contexts against
# the reference ones. When those are the same object, both are 1.0 by
# construction and carry no retrieval signal at all.
_RETRIEVAL_COMPARISON_METRICS = ("context_precision", "context_recall")


class RagasMetricUnavailableError(RuntimeError):
    """The RAGAS judge cannot run, so no RAGAS-backed metric can be built.

    Raised at construction (see the module docstring): a per-example refusal
    would be swallowed by dspy into ``failure_score`` 0.0.
    """


class RagasUnjudgeableExampleError(RuntimeError):
    """This example carries no RAGAS signal, so no honest score exists for it.

    Either nothing was judged, or the example is not RAG-shaped (no retrieved
    contexts). Both are dataset/wiring defects, not candidate quality.
    """


def _field(obj: Any, names: Sequence[str]) -> Optional[Any]:
    """First present, non-empty value among ``names`` on a dspy Example/Prediction.

    ``dspy.Example.__getattr__`` raises ``AttributeError`` for absent keys, and
    a dict may arrive instead of an Example, so both access styles are tried.
    """
    for name in names:
        value = None
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            try:
                value = getattr(obj, name)
            except AttributeError:
                value = None
        if value is not None and value != "" and value != []:
            return value
    return None


def _as_context_list(value: Any) -> List[str]:
    """Normalise a contexts field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item if isinstance(item, str) else str(item) for item in value]
    return [str(value)]


class RagasGEPAMetric:
    """GEPA metric scoring RAG prompt candidates with RAGAS judgements.

    Attributes:
        name: Metric name identifier.
        description: Metric description for logging.
        config: Weights and thresholds, reused from the ragas_feedback module.
    """

    name = "ragas_gepa"
    description = "GEPA metric for RAG agents - RAGAS retrieval and grounding quality"

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        config: Optional[RAGASFeedbackConfig] = None,
        evaluator: Optional[Any] = None,
    ) -> None:
        """Build the metric, verifying the RAGAS judge can actually run.

        Args:
            weights: Optional per-metric weights; defaults to RAGASFeedbackConfig's.
            config: Optional pre-built config (takes precedence over ``weights``).
            evaluator: Optional pre-built RAGASEvaluator, for tests and callers
                that already hold one.

        Raises:
            RagasMetricUnavailableError: the judged RAGAS path is blocked.
        """
        if config is None:
            config = RAGASFeedbackConfig()
            if weights:
                # __post_init__ normalises; assigning after it means doing so here.
                total = sum(weights.values())
                config.weights = (
                    {k: v / total for k, v in weights.items()} if total else dict(weights)
                )
        self.config = config
        self._evaluator = evaluator if evaluator is not None else self._resolve_evaluator()
        # Count of examples the JUDGE failed to score (heuristic fallback mid-run,
        # timeout, rate-limit) as opposed to examples the CANDIDATE made
        # unjudgeable. Both raise, and dspy converts either into failure_score
        # 0.0 — but only the first is noise: it zeroes a possibly-good candidate
        # for an environmental reason, so a run containing any of it must not be
        # persisted or deduped as a success. Locked because GEPA evaluates on
        # worker threads.
        self._degraded_examples = 0
        self._degradation_lock = threading.Lock()

    @property
    def degraded_examples(self) -> int:
        """Examples whose score was lost to judge degradation, not candidate quality."""
        with self._degradation_lock:
            return self._degraded_examples

    def reset_degradation(self) -> None:
        """Clear the counter so one metric instance can serve several runs."""
        with self._degradation_lock:
            self._degraded_examples = 0

    def _record_degradation(self) -> None:
        with self._degradation_lock:
            self._degraded_examples += 1

    @staticmethod
    def _resolve_evaluator() -> Any:
        """Return a RAGASEvaluator whose judged path is verified to be open.

        Imported lazily so that importing this module never pulls in the RAGAS
        dependency tree (and never touches dspy's global configuration).
        """
        try:
            from src.rag.evaluation import get_ragas_evaluator
        except ImportError as e:  # pragma: no cover - RAGAS is a hard dep in prod
            raise RagasMetricUnavailableError(
                f"RAGAS evaluator is unavailable ({e}); refusing to build a "
                "RAGAS-backed GEPA metric. Optimizing against substitute scores "
                "would evolve prompts toward fabricated signal (see issue #491)."
            ) from e

        evaluator = get_ragas_evaluator()

        # judged_path_blockers / verify_dependencies arrive with #1488. Absent
        # them the gate degrades to "an evaluator exists", which is why this
        # metric additionally refuses any heuristic-stamped result per example.
        blockers: Tuple[str, ...] = tuple(getattr(evaluator, "judged_path_blockers", ()) or ())
        if blockers:
            raise RagasMetricUnavailableError(
                "RAGAS judged path is unavailable, so every candidate would be "
                "scored by heuristics: " + "; ".join(blockers) + ". Refusing to "
                "build a RAGAS-backed GEPA metric."
            )
        verify = getattr(evaluator, "verify_dependencies", None)
        if callable(verify):
            # find_spec presence != importability; this is the #491 break class.
            verify()
        return evaluator

    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> Prediction:
        """Score a RAG candidate against RAGAS judgements.

        Args:
            gold: Ground-truth Example (supplies the query and, when present,
                the reference contexts used to detect fixture tautology).
            pred: Candidate Prediction (supplies the answer and retrieved contexts).
            trace: Full DSPy execution trace (unused).
            pred_name: Name of the predictor being optimized (unused).
            pred_trace: Trace for this predictor (unused).

        Returns:
            ``dspy.Prediction(score: float in [0, 1], feedback: str)`` — the only
            shape GEPA can consume.

        Raises:
            RagasUnjudgeableExampleError: the example is not RAG-shaped, or the
                judge measured nothing.
        """
        import dspy

        question, answer, retrieved, reference = self._extract(gold, pred)
        result = self._judge(question, answer, retrieved, reference)
        score, feedback = self._compose(result, retrieved, reference)
        return dspy.Prediction(score=float(score), feedback=feedback)

    def _extract(
        self, gold: Example, pred: Prediction
    ) -> Tuple[str, str, List[str], Optional[List[str]]]:
        """Pull (question, answer, retrieved_contexts, reference_contexts).

        Raises:
            RagasUnjudgeableExampleError: no question, no answer, or — the case
                that makes ``explainer`` the wrong registry key — no retrieved
                contexts, meaning there is no retrieval to judge.
        """
        question = _field(gold, _QUESTION_FIELDS) or _field(pred, _QUESTION_FIELDS)
        answer = _field(pred, _ANSWER_FIELDS)
        # Retrieval normally rides the prediction; a captured example may carry
        # it instead, so fall back to gold before giving up.
        retrieved = _as_context_list(
            _field(pred, _RETRIEVED_CONTEXT_FIELDS) or _field(gold, _RETRIEVED_CONTEXT_FIELDS)
        )
        reference = _field(gold, _REFERENCE_CONTEXT_FIELDS)

        missing = []
        if not question:
            missing.append("question")
        if not answer:
            missing.append("answer")
        if not retrieved:
            missing.append("retrieved contexts")
        if missing:
            message = (
                "Example is not RAG-shaped (no "
                + ", ".join(missing)
                + f"); searched {_QUESTION_FIELDS}, {_ANSWER_FIELDS} and "
                f"{_RETRIEVED_CONTEXT_FIELDS}. RAGAS cannot judge retrieval "
                "quality where no retrieval is recorded, and a number invented "
                "here would be indistinguishable from a judged one."
            )
            logger.error("RAGAS GEPA metric: %s", message)
            raise RagasUnjudgeableExampleError(message)

        reference_list = _as_context_list(reference) if reference is not None else None
        return str(question), str(answer), retrieved, reference_list

    def _judge(
        self,
        question: str,
        answer: str,
        retrieved: List[str],
        reference: Optional[List[str]],
    ) -> Any:
        """Run the RAGAS judge for one sample, bridging its async API to GEPA's sync call."""
        from src.rag.evaluation import EvaluationSample

        sample = EvaluationSample(
            query=question,
            ground_truth=answer,
            contexts=reference or [],
            answer=answer,
            retrieved_contexts=retrieved,
        )
        try:
            return _run_sync(self._evaluator.evaluate_sample(sample))
        except Exception:
            # A timeout, rate-limit or transport error is the ENVIRONMENT
            # failing, not the candidate. dspy will still turn this into
            # failure_score 0.0, so record it: the caller refuses to persist a
            # run whose scores are partly noise.
            self._record_degradation()
            raise

    def _compose(
        self,
        result: Any,
        retrieved: List[str],
        reference: Optional[List[str]],
    ) -> Tuple[float, str]:
        """Turn an EvaluationResult into a weighted score plus reflective feedback.

        Raises:
            RagasUnjudgeableExampleError: the result is heuristic rather than
                judged, or nothing survived exclusion.
        """
        metadata = getattr(result, "metadata", None) or {}

        # The evaluator stamps its own heuristic path so consumers can tell
        # synthetic scores from judged ones. Construction already refuses a
        # statically-blocked judge; this catches degradation DURING a run.
        if metadata.get("evaluation_method") == "fallback_heuristic":
            message = (
                "RAGASEvaluator returned fallback_heuristic scores rather than judged "
                "ones (the judge degraded mid-run); refusing to feed heuristics to "
                "GEPA as optimization signal."
            )
            logger.error("RAGAS GEPA metric: %s", message)
            # Judge-caused, not candidate-caused: see _record_degradation.
            self._record_degradation()
            raise RagasUnjudgeableExampleError(message)

        # `is not None` rather than truthiness — a judged 0.0 is a real score.
        judged = {
            name: value
            for name in _RAGAS_METRICS
            if (value := getattr(result, name, None)) is not None
        }
        unmeasured = [name for name in _RAGAS_METRICS if name not in judged]

        tautological: List[str] = []
        if _is_tautological(retrieved, reference):
            tautological = [name for name in _RETRIEVAL_COMPARISON_METRICS if name in judged]
            for name in tautological:
                del judged[name]

        if not judged:
            message = (
                "No RAGAS metric carries signal for this example ("
                + f"unmeasured={unmeasured or 'none'}, "
                + f"tautological={tautological or 'none'}"
                + "); refusing to invent a score."
            )
            logger.error("RAGAS GEPA metric: %s", message)
            raise RagasUnjudgeableExampleError(message)

        score = self._weighted(judged)
        return score, _feedback(score, judged, unmeasured, tautological)

    def _weighted(self, judged: Dict[str, float]) -> float:
        """Weighted mean over the judged metrics, renormalised to their weights.

        Renormalisation is what keeps an excluded metric from acting like a
        zero: dropping ``faithfulness`` from four equal weights leaves three
        weights summing to 0.75, so dividing by 0.75 reports the mean of what
        was judged instead of diluting it toward 0.
        """
        weights = {name: self.config.weights.get(name, 0.0) for name in judged}
        total_weight = sum(weights.values())
        if total_weight <= 0:
            # Configured weights exclude everything judged; fall back to a plain
            # mean rather than dividing by zero.
            return sum(judged.values()) / len(judged)
        return sum(judged[name] * weights[name] for name in judged) / total_weight


def _is_tautological(retrieved: List[str], reference: Optional[List[str]]) -> bool:
    """Whether retrieved contexts are the reference contexts (fixture-shaped).

    When a golden-set fixture supplies ``retrieved_contexts == contexts``,
    context_precision and context_recall are 1.0 by construction: the retriever
    is being graded against the exact passages it was handed. Those two metrics
    then measure nothing about retrieval, and letting them into the average
    inflates every candidate equally.
    """
    if not reference or not retrieved:
        return False
    return [c.strip() for c in retrieved] == [c.strip() for c in reference]


def _feedback(
    score: float,
    judged: Dict[str, float],
    unmeasured: List[str],
    tautological: List[str],
) -> str:
    """Reflective feedback naming both the scores and everything excluded."""
    parts = [f"RAG quality {score:.3f} over {len(judged)} judged metric(s)."]
    parts.append(" ".join(f"{name}={value:.2f}." for name, value in sorted(judged.items())))

    if unmeasured:
        parts.append(
            "Excluded as unmeasured (the judge returned no value; NOT a zero): "
            + ", ".join(sorted(unmeasured))
            + "."
        )
    if tautological:
        parts.append(
            "Excluded as tautological (retrieved contexts are identical to the "
            "reference contexts, so these are 1.0 by construction and carry no "
            "retrieval signal): " + ", ".join(sorted(tautological)) + "."
        )

    suggestions = []
    if judged.get("faithfulness", 1.0) < 0.7:
        suggestions.append("ground every claim in the retrieved context")
    if judged.get("answer_relevancy", 1.0) < 0.7:
        suggestions.append("answer the question asked, more directly")
    if judged.get("context_precision", 1.0) < 0.7:
        suggestions.append("retrieve fewer irrelevant passages")
    if judged.get("context_recall", 1.0) < 0.7:
        suggestions.append("retrieve the passages the answer still misses")
    if suggestions:
        parts.append("Improve by: " + "; ".join(suggestions) + ".")

    return " ".join(parts)


def _run_sync(coro: Any) -> Any:
    """Run a coroutine from GEPA's synchronous metric call.

    GEPA evaluates on worker threads with no running loop, where ``asyncio.run``
    is correct. But the nightly optimizer entry points are ``async``, and a
    single-threaded evaluation can run inline on that loop — where ``asyncio.run``
    raises. Mirrors ``recipient_optimizer._fetch_recipient_signals``.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


__all__ = [
    "RagasGEPAMetric",
    "RagasMetricUnavailableError",
    "RagasUnjudgeableExampleError",
]
