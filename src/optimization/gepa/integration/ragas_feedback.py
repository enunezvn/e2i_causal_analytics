"""RAGAS Feedback Provider for GEPA Optimization.

This module integrates RAGAS evaluation metrics as feedback signals for GEPA
optimization, specifically for RAG-based agents (cognitive_rag, explainer).

RAGAS provides structured evaluation of RAG quality:
- Faithfulness: Answer grounded in retrieved context
- Answer Relevancy: Answer addresses the question
- Context Precision: Retrieved context is relevant
- Context Recall: All relevant context retrieved

These scores become feedback signals that GEPA uses to evolve better prompts.

Usage:
    from src.optimization.gepa.integration import RAGASFeedbackProvider, create_ragas_metric

    # Create feedback provider
    provider = RAGASFeedbackProvider(
        weights={"faithfulness": 0.3, "answer_relevancy": 0.3, "context_precision": 0.4}
    )

    # Get GEPA-compatible metric
    metric = create_ragas_metric(provider, agent_name="cognitive_rag")

Failure contract (issue #1488):
    A score from this module always means "the RAGAS judge ran and returned
    this number". Evaluation failures are never represented as a score — they
    raise (``RagasDependencyError`` propagates; everything else surfaces as
    ``RAGASFeedbackUnavailableError`` or the original exception). This mirrors
    the #491 discipline in :mod:`src.rag.evaluation`, where heuristic fallbacks
    are kept outside the broad ``except`` because they "look like real
    (failing) RAG metrics and masquerade as a quality regression".

    The load-bearing seam is *construction*, not per-example evaluation.
    Measured against dspy 3.1.0: a metric that raises inside
    ``dspy.Evaluate`` is caught by ``ParallelExecutor`` and converted to
    ``failure_score`` (0.0) for that example, so a per-example raise cannot
    abort a GEPA run. Building the provider/metric happens outside that
    executor, so raising there does abort — which is why an unavailable
    evaluator fails at ``__post_init__`` rather than at first use.

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional, Protocol, Union

logger = logging.getLogger(__name__)


class RAGASFeedbackUnavailableError(RuntimeError):
    """Raised when RAGAS-backed GEPA feedback cannot be produced for real.

    Mirrors the #491 discipline of :class:`src.rag.evaluation.RagasDependencyError`
    one layer up: rather than substituting heuristics or a 0.0, refuse to
    produce optimization signal that GEPA cannot tell apart from a judged score.
    """


# Type alias for GEPA's expected feedback format
ScoreWithFeedback = dict[str, Union[float, str]]


class RAGEvaluationResult(Protocol):
    """Protocol for RAG evaluation results from RAGAS or similar."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


@dataclass
class RAGASFeedbackConfig:
    """Configuration for RAGAS feedback generation.

    Attributes:
        weights: Weights for combining RAGAS metrics
        feedback_template: Template for generating textual feedback
        min_score_threshold: Minimum acceptable score
        include_subscores: Whether to include individual metric scores in feedback
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "faithfulness": 0.25,
            "answer_relevancy": 0.25,
            "context_precision": 0.25,
            "context_recall": 0.25,
        }
    )
    feedback_template: str = (
        "RAG Quality Assessment:\n"
        "- Faithfulness: {faithfulness:.2f} ({faithfulness_feedback})\n"
        "- Relevancy: {answer_relevancy:.2f} ({relevancy_feedback})\n"
        "- Precision: {context_precision:.2f} ({precision_feedback})\n"
        "- Recall: {context_recall:.2f} ({recall_feedback})\n"
        "Overall: {overall_feedback}"
    )
    min_score_threshold: float = 0.6
    include_subscores: bool = True

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"RAGAS weights sum to {total}, normalizing to 1.0")
            self.weights = {k: v / total for k, v in self.weights.items()}


@dataclass
class RAGASFeedbackProvider:
    """Provides GEPA-compatible feedback from RAGAS evaluations.

    Converts RAGAS evaluation results into the ScoreWithFeedback format
    expected by GEPA metrics, enabling evolutionary optimization of
    RAG-based agents.

    Example:
        >>> provider = RAGASFeedbackProvider()
        >>> result = await provider.evaluate(
        ...     question="What caused the TRx drop?",
        ...     answer="The drop was caused by...",
        ...     contexts=["Context 1...", "Context 2..."],
        ... )
        >>> result
        {'score': 0.82, 'feedback': 'RAG Quality Assessment: ...'}
    """

    config: RAGASFeedbackConfig = field(default_factory=RAGASFeedbackConfig)
    _ragas_evaluator: Any = None
    _evaluation_sample_class: Any = None

    def __post_init__(self) -> None:
        """Initialize the RAGAS evaluator, or refuse to construct.

        Verifies the judge can actually RUN, not merely that it imports.
        Constructing ``RAGASEvaluator`` proves nothing: it sets availability
        flags and warns rather than failing, so a keyless environment would
        yield a provider that routes every sample to the stamped heuristic
        fallback. That would surface as a per-example refusal, which DSPy
        converts to ``failure_score`` 0.0 — the very masquerade this module
        exists to prevent.

        Raises:
            RAGASFeedbackUnavailableError: the evaluator cannot be imported, or
                its judged path is blocked (no LLM key, ragas not installed).
            RagasDependencyError: the RAGAS dependency tree is broken (#491).
        """
        try:
            from src.rag.evaluation import EvaluationSample, get_ragas_evaluator

            evaluator = get_ragas_evaluator()
        except ImportError as e:
            raise RAGASFeedbackUnavailableError(
                f"RAGAS evaluator is unavailable ({e}); refusing to produce GEPA "
                "feedback. Optimizing against substitute scores would evolve "
                "prompts toward fabricated signal — fix the RAGAS dependency "
                "tree (see issue #491) instead."
            ) from e

        blockers = evaluator.judged_path_blockers
        if blockers:
            raise RAGASFeedbackUnavailableError(
                "RAGAS judged path is unavailable, so every candidate would be "
                "scored by heuristics: "
                + "; ".join(blockers)
                + ". Refusing to produce GEPA feedback."
            )

        # find_spec presence != importability; this is the #491 break class.
        evaluator.verify_dependencies()

        self._ragas_evaluator = evaluator
        self._evaluation_sample_class = EvaluationSample
        logger.debug("RAGASFeedbackProvider initialized with RAGAS evaluator")

    @property
    def enabled(self) -> bool:
        """Whether RAGAS evaluation is available.

        Always ``True`` for a successfully constructed provider — construction
        raises otherwise (#1488). Retained for API compatibility.
        """
        return self._ragas_evaluator is not None

    def _get_feedback_text(self, score: float, metric_name: str) -> str:
        """Generate feedback text for a metric score.

        Args:
            score: Metric score (0-1)
            metric_name: Name of the metric

        Returns:
            Human-readable feedback text
        """
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "acceptable"
        elif score >= 0.6:
            return "needs improvement"
        else:
            return "poor"

    def _compute_weighted_score(self, scores: dict[str, float]) -> float:
        """Compute weighted average of RAGAS scores.

        Args:
            scores: Dictionary of metric name to score

        Returns:
            Weighted average score
        """
        total = 0.0
        for metric, weight in self.config.weights.items():
            total += scores.get(metric, 0.0) * weight
        return total

    def _generate_overall_feedback(
        self,
        weighted_score: float,
        scores: dict[str, float],
    ) -> str:
        """Generate overall feedback text.

        Args:
            weighted_score: Weighted average score
            scores: Individual metric scores

        Returns:
            Overall feedback text with suggestions
        """
        suggestions = []

        if scores.get("faithfulness", 1.0) < 0.7:
            suggestions.append("Improve answer grounding in retrieved context")

        if scores.get("answer_relevancy", 1.0) < 0.7:
            suggestions.append("Focus answer more directly on the question")

        if scores.get("context_precision", 1.0) < 0.7:
            suggestions.append("Improve retrieval to get more relevant context")

        if scores.get("context_recall", 1.0) < 0.7:
            suggestions.append("Ensure all relevant information is retrieved")

        if weighted_score >= 0.8:
            base = "Strong RAG performance."
        elif weighted_score >= 0.6:
            base = "Acceptable RAG performance with room for improvement."
        else:
            base = "RAG performance needs significant improvement."

        if suggestions:
            return f"{base} Suggestions: {'; '.join(suggestions)}"
        return base

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreWithFeedback:
        """Evaluate a RAG response and return GEPA-compatible feedback.

        Args:
            question: The user question
            answer: The generated answer
            contexts: Retrieved context documents
            ground_truth: Optional ground truth answer
            run_id: Optional run ID for Opik tracing
            **kwargs: Additional evaluation parameters

        Returns:
            ScoreWithFeedback dict with 'score' and 'feedback' keys

        Raises:
            RagasDependencyError: the RAGAS dependency tree is broken (#491).
                Propagated deliberately — collapsing it to 0.0 would reach GEPA
                as "this candidate scored zero".
            RAGASFeedbackUnavailableError: the evaluator produced heuristic
                rather than judged scores.
        """
        if not (self._ragas_evaluator and self._evaluation_sample_class):
            raise RAGASFeedbackUnavailableError(
                "RAGASFeedbackProvider has no evaluator; cannot produce GEPA feedback."
            )

        sample = self._evaluation_sample_class(
            query=question,
            ground_truth=ground_truth or answer,
            answer=answer,
            retrieved_contexts=contexts,
            metadata=kwargs.get("metadata", {}),
        )

        # Log-and-reraise: DSPy swallows the propagated exception into
        # failure_score, so without a module-owned record the operator sees
        # nothing at all. Never substitute a score.
        try:
            result = await self._ragas_evaluator.evaluate_sample(sample, run_id=run_id)

            # RAGASEvaluator stamps its own heuristic path so consumers can tell
            # synthetic scores from judged ones (evaluation.py, "fallback_heuristic").
            # Construction already refuses a statically-unavailable judge; this
            # catches degradation DURING a run (an expired or rate-limited key).
            evaluation_method = (getattr(result, "metadata", None) or {}).get("evaluation_method")
            if evaluation_method == "fallback_heuristic":
                raise RAGASFeedbackUnavailableError(
                    "RAGASEvaluator returned fallback_heuristic scores rather than "
                    "judged ones (the judge degraded mid-run); refusing to feed "
                    "heuristics to GEPA as optimization signal."
                )

            scores = {
                "faithfulness": result.faithfulness or 0.0,
                "answer_relevancy": result.answer_relevancy or 0.0,
                "context_precision": result.context_precision or 0.0,
                "context_recall": result.context_recall or 0.0,
            }
        except Exception as e:
            logger.exception(
                "RAGAS evaluation failed for GEPA feedback; propagating rather "
                "than scoring it: %s (question=%.80r)",
                e,
                question,
            )
            raise

        weighted_score = self._compute_weighted_score(scores)

        feedback = self.config.feedback_template.format(
            faithfulness=scores.get("faithfulness", 0),
            faithfulness_feedback=self._get_feedback_text(
                scores.get("faithfulness", 0), "faithfulness"
            ),
            answer_relevancy=scores.get("answer_relevancy", 0),
            relevancy_feedback=self._get_feedback_text(
                scores.get("answer_relevancy", 0), "relevancy"
            ),
            context_precision=scores.get("context_precision", 0),
            precision_feedback=self._get_feedback_text(
                scores.get("context_precision", 0), "precision"
            ),
            context_recall=scores.get("context_recall", 0),
            recall_feedback=self._get_feedback_text(scores.get("context_recall", 0), "recall"),
            overall_feedback=self._generate_overall_feedback(weighted_score, scores),
        )

        return {
            "score": weighted_score,
            "feedback": feedback,
        }

    async def evaluate_batch(
        self,
        examples: list[dict[str, Any]],
    ) -> list[ScoreWithFeedback]:
        """Evaluate a batch of RAG examples.

        Args:
            examples: List of dicts with 'question', 'answer', 'contexts' keys

        Returns:
            List of ScoreWithFeedback results
        """
        results = []
        for example in examples:
            result = await self.evaluate(
                question=example["question"],
                answer=example["answer"],
                contexts=example.get("contexts", []),
                ground_truth=example.get("ground_truth"),
            )
            results.append(result)
        return results


def create_ragas_metric(
    provider: Optional[RAGASFeedbackProvider] = None,
    agent_name: str = "cognitive_rag",
    weights: Optional[dict[str, float]] = None,
) -> Callable[[Any, Any, Optional[Any]], Coroutine[Any, Any, ScoreWithFeedback]]:
    """Create a GEPA-compatible metric function using RAGAS evaluation.

    This factory creates a metric function that can be passed to GEPA's
    optimizer for RAG-based agents.

    Args:
        provider: Optional pre-configured RAGASFeedbackProvider
        agent_name: Name of the RAG agent
        weights: Optional custom weights for RAGAS metrics

    Returns:
        Metric function compatible with GEPA

    Example:
        >>> metric = create_ragas_metric(agent_name="cognitive_rag")
        >>> optimizer = GEPA(metric=metric, ...)
    """
    if provider is None:
        config = RAGASFeedbackConfig()
        if weights:
            config.weights = weights
        provider = RAGASFeedbackProvider(config=config)

    async def ragas_metric(
        example: Any,
        pred: Any,
        trace: Optional[Any] = None,
    ) -> ScoreWithFeedback:
        """GEPA-compatible metric using RAGAS evaluation.

        Args:
            example: Training example with question and ground truth
            pred: Prediction with answer and contexts
            trace: Optional DSPy trace

        Returns:
            ScoreWithFeedback dict

        Raises:
            Exception: evaluation failures propagate rather than becoming 0.0,
                so a returned score always means the judge ran (#1488).
        """
        # Extract fields from example and prediction
        question = getattr(example, "question", str(example))
        answer = getattr(pred, "answer", str(pred))
        contexts = getattr(pred, "contexts", [])
        ground_truth = getattr(example, "answer", None)

        # Handle different input formats
        if isinstance(example, dict):
            question = example.get("question", question)
            ground_truth = example.get("answer", ground_truth)

        if isinstance(pred, dict):
            answer = pred.get("answer", answer)
            contexts = pred.get("contexts", contexts)

        # Ensure contexts is a list of strings
        if isinstance(contexts, str):
            contexts = [contexts]

        return await provider.evaluate(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

    # Set function metadata for GEPA
    ragas_metric.__name__ = f"ragas_metric_{agent_name}"
    ragas_metric.__doc__ = f"RAGAS feedback metric for {agent_name}"

    return ragas_metric


__all__ = [
    "RAGASFeedbackConfig",
    "RAGASFeedbackProvider",
    "RAGASFeedbackUnavailableError",
    "RAGEvaluationResult",
    "ScoreWithFeedback",
    "create_ragas_metric",
]
