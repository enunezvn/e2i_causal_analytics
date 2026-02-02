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

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Union

logger = logging.getLogger(__name__)


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
        """Initialize RAGAS evaluator if available."""
        try:
            from src.rag.evaluation import EvaluationSample, get_ragas_evaluator

            self._ragas_evaluator = get_ragas_evaluator()
            self._evaluation_sample_class = EvaluationSample
            logger.debug("RAGASFeedbackProvider initialized with RAGAS evaluator")
        except ImportError as e:
            logger.warning(f"RAGAS evaluator not available ({e}), using mock evaluation")
            self._ragas_evaluator = None
            self._evaluation_sample_class = None

    @property
    def enabled(self) -> bool:
        """Check if RAGAS evaluation is available."""
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
        """
        try:
            if self._ragas_evaluator and self._evaluation_sample_class:
                # Use real RAGAS evaluation via RAGASEvaluator
                sample = self._evaluation_sample_class(
                    query=question,
                    ground_truth=ground_truth or answer,
                    answer=answer,
                    retrieved_contexts=contexts,
                    metadata=kwargs.get("metadata", {}),
                )
                result = await self._ragas_evaluator.evaluate_sample(
                    sample,
                    run_id=run_id,
                )
                scores = {
                    "faithfulness": result.faithfulness or 0.0,
                    "answer_relevancy": result.answer_relevancy or 0.0,
                    "context_precision": result.context_precision or 0.0,
                    "context_recall": result.context_recall or 0.0,
                }
            else:
                # Mock evaluation for testing
                scores = self._mock_evaluate(question, answer, contexts)

            # Compute weighted score
            weighted_score = self._compute_weighted_score(scores)

            # Generate feedback text
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

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {
                "score": 0.0,
                "feedback": f"Evaluation failed: {str(e)}",
            }

    def _mock_evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, float]:
        """Mock RAGAS evaluation for testing.

        Uses simple heuristics to generate plausible scores.

        Args:
            question: The user question
            answer: The generated answer
            contexts: Retrieved context documents

        Returns:
            Dictionary of metric scores
        """
        # Simple heuristics for mock evaluation
        answer_len = len(answer)
        context_len = sum(len(c) for c in contexts)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        # Faithfulness: higher if answer is shorter relative to context
        faithfulness = min(1.0, context_len / (answer_len + 1) * 0.1)

        # Relevancy: higher if answer words overlap with question
        overlap = len(question_words & answer_words) / len(question_words) if question_words else 0
        relevancy = min(1.0, overlap * 2)

        # Precision: higher if contexts are not too long
        precision = min(1.0, 1000 / (context_len / len(contexts) + 1)) if contexts else 0.5

        # Recall: higher if more contexts retrieved
        recall = min(1.0, len(contexts) / 3)

        return {
            "faithfulness": max(0.3, min(0.9, faithfulness + 0.4)),
            "answer_relevancy": max(0.3, min(0.9, relevancy + 0.3)),
            "context_precision": max(0.3, min(0.9, precision + 0.3)),
            "context_recall": max(0.3, min(0.9, recall + 0.3)),
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
) -> Callable[[Any, Any, Any], ScoreWithFeedback]:
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
        """
        try:
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

        except Exception as e:
            logger.error(f"RAGAS metric evaluation failed: {e}")
            return {
                "score": 0.0,
                "feedback": f"Metric evaluation failed for {agent_name}: {str(e)}",
            }

    # Set function metadata for GEPA
    ragas_metric.__name__ = f"ragas_metric_{agent_name}"
    ragas_metric.__doc__ = f"RAGAS feedback metric for {agent_name}"

    return ragas_metric


__all__ = [
    "RAGASFeedbackConfig",
    "RAGASFeedbackProvider",
    "RAGEvaluationResult",
    "ScoreWithFeedback",
    "create_ragas_metric",
]
