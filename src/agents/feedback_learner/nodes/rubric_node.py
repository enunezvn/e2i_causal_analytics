"""
E2I Feedback Learner Agent - Rubric Evaluation Node
Version: 4.2
Purpose: Evaluate agent responses against the E2I causal analytics rubric

This node integrates with the feedback learner pipeline to:
1. Evaluate response quality using AI-as-judge methodology
2. Store evaluation results for learning
3. Trigger improvement actions based on scores

Integration points:
- Called with evaluation context from state
- Stores results in learning_signals table
- Triggers appropriate improvement decisions
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from ..evaluation import (
    EvaluationContext,
    ImprovementDecision,
    RubricEvaluation,
    RubricEvaluator,
)
from ..state import FeedbackLearnerState

logger = logging.getLogger(__name__)


class RubricNode:
    """
    Evaluate agent responses against the E2I causal analytics rubric.

    This node performs AI-as-judge evaluation on responses and stores
    the results for self-improvement learning.

    Attributes:
        evaluator: RubricEvaluator instance
        db_client: Optional database client for storing results
    """

    def __init__(
        self,
        evaluator: Optional[RubricEvaluator] = None,
        db_client: Optional[Any] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize rubric node.

        Args:
            evaluator: Optional RubricEvaluator instance.
                      Created with defaults if not provided.
            db_client: Optional database client for storing evaluation results.
            model: Anthropic model to use for evaluation.
        """
        self.evaluator = evaluator or RubricEvaluator(model=model)
        self.db_client = db_client

    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        """
        Execute rubric evaluation on provided context.

        Args:
            state: Current pipeline state, should contain:
                - rubric_evaluation_context: EvaluationContext to evaluate
                - session_id: Optional session identifier

        Returns:
            Updated state with rubric evaluation results
        """
        start_time = time.time()

        # Check if already failed
        if state.get("status") == "failed":
            return state

        # Get evaluation context from state
        eval_context = state.get("rubric_evaluation_context")

        if not eval_context:
            logger.debug("No rubric evaluation context provided, skipping")
            return {
                **state,
                "rubric_evaluation": None,
                "rubric_latency_ms": int((time.time() - start_time) * 1000),
            }

        try:
            # Convert to EvaluationContext if dict
            if isinstance(eval_context, dict):
                eval_context = EvaluationContext(**eval_context)

            # Run evaluation
            evaluation = await self.evaluator.evaluate(eval_context)

            # Store results if db_client provided
            if self.db_client:
                await self._store_evaluation(evaluation, eval_context)

            # Log result
            logger.info(
                "Rubric evaluation complete: score=%.2f decision=%s patterns=%d",
                evaluation.weighted_score,
                evaluation.decision.value,
                len(evaluation.pattern_flags),
            )

            rubric_latency = int((time.time() - start_time) * 1000)

            return {
                **state,
                "rubric_evaluation": evaluation.model_dump(),
                "rubric_weighted_score": evaluation.weighted_score,
                "rubric_decision": evaluation.decision.value,
                "rubric_pattern_flags": [p.model_dump() for p in evaluation.pattern_flags],
                "rubric_improvement_suggestion": evaluation.improvement_suggestion,
                "rubric_latency_ms": rubric_latency,
            }

        except Exception as e:
            logger.error("Rubric evaluation failed: %s", e)
            return {
                **state,
                "rubric_evaluation": None,
                "rubric_error": str(e),
                "errors": (state.get("errors") or []) + [{"node": "rubric_node", "error": str(e)}],
                "warnings": (state.get("warnings") or []) + [f"Rubric evaluation failed: {e}"],
            }

    async def _store_evaluation(
        self,
        evaluation: RubricEvaluation,
        context: EvaluationContext,
    ) -> None:
        """
        Store evaluation results in learning_signals table.

        Args:
            evaluation: The completed rubric evaluation
            context: The evaluation context
        """
        if not self.db_client:
            return

        try:
            signal_data = {
                "signal_type": "rubric_evaluation",
                "source_agent": "feedback_learner",
                "session_id": context.session_id,
                "rubric_scores": {
                    s.criterion: {"score": s.score, "reasoning": s.reasoning}
                    for s in evaluation.criterion_scores
                },
                "rubric_total": evaluation.weighted_score,
                "improvement_type": self._determine_improvement_type(evaluation),
                "improvement_priority": self._determine_priority(evaluation),
                "improvement_details": {
                    "decision": evaluation.decision.value,
                    "pattern_flags": [p.model_dump() for p in evaluation.pattern_flags],
                    "suggestion": evaluation.improvement_suggestion,
                    "overall_analysis": evaluation.overall_analysis,
                },
                "context_summary": {
                    "user_query": context.user_query[:500],  # Truncate for storage
                    "agents_used": context.agent_names,
                    "messages_evaluated": context.messages_evaluated,
                },
            }

            await self.db_client.table("learning_signals").insert(signal_data).execute()

            logger.debug("Stored rubric evaluation in learning_signals")

        except Exception as e:
            logger.warning("Failed to store rubric evaluation: %s", e)

    def _determine_improvement_type(self, evaluation: RubricEvaluation) -> str:
        """Determine the type of improvement needed based on evaluation."""
        if evaluation.decision == ImprovementDecision.ACCEPTABLE:
            return "none"

        # Check which criteria scored lowest
        lowest_score = min(evaluation.criterion_scores, key=lambda s: s.score)

        # Map criteria to improvement types
        criteria_to_type = {
            "causal_validity": "prompt",
            "actionability": "prompt",
            "evidence_chain": "retrieval",
            "regulatory_awareness": "prompt",
            "uncertainty_communication": "prompt",
        }

        return criteria_to_type.get(lowest_score.criterion, "workflow")

    def _determine_priority(self, evaluation: RubricEvaluation) -> str:
        """Determine improvement priority based on evaluation."""
        if evaluation.decision == ImprovementDecision.ESCALATE:
            return "critical"
        elif evaluation.decision == ImprovementDecision.AUTO_UPDATE:
            return "high"
        elif evaluation.decision == ImprovementDecision.SUGGESTION:
            return "medium"
        else:
            return "low"

    async def evaluate_and_decide(
        self,
        context: EvaluationContext,
    ) -> Dict[str, Any]:
        """
        Convenience method for standalone evaluation.

        Args:
            context: Evaluation context

        Returns:
            Dictionary with evaluation results and decision
        """
        evaluation = await self.evaluator.evaluate(context)

        return {
            "weighted_score": evaluation.weighted_score,
            "criterion_scores": {
                s.criterion: {"score": s.score, "reasoning": s.reasoning}
                for s in evaluation.criterion_scores
            },
            "decision": evaluation.decision.value,
            "is_acceptable": evaluation.is_acceptable,
            "needs_action": evaluation.needs_action,
            "improvement_suggestion": evaluation.improvement_suggestion,
            "pattern_flags": [p.model_dump() for p in evaluation.pattern_flags],
            "overall_analysis": evaluation.overall_analysis,
        }
