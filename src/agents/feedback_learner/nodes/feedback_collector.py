"""
E2I Feedback Learner Agent - Feedback Collector Node
Version: 4.2
Purpose: Collect feedback from various sources
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..state import FeedbackLearnerState, FeedbackItem, FeedbackSummary

logger = logging.getLogger(__name__)


class FeedbackCollectorNode:
    """
    Collect feedback from various sources.
    Prepares data for pattern analysis.
    """

    def __init__(
        self,
        feedback_store: Optional[Any] = None,
        outcome_store: Optional[Any] = None,
    ):
        """
        Initialize feedback collector.

        Args:
            feedback_store: Store for user feedback
            outcome_store: Store for outcome data
        """
        self.feedback_store = feedback_store
        self.outcome_store = outcome_store

    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        """Execute feedback collection."""
        start_time = time.time()

        # Check if already failed
        if state.get("status") == "failed":
            return state

        try:
            # Collect feedback from all sources
            user_feedback = await self._collect_user_feedback(state)
            outcome_feedback = await self._collect_outcome_feedback(state)
            implicit_feedback = await self._collect_implicit_feedback(state)

            # Combine all feedback
            all_feedback = user_feedback + outcome_feedback + implicit_feedback

            if not all_feedback:
                return {
                    **state,
                    "feedback_items": [],
                    "feedback_summary": self._generate_summary([]),
                    "collection_latency_ms": int((time.time() - start_time) * 1000),
                    "status": "analyzing",
                    "warnings": ["No feedback items collected"],
                }

            # Generate summary
            summary = self._generate_summary(all_feedback)

            collection_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Collected {len(all_feedback)} feedback items: "
                f"user={len(user_feedback)}, outcome={len(outcome_feedback)}"
            )

            return {
                **state,
                "feedback_items": all_feedback,
                "feedback_summary": summary,
                "collection_latency_ms": collection_time,
                "status": "analyzing",
            }

        except Exception as e:
            logger.error(f"Feedback collection failed: {e}")
            return {
                **state,
                "errors": [{"node": "feedback_collector", "error": str(e)}],
                "status": "failed",
            }

    async def _collect_user_feedback(
        self, state: FeedbackLearnerState
    ) -> List[FeedbackItem]:
        """Collect explicit user feedback (ratings, corrections)."""
        if not self.feedback_store:
            return []

        try:
            raw_feedback = await self.feedback_store.get_feedback(
                start_time=state.get("time_range_start"),
                end_time=state.get("time_range_end"),
                agents=state.get("focus_agents"),
            )

            items = []
            for fb in raw_feedback:
                items.append(
                    FeedbackItem(
                        feedback_id=fb.get("id", ""),
                        timestamp=fb.get("timestamp", ""),
                        feedback_type="rating" if "rating" in fb else "correction",
                        source_agent=fb.get("agent", "unknown"),
                        query=fb.get("query", ""),
                        agent_response=fb.get("response", ""),
                        user_feedback=fb.get("rating") or fb.get("correction"),
                        metadata=fb.get("metadata", {}),
                    )
                )

            return items
        except Exception as e:
            logger.warning(f"Failed to collect user feedback: {e}")
            return []

    async def _collect_outcome_feedback(
        self, state: FeedbackLearnerState
    ) -> List[FeedbackItem]:
        """Collect outcome-based feedback (predictions vs actuals)."""
        if not self.outcome_store:
            return []

        try:
            outcomes = await self.outcome_store.get_outcomes(
                start_time=state.get("time_range_start"),
                end_time=state.get("time_range_end"),
            )

            items = []
            for outcome in outcomes:
                predicted = outcome.get("prediction", 0)
                actual = outcome.get("actual", 0)

                items.append(
                    FeedbackItem(
                        feedback_id=f"outcome_{outcome.get('id', '')}",
                        timestamp=outcome.get("timestamp", ""),
                        feedback_type="outcome",
                        source_agent=outcome.get("agent", "unknown"),
                        query=outcome.get("original_query", ""),
                        agent_response=str(predicted),
                        user_feedback={
                            "predicted": predicted,
                            "actual": actual,
                            "error": actual - predicted,
                        },
                        metadata=outcome.get("metadata", {}),
                    )
                )

            return items
        except Exception as e:
            logger.warning(f"Failed to collect outcome feedback: {e}")
            return []

    async def _collect_implicit_feedback(
        self, state: FeedbackLearnerState
    ) -> List[FeedbackItem]:
        """Collect implicit feedback from user behavior."""
        # Could include: follow-up questions (confusion), session abandonment, etc.
        # For now, returns empty list
        return []

    def _generate_summary(self, feedback: List[FeedbackItem]) -> FeedbackSummary:
        """Generate feedback summary statistics."""
        if not feedback:
            return FeedbackSummary(
                total_count=0,
                by_type={},
                by_agent={},
                average_rating=None,
                rating_count=0,
            )

        by_type: Dict[str, int] = {}
        by_agent: Dict[str, int] = {}
        ratings: List[float] = []

        for item in feedback:
            # By type
            fb_type = item["feedback_type"]
            by_type[fb_type] = by_type.get(fb_type, 0) + 1

            # By agent
            agent = item["source_agent"]
            by_agent[agent] = by_agent.get(agent, 0) + 1

            # Ratings
            if fb_type == "rating" and isinstance(
                item["user_feedback"], (int, float)
            ):
                ratings.append(float(item["user_feedback"]))

        return FeedbackSummary(
            total_count=len(feedback),
            by_type=by_type,
            by_agent=by_agent,
            average_rating=sum(ratings) / len(ratings) if ratings else None,
            rating_count=len(ratings),
        )
