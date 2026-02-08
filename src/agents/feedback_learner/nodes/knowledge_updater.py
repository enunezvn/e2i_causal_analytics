"""
E2I Feedback Learner Agent - Knowledge Updater Node
Version: 4.2
Purpose: Apply learnings to knowledge bases
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

from ..state import FeedbackLearnerState, KnowledgeUpdate

logger = logging.getLogger(__name__)


class KnowledgeUpdaterNode:
    """
    Apply learnings to knowledge bases.
    Updates organizational knowledge.
    """

    def __init__(self, knowledge_stores: Optional[Dict[str, Any]] = None):
        """
        Initialize knowledge updater.

        Args:
            knowledge_stores: Dictionary of knowledge store instances by type
        """
        self.stores = knowledge_stores or {}

    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        """Execute knowledge updates."""
        start_time = time.time()

        # Check if already failed
        if state.get("status") == "failed":
            return state

        try:
            recommendations = state.get("learning_recommendations") or []

            # Generate proposed updates
            proposed_updates = self._generate_updates(
                cast(List[Dict[str, Any]], recommendations)
            )

            # Apply updates (with validation)
            applied = []
            for update in proposed_updates:
                success = await self._apply_update(update)
                if success:
                    applied.append(update["update_id"])

            # Generate summary
            summary = self._generate_summary(state, proposed_updates, applied)

            update_time = int((time.time() - start_time) * 1000)
            total_time = (
                state.get("collection_latency_ms", 0)
                + state.get("analysis_latency_ms", 0)
                + state.get("extraction_latency_ms", 0)
                + update_time
            )

            logger.info(
                f"Knowledge update complete: applied {len(applied)} of {len(proposed_updates)} updates"
            )

            return {
                **state,
                "proposed_updates": proposed_updates,
                "applied_updates": applied,
                "learning_summary": summary,
                "update_latency_ms": update_time,
                "total_latency_ms": total_time,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Knowledge update failed: {e}")
            return {
                **state,
                "errors": [{"node": "knowledge_updater", "error": str(e)}],
                "status": "failed",
            }

    def _generate_updates(self, recommendations: List[Dict[str, Any]]) -> List[KnowledgeUpdate]:
        """Generate knowledge updates from recommendations."""
        updates: List[KnowledgeUpdate] = []
        now = datetime.now(timezone.utc).isoformat()

        for rec in recommendations:
            category = rec.get("category", "")
            affected_agents = rec.get("affected_agents", ["unknown"])
            primary_agent = affected_agents[0] if affected_agents else "unknown"

            if category == "data_update":
                updates.append(
                    KnowledgeUpdate(
                        update_id=f"U_{rec.get('recommendation_id', 'R?')}",
                        knowledge_type="baseline",
                        key=primary_agent,
                        old_value=None,
                        new_value=rec.get("proposed_change"),
                        justification=rec.get("description", ""),
                        effective_date=now,
                    )
                )

            elif category == "config_change":
                updates.append(
                    KnowledgeUpdate(
                        update_id=f"U_{rec.get('recommendation_id', 'R?')}",
                        knowledge_type="agent_config",
                        key=primary_agent,
                        old_value=None,
                        new_value=rec.get("proposed_change"),
                        justification=rec.get("description", ""),
                        effective_date=now,
                    )
                )

            elif category == "prompt_update":
                updates.append(
                    KnowledgeUpdate(
                        update_id=f"U_{rec.get('recommendation_id', 'R?')}",
                        knowledge_type="prompt",
                        key=primary_agent,
                        old_value=None,
                        new_value=rec.get("proposed_change"),
                        justification=rec.get("description", ""),
                        effective_date=now,
                    )
                )

            elif category == "threshold":
                updates.append(
                    KnowledgeUpdate(
                        update_id=f"U_{rec.get('recommendation_id', 'R?')}",
                        knowledge_type="threshold",
                        key=primary_agent,
                        old_value=None,
                        new_value=rec.get("proposed_change"),
                        justification=rec.get("description", ""),
                        effective_date=now,
                    )
                )

        return updates

    async def _apply_update(self, update: KnowledgeUpdate) -> bool:
        """Apply a single update to knowledge store."""
        knowledge_type = update["knowledge_type"]

        if knowledge_type not in self.stores:
            logger.debug(f"No store available for knowledge type: {knowledge_type}")
            return False

        store = self.stores[knowledge_type]

        try:
            await store.update(
                key=update["key"],
                value=update["new_value"],
                justification=update["justification"],
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to apply update {update['update_id']}: {e}")
            return False

    def _generate_summary(
        self,
        state: FeedbackLearnerState,
        proposed: List[KnowledgeUpdate],
        applied: List[str],
    ) -> str:
        """Generate learning summary."""
        feedback_count = len(state.get("feedback_items") or [])
        pattern_count = len(state.get("detected_patterns") or [])
        rec_count = len(state.get("learning_recommendations") or [])

        parts = [
            "Learning cycle complete.",
            f"Processed {feedback_count} feedback items.",
            f"Detected {pattern_count} patterns.",
            f"Generated {rec_count} recommendations.",
            f"Applied {len(applied)} of {len(proposed)} proposed updates.",
        ]

        # Add top priorities if available
        priorities = state.get("priority_improvements") or []
        if priorities:
            parts.append(f"Top priority: {priorities[0]}")

        return " ".join(parts)
