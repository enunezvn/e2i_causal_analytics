"""
Learning-signals feedback store.

Adapts the ``learning_signals`` table — per-conversation reward evaluations the
cognitive workflow writes for every real chat turn (``SignalCollector`` in
``src/rag/cognitive_backends.py``, signals built in
``cognitive_rag_dspy._collect_training_signals``) — into the feedback-item
shape ``FeedbackLearnerAgent``'s collector expects.

Why this exists: the learner's only other real source is explicit chat thumbs
(``chatbot_message_feedback``), whose volume depends on users clicking. The
reward stream flows on EVERY chat turn, so cycles have real material even at
low thumbs volume. Synthetic rows (``is_synthetic``) are excluded — the learner
must never learn from showcase substrate.
"""

from typing import Any, Dict, List, Optional

from src.repositories.base import BaseRepository


class LearningSignalsFeedbackStore(BaseRepository):
    """Read-only ``get_feedback`` view over ``learning_signals`` dspy rows.

    Exposes the same contract as ``ChatbotFeedbackRepository.get_feedback``
    (``id``/``timestamp``/``rating``/``agent``/``query``/``response``/
    ``metadata``), so the two can be composed behind one feedback_store.
    """

    table_name = "learning_signals"
    model_class = None

    async def get_feedback(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        agents: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Return real cognitive-workflow reward signals as feedback items.

        ``rating`` carries the reward mapped onto the analyzer's 1–5 scale
        (``pattern_analyzer._rating_to_numeric`` passes numerics through
        unchanged and flags ``avg < 3.0`` as a low-ratings pattern — raw 0..1
        rewards would read as abysmal 1–5 ratings and fabricate that pattern on
        every cycle). The raw 0..1 reward is preserved in ``metadata.reward``.
        ``agent`` attribution: ``type=agent`` signals are credited to the first
        routed agent (the primary responder); ``investigator``/``summarizer``
        signals are orchestrator components, credited as ``cognitive_<type>``.
        """
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("signal_id, created_at, signal_details")
            .eq("is_synthetic", False)
            .eq("signal_details->>domain_signal", "dspy_signal")
        )
        if start_time:
            query = query.gte("created_at", start_time)
        if end_time:
            query = query.lte("created_at", end_time)

        result = await query.order("created_at", desc=True).limit(limit).execute()
        rows = result.data or []

        mapped: List[Dict[str, Any]] = []
        for r in rows:
            details = r.get("signal_details") or {}
            reward = details.get("reward")
            if not isinstance(reward, (int, float)):
                continue  # not a graded signal — nothing to learn from

            component = str(details.get("type") or "unknown")
            meta = details.get("metadata") or {}
            routed = meta.get("routed_agents") or []
            if component == "agent" and routed:
                agent = str(routed[0])
            elif component == "agent":
                agent = "orchestrator"
            else:
                agent = f"cognitive_{component}"

            if agents and agent not in agents:
                continue

            # reward 0..1 → rating 1..5 (clamped): same scale thumbs map onto
            # (thumbs_up→5.0, thumbs_down→1.0), so the analyzer's avg<3.0
            # low-ratings gate reads both sources coherently.
            rating_1to5 = max(1.0, min(5.0, 1.0 + 4.0 * float(reward)))
            mapped.append(
                {
                    "id": str(r.get("signal_id", "")),
                    "timestamp": r.get("created_at", ""),
                    "rating": rating_1to5,
                    "agent": agent,
                    "query": str(details.get("query") or ""),
                    "response": str(details.get("response") or ""),
                    "metadata": {
                        "source": "learning_signals",
                        "signal_component": component,
                        "conversation_id": meta.get("conversation_id"),
                        "routed_agents": routed,
                        "reward": float(reward),
                        "workflow_feedback": details.get("feedback"),
                    },
                }
            )
        return mapped


def get_learning_signals_feedback_store(
    supabase_client=None,
) -> LearningSignalsFeedbackStore:
    """Get a LearningSignalsFeedbackStore instance."""
    return LearningSignalsFeedbackStore(supabase_client)
