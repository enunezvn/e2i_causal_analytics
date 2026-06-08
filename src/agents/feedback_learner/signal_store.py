"""Durable persistence for feedback_learner training signals (audit F5).

The finalized FeedbackLearnerTrainingSignal is written to the
`dspy_agent_training_signals` table (migration 014) so the self-improver's
own signals survive process restart and can be read back by the optimizer
(Shard 03/05). The CognitiveRAG path already writes other agents' signals to
the same table via src/rag/memory_adapters.py; this closes the gap for the
feedback_learner itself.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional

from .dspy_integration import FeedbackLearnerTrainingSignal

logger = logging.getLogger(__name__)

TABLE = "dspy_agent_training_signals"


def build_signal_record(signal: FeedbackLearnerTrainingSignal) -> Dict[str, Any]:
    """Map a FeedbackLearnerTrainingSignal to a migration-014 row dict.

    Pure: no I/O. Column names match database/memory/014_dspy_training_signals.sql.
    """
    d = signal.to_dict()
    quality_metrics = dict(d.get("quality_metrics") or {})
    # Fold rubric evaluation into quality_metrics so the row stays single-table.
    rubric = d.get("rubric_evaluation") or {}
    if rubric:
        quality_metrics["rubric"] = rubric
    return {
        "source_agent": "feedback_learner",
        "batch_id": signal.batch_id or None,
        "input_context": {
            **(d.get("input_context") or {}),
            # carries the bounded real content (feedback_batch, etc.) added by
            # Shard 04 so the optimizer's signal->example conversion has content.
        },
        "output": d.get("output") or {},
        "quality_metrics": quality_metrics,
        "reward": d.get("reward", 0.0),
        "latency_breakdown": d.get("latency") or {},
        "total_latency_ms": int(signal.total_latency_ms or 0),
        "model_used": signal.model_used or "deterministic",
        "llm_calls": int(signal.llm_calls or 0),
        "total_tokens": int(signal.total_tokens or 0),
        "has_cognitive_context": signal.cognitive_context is not None,
        "is_training_example": True,
    }


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def persist_training_signal(
    signal: FeedbackLearnerTrainingSignal,
    client: Optional[Any] = None,
) -> bool:
    """Insert one finalized signal. Returns True on success, False otherwise.

    Never raises: a DB outage must not fail a learning cycle.
    """
    record = build_signal_record(signal)
    try:
        if client is None:
            from src.memory.services.factories import get_supabase_client

            client = await _maybe_await(get_supabase_client())
        if client is None:
            logger.warning("No Supabase client; feedback_learner signal not persisted")
            return False
        await _maybe_await(client.table(TABLE).insert(record).execute())
        logger.info("Persisted feedback_learner training signal batch=%s", signal.batch_id)
        return True
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        logger.error("Failed to persist feedback_learner training signal: %s", e)
        return False
