"""
Classification Log Repository.

Persists 4-stage ClassificationPipeline decisions to ``classification_logs``
(database/ml/013_tool_composer_tables.sql) for routing-quality analysis —
the shadow-vs-active agreement data that gates ORCHESTRATOR_CLASSIFIER_MODE
promotion.

Writes are strictly fail-open: a failed insert logs a warning and returns
None; it must never fail or delay the chat turn.
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.agents.orchestrator.classifier.schemas import ClassificationResult
from src.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ClassificationLogRepository(BaseRepository):
    """Repository for the ``classification_logs`` audit table."""

    table_name = "classification_logs"
    model_class = None
    id_column = "classification_id"

    async def record_classification(
        self,
        *,
        query_text: str,
        result: ClassificationResult,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Insert one classification decision; fail-open, never raises.

        Args:
            query_text: The raw user query that was classified.
            result: The pipeline's ClassificationResult (with ``stages``
                populated for the JSONB stage columns).
            session_id: Chat session id (``user_id~uuid`` format, ≤100 chars).
            user_id: Authenticated user id.

        Returns:
            The inserted row, or None when there is no client / insert failed.
        """
        if not self.client:
            return None

        dump = result.model_dump(mode="json")
        stages = dump.get("stages") or {}

        data = {
            "query_text": query_text,
            "query_hash": hashlib.sha256(query_text.encode()).hexdigest(),
            "routing_pattern": dump["routing_pattern"],
            "target_agents": dump.get("target_agents") or [],
            "confidence": dump.get("confidence", 0.0),
            "features_extracted": stages.get("features") or {},
            "domain_mapping": stages.get("domain_mapping") or {},
            "dependency_analysis": stages.get("dependency_analysis") or {},
            # Dependency dumps use field names (from_id/to_id), not aliases.
            "sub_questions": dump.get("sub_questions") or [],
            "dependencies": dump.get("dependencies") or [],
            "used_llm_layer": dump.get("used_llm_layer", False),
            "classification_latency_ms": dump.get("classification_latency_ms", 0.0),
            "session_id": session_id[:100] if session_id else None,
            "user_id": user_id[:100] if user_id else None,
            "is_followup": dump.get("is_followup", False),
        }
        data = {k: v for k, v in data.items() if v is not None}

        try:
            insert_result = await self.client.table(self.table_name).insert(data).execute()
            return insert_result.data[0] if insert_result.data else None
        except Exception as e:
            logger.warning("Failed to record classification log (fail-open): %s", e)
            return None

    async def fetch_unlabeled(
        self, *, lookback_days: int = 30, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Rows still awaiting a routing-correctness label (#1341 Phase 1).

        Returns recent rows where ``was_correct IS NULL``, newest first.
        Rows already visited by the judge (``feedback_notes`` set) are
        included — late explicit feedback may still auto-label them; the
        labeler itself excludes visited rows from re-judging.
        """
        if not self.client:
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        try:
            result = await (
                self.client.table(self.table_name)
                .select(
                    "classification_id,query_text,routing_pattern,target_agents,"
                    "confidence,session_id,user_id,created_at,feedback_notes"
                )
                .is_("was_correct", "null")
                .gte("created_at", cutoff)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return list(result.data or [])
        except Exception as e:
            logger.warning("Failed to fetch unlabeled classifications (fail-open): %s", e)
            return []

    async def fetch_for_metrics(
        self, *, lookback_days: int = 30, limit: int = 2000
    ) -> List[Dict[str, Any]]:
        """Recent rows (labeled + unlabeled) for Phase-2 telemetry (#1341).

        Returns the columns ``compute_run_metrics`` needs. Unlike
        ``fetch_unlabeled`` this includes labeled rows — the aggregation reports
        accuracy over the labeled subset and awaiting-feedback backlog together.
        """
        if not self.client:
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        try:
            result = await (
                self.client.table(self.table_name)
                .select(
                    "routing_pattern,target_agents,confidence,used_llm_layer,"
                    "was_correct,correct_pattern,feedback_notes,created_at"
                )
                .gte("created_at", cutoff)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return list(result.data or [])
        except Exception as e:
            logger.warning("Failed to fetch classifications for metrics (fail-open): %s", e)
            return []

    async def record_metrics_snapshot(
        self, metrics: Dict[str, Any], *, task_id: Optional[str], window_days: int
    ) -> bool:
        """Persist one Phase-2 telemetry snapshot to routing_classifier_metrics.

        Strictly fail-open: if the table is absent (migration 032 not yet
        applied) or the insert fails, log a warning and return False — the
        labeler still logs the metrics in its run summary. Never raises.
        """
        if not self.client:
            return False
        ab = metrics.get("abstention") or {}
        data = {
            "task_id": (task_id or "")[:100] or None,
            "window_days": window_days,
            "total": metrics.get("total", 0),
            "labeled": metrics.get("labeled", 0),
            "overall_accuracy_pct": metrics.get("overall_accuracy_pct"),
            "engagement_rate": metrics.get("engagement_rate"),
            "active_floor": metrics.get("active_floor"),
            "llm_layer_share": metrics.get("llm_layer_share"),
            "abstention_total": ab.get("total", 0),
            "abstention_correct": ab.get("judged_correct", 0),
            "abstention_incorrect": ab.get("judged_incorrect", 0),
            "per_pattern": metrics.get("per_pattern") or {},
            "label_sources": metrics.get("label_sources") or {},
        }
        data = {k: v for k, v in data.items() if v is not None}
        try:
            result = await self.client.table("routing_classifier_metrics").insert(data).execute()
            return bool(result.data)
        except Exception as e:
            logger.warning("Failed to record routing metrics snapshot (fail-open): %s", e)
            return False

    async def apply_label(
        self,
        classification_id: str,
        *,
        was_correct: Optional[bool] = None,
        correct_pattern: Optional[str] = None,
        feedback_notes: Optional[str] = None,
    ) -> bool:
        """Write a routing-correctness label onto one row; fail-open.

        ``was_correct=None`` with ``feedback_notes`` set records a judge
        abstention (row stays awaiting_feedback in v_classification_accuracy
        but is marked visited).
        """
        if not self.client:
            return False
        update: Dict[str, Any] = {}
        if was_correct is not None:
            update["was_correct"] = was_correct
        if correct_pattern is not None:
            update["correct_pattern"] = correct_pattern
        if feedback_notes is not None:
            update["feedback_notes"] = feedback_notes
        if not update:
            return False
        try:
            result = await (
                self.client.table(self.table_name)
                .update(update)
                .eq(self.id_column, classification_id)
                .execute()
            )
            return bool(result.data)
        except Exception as e:
            logger.warning("Failed to apply classification label (fail-open): %s", e)
            return False


def get_classification_log_repository(supabase_client=None) -> ClassificationLogRepository:
    """Get a ClassificationLogRepository instance."""
    return ClassificationLogRepository(supabase_client)
