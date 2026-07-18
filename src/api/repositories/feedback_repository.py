"""Supabase persistence for the feedback route stores (M2).

Replaces the process-local _learning_store / _patterns_store / _updates_store /
_feedback_store dicts in src/api/routes/feedback.py. Same canonical pattern as
GapsRepository (service-role client, asyncio.to_thread(execute), JSONB payload).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from src.api.routes.feedback import (
    DetectedPattern,
    FeedbackItem,
    KnowledgeUpdate,
    LearningResponse,
)

logger = logging.getLogger(__name__)

_BATCHES = "feedback_learning_batches"
_PATTERNS = "feedback_patterns"
_UPDATES = "feedback_knowledge_updates"
_ITEMS = "feedback_items"


class FeedbackRepository:
    """Thin async repository over the four feedback tables."""

    def __init__(self, client: Any = None) -> None:
        if client is None:
            from src.memory.services.factories import get_supabase_client

            client = get_supabase_client()
        self._client = client

    # ---- learning batches (_learning_store) --------------------------------
    async def upsert_batch(self, response: LearningResponse) -> None:
        row = {
            "batch_id": response.batch_id,
            "status": response.status.value,
            "payload": response.model_dump(mode="json"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        query = self._client.table(_BATCHES).upsert(row, on_conflict="batch_id")
        await asyncio.to_thread(query.execute)

    async def get_batch(self, batch_id: str) -> Optional[LearningResponse]:
        query = self._client.table(_BATCHES).select("payload").eq("batch_id", batch_id).limit(1)
        result = await asyncio.to_thread(query.execute)
        rows = (result.data) or []
        if not rows:
            return None
        return LearningResponse.model_validate(rows[0]["payload"])

    async def count_recent_and_last(self) -> List[LearningResponse]:
        query = self._client.table(_BATCHES).select("payload")
        result = await asyncio.to_thread(query.execute)
        rows = (result.data) or []
        return [LearningResponse.model_validate(r["payload"]) for r in rows]

    # ---- patterns (_patterns_store) ----------------------------------------
    async def upsert_pattern(self, pattern: DetectedPattern) -> None:
        row = {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type.value,
            "severity": pattern.severity.value,
            "payload": pattern.model_dump(mode="json"),
        }
        query = self._client.table(_PATTERNS).upsert(row, on_conflict="pattern_id")
        await asyncio.to_thread(query.execute)

    async def list_patterns(self) -> List[DetectedPattern]:
        query = self._client.table(_PATTERNS).select("payload, created_at")
        result = await asyncio.to_thread(query.execute)
        rows = (result.data) or []
        patterns = []
        for r in rows:
            payload = dict(r["payload"])
            # #1244: legacy payloads carry no detected_at — backfill from the
            # row's created_at (DB default now() at insert) so the API always
            # reports when the pattern was detected. Payload-carried values
            # (stamped by _convert_patterns since #1256) win.
            if not payload.get("detected_at") and r.get("created_at"):
                payload["detected_at"] = r["created_at"]
            patterns.append(DetectedPattern.model_validate(payload))
        return patterns

    # ---- knowledge updates (_updates_store) --------------------------------
    async def upsert_update(self, update: KnowledgeUpdate) -> None:
        row = {
            "update_id": update.update_id,
            "update_type": update.update_type.value,
            "status": update.status.value,
            "target_agent": update.target_agent,
            "payload": update.model_dump(mode="json"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        query = self._client.table(_UPDATES).upsert(row, on_conflict="update_id")
        await asyncio.to_thread(query.execute)

    async def get_update(self, update_id: str) -> Optional[KnowledgeUpdate]:
        query = self._client.table(_UPDATES).select("payload").eq("update_id", update_id).limit(1)
        result = await asyncio.to_thread(query.execute)
        rows = (result.data) or []
        if not rows:
            return None
        return KnowledgeUpdate.model_validate(rows[0]["payload"])

    async def list_updates(self) -> List[KnowledgeUpdate]:
        query = self._client.table(_UPDATES).select("payload")
        result = await asyncio.to_thread(query.execute)
        rows = (result.data) or []
        return [KnowledgeUpdate.model_validate(r["payload"]) for r in rows]

    # ---- raw items (_feedback_store) ---------------------------------------
    async def append_item(self, item: FeedbackItem) -> None:
        row = {
            "feedback_id": item.feedback_id,
            "source_agent": item.source_agent,
            "payload": item.model_dump(mode="json"),
        }
        query = self._client.table(_ITEMS).upsert(row, on_conflict="feedback_id")
        await asyncio.to_thread(query.execute)
