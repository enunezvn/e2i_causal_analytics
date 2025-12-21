"""
E2I Agentic Memory - Working Memory (Redis + LangGraph)
Short-term memory for current context, scratchpad, and immediate message history.

Technology: Redis + LangGraph MemorySaver checkpointer

Features:
- Session management with TTL (default: 24 hours)
- Message history with configurable window size
- Evidence board for multi-hop investigation
- E2I entity context (brand, region, patient/HCP IDs)
- LangGraph checkpointer integration for workflow state

Usage:
    from src.memory.working_memory import get_working_memory, get_langgraph_checkpointer

    # Get working memory instance
    wm = get_working_memory()

    # Create a session
    session_id = await wm.create_session(user_id="user123")

    # Add messages
    await wm.add_message(session_id, "user", "Why did TRx drop?")

    # Get LangGraph checkpointer for workflow
    checkpointer = get_langgraph_checkpointer()
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.memory.services.config import get_config
from src.memory.services.factories import get_redis_client

logger = logging.getLogger(__name__)


class RedisWorkingMemory:
    """
    Redis-based working memory with LangGraph MemorySaver integration.

    Stores:
    - Session state (user context, preferences, filters)
    - Message history (conversation with configurable window)
    - Evidence board (items collected during investigation)
    - E2I entity context (brand, region, patient/HCP IDs)
    """

    def __init__(self):
        """Initialize working memory with configuration."""
        self._config = get_config()
        self._working_config = self._config.working
        self._client = None
        self._checkpointer = None

    @property
    def session_prefix(self) -> str:
        """Redis key prefix for sessions."""
        return self._working_config.session_prefix

    @property
    def evidence_prefix(self) -> str:
        """Redis key prefix for evidence."""
        return self._working_config.evidence_prefix

    @property
    def ttl_seconds(self) -> int:
        """Session TTL in seconds."""
        return self._working_config.ttl_seconds

    @property
    def context_window_messages(self) -> int:
        """Maximum messages to keep in context window."""
        return self._working_config.context_window_messages

    async def get_client(self):
        """Lazy Redis client initialization."""
        if self._client is None:
            self._client = get_redis_client()
        return self._client

    def get_langgraph_checkpointer(self):
        """
        Get LangGraph checkpointer backed by Redis.

        Returns:
            RedisSaver: LangGraph checkpointer for workflow state persistence
        """
        if self._checkpointer is None:
            try:
                from langgraph.checkpoint.redis import RedisSaver

                redis_url = os.environ.get("REDIS_URL", "redis://localhost:6382")
                self._checkpointer = RedisSaver.from_conn_string(redis_url)
                logger.info("LangGraph RedisSaver checkpointer initialized")
            except ImportError:
                logger.warning("langgraph-checkpoint-redis not installed, using memory checkpointer")
                from langgraph.checkpoint.memory import MemorySaver

                self._checkpointer = MemorySaver()
        return self._checkpointer

    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================

    async def create_session(
        self,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new working memory session.

        Args:
            user_id: Optional user identifier
            initial_context: Optional initial context with:
                - preferences: User preferences dict
                - filters: Active filter settings
                - brand: Active E2I brand
                - region: Active E2I region

        Returns:
            str: New session ID (UUID)
        """
        redis = await self.get_client()
        session_id = str(uuid.uuid4())
        session_key = f"{self.session_prefix}{session_id}"

        context = initial_context or {}
        session_data = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity_at": datetime.now(timezone.utc).isoformat(),
            "message_count": "0",
            "current_phase": "init",
            "user_preferences": json.dumps(context.get("preferences", {})),
            "active_filters": json.dumps(context.get("filters", {})),
            "active_brand": context.get("brand"),
            "active_region": context.get("region"),
        }

        # Filter out None values
        session_data = {k: v for k, v in session_data.items() if v is not None}

        await redis.hset(session_key, mapping=session_data)
        await redis.expire(session_key, self.ttl_seconds)

        logger.debug(f"Created session {session_id} for user {user_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data.

        Args:
            session_id: Session identifier

        Returns:
            Dict with session data or None if not found
        """
        redis = await self.get_client()
        session_key = f"{self.session_prefix}{session_id}"

        data = await redis.hgetall(session_key)
        if not data:
            return None

        # Deserialize JSON fields
        if data.get("user_preferences"):
            data["user_preferences"] = json.loads(data["user_preferences"])
        if data.get("active_filters"):
            data["active_filters"] = json.loads(data["active_filters"])
        if data.get("message_count"):
            data["message_count"] = int(data["message_count"])

        return data

    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update session fields.

        Args:
            session_id: Session identifier
            updates: Fields to update
        """
        redis = await self.get_client()
        session_key = f"{self.session_prefix}{session_id}"

        # Serialize complex fields
        for key in ["user_preferences", "active_filters", "active_entities"]:
            if key in updates and isinstance(updates[key], dict):
                updates[key] = json.dumps(updates[key])

        # Convert non-string values
        for key, value in updates.items():
            if isinstance(value, (int, float)):
                updates[key] = str(value)

        updates["last_activity_at"] = datetime.now(timezone.utc).isoformat()

        await redis.hset(session_key, mapping=updates)
        await redis.expire(session_key, self.ttl_seconds)

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all associated data.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session was deleted
        """
        redis = await self.get_client()

        # Delete all session-related keys
        keys_to_delete = [
            f"{self.session_prefix}{session_id}",
            f"{self.session_prefix}{session_id}:messages",
            f"{self.evidence_prefix}{session_id}",
        ]

        deleted_count = 0
        for key in keys_to_delete:
            deleted_count += await redis.delete(key)

        logger.debug(f"Deleted session {session_id} ({deleted_count} keys)")
        return deleted_count > 0

    # ========================================================================
    # E2I CONTEXT MANAGEMENT
    # ========================================================================

    async def set_e2i_context(
        self,
        session_id: str,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        patient_ids: Optional[List[str]] = None,
        hcp_ids: Optional[List[str]] = None
    ) -> None:
        """
        Set E2I entity context for the session.

        Args:
            session_id: Session identifier
            brand: Active brand (Remibrutinib, Fabhalta, Kisqali)
            region: Active region (northeast, south, midwest, west)
            patient_ids: List of active patient IDs
            hcp_ids: List of active HCP IDs
        """
        updates = {}
        if brand:
            updates["active_brand"] = brand
        if region:
            updates["active_region"] = region
        if patient_ids:
            updates["active_patient_ids"] = json.dumps(patient_ids)
        if hcp_ids:
            updates["active_hcp_ids"] = json.dumps(hcp_ids)

        if updates:
            await self.update_session(session_id, updates)

    async def get_e2i_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get E2I entity context from session.

        Args:
            session_id: Session identifier

        Returns:
            Dict with E2I context (brand, region, patient_ids, hcp_ids)
        """
        session = await self.get_session(session_id)
        if not session:
            return {}

        context = {
            "brand": session.get("active_brand"),
            "region": session.get("active_region"),
        }

        if session.get("active_patient_ids"):
            context["patient_ids"] = json.loads(session["active_patient_ids"])
        if session.get("active_hcp_ids"):
            context["hcp_ids"] = json.loads(session["active_hcp_ids"])

        return context

    # ========================================================================
    # MESSAGE HISTORY
    # ========================================================================

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add message to conversation history.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata dict
        """
        redis = await self.get_client()
        messages_key = f"{self.session_prefix}{session_id}:messages"

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": json.dumps(metadata or {})
        }

        await redis.rpush(messages_key, json.dumps(message))
        await redis.expire(messages_key, self.ttl_seconds)

        # Trim to context window size
        await redis.ltrim(messages_key, -self.context_window_messages, -1)

        # Increment message count
        session_key = f"{self.session_prefix}{session_id}"
        await redis.hincrby(session_key, "message_count", 1)

    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages from conversation.

        Args:
            session_id: Session identifier
            limit: Maximum messages to return (None = all in window)

        Returns:
            List of message dicts with role, content, timestamp, metadata
        """
        redis = await self.get_client()
        messages_key = f"{self.session_prefix}{session_id}:messages"

        if limit:
            messages = await redis.lrange(messages_key, -limit, -1)
        else:
            messages = await redis.lrange(messages_key, 0, -1)

        result = []
        for m in messages:
            msg = json.loads(m)
            # Deserialize nested metadata
            if msg.get("metadata"):
                msg["metadata"] = json.loads(msg["metadata"])
            result.append(msg)

        return result

    async def clear_messages(self, session_id: str) -> None:
        """
        Clear all messages for a session.

        Args:
            session_id: Session identifier
        """
        redis = await self.get_client()
        messages_key = f"{self.session_prefix}{session_id}:messages"
        await redis.delete(messages_key)

        # Reset message count
        session_key = f"{self.session_prefix}{session_id}"
        await redis.hset(session_key, "message_count", "0")

    # ========================================================================
    # EVIDENCE BOARD (for multi-hop investigation)
    # ========================================================================

    async def append_evidence(
        self,
        session_id: str,
        evidence: Dict[str, Any]
    ) -> None:
        """
        Append evidence item to evidence board.

        Args:
            session_id: Session identifier
            evidence: Evidence item with source, content, relevance, etc.
        """
        redis = await self.get_client()
        evidence_key = f"{self.evidence_prefix}{session_id}"

        # Add timestamp if not present
        if "timestamp" not in evidence:
            evidence["timestamp"] = datetime.now(timezone.utc).isoformat()

        await redis.rpush(evidence_key, json.dumps(evidence))
        await redis.expire(evidence_key, self.ttl_seconds)

    async def get_evidence_trail(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all evidence from current investigation.

        Args:
            session_id: Session identifier

        Returns:
            List of evidence items
        """
        redis = await self.get_client()
        evidence_key = f"{self.evidence_prefix}{session_id}"

        evidence = await redis.lrange(evidence_key, 0, -1)
        return [json.loads(e) for e in evidence]

    async def clear_evidence(self, session_id: str) -> None:
        """
        Clear evidence board.

        Args:
            session_id: Session identifier
        """
        redis = await self.get_client()
        evidence_key = f"{self.evidence_prefix}{session_id}"
        await redis.delete(evidence_key)

    async def get_evidence_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary of evidence board.

        Args:
            session_id: Session identifier

        Returns:
            Dict with count, sources, and top relevance scores
        """
        evidence = await self.get_evidence_trail(session_id)

        if not evidence:
            return {"count": 0, "sources": [], "max_relevance": 0.0}

        sources = list(set(e.get("source", "unknown") for e in evidence))
        relevance_scores = [e.get("relevance", 0.0) for e in evidence]

        return {
            "count": len(evidence),
            "sources": sources,
            "max_relevance": max(relevance_scores) if relevance_scores else 0.0,
            "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
        }

    # ========================================================================
    # WORKFLOW STATE
    # ========================================================================

    async def set_workflow_phase(self, session_id: str, phase: str) -> None:
        """
        Set current workflow phase.

        Args:
            session_id: Session identifier
            phase: Phase name (summarizer, investigator, agent, reflector)
        """
        await self.update_session(session_id, {"current_phase": phase})

    async def get_workflow_phase(self, session_id: str) -> Optional[str]:
        """
        Get current workflow phase.

        Args:
            session_id: Session identifier

        Returns:
            Current phase name or None
        """
        session = await self.get_session(session_id)
        return session.get("current_phase") if session else None


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_working_memory: Optional[RedisWorkingMemory] = None


def get_working_memory() -> RedisWorkingMemory:
    """
    Get or create working memory singleton.

    Returns:
        RedisWorkingMemory: Singleton instance
    """
    global _working_memory
    if _working_memory is None:
        _working_memory = RedisWorkingMemory()
    return _working_memory


def get_langgraph_checkpointer():
    """
    Get the LangGraph checkpointer for workflow compilation.

    Returns:
        Checkpointer: RedisSaver or MemorySaver instance
    """
    working_memory = get_working_memory()
    return working_memory.get_langgraph_checkpointer()


def reset_working_memory() -> None:
    """Reset the working memory singleton (for testing)."""
    global _working_memory
    _working_memory = None
