"""
Chatbot Message Repository.

Handles individual messages in chatbot conversations with agent attribution.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.repositories.base import BaseRepository


class ChatbotMessageRepository(BaseRepository):
    """
    Repository for chatbot_messages table.

    Manages messages with:
    - Agent attribution (which agent generated the response)
    - Tool call tracking
    - RAG context storage
    - Performance metrics
    """

    table_name = "chatbot_messages"
    model_class = None  # Set to ChatbotMessage model when available

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        agent_name: Optional[str] = None,
        agent_tier: Optional[int] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        rag_context: Optional[List[Dict[str, Any]]] = None,
        rag_sources: Optional[List[str]] = None,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[int] = None,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add a message to a conversation.

        Args:
            session_id: Conversation session ID
            role: Message role (user/assistant/system/tool)
            content: Message content
            agent_name: E2I agent that generated this (for assistant messages)
            agent_tier: Agent tier (0-5)
            tool_calls: Array of tool calls made
            tool_results: Results from tool calls
            rag_context: Retrieved documents used
            rag_sources: Source identifiers for RAG
            model_used: LLM model used
            tokens_used: Token count
            latency_ms: Response latency in milliseconds
            confidence_score: Confidence score (0-1)
            metadata: Additional metadata

        Returns:
            Created message or None
        """
        if not self.client:
            return None

        data = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "agent_name": agent_name,
            "agent_tier": agent_tier,
            "tool_calls": tool_calls or [],
            "tool_results": tool_results or [],
            "rag_context": rag_context or [],
            "rag_sources": rag_sources or [],
            "model_used": model_used,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "confidence_score": confidence_score,
            "metadata": metadata or {},
        }

        result = await self.client.table(self.table_name).insert(data).execute()

        return self._to_model(result.data[0]) if result.data else None

    async def get_session_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        ascending: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation session.

        Args:
            session_id: Conversation session ID
            limit: Maximum records
            offset: Pagination offset
            ascending: Order by created_at ascending

        Returns:
            List of messages
        """
        if not self.client:
            return []

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=not ascending)
            .limit(limit)
            .offset(offset)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_recent_messages(
        self,
        session_id: str,
        count: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get most recent messages for context.

        Args:
            session_id: Conversation session ID
            count: Number of recent messages

        Returns:
            List of recent messages (newest first)
        """
        if not self.client:
            return []

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(count)
            .execute()
        )

        # Reverse to get chronological order
        messages = [self._to_model(row) for row in result.data]
        return list(reversed(messages))

    async def get_by_role(
        self,
        session_id: str,
        role: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get messages by role.

        Args:
            session_id: Conversation session ID
            role: Message role (user/assistant/system/tool)
            limit: Maximum records

        Returns:
            List of messages
        """
        if not self.client:
            return []

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("session_id", session_id)
            .eq("role", role)
            .order("created_at")
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_by_agent(
        self,
        agent_name: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get messages generated by a specific agent.

        Args:
            agent_name: E2I agent name
            limit: Maximum records

        Returns:
            List of messages
        """
        if not self.client:
            return []

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_name", agent_name)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_message_count(
        self,
        session_id: str,
    ) -> int:
        """
        Get message count for a session.

        Args:
            session_id: Conversation session ID

        Returns:
            Message count
        """
        if not self.client:
            return 0

        result = await (
            self.client.table(self.table_name)
            .select("id", count="exact")
            .eq("session_id", session_id)
            .execute()
        )

        return result.count or 0

    async def get_tool_usage_stats(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Get tool usage statistics.

        Args:
            session_id: Optional filter by session

        Returns:
            Dict mapping tool names to usage count
        """
        if not self.client:
            return {}

        query = (
            self.client.table(self.table_name)
            .select("tool_calls")
            .neq("tool_calls", [])
        )

        if session_id:
            query = query.eq("session_id", session_id)

        result = await query.limit(10000).execute()

        tool_counts: Dict[str, int] = {}
        for row in result.data:
            tool_calls = row.get("tool_calls", [])
            for call in tool_calls:
                tool_name = call.get("tool_name", "unknown")
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        return tool_counts

    async def get_agent_stats(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get agent usage statistics.

        Args:
            session_id: Optional filter by session

        Returns:
            Dict mapping agent names to stats
        """
        if not self.client:
            return {}

        query = (
            self.client.table(self.table_name)
            .select("agent_name, tokens_used, latency_ms")
            .not_.is_("agent_name", "null")
        )

        if session_id:
            query = query.eq("session_id", session_id)

        result = await query.limit(10000).execute()

        agent_stats: Dict[str, Dict[str, Any]] = {}
        for row in result.data:
            agent_name = row.get("agent_name")
            if not agent_name:
                continue

            if agent_name not in agent_stats:
                agent_stats[agent_name] = {
                    "message_count": 0,
                    "total_tokens": 0,
                    "total_latency_ms": 0,
                }

            agent_stats[agent_name]["message_count"] += 1
            agent_stats[agent_name]["total_tokens"] += row.get("tokens_used", 0) or 0
            agent_stats[agent_name]["total_latency_ms"] += (
                row.get("latency_ms", 0) or 0
            )

        # Calculate averages
        for stats in agent_stats.values():
            count = stats["message_count"]
            if count > 0:
                stats["avg_tokens"] = stats["total_tokens"] / count
                stats["avg_latency_ms"] = stats["total_latency_ms"] / count

        return agent_stats

    async def search_messages(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search messages by content.

        Args:
            query: Search query
            session_id: Optional filter by session
            limit: Maximum records

        Returns:
            List of matching messages
        """
        if not self.client:
            return []

        db_query = (
            self.client.table(self.table_name)
            .select("*")
            .ilike("content", f"%{query}%")
            .order("created_at", desc=True)
            .limit(limit)
        )

        if session_id:
            db_query = db_query.eq("session_id", session_id)

        result = await db_query.execute()

        return [self._to_model(row) for row in result.data]


# Factory function for dependency injection
def get_chatbot_message_repository(
    supabase_client=None,
) -> ChatbotMessageRepository:
    """
    Get a ChatbotMessageRepository instance.

    Args:
        supabase_client: Optional Supabase client

    Returns:
        Repository instance
    """
    return ChatbotMessageRepository(supabase_client)
