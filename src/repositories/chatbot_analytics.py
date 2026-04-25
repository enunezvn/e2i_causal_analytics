"""
Chatbot Analytics Repository.

Handles usage analytics for the E2I chatbot - tracks queries, performance, and tool usage.
Used for monitoring, capacity planning, and optimization.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.repositories.base import BaseRepository


class ChatbotAnalyticsRepository(BaseRepository):
    """
    Repository for chatbot_analytics table.

    Manages analytics with:
    - Query tracking (type, complexity, performance)
    - Tool usage recording
    - Agent performance metrics
    - Error tracking
    """

    table_name = "chatbot_analytics"
    model_class = None  # Set to ChatbotAnalytics model when available

    async def record_analytics(
        self,
        session_id: str,
        query_type: str,
        response_time_ms: Optional[int] = None,
        first_token_time_ms: Optional[int] = None,
        total_tokens: Optional[int] = None,
        query_complexity: str = "simple",
        tools_invoked: Optional[List[str]] = None,
        tools_succeeded: Optional[List[str]] = None,
        tools_failed: Optional[List[str]] = None,
        primary_agent: Optional[str] = None,
        agents_consulted: Optional[List[str]] = None,
        orchestrator_used: bool = False,
        tool_composer_used: bool = False,
        rag_queries: int = 0,
        rag_documents_retrieved: int = 0,
        memory_context_loaded: bool = False,
        episodic_memory_saved: bool = False,
        error_occurred: bool = False,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        message_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Record analytics for a chatbot interaction.

        Args:
            session_id: Conversation session ID
            query_type: Classification of the query
            response_time_ms: Total response time in milliseconds
            first_token_time_ms: Time to first streaming token
            total_tokens: Approximate token count in response
            query_complexity: simple, moderate, complex, multi_faceted
            tools_invoked: Array of tool names used
            tools_succeeded: Tools that completed successfully
            tools_failed: Tools that failed
            primary_agent: Main agent that handled the query
            agents_consulted: All agents consulted during execution
            orchestrator_used: Whether orchestrator was used
            tool_composer_used: Whether tool composer was used
            rag_queries: Number of RAG retrievals
            rag_documents_retrieved: Number of documents retrieved
            memory_context_loaded: Whether memory context was loaded
            episodic_memory_saved: Whether episodic memory was saved
            error_occurred: Whether an error occurred
            error_type: Type of error
            error_message: Error message
            message_id: Associated message ID
            metadata: Additional metadata

        Returns:
            Created analytics record or None
        """
        if not self.client:
            return None

        data = {
            "session_id": session_id,
            "query_type": query_type,
            "query_complexity": query_complexity,
            "response_time_ms": response_time_ms,
            "first_token_time_ms": first_token_time_ms,
            "total_tokens": total_tokens,
            "tools_invoked": tools_invoked or [],
            "tools_succeeded": tools_succeeded or [],
            "tools_failed": tools_failed or [],
            "primary_agent": primary_agent,
            "agents_consulted": agents_consulted or [],
            "orchestrator_used": orchestrator_used,
            "tool_composer_used": tool_composer_used,
            "rag_queries": rag_queries,
            "rag_documents_retrieved": rag_documents_retrieved,
            "memory_context_loaded": memory_context_loaded,
            "episodic_memory_saved": episodic_memory_saved,
            "error_occurred": error_occurred,
            "error_type": error_type,
            "error_message": error_message,
            "message_id": message_id,
            "metadata": metadata or {},
            "response_completed_at": datetime.now(timezone.utc).isoformat()
            if response_time_ms
            else None,
        }

        # Remove None values for cleaner inserts
        data = {k: v for k, v in data.items() if v is not None}

        try:
            result = await self.client.table(self.table_name).insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            # Log error but don't fail the request
            print(f"[ANALYTICS] Failed to record analytics: {e}")
            return None

    def record_analytics_sync(
        self,
        session_id: str,
        query_type: str,
        response_time_ms: Optional[int] = None,
        tools_invoked: Optional[List[str]] = None,
        primary_agent: Optional[str] = None,
        error_occurred: bool = False,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Record analytics synchronously (for use in sync contexts).

        Args:
            session_id: Conversation session ID
            query_type: Classification of the query
            response_time_ms: Total response time in milliseconds
            tools_invoked: Array of tool names used
            primary_agent: Main agent that handled the query
            error_occurred: Whether an error occurred
            error_type: Type of error
            metadata: Additional metadata

        Returns:
            Created analytics record or None
        """
        if not self.client:
            return None

        data = {
            "session_id": session_id,
            "query_type": query_type,
            "response_time_ms": response_time_ms,
            "tools_invoked": tools_invoked or [],
            "primary_agent": primary_agent,
            "error_occurred": error_occurred,
            "error_type": error_type,
            "metadata": metadata or {},
            "response_completed_at": datetime.now(timezone.utc).isoformat()
            if response_time_ms
            else None,
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        try:
            # Synchronous call - supabase-py is sync
            result = self.client.table(self.table_name).insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"[ANALYTICS] Failed to record analytics: {e}")
            return None

    async def get_usage_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get usage summary for a date range.

        Args:
            start_date: Start of period (default: 7 days ago)
            end_date: End of period (default: now)

        Returns:
            Summary with aggregated stats
        """
        if not self.client:
            return {}

        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        try:
            result = await self.client.rpc(
                "get_chatbot_usage_summary",
                {
                    "p_start_date": start_date.isoformat(),
                    "p_end_date": end_date.isoformat(),
                },
            ).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            print(f"[ANALYTICS] Failed to get usage summary: {e}")
            return {}

    async def get_agent_performance(
        self,
        agent_name: Optional[str] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get performance metrics per agent.

        Args:
            agent_name: Filter by specific agent (optional)
            days: Number of days to analyze

        Returns:
            List of agent performance metrics
        """
        if not self.client:
            return []

        try:
            result = await self.client.rpc(
                "get_agent_performance_metrics",
                {"p_agent_name": agent_name, "p_days": days},
            ).execute()
            return result.data or []
        except Exception as e:
            print(f"[ANALYTICS] Failed to get agent performance: {e}")
            return []

    async def get_hourly_pattern(
        self,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Get usage distribution by hour of day.

        Args:
            days: Number of days to analyze

        Returns:
            List of hourly usage patterns
        """
        if not self.client:
            return []

        try:
            result = await self.client.rpc(
                "get_hourly_usage_pattern",
                {"p_days": days},
            ).execute()
            return result.data or []
        except Exception as e:
            print(f"[ANALYTICS] Failed to get hourly pattern: {e}")
            return []

    async def get_recent_errors(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get recent error analytics entries.

        Args:
            limit: Maximum number of records

        Returns:
            List of error entries
        """
        if not self.client:
            return []

        try:
            result = await (
                self.client.table(self.table_name)
                .select("*")
                .eq("error_occurred", True)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            print(f"[ANALYTICS] Failed to get recent errors: {e}")
            return []

    async def get_session_analytics(
        self,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all analytics entries for a session.

        Args:
            session_id: Conversation session ID

        Returns:
            List of analytics entries
        """
        if not self.client:
            return []

        try:
            result = await (
                self.client.table(self.table_name)
                .select("*")
                .eq("session_id", session_id)
                .order("created_at", desc=False)
                .execute()
            )
            return result.data or []
        except Exception as e:
            print(f"[ANALYTICS] Failed to get session analytics: {e}")
            return []

    async def get_tool_usage_stats(
        self,
        days: int = 30,
    ) -> Dict[str, int]:
        """
        Get tool usage frequency statistics.

        Args:
            days: Number of days to analyze

        Returns:
            Dict of tool_name -> usage_count
        """
        if not self.client:
            return {}

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            result = await (
                self.client.table(self.table_name)
                .select("tools_invoked")
                .gte("created_at", cutoff.isoformat())
                .execute()
            )

            if not result.data:
                return {}

            # Aggregate tool usage
            tool_counts: Dict[str, int] = {}
            for record in result.data:
                for tool in record.get("tools_invoked", []):
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1

            return dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"[ANALYTICS] Failed to get tool usage stats: {e}")
            return {}

    async def get_query_type_distribution(
        self,
        days: int = 30,
    ) -> Dict[str, int]:
        """
        Get query type distribution.

        Args:
            days: Number of days to analyze

        Returns:
            Dict of query_type -> count
        """
        if not self.client:
            return {}

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            result = await (
                self.client.table(self.table_name)
                .select("query_type")
                .gte("created_at", cutoff.isoformat())
                .execute()
            )

            if not result.data:
                return {}

            # Aggregate query types
            type_counts: Dict[str, int] = {}
            for record in result.data:
                qtype = record.get("query_type", "unknown")
                type_counts[qtype] = type_counts.get(qtype, 0) + 1

            return dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"[ANALYTICS] Failed to get query type distribution: {e}")
            return {}


# Factory function for dependency injection
def get_chatbot_analytics_repository(
    supabase_client=None,
) -> ChatbotAnalyticsRepository:
    """
    Get a ChatbotAnalyticsRepository instance.

    Args:
        supabase_client: Optional Supabase client

    Returns:
        Repository instance
    """
    return ChatbotAnalyticsRepository(supabase_client)
