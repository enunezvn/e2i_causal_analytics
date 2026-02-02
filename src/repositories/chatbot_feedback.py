"""
Chatbot Feedback Repository.

Handles user feedback (thumbs up/down) on chatbot assistant responses.
Used for quality improvement and prompt optimization.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from src.repositories.base import BaseRepository


class ChatbotFeedbackRepository(BaseRepository):
    """
    Repository for chatbot_message_feedback table.

    Manages feedback with:
    - Rating storage (thumbs_up/thumbs_down)
    - Optional user comments
    - Agent attribution for performance analysis
    - Analytics functions for feedback trends
    """

    table_name = "chatbot_message_feedback"
    model_class = None  # Set to ChatbotFeedback model when available

    async def add_feedback(
        self,
        message_id: int,
        session_id: str,
        rating: Literal["thumbs_up", "thumbs_down"],
        comment: Optional[str] = None,
        query_text: Optional[str] = None,
        response_preview: Optional[str] = None,
        agent_name: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add feedback for a chatbot message.

        Args:
            message_id: ID of the message being rated
            session_id: Conversation session ID
            rating: 'thumbs_up' or 'thumbs_down'
            comment: Optional user comment explaining the rating
            query_text: The user query that led to this response
            response_preview: First 500 chars of the response
            agent_name: Which agent generated the response
            tools_used: Tools used in generating the response
            metadata: Additional metadata

        Returns:
            Created feedback record or None
        """
        if not self.client:
            return None

        data = {
            "message_id": message_id,
            "session_id": session_id,
            "rating": rating,
            "comment": comment,
            "query_text": query_text,
            "response_preview": response_preview[:500] if response_preview else None,
            "agent_name": agent_name,
            "tools_used": tools_used or [],
            "metadata": metadata or {},
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        try:
            result = await self.client.table(self.table_name).insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            # Handle duplicate feedback (unique constraint)
            if "unique_message_feedback" in str(e):
                # Update existing feedback instead
                return await self.update_feedback(message_id, session_id, rating, comment)
            raise

    async def update_feedback(
        self,
        message_id: int,
        session_id: str,
        rating: Literal["thumbs_up", "thumbs_down"],
        comment: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update existing feedback for a message.

        Args:
            message_id: ID of the message
            session_id: Conversation session ID
            rating: New rating
            comment: New comment (optional)

        Returns:
            Updated feedback record or None
        """
        if not self.client:
            return None

        data = {"rating": rating}
        if comment is not None:
            data["comment"] = comment

        result = await (
            self.client.table(self.table_name)
            .update(data)
            .eq("message_id", message_id)
            .eq("session_id", session_id)
            .execute()
        )
        return result.data[0] if result.data else None

    async def get_feedback_by_message(
        self,
        message_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get feedback for a specific message.

        Args:
            message_id: ID of the message

        Returns:
            Feedback record or None
        """
        if not self.client:
            return None

        result = await (
            self.client.table(self.table_name).select("*").eq("message_id", message_id).execute()
        )
        return result.data[0] if result.data else None

    async def get_session_feedback(
        self,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all feedback for a session.

        Args:
            session_id: Conversation session ID

        Returns:
            List of feedback records
        """
        if not self.client:
            return []

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data or []

    async def get_agent_stats(
        self,
        agent_name: Optional[str] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get feedback statistics per agent.

        Args:
            agent_name: Filter by specific agent (optional)
            days: Number of days to analyze

        Returns:
            List of agent stats with approval rates
        """
        if not self.client:
            return []

        # Use the database function for aggregation
        result = await self.client.rpc(
            "get_agent_feedback_stats",
            {"p_agent_name": agent_name, "p_days": days},
        ).execute()
        return result.data or []

    async def get_negative_feedback(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recent negative feedback for review.

        Args:
            limit: Maximum number of records

        Returns:
            List of negative feedback records
        """
        if not self.client:
            return []

        # Use the database function
        result = await self.client.rpc(
            "get_negative_feedback",
            {"p_limit": limit},
        ).execute()
        return result.data or []

    async def get_feedback_summary(
        self,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get a summary of feedback for the specified period.

        Args:
            days: Number of days to analyze

        Returns:
            Summary with counts and rates
        """
        if not self.client:
            return {}

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Get all feedback in period
        result = await (
            self.client.table(self.table_name)
            .select("rating, agent_name")
            .gte("created_at", cutoff.isoformat())
            .execute()
        )

        if not result.data:
            return {
                "period_days": days,
                "total_feedback": 0,
                "thumbs_up": 0,
                "thumbs_down": 0,
                "approval_rate": None,
            }

        total = len(result.data)
        thumbs_up = sum(1 for r in result.data if r["rating"] == "thumbs_up")
        thumbs_down = total - thumbs_up

        return {
            "period_days": days,
            "total_feedback": total,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "approval_rate": round(thumbs_up / total * 100, 2) if total > 0 else None,
        }

    async def delete_feedback(
        self,
        message_id: int,
        session_id: str,
    ) -> bool:
        """
        Delete feedback for a message.

        Args:
            message_id: ID of the message
            session_id: Conversation session ID

        Returns:
            True if deleted, False otherwise
        """
        if not self.client:
            return False

        result = await (
            self.client.table(self.table_name)
            .delete()
            .eq("message_id", message_id)
            .eq("session_id", session_id)
            .execute()
        )
        return bool(result.data)


# Factory function for dependency injection
def get_chatbot_feedback_repository(
    supabase_client=None,
) -> ChatbotFeedbackRepository:
    """
    Get a ChatbotFeedbackRepository instance.

    Args:
        supabase_client: Optional Supabase client

    Returns:
        Repository instance
    """
    return ChatbotFeedbackRepository(supabase_client)
