"""
Conversation Repository.

Handles chat history for RAG and self-improvement.
"""

from typing import List, Optional
from src.repositories.base import BaseRepository


class ConversationRepository(BaseRepository):
    """
    Repository for conversations table.

    Supports:
    - Chat history queries
    - RAG context retrieval
    - Feedback tracking
    """

    table_name = "conversations"
    model_class = None  # Set to Conversation model when available

    async def get_by_session(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum records

        Returns:
            List of Conversation records
        """
        return await self.get_many(
            filters={"session_id": session_id},
            limit=limit,
        )

    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List:
        """
        Get conversation history for a user.

        Args:
            user_id: User identifier
            limit: Maximum records

        Returns:
            List of Conversation records
        """
        return await self.get_many(
            filters={"user_id": user_id},
            limit=limit,
        )

    async def get_with_feedback(
        self,
        feedback_type: Optional[str] = None,
        limit: int = 100,
    ) -> List:
        """
        Get conversations that have feedback.

        Used by Feedback Learner for self-improvement.

        Args:
            feedback_type: Optional filter ('positive', 'negative')
            limit: Maximum records

        Returns:
            List of Conversation records with feedback
        """
        filters = {}
        if feedback_type:
            filters["feedback_type"] = feedback_type
        # TODO: Filter for non-null feedback
        return await self.get_many(filters=filters, limit=limit)

    async def get_similar_queries(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List:
        """
        Find similar past queries using vector similarity.

        Args:
            query_embedding: Query vector embedding
            top_k: Number of similar queries to return

        Returns:
            Similar past conversations
        """
        # TODO: Implement with pgvector similarity search
        return []
