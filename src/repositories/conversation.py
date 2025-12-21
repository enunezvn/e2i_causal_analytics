"""
Conversation Repository.

Handles chat history for RAG and self-improvement.

Uses cognitive_cycles table which stores user queries with embeddings
for vector similarity search.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.repositories.base import BaseRepository


class ConversationRepository(BaseRepository):
    """
    Repository for cognitive_cycles table (conversation/query history).

    Supports:
    - Chat history queries
    - RAG context retrieval via vector similarity search
    - Feedback tracking for self-improvement

    Note: Uses cognitive_cycles table which contains:
    - user_query: The user's natural language query
    - query_embedding: vector(1536) for similarity search
    - agent_response: The system's response
    - feedback_type/feedback_text: User feedback
    """

    table_name = "cognitive_cycles"
    model_class = None  # Set to CognitiveCycle model when available

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
            List of conversation records
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
            List of conversation records
        """
        return await self.get_many(
            filters={"user_id": user_id},
            limit=limit,
        )

    async def get_with_feedback(
        self,
        feedback_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get conversations that have feedback.

        Used by Feedback Learner for self-improvement.

        Args:
            feedback_type: Optional filter ('positive', 'negative', 'neutral')
            limit: Maximum records
            offset: Pagination offset
            user_id: Optional filter by user
            since: Optional filter by date

        Returns:
            List of conversation records with feedback
        """
        if not self.client:
            return []

        # Use the RPC function for efficient feedback queries
        params: Dict[str, Any] = {
            "p_limit": limit,
            "p_offset": offset,
        }

        if feedback_type:
            params["p_feedback_type"] = feedback_type
        if user_id:
            params["p_user_id"] = user_id
        if since:
            params["p_since"] = since.isoformat()

        result = await self.client.rpc("get_conversations_with_feedback", params).execute()

        return result.data if result.data else []

    async def get_similar_queries(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find similar past queries using pgvector similarity search.

        Uses the search_similar_conversations RPC function which performs
        cosine similarity search on the query_embedding column using
        pgvector's IVFFlat index.

        Args:
            query_embedding: Query vector embedding (1536 dimensions)
            top_k: Number of similar queries to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            user_id: Optional filter to search within user's history
            session_id: Optional filter to search within session

        Returns:
            List of similar past conversations with similarity scores

        Example:
            ```python
            similar = await repo.get_similar_queries(
                query_embedding=embedding_model.embed("What is TRx trend?"),
                top_k=5,
                min_similarity=0.7
            )
            for conv in similar:
                print(f"Query: {conv['user_query']}")
                print(f"Similarity: {conv['similarity']:.2f}")
            ```
        """
        if not self.client:
            return []

        # Validate embedding dimensions
        if len(query_embedding) != 1536:
            raise ValueError(f"Embedding must be 1536 dimensions, got {len(query_embedding)}")

        # Build RPC parameters
        params: Dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "min_similarity": min_similarity,
        }

        if user_id:
            params["filter_user_id"] = user_id
        if session_id:
            params["filter_session_id"] = session_id

        # Call the pgvector RPC function
        result = await self.client.rpc("search_similar_conversations", params).execute()

        return result.data if result.data else []

    async def get_conversation_context(
        self,
        query_embedding: List[float],
        context_size: int = 3,
        min_similarity: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Get relevant conversation context for RAG.

        Retrieves similar past conversations to provide context
        for the current query. Optimized for RAG use cases.

        Args:
            query_embedding: Current query embedding
            context_size: Number of context items to return
            min_similarity: Minimum relevance threshold

        Returns:
            List of relevant past conversations for context
        """
        similar = await self.get_similar_queries(
            query_embedding=query_embedding,
            top_k=context_size,
            min_similarity=min_similarity,
        )

        # Format for RAG context
        return [
            {
                "query": conv.get("user_query"),
                "response": conv.get("agent_response"),
                "intent": conv.get("detected_intent"),
                "similarity": conv.get("similarity"),
            }
            for conv in similar
        ]

    async def record_feedback(
        self,
        cycle_id: str,
        feedback_type: str,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Record user feedback for a conversation.

        Args:
            cycle_id: The cognitive cycle ID
            feedback_type: 'positive', 'negative', or 'neutral'
            feedback_text: Optional detailed feedback text
            feedback_score: Optional numeric score (1-5)

        Returns:
            Updated conversation record
        """
        if not self.client:
            return None

        updates: Dict[str, Any] = {
            "feedback_type": feedback_type,
            "feedback_at": datetime.now().isoformat(),
        }

        if feedback_text:
            updates["feedback_text"] = feedback_text
        if feedback_score is not None:
            updates["feedback_score"] = feedback_score

        result = await (
            self.client.table(self.table_name).update(updates).eq("cycle_id", cycle_id).execute()
        )

        return result.data[0] if result.data else None
