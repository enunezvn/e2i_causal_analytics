"""
Chatbot Conversation Repository.

Handles conversation sessions for E2I chatbot with brand/region context.
"""

import uuid
from typing import Any, Dict, List, Optional

from src.repositories.base import BaseRepository


class ChatbotConversationRepository(BaseRepository):
    """
    Repository for chatbot_conversations table.

    Manages conversation sessions with:
    - E2I context (brand, region, query type)
    - Session-based organization
    - Archiving and pinning
    """

    table_name = "chatbot_conversations"
    model_class = None  # Set to ChatbotConversation model when available

    def generate_session_id(self, user_id: str) -> str:
        """
        Generate a new session ID.

        Format: user_id~uuid

        Args:
            user_id: User UUID

        Returns:
            New session ID string
        """
        return f"{user_id}~{uuid.uuid4()}"

    async def create_conversation(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        brand_context: Optional[str] = None,
        region_context: Optional[str] = None,
        query_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new conversation.

        Args:
            user_id: User UUID
            session_id: Optional custom session ID (auto-generated if not provided)
            title: Conversation title
            brand_context: Brand filter for this conversation
            region_context: Region filter for this conversation
            query_type: Classification of query type
            metadata: Additional metadata

        Returns:
            Created conversation or None
        """
        if not self.client:
            return None

        if session_id is None:
            session_id = self.generate_session_id(user_id)

        data = {
            "session_id": session_id,
            "user_id": user_id,
            "title": title,
            "brand_context": brand_context,
            "region_context": region_context,
            "query_type": query_type,
            "metadata": metadata or {},
        }

        result = await self.client.table(self.table_name).insert(data).execute()

        return self._to_model(result.data[0]) if result.data else None

    async def get_by_session_id(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get conversation by session ID.

        Args:
            session_id: Session identifier

        Returns:
            Conversation dict or None
        """
        if not self.client:
            return None

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("session_id", session_id)
            .single()
            .execute()
        )

        return self._to_model(result.data) if result.data else None

    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 20,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get user's conversations.

        Args:
            user_id: User UUID
            limit: Maximum records
            include_archived: Include archived conversations

        Returns:
            List of conversations
        """
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("user_id", user_id)
            .order("last_message_at", desc=True)
            .limit(limit)
        )

        if not include_archived:
            query = query.eq("is_archived", False)

        result = await query.execute()
        return [self._to_model(row) for row in result.data]

    async def update_title(
        self,
        session_id: str,
        title: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Update conversation title.

        Args:
            session_id: Session identifier
            title: New title

        Returns:
            Updated conversation or None
        """
        if not self.client:
            return None

        result = await (
            self.client.table(self.table_name)
            .update({"title": title})
            .eq("session_id", session_id)
            .execute()
        )

        return self._to_model(result.data[0]) if result.data else None

    async def archive_conversation(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Archive a conversation.

        Args:
            session_id: Session identifier

        Returns:
            Updated conversation or None
        """
        if not self.client:
            return None

        result = await (
            self.client.table(self.table_name)
            .update({"is_archived": True})
            .eq("session_id", session_id)
            .execute()
        )

        return self._to_model(result.data[0]) if result.data else None

    async def pin_conversation(
        self,
        session_id: str,
        pinned: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Pin or unpin a conversation.

        Args:
            session_id: Session identifier
            pinned: Pin state

        Returns:
            Updated conversation or None
        """
        if not self.client:
            return None

        result = await (
            self.client.table(self.table_name)
            .update({"is_pinned": pinned})
            .eq("session_id", session_id)
            .execute()
        )

        return self._to_model(result.data[0]) if result.data else None

    async def update_query_type(
        self,
        session_id: str,
        query_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Update the query type classification.

        Args:
            session_id: Session identifier
            query_type: New query type

        Returns:
            Updated conversation or None
        """
        if not self.client:
            return None

        result = await (
            self.client.table(self.table_name)
            .update({"query_type": query_type})
            .eq("session_id", session_id)
            .execute()
        )

        return self._to_model(result.data[0]) if result.data else None

    async def update_tools_used(
        self,
        session_id: str,
        tools: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Update the tools used in this conversation.

        Args:
            session_id: Session identifier
            tools: List of tool names

        Returns:
            Updated conversation or None
        """
        if not self.client:
            return None

        # Get current tools and merge
        conv = await self.get_by_session_id(session_id)
        if not conv:
            return None

        current_tools = set(conv.get("tools_used", []) or [])
        current_tools.update(tools)

        result = await (
            self.client.table(self.table_name)
            .update({"tools_used": list(current_tools)})
            .eq("session_id", session_id)
            .execute()
        )

        return self._to_model(result.data[0]) if result.data else None

    async def get_by_brand(
        self,
        brand: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get conversations filtered by brand.

        Args:
            brand: Brand name
            limit: Maximum records

        Returns:
            List of conversations
        """
        if not self.client:
            return []

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("brand_context", brand)
            .eq("is_archived", False)
            .order("last_message_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_by_query_type(
        self,
        query_type: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get conversations by query type.

        Args:
            query_type: Query type classification
            limit: Maximum records

        Returns:
            List of conversations
        """
        return await self.get_many(
            filters={"query_type": query_type, "is_archived": False},
            limit=limit,
        )


# Factory function for dependency injection
def get_chatbot_conversation_repository(
    supabase_client=None,
) -> ChatbotConversationRepository:
    """
    Get a ChatbotConversationRepository instance.

    Args:
        supabase_client: Optional Supabase client

    Returns:
        Repository instance
    """
    return ChatbotConversationRepository(supabase_client)
