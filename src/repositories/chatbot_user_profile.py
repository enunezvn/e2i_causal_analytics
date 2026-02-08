"""
Chatbot User Profile Repository.

Handles user preferences and settings for E2I chatbot.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

from src.repositories.base import BaseRepository


class ChatbotUserProfileRepository(BaseRepository):
    """
    Repository for chatbot_user_profiles table.

    Manages user profiles with E2I-specific preferences:
    - Brand preference (Kisqali, Fabhalta, Remibrutinib)
    - Region preference
    - Expertise level
    - Usage tracking
    """

    table_name = "chatbot_user_profiles"
    model_class = None  # Set to ChatbotUserProfile model when available

    async def get_by_user_id(
        self,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get profile for a specific user.

        Args:
            user_id: User UUID

        Returns:
            User profile dict or None
        """
        return await self.get_by_id(user_id)

    async def update_preferences(
        self,
        user_id: str,
        brand_preference: Optional[str] = None,
        region_preference: Optional[str] = None,
        expertise_level: Optional[str] = None,
        show_technical_details: Optional[bool] = None,
        enable_recommendations: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update user preferences.

        Args:
            user_id: User UUID
            brand_preference: Default brand context
            region_preference: Default region filter
            expertise_level: User expertise (basic/intermediate/advanced/expert)
            show_technical_details: Show technical info in responses
            enable_recommendations: Include recommendations

        Returns:
            Updated profile or None
        """
        updates: Dict[str, Any] = {}

        if brand_preference is not None:
            updates["brand_preference"] = brand_preference
        if region_preference is not None:
            updates["region_preference"] = region_preference
        if expertise_level is not None:
            updates["expertise_level"] = expertise_level
        if show_technical_details is not None:
            updates["show_technical_details"] = show_technical_details
        if enable_recommendations is not None:
            updates["enable_recommendations"] = enable_recommendations

        if not updates:
            return await self.get_by_id(user_id)

        return await self.update(user_id, updates)

    async def get_brand_preference(
        self,
        user_id: str,
    ) -> Optional[str]:
        """
        Get user's default brand preference.

        Args:
            user_id: User UUID

        Returns:
            Brand name or None
        """
        profile = await self.get_by_id(user_id)
        if profile:
            return cast(Optional[str], profile.get("brand_preference"))
        return None

    async def update_activity(
        self,
        user_id: str,
        new_conversation: bool = False,
        new_messages: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Update user activity statistics.

        Args:
            user_id: User UUID
            new_conversation: Whether a new conversation was started
            new_messages: Number of new messages

        Returns:
            Updated profile or None
        """
        if not self.client:
            return None

        # Use RPC function if available, otherwise manual update
        try:
            await self.client.rpc(
                "update_chatbot_user_activity",
                {
                    "p_user_id": user_id,
                    "p_new_conversation": new_conversation,
                    "p_new_messages": new_messages,
                },
            ).execute()
        except Exception:
            # Fallback to manual update
            profile = await self.get_by_id(user_id)
            if profile:
                updates = {
                    "last_active_at": datetime.now(timezone.utc).isoformat(),
                }
                if new_conversation:
                    updates["total_conversations"] = profile.get("total_conversations", 0) + 1
                if new_messages > 0:
                    updates["total_messages"] = profile.get("total_messages", 0) + new_messages
                return await self.update(user_id, updates)

        return await self.get_by_id(user_id)

    async def get_active_users(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recently active users.

        Args:
            since: Filter by last_active_at >= since
            limit: Maximum records

        Returns:
            List of user profiles
        """
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("*")
            .order("last_active_at", desc=True)
            .limit(limit)
        )

        if since:
            query = query.gte("last_active_at", since.isoformat())

        result = await query.execute()
        return [self._to_model(row) for row in result.data]

    async def get_by_expertise_level(
        self,
        expertise_level: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get users by expertise level.

        Args:
            expertise_level: basic/intermediate/advanced/expert
            limit: Maximum records

        Returns:
            List of user profiles
        """
        return await self.get_many(
            filters={"expertise_level": expertise_level},
            limit=limit,
        )


# Factory function for dependency injection
def get_chatbot_user_profile_repository(
    supabase_client=None,
) -> ChatbotUserProfileRepository:
    """
    Get a ChatbotUserProfileRepository instance.

    Args:
        supabase_client: Optional Supabase client

    Returns:
        Repository instance
    """
    return ChatbotUserProfileRepository(supabase_client)
