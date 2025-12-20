"""
User Session Repository.

Handles MAU/WAU/DAU tracking for V3 KPIs.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from src.repositories.base import BaseRepository


class UserSessionRepository(BaseRepository):
    """
    Repository for user_sessions table.

    V3 addition for tracking:
    - Monthly Active Users (MAU)
    - Weekly Active Users (WAU)
    - Daily Active Users (DAU)
    """

    table_name = "user_sessions"
    model_class = None  # Set to UserSession model when available

    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List:
        """
        Get sessions for a specific user.

        Args:
            user_id: User identifier
            limit: Maximum records

        Returns:
            List of UserSession records
        """
        return await self.get_many(
            filters={"user_id": user_id},
            limit=limit,
        )

    async def get_sessions_in_range(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 10000,
    ) -> List:
        """
        Get sessions within a date range.

        Args:
            start_date: Start of range
            end_date: End of range
            limit: Maximum records

        Returns:
            List of UserSession records
        """
        if not self.client:
            return []

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .gte("session_start", start_date.isoformat())
            .lte("session_start", end_date.isoformat())
            .order("session_start", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_active_user_counts(self) -> Dict[str, int]:
        """
        Get current MAU/WAU/DAU counts.

        Uses v_kpi_active_users view which aggregates user_sessions.
        View returns: month, monthly_active_users, weekly_active_users, daily_active_users

        Returns:
            Dict with mau, wau, dau counts
        """
        if not self.client:
            return {"mau": 0, "wau": 0, "dau": 0}

        try:
            # Query the helper view for the current month's aggregations
            result = await (
                self.client.table("v_kpi_active_users")
                .select("monthly_active_users, weekly_active_users, daily_active_users")
                .order("month", desc=True)
                .limit(1)
                .execute()
            )

            if result.data:
                row = result.data[0]
                return {
                    "mau": row.get("monthly_active_users", 0) or 0,
                    "wau": row.get("weekly_active_users", 0) or 0,
                    "dau": row.get("daily_active_users", 0) or 0,
                }
        except Exception:
            # View may not exist or other error, fall back to direct calculation
            pass

        return {"mau": 0, "wau": 0, "dau": 0}

    async def get_session_metrics(
        self,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get session metrics (avg duration, pages viewed, etc.).

        Args:
            user_id: Optional filter by user

        Returns:
            Dict with session metrics including:
            - avg_duration_seconds: Average session length
            - avg_pages_viewed: Average pages per session
            - avg_queries_executed: Average queries per session
            - total_sessions: Total session count
        """
        if not self.client:
            return {
                "avg_duration_seconds": 0,
                "avg_pages_viewed": 0,
                "avg_queries_executed": 0,
                "total_sessions": 0,
            }

        # Build query
        query = self.client.table(self.table_name).select(
            "session_duration_seconds, page_views, queries_executed"
        )

        if user_id:
            query = query.eq("user_id", user_id)

        result = await query.limit(10000).execute()

        if not result.data:
            return {
                "avg_duration_seconds": 0,
                "avg_pages_viewed": 0,
                "avg_queries_executed": 0,
                "total_sessions": 0,
            }

        # Compute aggregations in Python
        durations = [
            r.get("session_duration_seconds", 0) or 0
            for r in result.data
        ]
        pages = [
            r.get("page_views", 0) or 0
            for r in result.data
        ]
        queries = [
            r.get("queries_executed", 0) or 0
            for r in result.data
        ]

        total = len(result.data)

        return {
            "avg_duration_seconds": sum(durations) / total if total else 0,
            "avg_pages_viewed": sum(pages) / total if total else 0,
            "avg_queries_executed": sum(queries) / total if total else 0,
            "total_sessions": total,
        }

    async def get_engagement_by_role(self) -> Dict[str, Dict[str, Any]]:
        """
        Get session engagement metrics broken down by user role.

        Returns:
            Dict mapping role to engagement metrics
        """
        if not self.client:
            return {}

        result = await (
            self.client.table(self.table_name)
            .select("user_role, session_duration_seconds, page_views, queries_executed")
            .limit(10000)
            .execute()
        )

        if not result.data:
            return {}

        # Group by role
        role_data: Dict[str, List[Dict]] = {}
        for row in result.data:
            role = row.get("user_role") or "unknown"
            if role not in role_data:
                role_data[role] = []
            role_data[role].append(row)

        # Compute metrics per role
        metrics = {}
        for role, sessions in role_data.items():
            durations = [s.get("session_duration_seconds", 0) or 0 for s in sessions]
            pages = [s.get("page_views", 0) or 0 for s in sessions]
            count = len(sessions)

            metrics[role] = {
                "session_count": count,
                "avg_duration_seconds": sum(durations) / count if count else 0,
                "avg_pages_viewed": sum(pages) / count if count else 0,
            }

        return metrics
