"""
Agent Activity Repository.

Handles agent analysis outputs and activity tracking.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.repositories.base import BaseRepository


class AgentActivityRepository(BaseRepository):
    """
    Repository for agent_activities table.

    Supports:
    - Agent output queries
    - Analysis result retrieval
    - Activity tracking
    - Tier-based filtering

    Table schema:
    - activity_id (PK)
    - agent_name (VARCHAR(50); roster = AGENT_REGISTRY_CONFIG, 22 agents)
    - agent_tier (workstream_type)
    - activity_timestamp (TIMESTAMPTZ)
    - workstream (workstream_type)
    - analysis_results (JSONB)
    """

    table_name = "agent_activities"
    model_class = None  # Set to AgentActivity model when available
    id_column = "activity_id"  # live PK (#894: .eq("id") was a latent 42703)
    # Provenance (#894): agent_activities carries is_synthetic (migration 063).
    # Since #1355 the synthetic loader DOES seed agent_activities
    # (AgentActivitiesGenerator, is_synthetic=true) — the default-exclude
    # predicate keeps planted "agent analyses" out of real-mode reads
    # (_query_agent_analysis reads via query_activities); synthetic-gold
    # showcase instances surface them via the E2I_INCLUDE_SYNTHETIC gate.
    HAS_PROVENANCE = True

    async def query_activities(
        self,
        agent_name: Optional[str] = None,
        brand: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """Chat agent-analysis read path (#1355): windowed, brand-resolvable.

        The table has NO brand column; brand-scoped analyses carry the brand
        INSIDE the ``analysis_results`` JSONB — writers (the DGP seed
        generator and the runtime activity writer,
        ``src/agents/activity_writer.py``) stamp ``analysis_results.brand``,
        and this method filters on ``analysis_results->>'brand'``.

        Args:
            agent_name: Optional agent filter (e.g. 'heterogeneous_optimizer')
            brand: Optional brand filter via ``analysis_results->>'brand'``
            since: Optional window start applied to ``activity_timestamp``
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows
                (the deployment gate ``E2I_INCLUDE_SYNTHETIC`` also lifts the
                exclusion on synthetic-gold showcase instances).

        Returns:
            Activities ordered by activity_timestamp descending.
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        query = self.client.table(self.table_name).select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        if brand:
            query = query.eq("analysis_results->>brand", brand)
        if since:
            query = query.gte("activity_timestamp", since.isoformat())

        query = apply_provenance_filter(query, include_synthetic)
        result = await query.order("activity_timestamp", desc=True).limit(limit).execute()

        return [self._to_model(row) for row in result.data]

    async def get_by_agent(
        self,
        agent_type: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get activities for a specific agent type.

        Args:
            agent_type: Agent name (e.g., 'orchestrator', 'causal_impact')
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of AgentActivity records ordered by timestamp descending
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        query = self.client.table(self.table_name).select("*").eq("agent_name", agent_type)
        result = await (
            apply_provenance_filter(query, include_synthetic)
            .order("activity_timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_by_tier(
        self,
        tier: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get activities for all agents in a tier.

        Args:
            tier: Agent tier (workstream_type: 'coordination', 'causal_analytics', etc.)
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of AgentActivity records
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        query = self.client.table(self.table_name).select("*").eq("agent_tier", tier)
        result = await (
            apply_provenance_filter(query, include_synthetic)
            .order("activity_timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_analysis_results(
        self,
        agent_type: str,
        limit: int = 50,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get analysis results from a specific agent.

        Used by RAG for indexing agent outputs.
        Only returns activities that have non-null analysis_results.

        Args:
            agent_type: Agent name (e.g., 'causal_impact', 'gap_analyzer')
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of AgentActivity records with analysis_results
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_name", agent_type)
            .not_.is_("analysis_results", "null")
        )
        result = await (
            apply_provenance_filter(query, include_synthetic)
            .order("activity_timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_recent_activities(
        self,
        hours: int = 24,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get recent agent activities.

        Args:
            hours: Number of hours to look back
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Recent activities ordered by timestamp descending
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        query = (
            self.client.table(self.table_name)
            .select("*")
            .gte("activity_timestamp", cutoff_time.isoformat())
        )
        result = await (
            apply_provenance_filter(query, include_synthetic)
            .order("activity_timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_by_workstream(
        self,
        workstream: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get activities for a specific workstream.

        Args:
            workstream: Workstream type (WS1, WS2, WS3)
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of AgentActivity records
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        query = self.client.table(self.table_name).select("*").eq("workstream", workstream)
        result = await (
            apply_provenance_filter(query, include_synthetic)
            .order("activity_timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_agent_activity_summary(
        self,
        hours: int = 24,
        include_synthetic: bool = False,
    ) -> Dict[str, Any]:
        """
        Get summary of agent activities over a time period.

        Args:
            hours: Number of hours to look back
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Dict with activity counts by agent and tier
        """
        if not self.client:
            return {
                "total_activities": 0,
                "by_agent": {},
                "by_tier": {},
                "with_results": 0,
            }

        from src.repositories.provenance import apply_provenance_filter

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        query = (
            self.client.table(self.table_name)
            .select("agent_name, agent_tier, analysis_results")
            .gte("activity_timestamp", cutoff_time.isoformat())
        )
        result = await apply_provenance_filter(query, include_synthetic).limit(10000).execute()

        if not result.data:
            return {
                "total_activities": 0,
                "by_agent": {},
                "by_tier": {},
                "with_results": 0,
            }

        # Aggregate counts
        by_agent: Dict[str, int] = {}
        by_tier: Dict[str, int] = {}
        with_results = 0

        for row in result.data:
            agent = row.get("agent_name") or "unknown"
            tier = row.get("agent_tier") or "unknown"

            by_agent[agent] = by_agent.get(agent, 0) + 1
            by_tier[tier] = by_tier.get(tier, 0) + 1

            if row.get("analysis_results") is not None:
                with_results += 1

        return {
            "total_activities": len(result.data),
            "by_agent": by_agent,
            "by_tier": by_tier,
            "with_results": with_results,
        }

    async def get_activities_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
        agent_type: Optional[str] = None,
        limit: int = 5000,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get activities within a time range.

        Args:
            start_time: Start of range
            end_time: End of range
            agent_type: Optional filter by agent name
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            Activities within range, ordered by timestamp
        """
        if not self.client:
            return []

        from src.repositories.provenance import apply_provenance_filter

        query = (
            self.client.table(self.table_name)
            .select("*")
            .gte("activity_timestamp", start_time.isoformat())
            .lte("activity_timestamp", end_time.isoformat())
        )

        if agent_type:
            query = query.eq("agent_name", agent_type)
        query = apply_provenance_filter(query, include_synthetic)

        result = await query.order("activity_timestamp", desc=True).limit(limit).execute()

        return [self._to_model(row) for row in result.data]
