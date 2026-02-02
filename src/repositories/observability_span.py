"""
Observability Span Repository.

Handles storage and retrieval of observability spans from ml_observability_spans table.
Provides methods for span persistence, querying, and latency statistics.

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.agents.ml_foundation.observability_connector.models import (
    AgentNameEnum,
    AgentTierEnum,
    LatencyStats,
    ObservabilitySpan,
    QualityMetrics,
)
from src.repositories.base import BaseRepository


class ObservabilitySpanRepository(BaseRepository[ObservabilitySpan]):
    """
    Repository for ml_observability_spans table.

    Supports:
    - Single and batch span insertion
    - Time-based span queries
    - Trace reconstruction
    - Agent-specific queries
    - Latency statistics from v_agent_latency_summary view
    - Retention cleanup

    Table schema:
    - id (UUID PK)
    - trace_id, span_id (unique constraint)
    - agent_name, agent_tier
    - started_at, ended_at, duration_ms
    - model_name, input_tokens, output_tokens, total_tokens
    - status, error_type, error_message
    - fallback_used, fallback_chain
    - attributes (JSONB)
    - experiment_id, training_run_id, deployment_id (FK)
    - user_id, session_id
    """

    table_name = "ml_observability_spans"
    model_class = ObservabilitySpan

    # =========================================================================
    # INSERT OPERATIONS
    # =========================================================================

    async def insert_span(self, span: ObservabilitySpan) -> Optional[ObservabilitySpan]:
        """
        Insert a single observability span.

        Args:
            span: ObservabilitySpan instance to insert

        Returns:
            Inserted span with ID, or None if insert failed
        """
        if not self.client:
            return None

        try:
            data = span.to_db_dict()
            # Note: Using sync Supabase client, no await needed
            result = self.client.table(self.table_name).insert(data).execute()

            if result.data:
                return self._to_model(result.data[0])
            return None
        except Exception as e:
            # Log error and return None for graceful degradation
            import logging

            logging.getLogger(__name__).warning(f"insert_span failed: {e}")
            return None

    async def insert_spans_batch(self, spans: List[ObservabilitySpan]) -> Dict[str, Any]:
        """
        Insert multiple spans in a batch operation.

        Args:
            spans: List of ObservabilitySpan instances

        Returns:
            Dict with insert results:
            {
                "success": bool,
                "inserted_count": int,
                "failed_count": int,
                "span_ids": List[str]
            }
        """
        if not self.client or not spans:
            return {
                "success": False,
                "inserted_count": 0,
                "failed_count": len(spans) if spans else 0,
                "span_ids": [],
            }

        try:
            data = [span.to_db_dict() for span in spans]
            # Note: Using sync Supabase client, no await needed
            result = self.client.table(self.table_name).insert(data).execute()

            inserted_count = len(result.data) if result.data else 0
            span_ids = [row.get("span_id", "") for row in (result.data or [])]

            return {
                "success": inserted_count == len(spans),
                "inserted_count": inserted_count,
                "failed_count": len(spans) - inserted_count,
                "span_ids": span_ids,
            }
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"insert_spans_batch failed: {e}")
            return {
                "success": False,
                "inserted_count": 0,
                "failed_count": len(spans),
                "span_ids": [],
            }

    # =========================================================================
    # TIME-BASED QUERIES
    # =========================================================================

    async def get_spans_by_time_window(
        self,
        window: str = "24h",
        agent_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 1000,
    ) -> List[ObservabilitySpan]:
        """
        Get spans within a time window.

        Args:
            window: Time window string (1h, 24h, 7d)
            agent_name: Optional filter by agent name
            status: Optional filter by status (success, error, timeout)
            limit: Maximum spans to return

        Returns:
            List of ObservabilitySpan instances
        """
        if not self.client:
            return []

        # Parse time window
        hours = self._parse_time_window(window)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        query = (
            self.client.table(self.table_name)
            .select("*")
            .gte("started_at", cutoff_time.isoformat())
        )

        if agent_name:
            query = query.eq("agent_name", agent_name)

        if status:
            query = query.eq("status", status)

        # Note: Using sync Supabase client, no await needed
        result = query.order("started_at", desc=True).limit(limit).execute()

        return [self._to_model(row) for row in (result.data or [])]

    def _parse_time_window(self, window: str) -> int:
        """
        Parse time window string to hours.

        Args:
            window: Time window string (1h, 24h, 7d, 30d)

        Returns:
            Number of hours
        """
        window = window.lower().strip()

        try:
            if window.endswith("h"):
                return int(window[:-1])
            elif window.endswith("d"):
                return int(window[:-1]) * 24
            elif window.endswith("w"):
                return int(window[:-1]) * 24 * 7
            else:
                # Default to 24 hours
                return 24
        except (ValueError, IndexError):
            # Invalid format, default to 24 hours
            return 24

    # =========================================================================
    # TRACE QUERIES
    # =========================================================================

    async def get_spans_by_trace_id(self, trace_id: str) -> List[ObservabilitySpan]:
        """
        Get all spans for a trace.

        Used for trace reconstruction and visualization.

        Args:
            trace_id: Trace identifier

        Returns:
            List of spans in the trace, ordered by start time
        """
        if not self.client:
            return []

        # Note: Using sync Supabase client, no await needed
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("trace_id", trace_id)
            .order("started_at", desc=False)
            .execute()
        )

        return [self._to_model(row) for row in (result.data or [])]

    async def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a trace.

        Args:
            trace_id: Trace identifier

        Returns:
            Dict with trace summary:
            {
                "trace_id": str,
                "span_count": int,
                "total_duration_ms": int,
                "agent_count": int,
                "error_count": int,
                "total_tokens": int
            }
        """
        spans = await self.get_spans_by_trace_id(trace_id)

        if not spans:
            return {
                "trace_id": trace_id,
                "span_count": 0,
                "total_duration_ms": 0,
                "agent_count": 0,
                "error_count": 0,
                "total_tokens": 0,
            }

        agents = set()
        error_count = 0
        total_tokens = 0
        min_start = spans[0].started_at
        max_end = spans[0].ended_at or spans[0].started_at

        for span in spans:
            agents.add(span.agent_name)
            if span.status.value == "error":
                error_count += 1
            if span.total_tokens:
                total_tokens += span.total_tokens
            if span.started_at < min_start:
                min_start = span.started_at
            if span.ended_at and span.ended_at > max_end:
                max_end = span.ended_at

        total_duration = int((max_end - min_start).total_seconds() * 1000)

        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "total_duration_ms": total_duration,
            "agent_count": len(agents),
            "error_count": error_count,
            "total_tokens": total_tokens,
        }

    # =========================================================================
    # AGENT QUERIES
    # =========================================================================

    async def get_spans_by_agent(
        self,
        agent_name: str,
        hours: int = 24,
        limit: int = 500,
    ) -> List[ObservabilitySpan]:
        """
        Get spans for a specific agent.

        Args:
            agent_name: Agent name
            hours: Number of hours to look back
            limit: Maximum spans

        Returns:
            List of spans for the agent
        """
        if not self.client:
            return []

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Note: Using sync Supabase client, no await needed
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_name", agent_name)
            .gte("started_at", cutoff_time.isoformat())
            .order("started_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in (result.data or [])]

    async def get_spans_by_tier(
        self,
        agent_tier: str,
        hours: int = 24,
        limit: int = 1000,
    ) -> List[ObservabilitySpan]:
        """
        Get spans for all agents in a tier.

        Args:
            agent_tier: Agent tier
            hours: Number of hours to look back
            limit: Maximum spans

        Returns:
            List of spans for agents in the tier
        """
        if not self.client:
            return []

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Note: Using sync Supabase client, no await needed
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_tier", agent_tier)
            .gte("started_at", cutoff_time.isoformat())
            .order("started_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in (result.data or [])]

    # =========================================================================
    # LATENCY STATISTICS (from v_agent_latency_summary view)
    # =========================================================================

    async def get_latency_stats(
        self,
        agent_name: Optional[str] = None,
    ) -> List[LatencyStats]:
        """
        Get latency statistics from v_agent_latency_summary view.

        The view provides pre-computed percentiles for the last 24 hours.

        Args:
            agent_name: Optional filter by agent name

        Returns:
            List of LatencyStats for each agent/tier
        """
        if not self.client:
            return []

        try:
            query = self.client.table("v_agent_latency_summary").select("*")

            if agent_name:
                query = query.eq("agent_name", agent_name)

            # Note: Using sync Supabase client, no await needed
            result = query.execute()

            stats_list = []
            for row in result.data or []:
                try:
                    # Map agent name string to enum
                    agent_enum = None
                    if row.get("agent_name"):
                        try:
                            agent_enum = AgentNameEnum(row["agent_name"])
                        except ValueError:
                            pass

                    tier_enum = None
                    if row.get("agent_tier"):
                        try:
                            tier_enum = AgentTierEnum(row["agent_tier"])
                        except ValueError:
                            pass

                    stats = LatencyStats(
                        agent_name=agent_enum,
                        agent_tier=tier_enum,
                        total_spans=row.get("total_spans", 0),
                        avg_duration_ms=float(row.get("avg_duration_ms", 0) or 0),
                        p50_ms=float(row.get("p50_ms", 0) or 0),
                        p95_ms=float(row.get("p95_ms", 0) or 0),
                        p99_ms=float(row.get("p99_ms", 0) or 0),
                        error_rate=float(row.get("error_rate", 0) or 0),
                        fallback_rate=float(row.get("fallback_rate", 0) or 0),
                        total_tokens_used=row.get("total_tokens_used", 0) or 0,
                    )
                    stats_list.append(stats)
                except Exception:
                    # Skip malformed rows
                    continue

            return stats_list
        except Exception:
            return []

    async def get_quality_metrics(
        self,
        time_window: str = "24h",
    ) -> QualityMetrics:
        """
        Compute quality metrics for a time window.

        Aggregates spans and computes overall system health metrics.

        Args:
            time_window: Time window string (1h, 24h, 7d)

        Returns:
            QualityMetrics instance
        """
        self._parse_time_window(time_window)
        spans = await self.get_spans_by_time_window(window=time_window, limit=10000)

        if not spans:
            return QualityMetrics(time_window=time_window)

        # Aggregate metrics
        total_spans = len(spans)
        success_count = 0
        error_count = 0
        timeout_count = 0
        fallback_count = 0
        total_tokens = 0
        durations = []

        by_agent: Dict[str, List[ObservabilitySpan]] = {}
        by_tier: Dict[str, List[ObservabilitySpan]] = {}

        for span in spans:
            status_val = span.status.value
            if status_val == "success":
                success_count += 1
            elif status_val == "error":
                error_count += 1
            elif status_val == "timeout":
                timeout_count += 1

            if span.fallback_used:
                fallback_count += 1

            if span.total_tokens:
                total_tokens += span.total_tokens

            if span.duration_ms:
                durations.append(span.duration_ms)

            # Group by agent
            agent_key = span.agent_name.value
            if agent_key not in by_agent:
                by_agent[agent_key] = []
            by_agent[agent_key].append(span)

            # Group by tier
            tier_key = span.agent_tier.value
            if tier_key not in by_tier:
                by_tier[tier_key] = []
            by_tier[tier_key].append(span)

        # Compute rates
        success_rate = success_count / total_spans if total_spans > 0 else 1.0
        error_rate = error_count / total_spans if total_spans > 0 else 0.0
        fallback_rate = fallback_count / total_spans if total_spans > 0 else 0.0

        # Compute percentiles
        avg_latency = sum(durations) / len(durations) if durations else 0.0
        sorted_durations = sorted(durations)
        p50 = self._percentile(sorted_durations, 0.5)
        p95 = self._percentile(sorted_durations, 0.95)
        p99 = self._percentile(sorted_durations, 0.99)

        # Build latency by agent
        latency_by_agent = {}
        for agent_name, agent_spans in by_agent.items():
            agent_durations = [s.duration_ms for s in agent_spans if s.duration_ms]
            agent_errors = sum(1 for s in agent_spans if s.status.value == "error")
            agent_fallbacks = sum(1 for s in agent_spans if s.fallback_used)
            agent_tokens = sum(s.total_tokens or 0 for s in agent_spans)

            if agent_durations:
                sorted_agent = sorted(agent_durations)
                latency_by_agent[agent_name] = LatencyStats(
                    total_spans=len(agent_spans),
                    avg_duration_ms=sum(agent_durations) / len(agent_durations),
                    p50_ms=self._percentile(sorted_agent, 0.5),
                    p95_ms=self._percentile(sorted_agent, 0.95),
                    p99_ms=self._percentile(sorted_agent, 0.99),
                    error_rate=agent_errors / len(agent_spans),
                    fallback_rate=agent_fallbacks / len(agent_spans),
                    total_tokens_used=agent_tokens,
                )

        # Build latency by tier
        latency_by_tier = {}
        for tier_name, tier_spans in by_tier.items():
            tier_durations = [s.duration_ms for s in tier_spans if s.duration_ms]
            tier_errors = sum(1 for s in tier_spans if s.status.value == "error")
            tier_fallbacks = sum(1 for s in tier_spans if s.fallback_used)
            tier_tokens = sum(s.total_tokens or 0 for s in tier_spans)

            if tier_durations:
                sorted_tier = sorted(tier_durations)
                latency_by_tier[tier_name] = LatencyStats(
                    total_spans=len(tier_spans),
                    avg_duration_ms=sum(tier_durations) / len(tier_durations),
                    p50_ms=self._percentile(sorted_tier, 0.5),
                    p95_ms=self._percentile(sorted_tier, 0.95),
                    p99_ms=self._percentile(sorted_tier, 0.99),
                    error_rate=tier_errors / len(tier_spans),
                    fallback_rate=tier_fallbacks / len(tier_spans),
                    total_tokens_used=tier_tokens,
                )

        metrics = QualityMetrics(
            time_window=time_window,
            total_spans=total_spans,
            success_count=success_count,
            error_count=error_count,
            timeout_count=timeout_count,
            success_rate=success_rate,
            error_rate=error_rate,
            fallback_rate=fallback_rate,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            total_tokens=total_tokens,
            avg_tokens_per_span=total_tokens / total_spans if total_spans > 0 else 0.0,
            latency_by_agent=latency_by_agent,
            latency_by_tier=latency_by_tier,
            status_distribution={
                "success": success_count,
                "error": error_count,
                "timeout": timeout_count,
            },
        )

        metrics.compute_quality_score()
        return metrics

    def _percentile(self, sorted_data: List[int], p: float) -> float:
        """
        Compute percentile from sorted data.

        Args:
            sorted_data: Sorted list of values
            p: Percentile (0.0-1.0)

        Returns:
            Percentile value
        """
        if not sorted_data:
            return 0.0

        n = len(sorted_data)
        k = (n - 1) * p
        f = int(k)
        c = f + 1 if f < n - 1 else f

        if f == c:
            return float(sorted_data[f])

        d0 = sorted_data[f] * (c - k)
        d1 = sorted_data[c] * (k - f)
        return d0 + d1

    # =========================================================================
    # RETENTION CLEANUP
    # =========================================================================

    async def delete_old_spans(
        self,
        retention_days: int = 30,
        batch_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Delete spans older than retention period.

        Args:
            retention_days: Number of days to retain
            batch_size: Maximum spans to delete per call

        Returns:
            Dict with deletion results:
            {
                "deleted_count": int,
                "cutoff_date": str
            }
        """
        if not self.client:
            return {"deleted_count": 0, "cutoff_date": ""}

        cutoff_time = datetime.now(timezone.utc) - timedelta(days=retention_days)

        try:
            # Note: Using sync Supabase client, no await needed
            result = (
                self.client.table(self.table_name)
                .delete()
                .lt("started_at", cutoff_time.isoformat())
                .limit(batch_size)
                .execute()
            )

            return {
                "deleted_count": len(result.data) if result.data else 0,
                "cutoff_date": cutoff_time.isoformat(),
            }
        except Exception:
            return {"deleted_count": 0, "cutoff_date": cutoff_time.isoformat()}

    # =========================================================================
    # ERROR ANALYSIS
    # =========================================================================

    async def get_error_spans(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> List[ObservabilitySpan]:
        """
        Get recent error spans for analysis.

        Args:
            hours: Number of hours to look back
            limit: Maximum spans

        Returns:
            List of error spans
        """
        if not self.client:
            return []

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Note: Using sync Supabase client, no await needed
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("status", "error")
            .gte("started_at", cutoff_time.isoformat())
            .order("started_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in (result.data or [])]

    async def get_fallback_spans(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> List[ObservabilitySpan]:
        """
        Get recent spans where fallback was used.

        Args:
            hours: Number of hours to look back
            limit: Maximum spans

        Returns:
            List of spans with fallback_used=True
        """
        if not self.client:
            return []

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Note: Using sync Supabase client, no await needed
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("fallback_used", True)
            .gte("started_at", cutoff_time.isoformat())
            .order("started_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in (result.data or [])]
