"""
Experiment Monitor Agent Memory Hooks
=====================================

Memory integration hooks for the Experiment Monitor agent's memory architecture.

The Experiment Monitor agent uses these hooks to:
1. Retrieve context from working memory (Redis - cached experiment status)
2. Search episodic memory (Supabase - historical monitoring events)
3. Store significant alerts and monitoring results in episodic memory
4. Cache monitoring results for dashboard refresh

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class MonitoringContext:
    """Context retrieved from memory systems for experiment monitoring."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    cached_status: Optional[Dict[str, Any]] = None
    historical_alerts: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MonitoringRecord:
    """Record of a monitoring check for storage."""

    session_id: str
    experiments_checked: int
    healthy_count: int
    warning_count: int
    critical_count: int
    alert_count: int
    alert_types: List[str]
    check_latency_ms: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class ExperimentMonitorMemoryHooks:
    """
    Memory integration hooks for the Experiment Monitor agent.

    Provides methods to:
    - Retrieve cached experiment status from working memory
    - Cache monitoring results with short TTL (for dashboard refresh)
    - Store significant alerts in episodic memory
    - Track monitoring trends over time
    """

    # Cache TTL in seconds (3 minutes for experiment status)
    STATUS_CACHE_TTL = 180
    # Cache TTL for alerts (10 minutes)
    ALERT_CACHE_TTL = 600

    def __init__(self):
        """Initialize memory hooks with lazy-loaded clients."""
        self._working_memory = None

    # =========================================================================
    # LAZY-LOADED MEMORY CLIENTS
    # =========================================================================

    @property
    def working_memory(self):
        """Lazy-load Redis working memory (port 6382)."""
        if self._working_memory is None:
            try:
                from src.memory.working_memory import get_working_memory

                self._working_memory = get_working_memory()
                logger.debug("Working memory client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize working memory: {e}")
                self._working_memory = None
        return self._working_memory

    # =========================================================================
    # CONTEXT RETRIEVAL
    # =========================================================================

    async def get_context(
        self,
        session_id: str,
        experiment_ids: Optional[List[str]] = None,
        include_history: bool = False,
    ) -> MonitoringContext:
        """
        Retrieve context from working and episodic memory.

        Args:
            session_id: Session identifier for working memory lookup
            experiment_ids: Optional list of experiment IDs to filter by
            include_history: Whether to include historical alerts

        Returns:
            MonitoringContext with data from memory systems
        """
        context = MonitoringContext(session_id=session_id)

        # 1. Get working memory (session context)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Check for cached monitoring status
        context.cached_status = await self._get_cached_status(experiment_ids)

        # 3. Get historical alerts if requested
        if include_history:
            context.historical_alerts = await self._get_alert_history(experiment_ids)

        logger.debug(
            f"Retrieved monitoring context for session {session_id}: "
            f"cached={context.cached_status is not None}, "
            f"history={len(context.historical_alerts)}"
        )

        return context

    async def _get_working_memory_context(
        self,
        session_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversation from working memory."""
        if not self.working_memory:
            return []

        try:
            messages = await self.working_memory.get_messages(session_id, limit=limit)
            return cast(List[Dict[str, Any]], messages)
        except Exception as e:
            logger.warning(f"Failed to get working memory: {e}")
            return []

    async def _get_cached_status(
        self,
        experiment_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get recently cached monitoring status."""
        if not self.working_memory:
            return None

        try:
            redis = await self.working_memory.get_client()

            # Build cache key based on experiment IDs
            if experiment_ids:
                cache_key = f"experiment_monitor:status:{':'.join(sorted(experiment_ids)[:5])}"
            else:
                cache_key = "experiment_monitor:status:all"

            cached = await redis.get(cache_key)
            if cached:
                return cast(Dict[str, Any], json.loads(cached))
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached status: {e}")
            return None

    async def _get_alert_history(
        self,
        experiment_ids: Optional[List[str]] = None,
        hours: int = 24,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get historical alerts from episodic memory."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            # Build query based on experiment IDs
            if experiment_ids:
                query_text = f"experiment monitor alert {' '.join(experiment_ids[:3])}"
            else:
                query_text = "experiment monitor alert srm enrollment fidelity"

            filters = EpisodicSearchFilters(
                event_type="experiment_alert_generated",
                agent_name="experiment_monitor",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.5,
                include_entity_context=False,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get alert history: {e}")
            return []

    # =========================================================================
    # MONITORING STATUS CACHING (Working Memory)
    # =========================================================================

    async def cache_monitoring_status(
        self,
        experiment_ids: Optional[List[str]],
        monitoring_result: Dict[str, Any],
    ) -> bool:
        """
        Cache monitoring status in working memory.

        Args:
            experiment_ids: Experiment IDs that were monitored (None for all)
            monitoring_result: Monitoring output to cache

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()

            # Build cache key
            if experiment_ids:
                cache_key = f"experiment_monitor:status:{':'.join(sorted(experiment_ids)[:5])}"
            else:
                cache_key = "experiment_monitor:status:all"

            await redis.setex(
                cache_key,
                self.STATUS_CACHE_TTL,
                json.dumps(monitoring_result, default=str),
            )

            logger.debug(f"Cached monitoring status: {cache_key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache monitoring status: {e}")
            return False

    async def cache_alert(
        self,
        alert: Dict[str, Any],
    ) -> bool:
        """
        Cache individual alert for quick retrieval.

        Args:
            alert: Alert dictionary to cache

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()

            alert_id = alert.get("alert_id", "unknown")
            cache_key = f"experiment_monitor:alert:{alert_id}"

            await redis.setex(
                cache_key,
                self.ALERT_CACHE_TTL,
                json.dumps(alert, default=str),
            )

            return True
        except Exception as e:
            logger.warning(f"Failed to cache alert: {e}")
            return False

    async def invalidate_cache(
        self,
        experiment_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Invalidate cached monitoring status.

        Args:
            experiment_ids: Specific experiments to invalidate, or None for all

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()

            if experiment_ids:
                cache_key = f"experiment_monitor:status:{':'.join(sorted(experiment_ids)[:5])}"
                await redis.delete(cache_key)
            else:
                # Delete the "all" key
                await redis.delete("experiment_monitor:status:all")

            return True
        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")
            return False

    # =========================================================================
    # ALERT STORAGE (Episodic Memory)
    # =========================================================================

    async def store_alert(
        self,
        session_id: str,
        alert: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[str]:
        """
        Store alert in episodic memory.

        Args:
            session_id: Session identifier
            alert: Alert dictionary
            state: Monitor state

        Returns:
            Memory ID if stored, None otherwise
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicMemoryInput,
                insert_episodic_memory_with_text,
            )

            alert_type = alert.get("alert_type", "unknown")
            severity = alert.get("severity", "info")
            experiment_name = alert.get("experiment_name", "Unknown")
            experiment_id = alert.get("experiment_id", "")
            message = alert.get("message", "")

            # Build description
            description = (
                f"Experiment alert ({severity}): {alert_type} detected "
                f"for '{experiment_name}' ({experiment_id}). {message}"
            )

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="experiment_alert_generated",
                event_subtype=f"alert_{alert_type}",
                description=description,
                raw_content={
                    "alert_id": alert.get("alert_id"),
                    "alert_type": alert_type,
                    "severity": severity,
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_name,
                    "message": message,
                    "details": alert.get("details", {}),
                    "recommended_action": alert.get("recommended_action", ""),
                    "timestamp": alert.get("timestamp"),
                },
                entities=None,
                outcome_type="alert_generated",
                agent_name="experiment_monitor",
                importance_score=self._calculate_alert_importance(alert),
                e2i_refs=None,
            )

            # Insert with auto-generated embedding
            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=f"{alert_type} {severity} {experiment_name} {message}",
                session_id=session_id,
            )

            logger.info(f"Stored alert in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store alert in episodic memory: {e}")
            return None

    async def store_monitoring_check(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[str]:
        """
        Store significant monitoring check in episodic memory.

        Only stores checks with:
        - Critical issues detected
        - Multiple alerts generated
        - First check in a while (periodic)

        Args:
            session_id: Session identifier
            result: Monitoring output
            state: Monitor state

        Returns:
            Memory ID if stored, None otherwise
        """
        # Only store significant events
        if not self._is_significant_check(result):
            logger.debug("Monitoring check not significant enough to store")
            return None

        try:
            from src.memory.episodic_memory import (
                EpisodicMemoryInput,
                insert_episodic_memory_with_text,
            )

            experiments_checked = result.get("experiments_checked", 0)
            healthy_count = result.get("healthy_count", 0)
            warning_count = result.get("warning_count", 0)
            critical_count = result.get("critical_count", 0)
            alerts = result.get("alerts", [])

            # Build description
            description = (
                f"Experiment monitoring: {experiments_checked} experiments checked. "
                f"Healthy: {healthy_count}, Warnings: {warning_count}, Critical: {critical_count}. "
                f"Alerts generated: {len(alerts)}."
            )

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="experiment_monitoring_completed",
                event_subtype="significant_check",
                description=description,
                raw_content={
                    "experiments_checked": experiments_checked,
                    "healthy_count": healthy_count,
                    "warning_count": warning_count,
                    "critical_count": critical_count,
                    "alert_count": len(alerts),
                    "alert_types": list({a.get("alert_type", "") for a in alerts}),
                    "check_latency_ms": result.get("check_latency_ms", 0),
                    "summary": result.get("monitor_summary", ""),
                },
                entities=None,
                outcome_type="monitoring_completed",
                agent_name="experiment_monitor",
                importance_score=self._calculate_check_importance(result),
                e2i_refs=None,
            )

            # Insert with auto-generated embedding
            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=f"{state.get('query', '')} {description}",
                session_id=session_id,
            )

            logger.info(f"Stored monitoring check in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store monitoring check: {e}")
            return None

    def _is_significant_check(
        self,
        result: Dict[str, Any],
    ) -> bool:
        """Determine if monitoring check is significant enough to store."""
        # Critical experiments are always significant
        if result.get("critical_count", 0) > 0:
            return True

        # Multiple alerts are significant
        alerts = result.get("alerts", [])
        if len(alerts) >= 2:
            return True

        # Multiple warnings are significant
        if result.get("warning_count", 0) >= 3:
            return True

        return False

    def _calculate_alert_importance(
        self,
        alert: Dict[str, Any],
    ) -> float:
        """Calculate importance score for alert."""
        severity = alert.get("severity", "info")

        severity_scores = {
            "critical": 0.9,
            "warning": 0.6,
            "info": 0.3,
        }

        return severity_scores.get(severity, 0.3)

    def _calculate_check_importance(
        self,
        result: Dict[str, Any],
    ) -> float:
        """Calculate importance score for monitoring check."""
        critical_count = result.get("critical_count", 0)
        warning_count = result.get("warning_count", 0)
        alert_count = len(result.get("alerts", []))

        # Base importance
        base_importance = 0.3

        # Boost for critical issues
        critical_boost = min(0.4, critical_count * 0.2)

        # Boost for warnings
        warning_boost = min(0.2, warning_count * 0.05)

        # Boost for alerts
        alert_boost = min(0.1, alert_count * 0.02)

        return float(min(1.0, base_importance + critical_boost + warning_boost + alert_boost))

    # =========================================================================
    # HISTORICAL ANALYSIS
    # =========================================================================

    async def get_experiment_alert_history(
        self,
        experiment_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get alert history for a specific experiment.

        Args:
            experiment_id: Experiment ID
            limit: Maximum results

        Returns:
            List of historical alerts for the experiment
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"experiment alert {experiment_id}"

            filters = EpisodicSearchFilters(
                event_type="experiment_alert_generated",
                agent_name="experiment_monitor",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.5,
                include_entity_context=False,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get experiment alert history: {e}")
            return []

    async def get_srm_history(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get history of SRM alerts.

        Args:
            limit: Maximum results

        Returns:
            List of historical SRM alerts
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = "sample ratio mismatch SRM chi-squared p-value"

            filters = EpisodicSearchFilters(
                event_type="experiment_alert_generated",
                agent_name="experiment_monitor",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.5,
                include_entity_context=False,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get SRM history: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION (Per Specialist Document)
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[ExperimentMonitorMemoryHooks] = None,
    session_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute monitoring results to CognitiveRAG's memory systems.

    This is the primary interface for storing experiment monitoring
    results in the memory architecture.

    Args:
        result: ExperimentMonitorOutput dictionary
        state: ExperimentMonitorState dictionary
        memory_hooks: Optional memory hooks instance (creates new if not provided)
        session_id: Session identifier (generates UUID if not provided)

    Returns:
        Dictionary with counts of stored memories:
        - alerts_stored: Number of alerts stored in episodic memory
        - check_stored: 1 if check stored (significant events only), 0 otherwise
        - working_cached: 1 if cached, 0 otherwise
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_experiment_monitor_memory_hooks()

    if session_id is None:
        session_id = str(uuid.uuid4())

    counts = {
        "alerts_stored": 0,
        "check_stored": 0,
        "working_cached": 0,
    }

    # Skip storage if check failed
    if state.get("status") == "failed":
        logger.info("Skipping memory storage for failed monitoring check")
        return counts

    experiment_ids = state.get("experiment_ids")

    # 1. Always cache in working memory
    cached = await memory_hooks.cache_monitoring_status(experiment_ids, result)
    if cached:
        counts["working_cached"] = 1

    # 2. Store alerts in episodic memory
    alerts = result.get("alerts", [])
    for alert in alerts:
        # Only store warning and critical alerts
        if alert.get("severity") in ("warning", "critical"):
            memory_id = await memory_hooks.store_alert(
                session_id=session_id,
                alert=alert,
                state=state,
            )
            if memory_id:
                counts["alerts_stored"] += 1

            # Also cache the alert
            await memory_hooks.cache_alert(alert)

    # 3. Store monitoring check in episodic memory (only significant events)
    memory_id = await memory_hooks.store_monitoring_check(
        session_id=session_id,
        result=result,
        state=state,
    )
    if memory_id:
        counts["check_stored"] = 1

    logger.info(
        f"Memory contribution complete: "
        f"alerts={counts['alerts_stored']}, "
        f"check={counts['check_stored']}, "
        f"working_cached={counts['working_cached']}"
    )

    return counts


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[ExperimentMonitorMemoryHooks] = None


def get_experiment_monitor_memory_hooks() -> ExperimentMonitorMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = ExperimentMonitorMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
