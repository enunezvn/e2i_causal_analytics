"""
Data Preparer Agent Memory Hooks
=================================

Memory integration hooks for the Data Preparer agent's tri-memory architecture.

The Data Preparer agent uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - similar past QC reports)
3. Query semantic memory (FalkorDB - data quality patterns, leakage incidents)
4. Store QC reports for future retrieval and RAG

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class DataPreparationContext:
    """Context retrieved from all memory systems for data preparation."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QCReportRecord:
    """Record of a QC report for storage in episodic memory."""

    session_id: str
    experiment_id: str
    report_id: str
    qc_status: str
    overall_score: float
    gate_passed: bool
    leakage_detected: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class DataPreparerMemoryHooks:
    """
    Memory integration hooks for the Data Preparer agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache QC reports in working memory (24h TTL)
    - Store QC reports in episodic memory for future retrieval
    - Store data quality patterns in semantic memory for knowledge graph
    """

    CACHE_TTL_SECONDS = 86400

    def __init__(self):
        """Initialize memory hooks with lazy-loaded clients."""
        self._working_memory = None
        self._semantic_memory = None

    @property
    def working_memory(self):
        """Lazy-load Redis working memory."""
        if self._working_memory is None:
            try:
                from src.memory.working_memory import get_working_memory

                self._working_memory = get_working_memory()
                logger.debug("Working memory client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize working memory: {e}")
                self._working_memory = None
        return self._working_memory

    @property
    def semantic_memory(self):
        """Lazy-load FalkorDB semantic memory."""
        if self._semantic_memory is None:
            try:
                from src.memory.semantic_memory import get_semantic_memory

                self._semantic_memory = get_semantic_memory()
                logger.debug("Semantic memory client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic memory: {e}")
                self._semantic_memory = None
        return self._semantic_memory

    # =========================================================================
    # CONTEXT RETRIEVAL
    # =========================================================================

    async def get_context(
        self,
        session_id: str,
        experiment_id: str,
        data_source: Optional[str] = None,
        max_episodic_results: int = 5,
    ) -> DataPreparationContext:
        """
        Retrieve context from all three memory systems.

        Args:
            session_id: Session identifier
            experiment_id: Experiment identifier
            data_source: Optional data source for filtering
            max_episodic_results: Maximum episodic memories to retrieve

        Returns:
            DataPreparationContext with data from all memory systems
        """
        context = DataPreparationContext(session_id=session_id)

        context.working_memory = await self._get_working_memory_context(session_id)
        context.episodic_context = await self._get_episodic_context(
            experiment_id=experiment_id,
            data_source=data_source,
            limit=max_episodic_results,
        )
        context.semantic_context = await self._get_semantic_context(
            experiment_id=experiment_id,
            data_source=data_source,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_patterns={len(context.semantic_context.get('quality_patterns', []))}"
        )

        return context

    async def _get_working_memory_context(
        self, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversation from working memory."""
        if not self.working_memory:
            return []

        try:
            messages = await self.working_memory.get_messages(session_id, limit=limit)
            return messages
        except Exception as e:
            logger.warning(f"Failed to get working memory: {e}")
            return []

    async def _get_episodic_context(
        self,
        experiment_id: str,
        data_source: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar past QC reports."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"data quality QC report {experiment_id} {data_source or ''}"

            filters = EpisodicSearchFilters(
                event_type="qc_report_completed",
                agent_name="data_preparer",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.5,
                include_entity_context=True,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get episodic context: {e}")
            return []

    async def _get_semantic_context(
        self,
        experiment_id: str,
        data_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get semantic memory context for data quality patterns."""
        if not self.semantic_memory:
            return {}

        try:
            context = {
                "quality_patterns": [],
                "leakage_incidents": [],
                "data_source_history": [],
            }

            # Query past leakage incidents
            leakage = self.semantic_memory.query(
                "MATCH (l:LeakageIncident) RETURN l ORDER BY l.timestamp DESC LIMIT 10"
            )
            context["leakage_incidents"] = leakage

            # Query data source history
            if data_source:
                history = self.semantic_memory.query(
                    f"MATCH (d:DataSource {{name: '{data_source}'}})-[:HAS_QC]->(q:QCReport) "
                    f"RETURN q ORDER BY q.timestamp DESC LIMIT 10"
                )
                context["data_source_history"] = history

            return context
        except Exception as e:
            logger.warning(f"Failed to get semantic context: {e}")
            return {}

    # =========================================================================
    # STORAGE: WORKING MEMORY (CACHE)
    # =========================================================================

    async def cache_qc_report(
        self,
        session_id: str,
        qc_report: Dict[str, Any],
    ) -> bool:
        """Cache QC report in working memory."""
        if not self.working_memory:
            return False

        try:
            cache_key = f"data_preparer:qc_report:{session_id}"
            await self.working_memory.set(
                cache_key,
                json.dumps(qc_report),
                ex=self.CACHE_TTL_SECONDS,
            )
            logger.debug(f"Cached QC report for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache QC report: {e}")
            return False

    # =========================================================================
    # STORAGE: EPISODIC MEMORY
    # =========================================================================

    async def store_qc_report(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """Store QC report in episodic memory."""
        try:
            from src.memory.episodic_memory import store_episodic_memory

            content = {
                "experiment_id": state.get("experiment_id"),
                "report_id": result.get("report_id"),
                "qc_status": result.get("qc_status"),
                "overall_score": state.get("overall_score"),
                "gate_passed": result.get("gate_passed"),
                "leakage_detected": state.get("leakage_detected"),
                "total_samples": result.get("total_samples"),
                "train_samples": result.get("train_samples"),
                "validation_samples": result.get("validation_samples"),
                "test_samples": result.get("test_samples"),
                "holdout_samples": result.get("holdout_samples"),
                "dimension_scores": {
                    "completeness": state.get("completeness_score"),
                    "validity": state.get("validity_score"),
                    "consistency": state.get("consistency_score"),
                    "uniqueness": state.get("uniqueness_score"),
                    "timeliness": state.get("timeliness_score"),
                },
                "blocking_issues": state.get("blocking_issues", []),
            }

            summary = (
                f"QC Report: {result.get('qc_status', 'unknown')}. "
                f"Score: {state.get('overall_score', 0):.2f}. "
                f"Gate: {'PASSED' if result.get('gate_passed') else 'FAILED'}. "
                f"Leakage: {'DETECTED' if state.get('leakage_detected') else 'none'}."
            )

            memory_id = await store_episodic_memory(
                session_id=session_id,
                event_type="qc_report_completed",
                agent_name="data_preparer",
                summary=summary,
                raw_content=content,
                brand=brand,
                region=region,
            )

            logger.info(f"Stored QC report in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store QC report: {e}")
            return None

    # =========================================================================
    # STORAGE: SEMANTIC MEMORY
    # =========================================================================

    async def store_data_quality_pattern(
        self,
        experiment_id: str,
        data_source: str,
        qc_status: str,
        overall_score: float,
        leakage_detected: bool,
        blocking_issues: List[str],
    ) -> bool:
        """Store data quality pattern in semantic memory."""
        if not self.semantic_memory:
            logger.warning("Semantic memory not available")
            return False

        try:
            # Create data source node
            self.semantic_memory.add_e2i_entity(
                entity_type="DataSource",
                entity_id=f"ds:{data_source}",
                properties={
                    "name": data_source,
                    "agent": "data_preparer",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Create QC report node
            self.semantic_memory.add_e2i_entity(
                entity_type="QCReport",
                entity_id=f"qc:{experiment_id}",
                properties={
                    "experiment_id": experiment_id,
                    "status": qc_status,
                    "overall_score": overall_score,
                    "leakage_detected": leakage_detected,
                    "blocking_issues_count": len(blocking_issues),
                    "agent": "data_preparer",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Create relationship
            self.semantic_memory.add_relationship(
                from_entity_id=f"ds:{data_source}",
                to_entity_id=f"qc:{experiment_id}",
                relationship_type="HAS_QC",
                properties={"agent": "data_preparer"},
            )

            # Store leakage incident if detected
            if leakage_detected:
                self.semantic_memory.add_e2i_entity(
                    entity_type="LeakageIncident",
                    entity_id=f"leak:{experiment_id}",
                    properties={
                        "experiment_id": experiment_id,
                        "data_source": data_source,
                        "agent": "data_preparer",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

            logger.info(f"Stored data quality pattern: {experiment_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store data quality pattern: {e}")
            return False


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[DataPreparerMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute data preparation results to memory systems.

    Args:
        result: DataPreparerState output fields
        state: Full DataPreparerState
        memory_hooks: Optional memory hooks instance
        session_id: Session identifier
        brand: Optional brand context
        region: Optional region context

    Returns:
        Dictionary with counts of stored memories
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_data_preparer_memory_hooks()

    if session_id is None:
        session_id = state.get("session_id") or str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "semantic_stored": 0,
        "working_cached": 0,
    }

    # Skip if error occurred
    if state.get("error"):
        logger.info("Skipping memory storage due to error")
        return counts

    # 1. Cache in working memory
    qc_report = {
        "report_id": result.get("report_id"),
        "qc_status": result.get("qc_status"),
        "gate_passed": result.get("gate_passed"),
        "overall_score": state.get("overall_score"),
    }
    cached = await memory_hooks.cache_qc_report(session_id, qc_report)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_qc_report(
        session_id=session_id,
        result=result,
        state=state,
        brand=brand,
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 3. Store pattern in semantic memory
    data_source = state.get("data_source")
    experiment_id = state.get("experiment_id")
    if data_source and experiment_id:
        stored = await memory_hooks.store_data_quality_pattern(
            experiment_id=experiment_id,
            data_source=data_source,
            qc_status=result.get("qc_status", "unknown"),
            overall_score=state.get("overall_score", 0),
            leakage_detected=state.get("leakage_detected", False),
            blocking_issues=state.get("blocking_issues", []),
        )
        if stored:
            counts["semantic_stored"] = 1

    logger.info(
        f"Memory contribution complete: "
        f"episodic={counts['episodic_stored']}, "
        f"semantic={counts['semantic_stored']}, "
        f"working_cached={counts['working_cached']}"
    )

    return counts


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[DataPreparerMemoryHooks] = None


def get_data_preparer_memory_hooks() -> DataPreparerMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = DataPreparerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
