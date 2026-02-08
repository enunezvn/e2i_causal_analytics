"""Drift Monitor Memory Hooks.

Integrates the Drift Monitor agent with the 4-Memory Architecture:
- Working Memory (Redis): Cache drift detection results with 24h TTL
- Episodic Memory (Supabase + pgvector): Store past drift detections for pattern recognition
- Semantic Memory (FalkorDB + Graphity): Store drift patterns and affected feature relationships

Contract: .claude/contracts/tier3-contracts.md
Architecture: .claude/contracts/4-MEMORY_ARCHITECTURE_CONTRACT.md

Memory Types Required (per CONTRACT_VALIDATION.md):
- Working: Yes (drift result caching)
- Episodic: Yes (drift detection history)
- Semantic: Yes (drift pattern relationships)
- Procedural: No (statistical computation, no LLM prompts to optimize)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class DriftDetectionContext:
    """Context retrieved from memory systems for drift detection.

    Contains relevant prior drift detections, feature patterns, and cached results
    to inform the current drift analysis.
    """

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "working_memory": self.working_memory,
            "episodic_context": self.episodic_context,
            "semantic_context": self.semantic_context,
            "retrieval_timestamp": self.retrieval_timestamp.isoformat(),
        }


@dataclass
class DriftDetectionRecord:
    """Episodic memory record for drift detection results.

    Stores the full drift detection analysis for future retrieval and pattern matching.
    """

    # Identity
    record_id: str
    session_id: str
    timestamp: datetime

    # Detection context
    query: str
    model_id: Optional[str]
    features_monitored: List[str]
    time_window: str
    brand: Optional[str]

    # Results
    overall_drift_score: float
    features_with_drift: List[str]
    data_drift_count: int
    model_drift_count: int
    concept_drift_count: int
    alert_count: int
    max_severity: str

    # Summary
    drift_summary: str
    recommended_actions: List[str]

    # Metadata (Contract field names)
    total_latency_ms: int  # Was detection_latency_ms
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "model_id": self.model_id,
            "features_monitored": self.features_monitored,
            "time_window": self.time_window,
            "brand": self.brand,
            "overall_drift_score": self.overall_drift_score,
            "features_with_drift": self.features_with_drift,
            "data_drift_count": self.data_drift_count,
            "model_drift_count": self.model_drift_count,
            "concept_drift_count": self.concept_drift_count,
            "alert_count": self.alert_count,
            "max_severity": self.max_severity,
            "drift_summary": self.drift_summary,
            "recommended_actions": self.recommended_actions,
            "total_latency_ms": self.total_latency_ms,
            "warnings": self.warnings,
        }


@dataclass
class DriftPatternRecord:
    """Semantic memory record for drift patterns.

    Stores relationships between features, drift types, and models
    for pattern recognition in the knowledge graph.
    """

    # Identity
    pattern_id: str
    timestamp: datetime

    # Pattern definition
    feature_name: str
    drift_type: str  # "data" | "model" | "concept"
    model_id: Optional[str]
    severity: str

    # Statistical details
    test_statistic: float
    p_value: float
    psi_score: Optional[float]

    # Context
    baseline_period: str
    current_period: str
    brand: Optional[str]

    # Relationships
    co_drifting_features: List[str]
    related_models: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pattern_id": self.pattern_id,
            "timestamp": self.timestamp.isoformat(),
            "feature_name": self.feature_name,
            "drift_type": self.drift_type,
            "model_id": self.model_id,
            "severity": self.severity,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "psi_score": self.psi_score,
            "baseline_period": self.baseline_period,
            "current_period": self.current_period,
            "brand": self.brand,
            "co_drifting_features": self.co_drifting_features,
            "related_models": self.related_models,
        }


# ============================================================================
# Memory Hooks Class
# ============================================================================


class DriftMonitorMemoryHooks:
    """Memory hooks for Drift Monitor agent.

    Provides integration with the 4-Memory Architecture:
    - Working Memory: Redis-based caching for drift results
    - Episodic Memory: Supabase storage for drift detection history
    - Semantic Memory: FalkorDB for drift pattern relationships

    Usage:
        hooks = get_drift_monitor_memory_hooks()
        context = await hooks.get_context(session_id, query, features)
        await hooks.cache_drift_result(session_id, result)
        await hooks.store_drift_detection(session_id, result, state)
    """

    # Cache TTL: 24 hours (consistent with other agents)
    CACHE_TTL_SECONDS = 86400

    def __init__(self):
        """Initialize with lazy-loaded memory clients."""
        self._working_memory = None
        self._semantic_memory = None
        self._supabase_client = None

    # ========================================================================
    # Lazy-Loaded Memory Clients
    # ========================================================================

    @property
    def working_memory(self):
        """Lazy-load Redis working memory client (port 6382)."""
        if self._working_memory is None:
            try:
                from src.memory.working_memory import get_working_memory

                self._working_memory = get_working_memory()
                logger.debug("Working memory client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize working memory: {e}")
        return self._working_memory

    @property
    def semantic_memory(self):
        """Lazy-load FalkorDB semantic memory client (port 6380)."""
        if self._semantic_memory is None:
            try:
                from src.memory.semantic_memory import get_semantic_memory

                self._semantic_memory = get_semantic_memory()
                logger.debug("Semantic memory client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize semantic memory: {e}")
        return self._semantic_memory

    @property
    def supabase_client(self):
        """Lazy-load Supabase client for episodic memory."""
        if self._supabase_client is None:
            try:
                from src.repositories import get_supabase_client

                self._supabase_client = get_supabase_client()
                logger.debug("Supabase client initialized for episodic memory")
            except Exception as e:
                logger.warning(f"Could not initialize Supabase client: {e}")
        return self._supabase_client

    # ========================================================================
    # Context Retrieval
    # ========================================================================

    async def get_context(
        self,
        session_id: str,
        query: str,
        features: List[str],
        model_id: Optional[str] = None,
        brand: Optional[str] = None,
        max_episodic: int = 5,
        max_semantic: int = 10,
    ) -> DriftDetectionContext:
        """Retrieve context from all three memory systems.

        Args:
            session_id: Current session ID
            query: User query or description
            features: Features being monitored
            model_id: Optional model ID filter
            brand: Optional brand filter
            max_episodic: Maximum episodic records to retrieve
            max_semantic: Maximum semantic patterns to retrieve

        Returns:
            DriftDetectionContext with relevant prior knowledge
        """
        context = DriftDetectionContext(session_id=session_id)

        # Retrieve from each memory system (parallel-safe, each handles errors)
        context.working_memory = await self._get_working_memory_context(
            session_id, features, model_id
        )
        context.episodic_context = await self._get_episodic_context(
            features, model_id, brand, max_episodic
        )
        context.semantic_context = await self._get_semantic_context(
            features, model_id, max_semantic
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_patterns={len(context.semantic_context.get('patterns', []))}"
        )

        return context

    async def _get_working_memory_context(
        self,
        session_id: str,
        features: List[str],
        model_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve cached drift results from working memory.

        Args:
            session_id: Current session ID
            features: Features to check for cached results
            model_id: Optional model ID filter

        Returns:
            List of cached drift results
        """
        if not self.working_memory:
            return []

        cached_results = []
        try:
            # Check for session-specific cache
            session_key = f"drift_monitor:session:{session_id}"
            session_cache = await self.working_memory.get(session_key)
            if session_cache:
                cached_results.append(session_cache)

            # Check for feature-specific caches
            for feature in features[:10]:  # Limit to prevent cache storm
                feature_key = f"drift_monitor:feature:{feature}"
                if model_id:
                    feature_key = f"{feature_key}:model:{model_id}"
                feature_cache = await self.working_memory.get(feature_key)
                if feature_cache:
                    cached_results.append(feature_cache)

        except Exception as e:
            logger.warning(f"Error retrieving working memory context: {e}")

        return cached_results

    async def _get_episodic_context(
        self,
        features: List[str],
        model_id: Optional[str] = None,
        brand: Optional[str] = None,
        max_records: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve similar past drift detections from episodic memory.

        Args:
            features: Features being monitored
            model_id: Optional model ID filter
            brand: Optional brand filter
            max_records: Maximum records to retrieve

        Returns:
            List of past drift detection records
        """
        if not self.supabase_client:
            return []

        try:
            # Query agent_activities for drift_monitor records
            query = self.supabase_client.table("agent_activities").select("*")

            # Filter by agent
            query = query.eq("agent_type", "drift_monitor")

            # Filter by model if specified
            if model_id:
                query = query.contains("metadata", {"model_id": model_id})

            # Filter by brand if specified
            if brand:
                query = query.contains("metadata", {"brand": brand})

            # Order by recency and limit
            query = query.order("created_at", desc=True).limit(max_records)

            response = query.execute()

            if response.data:
                # Filter for feature overlap
                relevant_records = []
                for record in response.data:
                    record_features = record.get("metadata", {}).get("features_monitored", [])
                    if set(features) & set(record_features):
                        relevant_records.append(record)
                return relevant_records[:max_records]

        except Exception as e:
            logger.warning(f"Error retrieving episodic context: {e}")

        return []

    async def _get_semantic_context(
        self,
        features: List[str],
        model_id: Optional[str] = None,
        max_patterns: int = 10,
    ) -> Dict[str, Any]:
        """Retrieve drift patterns from semantic memory.

        Args:
            features: Features to find patterns for
            model_id: Optional model ID filter
            max_patterns: Maximum patterns to retrieve

        Returns:
            Dict with patterns, feature relationships, and statistics
        """
        if not self.semantic_memory:
            return {"patterns": [], "feature_clusters": [], "drift_history": {}}

        semantic_context: Dict[str, Any] = {
            "patterns": [],
            "feature_clusters": [],
            "drift_history": {},
        }

        try:
            # Query for drift patterns related to features
            for feature in features[:5]:  # Limit queries
                query = f"""
                MATCH (f:Feature {{name: '{feature}'}})-[:HAS_DRIFT]->(d:DriftPattern)
                RETURN d.drift_type as drift_type,
                       d.severity as severity,
                       d.last_detected as last_detected,
                       d.occurrence_count as occurrence_count
                ORDER BY d.last_detected DESC
                LIMIT {max_patterns}
                """
                try:
                    result = await self.semantic_memory.query(query)
                    if result:
                        patterns_list: List[Dict[str, Any]] = semantic_context["patterns"]
                        for row in result:
                            patterns_list.append(
                                {
                                    "feature": feature,
                                    "drift_type": row.get("drift_type"),
                                    "severity": row.get("severity"),
                                    "last_detected": row.get("last_detected"),
                                    "occurrence_count": row.get("occurrence_count"),
                                }
                            )
                except Exception as query_error:
                    logger.debug(f"Pattern query failed for {feature}: {query_error}")

            # Query for co-drifting feature clusters
            if len(features) >= 2:
                cluster_query = f"""
                MATCH (f1:Feature)-[:CO_DRIFTS_WITH]->(f2:Feature)
                WHERE f1.name IN {features[:10]}
                RETURN f1.name as feature1, f2.name as feature2,
                       count(*) as co_occurrence
                ORDER BY co_occurrence DESC
                LIMIT 10
                """
                try:
                    cluster_result = await self.semantic_memory.query(cluster_query)
                    if cluster_result:
                        semantic_context["feature_clusters"] = [
                            {
                                "feature1": row.get("feature1"),
                                "feature2": row.get("feature2"),
                                "co_occurrence": row.get("co_occurrence"),
                            }
                            for row in cluster_result
                        ]
                except Exception as cluster_error:
                    logger.debug(f"Cluster query failed: {cluster_error}")

        except Exception as e:
            logger.warning(f"Error retrieving semantic context: {e}")

        return semantic_context

    # ========================================================================
    # Working Memory Caching
    # ========================================================================

    async def cache_drift_result(
        self,
        session_id: str,
        result: Dict[str, Any],
        features: Optional[List[str]] = None,
        model_id: Optional[str] = None,
    ) -> bool:
        """Cache drift detection result in working memory.

        Args:
            session_id: Session ID for the detection
            result: Drift detection result to cache
            features: Optional features to create individual caches
            model_id: Optional model ID for cache key

        Returns:
            True if caching successful
        """
        if not self.working_memory:
            return False

        try:
            # Cache session result
            session_key = f"drift_monitor:session:{session_id}"
            cache_data = {
                "result": result,
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "model_id": model_id,
            }
            await self.working_memory.set(session_key, cache_data, ttl=self.CACHE_TTL_SECONDS)

            # Cache per-feature results for quick lookup
            if features:
                features_with_drift = result.get("features_with_drift", [])
                for feature in features[:20]:  # Limit feature caches
                    feature_key = f"drift_monitor:feature:{feature}"
                    if model_id:
                        feature_key = f"{feature_key}:model:{model_id}"

                    feature_data = {
                        "has_drift": feature in features_with_drift,
                        "overall_score": result.get("overall_drift_score", 0.0),
                        "cached_at": datetime.now(timezone.utc).isoformat(),
                        "session_id": session_id,
                    }
                    await self.working_memory.set(
                        feature_key, feature_data, ttl=self.CACHE_TTL_SECONDS
                    )

            logger.debug(f"Cached drift result for session {session_id}")
            return True

        except Exception as e:
            logger.warning(f"Error caching drift result: {e}")
            return False

    async def get_cached_drift_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached drift result for session.

        Args:
            session_id: Session ID to lookup

        Returns:
            Cached result or None if not found
        """
        if not self.working_memory:
            return None

        try:
            session_key = f"drift_monitor:session:{session_id}"
            cached = await self.working_memory.get(session_key)
            if cached:
                return cast(Dict[str, Any], cached.get("result"))
        except Exception as e:
            logger.warning(f"Error retrieving cached drift result: {e}")

        return None

    async def invalidate_cache(
        self,
        session_id: Optional[str] = None,
        feature: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> bool:
        """Invalidate cached drift results.

        Args:
            session_id: Optional session to invalidate
            feature: Optional feature to invalidate
            model_id: Optional model ID for feature invalidation

        Returns:
            True if invalidation successful
        """
        if not self.working_memory:
            return False

        try:
            if session_id:
                session_key = f"drift_monitor:session:{session_id}"
                await self.working_memory.delete(session_key)
                logger.debug(f"Invalidated cache for session {session_id}")

            if feature:
                feature_key = f"drift_monitor:feature:{feature}"
                if model_id:
                    feature_key = f"{feature_key}:model:{model_id}"
                await self.working_memory.delete(feature_key)
                logger.debug(f"Invalidated cache for feature {feature}")

            return True

        except Exception as e:
            logger.warning(f"Error invalidating cache: {e}")
            return False

    # ========================================================================
    # Episodic Memory Storage
    # ========================================================================

    async def store_drift_detection(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[str]:
        """Store drift detection in episodic memory.

        Args:
            session_id: Session ID for the detection
            result: Drift detection result
            state: Full state from drift detection

        Returns:
            Record ID if stored successfully, None otherwise
        """
        if not self.supabase_client:
            return None

        try:
            # Generate record ID
            record_id = self._generate_record_id(session_id, result)

            # Determine max severity from alerts
            alerts = result.get("alerts", [])
            max_severity = "none"
            if alerts:
                severity_order = ["none", "low", "medium", "high", "critical"]
                for alert in alerts:
                    alert_severity = alert.get("severity", "none")
                    if severity_order.index(alert_severity) > severity_order.index(max_severity):
                        max_severity = alert_severity

            # Count drift types
            data_drift_count = len(result.get("data_drift_results", []))
            model_drift_count = len(result.get("model_drift_results", []))
            concept_drift_count = len(result.get("concept_drift_results", []))

            # Create record
            record = DriftDetectionRecord(
                record_id=record_id,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                query=state.get("query", ""),
                model_id=state.get("model_id"),
                features_monitored=state.get("features_to_monitor", []),
                time_window=state.get("time_window", "7d"),
                brand=state.get("brand"),
                overall_drift_score=result.get("overall_drift_score", 0.0),
                features_with_drift=result.get("features_with_drift", []),
                data_drift_count=data_drift_count,
                model_drift_count=model_drift_count,
                concept_drift_count=concept_drift_count,
                alert_count=len(alerts),
                max_severity=max_severity,
                drift_summary=result.get("drift_summary", ""),
                recommended_actions=result.get("recommended_actions", []),
                total_latency_ms=result.get("total_latency_ms", 0),
                warnings=result.get("warnings", []),
            )

            # Store in agent_activities table
            activity_data = {
                "agent_type": "drift_monitor",
                "activity_type": "drift_detection",
                "session_id": session_id,
                "query_text": state.get("query", ""),
                "result_summary": result.get("drift_summary", ""),
                "metadata": record.to_dict(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            response = (
                self.supabase_client.table("agent_activities").insert(activity_data).execute()
            )

            if response.data:
                logger.info(f"Stored drift detection record {record_id}")
                return record_id

        except Exception as e:
            logger.warning(f"Error storing drift detection: {e}")

        return None

    # ========================================================================
    # Semantic Memory Storage
    # ========================================================================

    async def store_drift_pattern(
        self,
        feature: str,
        drift_type: str,
        severity: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
    ) -> bool:
        """Store drift pattern in semantic memory.

        Creates or updates nodes and relationships in the knowledge graph
        for the detected drift pattern.

        Args:
            feature: Feature name
            drift_type: Type of drift (data/model/concept)
            severity: Drift severity
            result: Full drift result
            state: Detection state

        Returns:
            True if stored successfully
        """
        if not self.semantic_memory:
            return False

        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            pattern_id = self._generate_pattern_id(feature, drift_type, state)

            # Find the specific drift result for this feature
            drift_results = []
            if drift_type == "data":
                drift_results = result.get("data_drift_results", [])
            elif drift_type == "model":
                drift_results = result.get("model_drift_results", [])
            elif drift_type == "concept":
                drift_results = result.get("concept_drift_results", [])

            feature_result: Dict[str, Any] = next((r for r in drift_results if r.get("feature") == feature), {})

            # Create/update Feature node
            feature_query = f"""
            MERGE (f:Feature {{name: '{feature}'}})
            ON CREATE SET f.created_at = '{timestamp}'
            SET f.last_drift_check = '{timestamp}'
            RETURN f
            """
            await self.semantic_memory.query(feature_query)

            # Create DriftPattern node
            pattern_query = f"""
            MERGE (d:DriftPattern {{pattern_id: '{pattern_id}'}})
            ON CREATE SET
                d.drift_type = '{drift_type}',
                d.first_detected = '{timestamp}',
                d.occurrence_count = 1
            ON MATCH SET
                d.occurrence_count = d.occurrence_count + 1
            SET
                d.severity = '{severity}',
                d.last_detected = '{timestamp}',
                d.test_statistic = {feature_result.get("test_statistic", 0.0)},
                d.p_value = {feature_result.get("p_value", 1.0)}
            RETURN d
            """
            await self.semantic_memory.query(pattern_query)

            # Create Feature -> DriftPattern relationship
            rel_query = f"""
            MATCH (f:Feature {{name: '{feature}'}})
            MATCH (d:DriftPattern {{pattern_id: '{pattern_id}'}})
            MERGE (f)-[:HAS_DRIFT]->(d)
            """
            await self.semantic_memory.query(rel_query)

            # Create co-drifting relationships
            features_with_drift = result.get("features_with_drift", [])
            for other_feature in features_with_drift:
                if other_feature != feature:
                    co_drift_query = f"""
                    MERGE (f1:Feature {{name: '{feature}'}})
                    MERGE (f2:Feature {{name: '{other_feature}'}})
                    MERGE (f1)-[:CO_DRIFTS_WITH]->(f2)
                    """
                    await self.semantic_memory.query(co_drift_query)

            # Link to model if specified
            model_id = state.get("model_id")
            if model_id:
                model_query = f"""
                MERGE (m:Model {{model_id: '{model_id}'}})
                MATCH (d:DriftPattern {{pattern_id: '{pattern_id}'}})
                MERGE (d)-[:AFFECTS_MODEL]->(m)
                """
                await self.semantic_memory.query(model_query)

            logger.debug(f"Stored drift pattern for {feature} ({drift_type})")
            return True

        except Exception as e:
            logger.warning(f"Error storing drift pattern: {e}")
            return False

    async def store_all_drift_patterns(
        self,
        result: Dict[str, Any],
        state: Dict[str, Any],
    ) -> int:
        """Store all detected drift patterns in semantic memory.

        Args:
            result: Full drift detection result
            state: Detection state

        Returns:
            Number of patterns stored
        """
        stored_count = 0

        features_with_drift = result.get("features_with_drift", [])
        if not features_with_drift:
            return 0

        # Process each drift type
        for drift_type, results_key in [
            ("data", "data_drift_results"),
            ("model", "model_drift_results"),
            ("concept", "concept_drift_results"),
        ]:
            drift_results = result.get(results_key, [])
            for drift_result in drift_results:
                feature = drift_result.get("feature")
                if feature and drift_result.get("drift_detected"):
                    severity = drift_result.get("severity", "low")
                    success = await self.store_drift_pattern(
                        feature=feature,
                        drift_type=drift_type,
                        severity=severity,
                        result=result,
                        state=state,
                    )
                    if success:
                        stored_count += 1

        return stored_count

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _generate_record_id(self, session_id: str, result: Dict[str, Any]) -> str:
        """Generate unique record ID for episodic memory."""
        content = f"{session_id}:{result.get('overall_drift_score', 0)}"
        content += f":{len(result.get('features_with_drift', []))}"
        content += f":{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_pattern_id(self, feature: str, drift_type: str, state: Dict[str, Any]) -> str:
        """Generate unique pattern ID for semantic memory."""
        model_id = state.get("model_id", "no_model")
        brand = state.get("brand", "no_brand")
        content = f"{feature}:{drift_type}:{model_id}:{brand}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================================================
# Contribute to Memory Function
# ============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    session_id: str,
) -> Dict[str, int]:
    """Contribute drift detection results to CognitiveRAG's memory systems.

    This function is called after drift detection completes to:
    1. Cache results in working memory (24h TTL)
    2. Store detection record in episodic memory
    3. Store drift patterns in semantic memory

    Args:
        result: Drift detection result
        state: Full detection state
        session_id: Session identifier

    Returns:
        Dict with counts of items stored in each memory system
    """
    hooks = get_drift_monitor_memory_hooks()
    counts = {"working": 0, "episodic": 0, "semantic": 0}

    # 1. Cache in working memory
    features = state.get("features_to_monitor", [])
    model_id = state.get("model_id")
    cache_success = await hooks.cache_drift_result(
        session_id=session_id,
        result=result,
        features=features,
        model_id=model_id,
    )
    if cache_success:
        counts["working"] = 1 + len(features[:20])  # Session + feature caches

    # 2. Store in episodic memory
    record_id = await hooks.store_drift_detection(
        session_id=session_id,
        result=result,
        state=state,
    )
    if record_id:
        counts["episodic"] = 1

    # 3. Store in semantic memory (only if drift detected)
    if result.get("features_with_drift"):
        patterns_stored = await hooks.store_all_drift_patterns(
            result=result,
            state=state,
        )
        counts["semantic"] = patterns_stored

    logger.info(
        f"Contributed to memory: working={counts['working']}, "
        f"episodic={counts['episodic']}, semantic={counts['semantic']}"
    )

    return counts


# ============================================================================
# Singleton Access
# ============================================================================

_memory_hooks: Optional[DriftMonitorMemoryHooks] = None


def get_drift_monitor_memory_hooks() -> DriftMonitorMemoryHooks:
    """Get singleton DriftMonitorMemoryHooks instance.

    Returns:
        Shared DriftMonitorMemoryHooks instance
    """
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = DriftMonitorMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset singleton for testing purposes."""
    global _memory_hooks
    _memory_hooks = None
