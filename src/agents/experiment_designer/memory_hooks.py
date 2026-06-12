"""Experiment Designer Memory Hooks.

Integrates the Experiment Designer agent with the 4-Memory Architecture:
- Working Memory (Redis): Cache experiment designs with 24h TTL
- Episodic Memory (Supabase + pgvector): Store past experiment designs for learning

Contract: .claude/contracts/tier3-contracts.md
Architecture: .claude/contracts/4-MEMORY_ARCHITECTURE_CONTRACT.md

Memory Types Required (per CONTRACT_VALIDATION.md):
- Working: Yes (design caching)
- Episodic: Yes (experiment design history)
- Semantic: No (experiments are self-contained, no knowledge graph needed)
- Procedural: No (LLM prompts handled by DSPy integration)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class ExperimentDesignContext:
    """Context retrieved from memory systems for experiment design.

    Contains relevant prior experiment designs and cached results
    to inform the current design process.
    """

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "working_memory": self.working_memory,
            "episodic_context": self.episodic_context,
            "retrieval_timestamp": self.retrieval_timestamp.isoformat(),
        }


@dataclass
class ExperimentDesignRecord:
    """Episodic memory record for experiment design results.

    Stores the full experiment design for future retrieval and learning.
    """

    # Identity
    record_id: str
    session_id: str
    timestamp: datetime

    # Design context
    business_question: str
    brand: Optional[str]
    constraints: Dict[str, Any]

    # Design outputs
    design_type: str
    design_rationale: str
    randomization_unit: str
    randomization_method: str

    # Power analysis
    required_sample_size: int
    achieved_power: float
    duration_estimate_days: int

    # Validity assessment
    overall_validity_score: float
    validity_confidence: str
    threat_count: int
    critical_threat_count: int

    # Iteration metadata
    redesign_iterations: int
    total_latency_ms: int
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "business_question": self.business_question,
            "brand": self.brand,
            "constraints": self.constraints,
            "design_type": self.design_type,
            "design_rationale": self.design_rationale,
            "randomization_unit": self.randomization_unit,
            "randomization_method": self.randomization_method,
            "required_sample_size": self.required_sample_size,
            "achieved_power": self.achieved_power,
            "duration_estimate_days": self.duration_estimate_days,
            "overall_validity_score": self.overall_validity_score,
            "validity_confidence": self.validity_confidence,
            "threat_count": self.threat_count,
            "critical_threat_count": self.critical_threat_count,
            "redesign_iterations": self.redesign_iterations,
            "total_latency_ms": self.total_latency_ms,
            "warnings": self.warnings,
        }


@dataclass
class ValidityThreatRecord:
    """Record for tracking validity threats across experiments.

    Enables learning from past validity issues to proactively
    identify similar threats in new designs.
    """

    # Identity
    threat_id: str
    experiment_record_id: str
    timestamp: datetime

    # Threat details
    threat_type: str  # "internal" | "external" | "construct" | "statistical_conclusion"
    threat_name: str
    severity: str  # "low" | "medium" | "high" | "critical"
    description: str

    # Context
    design_type: str
    business_question_keywords: List[str]
    affected_outcomes: List[str]

    # Resolution
    mitigation_possible: bool
    mitigation_strategy: Optional[str]
    mitigation_effectiveness: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "threat_id": self.threat_id,
            "experiment_record_id": self.experiment_record_id,
            "timestamp": self.timestamp.isoformat(),
            "threat_type": self.threat_type,
            "threat_name": self.threat_name,
            "severity": self.severity,
            "description": self.description,
            "design_type": self.design_type,
            "business_question_keywords": self.business_question_keywords,
            "affected_outcomes": self.affected_outcomes,
            "mitigation_possible": self.mitigation_possible,
            "mitigation_strategy": self.mitigation_strategy,
            "mitigation_effectiveness": self.mitigation_effectiveness,
        }


# ============================================================================
# Memory Hooks Class
# ============================================================================


class ExperimentDesignerMemoryHooks:
    """Memory hooks for Experiment Designer agent.

    Provides integration with the 4-Memory Architecture:
    - Working Memory: Redis-based caching for experiment designs
    - Episodic Memory: Supabase storage for design history and learning

    Usage:
        hooks = get_experiment_designer_memory_hooks()
        context = await hooks.get_context(session_id, business_question)
        await hooks.cache_experiment_design(session_id, result)
        await hooks.store_experiment_design(session_id, result, state)
    """

    # Cache TTL: 24 hours (consistent with other agents)
    CACHE_TTL_SECONDS = 86400

    def __init__(self):
        """Initialize with lazy-loaded memory clients."""
        self._working_memory = None
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

    # ------------------------------------------------------------------
    # Working-memory raw-client helpers (#883 PR B): RedisWorkingMemory has
    # no get/set/delete methods — the hook predated that API, so every cache
    # call AttributeError'd into its except (surfaced by the agent-path
    # wiring proof). Route through the raw client like the sibling hooks.
    # ------------------------------------------------------------------

    async def _wm_get_json(self, key: str) -> Optional[Dict[str, Any]]:
        redis = await self.working_memory.get_client()
        raw = await redis.get(key)
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    async def _wm_set_json(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        redis = await self.working_memory.get_client()
        await redis.setex(key, ttl, json.dumps(value, default=str))

    async def _wm_delete(self, key: str) -> None:
        redis = await self.working_memory.get_client()
        await redis.delete(key)

    # ========================================================================
    # Hybrid Retriever Wire-in (Phase 2 finishing, issue #373)
    # ========================================================================

    async def get_hybrid_context(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        max_staleness: Optional[float] = 0.0,
    ) -> List[Any]:
        """Retrieve cross-source context via the hybrid retriever.

        Wraps :func:`src.rag.retriever.hybrid_search` with the experiment_designer
        agent_name default filter and freshness-only retrieval
        (``max_staleness=0.0``). Phase 2 finishing per
        ``.claude/plans/e2i_memory_subsystems_implementation_plan.md``
        §Recommended-sequencing item 1.

        Args:
            query: Search query text.
            k: Number of results to return.
            filters: Optional caller filters; merged with agent default
                ``{"agent_name": "experiment_designer"}`` (caller overrides).
            max_staleness: Freshness ceiling. Default ``0.0`` excludes
                invalidated rows under Decision 3 = KEEP BINARY.

        Returns:
            List of :class:`RetrievalResult` from the hybrid retriever.
        """
        from src.rag.retriever import hybrid_search

        merged_filters: Dict[str, Any] = {"agent_name": "experiment_designer"}
        if filters:
            merged_filters.update(filters)

        return await hybrid_search(
            query=query,
            k=k,
            filters=merged_filters,
            max_staleness=max_staleness,
        )

    # ========================================================================
    # Context Retrieval
    # ========================================================================

    async def get_context(
        self,
        session_id: str,
        business_question: str,
        brand: Optional[str] = None,
        design_type: Optional[str] = None,
        max_episodic: int = 5,
    ) -> ExperimentDesignContext:
        """Retrieve context from memory systems.

        Args:
            session_id: Current session ID
            business_question: Business question being designed for
            brand: Optional brand filter
            design_type: Optional design type filter
            max_episodic: Maximum episodic records to retrieve

        Returns:
            ExperimentDesignContext with relevant prior knowledge
        """
        context = ExperimentDesignContext(session_id=session_id)

        # Retrieve from each memory system
        context.working_memory = await self._get_working_memory_context(
            session_id, business_question
        )
        context.episodic_context = await self._get_episodic_context(
            business_question, brand, design_type, max_episodic
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}"
        )

        return context

    async def _get_working_memory_context(
        self,
        session_id: str,
        business_question: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve cached experiment designs from working memory.

        Args:
            session_id: Current session ID
            business_question: Business question for lookup

        Returns:
            List of cached experiment designs
        """
        if not self.working_memory:
            return []

        cached_results = []
        try:
            # Check for session-specific cache
            session_key = f"experiment_designer:session:{session_id}"
            session_cache = await self._wm_get_json(session_key)
            if session_cache:
                cached_results.append(session_cache)

            # Check for question-specific cache (using hash)
            question_hash = self._hash_question(business_question)
            question_key = f"experiment_designer:question:{question_hash}"
            question_cache = await self._wm_get_json(question_key)
            if question_cache:
                cached_results.append(question_cache)

        except Exception as e:
            logger.warning(f"Error retrieving working memory context: {e}")

        return cached_results

    async def _get_episodic_context(
        self,
        business_question: str,
        brand: Optional[str] = None,
        design_type: Optional[str] = None,
        max_records: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve similar past experiment designs from episodic memory.

        Args:
            business_question: Business question to find similar designs for
            brand: Optional brand filter
            design_type: Optional design type filter
            max_records: Maximum records to retrieve

        Returns:
            List of past experiment design records
        """
        if not self.supabase_client:
            return []

        try:
            # Query agent_activities for experiment_designer records using its
            # REAL columns (#883 §5): the original filters (.eq("agent_type")
            # / .contains("metadata")) named nonexistent columns, so every
            # call errored server-side, was swallowed below, and this reader
            # returned [] forever.
            query = self.supabase_client.table("agent_activities").select("*")

            # Filter by agent and the design-record activity rows
            query = query.eq("agent_name", "experiment_designer")
            query = query.eq("activity_type", "experiment_design")

            # Filter by brand if specified (record.to_dict embeds brand)
            if brand:
                query = query.contains("analysis_results", {"brand": brand})

            # Filter by design type if specified
            if design_type:
                query = query.contains("analysis_results", {"design_type": design_type})

            # Order by recency and limit
            query = query.order("created_at", desc=True).limit(max_records * 2)

            response = query.execute()

            if response.data:
                # Score by relevance to business question
                scored_records = []
                question_words = set(business_question.lower().split())

                for record in response.data:
                    analysis = record.get("analysis_results") or {}
                    record_question = (analysis.get("business_question") or "").lower()
                    record_words = set(record_question.split())

                    # Simple keyword overlap scoring
                    overlap = len(question_words & record_words)
                    if overlap > 0:
                        scored_records.append((overlap, record))

                # Sort by score and return top matches
                scored_records.sort(key=lambda x: x[0], reverse=True)
                return [r[1] for r in scored_records[:max_records]]

        except Exception as e:
            logger.warning(f"Error retrieving episodic context: {e}")

        return []

    async def get_similar_validity_threats(
        self,
        design_type: str,
        max_threats: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve past validity threats for similar design types.

        Helps proactively identify threats based on historical data.

        Args:
            design_type: Type of experimental design
            max_threats: Maximum threats to retrieve

        Returns:
            List of past validity threats
        """
        if not self.supabase_client:
            return []

        try:
            # Query for past experiments with this design type using the REAL
            # columns (#883 §5; both the design rows and the dedicated
            # validity_threats batch rows carry analysis_results.design_type +
            # .validity_threats, and the dedup below collapses duplicates).
            query = (
                self.supabase_client.table("agent_activities")
                .select("analysis_results")
                .eq("agent_name", "experiment_designer")
                .contains("analysis_results", {"design_type": design_type})
                .order("created_at", desc=True)
                .limit(20)
            )

            response = query.execute()

            if response.data:
                # Extract and deduplicate threats
                threats_seen = set()
                unique_threats = []

                for record in response.data:
                    analysis = record.get("analysis_results") or {}
                    for threat in analysis.get("validity_threats", []):
                        threat_key = f"{threat.get('threat_type')}:{threat.get('threat_name')}"
                        if threat_key not in threats_seen:
                            threats_seen.add(threat_key)
                            unique_threats.append(threat)

                            if len(unique_threats) >= max_threats:
                                return unique_threats

                return unique_threats

        except Exception as e:
            logger.warning(f"Error retrieving validity threats: {e}")

        return []

    # ========================================================================
    # Working Memory Caching
    # ========================================================================

    async def cache_experiment_design(
        self,
        session_id: str,
        result: Dict[str, Any],
        business_question: Optional[str] = None,
    ) -> bool:
        """Cache experiment design in working memory.

        Args:
            session_id: Session ID for the design
            result: Experiment design result to cache
            business_question: Optional question for additional cache key

        Returns:
            True if caching successful
        """
        if not self.working_memory:
            return False

        try:
            # Cache session result
            session_key = f"experiment_designer:session:{session_id}"
            cache_data = {
                "result": result,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._wm_set_json(session_key, cache_data, self.CACHE_TTL_SECONDS)

            # Also cache by question hash for reuse
            if business_question:
                question_hash = self._hash_question(business_question)
                question_key = f"experiment_designer:question:{question_hash}"
                await self._wm_set_json(question_key, cache_data, self.CACHE_TTL_SECONDS)

            logger.debug(f"Cached experiment design for session {session_id}")
            return True

        except Exception as e:
            logger.warning(f"Error caching experiment design: {e}")
            return False

    async def get_cached_experiment_design(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached experiment design for session.

        Args:
            session_id: Session ID to lookup

        Returns:
            Cached result or None if not found
        """
        if not self.working_memory:
            return None

        try:
            session_key = f"experiment_designer:session:{session_id}"
            cached = await self._wm_get_json(session_key)
            if cached:
                return cast(Dict[str, Any], cached.get("result"))
        except Exception as e:
            logger.warning(f"Error retrieving cached experiment design: {e}")

        return None

    async def invalidate_cache(
        self,
        session_id: Optional[str] = None,
        business_question: Optional[str] = None,
    ) -> bool:
        """Invalidate cached experiment designs.

        Args:
            session_id: Optional session to invalidate
            business_question: Optional question cache to invalidate

        Returns:
            True if invalidation successful
        """
        if not self.working_memory:
            return False

        try:
            if session_id:
                session_key = f"experiment_designer:session:{session_id}"
                await self._wm_delete(session_key)
                logger.debug(f"Invalidated cache for session {session_id}")

            if business_question:
                question_hash = self._hash_question(business_question)
                question_key = f"experiment_designer:question:{question_hash}"
                await self._wm_delete(question_key)
                logger.debug(f"Invalidated cache for question hash {question_hash}")

            return True

        except Exception as e:
            logger.warning(f"Error invalidating cache: {e}")
            return False

    # ========================================================================
    # Episodic Memory Storage
    # ========================================================================

    async def store_experiment_design(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        brand: Optional[str] = None,
    ) -> Optional[str]:
        """Store experiment design in episodic memory.

        Args:
            session_id: Session ID for the design
            result: Experiment design result
            state: Full state from design workflow
            brand: Optional brand context

        Returns:
            Record ID if stored successfully, None otherwise
        """
        if not self.supabase_client:
            return None

        try:
            # Generate record ID
            record_id = self._generate_record_id(session_id, result)

            # Extract power analysis data
            power_analysis = result.get("power_analysis", {})
            if power_analysis is None:
                power_analysis = {}

            # Count validity threats
            validity_threats = result.get("validity_threats", [])
            threat_count = len(validity_threats)
            critical_threat_count = sum(
                1 for t in validity_threats if t.get("severity") == "critical"
            )

            # Create record
            record = ExperimentDesignRecord(
                record_id=record_id,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                business_question=state.get("business_question", ""),
                brand=brand,
                constraints=state.get("constraints", {}),
                design_type=result.get("design_type", "RCT"),
                design_rationale=result.get("design_rationale", ""),
                randomization_unit=result.get("randomization_unit", "individual"),
                randomization_method=result.get("randomization_method", "simple"),
                required_sample_size=power_analysis.get("required_sample_size", 0),
                achieved_power=power_analysis.get("achieved_power", 0.0),
                duration_estimate_days=result.get("duration_estimate_days", 0),
                overall_validity_score=result.get("overall_validity_score", 0.0),
                validity_confidence=result.get("validity_confidence", "low"),
                threat_count=threat_count,
                critical_threat_count=critical_threat_count,
                redesign_iterations=result.get("redesign_iterations", 0),
                total_latency_ms=result.get("total_latency_ms", 0),
                warnings=result.get("warnings", []),
            )

            # Prepare validity threats for storage
            validity_threats_data = [
                {
                    "threat_type": t.get("threat_type", ""),
                    "threat_name": t.get("threat_name", ""),
                    "severity": t.get("severity", "medium"),
                    "description": t.get("description", ""),
                    "mitigation_possible": t.get("mitigation_possible", True),
                    "mitigation_strategy": t.get("mitigation_strategy"),
                }
                for t in validity_threats
            ]

            # Store in agent_activities using its REAL columns (#883 §5). The
            # original payload wrote 5 nonexistent columns (agent_type /
            # session_id / query_text / result_summary / metadata) and omitted
            # the defaultless NOT NULL PK activity_id + NOT NULL
            # activity_timestamp — PGRST204/23502 on every call, swallowed
            # below, so NO design was ever stored. Schema SSOT:
            # database/core/e2i_ml_complete_v3_schema.sql:608 (re-verified
            # against the live DB).
            activity_data = {
                "activity_id": f"expd_{record_id}",  # VARCHAR(30): 5 + 16 hex
                "agent_name": "experiment_designer",
                "agent_tier": "monitoring",  # tier 3 (agent_tier_mapping)
                "activity_timestamp": datetime.now(timezone.utc).isoformat(),
                "activity_type": "experiment_design",
                "processing_duration_ms": result.get("total_latency_ms", 0),
                "input_data": {
                    "session_id": session_id,
                    "business_question": state.get("business_question", ""),
                    "constraints": state.get("constraints", {}),
                },
                "analysis_results": {
                    **record.to_dict(),
                    "validity_threats": validity_threats_data,
                    "treatments": result.get("treatments", []),
                    "outcomes": result.get("outcomes", []),
                },
                "status": "completed",
            }

            response = (
                self.supabase_client.table("agent_activities").insert(activity_data).execute()
            )

            if response.data:
                logger.info(f"Stored experiment design record {record_id}")
                return record_id

        except Exception as e:
            logger.warning(f"Error storing experiment design: {e}")

        return None

    async def store_validity_threats(
        self,
        experiment_record_id: str,
        validity_threats: List[Dict[str, Any]],
        design_type: str,
        business_question: str,
        session_id: Optional[str] = None,
    ) -> int:
        """Store validity threats for learning.

        Args:
            experiment_record_id: Parent experiment record ID
            validity_threats: List of validity threats
            design_type: Experiment design type
            business_question: Business question context

        Returns:
            Number of threats actually persisted (0 when the insert did not
            land — never a fabricated count).
        """
        if not self.supabase_client or not validity_threats:
            return 0

        keywords = self._extract_keywords(business_question)

        try:
            threat_records = [
                ValidityThreatRecord(
                    threat_id=self._generate_threat_id(experiment_record_id, threat),
                    experiment_record_id=experiment_record_id,
                    timestamp=datetime.now(timezone.utc),
                    threat_type=threat.get("threat_type", "internal"),
                    threat_name=threat.get("threat_name", ""),
                    severity=threat.get("severity", "medium"),
                    description=threat.get("description", ""),
                    design_type=design_type,
                    business_question_keywords=keywords,
                    affected_outcomes=threat.get("affected_outcomes", []),
                    mitigation_possible=threat.get("mitigation_possible", True),
                    mitigation_strategy=threat.get("mitigation_strategy"),
                    mitigation_effectiveness=threat.get("effectiveness_rating"),
                )
                for threat in validity_threats
            ]

            # #883 §5: the original implementation constructed the records,
            # made NO database call, yet returned len(threats) — a fabricated
            # "N threats stored" success. Persist the batch as a dedicated
            # threat-specific agent_activities row (real columns only) and
            # return the count ONLY when the insert actually landed. The
            # reader (get_similar_validity_threats) dedupes by
            # threat_type:threat_name, so the copy embedded in the design row
            # cannot double-count.
            batch_id = hashlib.sha256(
                f"{experiment_record_id}:threats:{datetime.now(timezone.utc).isoformat()}".encode()
            ).hexdigest()[:16]
            activity_data = {
                "activity_id": f"expdt_{batch_id}",  # VARCHAR(30): 6 + 16 hex
                "agent_name": "experiment_designer",
                "agent_tier": "monitoring",
                "activity_timestamp": datetime.now(timezone.utc).isoformat(),
                "activity_type": "validity_threats",
                "records_processed": len(threat_records),
                "input_data": {
                    "session_id": session_id,
                    "experiment_record_id": experiment_record_id,
                    "business_question_keywords": keywords,
                },
                "analysis_results": {
                    "design_type": design_type,
                    "validity_threats": [r.to_dict() for r in threat_records],
                },
                "status": "completed",
            }

            response = (
                self.supabase_client.table("agent_activities").insert(activity_data).execute()
            )
            if response.data:
                logger.info(
                    f"Stored {len(threat_records)} validity threats for "
                    f"experiment {experiment_record_id}"
                )
                return len(threat_records)

        except Exception as e:
            logger.warning(f"Error storing validity threats: {e}")

        return 0

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _hash_question(self, question: str) -> str:
        """Generate hash for business question caching."""
        normalized = question.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _generate_record_id(self, session_id: str, result: Dict[str, Any]) -> str:
        """Generate unique record ID for episodic memory."""
        content = f"{session_id}:{result.get('design_type', 'RCT')}"
        content += f":{result.get('overall_validity_score', 0)}"
        content += f":{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_threat_id(self, experiment_id: str, threat: Dict[str, Any]) -> str:
        """Generate unique threat ID."""
        content = f"{experiment_id}:{threat.get('threat_type', '')}"
        content += f":{threat.get('threat_name', '')}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from business question for similarity matching."""
        # Simple keyword extraction (stopword removal)
        stopwords = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "what",
            "which",
            "who",
            "whom",
        }

        words = question.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return keywords[:10]  # Limit to 10 keywords


# ============================================================================
# Contribute to Memory Function
# ============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    session_id: str,
    brand: Optional[str] = None,
) -> Dict[str, int]:
    """Contribute experiment design results to CognitiveRAG's memory systems.

    This function is called after experiment design completes to:
    1. Cache results in working memory (24h TTL)
    2. Store design record in episodic memory

    Args:
        result: Experiment design result
        state: Full design state
        session_id: Session identifier
        brand: Optional brand context

    Returns:
        Dict with counts of items stored in each memory system
    """
    hooks = get_experiment_designer_memory_hooks()
    counts = {"working": 0, "episodic": 0}

    # 1. Cache in working memory
    business_question = state.get("business_question", "")
    cache_success = await hooks.cache_experiment_design(
        session_id=session_id,
        result=result,
        business_question=business_question,
    )
    if cache_success:
        counts["working"] = 2  # Session cache + question cache

    # 2. Store in episodic memory
    record_id = await hooks.store_experiment_design(
        session_id=session_id,
        result=result,
        state=state,
        brand=brand,
    )
    if record_id:
        counts["episodic"] = 1

        # Also track validity threats for learning
        validity_threats = result.get("validity_threats", [])
        if validity_threats:
            threats_stored = await hooks.store_validity_threats(
                experiment_record_id=record_id,
                validity_threats=validity_threats,
                design_type=result.get("design_type", "RCT"),
                business_question=business_question,
                session_id=session_id,
            )
            counts["episodic"] += threats_stored

    logger.info(
        f"Contributed to memory: working={counts['working']}, episodic={counts['episodic']}"
    )

    return counts


# ============================================================================
# Singleton Access
# ============================================================================

_memory_hooks: Optional[ExperimentDesignerMemoryHooks] = None


def get_experiment_designer_memory_hooks() -> ExperimentDesignerMemoryHooks:
    """Get singleton ExperimentDesignerMemoryHooks instance.

    Returns:
        Shared ExperimentDesignerMemoryHooks instance
    """
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = ExperimentDesignerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset singleton for testing purposes."""
    global _memory_hooks
    _memory_hooks = None
