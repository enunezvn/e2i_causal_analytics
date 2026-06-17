"""
Cohort Constructor Agent Memory Hooks
=====================================

Memory integration hooks for the Cohort Constructor agent's tri-memory architecture.

The Cohort Constructor agent uses these hooks to:
1. Retrieve context from working memory (Redis - recent cohort configurations)
2. Search episodic memory (Supabase - similar past cohort constructions)
3. Query semantic memory (FalkorDB - eligibility rule relationships, brand patterns)
4. Store cohort results for future retrieval and RAG

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
class CohortConstructionContext:
    """Context retrieved from all memory systems for cohort construction."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CohortConstructionRecord:
    """Record of a cohort construction for storage in episodic memory."""

    session_id: str
    cohort_id: str
    cohort_name: str
    brand: str
    indication: str
    total_patients: int
    eligible_patients: int
    eligibility_rate: float
    criteria_count: int
    config_hash: str
    execution_time_ms: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class CohortConstructorMemoryHooks:
    """
    Memory integration hooks for the Cohort Constructor agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache cohort configurations in working memory (24h TTL)
    - Store cohort results in episodic memory for future retrieval
    - Store eligibility rule patterns in semantic memory for knowledge graph
    """

    # Cache TTL in seconds (24 hours)
    CACHE_TTL_SECONDS = 86400

    def __init__(self):
        """Initialize memory hooks with lazy-loaded clients."""
        self._working_memory = None
        self._semantic_memory = None

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

    @property
    def semantic_memory(self):
        """Lazy-load FalkorDB semantic memory (port 6381)."""
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
        brand: str,
        indication: Optional[str] = None,
        max_episodic_results: int = 5,
    ) -> CohortConstructionContext:
        """
        Retrieve context from all three memory systems.

        Args:
            session_id: Session identifier for working memory lookup
            brand: Brand name for filtering (Remibrutinib, Fabhalta, Kisqali)
            indication: Optional indication for filtering
            max_episodic_results: Maximum episodic memories to retrieve

        Returns:
            CohortConstructionContext with data from all memory systems
        """
        context = CohortConstructionContext(session_id=session_id)

        # 1. Get working memory (recent session context)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Get episodic memory (similar past cohort constructions)
        context.episodic_context = await self._get_episodic_context(
            brand=brand,
            indication=indication,
            limit=max_episodic_results,
        )

        # 3. Get semantic memory (existing eligibility patterns, brand relationships)
        context.semantic_context = await self._get_semantic_context(
            brand=brand,
            indication=indication,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_rules={len(context.semantic_context.get('eligibility_rules', []))}"
        )

        return context

    async def _get_working_memory_context(
        self,
        session_id: str,
        limit: int = 10,
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

    async def _get_episodic_context(
        self,
        brand: str,
        indication: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar past cohort constructions."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"cohort construction {brand} {indication or ''}"

            filters = EpisodicSearchFilters(
                event_type="cohort_construction_completed",
                agent_name="cohort_constructor",
                brand=brand,
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
        brand: str,
        indication: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get semantic memory context for eligibility rules."""
        if not self.semantic_memory:
            return {}

        try:
            context: Dict[str, Any] = {
                "eligibility_rules": [],
                "brand_patterns": [],
                "prior_cohorts": [],
            }

            # Query eligibility rules for brand
            if brand:
                rules = self.semantic_memory.query(
                    "MATCH (r:EligibilityRule)-[:APPLIES_TO]->(b:Brand {name: $brand}) "
                    "RETURN r LIMIT 20",
                    {"brand": brand},
                )
                context["eligibility_rules"] = rules

            # Query prior cohorts for same brand/indication
            if brand and indication:
                cohorts = self.semantic_memory.query(
                    "MATCH (c:CohortConfig)-[:FOR_BRAND]->(b:Brand {name: $brand}) "
                    "WHERE c.indication = $indication "
                    "RETURN c LIMIT 10",
                    {"brand": brand, "indication": indication},
                )
                context["prior_cohorts"] = cohorts

            return context
        except Exception as e:
            logger.warning(f"Failed to get semantic context: {e}")
            return {}

    # =========================================================================
    # STORAGE: WORKING MEMORY (CACHE)
    # =========================================================================

    async def cache_cohort_config(
        self,
        session_id: str,
        config: Dict[str, Any],
    ) -> bool:
        """
        Cache cohort configuration in working memory.

        Args:
            session_id: Session identifier
            config: The cohort configuration to cache

        Returns:
            True if successfully cached
        """
        if not self.working_memory:
            return False

        try:
            cache_key = f"cohort_constructor:config:{session_id}"
            await self.working_memory.set(
                cache_key,
                json.dumps(config),
                ex=self.CACHE_TTL_SECONDS,
            )
            logger.debug(f"Cached cohort config for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache cohort config: {e}")
            return False

    async def cache_cohort_result(
        self,
        session_id: str,
        result: Dict[str, Any],
    ) -> bool:
        """
        Cache cohort construction result in working memory.

        Args:
            session_id: Session identifier
            result: The cohort construction result to cache

        Returns:
            True if successfully cached
        """
        if not self.working_memory:
            return False

        try:
            # RedisWorkingMemory has no ``set`` method — the hook predated the
            # API and every call AttributeError'd into the except below (#883
            # PR B, surfaced by the agent-path wiring proof). Use the raw
            # client like the sibling hooks do.
            redis = await self.working_memory.get_client()
            cache_key = f"cohort_constructor:result:{session_id}"
            await redis.setex(
                cache_key,
                self.CACHE_TTL_SECONDS,
                json.dumps(result, default=str),
            )
            logger.debug(f"Cached cohort result for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache cohort result: {e}")
            return False

    # =========================================================================
    # STORAGE: EPISODIC MEMORY
    # =========================================================================

    async def store_cohort_result(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        region: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store cohort construction result in episodic memory.

        Args:
            session_id: Session identifier
            result: CohortConstructorState output fields
            state: Full CohortConstructorState
            region: Optional region context

        Returns:
            Memory entry ID if successful, None otherwise
        """
        try:
            from src.memory.episodic_memory import (
                E2IEntityReferences,
                EpisodicMemoryInput,
                insert_episodic_memory_with_text,
            )

            # Extract key information
            config = state.get("config", {})
            stats = state.get("eligibility_stats", {})
            metadata = state.get("execution_metadata", {})

            brand = config.get("brand", "unknown")
            indication = config.get("indication", "unknown")
            cohort_name = config.get("cohort_name", "unknown")
            cohort_id = result.get("cohort_id", "unknown")

            content = {
                "cohort_id": cohort_id,
                "cohort_name": cohort_name,
                "brand": brand,
                "indication": indication,
                "kpi_category": "cohort",
                "total_patients": stats.get("total_input_patients", 0),
                "eligible_patients": stats.get("eligible_patient_count", 0),
                "eligibility_rate": 1.0 - stats.get("exclusion_rate", 0),
                "criteria_count": len(config.get("inclusion_criteria", []))
                + len(config.get("exclusion_criteria", [])),
                "execution_time_ms": metadata.get("execution_time_ms", 0),
                "status": result.get("status", "unknown"),
            }

            # Build searchable summary
            eligibility_rate = content["eligibility_rate"] * 100
            summary = (
                f"Cohort Construction: {cohort_name}. "
                f"Brand: {brand}. Indication: {indication}. "
                f"Eligible: {content['eligible_patients']}/{content['total_patients']} ({eligibility_rate:.1f}%)."
            )

            # Canonical EpisodicMemoryInput pattern (the legacy #749 compat
            # kwargs path folds brand into raw_content but leaves the BRAND
            # COLUMN null, so the brand-filtered search RPC behind
            # get_prior_cohorts could never match a row — #883 PR B).
            # outcome_type is the constrained memory_outcome_type enum: map
            # the construction status (the hook gate admits success/partial).
            memory_input = EpisodicMemoryInput(
                event_type="cohort_construction_completed",
                event_subtype="cohort_construction",
                description=summary,
                raw_content=content,
                outcome_type=(
                    "partial_success" if result.get("status") == "partial" else "success"
                ),
                agent_name="cohort_constructor",
                importance_score=0.6,
                e2i_refs=E2IEntityReferences(
                    brand=brand if brand != "unknown" else None,
                    region=region,
                ),
            )

            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=summary,
                session_id=session_id,
            )

            logger.info(f"Stored cohort result in episodic memory: {memory_id}")
            return cast(Optional[str], memory_id)
        except Exception as e:
            logger.warning(f"Failed to store cohort result: {e}")
            return None

    # =========================================================================
    # STORAGE: SEMANTIC MEMORY
    # =========================================================================

    async def store_eligibility_rule(
        self,
        rule_name: str,
        criterion: Dict[str, Any],
        brand: str,
        effectiveness_score: float,
        cohort_id: str,
    ) -> bool:
        """
        Store discovered eligibility rule in semantic memory (knowledge graph).

        Args:
            rule_name: Human-readable rule name
            criterion: Criterion specification (field, operator, value, etc.)
            brand: Associated brand (Remibrutinib, Fabhalta, Kisqali)
            effectiveness_score: How effective this rule is (0-1)
            cohort_id: Associated cohort ID

        Returns:
            True if successfully stored
        """
        if not self.semantic_memory:
            logger.warning("Semantic memory not available for rule storage")
            return False

        try:
            rule_id = f"rule:{brand}:{criterion.get('field', 'unknown')}:{criterion.get('operator', 'eq')}"

            # Create eligibility rule node
            self.semantic_memory.add_e2i_entity(
                entity_type="EligibilityRule",
                entity_id=rule_id,
                properties={
                    "name": rule_name,
                    "field": criterion.get("field"),
                    "operator": criterion.get("operator"),
                    "value": json.dumps(criterion.get("value")),
                    "criterion_type": criterion.get("criterion_type"),
                    "description": criterion.get("description", ""),
                    "clinical_rationale": criterion.get("clinical_rationale", ""),
                    "effectiveness_score": effectiveness_score,
                    "agent": "cohort_constructor",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Create brand node and relationship. Canonicalise the brand casing
            # (Remibrutinib, not remibrutinib) so this write MERGEs onto the one
            # canonical Brand node instead of minting a lowercase duplicate.
            from src.memory.episodic_memory import canonical_brand

            brand_name = canonical_brand(brand) or brand
            brand_id = f"brand:{brand_name}"
            self.semantic_memory.add_e2i_entity(
                entity_type="Brand",
                entity_id=brand_id,
                properties={
                    "name": brand_name,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self.semantic_memory.add_relationship(
                from_entity_id=rule_id,
                to_entity_id=brand_id,
                relationship_type="APPLIES_TO",
                properties={"agent": "cohort_constructor"},
            )

            # Link to cohort
            cohort_entity_id = f"cohort:{cohort_id}"
            self.semantic_memory.add_relationship(
                from_entity_id=rule_id,
                to_entity_id=cohort_entity_id,
                relationship_type="USED_IN",
                properties={
                    "agent": "cohort_constructor",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.info(f"Stored eligibility rule: {rule_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store eligibility rule: {e}")
            return False

    async def store_cohort_pattern(
        self,
        cohort_id: str,
        cohort_name: str,
        brand: str,
        indication: str,
        criteria_summary: Dict[str, Any],
        eligibility_rate: float,
    ) -> bool:
        """
        Store cohort configuration pattern in semantic memory.

        Args:
            cohort_id: Unique cohort identifier
            cohort_name: Human-readable cohort name
            brand: Brand name
            indication: Medical indication
            criteria_summary: Summary of criteria used
            eligibility_rate: Final eligibility rate

        Returns:
            True if successfully stored
        """
        if not self.semantic_memory:
            logger.warning("Semantic memory not available for cohort pattern storage")
            return False

        try:
            cohort_entity_id = f"cohort:{cohort_id}"

            # Canonicalise the brand casing once so the CohortConfig property and
            # the Brand node below share the one canonical identity (no lowercase
            # duplicate Brand node).
            from src.memory.episodic_memory import canonical_brand

            brand = canonical_brand(brand) or brand

            # Create cohort config node
            self.semantic_memory.add_e2i_entity(
                entity_type="CohortConfig",
                entity_id=cohort_entity_id,
                properties={
                    "cohort_id": cohort_id,
                    "name": cohort_name,
                    "brand": brand,
                    "indication": indication,
                    "inclusion_count": criteria_summary.get("inclusion_count", 0),
                    "exclusion_count": criteria_summary.get("exclusion_count", 0),
                    "eligibility_rate": eligibility_rate,
                    "agent": "cohort_constructor",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Link to brand
            brand_id = f"brand:{brand}"
            self.semantic_memory.add_e2i_entity(
                entity_type="Brand",
                entity_id=brand_id,
                properties={
                    "name": brand,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self.semantic_memory.add_relationship(
                from_entity_id=cohort_entity_id,
                to_entity_id=brand_id,
                relationship_type="FOR_BRAND",
                properties={"agent": "cohort_constructor"},
            )

            logger.info(f"Stored cohort pattern: {cohort_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store cohort pattern: {e}")
            return False

    # =========================================================================
    # PRIOR COHORTS (For DSPy Training)
    # =========================================================================

    async def get_prior_cohorts(
        self,
        brand: Optional[str] = None,
        min_eligibility_rate: float = 0.1,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get prior cohort constructions for DSPy training signals.

        Args:
            brand: Optional filter by brand
            min_eligibility_rate: Minimum eligibility rate threshold
            limit: Maximum results to return

        Returns:
            List of prior cohort constructions
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                content_filter_fetch_limit,
                hydrate_raw_content,
                search_episodic_by_text,
            )

            query_text = f"cohort construction {brand or ''} successful"

            filters = EpisodicSearchFilters(
                event_type="cohort_construction_completed",
                agent_name="cohort_constructor",
            )
            if brand:
                filters.brand = brand

            # Bounded candidate window, NOT a small fixed multiple: the
            # eligibility/status post-filter below runs on hydrated content,
            # and a `limit * 2` window starves when high-similarity
            # non-matching rows fill it (codex R1 on the #883 read-side fix).
            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=content_filter_fetch_limit(limit),
                min_similarity=0.5,
                include_entity_context=False,
            )
            if not results:
                return []

            # The search RPC's TABLE shape does NOT include raw_content, so
            # the original post-filter (`r.get("raw_content", {})`) saw {} on
            # every row and dropped them all — the reader stayed dead even
            # once writes landed (#883 PR B). Hydrate raw_content by memory_id
            # via the shared helper (one batched select + JSON-string-scalar
            # parse-back; this hook's inline copy was lifted into
            # episodic_memory.hydrate_raw_content when gap_analyzer's three
            # readers needed the identical fix).
            results = await hydrate_raw_content(results)

            # Filter by eligibility rate and construction status
            filtered = []
            for r in results:
                rc = r.get("raw_content", {})
                if (
                    rc.get("eligibility_rate", 0) >= min_eligibility_rate
                    and rc.get("status") == "success"
                ):
                    filtered.append(r)

            return filtered[:limit]
        except Exception as e:
            logger.warning(f"Failed to get prior cohorts: {e}")
            return []

    async def get_effective_rules_for_brand(
        self,
        brand: str,
        min_effectiveness: float = 0.7,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve effective eligibility rules for a brand from semantic memory.

        Args:
            brand: Target brand
            min_effectiveness: Minimum effectiveness score
            limit: Maximum rules to return

        Returns:
            List of effective eligibility rules
        """
        if not self.semantic_memory:
            return []

        try:
            rules = self.semantic_memory.query(
                "MATCH (r:EligibilityRule)-[:APPLIES_TO]->(b:Brand {name: $brand}) "
                "WHERE r.effectiveness_score >= $min_effectiveness "
                # LIMIT cannot be a bound param in FalkorDB; int-coerce to stay injection-safe.
                f"RETURN r ORDER BY r.effectiveness_score DESC LIMIT {int(limit)}",
                {"brand": brand, "min_effectiveness": min_effectiveness},
            )
            return cast(List[Dict[str, Any]], rules)
        except Exception as e:
            logger.warning(f"Failed to get effective rules: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[CohortConstructorMemoryHooks] = None,
    session_id: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute cohort construction results to CognitiveRAG's memory systems.

    This is the primary interface for storing cohort constructor results
    in the tri-memory architecture.

    Args:
        result: CohortConstructorState output fields
        state: Full CohortConstructorState
        memory_hooks: Optional memory hooks instance
        session_id: Session identifier
        region: Optional region context

    Returns:
        Dictionary with counts of stored memories
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_cohort_constructor_memory_hooks()

    if session_id is None:
        session_id = state.get("session_id") or str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "semantic_stored": 0,
        "working_cached": 0,
        "rules_stored": 0,
    }

    # Skip storage if construction failed
    status = result.get("status", "unknown")
    if status not in ("success", "partial"):
        logger.info(f"Skipping memory storage for failed construction: {status}")
        return counts

    config = state.get("config", {})
    brand = config.get("brand", "unknown")

    # 1. Cache result in working memory
    cached = await memory_hooks.cache_cohort_result(session_id, result)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_cohort_result(
        session_id=session_id,
        result=result,
        state=state,
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 3. Store cohort pattern in semantic memory
    cohort_id = result.get("cohort_id")
    if cohort_id:
        stats = state.get("eligibility_stats", {})
        eligibility_rate = 1.0 - stats.get("exclusion_rate", 0)

        stored = await memory_hooks.store_cohort_pattern(
            cohort_id=cohort_id,
            cohort_name=config.get("cohort_name", "unknown"),
            brand=brand,
            indication=config.get("indication", "unknown"),
            criteria_summary={
                "inclusion_count": len(config.get("inclusion_criteria", [])),
                "exclusion_count": len(config.get("exclusion_criteria", [])),
            },
            eligibility_rate=eligibility_rate,
        )
        if stored:
            counts["semantic_stored"] = 1

        # 4. Store effective eligibility rules in semantic memory
        for criterion in config.get("inclusion_criteria", []) + config.get(
            "exclusion_criteria", []
        ):
            # Calculate effectiveness based on removal count in log
            effectiveness = 0.5  # Default moderate effectiveness
            rule_stored = await memory_hooks.store_eligibility_rule(
                rule_name=criterion.get("description", criterion.get("field", "unknown")),
                criterion=criterion,
                brand=brand,
                effectiveness_score=effectiveness,
                cohort_id=cohort_id,
            )
            if rule_stored:
                counts["rules_stored"] += 1

    logger.info(
        f"Memory contribution complete: "
        f"episodic={counts['episodic_stored']}, "
        f"semantic={counts['semantic_stored']}, "
        f"rules={counts['rules_stored']}, "
        f"working_cached={counts['working_cached']}"
    )

    return counts


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[CohortConstructorMemoryHooks] = None


def get_cohort_constructor_memory_hooks() -> CohortConstructorMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = CohortConstructorMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
