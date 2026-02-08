"""
Explainer Agent Memory Hooks
============================

Memory integration hooks for the Explainer agent's tri-memory architecture.

The Explainer uses these hooks to:
1. Retrieve context from working memory (Redis - recent conversation)
2. Search episodic memory (Supabase - similar past explanations)
3. Query semantic memory (FalkorDB - entity relationships)
4. Cache generated explanations for future retrieval

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import json
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


# =============================================================================
# UUID CONVERSION HELPER
# =============================================================================


def _ensure_uuid(value: str) -> str:
    """Convert a non-UUID string to a deterministic UUID v5."""
    if not value:
        return str(_uuid.uuid4())
    try:
        _uuid.UUID(value)
        return value  # Already a valid UUID
    except ValueError:
        return str(_uuid.uuid5(_uuid.NAMESPACE_DNS, value))


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ExplanationContext:
    """Context retrieved from all memory systems for explanation generation."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExplanationRecord:
    """Record of a generated explanation for storage."""

    session_id: str
    query: str
    executive_summary: str
    detailed_explanation: str
    insights: List[Dict[str, Any]]
    audience: str
    output_format: str
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class ExplanationMemoryHooks:
    """
    Memory integration hooks for the Explainer agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache explanations in working memory (24h TTL)
    - Store explanations in episodic memory for future retrieval
    - Query entity relationships from semantic memory
    """

    # Cache TTL in seconds (24 hours)
    CACHE_TTL_SECONDS = 86400

    def __init__(self):
        """Initialize memory hooks with lazy-loaded clients."""
        self._working_memory = None
        self._semantic_memory = None
        self._entity_extractor = None

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

    @property
    def entity_extractor(self):
        """Lazy-load EntityExtractor for query entity extraction."""
        if self._entity_extractor is None:
            try:
                from src.rag.entity_extractor import EntityExtractor

                self._entity_extractor = EntityExtractor()
                logger.debug("Entity extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize entity extractor: {e}")
                self._entity_extractor = None
        return self._entity_extractor

    # =========================================================================
    # CONTEXT RETRIEVAL
    # =========================================================================

    async def get_context(
        self,
        session_id: str,
        query: str,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        max_episodic_results: int = 5,
    ) -> ExplanationContext:
        """
        Retrieve context from all three memory systems.

        Args:
            session_id: Session identifier for working memory lookup
            query: Query text for episodic similarity search
            brand: Optional brand filter for episodic search
            region: Optional region filter for episodic search
            max_episodic_results: Maximum episodic memories to retrieve

        Returns:
            ExplanationContext with data from all memory systems
        """
        context = ExplanationContext(session_id=session_id)

        # 1. Get working memory (recent conversation)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Get episodic memory (similar past explanations)
        context.episodic_context = await self._get_episodic_context(
            query=query,
            brand=brand,
            region=region,
            limit=max_episodic_results,
        )

        # 3. Get semantic memory (entity relationships)
        context.semantic_context = await self._get_semantic_context(query)

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_entities={len(context.semantic_context.get('entities', []))}"
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
        query: str,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar past explanations."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            filters = EpisodicSearchFilters(
                event_type="explanation_generated",
                agent_name="explainer",
                brand=brand,
                region=region,
            )

            results = await search_episodic_by_text(
                query_text=query,
                filters=filters,
                limit=limit,
                min_similarity=0.6,
                include_entity_context=False,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to search episodic memory: {e}")
            return []

    async def _get_semantic_context(self, query: str) -> Dict[str, Any]:
        """Query semantic memory for entity relationships.

        Uses EntityExtractor to identify domain entities in the query,
        then queries FalkorDB for those entities and their relationships.
        """
        if not self.semantic_memory:
            return {"entities": [], "relationships": [], "causal_paths": []}

        try:
            from src.memory.semantic_memory import E2IEntityType

            # Get graph statistics for context
            stats = self.semantic_memory.get_graph_stats()

            # Extract entities mentioned in the query
            extracted_entities = []
            relationships = []

            if self.entity_extractor:
                entities = self.entity_extractor.extract(query)

                # Query semantic memory for each extracted entity type
                # Brands map to general entity lookups
                for brand in entities.brands:
                    entity_data = self.semantic_memory.get_entity(E2IEntityType.TREATMENT, brand)
                    if entity_data:
                        extracted_entities.append(
                            {
                                "type": "brand",
                                "id": brand,
                                "data": entity_data,
                            }
                        )
                        # Get relationships for this entity
                        rels = self.semantic_memory.get_relationships(
                            E2IEntityType.TREATMENT, brand, direction="both"
                        )
                        relationships.extend(rels[:5])  # Limit to 5 per entity

                # KPIs - look up as triggers or causal paths
                for kpi in entities.kpis:
                    # Try as trigger first
                    entity_data = self.semantic_memory.get_entity(E2IEntityType.TRIGGER, kpi)
                    if entity_data:
                        extracted_entities.append(
                            {
                                "type": "kpi",
                                "id": kpi,
                                "data": entity_data,
                            }
                        )
                        rels = self.semantic_memory.get_relationships(
                            E2IEntityType.TRIGGER, kpi, direction="both"
                        )
                        relationships.extend(rels[:5])

                # Agents - look up agent activities
                for agent in entities.agents:
                    entity_data = self.semantic_memory.get_entity(
                        E2IEntityType.AGENT_ACTIVITY, agent
                    )
                    if entity_data:
                        extracted_entities.append(
                            {
                                "type": "agent",
                                "id": agent,
                                "data": entity_data,
                            }
                        )

                # HCP segments
                for segment in entities.hcp_segments:
                    entity_data = self.semantic_memory.get_entity(E2IEntityType.HCP, segment)
                    if entity_data:
                        extracted_entities.append(
                            {
                                "type": "hcp_segment",
                                "id": segment,
                                "data": entity_data,
                            }
                        )

                logger.debug(
                    f"Extracted {len(extracted_entities)} entities, "
                    f"{len(relationships)} relationships from query"
                )

            # Get causal paths if any causal-related entities found
            causal_paths = []
            if any(e.get("type") == "kpi" for e in extracted_entities):
                try:
                    # Query for causal paths related to extracted KPIs
                    kpi_ids = [e["id"] for e in extracted_entities if e.get("type") == "kpi"]
                    for kpi_id in kpi_ids[:2]:  # Limit to first 2 KPIs
                        paths = self.semantic_memory.get_relationships(
                            E2IEntityType.CAUSAL_PATH, kpi_id, direction="both"
                        )
                        causal_paths.extend(paths[:3])  # Limit paths per KPI
                except Exception as e:
                    logger.debug(f"Failed to get causal paths: {e}")

            return {
                "entities": extracted_entities,
                "relationships": relationships[:20],  # Cap total relationships
                "causal_paths": causal_paths[:10],  # Cap causal paths
                "graph_stats": stats,
                "extraction_summary": {
                    "brands_found": len(entities.brands) if self.entity_extractor else 0,
                    "kpis_found": len(entities.kpis) if self.entity_extractor else 0,
                    "agents_found": len(entities.agents) if self.entity_extractor else 0,
                },
            }
        except Exception as e:
            logger.warning(f"Failed to query semantic memory: {e}")
            return {"entities": [], "relationships": [], "causal_paths": []}

    # =========================================================================
    # EXPLANATION CACHING (Working Memory)
    # =========================================================================

    async def cache_explanation(
        self,
        session_id: str,
        explanation: Dict[str, Any],
    ) -> bool:
        """
        Cache explanation in working memory with 24h TTL.

        Args:
            session_id: Session identifier
            explanation: Explanation data to cache

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"explainer:cache:{session_id}"

            # Store as JSON with TTL
            await redis.setex(
                cache_key,
                self.CACHE_TTL_SECONDS,
                json.dumps(explanation),
            )

            logger.debug(f"Cached explanation for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache explanation: {e}")
            return False

    async def get_cached_explanation(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached explanation from working memory.

        Args:
            session_id: Session identifier

        Returns:
            Cached explanation or None if not found
        """
        if not self.working_memory:
            return None

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"explainer:cache:{session_id}"

            cached = await redis.get(cache_key)
            if cached:
                return cast(Dict[str, Any], json.loads(cached))
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached explanation: {e}")
            return None

    # =========================================================================
    # EXPLANATION STORAGE (Episodic Memory)
    # =========================================================================

    async def store_explanation(
        self,
        session_id: str,
        explanation: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store explanation in episodic memory for future retrieval.

        Args:
            session_id: Session identifier
            explanation: Explanation data to store
            brand: Optional brand context
            region: Optional region context

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            from src.memory.episodic_memory import (
                E2IEntityReferences,
                EpisodicMemoryInput,
                insert_episodic_memory_with_text,
            )

            # Build description for embedding
            description = f"Explanation: {explanation.get('executive_summary', '')}"

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="explanation_generated",
                event_subtype=explanation.get("output_format", "narrative"),
                description=description,
                raw_content=explanation,
                entities=explanation.get("entities"),
                outcome_type="explanation_delivered",
                agent_name="explainer",
                importance_score=0.7,
                e2i_refs=E2IEntityReferences(
                    brand=brand,
                    region=region,
                ),
            )

            # Insert with auto-generated embedding
            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=description,
                session_id=_ensure_uuid(session_id),
            )

            logger.info(f"Stored explanation in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store explanation in episodic memory: {e}")
            return None

    # =========================================================================
    # ENTITY CONTEXT RETRIEVAL
    # =========================================================================

    async def get_entity_relationships(
        self,
        entity_id: str,
        entity_type: str,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Get entity relationships from semantic memory.

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity (patient, hcp, etc.)
            max_depth: Maximum traversal depth

        Returns:
            Dict with entity network information
        """
        if not self.semantic_memory:
            return {"entity_id": entity_id, "network": {}}

        try:
            if entity_type == "patient":
                network = self.semantic_memory.get_patient_network(entity_id, max_depth)
            elif entity_type == "hcp":
                network = self.semantic_memory.get_hcp_influence_network(entity_id, max_depth)
            else:
                # Generic entity lookup
                network = {"entity_id": entity_id}

            return cast(Dict[str, Any], network)
        except Exception as e:
            logger.warning(f"Failed to get entity relationships: {e}")
            return {"entity_id": entity_id, "network": {}}

    async def get_causal_context(
        self,
        kpi_name: str,
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Get causal paths impacting a KPI from semantic memory.

        Args:
            kpi_name: Name of the KPI (e.g., "TRx", "NRx")
            min_confidence: Minimum confidence threshold

        Returns:
            List of causal paths with effect sizes
        """
        if not self.semantic_memory:
            return []

        try:
            paths = self.semantic_memory.find_causal_paths_for_kpi(
                kpi_name=kpi_name,
                min_confidence=min_confidence,
            )
            return cast(List[Dict[str, Any]], paths)
        except Exception as e:
            logger.warning(f"Failed to get causal context: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION (Per Specialist Document)
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[ExplanationMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute explanation results to CognitiveRAG's memory systems.

    This is the primary interface for storing explainer
    results in the memory architecture.

    Args:
        result: ExplainerOutput dictionary
        state: ExplainerState dictionary
        memory_hooks: Optional memory hooks instance (creates new if not provided)
        session_id: Session identifier (generates UUID if not provided)
        brand: Optional brand context
        region: Optional region context

    Returns:
        Dictionary with counts of stored memories:
        - episodic_stored: 1 if explanation stored, 0 otherwise
        - working_cached: 1 if cached, 0 otherwise
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_explanation_memory_hooks()

    if session_id is None:
        session_id = state.get("session_id") or str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "working_cached": 0,
    }

    # Skip storage if explanation failed
    if state.get("status") == "failed":
        logger.info("Skipping memory storage for failed explanation")
        return counts

    # Build explanation data for storage
    explanation_data = {
        "query": state.get("query", ""),
        "executive_summary": result.get("executive_summary", ""),
        "detailed_explanation": result.get("detailed_explanation", ""),
        "insights": result.get("extracted_insights", []),
        "audience": state.get("user_expertise", "analyst"),
        "output_format": state.get("output_format", "narrative"),
        "narrative_sections": result.get("narrative_sections", []),
        "visual_suggestions": result.get("visual_suggestions", []),
        "follow_up_questions": result.get("follow_up_questions", []),
    }

    # 1. Cache in working memory
    cached = await memory_hooks.cache_explanation(session_id, explanation_data)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_explanation(
        session_id=session_id,
        explanation=explanation_data,
        brand=brand,
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    logger.info(
        f"Memory contribution complete: "
        f"episodic={counts['episodic_stored']}, "
        f"working_cached={counts['working_cached']}"
    )

    return counts


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[ExplanationMemoryHooks] = None


def get_explanation_memory_hooks() -> ExplanationMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = ExplanationMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
