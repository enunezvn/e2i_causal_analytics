"""
E2I Agentic Memory - Episodic Memory (Supabase + pgvector)
Long-term storage for significant experiences with vector similarity search.

Technology: Supabase (PostgreSQL + pgvector)

Features:
- Vector similarity search for episodic memories
- E2I entity integration (patient, HCP, trigger, causal_path)
- Brand and region filtering
- Entity context enrichment
- Bulk insert for performance

Usage:
    from src.memory.episodic_memory import (
        search_episodic_memory,
        insert_episodic_memory,
        get_enriched_episodic_memory
    )

    # Search memories by embedding
    results = await search_episodic_memory(
        embedding=query_embedding,
        filters=EpisodicSearchFilters(brand="Kisqali"),
        limit=10
    )

    # Insert a new memory
    memory_id = await insert_episodic_memory(
        memory=EpisodicMemoryInput(
            event_type="query_answer",
            description="Answered TRx drop question"
        ),
        embedding=memory_embedding
    )
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.memory.services.factories import get_supabase_client, get_embedding_service

logger = logging.getLogger(__name__)


# ============================================================================
# E2I DATA LAYER ENTITY TYPES
# ============================================================================

class E2IEntityType(str, Enum):
    """E2I data layer entity types for foreign key references."""
    PATIENT = "patient"
    HCP = "hcp"
    TREATMENT = "treatment"
    TRIGGER = "trigger"
    PREDICTION = "prediction"
    CAUSAL_PATH = "causal_path"
    EXPERIMENT = "experiment"
    AGENT_ACTIVITY = "agent_activity"


class E2IBrand(str, Enum):
    """E2I brand values."""
    REMIBRUTINIB = "Remibrutinib"
    FABHALTA = "Fabhalta"
    KISQALI = "Kisqali"
    ALL = "all"


class E2IRegion(str, Enum):
    """E2I region values."""
    NORTHEAST = "northeast"
    SOUTH = "south"
    MIDWEST = "midwest"
    WEST = "west"
    ALL = "all"


class E2IAgentName(str, Enum):
    """E2I agent architecture names."""
    # Tier 1: Coordination
    ORCHESTRATOR = "orchestrator"
    TOOL_COMPOSER = "tool_composer"
    # Tier 2: Causal Analytics
    CAUSAL_IMPACT = "causal_impact"
    GAP_ANALYZER = "gap_analyzer"
    HETEROGENEOUS_OPTIMIZER = "heterogeneous_optimizer"
    # Tier 3: Monitoring
    DRIFT_MONITOR = "drift_monitor"
    EXPERIMENT_DESIGNER = "experiment_designer"
    HEALTH_SCORE = "health_score"
    # Tier 4: ML Predictions
    PREDICTION_SYNTHESIZER = "prediction_synthesizer"
    RESOURCE_OPTIMIZER = "resource_optimizer"
    # Tier 5: Self-Improvement
    FEEDBACK_LEARNER = "feedback_learner"
    EXPLAINER = "explainer"


# ============================================================================
# DATA CLASSES FOR E2I ENTITY CONTEXT
# ============================================================================

@dataclass
class E2IEntityContext:
    """Context about linked E2I entities for a memory."""
    patient: Optional[Dict[str, Any]] = None
    hcp: Optional[Dict[str, Any]] = None
    trigger: Optional[Dict[str, Any]] = None
    causal_path: Optional[Dict[str, Any]] = None
    treatment: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None
    experiment: Optional[Dict[str, Any]] = None
    agent_activity: Optional[Dict[str, Any]] = None


@dataclass
class E2IEntityReferences:
    """Foreign key references to E2I data layer entities."""
    patient_journey_id: Optional[str] = None
    patient_id: Optional[str] = None
    hcp_id: Optional[str] = None
    treatment_event_id: Optional[str] = None
    trigger_id: Optional[str] = None
    prediction_id: Optional[str] = None
    causal_path_id: Optional[str] = None
    experiment_id: Optional[str] = None
    agent_activity_id: Optional[str] = None
    brand: Optional[str] = None
    region: Optional[str] = None


@dataclass
class EpisodicMemoryInput:
    """Input for creating an episodic memory with E2I integration."""
    event_type: str
    description: str
    event_subtype: Optional[str] = None
    raw_content: Optional[Dict[str, Any]] = None
    entities: Optional[Dict[str, Any]] = None
    outcome_type: Optional[str] = None
    outcome_details: Optional[Dict[str, Any]] = None
    user_satisfaction_score: Optional[int] = None
    agent_name: Optional[str] = None
    importance_score: Optional[float] = 0.5
    e2i_refs: Optional[E2IEntityReferences] = None


@dataclass
class EpisodicSearchFilters:
    """Filters for episodic memory search with E2I entity support."""
    event_type: Optional[str] = None
    agent_name: Optional[str] = None
    brand: Optional[str] = None
    region: Optional[str] = None
    patient_id: Optional[str] = None
    hcp_id: Optional[str] = None
    trigger_id: Optional[str] = None
    causal_path_id: Optional[str] = None
    days_back: Optional[int] = None
    min_importance: Optional[float] = None


@dataclass
class EnrichedEpisodicMemory:
    """Episodic memory with full E2I entity context attached."""
    memory_id: str
    event_type: str
    description: str
    occurred_at: str
    event_subtype: Optional[str] = None
    outcome_type: Optional[str] = None
    agent_name: Optional[str] = None
    confidence_score: Optional[float] = None
    patient_context: Optional[Dict[str, Any]] = None
    hcp_context: Optional[Dict[str, Any]] = None
    trigger_context: Optional[Dict[str, Any]] = None
    causal_path_context: Optional[Dict[str, Any]] = None
    treatment_context: Optional[Dict[str, Any]] = None
    prediction_context: Optional[Dict[str, Any]] = None


@dataclass
class AgentActivityContext:
    """Full context for an agent activity."""
    activity_id: str
    agent_name: str
    action_type: str
    started_at: str
    status: str
    completed_at: Optional[str] = None
    trigger: Optional[Dict[str, Any]] = None
    causal_paths: Optional[List[Dict[str, Any]]] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    duration_ms: Optional[int] = None
    tokens_used: Optional[int] = None


# ============================================================================
# MEMORY STATISTICS TRACKING
# ============================================================================

async def _increment_memory_stats(memory_type: str, event_type: str) -> None:
    """
    Track memory statistics for monitoring.

    Args:
        memory_type: Type of memory (episodic, procedural, semantic)
        event_type: The event type being recorded
    """
    # This would typically update a stats table or send to metrics
    # For now, just log for observability
    logger.debug(f"Memory stat: {memory_type}/{event_type} +1")


# ============================================================================
# EPISODIC MEMORY SEARCH FUNCTIONS
# ============================================================================

async def search_episodic_memory(
    embedding: List[float],
    filters: Optional[EpisodicSearchFilters] = None,
    limit: int = 10,
    min_similarity: float = 0.5,
    include_entity_context: bool = False
) -> List[Dict[str, Any]]:
    """
    Search episodic memories by embedding similarity with E2I entity filters.

    Args:
        embedding: Query embedding vector
        filters: EpisodicSearchFilters with E2I entity support
        limit: Maximum results to return
        min_similarity: Minimum cosine similarity threshold
        include_entity_context: If True, fetch linked E2I entity details

    Returns:
        List of matching episodic memories with similarity scores
    """
    client = get_supabase_client()

    # Build filter params for RPC call
    filter_params = {
        "query_embedding": embedding,
        "match_threshold": min_similarity,
        "match_count": limit,
        "filter_event_type": None,
        "filter_agent": None,
        "filter_brand": None,
        "filter_region": None,
        "filter_patient_id": None,
        "filter_hcp_id": None
    }

    if filters:
        filter_params["filter_event_type"] = filters.event_type
        filter_params["filter_agent"] = filters.agent_name
        filter_params["filter_brand"] = filters.brand
        filter_params["filter_region"] = filters.region
        filter_params["filter_patient_id"] = filters.patient_id
        filter_params["filter_hcp_id"] = filters.hcp_id

    result = client.rpc("search_episodic_memory", filter_params).execute()
    memories = result.data or []

    # Optionally enrich with E2I entity context
    if include_entity_context and memories:
        for memory in memories:
            context = await get_memory_entity_context(memory["memory_id"])
            memory["e2i_context"] = asdict(context)

    logger.debug(f"Episodic search found {len(memories)} results")
    return memories


async def search_episodic_by_text(
    query_text: str,
    filters: Optional[EpisodicSearchFilters] = None,
    limit: int = 10,
    min_similarity: float = 0.5,
    include_entity_context: bool = False
) -> List[Dict[str, Any]]:
    """
    Search episodic memories by text query (auto-generates embedding).

    Args:
        query_text: Text query to search for
        filters: EpisodicSearchFilters with E2I entity support
        limit: Maximum results to return
        min_similarity: Minimum cosine similarity threshold
        include_entity_context: If True, fetch linked E2I entity details

    Returns:
        List of matching episodic memories with similarity scores
    """
    embedding_service = get_embedding_service()
    embedding = await embedding_service.embed(query_text)

    return await search_episodic_memory(
        embedding=embedding,
        filters=filters,
        limit=limit,
        min_similarity=min_similarity,
        include_entity_context=include_entity_context
    )


async def search_episodic_by_e2i_entity(
    entity_type: E2IEntityType,
    entity_id: str,
    limit: int = 20,
    event_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Search episodic memories linked to a specific E2I entity.

    Args:
        entity_type: Type of E2I entity (patient, hcp, trigger, etc.)
        entity_id: ID of the entity
        limit: Maximum results
        event_types: Optional filter by event types

    Returns:
        List of episodic memories linked to this entity
    """
    client = get_supabase_client()

    # Map entity type to column name
    column_map = {
        E2IEntityType.PATIENT: "patient_journey_id",
        E2IEntityType.HCP: "hcp_id",
        E2IEntityType.TRIGGER: "trigger_id",
        E2IEntityType.PREDICTION: "prediction_id",
        E2IEntityType.CAUSAL_PATH: "causal_path_id",
        E2IEntityType.EXPERIMENT: "experiment_id",
        E2IEntityType.TREATMENT: "treatment_event_id",
        E2IEntityType.AGENT_ACTIVITY: "agent_activity_id"
    }

    column = column_map.get(entity_type)
    if not column:
        raise ValueError(f"Unknown entity type: {entity_type}")

    query = client.table("episodic_memories") \
        .select("*") \
        .eq(column, entity_id) \
        .order("occurred_at", desc=True) \
        .limit(limit)

    if event_types:
        query = query.in_("event_type", event_types)

    result = query.execute()
    logger.debug(f"Entity search ({entity_type.value}/{entity_id}) found {len(result.data or [])} results")
    return result.data or []


# ============================================================================
# EPISODIC MEMORY INSERT FUNCTIONS
# ============================================================================

async def insert_episodic_memory(
    memory: EpisodicMemoryInput,
    embedding: List[float],
    session_id: Optional[str] = None,
    cycle_id: Optional[str] = None
) -> str:
    """
    Insert new episodic memory with E2I entity references.

    Args:
        memory: EpisodicMemoryInput with E2I entity references
        embedding: Pre-computed embedding vector
        session_id: Optional session ID
        cycle_id: Optional cognitive cycle ID

    Returns:
        ID of inserted memory
    """
    client = get_supabase_client()

    memory_id = str(uuid.uuid4())

    record = {
        "memory_id": memory_id,
        "session_id": session_id,
        "cycle_id": cycle_id,
        "event_type": memory.event_type,
        "event_subtype": memory.event_subtype,
        "description": memory.description,
        "raw_content": json.dumps(memory.raw_content or {}),
        "entities": json.dumps(memory.entities or {}),
        "outcome_type": memory.outcome_type,
        "outcome_details": json.dumps(memory.outcome_details or {}),
        "user_satisfaction_score": memory.user_satisfaction_score,
        "agent_name": memory.agent_name,
        "importance_score": memory.importance_score,
        "embedding": embedding,
        "occurred_at": datetime.now(timezone.utc).isoformat()
    }

    # Filter out None values
    record = {k: v for k, v in record.items() if v is not None}

    # Add E2I entity references if provided
    if memory.e2i_refs:
        refs = memory.e2i_refs
        if refs.patient_journey_id:
            record["patient_journey_id"] = refs.patient_journey_id
        if refs.patient_id:
            record["patient_id"] = refs.patient_id
        if refs.hcp_id:
            record["hcp_id"] = refs.hcp_id
        if refs.treatment_event_id:
            record["treatment_event_id"] = refs.treatment_event_id
        if refs.trigger_id:
            record["trigger_id"] = refs.trigger_id
        if refs.prediction_id:
            record["prediction_id"] = refs.prediction_id
        if refs.causal_path_id:
            record["causal_path_id"] = refs.causal_path_id
        if refs.experiment_id:
            record["experiment_id"] = refs.experiment_id
        if refs.agent_activity_id:
            record["agent_activity_id"] = refs.agent_activity_id
        if refs.brand:
            record["brand"] = refs.brand
        if refs.region:
            record["region"] = refs.region

    client.table("episodic_memories").insert(record).execute()

    # Track memory statistics
    await _increment_memory_stats("episodic", memory.event_type)

    logger.info(f"Inserted episodic memory {memory_id} (type={memory.event_type})")
    return memory_id


async def insert_episodic_memory_with_text(
    memory: EpisodicMemoryInput,
    text_to_embed: Optional[str] = None,
    session_id: Optional[str] = None,
    cycle_id: Optional[str] = None
) -> str:
    """
    Insert episodic memory with auto-generated embedding.

    Args:
        memory: EpisodicMemoryInput with E2I entity references
        text_to_embed: Text to embed (defaults to description)
        session_id: Optional session ID
        cycle_id: Optional cognitive cycle ID

    Returns:
        ID of inserted memory
    """
    text = text_to_embed or memory.description
    embedding_service = get_embedding_service()
    embedding = await embedding_service.embed(text)

    return await insert_episodic_memory(
        memory=memory,
        embedding=embedding,
        session_id=session_id,
        cycle_id=cycle_id
    )


async def bulk_insert_episodic_memories(
    memories: List[Tuple[EpisodicMemoryInput, List[float]]],
    session_id: Optional[str] = None,
    cycle_id: Optional[str] = None
) -> List[str]:
    """
    Bulk insert multiple episodic memories for performance.

    Args:
        memories: List of (EpisodicMemoryInput, embedding) tuples
        session_id: Optional session ID for all memories
        cycle_id: Optional cycle ID for all memories

    Returns:
        List of inserted memory IDs
    """
    client = get_supabase_client()
    memory_ids = []
    records = []
    now = datetime.now(timezone.utc).isoformat()

    for memory, embedding in memories:
        memory_id = str(uuid.uuid4())
        memory_ids.append(memory_id)

        record = {
            "memory_id": memory_id,
            "session_id": session_id,
            "cycle_id": cycle_id,
            "event_type": memory.event_type,
            "event_subtype": memory.event_subtype,
            "description": memory.description,
            "raw_content": json.dumps(memory.raw_content or {}),
            "entities": json.dumps(memory.entities or {}),
            "outcome_type": memory.outcome_type,
            "outcome_details": json.dumps(memory.outcome_details or {}),
            "user_satisfaction_score": memory.user_satisfaction_score,
            "agent_name": memory.agent_name,
            "importance_score": memory.importance_score,
            "embedding": embedding,
            "occurred_at": now
        }

        # Filter out None values
        record = {k: v for k, v in record.items() if v is not None}

        # Add E2I entity references if provided
        if memory.e2i_refs:
            refs = memory.e2i_refs
            if refs.patient_journey_id:
                record["patient_journey_id"] = refs.patient_journey_id
            if refs.patient_id:
                record["patient_id"] = refs.patient_id
            if refs.hcp_id:
                record["hcp_id"] = refs.hcp_id
            if refs.treatment_event_id:
                record["treatment_event_id"] = refs.treatment_event_id
            if refs.trigger_id:
                record["trigger_id"] = refs.trigger_id
            if refs.prediction_id:
                record["prediction_id"] = refs.prediction_id
            if refs.causal_path_id:
                record["causal_path_id"] = refs.causal_path_id
            if refs.experiment_id:
                record["experiment_id"] = refs.experiment_id
            if refs.agent_activity_id:
                record["agent_activity_id"] = refs.agent_activity_id
            if refs.brand:
                record["brand"] = refs.brand
            if refs.region:
                record["region"] = refs.region

        records.append(record)

    if records:
        client.table("episodic_memories").insert(records).execute()

    logger.info(f"Bulk inserted {len(memory_ids)} episodic memories")
    return memory_ids


# ============================================================================
# EPISODIC MEMORY CONTEXT FUNCTIONS
# ============================================================================

async def get_memory_entity_context(memory_id: str) -> E2IEntityContext:
    """
    Get linked E2I entity details for a memory.
    Uses the get_memory_entity_context database function.

    Args:
        memory_id: UUID of the episodic memory

    Returns:
        E2IEntityContext with details of linked entities
    """
    client = get_supabase_client()

    try:
        result = client.rpc("get_memory_entity_context", {"p_memory_id": memory_id}).execute()

        context = E2IEntityContext()

        for row in result.data or []:
            entity_type = row.get("entity_type")
            details = {
                "id": row.get("entity_id"),
                "name": row.get("entity_name"),
                **row.get("entity_details", {})
            }

            if entity_type == "patient":
                context.patient = details
            elif entity_type == "hcp":
                context.hcp = details
            elif entity_type == "trigger":
                context.trigger = details
            elif entity_type == "causal_path":
                context.causal_path = details
            elif entity_type == "treatment":
                context.treatment = details
            elif entity_type == "prediction":
                context.prediction = details
            elif entity_type == "agent_activity":
                context.agent_activity = details

        return context
    except Exception as e:
        logger.warning(f"get_memory_entity_context failed: {e}")
        return E2IEntityContext()


async def get_enriched_episodic_memory(memory_id: str) -> Optional[EnrichedEpisodicMemory]:
    """
    Get episodic memory with full E2I data layer context attached.

    This function retrieves the memory and uses the get_memory_entity_context()
    RPC to attach patient, HCP, trigger, and causal path details.

    Args:
        memory_id: UUID of the episodic memory

    Returns:
        EnrichedEpisodicMemory with all linked entity context, or None if not found
    """
    client = get_supabase_client()

    # Fetch the base memory
    result = client.table("episodic_memories") \
        .select("*") \
        .eq("memory_id", memory_id) \
        .single() \
        .execute()

    if not result.data:
        return None

    memory = result.data

    # Fetch entity context
    context = await get_memory_entity_context(memory_id)

    return EnrichedEpisodicMemory(
        memory_id=memory["memory_id"],
        event_type=memory["event_type"],
        event_subtype=memory.get("event_subtype"),
        description=memory["description"],
        occurred_at=memory["occurred_at"],
        outcome_type=memory.get("outcome_type"),
        agent_name=memory.get("agent_name"),
        confidence_score=memory.get("importance_score"),
        patient_context=asdict(context)["patient"] if context.patient else None,
        hcp_context=asdict(context)["hcp"] if context.hcp else None,
        trigger_context=asdict(context)["trigger"] if context.trigger else None,
        causal_path_context=asdict(context)["causal_path"] if context.causal_path else None,
        treatment_context=asdict(context)["treatment"] if context.treatment else None,
        prediction_context=asdict(context)["prediction"] if context.prediction else None
    )


async def get_agent_activity_with_context(activity_id: str) -> Optional[AgentActivityContext]:
    """
    Get agent activity with full E2I context.
    Uses the get_agent_activity_context() database function.

    Args:
        activity_id: ID of the agent activity

    Returns:
        AgentActivityContext with linked triggers, causal paths, predictions
    """
    client = get_supabase_client()

    try:
        result = client.rpc("get_agent_activity_context", {"p_activity_id": activity_id}).execute()

        if not result.data or len(result.data) == 0:
            return None

        data = result.data[0]

        return AgentActivityContext(
            activity_id=data["activity_id"],
            agent_name=data["agent_name"],
            action_type=data["action_type"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            status=data["status"],
            trigger=data.get("trigger"),
            causal_paths=data.get("causal_paths"),
            predictions=data.get("predictions"),
            duration_ms=data.get("duration_ms"),
            tokens_used=data.get("tokens_used")
        )
    except Exception as e:
        logger.warning(f"get_agent_activity_context failed: {e}")
        return None


async def get_causal_path_context(path_id: str) -> Optional[Dict[str, Any]]:
    """
    Get full context for a causal path including linked memories.

    Args:
        path_id: ID of the causal path

    Returns:
        Dict with causal path details, discovery agent, and related memories
    """
    client = get_supabase_client()

    # Get causal path details
    path_result = client.table("causal_paths") \
        .select("*") \
        .eq("path_id", path_id) \
        .single() \
        .execute()

    if not path_result.data:
        return None

    path = path_result.data

    # Get related episodic memories
    memories = await search_episodic_by_e2i_entity(
        entity_type=E2IEntityType.CAUSAL_PATH,
        entity_id=path_id,
        limit=10
    )

    return {
        "path_id": path_id,
        "source_entity": path.get("source_entity"),
        "target_entity": path.get("target_entity"),
        "effect_size": path.get("effect_size"),
        "confidence": path.get("confidence"),
        "method_used": path.get("method_used"),
        "discovery_date": path.get("created_at"),
        "related_memories": memories
    }


# ============================================================================
# EPISODIC MEMORY UTILITY FUNCTIONS
# ============================================================================

async def get_recent_memories(
    limit: int = 20,
    event_types: Optional[List[str]] = None,
    agent_name: Optional[str] = None,
    brand: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get recent episodic memories ordered by time.

    Args:
        limit: Maximum results
        event_types: Optional filter by event types
        agent_name: Optional filter by agent
        brand: Optional filter by brand

    Returns:
        List of recent memories
    """
    client = get_supabase_client()

    query = client.table("episodic_memories") \
        .select("*") \
        .order("occurred_at", desc=True) \
        .limit(limit)

    if event_types:
        query = query.in_("event_type", event_types)
    if agent_name:
        query = query.eq("agent_name", agent_name)
    if brand:
        query = query.eq("brand", brand)

    result = query.execute()
    return result.data or []


async def get_memory_by_id(memory_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single episodic memory by ID.

    Args:
        memory_id: UUID of the memory

    Returns:
        Memory dict or None if not found
    """
    client = get_supabase_client()

    result = client.table("episodic_memories") \
        .select("*") \
        .eq("memory_id", memory_id) \
        .single() \
        .execute()

    return result.data


async def delete_memory(memory_id: str) -> bool:
    """
    Delete an episodic memory.

    Args:
        memory_id: UUID of the memory to delete

    Returns:
        True if deleted, False if not found
    """
    client = get_supabase_client()

    result = client.table("episodic_memories") \
        .delete() \
        .eq("memory_id", memory_id) \
        .execute()

    deleted = len(result.data or []) > 0
    if deleted:
        logger.info(f"Deleted episodic memory {memory_id}")
    return deleted


async def count_memories_by_type(
    event_type: Optional[str] = None,
    brand: Optional[str] = None,
    days_back: int = 30
) -> int:
    """
    Count episodic memories with optional filters.

    Args:
        event_type: Optional event type filter
        brand: Optional brand filter
        days_back: How many days back to count

    Returns:
        Count of matching memories
    """
    client = get_supabase_client()

    query = client.table("episodic_memories") \
        .select("memory_id", count="exact")

    if event_type:
        query = query.eq("event_type", event_type)
    if brand:
        query = query.eq("brand", brand)

    # Note: For date filtering, we'd need to use .gte() with a date
    # This is simplified for now
    result = query.execute()
    return result.count or 0


async def sync_treatment_relationships_to_cache() -> int:
    """
    Sync HCP-Patient treatment relationships to semantic_memory_cache.
    Uses the sync_hcp_patient_relationships_to_cache() database function.

    This should be called periodically to keep the cache updated with
    the latest treatment relationships from the data layer.

    Returns:
        Count of synced relationships
    """
    client = get_supabase_client()

    try:
        result = client.rpc("sync_hcp_patient_relationships_to_cache", {}).execute()

        count = result.data if isinstance(result.data, int) else 0
        logger.info(f"Synced {count} treatment relationships to semantic cache")

        return count
    except Exception as e:
        logger.warning(f"sync_hcp_patient_relationships_to_cache failed: {e}")
        return 0
