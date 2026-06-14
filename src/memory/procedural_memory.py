"""
E2I Agentic Memory - Procedural Memory (Pattern Learning)
Stores successful agent patterns and provides few-shot examples for DSPy.

Technology: Supabase (PostgreSQL + pgvector)

Features:
- Procedure discovery via embedding similarity
- Few-shot example retrieval for in-context learning
- Success rate tracking and optimization
- Learning signal recording for DSPy training
- E2I context filtering (brand, region, agent)

Usage:
    from src.memory.procedural_memory import (
        find_relevant_procedures,
        insert_procedural_memory,
        get_few_shot_examples,
        record_learning_signal
    )

    # Find relevant procedures for a query
    procedures = await find_relevant_procedures(
        embedding=query_embedding,
        intent="kpi_investigation",
        brand="Kisqali"
    )

    # Record feedback for DSPy training
    await record_learning_signal(
        signal=LearningSignalInput(
            signal_type="thumbs_up",
            rated_agent="causal_impact",
            is_training_example=True
        ),
        cycle_id="cycle_123"
    )
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast

from src.memory.services.config import get_config
from src.memory.services.factories import (
    get_embedding_service,
    get_supabase_client,
    validate_embedding_dimensions,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ProceduralMemoryInput:
    """Input for creating a procedural memory."""

    procedure_name: str
    tool_sequence: List[Dict[str, Any]]
    procedure_type: str = "tool_sequence"
    trigger_pattern: Optional[str] = None
    intent_keywords: Optional[List[str]] = None
    detected_intent: Optional[str] = None
    applicable_brands: Optional[List[str]] = None
    applicable_regions: Optional[List[str]] = None
    applicable_agents: Optional[List[str]] = None


@dataclass
class LearningSignalInput:
    """Input for recording a learning signal."""

    signal_type: str  # thumbs_up, thumbs_down, correction, rating
    signal_value: Optional[float] = None
    signal_details: Optional[Dict[str, Any]] = None
    applies_to_type: Optional[str] = None
    applies_to_id: Optional[str] = None
    # E2I context
    related_patient_id: Optional[str] = None
    related_hcp_id: Optional[str] = None
    related_trigger_id: Optional[str] = None
    brand: Optional[str] = None
    region: Optional[str] = None
    rated_agent: Optional[str] = None
    # DSPy training
    is_training_example: bool = False
    dspy_metric_name: Optional[str] = None
    dspy_metric_value: Optional[float] = None
    training_input: Optional[str] = None
    training_output: Optional[str] = None


# ============================================================================
# PROCEDURAL MEMORY FUNCTIONS
# ============================================================================


async def find_relevant_procedures(
    embedding: List[float],
    procedure_type: Optional[str] = None,
    intent: Optional[str] = None,
    brand: Optional[str] = None,
    limit: int = 5,
    min_similarity: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Find relevant procedures (few-shot examples) with E2I context matching.

    Args:
        embedding: Query embedding vector
        procedure_type: Filter by procedure type
        intent: Filter by detected intent
        brand: Filter by applicable brand
        limit: Maximum results
        min_similarity: Minimum similarity threshold

    Returns:
        List of matching procedures with similarity scores
    """
    client = get_supabase_client()

    result = client.rpc(
        "find_relevant_procedures",
        {
            "query_embedding": embedding,
            "match_threshold": min_similarity,
            "match_count": limit,
            "filter_type": procedure_type,
            "filter_intent": intent,
            "filter_brand": brand,
        },
    ).execute()

    logger.debug(f"Found {len(result.data or [])} relevant procedures")
    return result.data or []


async def find_relevant_procedures_by_text(
    query_text: str,
    procedure_type: Optional[str] = None,
    intent: Optional[str] = None,
    brand: Optional[str] = None,
    limit: int = 5,
    min_similarity: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Find relevant procedures by text query (auto-generates embedding).

    Args:
        query_text: Text query to search for
        procedure_type: Filter by procedure type
        intent: Filter by detected intent
        brand: Filter by applicable brand
        limit: Maximum results
        min_similarity: Minimum similarity threshold

    Returns:
        List of matching procedures with similarity scores
    """
    embedding_service = get_embedding_service()
    embedding = await embedding_service.embed(query_text)

    return await find_relevant_procedures(
        embedding=embedding,
        procedure_type=procedure_type,
        intent=intent,
        brand=brand,
        limit=limit,
        min_similarity=min_similarity,
    )


async def insert_procedural_memory(
    procedure: ProceduralMemoryInput,
    trigger_embedding: List[float],
    dedup_name_prefix: Optional[str] = None,
) -> str:
    """
    Insert or update procedural memory with E2I context.

    If a similar procedure exists (similarity > 0.9), updates usage counts.
    Otherwise, creates a new procedure.

    Args:
        procedure: ProceduralMemoryInput with procedure details
        trigger_embedding: Embedding of the trigger pattern
        dedup_name_prefix: When set, the similarity dedup only matches rows
            whose procedure_name starts with this prefix (#883: the dedup
            filters by procedure_type only, and types like 'optimization' are
            shared across writers — without a name guard a high-similarity
            FOREIGN row would be "updated" and returned, silently dropping the
            new pattern). Default None preserves the historical behavior.

    Returns:
        ID of inserted or updated procedure
    """
    # M1: reject a dimension-mismatched trigger embedding (e.g. a 384-dim fallback)
    # before any DB call — the similarity search below and the insert both target a
    # vector(1536) column.
    validate_embedding_dimensions(
        trigger_embedding, get_config().procedural.vector_dims, context="procedural embedding"
    )

    client = get_supabase_client()

    # Check for existing similar procedure. With a name-prefix guard we fetch a
    # few candidates because the single best match may be a foreign row.
    existing = await find_relevant_procedures(
        trigger_embedding,
        procedure.procedure_type,
        limit=5 if dedup_name_prefix is not None else 1,
        min_similarity=0.9,
    )
    if dedup_name_prefix is not None:
        existing = [
            row
            for row in existing
            if str(row.get("procedure_name", "")).startswith(dedup_name_prefix)
        ]

    if existing:
        procedure_id = existing[0]["procedure_id"]

        client.table("procedural_memories").update(
            {
                "usage_count": existing[0].get("usage_count", 0) + 1,
                "success_count": existing[0].get("success_count", 0) + 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("procedure_id", procedure_id).execute()

        logger.info(f"Updated existing procedure {procedure_id}")
        return cast(str, procedure_id)

    procedure_id = str(uuid.uuid4())

    record = {
        "procedure_id": procedure_id,
        "procedure_name": procedure.procedure_name,
        "procedure_type": procedure.procedure_type,
        # #883 deferred: pass the structure raw — postgrest JSON-encodes the
        # payload itself, so a pre-dumped string double-encodes (the column
        # stores a JSON *string scalar*, not an array; live DB had 1566/1566
        # such rows). Old string rows are repaired by migration 072 and
        # readers stay tolerant of both shapes.
        "tool_sequence": procedure.tool_sequence,
        "trigger_pattern": procedure.trigger_pattern,
        "trigger_embedding": trigger_embedding,
        "intent_keywords": procedure.intent_keywords or [],
        "detected_intent": procedure.detected_intent,
        "applicable_brands": procedure.applicable_brands or ["all"],
        "applicable_regions": procedure.applicable_regions or ["all"],
        "applicable_agents": procedure.applicable_agents or [],
        "usage_count": 1,
        "success_count": 1,
        "is_active": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Filter out None values
    record = {k: v for k, v in record.items() if v is not None}

    client.table("procedural_memories").insert(record).execute()

    # Track memory statistics
    await _increment_memory_stats("procedural", procedure.procedure_type)

    logger.info(f"Inserted new procedure {procedure_id} (type={procedure.procedure_type})")
    return procedure_id


async def insert_procedural_memory_with_text(
    procedure: ProceduralMemoryInput,
    trigger_text: Optional[str] = None,
    dedup_name_prefix: Optional[str] = None,
) -> str:
    """
    Insert procedural memory with auto-generated embedding.

    Args:
        procedure: ProceduralMemoryInput with procedure details
        trigger_text: Text to embed (defaults to trigger_pattern)
        dedup_name_prefix: Optional name-prefix guard for the similarity dedup
            (see :func:`insert_procedural_memory`)

    Returns:
        ID of inserted or updated procedure
    """
    text = trigger_text or procedure.trigger_pattern or procedure.procedure_name
    embedding_service = get_embedding_service()
    embedding = await embedding_service.embed(text)

    return await insert_procedural_memory(
        procedure=procedure,
        trigger_embedding=embedding,
        dedup_name_prefix=dedup_name_prefix,
    )


async def get_few_shot_examples(
    query_embedding: List[float],
    intent: Optional[str] = None,
    brand: Optional[str] = None,
    max_examples: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get few-shot examples for in-context learning with E2I context.

    Args:
        query_embedding: Query embedding vector
        intent: Filter by detected intent
        brand: Filter by applicable brand
        max_examples: Maximum number of examples

    Returns:
        List of formatted few-shot examples
    """
    procedures = await find_relevant_procedures(
        embedding=query_embedding,
        intent=intent,
        brand=brand,
        limit=max_examples,
        min_similarity=0.6,
    )

    examples = []
    for proc in procedures:
        tool_sequence = proc.get("tool_sequence", [])
        if isinstance(tool_sequence, str):
            # Pre-#883-fix rows hold a double-encoded JSON string scalar
            # (repaired by migration 072, but tolerate un-migrated envs); a
            # malformed survivor must not break the whole example set.
            try:
                tool_sequence = json.loads(tool_sequence)
            except (ValueError, TypeError):
                tool_sequence = []

        examples.append(
            {
                "trigger": proc.get("trigger_pattern", ""),
                "intent": proc.get("detected_intent"),
                "solution": tool_sequence,
                "success_rate": proc.get("success_rate", 0),
                "applicable_brands": proc.get("applicable_brands", []),
                "applicable_regions": proc.get("applicable_regions", []),
            }
        )

    logger.debug(f"Retrieved {len(examples)} few-shot examples")
    return examples


async def get_few_shot_examples_by_text(
    query_text: str,
    intent: Optional[str] = None,
    brand: Optional[str] = None,
    max_examples: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get few-shot examples by text query (auto-generates embedding).

    Args:
        query_text: Text query to search for
        intent: Filter by detected intent
        brand: Filter by applicable brand
        max_examples: Maximum number of examples

    Returns:
        List of formatted few-shot examples
    """
    embedding_service = get_embedding_service()
    embedding = await embedding_service.embed(query_text)

    return await get_few_shot_examples(
        query_embedding=embedding, intent=intent, brand=brand, max_examples=max_examples
    )


async def update_procedure_outcome(procedure_id: str, success: bool) -> None:
    """
    Update procedure usage and success counts.

    Args:
        procedure_id: ID of the procedure
        success: Whether the procedure was successful
    """
    client = get_supabase_client()

    # L2 (#694): atomic server-side increment (migration 036) instead of a
    # read-modify-write (SELECT counts -> UPDATE counts+1), which lost updates
    # under concurrent outcomes. The RPC (RETURNS TABLE) yields one row when a
    # procedure was updated and an EMPTY result when none matched, so an empty
    # result.data means "not found". success_rate is a GENERATED column and
    # recomputes from usage_count/success_count automatically.
    result = client.rpc(
        "increment_procedure_outcome",
        {"p_procedure_id": procedure_id, "p_success": success},
    ).execute()

    if not result.data:
        logger.warning(f"Procedure {procedure_id} not found for outcome update")

    logger.debug(f"Updated procedure {procedure_id} outcome (success={success})")


async def get_procedure_by_id(procedure_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single procedure by ID.

    Args:
        procedure_id: ID of the procedure

    Returns:
        Procedure dict or None if not found
    """
    client = get_supabase_client()

    result = (
        client.table("procedural_memories")
        .select("*")
        .eq("procedure_id", procedure_id)
        .single()
        .execute()
    )

    return cast(Optional[Dict[str, Any]], result.data)


async def deactivate_procedure(procedure_id: str) -> bool:
    """
    Deactivate a procedure (soft delete).

    Args:
        procedure_id: ID of the procedure

    Returns:
        True if deactivated, False if not found
    """
    client = get_supabase_client()

    result = (
        client.table("procedural_memories")
        .update({"is_active": False, "updated_at": datetime.now(timezone.utc).isoformat()})
        .eq("procedure_id", procedure_id)
        .execute()
    )

    deactivated = len(result.data or []) > 0
    if deactivated:
        logger.info(f"Deactivated procedure {procedure_id}")
    return deactivated


async def get_top_procedures(
    procedure_type: Optional[str] = None, brand: Optional[str] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get top procedures by success rate.

    Args:
        procedure_type: Filter by type
        brand: Filter by brand
        limit: Maximum results

    Returns:
        List of procedures ordered by success rate
    """
    client = get_supabase_client()

    query = (
        client.table("procedural_memories")
        .select("*")
        .eq("is_active", True)
        .order("success_count", desc=True)
        .limit(limit)
    )

    if procedure_type:
        query = query.eq("procedure_type", procedure_type)

    # Note: Brand filtering on array field requires custom handling
    # For now, we filter in Python
    result = query.execute()
    procedures = result.data or []

    if brand:
        procedures = [
            p
            for p in procedures
            if brand in p.get("applicable_brands", []) or "all" in p.get("applicable_brands", [])
        ]

    return procedures[:limit]


# ============================================================================
# LEARNING SIGNALS FUNCTIONS
# ============================================================================


async def record_learning_signal(
    signal: LearningSignalInput, cycle_id: Optional[str] = None, session_id: Optional[str] = None
) -> str:
    """
    Record a learning signal with E2I context.

    Args:
        signal: LearningSignalInput with signal details
        cycle_id: Optional cognitive cycle ID
        session_id: Optional session ID

    Returns:
        ID of the recorded signal
    """
    client = get_supabase_client()

    signal_id = str(uuid.uuid4())

    record = {
        "signal_id": signal_id,
        "cycle_id": cycle_id,
        "session_id": session_id,
        "signal_type": signal.signal_type,
        "signal_value": signal.signal_value,
        # #883 deferred: raw dict, NOT json.dumps — a pre-dumped string is
        # double-encoded by the client into a JSON string scalar (the exact
        # raw_content failure B2's get_prior_cohorts had to json.loads around).
        "signal_details": signal.signal_details or {},
        "applies_to_type": signal.applies_to_type,
        "applies_to_id": signal.applies_to_id,
        # E2I context
        "related_patient_id": signal.related_patient_id,
        "related_hcp_id": signal.related_hcp_id,
        "related_trigger_id": signal.related_trigger_id,
        "brand": signal.brand,
        "region": signal.region,
        "rated_agent": signal.rated_agent,
        # DSPy
        "is_training_example": signal.is_training_example,
        "dspy_metric_name": signal.dspy_metric_name,
        "dspy_metric_value": signal.dspy_metric_value,
        "training_input": signal.training_input,
        "training_output": signal.training_output,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Remove None values
    record = {k: v for k, v in record.items() if v is not None}

    client.table("learning_signals").insert(record).execute()

    logger.info(f"Recorded learning signal {signal_id} (type={signal.signal_type})")
    return signal_id


async def get_training_examples_for_agent(
    agent_name: str,
    brand: Optional[str] = None,
    min_score: float = 0.7,
    limit: int = 100,
    include_synthetic: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get high-quality training examples for a specific agent.
    Used for DSPy optimization.

    #894: ``learning_signals`` is is_synthetic-tagged (migration 069; live
    substrate 300/300 synthetic at filing time) — recalling planted signals
    into DSPy optimization would train self-learning on a synthetic twin.
    Real mode default-excludes; validation runs opt in.

    Args:
        agent_name: Name of the agent
        brand: Optional brand filter
        min_score: Minimum metric value threshold
        limit: Maximum results
        include_synthetic: When True, do not exclude synthetic rows (opt-in).

    Returns:
        List of training examples
    """
    from src.repositories.provenance import apply_provenance_filter

    client = get_supabase_client()

    query = (
        client.table("learning_signals")
        .select("*")
        .eq("rated_agent", agent_name)
        .eq("is_training_example", True)
        .gte("dspy_metric_value", min_score)
        .order("dspy_metric_value", desc=True)
        .limit(limit)
    )

    if brand:
        query = query.eq("brand", brand)
    query = apply_provenance_filter(query, include_synthetic)

    result = query.execute()
    logger.debug(f"Retrieved {len(result.data or [])} training examples for {agent_name}")
    return result.data or []


async def get_feedback_summary_for_trigger(
    trigger_id: str, include_synthetic: bool = False
) -> Dict[str, Any]:
    """
    Get aggregated feedback for a specific trigger.
    Useful for evaluating trigger effectiveness.

    Args:
        trigger_id: ID of the trigger
        include_synthetic: When True, do not exclude synthetic rows (opt-in, #894).

    Returns:
        Summary dict with feedback counts and ratings
    """
    from src.repositories.provenance import apply_provenance_filter

    client = get_supabase_client()

    query = (
        client.table("learning_signals")
        .select("signal_type, signal_value")
        .eq("related_trigger_id", trigger_id)
    )
    result = apply_provenance_filter(query, include_synthetic).execute()

    signals = result.data or []

    summary = {
        "trigger_id": trigger_id,
        "total_feedback": len(signals),
        "thumbs_up": sum(1 for s in signals if s["signal_type"] == "thumbs_up"),
        "thumbs_down": sum(1 for s in signals if s["signal_type"] == "thumbs_down"),
        "avg_rating": None,
        "corrections_count": sum(1 for s in signals if s["signal_type"] == "correction"),
    }

    ratings = [
        s["signal_value"] for s in signals if s["signal_type"] == "rating" and s["signal_value"]
    ]
    if ratings:
        summary["avg_rating"] = sum(ratings) / len(ratings)

    return summary


async def get_feedback_summary_for_agent(
    agent_name: str, include_synthetic: bool = False
) -> Dict[str, Any]:
    """
    Get aggregated feedback for a specific agent.

    Args:
        agent_name: Name of the agent
        include_synthetic: When True, do not exclude synthetic rows (opt-in, #894).

    Returns:
        Summary dict with feedback counts and ratings
    """
    from src.repositories.provenance import apply_provenance_filter

    client = get_supabase_client()

    query = (
        client.table("learning_signals")
        .select("signal_type, signal_value")
        .eq("rated_agent", agent_name)
    )
    result = apply_provenance_filter(query, include_synthetic).execute()

    signals = result.data or []

    summary = {
        "agent_name": agent_name,
        "total_feedback": len(signals),
        "thumbs_up": sum(1 for s in signals if s["signal_type"] == "thumbs_up"),
        "thumbs_down": sum(1 for s in signals if s["signal_type"] == "thumbs_down"),
        "avg_rating": None,
        "corrections_count": sum(1 for s in signals if s["signal_type"] == "correction"),
        "training_examples": sum(1 for s in signals if s.get("is_training_example")),
    }

    ratings = [
        s["signal_value"] for s in signals if s["signal_type"] == "rating" and s["signal_value"]
    ]
    if ratings:
        summary["avg_rating"] = sum(ratings) / len(ratings)

    return summary


async def get_recent_signals(
    limit: int = 50,
    signal_type: Optional[str] = None,
    agent_name: Optional[str] = None,
    include_synthetic: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get recent learning signals.

    Args:
        limit: Maximum results
        signal_type: Filter by signal type
        agent_name: Filter by agent
        include_synthetic: When True, do not exclude synthetic rows (opt-in, #894).

    Returns:
        List of recent signals
    """
    from src.repositories.provenance import apply_provenance_filter

    client = get_supabase_client()

    query = client.table("learning_signals").select("*").order("created_at", desc=True).limit(limit)

    if signal_type:
        query = query.eq("signal_type", signal_type)
    if agent_name:
        query = query.eq("rated_agent", agent_name)
    query = apply_provenance_filter(query, include_synthetic)

    result = query.execute()
    return result.data or []


# ============================================================================
# MEMORY STATISTICS FUNCTIONS
# ============================================================================


async def _increment_memory_stats(memory_type: str, subtype: Optional[str] = None) -> None:
    """
    Observability hook for a memory write (log-only, no DB persistence).

    The live ``memory_statistics`` table (database/memory/001_agentic_memory_schema_v1.3.sql,
    "Aggregated memory system metrics") is an *hourly aggregated rollup* keyed by
    ``UNIQUE(stat_date, stat_hour)`` with denormalized columns
    (``episodic_count`` / ``procedural_count`` / ``semantic_cache_count`` + cycle,
    performance, quality and distribution metrics). It is populated by a separate
    aggregation job, NOT by per-write increments.

    The previous implementation upserted a normalized ``(memory_type, subtype, count)``
    row with ``on_conflict="stat_date,memory_type,subtype"`` — none of those columns or
    constraints exist on the real table, so every call raised Postgres 42703
    (undefined_column) and the error was swallowed. The procedural card therefore read 0
    even though ``procedural_memories`` holds the real rows.

    Live memory counts are sourced directly from their source tables instead
    (:func:`count_procedures` here, mirroring ``count_memories_by_type`` for episodic),
    so this hook is now a no-op log line — matching the sibling episodic hook
    (``src/memory/episodic_memory.py``) and avoiding a silently-failing write.

    Args:
        memory_type: episodic, procedural, semantic
        subtype: Event type or procedure type
    """
    logger.debug(f"Memory stat: {memory_type}/{subtype or 'general'} +1")


async def count_procedures(
    procedure_type: Optional[str] = None,
    active_only: bool = True,
) -> int:
    """
    Count procedural memories directly from the ``procedural_memories`` table.

    This mirrors the episodic ``count_memories_by_type`` pattern: live memory counts
    are read from the source table, not from the (separate, aggregation-job-populated)
    ``memory_statistics`` rollup. ``procedural_memories`` has no ``is_synthetic``
    provenance column, so no provenance filter applies.

    Args:
        procedure_type: Optional filter by ``procedure_type``.
        active_only: When True (default) count only ``is_active = true`` procedures.

    Returns:
        Count of matching procedural memories.
    """
    client = get_supabase_client()

    query = client.table("procedural_memories").select("procedure_id", count="exact")

    if procedure_type:
        query = query.eq("procedure_type", procedure_type)
    if active_only:
        query = query.eq("is_active", True)

    result = query.execute()
    return cast(int, result.count or 0)


async def get_procedural_stats(active_only: bool = True) -> Dict[str, Any]:
    """
    Live procedural-memory statistics sourced directly from ``procedural_memories``.

    Returns the total procedure count and the average per-procedure ``success_rate``
    (the column maintained on each row by ``update_procedure_outcome``). This replaces
    the broken ``memory_statistics`` lookup that always returned 0 (see
    :func:`_increment_memory_stats` for the schema-drift root cause).

    Args:
        active_only: When True (default) consider only ``is_active = true`` procedures.

    Returns:
        Dict with ``total_procedures`` (int) and ``average_success_rate`` (float).
    """
    client = get_supabase_client()

    total = await count_procedures(active_only=active_only)

    # Average the per-row ``success_rate`` across ALL procedures. Without
    # ``.range()`` PostgREST silently caps the response at ~1000 rows, so at the
    # live volume (1,566 rows) a single SELECT would average only the first page
    # and return a plausible-but-wrong rate. Page through ``.range()`` windows,
    # ordered by the unique PK (``procedure_id``) for a stable offset, and
    # accumulate a running (sum, count) so we never hold the full set in memory.
    # Mirrors the L7/#694 pagination pattern in crystallizer/consolidator.
    page_size = 1000
    offset = 0
    rate_sum = 0.0
    rate_count = 0
    while True:
        page_query = client.table("procedural_memories").select("success_rate")
        if active_only:
            page_query = page_query.eq("is_active", True)
        page_query = page_query.order("procedure_id").range(offset, offset + page_size - 1)
        page = page_query.execute().data or []
        for row in page:
            value = row.get("success_rate")
            if value is not None:
                rate_sum += value
                rate_count += 1
        if len(page) < page_size:
            break
        offset += page_size

    average_success_rate = rate_sum / rate_count if rate_count else 0.0

    return {
        "total_procedures": total,
        "average_success_rate": average_success_rate,
    }


async def get_memory_statistics(
    days_back: int = 30, memory_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read the aggregated ``memory_statistics`` rollup (hourly snapshots).

    Reads the real, live columns of the rollup table
    (``episodic_count`` / ``procedural_count`` / ``semantic_cache_count`` keyed by
    ``stat_date`` / ``stat_hour``). The previous implementation selected a
    nonexistent ``memory_type`` / ``count`` long-format shape, which would have
    raised 42703 against the live schema.

    NOTE: This rollup is populated by a separate aggregation job and may be empty.
    For the live "total procedures" / "success rate" surface, use
    :func:`get_procedural_stats`, which reads ``procedural_memories`` directly.

    Args:
        days_back: Number of days to look back.
        memory_type: Optional logical type ("episodic" | "procedural" |
            "semantic_cache") — selects which rollup count column to aggregate.

    Returns:
        Dict with ``period_days``, ``totals_by_type`` (per-type rollup sums), and
        ``daily_breakdown`` (raw rollup rows).
    """
    client = get_supabase_client()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()

    result = (
        client.table("memory_statistics")
        .select("*")
        .gte("stat_date", cutoff)
        .order("stat_date", desc=True)
        .execute()
    )
    stats = result.data or []

    count_columns = {
        "episodic": "episodic_count",
        "procedural": "procedural_count",
        "semantic_cache": "semantic_cache_count",
    }
    selected = (
        {memory_type: count_columns[memory_type]} if memory_type in count_columns else count_columns
    )

    totals: Dict[str, int] = {}
    for stat in stats:
        for logical_type, column in selected.items():
            totals[logical_type] = totals.get(logical_type, 0) + (stat.get(column) or 0)

    return {"period_days": days_back, "totals_by_type": totals, "daily_breakdown": stats}
