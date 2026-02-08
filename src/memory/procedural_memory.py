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

from src.memory.services.factories import get_embedding_service, get_supabase_client

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
    procedure: ProceduralMemoryInput, trigger_embedding: List[float]
) -> str:
    """
    Insert or update procedural memory with E2I context.

    If a similar procedure exists (similarity > 0.9), updates usage counts.
    Otherwise, creates a new procedure.

    Args:
        procedure: ProceduralMemoryInput with procedure details
        trigger_embedding: Embedding of the trigger pattern

    Returns:
        ID of inserted or updated procedure
    """
    client = get_supabase_client()

    # Check for existing similar procedure
    existing = await find_relevant_procedures(
        trigger_embedding, procedure.procedure_type, limit=1, min_similarity=0.9
    )

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
        "tool_sequence": json.dumps(procedure.tool_sequence),
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
    procedure: ProceduralMemoryInput, trigger_text: Optional[str] = None
) -> str:
    """
    Insert procedural memory with auto-generated embedding.

    Args:
        procedure: ProceduralMemoryInput with procedure details
        trigger_text: Text to embed (defaults to trigger_pattern)

    Returns:
        ID of inserted or updated procedure
    """
    text = trigger_text or procedure.trigger_pattern or procedure.procedure_name
    embedding_service = get_embedding_service()
    embedding = await embedding_service.embed(text)

    return await insert_procedural_memory(procedure=procedure, trigger_embedding=embedding)


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
            tool_sequence = json.loads(tool_sequence)

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

    # Get current counts
    result = (
        client.table("procedural_memories")
        .select("usage_count, success_count")
        .eq("procedure_id", procedure_id)
        .single()
        .execute()
    )

    if not result.data:
        logger.warning(f"Procedure {procedure_id} not found for outcome update")
        return

    current = result.data
    updates = {
        "usage_count": current.get("usage_count", 0) + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if success:
        updates["success_count"] = current.get("success_count", 0) + 1

    client.table("procedural_memories").update(updates).eq("procedure_id", procedure_id).execute()

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
        "signal_details": json.dumps(signal.signal_details or {}),
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
    agent_name: str, brand: Optional[str] = None, min_score: float = 0.7, limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get high-quality training examples for a specific agent.
    Used for DSPy optimization.

    Args:
        agent_name: Name of the agent
        brand: Optional brand filter
        min_score: Minimum metric value threshold
        limit: Maximum results

    Returns:
        List of training examples
    """
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

    result = query.execute()
    logger.debug(f"Retrieved {len(result.data or [])} training examples for {agent_name}")
    return result.data or []


async def get_feedback_summary_for_trigger(trigger_id: str) -> Dict[str, Any]:
    """
    Get aggregated feedback for a specific trigger.
    Useful for evaluating trigger effectiveness.

    Args:
        trigger_id: ID of the trigger

    Returns:
        Summary dict with feedback counts and ratings
    """
    client = get_supabase_client()

    result = (
        client.table("learning_signals")
        .select("signal_type, signal_value")
        .eq("related_trigger_id", trigger_id)
        .execute()
    )

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


async def get_feedback_summary_for_agent(agent_name: str) -> Dict[str, Any]:
    """
    Get aggregated feedback for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        Summary dict with feedback counts and ratings
    """
    client = get_supabase_client()

    result = (
        client.table("learning_signals")
        .select("signal_type, signal_value")
        .eq("rated_agent", agent_name)
        .execute()
    )

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
    limit: int = 50, signal_type: Optional[str] = None, agent_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get recent learning signals.

    Args:
        limit: Maximum results
        signal_type: Filter by signal type
        agent_name: Filter by agent

    Returns:
        List of recent signals
    """
    client = get_supabase_client()

    query = client.table("learning_signals").select("*").order("created_at", desc=True).limit(limit)

    if signal_type:
        query = query.eq("signal_type", signal_type)
    if agent_name:
        query = query.eq("rated_agent", agent_name)

    result = query.execute()
    return result.data or []


# ============================================================================
# MEMORY STATISTICS FUNCTIONS
# ============================================================================


async def _increment_memory_stats(memory_type: str, subtype: Optional[str] = None) -> None:
    """
    Track memory usage statistics for monitoring.

    Args:
        memory_type: episodic, procedural, semantic
        subtype: Event type or procedure type
    """
    client = get_supabase_client()

    today = datetime.now(timezone.utc).date().isoformat()

    try:
        # Upsert stats record
        client.table("memory_statistics").upsert(
            {
                "stat_date": today,
                "memory_type": memory_type,
                "subtype": subtype or "general",
                "count": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="stat_date,memory_type,subtype",
        ).execute()
    except Exception as e:
        # Stats are non-critical, just log
        logger.debug(f"Failed to update memory stats: {e}")


async def get_memory_statistics(
    days_back: int = 30, memory_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get memory usage statistics for monitoring.

    Args:
        days_back: Number of days to look back
        memory_type: Optional filter by memory type

    Returns:
        Dict with counts by type and trends
    """
    client = get_supabase_client()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()

    query = (
        client.table("memory_statistics")
        .select("*")
        .gte("stat_date", cutoff)
        .order("stat_date", desc=True)
    )

    if memory_type:
        query = query.eq("memory_type", memory_type)

    result = query.execute()
    stats = result.data or []

    # Aggregate by type
    totals = {}
    for stat in stats:
        mt = stat["memory_type"]
        if mt not in totals:
            totals[mt] = 0
        totals[mt] += stat.get("count", 0)

    return {"period_days": days_back, "totals_by_type": totals, "daily_breakdown": stats}
