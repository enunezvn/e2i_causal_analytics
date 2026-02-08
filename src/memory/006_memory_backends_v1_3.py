"""
E2I Agentic Memory - Backend Implementations v1.3
Storage and retrieval operations with E2I Data Layer Integration

Architecture:
┌─────────────────┬─────────────────────────────────────┬─────────────────────────────┐
│ Memory Type     │ Function                            │ Technology                  │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────┤
│ Short-Term      │ Current context, scratchpad,        │ Redis + LangGraph           │
│ (Working)       │ immediate message history           │ MemorySaver checkpointer    │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────┤
│ Episodic        │ Experiences: "What did I do?"       │ Supabase (Postgres +        │
│ (Long-Term)     │ "What happened yesterday?"          │ pgvector) + E2I FKs         │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────┤
│ Semantic        │ Facts: "What is the relationship    │ FalkorDB + Graphity         │
│ (Long-Term)     │ between Project X and User Y?"      │ + Supabase cache            │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────┤
│ Procedural      │ Skills: "How did I solve this       │ Supabase (vector store of   │
│ (Long-Term)     │ error last time?"                   │ tool call sequences)        │
└─────────────────┴─────────────────────────────────────┴─────────────────────────────┘

Version: 1.3
Changes from v1.2:
  - Added get_enriched_episodic_memory() for full entity context retrieval
  - Added get_agent_activity_with_context() using 001b RPC function
  - Added sync_treatment_relationships_to_cache() for periodic cache updates
  - Added get_causal_path_context() for causal analysis enrichment
  - Added bulk memory operations for performance
  - Improved error handling and logging
  - Added memory statistics tracking

Requires: 001_agentic_memory_schema_v1.3.sql + 001b_add_foreign_keys_v3.sql
"""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================


def load_config() -> Dict[str, Any]:
    """Load memory configuration from YAML."""
    config_path = Path(__file__).parent / "005_memory_config.yaml"
    with open(config_path) as f:
        result: Dict[str, Any] = yaml.safe_load(f)
        return result


CONFIG = load_config()
ENVIRONMENT = CONFIG.get("environment", "local_pilot")


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
    """E2I 18-agent architecture names (6 tiers)."""

    # Tier 0: ML Foundation (7 agents)
    SCOPE_DEFINER = "scope_definer"
    DATA_PREPARER = "data_preparer"
    FEATURE_ANALYZER = "feature_analyzer"
    MODEL_SELECTOR = "model_selector"
    MODEL_TRAINER = "model_trainer"
    MODEL_DEPLOYER = "model_deployer"
    OBSERVABILITY_CONNECTOR = "observability_connector"
    # Tier 1: Coordination (2 agents)
    ORCHESTRATOR = "orchestrator"
    TOOL_COMPOSER = "tool_composer"
    # Tier 2: Causal Analytics (3 agents)
    CAUSAL_IMPACT = "causal_impact"
    GAP_ANALYZER = "gap_analyzer"
    HETEROGENEOUS_OPTIMIZER = "heterogeneous_optimizer"
    # Tier 3: Monitoring & Experimentation (3 agents)
    DRIFT_MONITOR = "drift_monitor"
    EXPERIMENT_DESIGNER = "experiment_designer"
    HEALTH_SCORE = "health_score"
    # Tier 4: ML Predictions (2 agents)
    PREDICTION_SYNTHESIZER = "prediction_synthesizer"
    RESOURCE_OPTIMIZER = "resource_optimizer"
    # Tier 5: Self-Improvement (2 agents)
    FEEDBACK_LEARNER = "feedback_learner"
    EXPLAINER = "explainer"
    # Legacy (kept for backwards compatibility)
    FAIRNESS_GUARDIAN = "fairness_guardian"


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
    agent_activity: Optional[Dict[str, Any]] = None  # Added in v1.3


@dataclass
class E2IEntityReferences:
    """Foreign key references to E2I data layer entities."""

    patient_journey_id: Optional[str] = None
    patient_id: Optional[str] = None  # Denormalized
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
    # E2I entity references
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
    """
    v1.3: Episodic memory with full E2I entity context attached.
    Returned by get_enriched_episodic_memory().
    """

    memory_id: str
    event_type: str
    event_subtype: Optional[str]
    description: str
    occurred_at: str
    outcome_type: Optional[str]
    agent_name: Optional[str]
    confidence_score: Optional[float]
    # Full entity context from 001b FK joins
    patient_context: Optional[Dict[str, Any]] = None
    hcp_context: Optional[Dict[str, Any]] = None
    trigger_context: Optional[Dict[str, Any]] = None
    causal_path_context: Optional[Dict[str, Any]] = None
    treatment_context: Optional[Dict[str, Any]] = None
    prediction_context: Optional[Dict[str, Any]] = None


@dataclass
class AgentActivityContext:
    """
    v1.3: Full context for an agent activity.
    Returned by get_agent_activity_with_context().
    """

    activity_id: str
    agent_name: str
    action_type: str
    started_at: str
    completed_at: Optional[str]
    status: str
    # Linked entities
    trigger: Optional[Dict[str, Any]] = None
    causal_paths: Optional[List[Dict[str, Any]]] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    # Performance metrics
    duration_ms: Optional[int] = None
    tokens_used: Optional[int] = None


# ============================================================================
# SERVICE FACTORIES
# ============================================================================


def get_embedding_service():
    """Get embedding service based on environment."""
    if ENVIRONMENT == "local_pilot":
        return OpenAIEmbeddingService()
    else:
        return BedrockEmbeddingService()


def get_llm_service():
    """Get LLM service based on environment."""
    if ENVIRONMENT == "local_pilot":
        return AnthropicLLMService()
    else:
        return BedrockLLMService()


def get_redis_client():
    """Get Redis client for working memory."""
    import redis.asyncio as redis

    url = os.environ.get("REDIS_URL", "redis://localhost:6382")
    return redis.from_url(url, decode_responses=True)


def get_supabase_client():
    """Get Supabase client for episodic/procedural memory."""
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

    return create_client(url, key)


def get_falkordb_client():
    """Get FalkorDB client for semantic memory."""
    from falkordb import FalkorDB

    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(
        os.environ.get("FALKORDB_PORT", "6379")
    )  # 6379 internal (docker), 6381 external (host)
    password = os.environ.get("FALKORDB_PASSWORD")

    return FalkorDB(host=host, port=port, password=password)


# ============================================================================
# EMBEDDING SERVICES
# ============================================================================


class OpenAIEmbeddingService:
    """OpenAI embeddings for local pilot."""

    def __init__(self):
        import openai

        self.client = openai.OpenAI()
        self.model = CONFIG["embeddings"]["local_pilot"]["model"]
        self._cache = {}

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[no-any-return]

        response = self.client.embeddings.create(model=self.model, input=text)

        embedding: List[float] = response.data[0].embedding
        self._cache[cache_key] = embedding

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]  # type: ignore[no-any-return]


class BedrockEmbeddingService:
    """AWS Bedrock embeddings for production."""

    def __init__(self):
        import boto3

        self.client = boto3.client("bedrock-runtime")
        self.model = CONFIG["embeddings"]["aws_production"]["model"]

    async def embed(self, text: str) -> List[float]:
        response = self.client.invoke_model(
            modelId=self.model, body=json.dumps({"inputText": text})
        )
        result = json.loads(response["body"].read())
        embedding: List[float] = result["embedding"]
        return embedding


# ============================================================================
# LLM SERVICES
# ============================================================================


class AnthropicLLMService:
    """Anthropic Claude for local pilot."""

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = CONFIG["llm"]["local_pilot"]["model"]
        self.max_tokens = CONFIG["llm"]["local_pilot"]["max_tokens"]
        self.temperature = CONFIG["llm"]["local_pilot"]["temperature"]

    async def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(response.content[0].text)


class BedrockLLMService:
    """AWS Bedrock Claude for production."""

    def __init__(self):
        import boto3

        self.client = boto3.client("bedrock-runtime")
        self.model = CONFIG["llm"]["aws_production"]["model"]
        self.max_tokens = CONFIG["llm"]["aws_production"]["max_tokens"]

    async def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens or self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ),
        )
        result = json.loads(response["body"].read())
        return str(result["content"][0]["text"])


# ============================================================================
# SHORT-TERM / WORKING MEMORY (Redis + LangGraph MemorySaver)
# ============================================================================


class RedisWorkingMemory:
    """
    Redis-based working memory with LangGraph MemorySaver integration.
    """

    def __init__(self):
        self.config = CONFIG["memory_backends"]["working"]["local_pilot"]
        self._client = None
        self._checkpointer = None

    async def get_client(self):
        """Lazy Redis client initialization."""
        if self._client is None:
            self._client = get_redis_client()
        return self._client

    def get_langgraph_checkpointer(self):
        """Get LangGraph checkpointer backed by Redis."""
        if self._checkpointer is None:
            from langgraph.checkpoint.redis import RedisSaver

            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6382")
            self._checkpointer = RedisSaver.from_conn_string(redis_url)
        return self._checkpointer

    async def create_session(
        self,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Create new working memory session."""
        redis = await self.get_client()
        session_id = session_id or str(uuid.uuid4())

        session_key = f"{self.config['session_prefix']}{session_id}"

        session_data = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity_at": datetime.now(timezone.utc).isoformat(),
            "message_count": "0",
            "current_phase": "init",
            "user_preferences": json.dumps(
                initial_context.get("preferences", {}) if initial_context else {}
            ),
            "active_filters": json.dumps(
                initial_context.get("filters", {}) if initial_context else {}
            ),
            # E2I context
            "active_brand": initial_context.get("brand") if initial_context else None,
            "active_region": initial_context.get("region") if initial_context else None,
        }

        await redis.hset(
            session_key, mapping={k: v for k, v in session_data.items() if v is not None}
        )
        await redis.expire(session_key, self.config["ttl_seconds"])

        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        redis = await self.get_client()
        session_key = f"{self.config['session_prefix']}{session_id}"

        data = await redis.hgetall(session_key)
        if not data:
            return None

        # Deserialize JSON fields
        if data.get("user_preferences"):
            data["user_preferences"] = json.loads(data["user_preferences"])
        if data.get("active_filters"):
            data["active_filters"] = json.loads(data["active_filters"])
        if data.get("message_count"):
            data["message_count"] = int(data["message_count"])

        return dict(data)

    async def update_session(self, session_id: str, updates: Dict[str, Any]):
        """Update session fields."""
        redis = await self.get_client()
        session_key = f"{self.config['session_prefix']}{session_id}"

        # Serialize complex fields
        for key in ["user_preferences", "active_filters", "active_entities"]:
            if key in updates and isinstance(updates[key], dict):
                updates[key] = json.dumps(updates[key])

        # Convert non-string values
        for key, value in updates.items():
            if isinstance(value, (int, float)):
                updates[key] = str(value)

        updates["last_activity_at"] = datetime.now(timezone.utc).isoformat()

        await redis.hset(session_key, mapping=updates)
        await redis.expire(session_key, self.config["ttl_seconds"])

    async def set_e2i_context(
        self,
        session_id: str,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        patient_ids: Optional[List[str]] = None,
        hcp_ids: Optional[List[str]] = None,
    ):
        """Set E2I entity context for the session."""
        updates = {}
        if brand:
            updates["active_brand"] = brand
        if region:
            updates["active_region"] = region
        if patient_ids:
            updates["active_patient_ids"] = json.dumps(patient_ids)
        if hcp_ids:
            updates["active_hcp_ids"] = json.dumps(hcp_ids)

        if updates:
            await self.update_session(session_id, updates)

    async def get_e2i_context(self, session_id: str) -> Dict[str, Any]:
        """Get E2I entity context from session."""
        session = await self.get_session(session_id)
        if not session:
            return {}

        context = {
            "brand": session.get("active_brand"),
            "region": session.get("active_region"),
        }

        if session.get("active_patient_ids"):
            context["patient_ids"] = json.loads(session["active_patient_ids"])
        if session.get("active_hcp_ids"):
            context["hcp_ids"] = json.loads(session["active_hcp_ids"])

        return context

    async def add_message(
        self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Add message to conversation history."""
        redis = await self.get_client()
        messages_key = f"{self.config['session_prefix']}{session_id}:messages"

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": json.dumps(metadata or {}),
        }

        await redis.rpush(messages_key, json.dumps(message))
        await redis.expire(messages_key, self.config["ttl_seconds"])

        max_messages = self.config["context_window_messages"]
        await redis.ltrim(messages_key, -max_messages, -1)

        session_key = f"{self.config['session_prefix']}{session_id}"
        await redis.hincrby(session_key, "message_count", 1)

    async def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recent messages from conversation."""
        redis = await self.get_client()
        messages_key = f"{self.config['session_prefix']}{session_id}:messages"

        if limit:
            messages = await redis.lrange(messages_key, -limit, -1)
        else:
            messages = await redis.lrange(messages_key, 0, -1)

        return [json.loads(m) for m in messages]

    async def append_evidence(self, session_id: str, evidence: Dict[str, Any]):
        """Append evidence item to evidence board."""
        redis = await self.get_client()
        evidence_key = f"{self.config['evidence_prefix']}{session_id}"

        await redis.rpush(evidence_key, json.dumps(evidence))
        await redis.expire(evidence_key, self.config["ttl_seconds"])

    async def get_evidence_trail(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all evidence from current investigation."""
        redis = await self.get_client()
        evidence_key = f"{self.config['evidence_prefix']}{session_id}"

        evidence = await redis.lrange(evidence_key, 0, -1)
        return [json.loads(e) for e in evidence]

    async def clear_evidence(self, session_id: str):
        """Clear evidence board."""
        redis = await self.get_client()
        evidence_key = f"{self.config['evidence_prefix']}{session_id}"
        await redis.delete(evidence_key)


# Global working memory instance
_working_memory: Optional[RedisWorkingMemory] = None


def get_working_memory() -> RedisWorkingMemory:
    """Get or create working memory instance."""
    global _working_memory
    if _working_memory is None:
        _working_memory = RedisWorkingMemory()
    return _working_memory


def get_langgraph_checkpointer():
    """Get the LangGraph checkpointer for workflow compilation."""
    working_memory = get_working_memory()
    return working_memory.get_langgraph_checkpointer()


# ============================================================================
# EPISODIC MEMORY (Supabase + pgvector) with E2I Integration
# ============================================================================


async def search_episodic_memory(
    embedding: List[float],
    filters: Optional[EpisodicSearchFilters] = None,
    limit: int = 10,
    min_similarity: float = 0.5,
    include_entity_context: bool = False,
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
        "filter_hcp_id": None,
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
            memory["e2i_context"] = context

    return memories


async def search_episodic_by_e2i_entity(
    entity_type: E2IEntityType,
    entity_id: str,
    limit: int = 20,
    event_types: Optional[List[str]] = None,
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
        E2IEntityType.AGENT_ACTIVITY: "agent_activity_id",
    }

    column = column_map.get(entity_type)
    if not column:
        raise ValueError(f"Unknown entity type: {entity_type}")

    query = (
        client.table("episodic_memories")
        .select("*")
        .eq(column, entity_id)
        .order("occurred_at", desc=True)
        .limit(limit)
    )

    if event_types:
        query = query.in_("event_type", event_types)

    result = query.execute()
    return result.data or []


async def insert_episodic_memory(
    memory: EpisodicMemoryInput,
    embedding: List[float],
    session_id: Optional[str] = None,
    cycle_id: Optional[str] = None,
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
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }

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

    # v1.3: Track memory statistics
    await _increment_memory_stats("episodic", memory.event_type)

    return memory_id


async def get_memory_entity_context(memory_id: str) -> E2IEntityContext:
    """
    Get linked E2I entity details for a memory.
    Uses the get_memory_entity_context database function from 001b.

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
                **row.get("entity_details", {}),
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
        logger.warning(f"get_memory_entity_context failed (001b may not be installed): {e}")
        return E2IEntityContext()


# ============================================================================
# v1.3 ENHANCED EPISODIC MEMORY FUNCTIONS
# These leverage the 001b foreign key functions for richer context
# ============================================================================


async def get_enriched_episodic_memory(memory_id: str) -> Optional[EnrichedEpisodicMemory]:
    """
    v1.3: Get episodic memory with full E2I data layer context attached.

    This function retrieves the memory and uses the get_memory_entity_context()
    RPC from 001b to attach patient, HCP, trigger, and causal path details.

    Args:
        memory_id: UUID of the episodic memory

    Returns:
        EnrichedEpisodicMemory with all linked entity context, or None if not found
    """
    client = get_supabase_client()

    # Fetch the base memory
    result = (
        client.table("episodic_memories").select("*").eq("memory_id", memory_id).single().execute()
    )

    if not result.data:
        return None

    memory = result.data

    # Fetch entity context using 001b function
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
        prediction_context=asdict(context)["prediction"] if context.prediction else None,
    )


async def get_agent_activity_with_context(activity_id: str) -> Optional[AgentActivityContext]:
    """
    v1.3: Get agent activity with full E2I context.
    Uses the get_agent_activity_context() function from 001b.

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
            tokens_used=data.get("tokens_used"),
        )
    except Exception as e:
        logger.warning(f"get_agent_activity_context failed (001b may not be installed): {e}")
        return None


async def get_causal_path_context(path_id: str) -> Optional[Dict[str, Any]]:
    """
    v1.3: Get full context for a causal path including linked memories.

    Args:
        path_id: ID of the causal path

    Returns:
        Dict with causal path details, discovery agent, and related memories
    """
    client = get_supabase_client()

    # Get causal path details
    path_result = client.table("causal_paths").select("*").eq("path_id", path_id).single().execute()

    if not path_result.data:
        return None

    path = path_result.data

    # Get related episodic memories
    memories = await search_episodic_by_e2i_entity(
        entity_type=E2IEntityType.CAUSAL_PATH, entity_id=path_id, limit=10
    )

    return {
        "path_id": path_id,
        "source_entity": path.get("source_entity"),
        "target_entity": path.get("target_entity"),
        "effect_size": path.get("effect_size"),
        "confidence": path.get("confidence"),
        "method_used": path.get("method_used"),
        "discovery_date": path.get("created_at"),
        "related_memories": memories,
    }


async def sync_treatment_relationships_to_cache() -> int:
    """
    v1.3: Sync HCP-Patient treatment relationships to semantic_memory_cache.
    Uses the sync_hcp_patient_relationships_to_cache() function from 001b.

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
        logger.warning(
            f"sync_hcp_patient_relationships_to_cache failed (001b may not be installed): {e}"
        )
        return 0


async def bulk_insert_episodic_memories(
    memories: List[Tuple[EpisodicMemoryInput, List[float]]],
    session_id: Optional[str] = None,
    cycle_id: Optional[str] = None,
) -> List[str]:
    """
    v1.3: Bulk insert multiple episodic memories for performance.

    Args:
        memories: List of (EpisodicMemoryInput, embedding) tuples
        session_id: Optional session ID for all memories
        cycle_id: Optional cycle ID for all memories

    Returns:
        List of inserted memory IDs
    """
    client = get_supabase_client()

    records = []
    memory_ids = []

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
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        }

        if memory.e2i_refs:
            refs = memory.e2i_refs
            if refs.patient_journey_id:
                record["patient_journey_id"] = refs.patient_journey_id
            if refs.hcp_id:
                record["hcp_id"] = refs.hcp_id
            if refs.trigger_id:
                record["trigger_id"] = refs.trigger_id
            if refs.brand:
                record["brand"] = refs.brand
            if refs.region:
                record["region"] = refs.region

        records.append(record)

    # Bulk insert
    client.table("episodic_memories").insert(records).execute()

    logger.info(f"Bulk inserted {len(records)} episodic memories")

    return memory_ids


async def get_recent_experiences(
    user_id: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    days_back: int = 7,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Get recent experiences filtered by E2I context.
    """
    client = get_supabase_client()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

    query = (
        client.table("episodic_memories")
        .select("*")
        .gte("occurred_at", cutoff)
        .order("occurred_at", desc=True)
        .limit(limit)
    )

    if event_types:
        query = query.in_("event_type", event_types)
    if brand:
        query = query.eq("brand", brand)
    if region:
        query = query.eq("region", region)

    result = query.execute()
    return result.data or []


async def get_patient_interaction_history(
    patient_journey_id: str, limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get all episodic memories related to a specific patient.
    Useful for building patient context in agent responses.
    """
    return await search_episodic_by_e2i_entity(
        entity_type=E2IEntityType.PATIENT, entity_id=patient_journey_id, limit=limit
    )


async def get_hcp_interaction_history(hcp_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get all episodic memories related to a specific HCP.
    Useful for building HCP context in agent responses.
    """
    return await search_episodic_by_e2i_entity(
        entity_type=E2IEntityType.HCP, entity_id=hcp_id, limit=limit
    )


async def get_trigger_feedback_history(trigger_id: str) -> List[Dict[str, Any]]:
    """
    Get all feedback and interactions related to a specific trigger.
    Useful for evaluating trigger effectiveness.
    """
    return await search_episodic_by_e2i_entity(
        entity_type=E2IEntityType.TRIGGER,
        entity_id=trigger_id,
        event_types=["feedback", "user_query"],
    )


# ============================================================================
# SEMANTIC MEMORY (FalkorDB + Graphity) with E2I Integration
# ============================================================================


class FalkorDBSemanticMemory:
    """
    FalkorDB-based semantic memory with Graphity integration and E2I entity support.
    """

    def __init__(self):
        self.config = CONFIG["memory_backends"]["semantic"]["local_pilot"]
        self._client = None
        self._graph = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_falkordb_client()
        return self._client

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.client.select_graph(self.config["graph_name"])
        return self._graph

    def add_e2i_entity(
        self,
        entity_type: E2IEntityType,
        entity_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add an E2I entity to the semantic graph.

        Args:
            entity_type: E2I entity type (patient, hcp, trigger, etc.)
            entity_id: ID from E2I data layer
            properties: Additional properties
        """
        # Map E2I types to graph node labels
        label_map = {
            E2IEntityType.PATIENT: "Patient",
            E2IEntityType.HCP: "HCP",
            E2IEntityType.TRIGGER: "Trigger",
            E2IEntityType.CAUSAL_PATH: "CausalPath",
            E2IEntityType.PREDICTION: "Prediction",
            E2IEntityType.TREATMENT: "Treatment",
            E2IEntityType.EXPERIMENT: "Experiment",
        }

        label = label_map.get(entity_type, "Entity")
        props = properties or {}
        props["e2i_entity_type"] = entity_type.value
        props["updated_at"] = datetime.now(timezone.utc).isoformat()

        prop_items = [f"{k}: ${k}" for k in props.keys()]
        prop_string = ", ".join(prop_items)

        query = f"""
        MERGE (e:{label} {{id: $entity_id}})
        ON CREATE SET e += {{{prop_string}}}
        ON MATCH SET e += {{{prop_string}}}
        RETURN e
        """

        params = {"entity_id": entity_id, **props}
        self.graph.query(query, params)

        return True

    def add_e2i_relationship(
        self,
        source_type: E2IEntityType,
        source_id: str,
        target_type: E2IEntityType,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a relationship between E2I entities.

        Common relationship types:
        - TREATED_BY: Patient → HCP
        - PRESCRIBED: Patient → Brand
        - PRESCRIBES: HCP → Brand
        - GENERATED: Trigger → from Prediction
        - CAUSES: CausalPath relationship
        - IMPACTS: CausalPath → KPI
        """
        # Ensure entities exist
        self.add_e2i_entity(source_type, source_id)
        self.add_e2i_entity(target_type, target_id)

        # Map types to labels
        label_map = {
            E2IEntityType.PATIENT: "Patient",
            E2IEntityType.HCP: "HCP",
            E2IEntityType.TRIGGER: "Trigger",
            E2IEntityType.CAUSAL_PATH: "CausalPath",
            E2IEntityType.PREDICTION: "Prediction",
        }

        source_label = label_map.get(source_type, "Entity")
        target_label = label_map.get(target_type, "Entity")

        props = properties or {}
        props["updated_at"] = datetime.now(timezone.utc).isoformat()

        prop_items = [f"{k}: ${k}" for k in props.keys()]
        prop_string = ", ".join(prop_items) if prop_items else ""

        query = f"""
        MATCH (s:{source_label} {{id: $source_id}})
        MATCH (t:{target_label} {{id: $target_id}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r += {{{prop_string}}}
        RETURN r
        """

        params = {"source_id": source_id, "target_id": target_id, **props}
        self.graph.query(query, params)

        return True

    def get_patient_network(self, patient_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Get the relationship network around a patient.
        Returns: HCPs, treatments, triggers associated with patient.
        """
        query = """
        MATCH (p:Patient {id: $patient_id})-[r*1..$max_depth]-(connected)
        RETURN p, r, connected
        """

        result = self.graph.query(query, {"patient_id": patient_id, "max_depth": max_depth})

        network: Dict[str, Any] = {
            "patient_id": patient_id,
            "hcps": [],
            "treatments": [],
            "triggers": [],
            "causal_paths": [],
        }

        for record in result.result_set:
            connected = record[2]
            labels = connected.labels if hasattr(connected, "labels") else []

            node_data = {
                "id": connected.properties.get("id"),
                "properties": dict(connected.properties),
            }

            if "HCP" in labels:
                network["hcps"].append(node_data)
            elif "Treatment" in labels:
                network["treatments"].append(node_data)
            elif "Trigger" in labels:
                network["triggers"].append(node_data)
            elif "CausalPath" in labels:
                network["causal_paths"].append(node_data)

        return network

    def get_hcp_influence_network(self, hcp_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Get the influence network around an HCP.
        Returns: Connected HCPs, patients, brands prescribed.
        """
        query = """
        MATCH (h:HCP {id: $hcp_id})-[r*1..$max_depth]-(connected)
        RETURN h, r, connected, type(r) as rel_type
        """

        result = self.graph.query(query, {"hcp_id": hcp_id, "max_depth": max_depth})

        network: Dict[str, Any] = {
            "hcp_id": hcp_id,
            "influenced_hcps": [],
            "patients": [],
            "brands_prescribed": [],
        }

        for record in result.result_set:
            connected = record[2]
            rel_type = record[3] if len(record) > 3 else None
            labels = connected.labels if hasattr(connected, "labels") else []

            node_data = {
                "id": connected.properties.get("id"),
                "relationship": rel_type,
                "properties": dict(connected.properties),
            }

            if "HCP" in labels:
                network["influenced_hcps"].append(node_data)
            elif "Patient" in labels:
                network["patients"].append(node_data)
            elif "Brand" in labels:
                network["brands_prescribed"].append(node_data)

        return network

    def traverse_causal_chain(
        self, start_entity_id: str, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Traverse causal relationships from a starting entity.
        """
        query = """
        MATCH path = (s {id: $start_id})-[:CAUSES|IMPACTS*1..$max_depth]->(t)
        RETURN
            [n IN nodes(path) | {id: n.id, type: labels(n)[0]}] as nodes,
            [r IN relationships(path) | {type: type(r), conf: r.confidence}] as rels
        """

        result = self.graph.query(query, {"start_id": start_entity_id, "max_depth": max_depth})

        chains = []
        for record in result.result_set:
            chains.append(
                {"nodes": record[0], "relationships": record[1], "path_length": len(record[1])}
            )

        return chains

    def find_causal_paths_for_kpi(
        self, kpi_name: str, min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find all causal paths that impact a specific KPI.
        Useful for understanding what drives KPI changes.
        """
        query = """
        MATCH (cp:CausalPath)-[r:IMPACTS]->(k:KPI {name: $kpi_name})
        WHERE r.confidence >= $min_confidence
        RETURN cp.id as path_id, cp.effect_size as effect_size,
               r.confidence as confidence, cp.method_used as method
        ORDER BY r.confidence DESC
        """

        result = self.graph.query(query, {"kpi_name": kpi_name, "min_confidence": min_confidence})

        return [
            {
                "path_id": record[0],
                "effect_size": record[1],
                "confidence": record[2],
                "method": record[3],
            }
            for record in result.result_set
        ]


# Global semantic memory instance
_semantic_memory: Optional[FalkorDBSemanticMemory] = None


def get_semantic_memory() -> FalkorDBSemanticMemory:
    """Get or create semantic memory instance."""
    global _semantic_memory
    if _semantic_memory is None:
        _semantic_memory = FalkorDBSemanticMemory()
    return _semantic_memory


async def query_semantic_graph(query: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Query the semantic graph (used by investigator node)."""
    semantic = get_semantic_memory()

    start_nodes = query.get("start_nodes", [])
    query.get("relationship_types")
    max_depth = query.get("max_depth", 2)

    results = []
    for node_id in start_nodes:
        # Check if this is an E2I entity query
        if query.get("entity_type") == "patient":
            network = semantic.get_patient_network(node_id, max_depth)
            results.append({"type": "patient_network", "data": network})
        elif query.get("entity_type") == "hcp":
            network = semantic.get_hcp_influence_network(node_id, max_depth)
            results.append({"type": "hcp_network", "data": network})
        elif query.get("follow_causal"):
            chains = semantic.traverse_causal_chain(node_id, max_depth)
            results.extend([{"type": "causal_chain", "data": chain} for chain in chains])

    return results


async def sync_to_semantic_graph(triplet: Dict[str, Any]) -> bool:
    """Add a triplet to the semantic graph."""
    semantic = get_semantic_memory()

    # Map subject/object to E2I types if applicable
    subject_type = triplet.get("subject_type", "Entity")
    object_type = triplet.get("object_type", "Entity")

    # Try to map to E2I entity types
    e2i_type_map = {
        "Patient": E2IEntityType.PATIENT,
        "HCP": E2IEntityType.HCP,
        "Trigger": E2IEntityType.TRIGGER,
        "CausalPath": E2IEntityType.CAUSAL_PATH,
    }

    source_e2i_type = e2i_type_map.get(subject_type)
    target_e2i_type = e2i_type_map.get(object_type)

    if source_e2i_type and target_e2i_type:
        return semantic.add_e2i_relationship(
            source_type=source_e2i_type,
            source_id=triplet["subject"],
            target_type=target_e2i_type,
            target_id=triplet["object"],
            rel_type=triplet["predicate"],
            properties={"confidence": triplet.get("confidence", 0.8)},
        )

    # Fall back to generic entity handling
    semantic.add_e2i_entity(E2IEntityType.PATIENT, triplet["subject"])  # Generic
    semantic.add_e2i_entity(E2IEntityType.PATIENT, triplet["object"])

    return True


async def sync_data_layer_to_semantic_cache():
    """
    Sync E2I data layer relationships to Supabase semantic cache.
    Calls the sync_hcp_patient_relationships_to_cache database function.
    """
    client = get_supabase_client()

    result = client.rpc("sync_hcp_patient_relationships_to_cache", {}).execute()

    return result.data


# ============================================================================
# GRAPHITY EXTRACTOR with E2I Context
# ============================================================================


class GraphityExtractor:
    """
    Graphity-style extraction with E2I entity awareness.
    """

    def __init__(self):
        self.config = CONFIG["memory_backends"]["semantic"]["local_pilot"]["graphity"]
        self.llm = get_llm_service()
        self.semantic_memory = get_semantic_memory()
        self.entity_types = self.config["entity_types"]
        self.relationship_types = self.config["relationship_types"]

    async def extract_and_store(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        known_e2i_entities: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships, linking to known E2I entities.

        Args:
            text: Text to extract from
            context: Optional context
            known_e2i_entities: Dict mapping entity names to E2I IDs
                Example: {"Dr. Smith": "HCP-123", "Patient A": "PAT-456"}
        """
        extraction_prompt = self._build_extraction_prompt(text, context, known_e2i_entities)
        response = await self.llm.complete(extraction_prompt)
        entities, relationships = self._parse_extraction(response)

        # Link to known E2I entities
        if known_e2i_entities:
            for entity in entities:
                entity_name = entity.get("properties", {}).get("name", entity.get("id"))
                if entity_name in known_e2i_entities:
                    entity["e2i_id"] = known_e2i_entities[entity_name]

        # Store in graph
        stored_entities = 0
        stored_relationships = 0

        for entity in entities:
            try:
                e2i_type = self._map_to_e2i_type(entity["type"])
                entity_id = entity.get("e2i_id", entity["id"])

                self.semantic_memory.add_e2i_entity(
                    entity_type=e2i_type,
                    entity_id=entity_id,
                    properties=entity.get("properties", {}),
                )
                stored_entities += 1
            except Exception as e:
                logger.error(f"Error storing entity: {e}")

        for rel in relationships:
            try:
                source_e2i_type = self._map_to_e2i_type(rel["subject_type"])
                target_e2i_type = self._map_to_e2i_type(rel["object_type"])

                self.semantic_memory.add_e2i_relationship(
                    source_type=source_e2i_type,
                    source_id=rel["subject"],
                    target_type=target_e2i_type,
                    target_id=rel["object"],
                    rel_type=rel["predicate"],
                    properties={"confidence": rel.get("confidence", 0.8)},
                )
                stored_relationships += 1
            except Exception as e:
                logger.error(f"Error storing relationship: {e}")

        return {
            "entities_extracted": len(entities),
            "entities_stored": stored_entities,
            "relationships_extracted": len(relationships),
            "relationships_stored": stored_relationships,
        }

    def _map_to_e2i_type(self, type_str: str) -> E2IEntityType:
        """Map extracted type string to E2IEntityType."""
        type_map = {
            "Patient": E2IEntityType.PATIENT,
            "HCP": E2IEntityType.HCP,
            "Trigger": E2IEntityType.TRIGGER,
            "CausalPath": E2IEntityType.CAUSAL_PATH,
            "Prediction": E2IEntityType.PREDICTION,
            "Treatment": E2IEntityType.TREATMENT,
            "Experiment": E2IEntityType.EXPERIMENT,
        }
        return type_map.get(type_str, E2IEntityType.PATIENT)  # Default

    def _build_extraction_prompt(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        known_entities: Optional[Dict[str, str]] = None,
    ) -> str:
        entity_types_str = ", ".join(self.entity_types)
        rel_types_str = ", ".join(self.relationship_types)

        known_str = ""
        if known_entities:
            known_str = f"\nKnown E2I entities to link: {json.dumps(known_entities)}"

        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context)}"

        return f"""Extract entities and relationships from the following text.
This is for the E2I Causal Analytics system.

Valid entity types: {entity_types_str}
Valid relationship types: {rel_types_str}
{known_str}
{context_str}

Text to analyze:
{text}

Return JSON:
{{
    "entities": [
        {{"id": "unique_id", "type": "EntityType", "properties": {{"name": "...", ...}}, "e2i_id": "optional_e2i_id"}}
    ],
    "relationships": [
        {{"subject": "entity_id", "subject_type": "Type", "predicate": "REL_TYPE", "object": "entity_id", "object_type": "Type", "confidence": 0.9}}
    ]
}}

JSON:"""

    def _parse_extraction(self, llm_response: str) -> Tuple[List[Dict], List[Dict]]:
        try:
            cleaned = llm_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]

            data = json.loads(cleaned)

            entities = [e for e in data.get("entities", []) if e.get("type") in self.entity_types]
            relationships = [
                r
                for r in data.get("relationships", [])
                if r.get("predicate") in self.relationship_types
            ]

            return entities, relationships
        except json.JSONDecodeError:
            return [], []


_graphity_extractor: Optional[GraphityExtractor] = None


def get_graphity_extractor() -> GraphityExtractor:
    """Get or create Graphity extractor instance."""
    global _graphity_extractor
    if _graphity_extractor is None:
        _graphity_extractor = GraphityExtractor()
    return _graphity_extractor


# ============================================================================
# PROCEDURAL MEMORY with E2I Context
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
    # E2I context
    applicable_brands: Optional[List[str]] = None
    applicable_regions: Optional[List[str]] = None
    applicable_agents: Optional[List[str]] = None


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

    return result.data or []


async def insert_procedural_memory(
    procedure: ProceduralMemoryInput, trigger_embedding: List[float]
) -> str:
    """
    Insert or update procedural memory with E2I context.
    """
    client = get_supabase_client()

    # Check for existing similar procedure
    existing = await find_relevant_procedures(
        trigger_embedding, procedure.procedure_type, limit=1, min_similarity=0.9
    )

    if existing:
        procedure_id: str = existing[0]["procedure_id"]

        client.table("procedural_memories").update(
            {
                "usage_count": existing[0].get("usage_count", 0) + 1,
                "success_count": existing[0].get("success_count", 0) + 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("procedure_id", procedure_id).execute()

        return procedure_id

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

    client.table("procedural_memories").insert(record).execute()

    # v1.3: Track memory statistics
    await _increment_memory_stats("procedural", procedure.procedure_type)

    return procedure_id


async def get_few_shot_examples(
    query_embedding: List[float],
    intent: Optional[str] = None,
    brand: Optional[str] = None,
    max_examples: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get few-shot examples for in-context learning with E2I context.
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
        tool_sequence = (
            json.loads(proc["tool_sequence"])
            if isinstance(proc["tool_sequence"], str)
            else proc["tool_sequence"]
        )

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

    return examples


# ============================================================================
# LEARNING SIGNALS with E2I Context
# ============================================================================


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


async def record_learning_signal(
    signal: LearningSignalInput, cycle_id: Optional[str] = None, session_id: Optional[str] = None
):
    """Record a learning signal with E2I context."""
    client = get_supabase_client()

    record = {
        "signal_id": str(uuid.uuid4()),
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


async def get_training_examples_for_agent(
    agent_name: str, brand: Optional[str] = None, min_score: float = 0.7, limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get high-quality training examples for a specific agent.
    Used for DSPy optimization.
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
    return result.data or []


async def get_feedback_summary_for_trigger(trigger_id: str) -> Dict[str, Any]:
    """
    Get aggregated feedback for a specific trigger.
    Useful for evaluating trigger effectiveness.
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


# ============================================================================
# v1.3: MEMORY STATISTICS TRACKING
# ============================================================================


async def _increment_memory_stats(memory_type: str, subtype: Optional[str] = None):
    """
    v1.3: Track memory usage statistics for monitoring.

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
    v1.3: Get memory usage statistics for monitoring.

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


# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================

__all__ = [
    # Entity types
    "E2IEntityType",
    "E2IBrand",
    "E2IRegion",
    "E2IAgentName",
    # Data classes
    "E2IEntityContext",
    "E2IEntityReferences",
    "EpisodicMemoryInput",
    "EpisodicSearchFilters",
    "EnrichedEpisodicMemory",
    "AgentActivityContext",
    "ProceduralMemoryInput",
    "LearningSignalInput",
    # Service factories
    "get_embedding_service",
    "get_llm_service",
    "get_supabase_client",
    "get_falkordb_client",
    "get_working_memory",
    "get_langgraph_checkpointer",
    "get_semantic_memory",
    "get_graphity_extractor",
    # Episodic memory
    "search_episodic_memory",
    "search_episodic_by_e2i_entity",
    "insert_episodic_memory",
    "get_memory_entity_context",
    "get_enriched_episodic_memory",  # v1.3
    "get_recent_experiences",
    "get_patient_interaction_history",
    "get_hcp_interaction_history",
    "get_trigger_feedback_history",
    "bulk_insert_episodic_memories",  # v1.3
    # Semantic memory
    "query_semantic_graph",
    "sync_to_semantic_graph",
    "sync_data_layer_to_semantic_cache",
    "sync_treatment_relationships_to_cache",  # v1.3
    # Agent activity (v1.3)
    "get_agent_activity_with_context",
    "get_causal_path_context",
    # Procedural memory
    "find_relevant_procedures",
    "insert_procedural_memory",
    "get_few_shot_examples",
    # Learning signals
    "record_learning_signal",
    "get_training_examples_for_agent",
    "get_feedback_summary_for_trigger",
    # Statistics (v1.3)
    "get_memory_statistics",
]
