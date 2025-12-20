"""
E2I Agentic Memory - Backend Implementations
Storage and retrieval operations for all memory types

Architecture:
┌─────────────────┬─────────────────────────────────────┬─────────────────────────────┐
│ Memory Type     │ Function                            │ Technology                  │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────┤
│ Short-Term      │ Current context, scratchpad,        │ Redis + LangGraph           │
│ (Working)       │ immediate message history           │ MemorySaver checkpointer    │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────┤
│ Episodic        │ Experiences: "What did I do?"       │ Supabase (Postgres +        │
│ (Long-Term)     │ "What happened yesterday?"          │ pgvector)                   │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────┤
│ Semantic        │ Facts: "What is the relationship    │ FalkorDB + Graphity         │
│ (Long-Term)     │ between Project X and User Y?"      │ Extractor node updates      │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────┤
│ Procedural      │ Skills: "How did I solve this       │ Supabase (vector store of   │
│ (Long-Term)     │ error last time?"                   │ tool call sequences)        │
└─────────────────┴─────────────────────────────────────┴─────────────────────────────┘

Version: 1.1
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import asyncio

import yaml
from pydantic import BaseModel

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config() -> Dict[str, Any]:
    """Load memory configuration from YAML."""
    config_path = Path(__file__).parent / "005_memory_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


CONFIG = load_config()
ENVIRONMENT = CONFIG.get("environment", "local_pilot")


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
    
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return redis.from_url(url, decode_responses=True)


def get_supabase_client():
    """Get Supabase client for episodic/procedural memory."""
    from supabase import create_client, Client
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
    
    return create_client(url, key)


def get_falkordb_client():
    """Get FalkorDB client for semantic memory."""
    from falkordb import FalkorDB
    
    # FalkorDB uses Redis protocol
    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6379"))
    
    return FalkorDB(host=host, port=port)


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
            return self._cache[cache_key]
        
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        
        embedding = response.data[0].embedding
        self._cache[cache_key] = embedding
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]


class BedrockEmbeddingService:
    """AWS Bedrock embeddings for production."""
    
    def __init__(self):
        import boto3
        self.client = boto3.client("bedrock-runtime")
        self.model = CONFIG["embeddings"]["aws_production"]["model"]
    
    async def embed(self, text: str) -> List[float]:
        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps({"inputText": text})
        )
        result = json.loads(response["body"].read())
        return result["embedding"]


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
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


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
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]


# ============================================================================
# SHORT-TERM / WORKING MEMORY (Redis + LangGraph MemorySaver)
# Holds current context, scratchpad, and immediate message history
# ============================================================================

class RedisWorkingMemory:
    """
    Redis-based working memory with LangGraph MemorySaver integration.
    
    Stores:
    - Session state and metadata
    - Conversation context (last N messages)
    - Evidence board (accumulated during investigation)
    - Scratchpad for intermediate computations
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
        """
        Get LangGraph checkpointer backed by Redis.
        Use this when compiling LangGraph workflows.
        
        Usage:
            from langgraph.graph import StateGraph
            from memory_backends import get_langgraph_checkpointer
            
            workflow = StateGraph(...)
            app = workflow.compile(checkpointer=get_langgraph_checkpointer())
        """
        if self._checkpointer is None:
            # LangGraph Redis checkpointer
            # Note: Requires langgraph-checkpoint-redis package
            from langgraph.checkpoint.redis import RedisSaver
            
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            self._checkpointer = RedisSaver.from_conn_string(redis_url)
        
        return self._checkpointer
    
    # --- Session Management ---
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new working memory session."""
        redis = await self.get_client()
        session_id = str(uuid.uuid4())
        
        session_key = f"{self.config['session_prefix']}{session_id}"
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "created_at": datetime.utcnow().isoformat(),
            "last_activity_at": datetime.utcnow().isoformat(),
            "message_count": "0",
            "current_phase": "init",
            "user_preferences": json.dumps(initial_context.get("preferences", {}) if initial_context else {}),
            "active_filters": json.dumps(initial_context.get("filters", {}) if initial_context else {})
        }
        
        await redis.hset(session_key, mapping=session_data)
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
        
        return data
    
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
        
        updates["last_activity_at"] = datetime.utcnow().isoformat()
        
        await redis.hset(session_key, mapping=updates)
        await redis.expire(session_key, self.config["ttl_seconds"])
    
    async def end_session(self, session_id: str):
        """End session (mark as ended, will expire via TTL)."""
        await self.update_session(session_id, {"ended_at": datetime.utcnow().isoformat()})
    
    # --- Conversation Context (In-Context Prompting) ---
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add message to conversation history."""
        redis = await self.get_client()
        messages_key = f"{self.config['session_prefix']}{session_id}:messages"
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": json.dumps(metadata or {})
        }
        
        await redis.rpush(messages_key, json.dumps(message))
        await redis.expire(messages_key, self.config["ttl_seconds"])
        
        # Trim to keep only last N messages (context window)
        max_messages = self.config["context_window_messages"]
        await redis.ltrim(messages_key, -max_messages, -1)
        
        # Update message count
        session_key = f"{self.config['session_prefix']}{session_id}"
        await redis.hincrby(session_key, "message_count", 1)
    
    async def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent messages from conversation."""
        redis = await self.get_client()
        messages_key = f"{self.config['session_prefix']}{session_id}:messages"
        
        if limit:
            messages = await redis.lrange(messages_key, -limit, -1)
        else:
            messages = await redis.lrange(messages_key, 0, -1)
        
        return [json.loads(m) for m in messages]
    
    async def set_conversation_summary(self, session_id: str, summary: str):
        """Store compressed conversation summary."""
        redis = await self.get_client()
        key = f"{self.config['session_prefix']}{session_id}:summary"
        
        await redis.set(key, summary)
        await redis.expire(key, self.config["ttl_seconds"])
    
    async def get_conversation_summary(self, session_id: str) -> Optional[str]:
        """Get conversation summary."""
        redis = await self.get_client()
        key = f"{self.config['session_prefix']}{session_id}:summary"
        
        return await redis.get(key)
    
    # --- Evidence Board ---
    
    async def append_evidence(
        self,
        session_id: str,
        evidence: Dict[str, Any]
    ):
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
        """Clear evidence board for new investigation."""
        redis = await self.get_client()
        evidence_key = f"{self.config['evidence_prefix']}{session_id}"
        
        await redis.delete(evidence_key)
    
    # --- Scratchpad ---
    
    async def set_scratchpad(self, session_id: str, key: str, value: Any):
        """Store value in session scratchpad."""
        redis = await self.get_client()
        scratchpad_key = f"{self.config['session_prefix']}{session_id}{self.config['scratchpad_key_suffix']}:{key}"
        
        await redis.set(scratchpad_key, json.dumps(value))
        await redis.expire(scratchpad_key, self.config["ttl_seconds"])
    
    async def get_scratchpad(self, session_id: str, key: str) -> Any:
        """Get value from session scratchpad."""
        redis = await self.get_client()
        scratchpad_key = f"{self.config['session_prefix']}{session_id}{self.config['scratchpad_key_suffix']}:{key}"
        
        value = await redis.get(scratchpad_key)
        return json.loads(value) if value else None


# Global working memory instance
_working_memory: Optional[RedisWorkingMemory] = None


def get_working_memory() -> RedisWorkingMemory:
    """Get or create working memory instance."""
    global _working_memory
    if _working_memory is None:
        _working_memory = RedisWorkingMemory()
    return _working_memory


def get_langgraph_checkpointer():
    """
    Get the LangGraph checkpointer for workflow compilation.
    
    Usage:
        from memory_backends import get_langgraph_checkpointer
        
        workflow = create_cognitive_workflow()
        app = workflow.compile(checkpointer=get_langgraph_checkpointer())
    """
    working_memory = get_working_memory()
    return working_memory.get_langgraph_checkpointer()


# ============================================================================
# EPISODIC MEMORY (Supabase + pgvector)
# Stores experiences: "What did I do?" "What happened yesterday?"
# ============================================================================

async def search_episodic_memory(
    embedding: List[float],
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    min_similarity: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Search episodic memories by embedding similarity.
    
    Args:
        embedding: Query embedding vector
        filters: Optional filters (event_type, entities, date_range)
        limit: Maximum results to return
        min_similarity: Minimum cosine similarity threshold
    
    Returns:
        List of matching episodic memories with similarity scores
    """
    client = get_supabase_client()
    
    result = client.rpc(
        "search_episodic_memory",
        {
            "query_embedding": embedding,
            "match_threshold": min_similarity,
            "match_count": limit,
            "filter_event_type": filters.get("event_type") if filters else None,
            "filter_agent": filters.get("agent") if filters else None
        }
    ).execute()
    
    return result.data or []


async def insert_episodic_memory(
    memory: Dict[str, Any],
    embedding: List[float],
    session_id: Optional[str] = None
) -> str:
    """
    Insert new episodic memory (User/AI interaction trace).
    
    Args:
        memory: Memory data (event_type, description, entities, etc.)
        embedding: Pre-computed embedding vector
        session_id: Optional session ID for linking
    
    Returns:
        ID of inserted memory
    """
    client = get_supabase_client()
    
    memory_id = str(uuid.uuid4())
    
    record = {
        "memory_id": memory_id,
        "session_id": session_id,
        "event_type": memory["event_type"],
        "event_subtype": memory.get("event_subtype"),
        "description": memory["description"],
        "raw_content": json.dumps(memory.get("raw_content", {})),
        "entities": json.dumps(memory.get("entities", {})),
        "outcome_type": memory.get("outcome_type"),
        "outcome_details": json.dumps(memory.get("outcome_details", {})),
        "user_satisfaction_score": memory.get("user_satisfaction_score"),
        "agent_name": memory.get("agent_name"),
        "embedding": embedding,
        "occurred_at": datetime.utcnow().isoformat()
    }
    
    client.table("episodic_memories").insert(record).execute()
    
    return memory_id


async def get_recent_experiences(
    user_id: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    days_back: int = 7,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Get recent experiences (for "What happened yesterday?" queries).
    """
    client = get_supabase_client()
    
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    
    query = client.table("episodic_memories") \
        .select("*") \
        .gte("occurred_at", cutoff) \
        .order("occurred_at", desc=True) \
        .limit(limit)
    
    if event_types:
        query = query.in_("event_type", event_types)
    
    result = query.execute()
    return result.data or []


# ============================================================================
# SEMANTIC MEMORY (FalkorDB + Graphity)
# Stores facts: "What is the relationship between Project X and User Y?"
# ============================================================================

class FalkorDBSemanticMemory:
    """
    FalkorDB-based semantic memory with Graphity integration.
    
    FalkorDB is a graph database that uses Redis protocol.
    Graphity provides automatic entity/relationship extraction from text.
    """
    
    def __init__(self):
        self.config = CONFIG["memory_backends"]["semantic"]["local_pilot"]
        self._client = None
        self._graph = None
    
    @property
    def client(self):
        """Lazy FalkorDB client initialization."""
        if self._client is None:
            self._client = get_falkordb_client()
        return self._client
    
    @property
    def graph(self):
        """Get or create the semantic graph."""
        if self._graph is None:
            self._graph = self.client.select_graph(self.config["graph_name"])
        return self._graph
    
    # --- Core Graph Operations ---
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add or update an entity (node) in the graph.
        
        Example entity types: Patient, HCP, Brand, Region, KPI, CausalPath
        """
        props = properties or {}
        props["updated_at"] = datetime.utcnow().isoformat()
        
        # Build property string for Cypher
        prop_items = [f"{k}: ${k}" for k in props.keys()]
        prop_string = ", ".join(prop_items)
        
        query = f"""
        MERGE (e:{entity_type} {{id: $entity_id}})
        ON CREATE SET e += {{{prop_string}}}
        ON MATCH SET e += {{{prop_string}}}
        RETURN e
        """
        
        params = {"entity_id": entity_id, **props}
        self.graph.query(query, params)
        
        return True
    
    def add_relationship(
        self,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add or update a relationship (edge) between entities.
        
        Example rel_types: TREATED_BY, PRESCRIBED, CAUSES, IMPACTS, INFLUENCES
        """
        props = properties or {}
        props["updated_at"] = datetime.utcnow().isoformat()
        
        prop_items = [f"{k}: ${k}" for k in props.keys()]
        prop_string = ", ".join(prop_items) if prop_items else ""
        
        query = f"""
        MATCH (s:{source_type} {{id: $source_id}})
        MATCH (t:{target_type} {{id: $target_id}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r += {{{prop_string}}}
        RETURN r
        """
        
        params = {
            "source_id": source_id,
            "target_id": target_id,
            **props
        }
        
        self.graph.query(query, params)
        return True
    
    def add_triplet(
        self,
        subject: str,
        subject_type: str,
        predicate: str,
        obj: str,
        object_type: str,
        confidence: float = 1.0
    ) -> bool:
        """
        Add a fact as a triplet (Subject -[Predicate]-> Object).
        
        This is the primary interface for the Graphity extractor.
        """
        # Ensure both entities exist
        self.add_entity(subject, subject_type)
        self.add_entity(obj, object_type)
        
        # Add relationship
        self.add_relationship(
            source_id=subject,
            source_type=subject_type,
            target_id=obj,
            target_type=object_type,
            rel_type=predicate,
            properties={"confidence": confidence}
        )
        
        return True
    
    # --- Query Operations ---
    
    def get_neighbors(
        self,
        entity_id: str,
        entity_type: Optional[str] = None,
        rel_types: Optional[List[str]] = None,
        direction: str = "both",  # "out", "in", "both"
        max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring entities via specified relationships.
        """
        # Build relationship pattern
        if rel_types:
            rel_pattern = "|".join(rel_types)
            rel_clause = f"[r:{rel_pattern}*1..{max_depth}]"
        else:
            rel_clause = f"[r*1..{max_depth}]"
        
        # Build direction pattern
        if direction == "out":
            pattern = f"-{rel_clause}->"
        elif direction == "in":
            pattern = f"<-{rel_clause}-"
        else:
            pattern = f"-{rel_clause}-"
        
        # Build entity match
        if entity_type:
            entity_match = f"(s:{entity_type} {{id: $entity_id}})"
        else:
            entity_match = "(s {id: $entity_id})"
        
        query = f"""
        MATCH {entity_match}{pattern}(t)
        RETURN t, type(r) as rel_type, r
        """
        
        result = self.graph.query(query, {"entity_id": entity_id})
        
        neighbors = []
        for record in result.result_set:
            node = record[0]
            neighbors.append({
                "entity_id": node.properties.get("id"),
                "entity_type": node.labels[0] if node.labels else "Unknown",
                "properties": dict(node.properties),
                "relationship_type": record[1],
                "relationship_properties": dict(record[2].properties) if record[2] else {}
            })
        
        return neighbors
    
    def traverse_causal_chain(
        self,
        start_entity_id: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Traverse causal relationships from a starting entity.
        Specifically follows CAUSES and IMPACTS relationships.
        """
        query = """
        MATCH path = (s {id: $start_id})-[:CAUSES|IMPACTS*1..$max_depth]->(t)
        RETURN path, 
               [n IN nodes(path) | {id: n.id, type: labels(n)[0]}] as nodes,
               [r IN relationships(path) | {type: type(r), conf: r.confidence}] as rels
        """
        
        result = self.graph.query(query, {"start_id": start_entity_id, "max_depth": max_depth})
        
        chains = []
        for record in result.result_set:
            chains.append({
                "nodes": record[1],
                "relationships": record[2],
                "path_length": len(record[2])
            })
        
        return chains
    
    def query_triplets(
        self,
        subject_type: Optional[str] = None,
        predicate: Optional[str] = None,
        object_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query triplets matching a pattern.
        """
        # Build match clause dynamically
        subject_match = f"(s:{subject_type})" if subject_type else "(s)"
        object_match = f"(o:{object_type})" if object_type else "(o)"
        rel_match = f"[r:{predicate}]" if predicate else "[r]"
        
        query = f"""
        MATCH {subject_match}-{rel_match}->{object_match}
        RETURN s.id as subject, labels(s)[0] as subject_type,
               type(r) as predicate, r.confidence as confidence,
               o.id as object, labels(o)[0] as object_type
        LIMIT $limit
        """
        
        result = self.graph.query(query, {"limit": limit})
        
        triplets = []
        for record in result.result_set:
            triplets.append({
                "subject": record[0],
                "subject_type": record[1],
                "predicate": record[2],
                "confidence": record[3] or 1.0,
                "object": record[4],
                "object_type": record[5]
            })
        
        return triplets
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two entities.
        """
        query = """
        MATCH path = shortestPath((s {id: $source})-[*1..$max_depth]-(t {id: $target}))
        RETURN [n IN nodes(path) | {id: n.id, type: labels(n)[0]}] as nodes,
               [r IN relationships(path) | {type: type(r)}] as rels
        """
        
        result = self.graph.query(query, {
            "source": source_id,
            "target": target_id,
            "max_depth": max_depth
        })
        
        if result.result_set:
            record = result.result_set[0]
            return {
                "nodes": record[0],
                "relationships": record[1]
            }
        
        return None


# ============================================================================
# GRAPHITY EXTRACTOR
# Automatic entity and relationship extraction using LLM
# ============================================================================

class GraphityExtractor:
    """
    Graphity-style automatic extraction of entities and relationships.
    
    As interactions happen, this extractor node updates the semantic graph
    with discovered facts.
    """
    
    def __init__(self):
        self.config = CONFIG["memory_backends"]["semantic"]["local_pilot"]["graphity"]
        self.llm = get_llm_service()
        self.semantic_memory = get_semantic_memory()
        
        # Valid types from config
        self.entity_types = self.config["entity_types"]
        self.relationship_types = self.config["relationship_types"]
    
    async def extract_and_store(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from text, store in graph.
        
        Args:
            text: Text to extract from (e.g., user query, agent response)
            context: Optional context (entities already known, etc.)
        
        Returns:
            Extraction results with counts
        """
        # Build extraction prompt
        extraction_prompt = self._build_extraction_prompt(text, context)
        
        # Call LLM for extraction
        response = await self.llm.complete(extraction_prompt)
        
        # Parse extraction results
        entities, relationships = self._parse_extraction(response)
        
        # Store in graph
        stored_entities = 0
        stored_relationships = 0
        
        for entity in entities:
            try:
                self.semantic_memory.add_entity(
                    entity_id=entity["id"],
                    entity_type=entity["type"],
                    properties=entity.get("properties", {})
                )
                stored_entities += 1
            except Exception as e:
                print(f"Error storing entity: {e}")
        
        for rel in relationships:
            try:
                self.semantic_memory.add_triplet(
                    subject=rel["subject"],
                    subject_type=rel["subject_type"],
                    predicate=rel["predicate"],
                    obj=rel["object"],
                    object_type=rel["object_type"],
                    confidence=rel.get("confidence", 0.8)
                )
                stored_relationships += 1
            except Exception as e:
                print(f"Error storing relationship: {e}")
        
        return {
            "entities_extracted": len(entities),
            "entities_stored": stored_entities,
            "relationships_extracted": len(relationships),
            "relationships_stored": stored_relationships
        }
    
    def _build_extraction_prompt(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for entity/relationship extraction."""
        
        entity_types_str = ", ".join(self.entity_types)
        rel_types_str = ", ".join(self.relationship_types)
        
        context_str = ""
        if context:
            context_str = f"\nKnown context: {json.dumps(context)}"
        
        return f"""Extract entities and relationships from the following text.

Valid entity types: {entity_types_str}
Valid relationship types: {rel_types_str}
{context_str}

Text to analyze:
{text}

Return your extraction as JSON with this structure:
{{
    "entities": [
        {{"id": "unique_id", "type": "EntityType", "properties": {{"name": "...", ...}}}}
    ],
    "relationships": [
        {{"subject": "entity_id", "subject_type": "Type", "predicate": "REL_TYPE", "object": "entity_id", "object_type": "Type", "confidence": 0.9}}
    ]
}}

Only extract facts that are clearly stated or strongly implied. Use confidence < 1.0 for inferred relationships.
If no entities or relationships can be extracted, return empty arrays.

JSON:"""
    
    def _parse_extraction(
        self,
        llm_response: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Parse LLM extraction response."""
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = llm_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            
            data = json.loads(cleaned)
            
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            
            # Validate entity types
            entities = [e for e in entities if e.get("type") in self.entity_types]
            
            # Validate relationship types
            relationships = [r for r in relationships if r.get("predicate") in self.relationship_types]
            
            return entities, relationships
            
        except json.JSONDecodeError:
            return [], []


# Global semantic memory instance
_semantic_memory: Optional[FalkorDBSemanticMemory] = None
_graphity_extractor: Optional[GraphityExtractor] = None


def get_semantic_memory() -> FalkorDBSemanticMemory:
    """Get or create semantic memory instance."""
    global _semantic_memory
    if _semantic_memory is None:
        _semantic_memory = FalkorDBSemanticMemory()
    return _semantic_memory


def get_graphity_extractor() -> GraphityExtractor:
    """Get or create Graphity extractor instance."""
    global _graphity_extractor
    if _graphity_extractor is None:
        _graphity_extractor = GraphityExtractor()
    return _graphity_extractor


# Convenience function for cognitive workflow
async def query_semantic_graph(query: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Query the semantic graph (used by investigator node).
    
    Args:
        query: Query specification with:
            - start_nodes: List of entity IDs to start from
            - relationship_types: Types of relationships to traverse
            - max_depth: Maximum traversal depth
    """
    semantic = get_semantic_memory()
    
    start_nodes = query.get("start_nodes", [])
    rel_types = query.get("relationship_types")
    max_depth = query.get("max_depth", 2)
    
    results = []
    for node_id in start_nodes:
        neighbors = semantic.get_neighbors(
            entity_id=node_id,
            rel_types=rel_types,
            max_depth=max_depth
        )
        
        for neighbor in neighbors:
            results.append({
                "subject": node_id,
                "predicate": neighbor["relationship_type"],
                "object": neighbor["entity_id"],
                "confidence": neighbor["relationship_properties"].get("confidence", 0.5)
            })
    
    return results


async def sync_to_semantic_graph(triplet: Dict[str, Any]) -> bool:
    """
    Add a triplet to the semantic graph (used by reflector node).
    """
    semantic = get_semantic_memory()
    
    return semantic.add_triplet(
        subject=triplet["subject"],
        subject_type=triplet.get("subject_type", "Entity"),
        predicate=triplet["predicate"],
        obj=triplet["object"],
        object_type=triplet.get("object_type", "Entity"),
        confidence=triplet.get("confidence", 0.8)
    )


# ============================================================================
# PROCEDURAL MEMORY (Supabase)
# Stores skills: "How did I solve this error last time?"
# ============================================================================

async def find_relevant_procedures(
    embedding: List[float],
    procedure_type: Optional[str] = None,
    limit: int = 5,
    min_similarity: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Find relevant procedures (few-shot examples) by query similarity.
    
    This answers: "How did I solve this type of problem before?"
    
    Args:
        embedding: Query embedding vector
        procedure_type: Optional filter (tool_sequence, query_pattern, error_recovery)
        limit: Maximum results
        min_similarity: Minimum similarity threshold
    
    Returns:
        List of matching procedures with success rates for few-shot prompting
    """
    client = get_supabase_client()
    
    result = client.rpc(
        "find_relevant_procedures",
        {
            "query_embedding": embedding,
            "match_threshold": min_similarity,
            "match_count": limit
        }
    ).execute()
    
    procedures = result.data or []
    
    # Filter by type if specified
    if procedure_type:
        procedures = [p for p in procedures if p.get("procedure_type") == procedure_type]
    
    # Sort by success rate (for few-shot, we want the best examples)
    procedures.sort(key=lambda p: p.get("success_rate", 0), reverse=True)
    
    return procedures


async def insert_procedural_memory(
    procedure: Dict[str, Any],
    trigger_embedding: List[float]
) -> str:
    """
    Insert or update procedural memory (successful tool call sequence).
    
    Args:
        procedure: Procedure data with:
            - procedure_name: Human-readable name
            - procedure_type: tool_sequence, query_pattern, error_recovery
            - tool_sequence: Ordered list of tool calls
            - intent_keywords: Keywords that suggest this procedure
        trigger_embedding: Embedding for matching similar queries
    
    Returns:
        ID of inserted/updated procedure
    """
    client = get_supabase_client()
    
    # Check if similar procedure exists (update if so)
    existing = await find_relevant_procedures(
        trigger_embedding,
        procedure.get("procedure_type"),
        limit=1,
        min_similarity=0.9  # High similarity = same procedure
    )
    
    if existing:
        # Update existing procedure's success count
        procedure_id = existing[0]["procedure_id"]
        
        client.table("procedural_memories") \
            .update({
                "usage_count": existing[0].get("usage_count", 0) + 1,
                "success_count": existing[0].get("success_count", 0) + procedure.get("success_count", 1),
                "updated_at": datetime.utcnow().isoformat()
            }) \
            .eq("procedure_id", procedure_id) \
            .execute()
        
        return procedure_id
    
    # Insert new procedure
    procedure_id = str(uuid.uuid4())
    
    record = {
        "procedure_id": procedure_id,
        "procedure_name": procedure["procedure_name"],
        "procedure_type": procedure.get("procedure_type", "tool_sequence"),
        "tool_sequence": json.dumps(procedure["tool_sequence"]),
        "trigger_pattern": procedure.get("trigger_pattern"),
        "trigger_embedding": trigger_embedding,
        "intent_keywords": procedure.get("intent_keywords", []),
        "usage_count": 1,
        "success_count": procedure.get("success_count", 1),
        "is_active": True,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    client.table("procedural_memories").insert(record).execute()
    
    return procedure_id


async def get_few_shot_examples(
    query_embedding: List[float],
    max_examples: int = 5
) -> List[Dict[str, Any]]:
    """
    Get few-shot examples for in-context learning.
    
    Returns successful tool sequences formatted for prompting.
    """
    config = CONFIG["memory_backends"]["procedural"]["local_pilot"]["few_shot"]
    
    procedures = await find_relevant_procedures(
        embedding=query_embedding,
        limit=config.get("max_examples", max_examples),
        min_similarity=config.get("min_similarity", 0.7)
    )
    
    # Format as few-shot examples
    examples = []
    for proc in procedures:
        tool_sequence = json.loads(proc["tool_sequence"]) if isinstance(proc["tool_sequence"], str) else proc["tool_sequence"]
        
        examples.append({
            "trigger": proc.get("trigger_pattern", ""),
            "solution": tool_sequence,
            "success_rate": proc.get("success_rate", 0)
        })
    
    return examples


# ============================================================================
# LEARNING SIGNAL OPERATIONS (for DSPy optimization)
# ============================================================================

async def record_learning_signal(
    signal: Dict[str, Any],
    cycle_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Record a learning signal for DSPy optimization."""
    client = get_supabase_client()
    
    record = {
        "signal_id": str(uuid.uuid4()),
        "cycle_id": cycle_id,
        "session_id": session_id,
        "signal_type": signal["signal_type"],
        "signal_value": signal.get("signal_value"),
        "signal_details": json.dumps(signal.get("signal_details", {})),
        "applies_to_type": signal.get("applies_to_type"),
        "applies_to_id": signal.get("applies_to_id"),
        "is_training_example": signal.get("is_training_example", False),
        "dspy_metric_name": signal.get("dspy_metric_name"),
        "dspy_metric_value": signal.get("dspy_metric_value"),
        "created_at": datetime.utcnow().isoformat()
    }
    
    client.table("learning_signals").insert(record).execute()
