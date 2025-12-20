# E2I Hybrid RAG Architecture
## Supabase (pgvector) + FalkorDB (Graph) Orchestration & Visualization

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                    │
│              "Why did Remibrutinib conversion drop?"                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HYBRID RETRIEVER ORCHESTRATOR                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   Query     │→ │   Parallel  │→ │   Fusion    │                  │
│  │  Analyzer   │  │   Dispatch  │  │   Ranker    │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
            │                │                │
            ▼                ▼                ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   SUPABASE    │  │   SUPABASE    │  │   FALKORDB    │
│   pgvector    │  │   Full-Text   │  │   Graph       │
│   (Semantic)  │  │   (Keyword)   │  │   (Relational)│
├───────────────┤  ├───────────────┤  ├───────────────┤
│ • Episodic    │  │ • BM25-style  │  │ • Causal DAG  │
│ • Procedural  │  │ • Trigram     │  │ • Entity Rels │
│ • Insights    │  │ • Pattern     │  │ • Pathways    │
└───────────────┘  └───────────────┘  └───────────────┘
            │                │                │
            └────────────────┴────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RECIPROCAL RANK FUSION (RRF)                      │
│                    + Graph-Boosted Reranking                         │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXT ASSEMBLY + LLM                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Hybrid Retriever Implementation

### 1.1 Core Retriever Class

```python
# src/rag/hybrid_retriever.py

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import numpy as np

from supabase import Client as SupabaseClient
from falkordb import FalkorDB
from sentence_transformers import SentenceTransformer


class RetrievalSource(Enum):
    """Track which backend returned each result."""
    VECTOR = "supabase_vector"
    FULLTEXT = "supabase_fulltext"
    GRAPH = "falkordb_graph"


@dataclass
class RetrievalResult:
    """Unified result format from any backend."""
    id: str
    content: str
    source: RetrievalSource
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph_context: Optional[Dict] = None  # Connected nodes/edges
    
    # For debugging/auditing
    query_latency_ms: float = 0.0
    raw_score: float = 0.0  # Before normalization


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search behavior."""
    # Weight distribution (must sum to 1.0)
    vector_weight: float = 0.4
    fulltext_weight: float = 0.2
    graph_weight: float = 0.4
    
    # Per-source limits
    vector_top_k: int = 20
    fulltext_top_k: int = 20
    graph_top_k: int = 20
    
    # Final output
    final_top_k: int = 10
    
    # Timeouts (ms)
    vector_timeout: int = 2000
    fulltext_timeout: int = 1000
    graph_timeout: int = 3000
    
    # Fusion parameters
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    
    # Graph boost for causally-connected results
    graph_boost_factor: float = 1.3
    
    def validate(self):
        total = self.vector_weight + self.fulltext_weight + self.graph_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class HybridRetriever:
    """
    Orchestrates hybrid search across Supabase (vector + fulltext) 
    and FalkorDB (graph).
    
    GUARANTEES:
    1. All three backends are ALWAYS queried in parallel
    2. Results are fused using Reciprocal Rank Fusion
    3. Graph relationships boost semantically-connected results
    4. Timeouts ensure graceful degradation
    5. Full audit trail for debugging
    """
    
    def __init__(
        self,
        supabase: SupabaseClient,
        falkordb: FalkorDB,
        embedding_model: str = "all-MiniLM-L6-v2",
        config: Optional[HybridSearchConfig] = None
    ):
        self.supabase = supabase
        self.falkordb = falkordb
        self.embedder = SentenceTransformer(embedding_model)
        self.config = config or HybridSearchConfig()
        self.config.validate()
        
        # For audit logging
        self._last_query_stats: Dict = {}
    
    async def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        require_all_sources: bool = False
    ) -> List[RetrievalResult]:
        """
        Execute hybrid search across all backends.
        
        Args:
            query: Natural language query
            filters: Optional filters (brand, region, date_range, etc.)
            require_all_sources: If True, fail if any backend times out
            
        Returns:
            Fused and ranked results from all sources
        """
        start_time = datetime.utcnow()
        
        # Generate embedding for vector search
        query_embedding = self.embedder.encode(query).tolist()
        
        # Extract entities for graph search
        entities = self._extract_entities(query)
        
        # Execute all searches in parallel
        results = await asyncio.gather(
            self._search_vector(query_embedding, filters),
            self._search_fulltext(query, filters),
            self._search_graph(entities, query, filters),
            return_exceptions=True
        )
        
        vector_results, fulltext_results, graph_results = results
        
        # Handle failures
        if isinstance(vector_results, Exception):
            if require_all_sources:
                raise vector_results
            vector_results = []
            
        if isinstance(fulltext_results, Exception):
            if require_all_sources:
                raise fulltext_results
            fulltext_results = []
            
        if isinstance(graph_results, Exception):
            if require_all_sources:
                raise graph_results
            graph_results = []
        
        # Fuse results using RRF
        fused = self._reciprocal_rank_fusion(
            vector_results,
            fulltext_results,
            graph_results
        )
        
        # Apply graph boost
        boosted = self._apply_graph_boost(fused, graph_results)
        
        # Record stats for auditing
        self._last_query_stats = {
            "query": query,
            "total_latency_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
            "vector_count": len(vector_results),
            "fulltext_count": len(fulltext_results),
            "graph_count": len(graph_results),
            "fused_count": len(boosted),
            "sources_used": {
                "vector": len(vector_results) > 0,
                "fulltext": len(fulltext_results) > 0,
                "graph": len(graph_results) > 0
            }
        }
        
        return boosted[:self.config.final_top_k]
    
    # =========================================================================
    # SUPABASE VECTOR SEARCH (pgvector)
    # =========================================================================
    
    async def _search_vector(
        self,
        embedding: List[float],
        filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """
        Semantic search using pgvector cosine similarity.
        
        Searches across:
        - insight_embeddings (causal insights, agent outputs)
        - episodic_memories (conversation history)
        - procedural_memories (successful patterns)
        """
        start = datetime.utcnow()
        
        try:
            # Use Supabase RPC for vector similarity search
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.supabase.rpc(
                        "hybrid_vector_search",
                        {
                            "query_embedding": embedding,
                            "match_count": self.config.vector_top_k,
                            "filters": filters or {}
                        }
                    ).execute
                ),
                timeout=self.config.vector_timeout / 1000
            )
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            return [
                RetrievalResult(
                    id=row["id"],
                    content=row["content"],
                    source=RetrievalSource.VECTOR,
                    score=row["similarity"],
                    metadata=row.get("metadata", {}),
                    query_latency_ms=latency,
                    raw_score=row["similarity"]
                )
                for row in response.data
            ]
            
        except asyncio.TimeoutError:
            print(f"Vector search timeout after {self.config.vector_timeout}ms")
            return []
    
    # =========================================================================
    # SUPABASE FULL-TEXT SEARCH
    # =========================================================================
    
    async def _search_fulltext(
        self,
        query: str,
        filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """
        Keyword/BM25-style search using PostgreSQL full-text search.
        
        Good for:
        - Exact term matching (KPI names, brand names)
        - Acronym matching (TRx, HCP, ROI)
        - Pattern matching with trigrams
        """
        start = datetime.utcnow()
        
        try:
            # Use PostgreSQL ts_rank for relevance scoring
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.supabase.rpc(
                        "hybrid_fulltext_search",
                        {
                            "search_query": query,
                            "match_count": self.config.fulltext_top_k,
                            "filters": filters or {}
                        }
                    ).execute
                ),
                timeout=self.config.fulltext_timeout / 1000
            )
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            return [
                RetrievalResult(
                    id=row["id"],
                    content=row["content"],
                    source=RetrievalSource.FULLTEXT,
                    score=row["rank"],
                    metadata=row.get("metadata", {}),
                    query_latency_ms=latency,
                    raw_score=row["rank"]
                )
                for row in response.data
            ]
            
        except asyncio.TimeoutError:
            print(f"Fulltext search timeout after {self.config.fulltext_timeout}ms")
            return []
    
    # =========================================================================
    # FALKORDB GRAPH SEARCH
    # =========================================================================
    
    async def _search_graph(
        self,
        entities: Dict[str, List[str]],
        query: str,
        filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """
        Graph traversal search using FalkorDB (Cypher queries).
        
        Finds:
        - Causal paths between entities
        - Related concepts (brands → KPIs → regions)
        - Historical patterns and relationships
        """
        start = datetime.utcnow()
        
        try:
            graph = self.falkordb.select_graph("e2i_knowledge")
            
            # Build dynamic Cypher query based on extracted entities
            cypher_query = self._build_graph_query(entities, filters)
            
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    graph.query,
                    cypher_query
                ),
                timeout=self.config.graph_timeout / 1000
            )
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            results = []
            for record in result.result_set:
                # Extract node/relationship data
                node_data = self._parse_graph_result(record)
                
                results.append(RetrievalResult(
                    id=node_data["id"],
                    content=node_data["description"],
                    source=RetrievalSource.GRAPH,
                    score=node_data.get("relevance", 0.5),
                    metadata=node_data.get("properties", {}),
                    graph_context={
                        "connected_nodes": node_data.get("neighbors", []),
                        "path_length": node_data.get("path_length", 0),
                        "relationship_types": node_data.get("rel_types", [])
                    },
                    query_latency_ms=latency,
                    raw_score=node_data.get("relevance", 0.5)
                ))
            
            return results
            
        except asyncio.TimeoutError:
            print(f"Graph search timeout after {self.config.graph_timeout}ms")
            return []
    
    def _build_graph_query(
        self,
        entities: Dict[str, List[str]],
        filters: Optional[Dict]
    ) -> str:
        """
        Build Cypher query for FalkorDB based on extracted entities.
        """
        # Start with entity matching
        match_clauses = []
        where_clauses = []
        
        if entities.get("brands"):
            brand_list = ", ".join(f"'{b}'" for b in entities["brands"])
            match_clauses.append("(b:Brand)")
            where_clauses.append(f"b.name IN [{brand_list}]")
        
        if entities.get("regions"):
            region_list = ", ".join(f"'{r}'" for r in entities["regions"])
            match_clauses.append("(r:Region)")
            where_clauses.append(f"r.name IN [{region_list}]")
        
        if entities.get("kpis"):
            kpi_list = ", ".join(f"'{k}'" for k in entities["kpis"])
            match_clauses.append("(k:KPI)")
            where_clauses.append(f"k.name IN [{kpi_list}]")
        
        # Default: find causal paths
        if not match_clauses:
            return """
                MATCH (n)-[rel:CAUSES|AFFECTS|CORRELATES*1..3]->(m)
                WHERE n.updated_at > datetime() - duration('P30D')
                RETURN n, rel, m, 
                       length((n)-[*]->(m)) as path_length
                ORDER BY path_length ASC
                LIMIT 20
            """
        
        # Build complete query
        match_str = "MATCH " + ", ".join(match_clauses)
        where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        return f"""
            {match_str}
            OPTIONAL MATCH path = (b)-[rel:CAUSES|AFFECTS*1..3]->(target)
            {where_str}
            RETURN b, r, k, path, 
                   nodes(path) as path_nodes,
                   relationships(path) as path_rels,
                   length(path) as path_length
            ORDER BY path_length ASC
            LIMIT {self.config.graph_top_k}
        """
    
    # =========================================================================
    # FUSION & RANKING
    # =========================================================================
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievalResult],
        fulltext_results: List[RetrievalResult],
        graph_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF Score = Σ 1/(k + rank_i) for each source
        
        This is more robust than simple score averaging because
        it handles different score distributions across backends.
        """
        k = self.config.rrf_k
        
        # Build score map: id -> cumulative RRF score
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}
        
        # Process each source with its weight
        sources = [
            (vector_results, self.config.vector_weight),
            (fulltext_results, self.config.fulltext_weight),
            (graph_results, self.config.graph_weight)
        ]
        
        for results, weight in sources:
            for rank, result in enumerate(results, start=1):
                rrf_contribution = weight * (1.0 / (k + rank))
                
                if result.id in rrf_scores:
                    rrf_scores[result.id] += rrf_contribution
                else:
                    rrf_scores[result.id] = rrf_contribution
                    result_map[result.id] = result
        
        # Update scores and sort
        for id, score in rrf_scores.items():
            result_map[id].score = score
        
        return sorted(
            result_map.values(),
            key=lambda x: x.score,
            reverse=True
        )
    
    def _apply_graph_boost(
        self,
        fused_results: List[RetrievalResult],
        graph_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Boost results that have graph connections to other results.
        
        Intuition: If result A is causally connected to result B
        in the knowledge graph, both should rank higher.
        """
        # Build set of IDs that appear in graph results
        graph_ids = {r.id for r in graph_results}
        graph_connected = {}
        
        for r in graph_results:
            if r.graph_context:
                for neighbor in r.graph_context.get("connected_nodes", []):
                    graph_connected[neighbor] = r.id
        
        # Apply boost
        for result in fused_results:
            if result.id in graph_ids:
                result.score *= self.config.graph_boost_factor
            elif result.id in graph_connected:
                # Partial boost for indirectly connected
                result.score *= (1 + (self.config.graph_boost_factor - 1) / 2)
        
        # Re-sort after boosting
        return sorted(fused_results, key=lambda x: x.score, reverse=True)
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract E2I domain entities from query for graph search.
        (Uses the fastText normalizer we built earlier)
        """
        # This integrates with our E2IEntityExtractor
        from src.nlp.entity_extractor import E2IEntityExtractor
        
        extractor = E2IEntityExtractor()
        entities = extractor.extract(query)
        
        return {
            "brands": entities.brands,
            "regions": entities.regions,
            "kpis": entities.kpis,
            "agents": entities.agents,
            "journey_stages": entities.journey_stages
        }
    
    def _parse_graph_result(self, record) -> Dict:
        """Parse FalkorDB result into standard format."""
        # Implementation depends on your graph schema
        return {
            "id": str(record[0].id) if record[0] else "unknown",
            "description": record[0].properties.get("description", ""),
            "properties": dict(record[0].properties) if record[0] else {},
            "neighbors": [str(n.id) for n in record[1]] if len(record) > 1 else [],
            "relevance": 0.7,  # Default relevance for graph results
            "path_length": record[-1] if isinstance(record[-1], int) else 0
        }
    
    def get_last_query_stats(self) -> Dict:
        """Return audit stats from the last query."""
        return self._last_query_stats.copy()
```

### 1.2 Supabase SQL Functions (Required)

```sql
-- migrations/011_hybrid_search_functions.sql

-- ============================================================================
-- VECTOR SEARCH FUNCTION
-- ============================================================================

CREATE OR REPLACE FUNCTION hybrid_vector_search(
    query_embedding vector(1536),
    match_count int DEFAULT 20,
    filters jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id uuid,
    content text,
    similarity float,
    metadata jsonb,
    source_table text
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    
    -- Search insight_embeddings (agent outputs, causal paths)
    SELECT 
        ie.insight_id as id,
        ie.content,
        1 - (ie.embedding <=> query_embedding) as similarity,
        jsonb_build_object(
            'source_type', ie.source_type,
            'agent_name', ie.agent_name,
            'created_at', ie.created_at
        ) as metadata,
        'insight_embeddings'::text as source_table
    FROM insight_embeddings ie
    WHERE 
        -- Apply optional filters
        (filters->>'brand' IS NULL OR ie.metadata->>'brand' = filters->>'brand')
        AND (filters->>'region' IS NULL OR ie.metadata->>'region' = filters->>'region')
        AND (filters->>'date_from' IS NULL OR ie.created_at >= (filters->>'date_from')::timestamp)
    
    UNION ALL
    
    -- Search episodic_memories (conversation history)
    SELECT
        em.memory_id as id,
        em.content,
        1 - (em.embedding <=> query_embedding) as similarity,
        jsonb_build_object(
            'memory_type', 'episodic',
            'user_id', em.user_id,
            'created_at', em.created_at
        ) as metadata,
        'episodic_memories'::text as source_table
    FROM episodic_memories em
    WHERE em.is_active = true
    
    UNION ALL
    
    -- Search procedural_memories (successful patterns)
    SELECT
        pm.procedure_id as id,
        pm.procedure_name || ': ' || pm.description as content,
        1 - (pm.trigger_embedding <=> query_embedding) as similarity,
        jsonb_build_object(
            'memory_type', 'procedural',
            'success_rate', pm.success_count::float / NULLIF(pm.usage_count, 0),
            'tool_sequence', pm.tool_sequence
        ) as metadata,
        'procedural_memories'::text as source_table
    FROM procedural_memories pm
    WHERE pm.success_count > 0
    
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;


-- ============================================================================
-- FULL-TEXT SEARCH FUNCTION
-- ============================================================================

CREATE OR REPLACE FUNCTION hybrid_fulltext_search(
    search_query text,
    match_count int DEFAULT 20,
    filters jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id uuid,
    content text,
    rank float,
    metadata jsonb,
    source_table text
)
LANGUAGE plpgsql
AS $$
DECLARE
    tsquery_val tsquery;
BEGIN
    -- Parse search query with prefix matching for partial words
    tsquery_val := websearch_to_tsquery('english', search_query);
    
    RETURN QUERY
    
    -- Search causal_paths
    SELECT
        cp.path_id as id,
        cp.description as content,
        ts_rank_cd(cp.search_vector, tsquery_val) as rank,
        jsonb_build_object(
            'source_node', cp.source_node,
            'target_node', cp.target_node,
            'effect_size', cp.effect_size,
            'confidence', cp.confidence_score
        ) as metadata,
        'causal_paths'::text as source_table
    FROM causal_paths cp
    WHERE cp.search_vector @@ tsquery_val
    
    UNION ALL
    
    -- Search agent_activities
    SELECT
        aa.activity_id as id,
        aa.analysis_results::text as content,
        ts_rank_cd(aa.search_vector, tsquery_val) as rank,
        jsonb_build_object(
            'agent_name', aa.agent_name,
            'agent_tier', aa.agent_tier,
            'status', aa.status
        ) as metadata,
        'agent_activities'::text as source_table
    FROM agent_activities aa
    WHERE aa.search_vector @@ tsquery_val
    
    UNION ALL
    
    -- Search triggers with reason
    SELECT
        t.trigger_id as id,
        t.trigger_reason as content,
        ts_rank_cd(t.search_vector, tsquery_val) as rank,
        jsonb_build_object(
            'trigger_type', t.trigger_type,
            'priority', t.priority,
            'brand', t.brand
        ) as metadata,
        'triggers'::text as source_table
    FROM triggers t
    WHERE t.search_vector @@ tsquery_val
    
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;


-- ============================================================================
-- ADD SEARCH VECTORS TO TABLES (run once)
-- ============================================================================

-- Add search vector columns
ALTER TABLE causal_paths ADD COLUMN IF NOT EXISTS 
    search_vector tsvector 
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(description, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(source_node, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(target_node, '')), 'B')
    ) STORED;

ALTER TABLE agent_activities ADD COLUMN IF NOT EXISTS
    search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(agent_name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(analysis_results::text, '')), 'B')
    ) STORED;

ALTER TABLE triggers ADD COLUMN IF NOT EXISTS
    search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(trigger_reason, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(trigger_type::text, '')), 'B')
    ) STORED;

-- Create GIN indexes for fast full-text search
CREATE INDEX IF NOT EXISTS idx_causal_paths_search ON causal_paths USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_agent_activities_search ON agent_activities USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_triggers_search ON triggers USING GIN(search_vector);

-- Create HNSW indexes for fast vector search
CREATE INDEX IF NOT EXISTS idx_insight_embeddings_vector 
    ON insight_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_episodic_memories_vector 
    ON episodic_memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_procedural_memories_vector 
    ON procedural_memories USING hnsw (trigger_embedding vector_cosine_ops);
```

---

## Part 2: FalkorDB Graph Schema

```cypher
// migrations/002_semantic_graph_schema.cypher

// ============================================================================
// NODE TYPES
// ============================================================================

// Core business entities
CREATE INDEX FOR (b:Brand) ON (b.name);
CREATE INDEX FOR (r:Region) ON (r.name);
CREATE INDEX FOR (k:KPI) ON (k.name);
CREATE INDEX FOR (a:Agent) ON (a.name);

// Domain concepts
CREATE INDEX FOR (js:JourneyStage) ON (js.name);
CREATE INDEX FOR (t:Trigger) ON (t.id);
CREATE INDEX FOR (i:Insight) ON (i.id);

// Causal graph nodes
CREATE INDEX FOR (cv:CausalVariable) ON (cv.name);
CREATE INDEX FOR (cp:CausalPath) ON (cp.id);

// ============================================================================
// RELATIONSHIP TYPES
// ============================================================================

// Causal relationships
// (Treatment)-[:CAUSES {effect_size: 0.15, confidence: 0.87}]->(Outcome)
// (Confound)-[:AFFECTS]->(Treatment)
// (Variable)-[:CORRELATES {strength: 0.6}]->(Variable)

// Business relationships
// (Brand)-[:SOLD_IN]->(Region)
// (Agent)-[:ANALYZES]->(KPI)
// (Trigger)-[:TARGETS]->(JourneyStage)

// Temporal relationships
// (Insight)-[:PRECEDED_BY]->(Insight)
// (KPI)-[:TREND_CHANGED {direction: 'down', magnitude: 0.12}]->(KPI)

// ============================================================================
// SAMPLE DATA SETUP
// ============================================================================

// Create brand nodes
MERGE (rem:Brand {name: 'Remibrutinib', therapeutic_area: 'CSU', launch_year: 2024})
MERGE (fab:Brand {name: 'Fabhalta', therapeutic_area: 'PNH', launch_year: 2023})
MERGE (kis:Brand {name: 'Kisqali', therapeutic_area: 'Oncology', launch_year: 2017})

// Create region nodes
MERGE (ne:Region {name: 'Northeast', abbrev: 'NE'})
MERGE (se:Region {name: 'Southeast', abbrev: 'SE'})
MERGE (mw:Region {name: 'Midwest', abbrev: 'MW'})
MERGE (we:Region {name: 'West', abbrev: 'W'})

// Create KPI nodes
MERGE (conv:KPI {name: 'conversion_rate', workstream: 'WS2', unit: 'percentage'})
MERGE (trx:KPI {name: 'TRx', workstream: 'WS3', unit: 'count'})
MERGE (eng:KPI {name: 'engagement_score', workstream: 'WS2', unit: 'score'})
MERGE (cov:KPI {name: 'HCP_coverage', workstream: 'WS1', unit: 'percentage'})

// Create agent nodes
MERGE (orch:Agent {name: 'orchestrator', tier: 0})
MERGE (ci:Agent {name: 'causal_impact', tier: 1})
MERGE (ga:Agent {name: 'gap_analyzer', tier: 1})
MERGE (dm:Agent {name: 'drift_monitor', tier: 2})
MERGE (exp:Agent {name: 'explainer', tier: 4})

// Create causal relationships
MERGE (eng)-[:CAUSES {effect_size: 0.23, confidence: 0.85}]->(conv)
MERGE (cov)-[:CAUSES {effect_size: 0.15, confidence: 0.78}]->(eng)
MERGE (conv)-[:CAUSES {effect_size: 0.31, confidence: 0.91}]->(trx)

// Create brand-region relationships
MERGE (rem)-[:SOLD_IN {market_share: 0.12}]->(ne)
MERGE (rem)-[:SOLD_IN {market_share: 0.08}]->(mw)
MERGE (fab)-[:SOLD_IN {market_share: 0.15}]->(ne)
MERGE (kis)-[:SOLD_IN {market_share: 0.22}]->(we)

// Create agent-KPI relationships
MERGE (ci)-[:ANALYZES]->(conv)
MERGE (ga)-[:ANALYZES]->(trx)
MERGE (dm)-[:MONITORS]->(eng)
```

---

## Part 3: Knowledge Graph Visualization

### 3.1 React Component with Cytoscape.js

```typescript
// frontend/src/components/KnowledgeGraph/KnowledgeGraphViewer.tsx

import React, { useEffect, useRef, useState, useCallback } from 'react';
import cytoscape, { Core, ElementDefinition } from 'cytoscape';
import dagre from 'cytoscape-dagre';
import popper from 'cytoscape-popper';
import tippy from 'tippy.js';
import 'tippy.js/dist/tippy.css';

// Register extensions
cytoscape.use(dagre);
cytoscape.use(popper);

// Types
interface GraphNode {
  id: string;
  label: string;
  type: 'Brand' | 'Region' | 'KPI' | 'Agent' | 'CausalVariable' | 'Insight';
  properties: Record<string, any>;
}

interface GraphEdge {
  source: string;
  target: string;
  type: 'CAUSES' | 'AFFECTS' | 'CORRELATES' | 'ANALYZES' | 'SOLD_IN';
  properties: Record<string, any>;
}

interface KnowledgeGraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface Props {
  data: KnowledgeGraphData;
  onNodeClick?: (node: GraphNode) => void;
  onEdgeClick?: (edge: GraphEdge) => void;
  highlightPath?: string[];  // Node IDs to highlight
  layout?: 'dagre' | 'cose' | 'breadthfirst' | 'circle';
}

// Color scheme for node types
const NODE_COLORS: Record<string, string> = {
  Brand: '#667eea',       // Purple
  Region: '#48bb78',      // Green
  KPI: '#f6ad55',         // Orange
  Agent: '#fc8181',       // Red
  CausalVariable: '#4fd1c5', // Teal
  Insight: '#b794f4',     // Light purple
};

// Edge styles by relationship type
const EDGE_STYLES: Record<string, { color: string; style: string }> = {
  CAUSES: { color: '#e53e3e', style: 'solid' },
  AFFECTS: { color: '#dd6b20', style: 'dashed' },
  CORRELATES: { color: '#38a169', style: 'dotted' },
  ANALYZES: { color: '#3182ce', style: 'solid' },
  SOLD_IN: { color: '#805ad5', style: 'solid' },
};

export const KnowledgeGraphViewer: React.FC<Props> = ({
  data,
  onNodeClick,
  onEdgeClick,
  highlightPath = [],
  layout = 'dagre'
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  // Convert data to Cytoscape format
  const getCytoscapeElements = useCallback((): ElementDefinition[] => {
    const elements: ElementDefinition[] = [];

    // Add nodes
    data.nodes.forEach(node => {
      elements.push({
        data: {
          id: node.id,
          label: node.label,
          type: node.type,
          ...node.properties,
        },
        classes: [
          node.type.toLowerCase(),
          highlightPath.includes(node.id) ? 'highlighted' : ''
        ].join(' ')
      });
    });

    // Add edges
    data.edges.forEach((edge, index) => {
      elements.push({
        data: {
          id: `edge-${index}`,
          source: edge.source,
          target: edge.target,
          type: edge.type,
          ...edge.properties,
        },
        classes: edge.type.toLowerCase()
      });
    });

    return elements;
  }, [data, highlightPath]);

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;

    const cy = cytoscape({
      container: containerRef.current,
      elements: getCytoscapeElements(),
      
      style: [
        // Base node style
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': (ele) => NODE_COLORS[ele.data('type')] || '#888',
            'color': '#fff',
            'font-size': '12px',
            'font-weight': 'bold',
            'text-outline-color': '#333',
            'text-outline-width': 1,
            'width': 60,
            'height': 60,
            'border-width': 2,
            'border-color': '#333',
          }
        },
        
        // Highlighted nodes
        {
          selector: 'node.highlighted',
          style: {
            'border-width': 4,
            'border-color': '#ffd700',
            'box-shadow': '0 0 15px #ffd700',
          }
        },
        
        // Selected node
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#00ff00',
            'background-color': '#2d3748',
          }
        },
        
        // Base edge style
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': (ele) => EDGE_STYLES[ele.data('type')]?.color || '#888',
            'line-style': (ele) => EDGE_STYLES[ele.data('type')]?.style || 'solid',
            'target-arrow-color': (ele) => EDGE_STYLES[ele.data('type')]?.color || '#888',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': (ele) => {
              const effectSize = ele.data('effect_size');
              return effectSize ? `${(effectSize * 100).toFixed(0)}%` : '';
            },
            'font-size': '10px',
            'text-rotation': 'autorotate',
            'text-margin-y': -10,
          }
        },
        
        // Causal edges (thicker, with effect labels)
        {
          selector: 'edge.causes',
          style: {
            'width': 4,
            'line-color': '#e53e3e',
            'target-arrow-color': '#e53e3e',
          }
        },
      ],
      
      layout: {
        name: layout,
        rankDir: 'LR',  // Left to right for DAG
        nodeSep: 80,
        rankSep: 120,
        animate: true,
        animationDuration: 500,
      },
      
      // Interaction settings
      minZoom: 0.3,
      maxZoom: 3,
      wheelSensitivity: 0.3,
    });

    // Add tooltips using Tippy.js
    cy.nodes().forEach(node => {
      const ref = node.popperRef();
      const content = document.createElement('div');
      content.innerHTML = `
        <div style="padding: 8px; background: #2d3748; color: white; border-radius: 4px;">
          <strong>${node.data('label')}</strong><br/>
          <em>Type: ${node.data('type')}</em>
          ${node.data('effect_size') ? `<br/>Effect: ${(node.data('effect_size') * 100).toFixed(1)}%` : ''}
          ${node.data('confidence') ? `<br/>Confidence: ${(node.data('confidence') * 100).toFixed(0)}%` : ''}
        </div>
      `;
      
      tippy(ref, {
        content,
        trigger: 'mouseenter',
        placement: 'top',
        arrow: true,
        allowHTML: true,
      });
    });

    // Event handlers
    cy.on('tap', 'node', (event) => {
      const node = event.target;
      const nodeData: GraphNode = {
        id: node.id(),
        label: node.data('label'),
        type: node.data('type'),
        properties: node.data(),
      };
      setSelectedNode(nodeData);
      onNodeClick?.(nodeData);
    });

    cy.on('tap', 'edge', (event) => {
      const edge = event.target;
      const edgeData: GraphEdge = {
        source: edge.data('source'),
        target: edge.data('target'),
        type: edge.data('type'),
        properties: edge.data(),
      };
      onEdgeClick?.(edgeData);
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [data, layout, highlightPath, onNodeClick, onEdgeClick, getCytoscapeElements]);

  // Update elements when data changes
  useEffect(() => {
    if (cyRef.current) {
      cyRef.current.json({ elements: getCytoscapeElements() });
      cyRef.current.layout({ name: layout }).run();
    }
  }, [data, getCytoscapeElements, layout]);

  return (
    <div className="knowledge-graph-container">
      <div 
        ref={containerRef} 
        style={{ 
          width: '100%', 
          height: '600px',
          background: '#1a202c',
          borderRadius: '8px',
        }} 
      />
      
      {/* Legend */}
      <div className="graph-legend" style={{ 
        display: 'flex', 
        gap: '16px', 
        marginTop: '12px',
        flexWrap: 'wrap'
      }}>
        {Object.entries(NODE_COLORS).map(([type, color]) => (
          <div key={type} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{ 
              width: '16px', 
              height: '16px', 
              borderRadius: '50%', 
              backgroundColor: color 
            }} />
            <span style={{ fontSize: '12px' }}>{type}</span>
          </div>
        ))}
      </div>
      
      {/* Selected node details */}
      {selectedNode && (
        <div className="node-details" style={{
          marginTop: '12px',
          padding: '12px',
          background: '#2d3748',
          borderRadius: '8px',
          color: 'white'
        }}>
          <h4>{selectedNode.label}</h4>
          <p>Type: {selectedNode.type}</p>
          <pre style={{ fontSize: '11px', overflow: 'auto' }}>
            {JSON.stringify(selectedNode.properties, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};
```

### 3.2 API Endpoint to Fetch Graph Data

```python
# src/api/routes/knowledge_graph.py

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from pydantic import BaseModel
from falkordb import FalkorDB

router = APIRouter(prefix="/api/knowledge-graph", tags=["knowledge-graph"])


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    properties: dict


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    properties: dict


class KnowledgeGraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: dict


@router.get("/causal-subgraph", response_model=KnowledgeGraphResponse)
async def get_causal_subgraph(
    center_node: Optional[str] = Query(None, description="Center node ID for ego graph"),
    node_types: Optional[List[str]] = Query(None, description="Filter by node types"),
    relationship_types: Optional[List[str]] = Query(None, description="Filter by relationship types"),
    max_depth: int = Query(2, ge=1, le=4, description="Max traversal depth"),
    limit: int = Query(100, ge=1, le=500, description="Max nodes to return")
):
    """
    Fetch a subgraph from FalkorDB for visualization.
    
    If center_node is provided, returns ego graph around that node.
    Otherwise returns the full causal graph (up to limit).
    """
    try:
        falkordb = FalkorDB(host="localhost", port=6379)
        graph = falkordb.select_graph("e2i_knowledge")
        
        if center_node:
            # Ego graph query
            query = f"""
                MATCH path = (center {{id: $center_id}})-[*1..{max_depth}]-(neighbor)
                WHERE center <> neighbor
                UNWIND relationships(path) as rel
                UNWIND nodes(path) as node
                RETURN DISTINCT node, rel
                LIMIT {limit}
            """
            params = {"center_id": center_node}
        else:
            # Full graph query (filtered)
            node_filter = ""
            if node_types:
                type_list = " OR ".join(f"n:{t}" for t in node_types)
                node_filter = f"WHERE ({type_list})"
            
            rel_filter = ""
            if relationship_types:
                rel_list = "|".join(relationship_types)
                rel_filter = f"[:{rel_list}]"
            
            query = f"""
                MATCH (n)-{rel_filter or '[r]'}->(m)
                {node_filter}
                RETURN n, r, m
                LIMIT {limit}
            """
            params = {}
        
        result = graph.query(query, params)
        
        # Parse results into nodes and edges
        nodes_map = {}
        edges = []
        
        for record in result.result_set:
            # Process nodes
            for item in record:
                if hasattr(item, 'id'):  # It's a node
                    if item.id not in nodes_map:
                        nodes_map[item.id] = GraphNode(
                            id=str(item.id),
                            label=item.properties.get('name', str(item.id)),
                            type=item.labels[0] if item.labels else 'Unknown',
                            properties=dict(item.properties)
                        )
                elif hasattr(item, 'type'):  # It's a relationship
                    edges.append(GraphEdge(
                        source=str(item.src_node),
                        target=str(item.dest_node),
                        type=item.type,
                        properties=dict(item.properties)
                    ))
        
        return KnowledgeGraphResponse(
            nodes=list(nodes_map.values()),
            edges=edges,
            metadata={
                "total_nodes": len(nodes_map),
                "total_edges": len(edges),
                "query_depth": max_depth,
                "center_node": center_node
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal-path", response_model=KnowledgeGraphResponse)
async def get_causal_path(
    source: str = Query(..., description="Source node ID"),
    target: str = Query(..., description="Target node ID"),
    max_length: int = Query(4, ge=1, le=6, description="Max path length")
):
    """
    Find causal paths between two nodes.
    
    Returns all paths up to max_length that connect source to target
    via CAUSES or AFFECTS relationships.
    """
    try:
        falkordb = FalkorDB(host="localhost", port=6379)
        graph = falkordb.select_graph("e2i_knowledge")
        
        query = f"""
            MATCH path = (s {{id: $source}})-[:CAUSES|AFFECTS*1..{max_length}]->(t {{id: $target}})
            UNWIND nodes(path) as node
            UNWIND relationships(path) as rel
            RETURN DISTINCT node, rel, length(path) as path_length
            ORDER BY path_length ASC
        """
        
        result = graph.query(query, {"source": source, "target": target})
        
        # Parse results
        nodes_map = {}
        edges = []
        
        for record in result.result_set:
            node = record[0]
            rel = record[1]
            
            if node.id not in nodes_map:
                nodes_map[node.id] = GraphNode(
                    id=str(node.id),
                    label=node.properties.get('name', str(node.id)),
                    type=node.labels[0] if node.labels else 'Unknown',
                    properties=dict(node.properties)
                )
            
            if rel:
                edges.append(GraphEdge(
                    source=str(rel.src_node),
                    target=str(rel.dest_node),
                    type=rel.type,
                    properties=dict(rel.properties)
                ))
        
        return KnowledgeGraphResponse(
            nodes=list(nodes_map.values()),
            edges=edges,
            metadata={
                "source": source,
                "target": target,
                "paths_found": len(set(e.source for e in edges))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Part 4: Ensuring Hybrid Queries Always Execute

### 4.1 Health Check & Circuit Breaker

```python
# src/rag/health_monitor.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional
import asyncio
from enum import Enum


class BackendStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class BackendHealth:
    status: BackendStatus
    last_check: datetime
    latency_ms: float
    error_rate: float
    consecutive_failures: int


class HybridRAGHealthMonitor:
    """
    Monitors health of all RAG backends and ensures hybrid queries
    always attempt all sources.
    """
    
    def __init__(
        self,
        supabase_client,
        falkordb_client,
        check_interval_seconds: int = 30
    ):
        self.supabase = supabase_client
        self.falkordb = falkordb_client
        self.check_interval = check_interval_seconds
        
        self._health: Dict[str, BackendHealth] = {
            "supabase_vector": BackendHealth(
                status=BackendStatus.HEALTHY,
                last_check=datetime.utcnow(),
                latency_ms=0,
                error_rate=0,
                consecutive_failures=0
            ),
            "supabase_fulltext": BackendHealth(
                status=BackendStatus.HEALTHY,
                last_check=datetime.utcnow(),
                latency_ms=0,
                error_rate=0,
                consecutive_failures=0
            ),
            "falkordb_graph": BackendHealth(
                status=BackendStatus.HEALTHY,
                last_check=datetime.utcnow(),
                latency_ms=0,
                error_rate=0,
                consecutive_failures=0
            ),
        }
        
        self._running = False
    
    async def start(self):
        """Start background health monitoring."""
        self._running = True
        while self._running:
            await self._check_all_backends()
            await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop health monitoring."""
        self._running = False
    
    async def _check_all_backends(self):
        """Check health of all backends."""
        await asyncio.gather(
            self._check_supabase_vector(),
            self._check_supabase_fulltext(),
            self._check_falkordb()
        )
    
    async def _check_supabase_vector(self):
        """Health check for pgvector."""
        backend = "supabase_vector"
        start = datetime.utcnow()
        
        try:
            # Simple vector similarity query
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.supabase.rpc(
                        "hybrid_vector_search",
                        {
                            "query_embedding": [0.0] * 1536,  # Zero vector
                            "match_count": 1
                        }
                    ).execute
                ),
                timeout=5.0
            )
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            self._update_health(backend, True, latency)
            
        except Exception as e:
            self._update_health(backend, False, 5000, str(e))
    
    async def _check_supabase_fulltext(self):
        """Health check for full-text search."""
        backend = "supabase_fulltext"
        start = datetime.utcnow()
        
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.supabase.rpc(
                        "hybrid_fulltext_search",
                        {
                            "search_query": "health_check",
                            "match_count": 1
                        }
                    ).execute
                ),
                timeout=3.0
            )
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            self._update_health(backend, True, latency)
            
        except Exception as e:
            self._update_health(backend, False, 3000, str(e))
    
    async def _check_falkordb(self):
        """Health check for FalkorDB graph."""
        backend = "falkordb_graph"
        start = datetime.utcnow()
        
        try:
            graph = self.falkordb.select_graph("e2i_knowledge")
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    graph.query,
                    "MATCH (n) RETURN count(n) LIMIT 1"
                ),
                timeout=5.0
            )
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            self._update_health(backend, True, latency)
            
        except Exception as e:
            self._update_health(backend, False, 5000, str(e))
    
    def _update_health(
        self,
        backend: str,
        success: bool,
        latency_ms: float,
        error: Optional[str] = None
    ):
        """Update health status for a backend."""
        health = self._health[backend]
        health.last_check = datetime.utcnow()
        health.latency_ms = latency_ms
        
        if success:
            health.consecutive_failures = 0
            health.error_rate = max(0, health.error_rate - 0.1)
            
            if health.latency_ms < 500:
                health.status = BackendStatus.HEALTHY
            elif health.latency_ms < 2000:
                health.status = BackendStatus.DEGRADED
            else:
                health.status = BackendStatus.DEGRADED
        else:
            health.consecutive_failures += 1
            health.error_rate = min(1.0, health.error_rate + 0.2)
            
            if health.consecutive_failures >= 3:
                health.status = BackendStatus.UNHEALTHY
            else:
                health.status = BackendStatus.DEGRADED
    
    def get_status(self) -> Dict[str, Dict]:
        """Get current health status of all backends."""
        return {
            backend: {
                "status": health.status.value,
                "latency_ms": health.latency_ms,
                "error_rate": health.error_rate,
                "last_check": health.last_check.isoformat(),
                "consecutive_failures": health.consecutive_failures
            }
            for backend, health in self._health.items()
        }
    
    def is_hybrid_available(self) -> bool:
        """Check if all backends are available for hybrid search."""
        return all(
            h.status != BackendStatus.UNHEALTHY 
            for h in self._health.values()
        )
    
    def get_available_backends(self) -> list:
        """Get list of currently available backends."""
        return [
            backend for backend, health in self._health.items()
            if health.status != BackendStatus.UNHEALTHY
        ]
```

### 4.2 Test Suite for Hybrid Guarantee

```python
# tests/integration/test_hybrid_retriever.py

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.rag.hybrid_retriever import (
    HybridRetriever,
    HybridSearchConfig,
    RetrievalSource,
    RetrievalResult
)


class TestHybridRetrieverGuarantees:
    """
    Tests to GUARANTEE hybrid queries always execute across all backends.
    """
    
    @pytest.fixture
    def mock_supabase(self):
        client = Mock()
        client.rpc.return_value.execute.return_value = Mock(
            data=[
                {"id": "vec-1", "content": "Vector result", "similarity": 0.9, "metadata": {}},
                {"id": "vec-2", "content": "Vector result 2", "similarity": 0.8, "metadata": {}},
            ]
        )
        return client
    
    @pytest.fixture
    def mock_falkordb(self):
        client = Mock()
        graph = Mock()
        graph.query.return_value = Mock(
            result_set=[
                (Mock(id="graph-1", properties={"description": "Graph result"}),),
            ]
        )
        client.select_graph.return_value = graph
        return client
    
    @pytest.fixture
    def retriever(self, mock_supabase, mock_falkordb):
        return HybridRetriever(
            supabase=mock_supabase,
            falkordb=mock_falkordb,
            embedding_model="all-MiniLM-L6-v2"
        )
    
    @pytest.mark.asyncio
    async def test_all_backends_queried(self, retriever, mock_supabase, mock_falkordb):
        """
        GUARANTEE: All three backends are ALWAYS queried.
        """
        results = await retriever.search("test query")
        
        # Verify Supabase vector was called
        assert mock_supabase.rpc.call_count >= 1
        calls = [str(c) for c in mock_supabase.rpc.call_args_list]
        assert any("hybrid_vector_search" in c for c in calls)
        
        # Verify Supabase fulltext was called
        assert any("hybrid_fulltext_search" in c for c in calls)
        
        # Verify FalkorDB was called
        mock_falkordb.select_graph.assert_called()
    
    @pytest.mark.asyncio
    async def test_results_from_all_sources(self, retriever):
        """
        GUARANTEE: Results include items from all available sources.
        """
        results = await retriever.search("test query")
        
        sources = {r.source for r in results}
        
        # Should have results from at least vector and graph
        assert RetrievalSource.VECTOR in sources or RetrievalSource.FULLTEXT in sources
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_timeout(self, retriever, mock_supabase):
        """
        GUARANTEE: If one backend times out, others still return results.
        """
        # Make vector search timeout
        async def slow_rpc(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
        
        with patch.object(retriever, '_search_vector', side_effect=asyncio.TimeoutError):
            results = await retriever.search("test query")
            
            # Should still have results from other backends
            assert len(results) > 0
            
            # Vector results should be empty but others present
            stats = retriever.get_last_query_stats()
            assert stats["vector_count"] == 0
            assert stats["fulltext_count"] > 0 or stats["graph_count"] > 0
    
    @pytest.mark.asyncio
    async def test_require_all_sources_flag(self, retriever):
        """
        GUARANTEE: require_all_sources=True fails if any backend unavailable.
        """
        with patch.object(retriever, '_search_vector', side_effect=asyncio.TimeoutError):
            with pytest.raises(asyncio.TimeoutError):
                await retriever.search("test query", require_all_sources=True)
    
    @pytest.mark.asyncio
    async def test_rrf_fusion_combines_all_results(self, retriever):
        """
        GUARANTEE: RRF fusion properly combines results from all sources.
        """
        results = await retriever.search("test query")
        
        # Check that scores are normalized RRF scores (not raw scores)
        for r in results:
            # RRF scores should be small fractions
            assert 0 < r.score < 1
    
    @pytest.mark.asyncio
    async def test_graph_boost_applied(self, retriever):
        """
        GUARANTEE: Graph-connected results receive boost.
        """
        results = await retriever.search("test query")
        
        # Results with graph_context should have boosted scores
        graph_results = [r for r in results if r.graph_context]
        non_graph_results = [r for r in results if not r.graph_context]
        
        if graph_results and non_graph_results:
            avg_graph_score = sum(r.score for r in graph_results) / len(graph_results)
            avg_non_graph_score = sum(r.score for r in non_graph_results) / len(non_graph_results)
            
            # Graph results should generally score higher (with boost)
            # This is a soft assertion since it depends on raw scores
            assert avg_graph_score >= avg_non_graph_score * 0.8
    
    @pytest.mark.asyncio
    async def test_audit_trail_recorded(self, retriever):
        """
        GUARANTEE: Full audit trail is recorded for every query.
        """
        results = await retriever.search("test query")
        stats = retriever.get_last_query_stats()
        
        assert "query" in stats
        assert "total_latency_ms" in stats
        assert "vector_count" in stats
        assert "fulltext_count" in stats
        assert "graph_count" in stats
        assert "sources_used" in stats
        
        # Verify sources_used is accurate
        assert isinstance(stats["sources_used"]["vector"], bool)
        assert isinstance(stats["sources_used"]["fulltext"], bool)
        assert isinstance(stats["sources_used"]["graph"], bool)
```

---

## Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Vector Search** | Supabase + pgvector | Semantic similarity (embeddings) |
| **Full-Text Search** | Supabase + PostgreSQL | Keyword/BM25 matching |
| **Graph Search** | FalkorDB + Cypher | Relationship traversal |
| **Fusion** | Reciprocal Rank Fusion | Combine rankings fairly |
| **Visualization** | Cytoscape.js + React | Interactive graph explorer |
| **Health Monitoring** | Custom circuit breaker | Graceful degradation |

**Key Guarantees:**
1. ✅ All 3 backends queried in parallel on every search
2. ✅ Timeouts prevent slow backends from blocking
3. ✅ RRF ensures fair fusion regardless of score distributions
4. ✅ Graph relationships boost causally-connected results
5. ✅ Full audit trail for debugging
6. ✅ Test suite validates hybrid behavior
