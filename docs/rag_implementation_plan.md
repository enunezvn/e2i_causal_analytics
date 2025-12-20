# E2I Hybrid RAG Implementation Plan
**Created**: 2025-12-15
**Status**: Planning Phase
**Priority**: High

---

## Executive Summary

This document outlines the comprehensive implementation plan for integrating a Hybrid RAG (Retrieval-Augmented Generation) system into the E2I Causal Analytics platform. The implementation combines:

1. **Supabase pgvector** - Semantic/vector search
2. **Supabase PostgreSQL full-text** - Keyword/BM25 search
3. **FalkorDB graph database** - Relationship/causal path traversal
4. **Reciprocal Rank Fusion** - Result combining algorithm
5. **Cytoscape.js visualization** - Interactive knowledge graph UI

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Backend Implementation](#backend-implementation)
3. [Database Changes](#database-changes)
4. [Frontend Integration](#frontend-integration)
5. [API Endpoints](#api-endpoints)
6. [Configuration & Dependencies](#configuration--dependencies)
7. [Testing Strategy](#testing-strategy)
8. [Documentation Updates](#documentation-updates)
9. [Deployment Considerations](#deployment-considerations)
10. [Gaps & Recommendations](#gaps--recommendations)

---

## Architecture Overview

### High-Level Data Flow

```
User Query
    ↓
Query Analyzer (extract entities)
    ↓
┌─────────────────────────────────────┐
│  Hybrid Retriever Orchestrator      │
│  (Parallel Dispatch)                │
└─────────────────────────────────────┘
    ↓           ↓           ↓
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Vector  │ │Fulltext │ │  Graph  │
│ Search  │ │ Search  │ │ Search  │
│(pgvector)│ │  (tsvec)│ │(FalkorDB)│
└─────────┘ └─────────┘ └─────────┘
    ↓           ↓           ↓
┌─────────────────────────────────────┐
│  Reciprocal Rank Fusion (RRF)       │
│  + Graph Boost                      │
└─────────────────────────────────────┘
    ↓
Context Assembly + LLM
    ↓
Response to User
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vector Search | Supabase + pgvector | Semantic similarity via embeddings |
| Full-Text Search | PostgreSQL tsvector | Keyword/BM25-style matching |
| Graph Search | FalkorDB + Cypher | Causal relationship traversal |
| Fusion Algorithm | RRF | Combine results from all sources |
| Visualization | Cytoscape.js | Interactive graph UI |
| Health Monitoring | Custom Circuit Breaker | Graceful degradation |

---

## Backend Implementation

### 1. Core RAG Module Structure

**New Directory**: `src/rag/`

```
src/rag/
├── __init__.py
├── hybrid_retriever.py        # Main HybridRetriever class
├── health_monitor.py          # HybridRAGHealthMonitor class
├── config.py                  # Configuration dataclasses
└── types.py                   # RetrievalResult, RetrievalSource enums
```

#### 1.1 Core Classes to Implement

**File**: `src/rag/hybrid_retriever.py` (~570 lines)

Key classes:
- `RetrievalSource(Enum)` - Track result sources (VECTOR, FULLTEXT, GRAPH)
- `RetrievalResult` - Unified result format
- `HybridSearchConfig` - Search configuration with weights
- `HybridRetriever` - Main orchestrator class

Methods to implement:
- `search()` - Main entry point for hybrid search
- `_search_vector()` - Supabase pgvector search
- `_search_fulltext()` - PostgreSQL full-text search
- `_search_graph()` - FalkorDB graph traversal
- `_reciprocal_rank_fusion()` - RRF algorithm
- `_apply_graph_boost()` - Boost graph-connected results
- `_extract_entities()` - Extract E2I domain entities
- `_build_graph_query()` - Build dynamic Cypher queries

**File**: `src/rag/embeddings.py` (NEW - ~150 lines)

OpenAI embedding client:
```python
from openai import OpenAI
import os
from typing import List, Union

class OpenAIEmbeddingClient:
    """Client for generating embeddings using OpenAI API."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1"
    ):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        self.model = model
        self.dimension = 1536  # text-embedding-3-small dimension

    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text."""
        if isinstance(text, str):
            text = [text]

        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )

        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if len(embeddings) == 1 else embeddings
```

Key features:
- OpenAI API integration
- Batch embedding support
- Error handling and retries
- Token usage tracking

**File**: `src/rag/health_monitor.py` (~195 lines)

Key classes:
- `BackendStatus(Enum)` - HEALTHY, DEGRADED, UNHEALTHY
- `BackendHealth` - Health metrics dataclass
- `HybridRAGHealthMonitor` - Monitor all backends

Methods:
- `start()` / `stop()` - Async health monitoring
- `_check_supabase_vector()` - Vector search health check
- `_check_supabase_fulltext()` - Fulltext health check
- `_check_falkordb()` - Graph database health check
- `get_status()` - Current health status
- `is_hybrid_available()` - Check all backends available

#### 1.2 Entity Extraction Enhancement

**File**: `src/nlp/entity_extractor.py` (NEW)

Must implement:
```python
class E2IEntityExtractor:
    """Extract E2I domain entities for graph search."""

    def extract(self, query: str) -> ExtractedEntities:
        """Extract brands, regions, KPIs, agents, journey stages."""
        pass
```

Leverages existing:
- `domain_vocabulary_v3.1.0.yaml` - Entity vocabularies
- `src/nlp/e2i_fasttext_trainer.py` - Typo-tolerant matching

---

## Database Changes

### 2. Supabase SQL Migrations

#### 2.1 Hybrid Search Functions

**New File**: `database/memory/011_hybrid_search_functions.sql` (~232 lines)

Creates 2 PostgreSQL functions:

1. **`hybrid_vector_search()`**
   - Searches insight_embeddings, episodic_memories, procedural_memories
   - Uses pgvector cosine similarity (`<=>` operator)
   - Accepts filters (brand, region, date_range)
   - Returns unified result format

2. **`hybrid_fulltext_search()`**
   - Searches causal_paths, agent_activities, triggers
   - Uses PostgreSQL `ts_rank_cd()` for relevance scoring
   - Converts query to `tsquery` with `websearch_to_tsquery()`
   - Returns ranked results

#### 2.2 Database Schema Changes

**Add to existing tables**:

```sql
-- Add full-text search vectors (generated columns)
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
```

**Add indexes**:

```sql
-- GIN indexes for fast full-text search
CREATE INDEX IF NOT EXISTS idx_causal_paths_search
    ON causal_paths USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_agent_activities_search
    ON agent_activities USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_triggers_search
    ON triggers USING GIN(search_vector);

-- HNSW indexes for fast vector search
CREATE INDEX IF NOT EXISTS idx_insight_embeddings_vector
    ON insight_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_episodic_memories_vector
    ON episodic_memories USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_procedural_memories_vector
    ON procedural_memories USING hnsw (trigger_embedding vector_cosine_ops);
```

**Migration Order**:
1. Add columns (if not exist)
2. Create indexes
3. Create RPC functions

### 3. FalkorDB Graph Schema

#### 3.1 Graph Schema Setup

**New File**: `database/memory/002_semantic_graph_schema.cypher` (~80 lines)

**Node Types**:
- `Brand` (Remibrutinib, Fabhalta, Kisqali)
- `Region` (Northeast, Southeast, Midwest, West)
- `KPI` (conversion_rate, TRx, engagement_score, HCP_coverage)
- `Agent` (orchestrator, causal_impact, gap_analyzer, etc.)
- `JourneyStage`, `Trigger`, `Insight`
- `CausalVariable`, `CausalPath`

**Relationship Types**:
- `CAUSES` - Causal effect (with effect_size, confidence)
- `AFFECTS` - Secondary influence
- `CORRELATES` - Correlation (with strength)
- `SOLD_IN` - Brand-region relationships
- `ANALYZES` - Agent-KPI relationships
- `MONITORS` - Agent monitoring relationships

**Sample Data to Seed**:
```cypher
// Brands
MERGE (rem:Brand {name: 'Remibrutinib', therapeutic_area: 'CSU', launch_year: 2024})
MERGE (fab:Brand {name: 'Fabhalta', therapeutic_area: 'PNH', launch_year: 2023})
MERGE (kis:Brand {name: 'Kisqali', therapeutic_area: 'Oncology', launch_year: 2017})

// KPIs with causal relationships
MERGE (eng:KPI {name: 'engagement_score', workstream: 'WS2'})
MERGE (conv:KPI {name: 'conversion_rate', workstream: 'WS2'})
MERGE (trx:KPI {name: 'TRx', workstream: 'WS3'})

// Causal chains
MERGE (eng)-[:CAUSES {effect_size: 0.23, confidence: 0.85}]->(conv)
MERGE (conv)-[:CAUSES {effect_size: 0.31, confidence: 0.91}]->(trx)
```

**Indexes to Create**:
```cypher
CREATE INDEX FOR (b:Brand) ON (b.name);
CREATE INDEX FOR (r:Region) ON (r.name);
CREATE INDEX FOR (k:KPI) ON (k.name);
CREATE INDEX FOR (a:Agent) ON (a.name);
CREATE INDEX FOR (cv:CausalVariable) ON (cv.name);
```

---

## Frontend Integration

### 4. React Knowledge Graph Component

#### 4.1 New Component Structure

**New Directory**: `frontend/src/components/KnowledgeGraph/`

```
frontend/src/components/KnowledgeGraph/
├── KnowledgeGraphViewer.tsx     # Main component (~315 lines)
├── KnowledgeGraphViewer.css     # Styles
├── types.ts                     # GraphNode, GraphEdge interfaces
└── utils.ts                     # Helper functions
```

#### 4.2 Component Implementation

**File**: `frontend/src/components/KnowledgeGraph/KnowledgeGraphViewer.tsx`

Dependencies:
```typescript
import cytoscape, { Core, ElementDefinition } from 'cytoscape';
import dagre from 'cytoscape-dagre';
import popper from 'cytoscape-popper';
import tippy from 'tippy.js';
```

Key features:
- Cytoscape.js graph rendering
- Dagre layout algorithm for DAGs
- Popper tooltips for node details
- Click handlers for node/edge selection
- Configurable color schemes by node type
- Edge styling by relationship type (CAUSES, AFFECTS, etc.)
- Legend for node types
- Selected node details panel

Props:
```typescript
interface Props {
  data: KnowledgeGraphData;        // Nodes and edges
  onNodeClick?: (node: GraphNode) => void;
  onEdgeClick?: (edge: GraphEdge) => void;
  highlightPath?: string[];         // Node IDs to highlight
  layout?: 'dagre' | 'cose' | 'breadthfirst' | 'circle';
}
```

#### 4.3 Dashboard Integration

**Update**: `frontend/E2I_Causal_Dashboard_V3.html`

Already includes:
- ✅ Cytoscape.js CDN links (lines 10-12)
- ✅ Knowledge Graph tab button (line 1205)
- ✅ Knowledge Graph content section (line 2615)
- ✅ Hybrid RAG query flow diagram (line 2743)

**Remaining work**:
1. Add actual graph container div
2. Wire up API calls to `/api/knowledge-graph/causal-subgraph`
3. Handle graph data state
4. Add loading/error states
5. Implement filters (node types, relationship types)

---

## API Endpoints

### 5. Knowledge Graph Routes

**New File**: `src/api/routes/knowledge_graph.py` (~180 lines)

Router prefix: `/api/knowledge-graph`

#### 5.1 Endpoints to Implement

**GET `/causal-subgraph`**
- Fetch subgraph from FalkorDB for visualization
- Query params:
  - `center_node: str` (optional) - Node ID for ego graph
  - `node_types: List[str]` (optional) - Filter by node types
  - `relationship_types: List[str]` (optional) - Filter by relationships
  - `max_depth: int` (default: 2) - Max traversal depth
  - `limit: int` (default: 100) - Max nodes to return
- Returns: `KnowledgeGraphResponse` (nodes, edges, metadata)

**GET `/causal-path`**
- Find causal paths between two nodes
- Query params:
  - `source: str` (required) - Source node ID
  - `target: str` (required) - Target node ID
  - `max_length: int` (default: 4) - Max path length
- Returns: All paths connecting source → target via CAUSES/AFFECTS

#### 5.2 Response Models

```python
class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # Brand, Region, KPI, Agent, etc.
    properties: dict

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str  # CAUSES, AFFECTS, CORRELATES, etc.
    properties: dict  # effect_size, confidence, etc.

class KnowledgeGraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: dict
```

---

## Configuration & Dependencies

### 6. New Dependencies

#### 6.1 Python Packages (requirements.txt)

```txt
# Already present:
supabase>=2.0.0          ✅
pgvector>=0.3.0          ✅
falkordb>=1.0.0          ✅

# Added for RAG:
openai>=1.54.0           ✅ (for embeddings)
ragas>=0.1.0             ✅ (for evaluation)
tiktoken>=0.8.0          ✅ (for token counting)
```

**Note**: Removed `sentence-transformers` dependency - using OpenAI embeddings instead!

#### 6.2 Frontend Packages (npm/yarn)

**For React TypeScript version** (future):
```json
{
  "dependencies": {
    "cytoscape": "^3.28.1",
    "cytoscape-dagre": "^2.5.0",
    "cytoscape-popper": "^2.0.0",
    "tippy.js": "^6.3.7",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/cytoscape": "^3.19.16",
    "@types/cytoscape-dagre": "^2.5.0"
  }
}
```

**For current HTML dashboard** (V3):
- Already using CDN links ✅
- No npm install needed for now

### 7. Configuration Files

#### 7.1 RAG Configuration

**New File**: `config/rag_config.yaml`

```yaml
hybrid_search:
  # Weight distribution (must sum to 1.0)
  weights:
    vector: 0.4
    fulltext: 0.2
    graph: 0.4

  # Per-source result limits
  limits:
    vector_top_k: 20
    fulltext_top_k: 20
    graph_top_k: 20
    final_top_k: 10

  # Timeouts (milliseconds)
  timeouts:
    vector: 2000
    fulltext: 1000
    graph: 3000

  # Reciprocal Rank Fusion constant
  rrf_k: 60

  # Graph boost for connected results
  graph_boost_factor: 1.3

embedding:
  provider: openai
  base_url: "https://api.openai.com/v1"
  model: "text-embedding-3-small"
  dimension: 1536  # text-embedding-3-small output dimension
  batch_size: 100  # Max texts per API call

health_check:
  interval_seconds: 30
  failure_threshold: 3  # Consecutive failures before UNHEALTHY
```

#### 7.2 Environment Variables

**Update**: `.env.example`

```bash
# OpenAI (for Embeddings)
OPENAI_API_KEY=sk-xxxxx

# Embedding Configuration
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL_CHOICE=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# RAG Configuration
RAG_VECTOR_WEIGHT=0.4
RAG_FULLTEXT_WEIGHT=0.2
RAG_GRAPH_WEIGHT=0.4
RAG_FINAL_TOP_K=10
RAG_RRF_K=60
RAG_GRAPH_BOOST=1.3

# RAG Health Monitoring
RAG_HEALTH_CHECK_INTERVAL=30
RAG_FAILURE_THRESHOLD=3
RAG_ENABLE_HEALTH_MONITORING=true

# RAG Evaluation (Ragas)
RAGAS_ENABLE_EVALUATION=true
RAGAS_EVALUATION_INTERVAL=daily
RAGAS_TEST_SET_SIZE=100
RAGAS_METRICS=faithfulness,answer_relevancy,context_precision,context_recall

# FalkorDB (if not already present)
FALKORDB_HOST=localhost
FALKORDB_PORT=6379  # Standard Redis port
FALKORDB_GRAPH_NAME=e2i_knowledge
```

---

## Testing Strategy

### 8. RAG Evaluation with Ragas

#### 8.1 Ragas Framework Integration

**Why Ragas?**
- Evaluates RAG pipelines using LLM-as-judge
- Reference-free metrics (faithfulness, answer relevancy)
- Reference-based metrics (context precision/recall, answer similarity)
- Minimal ground truth labels required
- Production-ready for continuous evaluation

**Key Metrics**:

| Metric | Measures | Target | Priority |
|--------|----------|--------|----------|
| Faithfulness | Answer grounded in context (no hallucinations) | >0.8 | Critical |
| Answer Relevancy | Answer addresses the question | >0.85 | Critical |
| Context Precision | Relevant chunks ranked high | >0.75 | High |
| Context Recall | All relevant info retrieved | >0.8 | High |
| Answer Similarity | Semantic similarity to ground truth | >0.7 | Medium |
| Answer Correctness | Factual correctness | >0.75 | Medium |

**Implementation**: See `docs/rag_evaluation_with_ragas.md` for complete guide.

**File**: `src/rag/evaluation.py` (~400 lines)

Key classes:
- `RAGEvaluator` - Main evaluation class
- `EvaluationResult` - Single evaluation result dataclass

Methods:
- `evaluate_test_set()` - Batch evaluation on golden dataset
- `evaluate_single_query()` - Single query evaluation
- `compare_configurations()` - A/B testing different RAG configs
- `log_to_mlflow()` - MLflow integration
- `log_to_opik()` - Opik observability integration

**Golden Test Dataset**: `tests/evaluation/golden_dataset.json`

Structure:
```json
{
  "test_cases": [
    {
      "id": "tc_001",
      "category": "causal_impact",
      "question": "What caused X?",
      "ground_truth": "Expected answer...",
      "expected_contexts": ["source1", "source2"],
      "difficulty": "medium",
      "requires_graph": true
    }
  ]
}
```

**Test Set Size**: 100 cases covering:
- Causal impact queries (20)
- Gap analysis queries (15)
- KPI lookup queries (20)
- Temporal analysis (15)
- Comparative queries (10)
- Multi-hop reasoning (10)
- Graph traversal (10)

#### 8.2 Automated Evaluation Workflow

**Daily Evaluation**:
```bash
# Run daily at 2 AM UTC
python scripts/run_daily_evaluation.py
```

Features:
- Evaluate full test set (100 cases)
- Log metrics to MLflow
- Check for regressions
- Send alerts if metrics below threshold
- Store results for trending

**CI/CD Integration**: GitHub Actions workflow
```yaml
# .github/workflows/rag_evaluation.yml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:      # Manual trigger
```

**Monitoring**:
- Grafana dashboards for metric trends
- Alerts on regression (>10% drop week-over-week)
- Production query sampling (1-5%)

#### 8.3 A/B Testing Framework

**Compare configurations**:
```python
evaluator.compare_configurations(
    test_dataset_path="tests/evaluation/golden_dataset.json",
    configs=[
        {"vector_weight": 0.4, "graph_weight": 0.4},  # Baseline
        {"vector_weight": 0.3, "graph_weight": 0.5},  # Graph-heavy
    ],
    config_names=["baseline", "graph_heavy"]
)
```

**Evaluation outputs**:
- Aggregate metrics per configuration
- Statistical significance testing
- Winner determination (weighted scoring)
- Per-category breakdown

### 9. Unit & Integration Tests

#### 9.1 Unit Tests

**File**: `tests/unit/test_hybrid_retriever.py`

Test coverage:
- Configuration validation (weights sum to 1.0)
- Entity extraction from queries
- Cypher query building
- RRF scoring algorithm
- Graph boost calculation

**File**: `tests/unit/test_embeddings.py`

Test coverage:
- OpenAI API client initialization
- Single text embedding
- Batch embedding
- Error handling

**File**: `tests/unit/test_ragas_metrics.py`

Test coverage:
- Metric calculation (mock data)
- Threshold validation
- Score normalization

#### 9.2 Integration Tests

**File**: `tests/integration/test_hybrid_retriever.py` (~160 lines from doc)

**CRITICAL TESTS** (validate guarantees):

1. **`test_all_backends_queried`**
   - Verify all 3 backends always called
   - Check vector, fulltext, graph queries executed

2. **`test_results_from_all_sources`**
   - Results include items from all sources
   - Source diversity

3. **`test_graceful_degradation_on_timeout`**
   - If vector times out, fulltext+graph still return
   - Partial results acceptable

4. **`test_require_all_sources_flag`**
   - `require_all_sources=True` fails if any backend down
   - Strict mode validation

5. **`test_rrf_fusion_combines_all_results`**
   - RRF scores normalized correctly
   - No source dominates unfairly

6. **`test_graph_boost_applied`**
   - Graph-connected results score higher
   - Boost factor applied correctly

7. **`test_audit_trail_recorded`**
   - Query stats captured
   - Latency, counts, sources_used tracked

#### 9.3 Health Monitor Tests

**File**: `tests/unit/test_health_monitor.py`

Test coverage:
- Backend health checks (mock responses)
- Status transitions (HEALTHY → DEGRADED → UNHEALTHY)
- Consecutive failure counting
- Circuit breaker behavior

#### 9.4 Evaluation Pipeline Tests

**File**: `tests/evaluation/test_evaluation_pipeline.py`

Test coverage:
- Golden dataset loading
- Batch evaluation execution
- Metric aggregation
- MLflow/Opik logging
- A/B testing workflow

---

## Documentation Updates

### 10. Documentation Changes

#### 10.1 Primary Documentation

**Update**: `docs/e2i_nlv_project_structure_v4.1.md`

Add section:
```markdown
### 2.8 Hybrid RAG Architecture

The E2I platform uses a hybrid retrieval system combining:
- **Vector Search** (Supabase pgvector): Semantic similarity
- **Full-Text Search** (PostgreSQL): Keyword matching
- **Graph Search** (FalkorDB): Relationship traversal

See: `docs/e2i_hybrid_rag_implementation.md` for complete implementation.
```

#### 10.2 Setup Guide

**New File**: `docs/rag_setup_guide.md`

Contents:
1. Prerequisites (Supabase, FalkorDB)
2. Database schema setup
3. FalkorDB initialization
4. Seeding sample data
5. Testing hybrid search
6. Monitoring health checks
7. Troubleshooting

#### 10.3 Ragas Evaluation Guide

**New File**: `docs/rag_evaluation_with_ragas.md` ✅ CREATED

Complete guide covering:
- Ragas framework introduction
- Metric definitions and interpretation
- Test dataset creation (100 golden cases)
- Evaluation pipeline implementation
- MLflow & Opik integration
- Automated evaluation workflows
- A/B testing framework
- Monitoring & alerting setup

#### 10.4 README Updates

**Update**: `README.md`

Add to **Key Features**:
```markdown
- **Hybrid RAG System** (Vector + Keyword + Graph search with RRF fusion)
- **Knowledge Graph Visualization** (Interactive Cytoscape.js UI)
```

Add to **Tech Stack** table:
```markdown
| RAG | Hybrid retrieval (pgvector + tsvector + FalkorDB), sentence-transformers |
```

#### 10.5 Architecture Diagram

**New File**: `docs/diagrams/hybrid_rag_architecture.md`

Include:
- System architecture diagram
- Data flow visualization
- Component interaction diagram
- Database schema relationships

#### 10.6 API Documentation

**Update**: `docs/api/README.md` (or create if missing)

Document:
- `/api/knowledge-graph/causal-subgraph` endpoint
- `/api/knowledge-graph/causal-path` endpoint
- Request/response schemas
- Example queries

---

## Deployment Considerations

### 11. Infrastructure & DevOps

#### 11.1 Docker Configuration

**Update**: `docker-compose.yml` (or create if missing)

Ensure services:
```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"  # FalkorDB uses Redis protocol
    volumes:
      - ./database/memory/002_semantic_graph_schema.cypher:/docker-entrypoint-initdb.d/init.cypher
```

**Note**: Redis and FalkorDB both use port 6379 by default
- Consider using different ports or separate containers
- Update `.env` accordingly

#### 11.2 Makefile Targets

**Update**: `Makefile`

Add targets:
```makefile
.PHONY: rag-setup
rag-setup: ## Initialize RAG system (DB + graph schema)
	@echo "Setting up Hybrid RAG system..."
	@python scripts/setup_rag.py

.PHONY: rag-test
rag-test: ## Run RAG integration tests
	@pytest tests/integration/test_hybrid_retriever.py -v

.PHONY: graph-seed
graph-seed: ## Seed FalkorDB with sample data
	@python scripts/seed_knowledge_graph.py

.PHONY: rag-health
rag-health: ## Check RAG backend health
	@python scripts/check_rag_health.py

.PHONY: rag-eval
rag-eval: ## Run Ragas evaluation on golden test set
	@python scripts/run_daily_evaluation.py

.PHONY: rag-eval-single
rag-eval-single: ## Evaluate a single query
	@python scripts/evaluate_single_query.py "$(QUERY)"
```

#### 11.3 Supabase Deployment

**Migration checklist**:
1. Run `011_hybrid_search_functions.sql` in Supabase SQL editor
2. Verify indexes created successfully
3. Test RPC functions manually
4. Check pgvector extension enabled
5. Monitor query performance

#### 11.4 FalkorDB Deployment

**Initialization steps**:
1. Start FalkorDB container
2. Execute `002_semantic_graph_schema.cypher`
3. Verify indexes created
4. Seed sample data (brands, KPIs, agents)
5. Test Cypher queries
6. Monitor memory usage

---

## Gaps & Recommendations

### 12. Missing Components & Considerations

#### 12.1 Identified Gaps

**1. Embedding Generation Pipeline**
- ❌ No automated embedding generation for new content
- **Recommendation**: Create background job to embed new:
  - Agent outputs → `insight_embeddings`
  - User queries → `episodic_memories`
  - Successful patterns → `procedural_memories`

**2. Graph Population Logic**
- ❌ No code to sync Supabase causal_paths → FalkorDB graph
- **Recommendation**: Create ETL job:
  - Read `causal_paths` table
  - Create/update FalkorDB nodes and relationships
  - Schedule: hourly or event-driven

**3. Entity Extraction Implementation**
- ❌ `E2IEntityExtractor` class not implemented yet
- **Dependencies**:
  - Needs `domain_vocabulary_v3.1.0.yaml` loaded
  - Should use existing fastText normalizer
  - Must extract: brands, regions, KPIs, agents, journey_stages

**4. Frontend State Management**
- ❌ Dashboard V3 is static HTML, not React
- **Recommendation**:
  - Phase 1: Add vanilla JS for API calls
  - Phase 2: Migrate to React TypeScript (future)

**5. Caching Strategy**
- ❌ No caching for frequent queries
- **Recommendation**: Add Redis caching:
  - Cache key: `hash(query + filters)`
  - TTL: 300 seconds
  - Invalidate on data updates

**6. Query History & Analytics**
- ❌ No tracking of which queries use which sources
- **Recommendation**: Log to `episodic_memories`:
  - Query text
  - Sources used (vector/fulltext/graph)
  - Result count per source
  - Total latency
  - User feedback (if any)

**7. Graph Visualization Performance**
- ❌ Large graphs (500+ nodes) may lag in browser
- **Recommendation**:
  - Implement server-side graph simplification
  - Add zoom levels with progressive loading
  - Consider WebGL renderer for large graphs

**8. Security & Access Control**
- ❌ No mention of Row-Level Security (RLS) for graph data
- **Recommendation**: Add RLS policies if multi-tenant

**9. Monitoring & Observability**
- ❌ No integration with Opik/MLflow for RAG metrics
- **Recommendation**: Log to Opik:
  - Query latency by source
  - Result relevance scores
  - Graph boost impact
  - User click-through rates

**10. Documentation - API Examples**
- ❌ No example curl commands or code snippets
- **Recommendation**: Add to `docs/api/`:
  - Python client examples
  - JavaScript fetch examples
  - curl command examples

#### 12.2 Future Enhancements

**1. Hybrid Search Weights Tuning**
- Auto-tune weights based on query performance
- A/B test different weight configurations
- Per-user preference learning

**2. Query Expansion**
- Use LLM to expand queries before search
- Add synonyms from domain vocabulary
- Handle abbreviations (TRx, HCP, etc.)

**3. Multi-Modal Search**
- Add image search for medical imagery
- PDF document ingestion
- Table extraction and search

**4. Real-Time Graph Updates**
- WebSocket connection for live graph changes
- Animated transitions when new nodes added
- Highlight recently updated paths

**5. Collaborative Filtering**
- "Users who searched X also searched Y"
- Query suggestion based on role/brand
- Personalized result ranking

---

## Implementation Checklist

### Phase 1: Backend Foundation (Week 1)

- [ ] Create `src/rag/` directory structure
- [ ] Implement `HybridRetriever` class
- [ ] Implement `HybridRAGHealthMonitor` class
- [ ] Create `E2IEntityExtractor` class
- [ ] Add Supabase SQL functions migration
- [ ] Create FalkorDB Cypher schema
- [ ] Add database indexes
- [ ] Create `config/rag_config.yaml`
- [ ] Update `.env.example`

### Phase 2: API & Frontend (Week 2)

- [ ] Implement knowledge graph API routes
- [ ] Create React `KnowledgeGraphViewer` component (or vanilla JS)
- [ ] Integrate into dashboard V3
- [ ] Add API error handling
- [ ] Implement loading states
- [ ] Add graph filters UI

### Phase 3: Testing & Validation (Week 3)

- [ ] Write unit tests for HybridRetriever
- [ ] Write integration tests (7 critical tests)
- [ ] Test health monitoring
- [ ] Load test with large graphs
- [ ] Validate RRF algorithm
- [ ] Test graceful degradation

### Phase 4: Documentation & Deployment (Week 4)

- [ ] Create RAG setup guide
- [ ] Update project structure docs
- [ ] Write API documentation with examples
- [ ] Create architecture diagrams
- [ ] Update README.md
- [ ] Add Makefile targets
- [ ] Create Docker initialization scripts
- [ ] Write deployment runbook

### Phase 5: Optimization & Polish (Ongoing)

- [ ] Implement query caching
- [ ] Add query analytics logging
- [ ] Optimize large graph rendering
- [ ] Tune search weights
- [ ] Add monitoring dashboards
- [ ] Collect user feedback

---

## Success Criteria

### Functional Requirements

1. ✅ All 3 backends queried on every search (verified by tests)
2. ✅ Results combined using RRF algorithm
3. ✅ Graph relationships boost causally-connected results
4. ✅ Graceful degradation if backend unavailable
5. ✅ Knowledge graph visualization works with 100+ nodes
6. ✅ API response time < 3 seconds for typical queries
7. ✅ Health monitoring detects backend failures within 30 seconds

### Non-Functional Requirements

1. ✅ Code coverage > 80% for RAG module
2. ✅ API documentation with working examples
3. ✅ Setup guide allows new developer to run RAG in < 1 hour
4. ✅ Audit trail captures all query metadata
5. ✅ System handles 10 concurrent queries without degradation

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| FalkorDB graph too large (OOM) | High | Implement pagination, graph pruning |
| Supabase pgvector slow (>2s) | Medium | Add HNSW index, tune parameters |
| Frontend lag with large graphs | Medium | Server-side simplification, WebGL |
| Entity extraction misses entities | Low | Expand vocabulary, use LLM fallback |
| Weight tuning hard to optimize | Low | Add A/B testing framework |

---

## Appendix

### A. File Manifest

**New Files to Create**:
```
src/rag/__init__.py
src/rag/hybrid_retriever.py
src/rag/health_monitor.py
src/rag/embeddings.py                    # NEW: OpenAI embedding client
src/rag/evaluation.py                    # NEW: Ragas evaluation
src/rag/config.py
src/rag/types.py
src/nlp/entity_extractor.py
src/api/routes/knowledge_graph.py
database/memory/011_hybrid_search_functions.sql
database/memory/002_semantic_graph_schema.cypher
config/rag_config.yaml
config/ragas_config.yaml                 # NEW: Ragas configuration
docs/rag_setup_guide.md
docs/rag_implementation_plan.md          # ✅ Created (this file)
docs/rag_evaluation_with_ragas.md       # ✅ Created
tests/unit/test_hybrid_retriever.py
tests/unit/test_embeddings.py            # NEW: OpenAI embedding tests
tests/unit/test_ragas_metrics.py         # NEW: Ragas metric tests
tests/integration/test_hybrid_retriever.py
tests/evaluation/test_evaluation_pipeline.py  # NEW: Evaluation tests
tests/evaluation/golden_dataset.json     # NEW: 100 test cases
tests/unit/test_health_monitor.py
scripts/setup_rag.py
scripts/seed_knowledge_graph.py
scripts/check_rag_health.py
scripts/run_daily_evaluation.py          # NEW: Daily evaluation job
scripts/generate_test_dataset.py         # NEW: Generate golden test set
frontend/src/components/KnowledgeGraph/KnowledgeGraphViewer.tsx
frontend/src/components/KnowledgeGraph/types.ts
.github/workflows/rag_evaluation.yml     # NEW: CI/CD evaluation
```

**Files to Update**:
```
requirements.txt                         # ✅ Updated (openai, ragas)
.env.example                             # ✅ Updated (OpenAI, Ragas configs)
README.md
docs/e2i_nlv_project_structure_v4.1.md
PROJECT_STRUCTURE.txt
Makefile
docker-compose.yml
frontend/E2I_Causal_Dashboard_V3.html
```

### B. Estimated Effort

| Phase | Estimated Hours | Dependencies |
|-------|----------------|--------------|
| Phase 1: Backend | 40-50 hours | Database access, FalkorDB setup |
| Phase 2: API/Frontend | 30-40 hours | Backend complete |
| Phase 3: Testing | 20-25 hours | Phase 1 & 2 complete |
| Phase 4: Docs | 15-20 hours | All phases complete |
| Phase 5: Optimization | 10-15 hours (ongoing) | Production usage data |
| **Total** | **115-150 hours** (~3-4 weeks full-time) | |

---

**Document Version**: 1.0
**Last Updated**: 2025-12-15
**Next Review**: After Phase 1 completion
