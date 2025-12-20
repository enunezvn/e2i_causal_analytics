# RAG Implementation Quick Reference

**Quick lookup guide for E2I RAG implementation**

---

## 🚀 Quick Start

### Before You Begin
```bash
# Verify environment
python -c "from openai import OpenAI; client = OpenAI(); print('✅ OpenAI OK')"
python -c "from supabase import create_client; import os; client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY')); print('✅ Supabase OK')"
python -c "from redis import Redis; r = Redis(port=6380); print('✅ FalkorDB OK')"

# Install dependencies
pip install openai>=1.54.0 ragas>=0.1.0

# Start services
docker-compose up -d  # Redis + FalkorDB
```

### Current Status
- **Checkpoint**: See `RAG_CHECKPOINT_STATUS.md`
- **Main Plan**: See `docs/RAG_LLM_IMPLEMENTATION_PLAN.md`
- **Progress**: See `RAG_CHECKPOINT_STATUS.md` Quick Status Summary

---

## 📁 File Locations

### Core Implementation
```
src/rag/
├── __init__.py
├── types.py                   # Enums, dataclasses
├── config.py                  # Configuration
├── embeddings.py              # OpenAI client
├── hybrid_retriever.py        # Main retriever
├── health_monitor.py          # Health checks
└── evaluation.py              # Ragas evaluator

src/nlp/
└── entity_extractor.py        # E2I entity extraction
```

### Database
```
database/memory/
├── 011_hybrid_search_functions.sql   # Supabase RPC
└── 002_semantic_graph_schema.cypher  # FalkorDB schema
```

### Tests
```
tests/
├── unit/
│   ├── test_embeddings.py
│   ├── test_hybrid_retriever.py
│   ├── test_health_monitor.py
│   └── test_rag_evaluator.py
├── integration/
│   └── test_hybrid_retriever_integration.py
└── evaluation/
    ├── golden_dataset.json
    └── test_evaluation_pipeline.py
```

### Configuration
```
config/
├── rag_config.yaml            # RAG settings
└── ragas_config.yaml          # Evaluation settings
```

---

## 🔧 Key Code Snippets

### OpenAI Embedding Client
```python
from src.rag.embeddings import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# Single embedding
embedding = client.encode("What is the adoption rate for Kisqali?")
# Returns: List[float] with 1536 dimensions

# Batch embedding
embeddings = client.encode(["query1", "query2", "query3"])
# Returns: List[List[float]]
```

### Hybrid Retriever
```python
from src.rag.hybrid_retriever import HybridRetriever, HybridSearchConfig

config = HybridSearchConfig(
    vector_weight=0.4,
    fulltext_weight=0.2,
    graph_weight=0.4,
    graph_boost_factor=1.3,
    top_k=10
)

retriever = HybridRetriever(
    config=config,
    supabase_client=supabase,
    falkordb_client=falkordb,
    embedding_client=embedding_client
)

# Search
results = retriever.search("What caused the drop in Kisqali NPS?")
# Returns: List[RetrievalResult]
```

### Reciprocal Rank Fusion
```python
def reciprocal_rank_fusion(
    results_by_source: Dict[str, List],
    k: int = 60
) -> List[Tuple[str, float]]:
    """Combine rankings using RRF."""
    scores = defaultdict(float)

    for source, results in results_by_source.items():
        for rank, result in enumerate(results, start=1):
            scores[result.id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### RAG Evaluator
```python
from src.rag.evaluation import RAGEvaluator

evaluator = RAGEvaluator(mlflow_client=mlflow_client)

# Evaluate single query
metrics = evaluator.evaluate_query(
    query="What is Remibrutinib adoption?",
    response="Adoption is 23% in US...",
    context=["chunk1", "chunk2"],
    ground_truth="Expected answer..."  # Optional
)
# Returns: {faithfulness: 0.85, answer_relevancy: 0.90, ...}

# Batch evaluation
results = evaluator.evaluate_batch(test_cases)
```

---

## 🗄️ Database Schemas

### Supabase RPC Functions

**Vector + Fulltext Search**:
```sql
CREATE OR REPLACE FUNCTION hybrid_search_vector_fulltext(
    query_text TEXT,
    query_embedding VECTOR(1536),
    top_k INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    vector_score FLOAT,
    fulltext_score FLOAT,
    combined_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT id, content, metadata,
               1 - (embedding <=> query_embedding) AS score
        FROM episodic_memories
        ORDER BY embedding <=> query_embedding
        LIMIT top_k
    ),
    fulltext_results AS (
        SELECT id, content, metadata,
               ts_rank(to_tsvector('english', content),
                      plainto_tsquery('english', query_text)) AS score
        FROM episodic_memories
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', query_text)
        ORDER BY score DESC
        LIMIT top_k
    )
    SELECT ... -- RRF fusion logic
END;
$$;
```

### FalkorDB Cypher Schema

**Nodes**:
```cypher
// Brand
CREATE (:Brand {name: 'Remibrutinib', category: 'BTK inhibitor', indication: 'CSU'})

// HCP
CREATE (:HCP {id: 'HCP001', specialty: 'Dermatology', tier: 'A'})

// Patient
CREATE (:Patient {id: 'PT001', diagnosis: 'CSU', journey_stage: 'Awareness'})

// KPI
CREATE (:KPI {name: 'Adoption Rate', category: 'engagement', unit: 'percentage'})

// Event
CREATE (:Event {type: 'rep_visit', timestamp: '2024-03-15', outcome: 'positive'})
```

**Relationships**:
```cypher
// Treatment
CREATE (:Brand)-[:TREATS]->(:Indication)

// HCP-Patient
CREATE (:HCP)-[:TREATS]->(:Patient)

// Causal
CREATE (:Event)-[:CAUSED_BY]->(:Event)
CREATE (:Event)-[:IMPACTS]->(:KPI)

// Graph traversal example
MATCH path = (e1:Event)-[:CAUSED_BY*1..3]->(e2:Event)-[:IMPACTS]->(k:KPI)
WHERE e1.type = 'rep_visit' AND k.name = 'Adoption Rate'
RETURN path
```

---

## 🧪 Testing Commands

### Unit Tests
```bash
# All unit tests
pytest tests/unit/ -v

# Specific module
pytest tests/unit/test_embeddings.py -v

# With coverage
pytest tests/unit/ -v --cov=src/rag --cov-report=html

# Watch mode (re-run on changes)
pytest-watch tests/unit/
```

### Integration Tests
```bash
# All integration tests
pytest tests/integration/ -v

# Critical tests only
pytest tests/integration/test_hybrid_retriever_integration.py -v -k "test_all_backends"

# With markers
pytest -m "critical" -v
```

### Evaluation Tests
```bash
# Run evaluation pipeline
python scripts/run_daily_evaluation.py

# Generate test dataset
python scripts/generate_test_dataset.py --size 100

# Check quality gates
python scripts/check_quality_gates.py
```

---

## 📊 Configuration Files

### `.env` Variables
```bash
# OpenAI
OPENAI_API_KEY=sk-...
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL_CHOICE=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Ragas
RAGAS_ENABLE_EVALUATION=true
RAGAS_EVALUATION_INTERVAL=daily
RAGAS_TEST_SET_SIZE=100
RAGAS_METRICS=faithfulness,answer_relevancy,context_precision,context_recall

# Supabase (already configured)
SUPABASE_URL=...
SUPABASE_KEY=...

# FalkorDB
FALKORDB_HOST=localhost
FALKORDB_PORT=6380
```

### `config/rag_config.yaml`
```yaml
hybrid_search:
  vector_weight: 0.4
  fulltext_weight: 0.2
  graph_weight: 0.4
  graph_boost_factor: 1.3
  top_k: 10
  rrf_k: 60
  require_all_sources: false
  timeout_seconds: 5

health_monitoring:
  check_interval_seconds: 30
  degraded_latency_threshold: 2.0
  unhealthy_failure_count: 3
  circuit_breaker_timeout: 60

embedding:
  model: "text-embedding-3-small"
  dimension: 1536
  batch_size: 100
  max_retries: 3
  retry_delay: 1.0
```

### `config/ragas_config.yaml`
```yaml
evaluation:
  metrics:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall

  thresholds:
    faithfulness: 0.80
    answer_relevancy: 0.85
    context_precision: 0.75
    context_recall: 0.80

  alerting:
    degradation_threshold: 0.10  # Alert if metrics drop >10%
    consecutive_failures: 2       # Alert after 2 days of regression
    channels:
      - slack
      - email

  test_set:
    size: 100
    categories:
      causal_impact: 20
      gap_analysis: 15
      kpi_lookup: 20
      temporal_analysis: 15
      comparative: 10
      multi_hop: 10
      graph_traversal: 10
```

---

## 📈 Ragas Metrics

### Metric Definitions

| Metric | Threshold | Description | How to Improve |
|--------|-----------|-------------|----------------|
| **Faithfulness** | >0.8 | Answer grounded in context, no hallucinations | Better context retrieval, reduce hallucination |
| **Answer Relevancy** | >0.85 | Answer addresses the question | Improve query understanding |
| **Context Precision** | >0.75 | Relevant chunks ranked high | Tune RRF weights, improve ranking |
| **Context Recall** | >0.8 | All relevant info retrieved | Increase top_k, improve search |

### Interpreting Results

**Good Performance**:
```
Faithfulness: 0.87 ✅
Answer Relevancy: 0.91 ✅
Context Precision: 0.79 ✅
Context Recall: 0.84 ✅
```

**Needs Improvement**:
```
Faithfulness: 0.72 ⚠️  → LLM hallucinating, improve context
Answer Relevancy: 0.68 ⚠️  → Answer off-topic, improve routing
Context Precision: 0.61 ❌ → Too many irrelevant results, tune weights
Context Recall: 0.55 ❌    → Missing relevant info, increase top_k
```

---

## 🔍 Debugging

### Common Issues

#### Issue: OpenAI Rate Limit
```python
# Error: openai.RateLimitError
# Solution: Add retry with exponential backoff

import time
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def get_embedding(text):
    return client.encode(text)
```

#### Issue: Supabase Timeout
```python
# Error: TimeoutError after 5s
# Solution: Increase timeout or optimize query

# Check slow queries
SELECT * FROM pg_stat_statements
WHERE query LIKE '%episodic_memories%'
ORDER BY mean_exec_time DESC;

# Add index
CREATE INDEX idx_embedding_hnsw
ON episodic_memories
USING hnsw (embedding vector_cosine_ops);
```

#### Issue: FalkorDB Connection Failed
```bash
# Error: redis.exceptions.ConnectionError
# Check if FalkorDB is running
docker ps | grep falkordb

# Restart FalkorDB
docker-compose restart falkordb

# Test connection
redis-cli -p 6380 PING
```

#### Issue: RRF Returns Empty Results
```python
# Debug RRF fusion
results_by_source = {
    'vector': vector_results,      # Check: non-empty?
    'fulltext': fulltext_results,  # Check: non-empty?
    'graph': graph_results         # Check: non-empty?
}

# Log results from each source
for source, results in results_by_source.items():
    print(f"{source}: {len(results)} results")

# Check RRF scores
fused_results = reciprocal_rank_fusion(results_by_source)
for doc_id, score in fused_results[:5]:
    print(f"{doc_id}: {score:.4f}")
```

### Logging

**Enable Debug Logging**:
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log RAG operations
logger = logging.getLogger('src.rag')
logger.setLevel(logging.DEBUG)
```

**Trace Retrieval**:
```python
import opik

@opik.track()
def hybrid_search(query):
    with opik.span("embedding_generation"):
        embedding = client.encode(query)

    with opik.span("vector_search"):
        vector_results = search_vector(embedding)

    # View traces in Opik dashboard
```

---

## 🎯 Performance Targets

### Latency SLAs

| Operation | P50 | P95 | P99 | Timeout |
|-----------|-----|-----|-----|---------|
| OpenAI Embedding | <100ms | <200ms | <300ms | 5s |
| Vector Search | <200ms | <500ms | <1s | 3s |
| Fulltext Search | <100ms | <300ms | <500ms | 2s |
| Graph Search | <300ms | <800ms | <1.5s | 3s |
| **Total RAG Pipeline** | **<1s** | **<2s** | **<3s** | **5s** |

### Quality Targets

| Metric | Target | Minimum | Current | Status |
|--------|--------|---------|---------|--------|
| Faithfulness | >0.85 | 0.80 | - | ⏳ |
| Answer Relevancy | >0.90 | 0.85 | - | ⏳ |
| Context Precision | >0.80 | 0.75 | - | ⏳ |
| Context Recall | >0.85 | 0.80 | - | ⏳ |

---

## 🚨 Troubleshooting Checklist

### Before Starting Implementation
- [ ] OpenAI API key in `.env` and valid
- [ ] Supabase connection works
- [ ] FalkorDB running on port 6380
- [ ] Redis running on port 6379
- [ ] Python 3.12+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] PostgreSQL pgvector extension enabled

### When Tests Fail
- [ ] Check environment variables loaded
- [ ] Verify database connections
- [ ] Check test fixtures properly set up
- [ ] Review test logs for specific errors
- [ ] Run tests in isolation (`pytest tests/unit/test_embeddings.py::test_single_embedding`)
- [ ] Clear any cached data

### When Retrieval Quality Poor
- [ ] Check if all 3 backends returning results
- [ ] Verify RRF fusion working correctly
- [ ] Test each backend individually
- [ ] Review entity extraction accuracy
- [ ] Check graph boost being applied
- [ ] Verify embeddings have correct dimension (1536)
- [ ] Test with known queries from golden dataset

### When Performance Slow
- [ ] Check database indexes exist
- [ ] Review slow query logs
- [ ] Verify caching enabled
- [ ] Check batch sizes not too large
- [ ] Monitor API rate limits (OpenAI)
- [ ] Review Opik traces for bottlenecks

---

## 📞 Getting Help

### Documentation
- **Main Plan**: `docs/RAG_LLM_IMPLEMENTATION_PLAN.md`
- **Implementation Details**: `docs/rag_implementation_plan.md`
- **Evaluation Guide**: `docs/rag_evaluation_with_ragas.md`
- **Checkpoint Status**: `RAG_CHECKPOINT_STATUS.md`
- **This File**: `docs/RAG_QUICK_REFERENCE.md`

### External Resources
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- Ragas Docs: https://docs.ragas.io/
- Supabase pgvector: https://supabase.com/docs/guides/ai/vector-embeddings
- FalkorDB: https://docs.falkordb.com/
- Cytoscape.js: https://js.cytoscape.org/

### Commands Reference
```bash
# See current status
cat RAG_CHECKPOINT_STATUS.md | grep "Current Checkpoint"

# Run tests
make test-rag  # If Makefile target exists
pytest tests/unit/ -v

# Check coverage
pytest --cov=src/rag --cov-report=html
open htmlcov/index.html

# Run evaluation
python scripts/run_daily_evaluation.py

# Check health
curl http://localhost:8000/api/v1/rag/health
```

---

## ⚡ Quick Command Reference

```bash
# Environment
python -c "import os; print('OpenAI:', 'OK' if os.getenv('OPENAI_API_KEY') else 'MISSING')"

# Testing
pytest tests/unit/test_embeddings.py -v
pytest tests/integration/ -v -k "critical"
pytest --cov=src/rag --cov-report=term-missing

# Database
psql $SUPABASE_URL -c "SELECT COUNT(*) FROM episodic_memories;"
redis-cli -p 6380 GRAPH.QUERY knowledge_graph "MATCH (n) RETURN count(n)"

# API
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Kisqali adoption?", "top_k": 10}'

# Monitoring
tail -f logs/rag.log
python scripts/check_quality_gates.py
```

---

**Last Updated**: 2025-12-17
**Version**: 1.0
**Maintainer**: E2I Development Team
