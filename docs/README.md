# E2I Agentic Memory System

## Natural Language Visualization + Self-Improving RAG

This package implements the Deep Agentic Memory architecture for the E2I Causal Analytics Dashboard, enabling iterative retrieval (investigation) and asynchronous learning (reflection).

---

## 📁 Package Contents

| File | Purpose |
|------|---------|
| `001_agentic_memory_schema.sql` | Supabase/PostgreSQL schema for Episodic, Procedural memory + cache tables |
| `002_semantic_graph_schema.cypher` | FalkorDB/Cypher schema for semantic memory graph |
| `003_memory_vocabulary.yaml` | Domain vocabulary: entity types, relationships, intents, event categories |
| `004_cognitive_workflow.py` | LangGraph state machine implementing 4-phase cognitive cycle |
| `005_memory_config.yaml` | Technology mappings for local pilot vs AWS production |
| `006_memory_backends.py` | Backend implementations: Redis, Supabase, FalkorDB + Graphity |

---

## 🧠 Architecture Overview

### Tri-Memory Architecture (As Specified)

| Memory Type | Function | Technology |
|-------------|----------|------------|
| **Short-Term (Working)** | Holds current context, scratchpad, and immediate message history | **Redis** (via LangGraph MemorySaver checkpointer) + In-Context Prompting |
| **Episodic (Long-Term)** | Stores experiences: "What did I do?" "What happened yesterday?" | **Supabase** (Postgres + pgvector). Stores User/AI interaction traces. |
| **Semantic (Long-Term)** | Stores facts: "What is the relationship between Project X and User Y?" | **FalkorDB + Graphity**. As interactions happen, an extractor node updates this graph. |
| **Procedural (Long-Term)** | Stores skills: "How did I solve this error last time?" | **Supabase**. Vector store of successful tool call sequences (few-shot examples). |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC MEMORY SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     SHORT-TERM / WORKING MEMORY                       │  │
│  │                     Redis + LangGraph MemorySaver                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │  Session    │  │  Messages   │  │  Evidence   │  │ Scratchpad  │  │  │
│  │  │  State      │  │  (Context)  │  │  Board      │  │             │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         │                          │                          │            │
│         ▼                          ▼                          ▼            │
│  ┌─────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐   │
│  │    EPISODIC     │  │      SEMANTIC        │  │    PROCEDURAL       │   │
│  │    MEMORY       │  │      MEMORY          │  │    MEMORY           │   │
│  │                 │  │                      │  │                     │   │
│  │  Supabase +     │  │  FalkorDB +          │  │  Supabase +         │   │
│  │  pgvector       │  │  Graphity            │  │  pgvector           │   │
│  │                 │  │                      │  │                     │   │
│  │ • User queries  │  │ • Entity nodes       │  │ • Tool sequences    │   │
│  │ • Agent actions │  │ • Relationships      │  │ • Query patterns    │   │
│  │ • Events/traces │  │ • Causal chains      │  │ • Error recoveries  │   │
│  │                 │  │ • Auto-extracted     │  │ • Few-shot examples │   │
│  └─────────────────┘  └──────────────────────┘  └─────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4-Phase Cognitive Cycle

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          COGNITIVE CYCLE                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐   │
│  │  PHASE 1   │───▶│  PHASE 2   │───▶│  PHASE 3   │───▶│  PHASE 4   │   │
│  │ SUMMARIZER │    │INVESTIGATOR│    │   AGENT    │    │ REFLECTOR  │   │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘   │
│                                                                          │
│  • Ingest query     • Multi-hop        • Route to        • Extract      │
│  • Compress old       retrieval          E2I agents       new facts     │
│    context          • Build evidence   • Synthesize      • Learn        │
│  • Extract entities   trail              response          procedures   │
│  • Detect intent    • Decide when      • Generate viz    • Store        │
│                       sufficient                           episodic     │
│                                                                          │
│  ◀─────────────────────────────────────────────────────────────────────▶│
│                         RUNS SYNCHRONOUSLY                               │
│                                                          (ASYNC)        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Local Pilot)

### 1. Prerequisites

- Docker Desktop installed and running
- Python 3.12+ with virtual environment
- Supabase project (cloud-hosted)

### 2. Start Docker Services

The agentic memory system requires two Docker containers:

```bash
# Start Redis for Working Memory (port 6379)
docker run -d --name redis-working-memory -p 6379:6379 redis:latest

# Start FalkorDB for Semantic Memory (port 6380)
docker run -d --name falkordb -p 6380:6379 falkordb/falkordb:latest
```

**With Corporate Proxy** (if behind a proxy like `163.116.128.80:2011`):

Ensure Docker daemon is configured with proxy settings in `/etc/systemd/system/docker.service.d/http-proxy.conf`:
```ini
[Service]
Environment="HTTP_PROXY=http://163.116.128.80:2011"
Environment="HTTPS_PROXY=http://163.116.128.80:2011"
Environment="NO_PROXY=localhost,127.0.0.1"
```

Then reload and restart Docker:
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 3. Install Python Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install required packages
pip install redis falkordb supabase langgraph langgraph-checkpoint-redis \
            graphiti-core anthropic openai pydantic pyyaml

# With proxy
pip install --proxy http://163.116.128.80:2011 <package-name>
```

### 4. Environment Variables

Create/update `.env` file in project root:

```bash
# Redis (Working Memory) - port 6379
REDIS_URL="redis://localhost:6379"

# FalkorDB (Semantic Memory) - port 6380
FALKORDB_HOST="localhost"
FALKORDB_PORT="6380"

# Supabase (Episodic + Procedural Memory)
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_SERVICE_KEY="your-service-key"

# LLM Services
ANTHROPIC_API_KEY="your-anthropic-key"
OPENAI_API_KEY="your-openai-key"  # For embeddings
```

### 5. Apply Database Schema

```bash
# Connect to Supabase SQL Editor and run:
# 001_agentic_memory_schema_v1.3.sql
```

### 6. Verify Installation

Run the health check to verify all backends are operational:

```bash
source venv/bin/activate && python3 << 'EOF'
import os
from dotenv import load_dotenv
load_dotenv()

print("E2I Agentic Memory Health Check")
print("=" * 40)

# Check Redis
try:
    import redis
    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    r.ping()
    print("✓ Redis (Working Memory) - OK")
except Exception as e:
    print(f"✗ Redis failed: {e}")

# Check FalkorDB
try:
    from falkordb import FalkorDB
    db = FalkorDB(host="localhost", port=6380)
    db.select_graph("e2i_semantic")
    print("✓ FalkorDB (Semantic Memory) - OK")
except Exception as e:
    print(f"✗ FalkorDB failed: {e}")

# Check Supabase
try:
    from supabase import create_client
    client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    client.table("episodic_memories").select("memory_id").limit(1).execute()
    print("✓ Supabase (Episodic/Procedural) - OK")
except Exception as e:
    print(f"✗ Supabase failed: {e}")

print("=" * 40)
EOF
```

### 7. Run Cognitive Cycle

```python
from e2i_agentic_memory.cognitive_workflow import run_cognitive_cycle
from e2i_agentic_memory.memory_backends import get_langgraph_checkpointer

# Get the Redis-backed checkpointer for state persistence
checkpointer = get_langgraph_checkpointer()

result = await run_cognitive_cycle(
    user_query="Why did Kisqali adoption increase in the Northeast last quarter?",
    user_id="analyst_001",
    checkpointer=checkpointer  # LangGraph MemorySaver backed by Redis
)

print(f"Response: {result['synthesized_response']}")
print(f"Confidence: {result['confidence_score']:.0%}")
print(f"Facts Learned: {len(result['new_facts'])}")
```

---

## 🔄 Service Management

### Container Status

Check running containers:
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected output:
```
NAMES                  STATUS       PORTS
redis-working-memory   Up X hours   0.0.0.0:6379->6379/tcp
falkordb              Up X hours   0.0.0.0:6380->6379/tcp
```

### Start Services

```bash
# Start Redis (Working Memory)
docker start redis-working-memory

# Start FalkorDB (Semantic Memory)
docker start falkordb

# Start both at once
docker start redis-working-memory falkordb
```

### Stop Services

```bash
# Stop Redis
docker stop redis-working-memory

# Stop FalkorDB
docker stop falkordb

# Stop both at once
docker stop redis-working-memory falkordb
```

### Restart Services

```bash
# Restart Redis
docker restart redis-working-memory

# Restart FalkorDB
docker restart falkordb

# Restart both at once
docker restart redis-working-memory falkordb
```

### If Containers Don't Exist (First Time or After Removal)

```bash
# Create and start Redis container
docker run -d --name redis-working-memory -p 6379:6379 redis:latest

# Create and start FalkorDB container
docker run -d --name falkordb -p 6380:6379 falkordb/falkordb:latest
```

### Remove and Recreate Containers (Clean Slate)

```bash
# Stop and remove existing containers
docker rm -f redis-working-memory falkordb

# Recreate fresh containers
docker run -d --name redis-working-memory -p 6379:6379 redis:latest
docker run -d --name falkordb -p 6380:6379 falkordb/falkordb:latest
```

### View Container Logs

```bash
# Redis logs
docker logs redis-working-memory

# FalkorDB logs
docker logs falkordb

# Follow logs in real-time
docker logs -f falkordb
```

### Test Connectivity

```bash
# Test Redis (Working Memory)
docker exec redis-working-memory redis-cli PING
# Expected: PONG

# Test FalkorDB (Semantic Memory)
docker exec falkordb redis-cli PING
# Expected: PONG

# Verify FalkorDB graph module is loaded
docker exec falkordb redis-cli MODULE LIST | grep -i graph
# Expected: graph (and path to falkordb.so)

# List graphs in FalkorDB
docker exec falkordb redis-cli GRAPH.LIST
# Expected: e2i_semantic (or empty if no graphs created yet)
```

### Auto-Start on System Boot (Optional)

To have containers start automatically when Docker starts:
```bash
docker update --restart unless-stopped redis-working-memory
docker update --restart unless-stopped falkordb
```

### Port Reference

| Service | Container Name | Internal Port | External Port | Purpose |
|---------|---------------|---------------|---------------|---------|
| Redis | redis-working-memory | 6379 | 6379 | Working Memory + LangGraph checkpoints |
| FalkorDB | falkordb | 6379 | 6380 | Semantic Graph (FalkorDB uses Redis protocol) |

---

## 📋 Schema Summary

### Episodic Memory (`episodic_memories`)
Stores event logs with vector embeddings for semantic search.

| Column | Type | Purpose |
|--------|------|---------|
| `memory_id` | UUID | Primary key |
| `event_type` | VARCHAR | user_query, agent_action, feedback |
| `description` | TEXT | Natural language summary |
| `entities` | JSONB | Referenced brands, regions, HCPs |
| `embedding` | vector(1536) | For similarity search |
| `outcome_type` | VARCHAR | success, partial, failure |

### Procedural Memory (`procedural_memories`)
Stores successful tool call sequences for few-shot learning.

| Column | Type | Purpose |
|--------|------|---------|
| `procedure_id` | UUID | Primary key |
| `procedure_name` | VARCHAR | Human-readable name |
| `tool_sequence` | JSONB | Ordered list of tool calls |
| `trigger_embedding` | vector(1536) | For matching similar queries |
| `success_count` | INTEGER | Success tracking |
| `usage_count` | INTEGER | Usage tracking |

### Semantic Memory Cache (`semantic_memory_cache`)
Hot cache of graph triplets for fast retrieval.

| Column | Type | Purpose |
|--------|------|---------|
| `subject_type` | VARCHAR | patient, hcp, brand, etc. |
| `subject_id` | VARCHAR | Entity ID |
| `predicate` | VARCHAR | TREATS, PRESCRIBES, CAUSES |
| `object_type` | VARCHAR | Entity type |
| `object_id` | VARCHAR | Entity ID |
| `confidence` | FLOAT | Triplet confidence |

### Working Memory (`working_memory_sessions`)
Active session state with evidence board.

| Column | Type | Purpose |
|--------|------|---------|
| `session_id` | UUID | Primary key |
| `conversation_summary` | TEXT | Compressed history |
| `evidence_trail` | JSONB | Accumulated evidence |
| `current_phase` | ENUM | summarizer, investigator, agent, reflector |
| `active_entities` | JSONB | Entities in focus |

---

## 🔌 Integration with E2I Agents

The cognitive workflow routes queries to E2I agents based on detected intent:

| Intent | Primary Agent | Secondary Agent | Visualization |
|--------|--------------|-----------------|---------------|
| Causal (why) | `causal_impact` | `explainer` | Sankey |
| Trend (change) | `drift_monitor` | `health_score` | Line |
| Comparison (vs) | `gap_analyzer` | `explainer` | Bar |
| Optimization | `heterogeneous_optimizer` | `resource_optimizer` | Heatmap |
| Experiment | `experiment_designer` | - | Table |

---

## 📊 Key Features

### Redis Working Memory with LangGraph MemorySaver
```python
# The checkpointer provides automatic state persistence across workflow steps
from memory_backends import get_langgraph_checkpointer

checkpointer = get_langgraph_checkpointer()  # Redis-backed

# Working memory holds:
# - Session state (user context, filters)
# - Message history (last N messages for in-context prompting)
# - Evidence board (accumulated during investigation)
# - Scratchpad (intermediate computations)

# All automatically persisted to Redis with TTL
```

### Multi-Hop Investigation
```python
# Phase 2 executes iterative retrieval across memory types:
# Hop 1: Query EPISODIC memory (Supabase) → "What happened before?"
# Hop 2: Query SEMANTIC memory (FalkorDB) → "Who owns that module?"
# Hop 3: Query PROCEDURAL memory (Supabase) → "How did we solve this?"
# Hop 4: Deep traversal if needed

# Evidence board in Redis tracks all findings
```

### Graphity Auto-Extraction
```python
# As interactions happen, Graphity extracts entities and relationships
from memory_backends import get_graphity_extractor

extractor = get_graphity_extractor()

# Automatically updates FalkorDB graph
result = await extractor.extract_and_store(
    text="Dr. Smith prescribed Kisqali for the HR+ breast cancer patient",
    context={"region": "northeast"}
)
# → Creates: (Dr_Smith)-[:PRESCRIBED]->(Kisqali)
# → Creates: (Patient_X)-[:TREATED_BY]->(Dr_Smith)
```

### Asynchronous Learning (Reflector Node)
```python
# Phase 4 runs AFTER response is sent:
# - Evaluates if interaction is worth remembering (selective attention)
# - Extracts new facts as triplets → FalkorDB semantic graph
# - Learns successful tool sequences → Supabase procedural memory
# - Creates episodic trace → Supabase with pgvector
# - Collects DSPy training signals
```

### Evidence Board Pattern
```python
# Evidence accumulates in Redis across hops:
evidence_trail = [
    {"hop": 1, "source": "episodic", "content": "User queried Kisqali trends...", "relevance": 0.89},
    {"hop": 2, "source": "semantic", "content": "(HCP_123)-[:PRESCRIBES]->(Kisqali)", "relevance": 0.92},
    {"hop": 3, "source": "procedural", "content": "Similar query used causal_impact agent...", "relevance": 0.85}
]
```

---

## 🔄 AWS Production Migration

When ready to deploy to AWS, update `005_memory_config.yaml`:

```yaml
environment: "aws_production"  # Change from "local_pilot"

memory_backends:
  working:
    backend: "redis"
    connection: "${ELASTICACHE_REDIS_URL}"
    cluster_mode: true
    langgraph_checkpointer: "RedisSaver"
  
  episodic:
    backend: "aurora_postgresql"
    connection: "${AURORA_ENDPOINT}"
    vector_extension: "pgvector"
  
  semantic:
    backend: "neptune"
    connection: "${NEPTUNE_ENDPOINT}"
    query_language: "opencypher"
    graphity:
      enabled: true
      model: "anthropic.claude-3-5-sonnet-20241022-v2:0"
  
  procedural:
    backend: "aurora_postgresql"
    connection: "${AURORA_ENDPOINT}"
```

### Technology Migration Path

| Memory Type | Local Pilot | AWS Production |
|-------------|-------------|----------------|
| Working | Redis | ElastiCache Redis (Cluster) |
| Episodic | Supabase + pgvector | Aurora PostgreSQL + pgvector |
| Semantic | FalkorDB + Graphity | Neptune + Graphity |
| Procedural | Supabase + pgvector | Aurora PostgreSQL + pgvector |

---

## 📚 Related Documentation

- **E2I PRD V3**: Full product requirements with 11-agent architecture
- **E2I Data Schema**: ML-compliant Supabase schema (18 tables)
- **E2I Project Structure V3**: Complete project directory layout
- **Memory Vocabulary**: Domain entities, relationships, intents

---

## 🧪 Testing

```python
# Test episodic memory search
from memory_backends import search_episodic_memory, get_embedding_service

embedding_service = get_embedding_service()
query_embedding = await embedding_service.embed("Kisqali adoption Northeast")

results = await search_episodic_memory(
    embedding=query_embedding,
    filters={"event_type": "user_query"},
    limit=5
)
```

```python
# Test semantic graph traversal
from memory_backends import query_semantic_graph

results = await query_semantic_graph({
    "start_nodes": ["Kisqali"],
    "relationship_types": ["PRESCRIBES", "CAUSES"],
    "max_depth": 2
})
```

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-02 | Initial release with Tri-Memory architecture |
| 1.1 | 2025-12-02 | Updated to use Redis + FalkorDB + Graphity as specified |
| 1.2 | 2025-12-04 | Fixed FalkorDB container (was using wrong image), added graphiti-core, added service management docs |

---

## 🔧 Service Dependencies

### Local Pilot Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL PILOT SERVICES                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────┐ │
│  │ redis-working-    │  │    falkordb       │  │  Supabase   │ │
│  │ memory            │  │                   │  │  (Cloud)    │ │
│  │ localhost:6379    │  │ localhost:6380    │  │             │ │
│  │                   │  │                   │  │             │ │
│  │ • Working Memory  │  │ • Semantic Graph  │  │ • Episodic  │ │
│  │ • LangGraph       │  │ • Graphity        │  │ • Procedural│ │
│  │   Checkpoints     │  │   Extraction      │  │ • pgvector  │ │
│  │ • Session State   │  │ • Entity/Rels     │  │             │ │
│  └───────────────────┘  └───────────────────┘  └─────────────┘ │
│         │                       │                    │          │
│         └───────────────────────┼────────────────────┘          │
│                                 │                               │
│                    ┌────────────┴────────────┐                  │
│                    │   LangGraph Cognitive   │                  │
│                    │   Workflow Engine       │                  │
│                    │                         │                  │
│                    │   summarizer_node       │                  │
│                    │   investigator_node     │                  │
│                    │   agent_node            │                  │
│                    │   reflector_node        │                  │
│                    └─────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Reference Commands

```bash
# Check status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Start all services
docker start redis-working-memory falkordb

# Stop all services
docker stop redis-working-memory falkordb

# Restart all services
docker restart redis-working-memory falkordb

# Health check
docker exec redis-working-memory redis-cli PING   # Working Memory
docker exec falkordb redis-cli PING               # Semantic Memory
docker exec falkordb redis-cli GRAPH.LIST         # List graphs
```

---

## 🐛 Troubleshooting

### Container won't start
```bash
# Check if container exists but is stopped
docker ps -a | grep -E "redis-working-memory|falkordb"

# If exists, start it
docker start <container-name>

# If doesn't exist, create it
docker run -d --name redis-working-memory -p 6379:6379 redis:latest
docker run -d --name falkordb -p 6380:6379 falkordb/falkordb:latest
```

### Port already in use
```bash
# Find what's using the port
sudo lsof -i :6379
sudo lsof -i :6380

# Kill the process or use different ports
docker run -d --name redis-working-memory -p 6381:6379 redis:latest
# Then update REDIS_URL in .env
```

### FalkorDB graph module not loaded
```bash
# Verify module is loaded
docker exec falkordb redis-cli MODULE LIST | grep graph

# If not showing, recreate container with correct image
docker rm -f falkordb
docker run -d --name falkordb -p 6380:6379 falkordb/falkordb:latest
```

### Connection refused errors
```bash
# Check Docker is running
docker info

# Check containers are running
docker ps

# Start containers if stopped
docker start redis-working-memory falkordb
```

---

*Built for E2I Causal Analytics Dashboard v3.0*
