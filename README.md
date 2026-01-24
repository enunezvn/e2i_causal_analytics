# E2I Causal Analytics

**Healthcare Engagement Intelligence Platform**
Multi-Agent Causal Analytics for Pharmaceutical Drug Adoption Analysis

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version 4.2.1](https://img.shields.io/badge/version-4.2.1-green.svg)]()
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)]()

## Overview

E2I Causal Analytics is a sophisticated 21-agent, 6-tier agentic system designed for pharmaceutical companies to understand and optimize drug adoption through causal inference and natural language querying.

### Key Features

- **21 AI Agents** across 6 tiers (ML Foundation, Coordination, Causal Analytics, Monitoring, Predictions, Self-Improvement)
- **Tri-Memory Architecture** (Working, Episodic, Procedural, Semantic)
- **Causal Validation** with 5 DoWhy refutation tests
- **MLOps Integration** (MLflow, Opik, Feast, Great Expectations, Optuna, SHAP, BentoML)
- **Real-Time Model Interpretability** (v4.1) - SHAP explanations in 50-500ms via REST API
- **Digital Twin Engine** ⭐ NEW v4.2 - A/B test pre-screening with ML-based simulations
- **Tool Composer** ⭐ NEW v4.2 - Multi-faceted query decomposition & dynamic tool orchestration
- **Natural Language Interface** with typo-tolerant query processing
- **37 Database Tables** for comprehensive data management (including 9 new v4.2 tables)
- **Hybrid RAG System** with vector + full-text + graph search

### Analyzed Brands

- **Remibrutinib** - BTK inhibitor for chronic spontaneous urticaria (CSU)
- **Fabhalta** - Factor B inhibitor for paroxysmal nocturnal hemoglobinuria (PNH)
- **Kisqali** - CDK4/6 inhibitor (ribociclib) for breast cancer

## Architecture

### 6-Tier Agent System

**TIER 0: ML FOUNDATION** (8 agents)
- scope_definer, cohort_constructor, data_preparer, feature_analyzer, model_selector, model_trainer, model_deployer, observability_connector

**TIER 1: COORDINATION** (2 agents) ⭐ Enhanced v4.2
- orchestrator (multi-agent routing & synthesis with 4-stage classifier)
- tool_composer ⭐ NEW v4.2 (multi-faceted query decomposition & tool orchestration)

**TIER 2: CAUSAL ANALYTICS** (3 agents) ⭐ Core
- causal_impact (effect estimation + 5 refutation tests)
- gap_analyzer (ROI opportunity identification)
- heterogeneous_optimizer (treatment effect heterogeneity)

**TIER 3: MONITORING** (3 agents) ⭐ Enhanced v4.2
- drift_monitor, experiment_designer (with Digital Twin pre-screening), health_score

**TIER 4: ML PREDICTIONS** (2 agents)
- prediction_synthesizer, resource_optimizer

**TIER 5: SELF-IMPROVEMENT** (2 agents)
- explainer, feedback_learner

## Project Structure

```
e2i_causal_analytics/
├── config/                    # YAML configurations (8 files)
│   ├── agent_config.yaml      # 21-agent definitions
│   ├── domain_vocabulary_v3.1.0.yaml
│   ├── kpi_definitions.yaml   # 46+ KPIs
│   └── ...
│
├── database/                  # SQL schemas (37 tables) ⭐ Enhanced v4.2
│   ├── core/                  # 8 core data tables
│   ├── ml/                    # 17 ML tables (8 foundation + 2 causal + 1 SHAP + 3 digital twin + 3 tool composer)
│   ├── memory/                # 4 memory tables + FalkorDB schema + 3 tool composer memory tables
│   ├── audit/                 # Audit trail
│   └── causal/                # 2 causal validation tables
│
├── data/
│   ├── synthetic/             # ~200 patients, ~50 HCPs (18 JSON files)
│   └── training/              # fastText corpus
│
├── src/                       # Main source code ⭐ Enhanced v4.2
│   ├── nlp/                   # Query processing, entity extraction
│   ├── agents/                # 21 agent implementations
│   │   ├── orchestrator/      # Tier 1 coordination agent
│   │   │   ├── classifier/    # ⭐ NEW v4.2: 4-stage query classification
│   │   │   ├── router.py      # ⭐ NEW v4.2: Enhanced routing logic
│   │   │   └── tools/         # Orchestrator tools
│   │   │       └── explain_tool.py  # SHAP chat integration
│   │   ├── tool_composer/     # ⭐ NEW v4.2: Multi-faceted query handler
│   │   │   ├── decomposer.py, planner.py, executor.py, synthesizer.py
│   │   │   └── models/        # Composition data models
│   │   └── experiment_designer/  # ⭐ Enhanced v4.2 with Digital Twin
│   │       └── tools/         # Includes digital twin simulation tools
│   ├── digital_twin/          # ⭐ NEW v4.2: Digital twin engine
│   │   ├── twin_generator.py, simulation_engine.py
│   │   └── models/            # Twin & simulation schemas
│   ├── tool_registry/         # ⭐ NEW v4.2: Tool discovery & management
│   ├── feature_store/         # ⭐ NEW v4.2: Lightweight feature store
│   │   ├── client.py          # Main FeatureStoreClient
│   │   ├── retrieval.py       # Feature retrieval with caching
│   │   ├── writer.py          # Feature writing & batch operations
│   │   └── models.py          # Feature store data models
│   ├── memory/                # Tri-memory backends
│   ├── causal/                # Causal inference engine
│   ├── ml/                    # ML operations & data management
│   ├── mlops/                 # MLOps components
│   │   └── shap_explainer_realtime.py  # Real-time SHAP engine
│   ├── api/                   # FastAPI endpoints
│   │   └── routes/
│   │       └── explain.py     # 5 SHAP API endpoints
│   └── utils/                 # Shared utilities
│
├── tests/                     # Unit & integration tests
├── scripts/                   # Utility scripts
├── frontend/                  # Dashboard UI mockups
├── docs/                      # Comprehensive documentation
└── docker/                    # Container configurations
```

## What's New in v4.2.0

### Lightweight Feature Store 🏪
Integrated feature store leveraging existing infrastructure:
- **Architecture**: Supabase (offline) + Redis (online) + MLflow (tracking)
- **Zero Overhead**: No additional services required
- **Online Serving**: <1ms cache hits, <50ms cache misses via Redis
- **Offline Storage**: PostgreSQL time-series with freshness monitoring
- **Batch Operations**: Efficient bulk feature writes with cache invalidation
- **MLflow Integration**: Automatic feature definition tracking
- **3 New Tables**: `feature_groups`, `features`, `feature_values`
- **E2I Use Cases**: HCP targeting, brand performance, causal features
- **Documentation**: Complete guide + quick start tutorial

### Digital Twin Engine 🔮
Pre-screen experiments before real-world deployment with ML-based digital twins:
- **Twin Generation**: Create ML models that simulate HCP, patient, or territory behavior
- **Intervention Simulation**: Test marketing interventions on 10,000+ digital twins in seconds
- **Fidelity Tracking**: Validate twin predictions against real A/B test outcomes
- **Smart Recommendations**: Get deploy/skip/refine decisions based on simulated ATE
- **3 New Tables**: `digital_twin_models`, `twin_simulations`, `twin_fidelity_tracking`
- **MLflow Integration**: Version and track twin models with full lineage

### Tool Composer 🛠️
Handle complex, multi-faceted queries with dynamic tool orchestration:
- **4-Stage Classifier**: Intent features → Domain mapping → Dependencies → Pattern selection
- **4-Phase Pipeline**: Decompose → Plan → Execute → Synthesize
- **Tool Registry**: Discover and compose from 14+ tools across all agents
- **Parallel Execution**: Run independent tool steps concurrently with dependency management
- **6 New Tables**: Tool registry, dependencies, composition episodes, classification logs, metrics, execution steps
- **Routing Patterns**: `SINGLE_AGENT`, `PARALLEL_DELEGATION`, `TOOL_COMPOSER`, `CLARIFICATION_NEEDED`

### Configuration Updates
- **Domain Vocabulary v4.2.0**: Enhanced with Tool Composer ENUMs and routing patterns
- **Enhanced Orchestrator**: 4-stage classifier for intelligent query routing
- **Enhanced Experiment Designer**: Digital twin pre-screening tools integrated

### Documentation
- `docs/digital_twin_component_update_list.md` - Digital twin implementation guide
- `docs/digital_twin_implementation.html` - Interactive digital twin guide
- `docs/tool_composer_component_update_list.md` - Tool composer implementation guide
- `docs/tool_composer_architecture.html` - Tool composer architecture

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (for Redis, FalkorDB, and Opik)
- Supabase account
- Anthropic API key
- Opik (optional, for LLM/Agent observability)

### Installation

1. **Clone and setup environment**
   ```bash
   cd e2i_causal_analytics
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   make install
   # Or manually: pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URLs
   ```

4. **Start Docker services**
   ```bash
   make docker-up
   # Starts Redis (port 6379) and FalkorDB (port 6380)
   ```

5. **Setup Opik Observability** (Optional)
   ```bash
   # Clone and start Opik locally
   git clone https://github.com/comet-ml/opik.git /tmp/opik
   cd /tmp/opik/deployment/docker-compose
   docker compose up -d

   # Wait for services to be healthy (~2-3 minutes)
   docker compose ps

   # Access Opik dashboard at http://localhost:5173
   ```

   Configure Python SDK:
   ```bash
   # For local Opik instance (no API key needed)
   export OPIK_URL_OVERRIDE=http://localhost:5173/api
   export OPIK_WORKSPACE=default
   ```

   Or create `~/.opik.config`:
   ```ini
   [opik]
   url_override = http://localhost:5173/api/
   workspace = default
   ```

6. **Initialize database**
   ```bash
   # Apply schemas in order:
   # 1. database/core/e2i_ml_complete_v3_schema.sql
   # 2. database/ml/mlops_tables.sql
   # 3. database/ml/010_causal_validation_tables.sql
   # 4. database/ml/011_realtime_shap_audit.sql
   # 5. database/ml/012_digital_twin_tables.sql (v4.2)
   # 6. database/ml/013_tool_composer_tables.sql (v4.2)
   # 7. database/migrations/004_create_feature_store_schema.sql (v4.2) ⭐ NEW
   # 8. database/memory/ (RAG functions)
   # 9. database/audit/

   # For feature store migration, use:
   python scripts/run_migration.py database/migrations/004_create_feature_store_schema.sql
   ```

7. **Generate synthetic data**
   ```bash
   make data-generate
   # Or: python src/ml/data_generator.py
   ```

8. **Start API server** (includes Real-Time SHAP endpoints)
   ```bash
   # See docs/realtime_shap_api.md for SHAP setup details
   uvicorn main:app --reload
   # Test SHAP: curl http://localhost:8000/api/v1/explain/health
   ```

## Development

### Available Commands

```bash
make help           # Show all available commands
make test           # Run test suite
make lint           # Check code quality
make format         # Format code with black
make clean          # Clean build artifacts
```

### Running Tests

```bash
pytest tests/ -v --cov=src
```

## Key Components

### Real-Time Model Interpretability ⭐ NEW v4.1

**SHAP Explanations API**
- 5 REST endpoints (/predict, /batch, /history, /models, /health)
- 50-500ms latency (TreeExplainer for tree models, KernelExplainer for others)
- Natural language chat integration ("Why is patient X flagged?")
- Compliance audit trail with row-level security
- Visualization support (waterfall charts, force plots, bar charts)

**Performance SLAs**
- P50: <100ms (tree models), P95: <300ms, P99: <500ms
- Explainer caching (1-hour TTL), thread pool optimization

**Use Cases**
- Field rep conversations: "Why is this patient recommended?"
- Regulatory audit: Complete explanation history
- Model debugging: Compare predictions over time
- A/B testing: Contextual explanation depth experiments

**Integration**
```python
# Import API routes
from src.api.routes.explain import router as explain_router
app.include_router(explain_router, prefix="/api/v1")

# Import chat tools
from src.agents.orchestrator.tools.explain_tool import ExplainIntentHandler

# Import SHAP engine
from src.mlops.shap_explainer_realtime import RealTimeSHAPExplainer
```

See `docs/realtime_shap_api.md` for complete documentation and integration guide.

### Lightweight Feature Store ⭐ NEW v4.2

**E2I Feature Store**
- Integrated solution using Supabase (offline) + Redis (online) + MLflow (tracking)
- Sub-millisecond online serving with automatic cache invalidation
- Feature freshness monitoring and time-series storage
- Multi-entity support with flexible schema
- Zero additional infrastructure (leverages existing services)

**Core Capabilities**
- Feature Groups: Logical organization of related features
- Online Serving: Redis-cached retrieval (<1ms cache hits, <50ms misses)
- Offline Storage: PostgreSQL time-series with freshness tracking
- Batch Operations: Efficient bulk feature writes
- MLflow Integration: Automatic feature definition tracking

**Quick Start**
```python
from src.feature_store import FeatureStoreClient

# Initialize
fs = FeatureStoreClient(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_ANON_KEY"),
    redis_url="redis://localhost:6379",
    mlflow_tracking_uri="http://localhost:5000"
)

# Get features for an HCP
features = fs.get_entity_features(
    entity_values={"hcp_id": "HCP123"},
    feature_group="hcp_demographics",
    use_cache=True
)
```

**E2I Use Cases**
- HCP targeting features (specialty, years_in_practice, practice_size)
- Brand performance metrics (NRx, market share, growth rates)
- Causal inference features (ATE, CATE by segment)
- Agent integration (Gap Analyzer, Prediction Synthesizer, etc.)

See `docs/FEATURE_STORE.md` for complete documentation and `docs/FEATURE_STORE_QUICKSTART.md` for setup guide.

### Opik LLM/Agent Observability

**Production-Grade Observability**
Full observability for LLM calls and agent workflows with the Opik integration:
- **Trace Visualization**: Hierarchical span view with timing, inputs, outputs
- **Multi-Backend**: Opik dashboard (primary) + Supabase (persistence)
- **Circuit Breaker**: Graceful degradation when Opik is unavailable
- **Batch Processing**: Efficient span buffering (100 spans or 5 seconds)
- **Self-Monitoring**: Health spans, latency tracking, alert thresholds
- **Metrics Caching**: Redis (primary) + in-memory fallback with TTL

**Architecture**
```
Agent Workflow → OpikConnector → Opik Dashboard
                     ↓
              BatchProcessor → ObservabilitySpanRepository → Supabase
                     ↓
              MetricsCache (Redis/Memory) ← SelfMonitor
```

**Quick Start**
```python
from src.mlops.opik_connector import OpikConnector
from opik import track

# Initialize connector (auto-configures from ~/.opik.config)
connector = OpikConnector()

# Use @track decorator for automatic tracing
@track(project_name='e2i-analytics')
def analyze_kpi(kpi_name: str, value: float) -> dict:
    # Your analysis logic
    return {'kpi': kpi_name, 'status': 'healthy' if value > 50 else 'warning'}

# Or use context manager for custom spans
async with connector.trace_agent("gap_analyzer", metadata={"brand": "Kisqali"}):
    result = await run_gap_analysis()
```

**Production Features**
- Circuit breaker: Opens after 5 failures, recovers after 30 seconds
- Batch flush: Every 100 spans or 5 seconds (configurable)
- Config file: `config/observability.yaml` with environment overrides
- Contract compliance: 100% (69/69 checks) validated

See `docs/OPIK_TODO.md` for implementation details and `config/observability.yaml` for configuration.

### Tri-Memory System

**Working Memory** (Redis)
- Session state, messages, evidence board
- TTL: 3600 seconds

**Episodic Memory** (Supabase + pgvector)
- User queries, agent actions, events

**Procedural Memory** (Supabase + pgvector)
- Tool sequences, query patterns

**Semantic Memory** (FalkorDB)
- Entity nodes, relationships, causal chains

### Causal Validation Pipeline

5 DoWhy refutation tests ensure causal estimate reliability:
1. Placebo treatment test
2. Random common cause test
3. Data subset validation
4. Bootstrap estimation
5. Sensitivity analysis (E-value)

Gate decisions: **proceed** | **review** | **block**

### Query Robustness

3-layer natural language processing:
1. **fastText normalization** - Handle typos via subword embeddings
2. **rapidfuzz matching** - Fuzzy match against domain vocabulary
3. **Claude disambiguation** - Resolve complex/ambiguous queries

## Database

28 tables across 5 categories:
- **Core Data** (8): patient_journeys, hcp_profiles, treatment_events, etc.
- **ML Foundation** (8): ml_experiments, ml_model_registry, ml_deployments, etc.
- **Memory** (4): episodic_memories, procedural_memories, semantic_cache, working_memory
- **Causal Validation** (2): causal_validations, expert_reviews
- **Supporting** (6): user_sessions, data_source_tracking, etc.

## Documentation

- **Primary**: `docs/e2i_nlv_project_structure_v4.1.md` (55KB)
- **Setup**: `docs/README.md` (Memory system)
- **Real-Time SHAP API**: `docs/realtime_shap_api.md` ⭐ NEW v4.1
- **Feature Store**: `docs/FEATURE_STORE.md` (complete guide) ⭐ NEW v4.2
- **Feature Store Quick Start**: `docs/FEATURE_STORE_QUICKSTART.md` ⭐ NEW v4.2
- **Opik Observability**: `docs/OPIK_TODO.md` (implementation status)
- **Observability Config**: `config/observability.yaml` (production settings)
- **RAG Implementation**: `docs/rag_implementation_plan.md`
- **Codebase Index**: `.claude/codebase_index.md`
- **Project Structure**: `PROJECT_STRUCTURE.txt`
- **API Docs**: Coming soon

## Tech Stack

| Category | Technologies |
|----------|-------------|
| AI/ML | LangGraph, LangChain, Claude (Anthropic) |
| Causal | DoWhy, EconML, NetworkX |
| MLOps | MLflow, Opik, Optuna, SHAP, BentoML, Great Expectations |
| Feature Store | Lightweight (Supabase + Redis + MLflow) ⭐ NEW v4.2 |
| Database | PostgreSQL/Supabase, pgvector, Redis, FalkorDB |
| NLP | fastText, rapidfuzz, sentence-transformers |
| API | FastAPI, Pydantic |
| Frontend | React, TypeScript (planned) |

## License

Proprietary - All rights reserved

## Support

For questions or issues, please contact the E2I development team.

---

**Version**: 4.2.1
**Last Updated**: December 2025
**Recent**: Added Opik LLM/Agent Observability integration
