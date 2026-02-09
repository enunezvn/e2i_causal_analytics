# E2I Causal Analytics

**Healthcare Engagement Intelligence Platform**
Multi-Agent Causal Analytics for Pharmaceutical Drug Adoption Analysis

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version 4.2.1](https://img.shields.io/badge/version-4.2.1-green.svg)]()
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)]()
[![Backend Tests](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/backend-tests.yml/badge.svg)](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/backend-tests.yml)
[![Frontend Tests](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/frontend-tests.yml/badge.svg)](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/frontend-tests.yml)
[![Security](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/security.yml/badge.svg)](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/security.yml)
[![Deploy](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/deploy.yml/badge.svg)](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/deploy.yml)
[![Type Check](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/verify-types.yml/badge.svg)](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/verify-types.yml)

## Overview

E2I Causal Analytics is a sophisticated 21-agent, 6-tier agentic system designed for pharmaceutical companies to understand and optimize drug adoption through causal inference and natural language querying.

### Key Features

- **21 AI Agents** across 6 tiers (ML Foundation, Coordination, Causal Analytics, Monitoring, Predictions, Self-Improvement)
- **Tri-Memory Architecture** (Working, Episodic, Procedural, Semantic)
- **Causal Validation** with 5 DoWhy refutation tests
- **MLOps Integration** (MLflow, Opik, Feast, Great Expectations, Optuna, SHAP, BentoML)
- **Real-Time Model Interpretability** (v4.1) - SHAP explanations in 50-500ms via REST API
- **Digital Twin Engine** (v4.2) - A/B test pre-screening with ML-based simulations
- **Tool Composer** (v4.2) - Multi-faceted query decomposition & dynamic tool orchestration
- **Natural Language Interface** with typo-tolerant query processing
- **120+ Database Tables** across core, ML, memory, chat, audit, and RAG schemas
- **Hybrid RAG System** with vector + full-text + graph search
- **Full-Stack Dashboard** — React 18 + TypeScript + Vite with 21 pages
- **Production Observability** — Prometheus, Grafana, Loki, Alertmanager

### Analyzed Brands

- **Remibrutinib** - BTK inhibitor for chronic spontaneous urticaria (CSU)
- **Fabhalta** - Factor B inhibitor for paroxysmal nocturnal hemoglobinuria (PNH)
- **Kisqali** - CDK4/6 inhibitor (ribociclib) for breast cancer

## Architecture

### 6-Tier Agent System

**TIER 0: ML FOUNDATION** (8 agents)
- scope_definer, cohort_constructor, data_preparer, feature_analyzer, model_selector, model_trainer, model_deployer, observability_connector

**TIER 1: COORDINATION** (2 agents)
- orchestrator (multi-agent routing & synthesis with 4-stage classifier)
- tool_composer (multi-faceted query decomposition & tool orchestration)

**TIER 2: CAUSAL ANALYTICS** (3 agents)
- causal_impact (effect estimation + 5 refutation tests)
- gap_analyzer (ROI opportunity identification)
- heterogeneous_optimizer (treatment effect heterogeneity)

**TIER 3: MONITORING** (4 agents)
- drift_monitor, experiment_designer (with Digital Twin pre-screening), experiment_monitor, health_score

**TIER 4: ML PREDICTIONS** (2 agents)
- prediction_synthesizer, resource_optimizer

**TIER 5: SELF-IMPROVEMENT** (2 agents)
- explainer, feedback_learner

## Project Structure

```
e2i_causal_analytics/
├── config/                    # YAML configurations (48 files)
│   ├── agent_config.yaml      # Agent definitions
│   ├── domain_vocabulary_v3.1.0.yaml
│   ├── kpi_definitions.yaml   # 50 KPIs
│   └── ...
│
├── database/                  # SQL schemas (120+ tables)
│   ├── core/                  # Core data tables (patients, HCPs, treatments, triggers)
│   ├── ml/                    # ML pipeline tables (experiments, models, digital twins, A/B testing, GEPA, etc.)
│   ├── memory/                # Memory tables + FalkorDB schema
│   ├── chat/                  # Chat, feedback, analytics tables
│   ├── rag/                   # RAG document chunks + search logs
│   ├── audit/                 # Audit trail + security audit log
│   ├── causal/                # Causal validation + energy score tables
│   └── migrations/            # Feature store, feedback loop, validation schemas
│
├── data/
│   ├── synthetic/             # ~200 patients, ~50 HCPs (20 JSON files)
│   └── training/              # fastText corpus
│
├── src/                       # Main source code
│   ├── nlp/                   # Query processing, entity extraction
│   ├── agents/                # 21 agent implementations (6 tiers)
│   │   ├── orchestrator/      # Tier 1 coordination (4-stage classifier + router)
│   │   ├── tool_composer/     # Multi-faceted query decomposition & orchestration
│   │   ├── experiment_designer/ # Experiment design with Digital Twin pre-screening
│   │   ├── ml_foundation/     # 7 Tier 0 agents (scope, data, features, models)
│   │   └── ...                # 11 more agents (causal, monitoring, predictions, etc.)
│   ├── digital_twin/          # A/B test pre-screening with ML-based simulations
│   ├── tool_registry/         # Tool discovery & management
│   ├── feature_store/         # Lightweight feature store (Supabase + Redis + MLflow)
│   ├── memory/                # Tri-memory backends (working, episodic, procedural, semantic)
│   ├── causal_engine/         # EconML CausalForestDML, CausalML, DoWhy integration
│   ├── rag/                   # Hybrid RAG (vector + full-text + graph via FalkorDB)
│   ├── ml/                    # ML operations & data management
│   ├── mlops/                 # MLflow, Opik, Feast, BentoML, SHAP connectors
│   ├── workers/               # Celery task definitions and event consumers
│   ├── api/                   # FastAPI endpoints & middleware
│   └── utils/                 # Shared utilities (circuit breaker, etc.)
│
├── tests/                     # 500+ test files (unit, integration, tier0-5)
├── scripts/                   # Utility scripts (deploy, health check, backups, migrations)
├── frontend/                  # React 18 + TypeScript + Vite dashboard (21 pages)
├── docs/                      # Comprehensive documentation
│   ├── ARCHITECTURE.md        # C4-model architecture documentation
│   ├── ONBOARDING.md          # Developer onboarding guide
│   ├── SYNTHETIC_DATA.md      # Synthetic data generation & validation reference
│   ├── api/                   # OpenAPI spec (auto-generated)
│   └── data/                  # Data dictionary & conversion docs
│       ├── 00-INDEX.md        # Master index & quick-start
│       ├── 01-07 *.md         # Schema docs (core, ML, graph, Feast, KPIs)
│       └── templates/         # CSV templates with example rows
└── docker/                    # Container configurations
```

## What's New in v4.2.1

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

- Docker Engine 24+ and Docker Compose v2
- Supabase account (or self-hosted Supabase)
- Anthropic API key

All services (API, frontend, workers, Redis, FalkorDB, MLflow, Opik, observability) run in Docker containers via Docker Compose.

### Installation

1. **Clone and configure environment**
   ```bash
   git clone https://github.com/enunezvn/e2i_causal_analytics.git
   cd e2i_causal_analytics
   cp .env.example .env
   # Edit .env with your API keys and database URLs
   ```

2. **Start all services**
   ```bash
   docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
   ```

   To include Opik (agent observability):
   ```bash
   docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml -f docker/docker-compose.opik.yml up -d
   ```

3. **Verify services are running**
   ```bash
   docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml ps
   curl http://localhost:8000/health
   ```

4. **Initialize database**
   ```bash
   # Apply schemas in order:
   # 1. database/core/e2i_ml_complete_v3_schema.sql
   # 2. database/ml/mlops_tables.sql
   # 3. database/ml/010_causal_validation_tables.sql
   # 4. database/ml/011_realtime_shap_audit.sql
   # 5. database/ml/012_digital_twin_tables.sql (v4.2)
   # 6. database/ml/013_tool_composer_tables.sql (v4.2)
   # 7. database/migrations/004_create_feature_store_schema.sql (v4.2)
   # 8. database/memory/ (RAG functions)
   # 9. database/audit/

   # For feature store migration, use:
   python scripts/run_migration.py database/migrations/004_create_feature_store_schema.sql
   ```

5. **Generate synthetic data**
   ```bash
   make data-generate
   # Or: python src/ml/data_generator.py
   ```

See `docker/README.md` for Docker Compose configuration and `DEPLOYMENT.md` for setup instructions.

## CI/CD Workflows

All workflows live in `.github/workflows/` and run on GitHub Actions:

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| Backend Tests | `backend-tests.yml` | Push/PR to main, develop | pytest with coverage gate (`--cov-fail-under=70`) |
| Frontend Tests | `frontend-tests.yml` | Push/PR to main, develop | Vitest + coverage thresholds |
| Deploy | `deploy.yml` | Push to main | Auto-deploy with health checks and rollback |
| Security | `security.yml` | Push/PR + daily cron | Bandit, safety, secrets scan |
| API Docs | `api-docs.yml` | Push/PR to main, develop | OpenAPI spec validation (Spectral) |
| Verify Types | `verify-types.yml` | Push/PR to main | MyPy type checking (non-blocking) |
| RAGAS Evaluation | `ragas-evaluation.yml` | Push/PR to main | RAG pipeline quality evaluation |
| Synthetic Benchmarks | `synthetic-benchmarks.yml` | Push/PR to main | Causal engine benchmark suite |

## Operational Scripts

Key scripts in `scripts/`:

**Core Operations**
- `deploy.sh` — Git pull, restart workers, seed FalkorDB, health check
- `health_check.sh` — Check all 24 services (HTTP, Redis, FalkorDB, Supabase, observability)
- `run_migrations.sh` — Apply SQL migrations to Supabase
- `backup_data_stores.sh` — Backup Redis, FalkorDB, MLflow artifacts
- `backup_cron.sh` — Scheduled backup wrapper

**Testing**
- `run_tests_batched.sh` — Full test suite in 43 batches (~20 min)
- `run_frontend_tests_batched.sh` — Frontend test suite in batches

**Infrastructure**
- `opik-manager.sh` — Start/stop/status for Opik overlay
- `setup_branch_protection.sh` — Configure GitHub branch protection via `gh api`
- `ssh-tunnels/tunnels.sh` — SSH tunnel launcher for remote management ports
- `seed_falkordb_all.sh` — Seed knowledge graph from Supabase tables

**Utilities**
- `droplet-connect.sh` — SSH into the production droplet
- `droplet_report.sh` — System resource and service status report
- `generate_api_docs.sh` — Regenerate OpenAPI spec
- `preflight-check.sh` — Pre-deploy validation checks

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

### Real-Time Model Interpretability (v4.1)

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

### Lightweight Feature Store (v4.2)

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

120+ tables across 8 categories:
- **Core Data** (19): patient_journeys, hcp_profiles, treatment_events, triggers, business_metrics, etc.
- **ML Pipeline** (60+): experiments, model registry, digital twins, causal validation, A/B testing, GEPA, cohort constructor, etc.
- **Memory** (7): episodic_memories, procedural_memories, semantic_cache, cognitive_cycles, etc.
- **RAG** (2): rag_document_chunks (HNSW), rag_search_logs
- **Chat** (10+): chat_threads, chat_messages, user_preferences (RLS), chatbot analytics, feedback, training signals
- **Audit** (3): audit_chain_entries (SHA-256 hash chain), verification_log, security_audit_log (partitioned)
- **FalkorDB Graph**: 8 node types, 15 edge types
- **Feast Feature Store**: 10 feature views, 48 features

See [`docs/data/00-INDEX.md`](docs/data/00-INDEX.md) for the complete data dictionary and schema documentation.

## Documentation

- **Architecture**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — C4-model system architecture
- **Onboarding**: [`docs/ONBOARDING.md`](docs/ONBOARDING.md) — Developer setup guide
- **Synthetic Data**: [`docs/SYNTHETIC_DATA.md`](docs/SYNTHETIC_DATA.md) — DGPs, generators, causal validation, digital twin
- **Data Documentation**: [`docs/data/00-INDEX.md`](docs/data/00-INDEX.md) — Master index for all data docs:
  - [Data Conversion Guide](docs/data/01-DATA-CONVERSION-GUIDE.md) — Real data onboarding playbook
  - [Core Data Dictionary](docs/data/02-CORE-DATA-DICTIONARY.md) — 19 core tables, 12 enums, 28 views
  - [ML Pipeline Schema](docs/data/03-ML-PIPELINE-SCHEMA.md) — 60+ ML lifecycle tables
  - [Knowledge Graph Ontology](docs/data/04-KNOWLEDGE-GRAPH-ONTOLOGY.md) — FalkorDB schema
  - [Feature Store Reference](docs/data/05-FEATURE-STORE-REFERENCE.md) — Feast entities & features
  - [KPI Reference](docs/data/06-KPI-REFERENCE.md) — All 46 KPIs with formulas & thresholds
  - [Supporting Schemas](docs/data/07-SUPPORTING-SCHEMAS.md) — Memory, RAG, Chat, Audit
  - [CSV Templates](docs/data/templates/) — Ready-to-use templates with example rows
- **API Reference**: [`docs/api/openapi.json`](docs/api/openapi.json) — OpenAPI 3.0 spec
- **Observability Config**: `config/observability.yaml` — Production settings
- **Developer Reference**: `CLAUDE.md` — Quick reference for AI-assisted development

## Tech Stack

| Category | Technologies |
|----------|-------------|
| AI/ML | LangGraph, LangChain, Claude (Anthropic) |
| Causal | DoWhy, EconML, NetworkX |
| MLOps | MLflow, Opik, Optuna, SHAP, BentoML, Great Expectations |
| Feature Store | Feast + Lightweight (Supabase + Redis + MLflow) |
| Database | PostgreSQL/Supabase, pgvector, Redis, FalkorDB |
| NLP | fastText, rapidfuzz, sentence-transformers |
| API | FastAPI, Pydantic |
| Frontend | React 18, TypeScript, Vite, TanStack Query, Tailwind, CopilotKit |
| Observability | Prometheus, Grafana, Loki, Promtail, Alertmanager |
| Infrastructure | Docker Compose, Nginx, Celery, Certbot |

## License

Proprietary - All rights reserved

## Support

For questions or issues, please contact the E2I development team.

---

**Version**: 4.2.1
**Last Updated**: February 2026
**Recent**: All-Docker deployment, observability stack, production hardening
