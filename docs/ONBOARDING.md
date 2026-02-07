# E2I Causal Analytics - Developer Onboarding Guide

**Healthcare Engagement Intelligence Platform** (v4.2.1)
21-Agent, 6-Tier Causal Analytics System for Pharmaceutical Drug Adoption Analysis

---

## Table of Contents

1. [Welcome](#1-welcome)
2. [System Requirements](#2-system-requirements)
3. [Environment Setup](#3-environment-setup)
4. [Project Overview](#4-project-overview)
5. [Architecture Deep Dive](#5-architecture-deep-dive)
6. [Development Workflow](#6-development-workflow)
7. [Testing](#7-testing)
8. [Deployment](#8-deployment)
9. [Security & Compliance](#9-security--compliance)
10. [Codebase Navigation](#10-codebase-navigation)
11. [First Tasks](#11-first-tasks)
12. [Tools & Access](#12-tools--access)
13. [Troubleshooting](#13-troubleshooting)
14. [FAQ](#14-faq)

---

## 1. Welcome

E2I Causal Analytics helps pharmaceutical companies understand and optimize drug adoption through **causal inference** and **natural language querying**. The platform analyzes three brands:

| Brand | Drug Class | Indication |
|-------|-----------|------------|
| **Remibrutinib** | BTK inhibitor | Chronic spontaneous urticaria (CSU) |
| **Fabhalta** | Factor B inhibitor | Paroxysmal nocturnal hemoglobinuria (PNH) |
| **Kisqali** | CDK4/6 inhibitor (ribociclib) | Breast cancer |

The system uses 21 AI agents organized in 6 tiers to perform causal analysis, predict drug adoption, design experiments, and explain model outputs in natural language.

### Key Capabilities

- **Causal Inference**: DoWhy refutation tests, EconML treatment effect estimation, CausalML uplift modeling
- **Natural Language Interface**: Typo-tolerant query processing (fastText + rapidfuzz + Claude)
- **Digital Twin Engine**: A/B test pre-screening with ML-based simulations
- **Real-Time Explainability**: SHAP explanations in 50-500ms via REST API
- **Knowledge Graph**: FalkorDB temporal graph with Cypher queries
- **Full MLOps**: MLflow tracking, Opik observability, Feast features, BentoML serving

---

## 2. System Requirements

### For Docker-Based Development (Recommended)

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Docker Engine** | 24+ | Latest stable |
| **Docker Compose** | v2+ | Latest stable |
| **RAM** | 8 GB | 16 GB+ |
| **Disk** | 20 GB free | 50 GB free |
| **OS** | Linux / macOS / Windows (WSL2) | Ubuntu 22.04+ |

### For Local Python Development

| Requirement | Version |
|-------------|---------|
| **Python** | 3.12+ |
| **Node.js** | 20+ (for frontend) |
| **npm** | 10+ |
| **Git** | 2.40+ |

### API Keys Required

| Service | Variable | How to Get |
|---------|----------|------------|
| **Anthropic** | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| **Supabase** | `SUPABASE_URL`, `SUPABASE_KEY` | Project settings at [supabase.com](https://supabase.com) |
| **OpenAI** (optional) | `OPENAI_API_KEY` | For RAGAS evaluation and embeddings |

---

## 3. Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/enunezvn/e2i_causal_analytics.git
cd e2i_causal_analytics
```

### Step 2: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials. Required variables:

```env
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJhbG...
SUPABASE_SERVICE_KEY=eyJhbG...

# Passwords (choose strong values, no defaults allowed)
REDIS_PASSWORD=your-redis-password
FALKORDB_PASSWORD=your-falkordb-password
GRAFANA_ADMIN_PASSWORD=your-grafana-password

# Database
SUPABASE_DB_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
```

> **Note**: `docker/.env` is a symlink to `../.env` so Docker Compose picks up these values automatically.

### Step 3: Start All Services (Docker)

```bash
# Start core services (API, frontend, workers, Redis, FalkorDB, MLflow, observability)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# Optional: include Opik agent observability (10 additional services)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml \
  -f docker/docker-compose.opik.yml up -d
```

Or use the Makefile shortcut:

```bash
make docker-up
```

### Step 4: Verify Services

```bash
# Check all containers are healthy
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml ps

# Check API
curl http://localhost:8000/health

# Check frontend
curl -s http://localhost:3002 | head -5

# Full health check (24 services)
./scripts/health_check.sh
```

### Step 5: Set Up Local Python Environment (for tests & linting)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### Step 6: Set Up Frontend Development (optional)

```bash
cd frontend
npm ci
cd ..
```

### Step 7: Initialize Database

Apply schemas in order:

```bash
# Core tables (8 tables)
psql $SUPABASE_DB_URL < database/core/e2i_ml_complete_v3_schema.sql

# ML tables (17 tables)
psql $SUPABASE_DB_URL < database/ml/mlops_tables.sql
psql $SUPABASE_DB_URL < database/ml/010_causal_validation_tables.sql
psql $SUPABASE_DB_URL < database/ml/011_realtime_shap_audit.sql
psql $SUPABASE_DB_URL < database/ml/012_digital_twin_tables.sql
psql $SUPABASE_DB_URL < database/ml/013_tool_composer_tables.sql

# Feature store
python scripts/run_migration.py database/migrations/004_create_feature_store_schema.sql

# Memory & audit tables
psql $SUPABASE_DB_URL < database/memory/*.sql
psql $SUPABASE_DB_URL < database/audit/*.sql
```

### Step 8: Generate Synthetic Data

```bash
make data-generate
# Or: python src/ml/data_generator.py
```

---

## 4. Project Overview

### Tech Stack

| Category | Technologies |
|----------|-------------|
| **AI/ML** | LangGraph, LangChain, Claude (Anthropic), scikit-learn, LightGBM |
| **Causal Inference** | DoWhy, EconML (CausalForestDML), CausalML (UpliftRandomForest), NetworkX |
| **MLOps** | MLflow, Opik, Feast, BentoML, Great Expectations, Optuna, SHAP |
| **Backend** | FastAPI, Pydantic, Celery, Redis |
| **Frontend** | React 18, TypeScript, Vite, TanStack Query, Tailwind CSS, CopilotKit |
| **Databases** | PostgreSQL (Supabase), pgvector, Redis, FalkorDB |
| **NLP** | fastText, rapidfuzz, sentence-transformers |
| **Observability** | Prometheus, Grafana, Loki, Promtail, Alertmanager |
| **Infrastructure** | Docker Compose, Nginx, DigitalOcean |

### Project Structure

```
e2i_causal_analytics/
├── src/                    # Backend source code
│   ├── agents/             # 21 LangGraph agent implementations
│   ├── api/                # FastAPI app, routes, middleware
│   ├── causal_engine/      # EconML, CausalML, DoWhy integration
│   ├── digital_twin/       # A/B test simulation engine
│   ├── memory/             # Tri-memory architecture
│   ├── rag/                # Hybrid RAG (vector + full-text + graph)
│   ├── mlops/              # MLflow, Opik, Feast, BentoML connectors
│   ├── workers/            # Celery task definitions
│   ├── nlp/                # Query processing, entity extraction
│   ├── feature_store/      # Feature store client
│   ├── kpi/                # 46+ KPI definitions
│   └── utils/              # Shared utilities
├── frontend/               # React/TypeScript/Vite frontend
│   └── src/
│       ├── components/     # React components (20+ pages)
│       ├── api/            # API client layer
│       ├── lib/            # Utility libraries
│       └── providers/      # Context providers
├── tests/                  # Test suite
│   ├── unit/               # Unit tests (43 batches)
│   ├── integration/        # Integration tests
│   ├── e2e/                # End-to-end tests
│   ├── synthetic/          # Causal model validation
│   └── security/           # Security-specific tests
├── config/                 # YAML configurations
│   ├── agent_config.yaml   # 21-agent definitions
│   ├── kpi_definitions.yaml
│   └── observability.yaml
├── database/               # SQL schemas (37+ tables)
│   ├── core/               # 8 core data tables
│   ├── ml/                 # 17 ML tables
│   ├── memory/             # 4 memory tables
│   └── audit/              # Audit trail
├── docker/                 # Docker Compose & Dockerfiles
│   ├── docker-compose.yml       # Base (21+ services)
│   ├── docker-compose.dev.yml   # Dev overlay (hot-reload)
│   └── docker-compose.opik.yml  # Opik overlay (10 services)
├── scripts/                # 36+ operational scripts
├── feature_repo/           # Feast feature definitions
├── .github/workflows/      # 8 CI/CD workflows
├── CLAUDE.md               # AI assistant instructions
├── DEPLOYMENT.md           # Deployment guide
└── pyproject.toml          # Python tool configuration
```

---

## 5. Architecture Deep Dive

### 6-Tier Agent System

```
TIER 0: ML Foundation (8 agents)
  scope_definer → cohort_constructor → data_preparer → feature_analyzer
  → model_selector → model_trainer → model_deployer → observability_connector

TIER 1: Coordination (2 agents)
  orchestrator (4-stage classifier + router)
  tool_composer (multi-faceted query decomposition)

TIER 2: Causal Analytics (3 agents)
  causal_impact (DoWhy refutation)
  gap_analyzer (ROI opportunity)
  heterogeneous_optimizer (CATE estimation)

TIER 3: Monitoring (3 agents)
  drift_monitor, experiment_designer (digital twin), health_score

TIER 4: Predictions (2 agents)
  prediction_synthesizer, resource_optimizer

TIER 5: Self-Improvement (2 agents)
  explainer (SHAP), feedback_learner
```

Each agent is a **LangGraph state machine** with:
- Typed state (TypedDict with `NotRequired` for optional fields)
- Node functions (data retrieval, analysis, synthesis)
- Conditional edges (routing logic)
- Tool bindings (API calls, database queries, ML inference)

### Container Architecture

| Service | Container | Port (Host) | Auto-Reload |
|---------|-----------|-------------|-------------|
| **API** | `e2i_api_dev` | 8000 | Yes (uvicorn --reload) |
| **Frontend** | `e2i_frontend_dev` | 3002 | Yes (Vite HMR) |
| **Worker Light** | auto-numbered (x2) | - | No (restart needed) |
| **Worker Medium** | `e2i_worker_medium_dev` | - | No |
| **Scheduler** | `e2i_scheduler_dev` | - | No |
| **Redis** | `e2i_redis_dev` | 6382 | N/A |
| **FalkorDB** | `e2i_falkordb_dev` | 6381 | N/A |
| **MLflow** | `e2i_mlflow_dev` | 5000 | N/A |
| **Prometheus** | `e2i_prometheus` | 9091 | N/A |
| **Grafana** | `e2i_grafana` | 3200 | N/A |

### Worker Queue Architecture

| Worker | Concurrency | Queues | Use Case |
|--------|-------------|--------|----------|
| **worker_light** (x2) | 4 | quick, api, default | Fast tasks (<30s) |
| **worker_medium** | 1 | analytics, reports, ml | Medium analysis (1-5 min) |
| **worker_heavy** | 1 | causal, training, heavy | Heavy ML (5-30 min, on-demand) |
| **scheduler** | - | celery-beat | Periodic tasks |

### Tri-Memory System

| Layer | Storage | Purpose | TTL |
|-------|---------|---------|-----|
| **Working** | Redis | Session state, messages, evidence board | 3600s |
| **Episodic** | Supabase + pgvector | User queries, agent actions, events | Permanent |
| **Procedural** | Supabase + pgvector | Tool sequences, query patterns | Permanent |
| **Semantic** | FalkorDB | Entity nodes, relationships, causal chains | Permanent |

### Database Schema

- **37+ tables** across PostgreSQL (Supabase) with pgvector extension
- **Core Data** (8): patient_journeys, hcp_profiles, treatment_events, etc.
- **ML Foundation** (8): ml_experiments, ml_model_registry, ml_deployments, etc.
- **Causal** (2): causal_validations, expert_reviews
- **Memory** (4): episodic_memories, procedural_memories, semantic_cache, working_memory
- **Digital Twin** (3): digital_twin_models, twin_simulations, twin_fidelity_tracking
- **Tool Composer** (6): tool registry, dependencies, composition episodes, etc.
- **Feature Store** (3): feature_groups, features, feature_values
- **FalkorDB Graph**: 8 node types, 15 edge types (HCP, Patient, Treatment, Brand, etc.)

---

## 6. Development Workflow

### Git Workflow

**Branch Protection**: `main` branch is protected - no direct pushes.

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes, then commit
git add <files>
git commit -m "feat: description of change"

# Push and create PR
git push -u origin feature/your-feature-name
gh pr create --title "feat: description" --body "## Summary\n..."
```

**Pre-commit hooks** run automatically on `git commit`:
- **ruff** (lint + auto-fix)
- **ruff-format** (code formatting, line-length=100)
- **trailing-whitespace** removal
- **end-of-file-fixer**
- **check-yaml** validation
- **check-added-large-files** (>1MB blocked)
- **no-commit-to-branch** (prevents commits to `main`)

### PR Requirements

1. All CI checks must pass (lint, type-check, unit tests, integration tests)
2. At least 1 approval required
3. CODEOWNERS review required (@enunez)
4. Stale reviews auto-dismissed on new pushes

### Code Style

| Tool | Purpose | Config Location |
|------|---------|----------------|
| **Ruff** | Linting + formatting | `pyproject.toml` [tool.ruff] |
| **Black** | Python formatting | `pyproject.toml` [tool.black] |
| **MyPy** | Type checking (non-blocking) | `pyproject.toml` [tool.mypy] |
| **ESLint** | Frontend linting | `frontend/eslint.config.js` |
| **TypeScript** | Frontend type checking | `frontend/tsconfig.app.json` |

Key conventions:
- Line length: **100 characters** (Python and TypeScript)
- Python target: **3.12**
- No `n_jobs=-1` in ML code (causes process deadlocks) - use `n_jobs=1`
- Module-level flags for testing (`TESTING_MODE`, `DEBUG_MODE`) - tests patch the module var
- Agent output TypedDicts use `NotRequired` for optional fields

### Useful Make Commands

```bash
make help           # Show all available commands
make test           # Run tests with coverage
make test-fast      # Run tests without coverage
make test-seq       # Sequential tests (for debugging)
make test-cov       # Full coverage (HTML + XML reports)
make lint           # Ruff check + MyPy
make format         # Black + Ruff fix
make docker-up      # Start all Docker services
make docker-down    # Stop all Docker services
make docker-logs    # Tail API + frontend logs
make deploy         # Quick deploy (git pull + restart workers)
make deploy-build   # Deploy with image rebuild
make api-docs       # Generate OpenAPI spec + Redoc HTML
make clean          # Remove build artifacts
```

---

## 7. Testing

### Test Hierarchy

| Level | Tool | Location | Parallelization | Typical Runtime |
|-------|------|----------|-----------------|-----------------|
| **Unit** | pytest | `tests/unit/` | 4 workers | ~5 min |
| **Integration** | pytest + Redis | `tests/integration/` | 2 workers | ~10 min |
| **E2E (Backend)** | pytest | `tests/e2e/` | Sequential | ~15 min |
| **E2E (Frontend)** | Playwright | `frontend/e2e/` | 4 shards | ~10 min |
| **Synthetic** | pytest | `tests/synthetic/` | Sequential | ~5 min |
| **RAGAS** | OpenAI + RAGAS | `scripts/run_ragas_eval.py` | Sequential | ~10 min |
| **Tier 0** | Custom runner | `scripts/run_tier0_test.py` | Sequential | ~20 min |
| **Tier 1-5** | Custom runner | `scripts/run_tier1_5_test.py` | Sequential | ~15 min |

### Running Tests

```bash
# Standard test suite (4 parallel workers, 30s timeout per test)
.venv/bin/pytest tests/

# With coverage (fail_under=70%)
.venv/bin/pytest tests/ --cov --cov-report=term-missing

# Batched suite (43 batches, RAM-aware, ~20 min)
scripts/run_tests_batched.sh

# Single test file (for debugging)
.venv/bin/pytest tests/unit/test_agents/test_orchestrator.py -v -n 0 -s

# Frontend unit tests
cd frontend && npm run test:run

# Frontend E2E tests
cd frontend && npm run test:e2e
```

### Test Markers

```bash
# Run only unit tests
pytest -m unit tests/

# Run only integration tests (requires Redis)
pytest -m integration tests/

# Skip slow tests
pytest -m "not slow" tests/

# Run only tests that need FalkorDB
pytest -m requires_falkordb tests/
```

Available markers: `unit`, `integration`, `e2e`, `slow`, `requires_redis`, `requires_falkordb`, `requires_supabase`, `heavy_ml`, `xdist_group`

### Tier Tests (ML Pipeline Validation)

```bash
# Tier 0: Full ML pipeline (generates 1500 patients, caches output)
.venv/bin/python scripts/run_tier0_test.py

# Tier 1-5: Test all 12 agents using cached Tier 0 output
.venv/bin/python scripts/run_tier1_5_test.py

# Run specific tiers
.venv/bin/python scripts/run_tier1_5_test.py --tiers 2,3

# Run specific agents
.venv/bin/python scripts/run_tier1_5_test.py --agents causal_impact,explainer
```

### Coverage Thresholds

| Component | Lines | Branches | Functions | Statements |
|-----------|-------|----------|-----------|------------|
| **Backend** | 70% | 70% | 70% | 70% |
| **Frontend** | 62% | 55% | 54% | 62% |

---

## 8. Deployment

### Architecture

Dev and prod are the **same machine** - a single DigitalOcean droplet (8 vCPU / 32 GB RAM). All services run via Docker Compose. Host nginx handles SSL termination.

### Deploy Process

```bash
# Most common: pull changes + restart workers
./scripts/deploy.sh

# When dependencies change (requirements.txt, package.json, Dockerfiles)
./scripts/deploy.sh --build

# Verify
./scripts/health_check.sh
```

- **API**: Auto-reloads via `uvicorn --reload` (bind mount)
- **Frontend**: Auto-reloads via Vite HMR (bind mount)
- **Workers**: Must be explicitly restarted (Celery doesn't auto-reload)

### CI/CD Pipeline

Push to `main` triggers the deploy workflow:

1. **Backend Tests** (lint, type-check, unit tests, integration tests)
2. **Build & Push** API image to GHCR
3. **Build & Push** Frontend image to GHCR
4. **SSH Deploy** to droplet (git pull + restart workers + health check)

### Accessing Services (via SSH Tunnel)

```bash
# Start all tunnels
bash scripts/ssh-tunnels/tunnels.sh

# Or minimal tunnel for frontend
ssh -N -L 8443:localhost:443 enunez@138.197.4.36
```

| Service | Local URL |
|---------|-----------|
| Frontend | https://localhost:8443 |
| API Docs | https://localhost:8443/api/docs |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3200 |
| Opik | http://localhost:5173 |
| FalkorDB Browser | http://localhost:3030 |
| Supabase Studio | http://localhost:3001 |
| Alertmanager | http://localhost:9093 |

---

## 9. Security & Compliance

### Authentication

- **JWT-based** via Supabase Auth
- Tokens validated against `SUPABASE_JWT_SECRET`
- Testing mode auto-bypasses auth (`E2I_TESTING_MODE=true`)

### Role-Based Access Control (4 Levels)

| Role | Level | Permissions |
|------|-------|-------------|
| **viewer** | 1 | Read-only dashboard access |
| **analyst** | 2 | Run analyses + viewer permissions |
| **operator** | 3 | Manage experiments/feedback + analyst permissions |
| **admin** | 4 | System management (full access) |

### Security Headers

All API responses include:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy` (self-based, with CDN exceptions for docs)
- `Strict-Transport-Security` (production only)
- `Permissions-Policy` (restricts camera, microphone, etc.)

### Rate Limiting

| Endpoint | Limit | Window |
|----------|-------|--------|
| Default | 100 req | 60s |
| Auth endpoints | 20 req | 60s |
| CopilotKit chat | 30 req | 1 hour |
| Batch operations | 10 req | 60s |
| Health checks | 300 req | 60s |

### CI Security Scanning (8 Checks)

| Scanner | Target | Blocks Deploy |
|---------|--------|---------------|
| **Gitleaks** | Secrets in git history | Yes |
| **Bandit** | Python SAST (HIGH/CRITICAL) | Yes |
| **Semgrep** | Multi-language SAST (OWASP Top 10) | Yes |
| **pip-audit** | Python dependency vulnerabilities | Yes |
| **npm audit** | Frontend dependency vulnerabilities | No (reports only) |
| **Trivy** | Container image scanning | No (reports only) |
| **Hadolint** | Dockerfile best practices | No (reports only) |
| **Spectral** | OpenAPI spec linting | No (reports only) |

### Important Security Practices

- **Never commit secrets** to git (`.env` is gitignored, Gitleaks scans history)
- **No default passwords** - all `REDIS_PASSWORD`, `FALKORDB_PASSWORD`, etc. are required
- **Management ports** bound to `127.0.0.1` only (MLflow, Grafana, etc.)
- **read_only: true** on API, scheduler, frontend, prometheus, grafana containers
- **FalkorDB auth** enabled via `--requirepass`
- **MLflow auth** via nginx `auth_basic`

---

## 10. Codebase Navigation

### Key Entry Points

| What You Want | Where to Look |
|---------------|---------------|
| API routes | `src/api/routes/` (80+ endpoints) |
| API middleware | `src/api/middleware/` (security, auth, rate limiting, CORS, timing) |
| Agent implementations | `src/agents/<agent_name>/` (each has graph.py, nodes, tools) |
| Agent configuration | `config/agent_config.yaml` |
| Causal engine | `src/causal_engine/` (EconML, CausalML, DoWhy) |
| KPI definitions | `config/kpi_definitions.yaml` + `src/kpi/` |
| Celery tasks | `src/tasks/` + `src/workers/` |
| Frontend components | `frontend/src/components/` |
| Frontend API layer | `frontend/src/api/` + `frontend/src/lib/api-client.ts` |
| Database schemas | `database/` (organized by domain) |
| Docker configs | `docker/docker-compose*.yml` |
| CI/CD workflows | `.github/workflows/` (8 YAML files) |

### How Agents Work

Each agent follows this pattern:

```python
# src/agents/<agent_name>/graph.py
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    """Agent state definition"""
    input_data: dict
    results: NotRequired[dict]
    errors: NotRequired[list]

def create_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_data", retrieve_data_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_edge("retrieve_data", "analyze")
    graph.add_edge("analyze", "synthesize")
    return graph.compile()
```

### How API Routes Work

```python
# src/api/routes/<domain>.py
from fastapi import APIRouter, Depends
from src.api.dependencies.auth import require_analyst

router = APIRouter(prefix="/api/v1/<domain>", tags=["domain"])

@router.get("/endpoint")
async def get_data(user=Depends(require_analyst)):
    ...
```

### How Celery Tasks Work

```python
# src/tasks/<task_module>.py
from src.workers.celery_app import celery_app

@celery_app.task(bind=True, queue="analytics")
def run_analysis(self, params: dict):
    ...
```

---

## 11. First Tasks

### Week 1: Get Oriented

1. **Run the full health check**
   ```bash
   ./scripts/health_check.sh
   ```
   Understand what each of the 24 services does and verify they're running.

2. **Explore the API docs**
   Open http://localhost:8000/api/docs (Swagger UI) and browse the 80+ endpoints.

3. **Run the test suite**
   ```bash
   make test-fast
   ```
   Understand what's being tested and the test organization.

4. **Read the architecture docs**
   - `docs/ARCHITECTURE.md` - System architecture with C4 diagrams
   - `config/agent_config.yaml` - All 21 agent definitions
   - `config/kpi_definitions.yaml` - 46+ KPI definitions

5. **Trace a query through the system**
   Follow how a natural language query flows:
   - `src/nlp/` - Query processing
   - `src/agents/orchestrator/` - Classification and routing
   - `src/agents/<target_agent>/` - Analysis execution
   - `src/api/routes/` - Response delivery

### Week 2: Make Small Changes

6. **Add a test for an existing agent**
   Pick any agent in `src/agents/` and write a unit test in `tests/unit/test_agents/`.

7. **Add a new KPI calculation**
   - Define it in `config/kpi_definitions.yaml`
   - Implement in `src/kpi/`
   - Add a test

8. **Fix a "good first issue"**
   Check GitHub Issues labeled `good-first-issue`.

### Week 3: Deeper Work

9. **Run the Tier 0 pipeline**
   ```bash
   .venv/bin/python scripts/run_tier0_test.py
   ```
   Understand the full ML pipeline from data generation through model deployment.

10. **Explore the causal engine**
    - Read `src/causal_engine/`
    - Run synthetic benchmarks: `pytest tests/synthetic/ -v`
    - Understand DoWhy refutation tests

### Ongoing

11. **Review PRs** - Read other developers' code to learn patterns
12. **Monitor Grafana dashboards** at http://localhost:3200
13. **Explore the knowledge graph** via FalkorDB Browser at http://localhost:3030

---

## 12. Tools & Access

### Required Accounts

| Tool | Purpose | Access |
|------|---------|--------|
| **GitHub** | Code, PRs, CI/CD | Repo collaborator access |
| **Supabase** | Database, auth | Project member |
| **Anthropic Console** | Claude API | API key |
| **Codecov** | Coverage tracking | Auto via GitHub |

### Development Tools

| Tool | Purpose | Install |
|------|---------|---------|
| **VS Code** | IDE | Recommended extensions: Python, Ruff, ESLint, Tailwind CSS IntelliSense |
| **Docker Desktop** | Container management | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **GitHub CLI** | PR management, issue tracking | `brew install gh` or [cli.github.com](https://cli.github.com/) |
| **HTTPie / curl** | API testing | `brew install httpie` |
| **DBeaver** | Database GUI | [dbeaver.io](https://dbeaver.io/) |

### VS Code Debugging

The dev overlay exposes debugpy on port 5678:

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to API",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "localhost", "port": 5678 },
      "pathMappings": [
        { "localRoot": "${workspaceFolder}/src", "remoteRoot": "/app/src" }
      ]
    }
  ]
}
```

---

## 13. Troubleshooting

### API Returns 502 Bad Gateway

API container not running or still starting:
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs api
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart api
```

### Frontend Shows Old Version

Vite HMR disconnected or browser cache:
```bash
# Hard refresh: Ctrl+Shift+R
# Or restart container:
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart frontend
```

### Workers Not Picking Up Code Changes

Celery workers don't auto-reload:
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml \
  restart worker_light worker_medium scheduler
```

### Tests Failing with Import Errors

Ensure you're using the project venv:
```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

### Redis / FalkorDB Connection Refused

Check containers are running:
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml ps redis falkordb
```

Redis is on port **6382** (external), FalkorDB on port **6381** (external). Inside Docker, both use port 6379 (internal).

### Pre-commit Hook Blocks Commit to Main

The `no-commit-to-branch` hook prevents direct commits to `main`. Create a feature branch instead:
```bash
git checkout -b feature/your-change
```

### Docker Compose "Port Already in Use"

Check for conflicting containers from other compose projects:
```bash
docker ps --format "{{.Names}}: {{.Ports}}" | grep <port>
```

### Out of Memory During Tests

Use the RAM-aware batched runner:
```bash
scripts/run_tests_batched.sh
```

Or reduce parallelism:
```bash
pytest tests/ -n 1  # Single worker
pytest tests/ -n 0  # Sequential
```

---

## 14. FAQ

**Q: Where are the environment variables documented?**
A: See `.env.example` at the project root. All required variables are listed in `docker/docker-compose.yml` (look for `${VAR:?error}` patterns).

**Q: How do I access the production droplet?**
A: Via SSH tunnel. Run `bash scripts/ssh-tunnels/tunnels.sh` from your local machine, then access services at `localhost` on their respective ports.

**Q: Why can't I push directly to main?**
A: Branch protection is enabled. Create a feature branch, push it, and open a PR. Pre-commit hooks also block commits to `main`.

**Q: How do I add a new API endpoint?**
A: Create a route in `src/api/routes/`, register it in `src/api/main.py`, add auth requirements via `Depends(require_analyst)`, and write tests in `tests/unit/test_api/`.

**Q: How do I add a new agent?**
A: Create a directory in `src/agents/<agent_name>/` with `graph.py` (LangGraph state machine), node functions, and tools. Add the agent definition to `config/agent_config.yaml`. Write tests in `tests/unit/test_agents/`.

**Q: Why is MyPy non-blocking?**
A: The codebase has `ignore_missing_imports = true` because many ML libraries lack type stubs. MyPy runs in CI for visibility but doesn't block merges.

**Q: How do I run just the causal engine tests?**
A: `pytest tests/unit/test_causal_engine/ -v` for unit tests, or `pytest tests/synthetic/ -v` for validation benchmarks.

**Q: What's the difference between worker_light, worker_medium, and worker_heavy?**
A: Light workers (x2, concurrency=4) handle quick API tasks. Medium (x1, concurrency=1) handles analytics/reports. Heavy (x0, on-demand) handles causal training and heavy ML. Heavy workers start at 0 replicas and scale up when needed.

**Q: How do I view agent traces?**
A: If Opik is running, access the dashboard at http://localhost:5173. Traces show hierarchical spans with timing, inputs, and outputs for each agent workflow.

**Q: How do I regenerate TypeScript types from the API?**
A: `cd frontend && npm run generate:types`. This reads the OpenAPI spec and generates `src/types/generated/api.ts`.

---

## Quick Reference Card

```bash
# Start everything
make docker-up

# Check health
./scripts/health_check.sh

# Run tests
make test           # With coverage
make test-fast      # Without coverage

# Lint & format
make lint           # Check
make format         # Fix

# View logs
make docker-logs

# Deploy
make deploy         # Quick (git pull + restart workers)
make deploy-build   # With image rebuild

# API docs
open http://localhost:8000/api/docs

# Monitoring
open http://localhost:3200  # Grafana (via tunnel)
open http://localhost:9091  # Prometheus (via tunnel)
```

---

*Last Updated: 2026-02-07*
