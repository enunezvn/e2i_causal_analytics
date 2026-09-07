# E2I Causal Analytics - Developer Onboarding Guide

**Healthcare Engagement Intelligence Platform** (v4.2.1)
22-Agent, 6-Tier Causal Analytics System for Pharmaceutical Drug Adoption Analysis

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

The system uses the agent roster in `config/agent_config.yaml` (22 agents today), organized in 6 tiers, to perform causal analysis, predict drug adoption, design experiments, and explain model outputs in natural language.

### Key Capabilities

- **Causal Inference**: DoWhy refutation tests, EconML treatment effect estimation, CausalML uplift modeling
- **Natural Language Interface**: Typo-tolerant query processing (fastText + rapidfuzz + Claude)
- **Digital Twin Engine**: A/B test pre-screening with ML-based simulations
- **Real-Time Explainability**: SHAP explanations in 50-500ms via REST API
- **Knowledge Graph**: FalkorDB temporal graph with Cypher queries
- **Full MLOps**: MLflow tracking, Feast features, BentoML serving, in-database LLM usage tracking (`llm_usage_events` + `/admin` → Observability)

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

| Service | Variable | Required? | How to Get |
|---------|----------|-----------|------------|
| **OpenAI** | `OPENAI_API_KEY` | **Always** | [platform.openai.com](https://platform.openai.com). The default provider is OpenAI (`LLM_PROVIDER` defaults to `openai` in `src/utils/llm_factory.py`), and RAG embeddings (`text-embedding-3-small`, `src/rag/config.py`) are OpenAI **regardless** of `LLM_PROVIDER`. Also used by the RAGAS evaluations. |
| **Supabase** | `SUPABASE_URL`, `SUPABASE_KEY` | **Always** | Self-hosted Supabase — see `config/supabase_self_hosted.example.env` |
| **Anthropic** | `ANTHROPIC_API_KEY` | With `LLM_PROVIDER=anthropic` | [console.anthropic.com](https://console.anthropic.com). Independently of the provider it also gates (fail-open — the feature disables itself, nothing errors) the nightly routing-label judge (`src/tasks/routing_label_tasks.py`) and the Layer-4 adaptive-validity evaluator (`src/data/causal_role_evaluator.py`). |

> **Do not configure only `ANTHROPIC_API_KEY`.** The code default is
> `LLM_PROVIDER=openai`, so an Anthropic-only `.env` leaves the configured provider
> with no key. Per-provider model tiers (`fast` / `standard` / `reasoning`) are
> defined in `src/utils/llm_factory.py`.
>
> **As deployed on the droplet**, both keys are set: `LLM_PROVIDER=anthropic` **and**
> `DSPY_LM_MODEL=openai/gpt-5.6-terra` — the LangChain agents run on Anthropic while
> the DSPy modules stay on OpenAI. See
> [`docs/decisions/adr-010-dspy-terra-scoped-anthropic-flip.md`](decisions/adr-010-dspy-terra-scoped-anthropic-flip.md)
> and `docs/LLM_CONFIGURATION.md`.

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
OPENAI_API_KEY=sk-...          # always required (default provider + RAG embeddings)
ANTHROPIC_API_KEY=sk-ant-...   # required when LLM_PROVIDER=anthropic
SUPABASE_URL=http://172.17.0.1:54321
SUPABASE_KEY=eyJhbG...
SUPABASE_SERVICE_KEY=eyJhbG...

# Passwords (choose strong values, no defaults allowed)
REDIS_PASSWORD=your-redis-password
FALKORDB_PASSWORD=your-falkordb-password
SUPABASE_POSTGRES_PASSWORD=your-supabase-db-password
GRAFANA_ADMIN_PASSWORD=your-grafana-password

# Database
SUPABASE_DB_URL=postgresql://postgres:PASSWORD@127.0.0.1:5432/postgres
```

> **`SUPABASE_POSTGRES_PASSWORD` is mandatory and easy to miss.** It is the password of
> the self-hosted `supabase-db` container (the same value as `POSTGRES_PASSWORD` in
> `/opt/supabase/docker/.env`). Compose builds the *container-internal*
> `SUPABASE_DB_URL` from it (`docker/docker-compose.yml`) and also injects it into the
> Feast services and `postgres-exporter`.
>
> Compose hard-fails on four `${VAR:?…}`-enforced variables — `REDIS_PASSWORD`,
> `FALKORDB_PASSWORD`, `SUPABASE_POSTGRES_PASSWORD` and `GRAFANA_ADMIN_PASSWORD`. The
> Grafana one is checked even when the `monitoring` profile is off, because `config`
> evaluates every service definition. Verify your `.env` before starting anything:
>
> ```bash
> docker compose --env-file .env -f docker/docker-compose.yml config -q   # rc 0 = OK
> grep -o '\${[A-Z_]*:?' docker/docker-compose.yml | sort -u             # the live list
> ```

> **Note**: `docker/.env` is a symlink to `../.env` so Docker Compose picks up these values automatically.

### Step 3: Start All Services (Docker)

```bash
# Start core services (API, frontend, workers, Redis, FalkorDB, MLflow, BentoML, Feast)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# Observability is OPT-IN: prometheus, alertmanager, node-exporter, postgres-exporter,
# loki, promtail and grafana sit behind the `monitoring` compose profile, so a plain
# `up -d` does NOT start them.
COMPOSE_PROFILES=monitoring \
  docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# falkordb-browser is behind the `debug` profile; Flower and Redis Commander are
# `dev-tools` in the dev overlay.

# Legacy (do NOT start by default): the Opik observability overlay.
# Opik was intentionally stopped in May 2026 — LLM usage tracking now lives in
# the llm_usage_events table (see /admin → Observability). The overlay is kept
# for reference only:
# docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml \
#   -f docker/docker-compose.opik.yml up -d
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

# Full health check
./scripts/health_check.sh
```

> `health_check.sh` derives its service list at runtime from
> `docker compose config --services`, so the total is not a fixed number. Anything
> behind a profile you did not start reports **SKIPPED**, not a failure. Opik reports
> **UNHEALTHY by design** — it was intentionally stopped in May 2026 and its containers
> are not started.

### Step 5: Set Up Local Python Environment (for tests & linting)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make dev-install        # pip install -r requirements.txt, THEN pip install -e ".[dev]"
pre-commit install
```

> **Install `requirements.txt` first.** `pip install -e ".[dev]"` on its own resolves
> unpinned tool versions and can give you a different toolchain from CI, which installs
> `requirements.txt` first — exactly what `make dev-install` does.

### Step 6: Set Up Frontend Development (optional)

```bash
cd frontend
npm ci
cd ..
```

### Step 7: Initialize Database

Use the migration runner. It is the **same script every deploy runs**
(`.github/workflows/deploy.yml`), so a fresh box ends up with exactly the deployed
schema:

```bash
./scripts/run_migrations.sh --dry-run   # list pending files, apply nothing
./scripts/run_migrations.sh             # apply every pending file
```

`scripts/run_migrations.sh`:

- Applies **every** `.sql` file, in alphabetical order, from the eight schema
  directories listed in its `MIGRATION_DIRS` array — `database/migrations`, `memory`,
  `core`, `ml`, `causal`, `chat`, `rag`, `audit`. Each directory contributes a key
  prefix so identically-numbered files in different directories (e.g. `ml/011` vs
  `migrations/011`) never collide in the ledger.
- Records what it applied in `public.schema_migrations`, so it is **idempotent** —
  re-running it is a clean no-op.
- Skips non-forward-migration files by name (`rollback_*.sql`, `*_rollback.sql`,
  `*_validation_queries.sql`); those DROP live objects and must never be auto-applied.
- Auto-detects its connection: `SUPABASE_DB_URL` if set (CI / remote), otherwise
  `docker exec` into `$SUPABASE_DB_CONTAINER` (default `supabase-db`) — which is how it
  runs **on the droplet**, where the self-hosted Supabase stack exposes REST
  credentials but no `SUPABASE_DB_URL`.

> **Do not hand-apply a subset with `psql`.** Picking individual files leaves a partial
> schema *and* an empty `public.schema_migrations`, so the next
> `./scripts/run_migrations.sh` cannot tell what is already there.
>
> **`make db-init` is a stub** — it only echoes "Run your database initialization
> scripts here" and touches no database.

Re-derive the directory list and the file counts at any time:

```bash
sed -n '/^MIGRATION_DIRS=(/,/^)/p' scripts/run_migrations.sh
for d in migrations memory core ml causal chat rag audit; do
  printf '%s: %s\n' "$d" "$(ls database/$d/*.sql 2>/dev/null | wc -l)"
done
```

### Step 8: Generate Synthetic Data

```bash
make data-generate
# Equivalent to the two steps it runs, in order:
#   python src/ml/data_generator.py   # generate
#   python src/ml/data_loader.py      # load into Supabase
```

---

## 4. Project Overview

### Tech Stack

| Category | Technologies |
|----------|-------------|
| **AI/ML** | LangGraph, LangChain, DSPy, OpenAI + Anthropic (see `src/utils/llm_factory.py`), scikit-learn, LightGBM |
| **Causal Inference** | DoWhy, EconML (CausalForestDML), CausalML (UpliftRandomForest), NetworkX |
| **MLOps** | MLflow, Feast, BentoML, Great Expectations, Optuna, SHAP (Opik: stopped May 2026) |
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
│   ├── agents/             # 22 LangGraph agent implementations
│   ├── api/                # FastAPI app, routes, middleware
│   ├── causal_engine/      # EconML, CausalML, DoWhy integration
│   ├── digital_twin/       # A/B test simulation engine
│   ├── memory/             # Tri-memory architecture
│   ├── rag/                # Hybrid RAG (vector + full-text + graph)
│   ├── mlops/              # MLflow, Opik, Feast, BentoML connectors
│   ├── workers/            # Celery task definitions
│   ├── nlp/                # Query processing, entity extraction
│   ├── feature_store/      # Feature store client
│   ├── kpi/                # KPI implementations (registry: config/kpi_definitions.yaml)
│   └── utils/              # Shared utilities
├── frontend/               # React/TypeScript/Vite frontend
│   └── src/
│       ├── pages/          # Route-level pages (31 page components)
│       ├── components/     # Shared React components
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
│   ├── agent_config.yaml   # 22-agent definitions
│   ├── kpi_definitions.yaml
│   └── observability.yaml
├── database/               # SQL schemas — 8 dirs applied by scripts/run_migrations.sh
│   ├── migrations/         # Numbered cross-cutting migrations
│   ├── core/               # Core data tables
│   ├── ml/                 # ML / MLOps tables
│   ├── memory/             # Tri-memory tables + the FalkorDB graph schema
│   ├── causal/ chat/ rag/  # Domain schemas
│   └── audit/              # Audit trail (hash-chained)
├── docker/                 # Docker Compose & Dockerfiles
│   ├── docker-compose.yml       # Base (21+ services)
│   ├── docker-compose.dev.yml   # Dev overlay (hot-reload)
│   └── docker-compose.opik.yml  # Opik overlay (legacy — stopped May 2026)
├── scripts/                # Operational scripts (~140 files)
├── feature_repo/           # Feast feature definitions
├── .github/workflows/      # 20+ CI/CD workflows
├── CLAUDE.md               # AI assistant instructions
├── DEPLOYMENT.md           # Deployment guide
└── pyproject.toml          # Python tool configuration
```

---

## 5. Architecture Deep Dive

### 6-Tier Agent System

```
TIER 0: ML Foundation (9 agents)
  scope_definer → cohort_constructor → data_preparer → feature_analyzer
  → model_selector → model_trainer → model_deployer → observability_connector
  cohort_profiler (dispatched by the orchestrator, not on the sequential chain)

TIER 1: Coordination (2 agents)
  orchestrator (intent classification + routing. Classification is a staged
                rule-based pipeline in src/agents/orchestrator/classifier/
                — feature extraction → domain mapping → dependency detection →
                pattern selection — with a fast-tier LLM fallback used only for
                ambiguous queries (Haiku on Anthropic, the fast GPT tier on
                OpenAI; see src/utils/llm_factory.py). Routes dependent
                multi-part queries to tool_composer)
  tool_composer (multi-part query decomposition: sub-questions + dependency DAG)

TIER 2: Causal Analytics (3 agents)
  causal_impact (DoWhy refutation)
  gap_analyzer (ROI opportunity)
  heterogeneous_optimizer (CATE estimation)

TIER 3: Monitoring (4 agents)
  drift_monitor, experiment_designer (digital twin), experiment_monitor, health_score

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

> **These are the dev-overlay names.** The `_dev` suffix and the auto-reload column come
> from `docker-compose.dev.yml`. **Production runs the base compose only**, where the
> same services are `e2i_api` (gunicorn, no reload) and `e2i_frontend` (nginx serving
> the built bundle on `3002:80`) — see [section 8](#8-deployment). Prometheus and
> Grafana carry no `_dev` suffix in either set, but a plain `up -d` does not start them
> at all: they sit behind the `monitoring` compose profile.

### Worker Queue Architecture

| Worker | Replicas | Concurrency | Queues | Memory limit | Use Case |
|--------|----------|-------------|--------|--------------|----------|
| **worker_light** | 2 | 2 | `default`, `quick`, `api` | 1.5G | Fast tasks (<30s) |
| **worker_medium** | 1 | 2 | `analytics`, `reports`, `aggregations` | 4G | Medium analysis (1–5 min) |
| **worker_heavy** | 0 (on-demand, scales 0–4) | 1 | `shap`, `causal`, `ml`, `twins` | 3G | Heavy ML — `--time-limit=3600`, `--soft-time-limit=3300` |
| **scheduler** | 1 | - | celery-beat | 1G | Periodic tasks |

Re-derive with `grep -nE 'queues=|concurrency=|replicas:' docker/docker-compose.yml`.

### Tri-Memory System

| Layer | Storage | Purpose | TTL |
|-------|---------|---------|-----|
| **Working** | Redis | Session state, messages, evidence board | 3600s |
| **Episodic** | Supabase + pgvector | User queries, agent actions, events | Permanent |
| **Procedural** | Supabase + pgvector | Tool sequences, query patterns | Permanent |
| **Semantic** | FalkorDB | Entity nodes, relationships, causal chains | Permanent |

### Database Schema

- **140+ tables** across PostgreSQL (Supabase) with pgvector extension
- **Core Data** (19): patient_journeys, hcp_profiles, treatment_events, triggers, business_metrics, etc.
- **ML Pipeline** (60+): experiments, model registry, digital twins, causal validation, A/B testing, GEPA, etc.
- **Memory** (7): episodic_memories, procedural_memories, semantic_memory_cache, cognitive_cycles, investigation_hops, learning_signals, memory_statistics
- **RAG** (2): rag_document_chunks (HNSW), rag_search_logs
- **Chat** (6+): chat_threads, chat_messages, user_preferences (RLS)
- **Audit** (2): audit_chain_entries (SHA-256 hash chain), verification_log
- **FalkorDB Graph**: the ontology in `config/ontology/` declares 8 node types
  (`node_types.yaml`) and 15 edge types (`edge_types.yaml`); the deployed memory graph
  schema (`database/memory/002_semantic_graph_schema.cypher`) uses a wider set —
  12 labels and 18 relationship types — because it adds the memory/causal layer on top
- **Feast Feature Store**: 11 feature views / 65 fields over 7 PostgreSQL sources
  (`feature_repo/features/*.py`), served by `feastdev/feature-server` pinned in
  `docker/Dockerfile.feast`. Canonical reference:
  [`docs/data/05-FEATURE-STORE-REFERENCE.md`](data/05-FEATURE-STORE-REFERENCE.md)

> **Full data documentation**: See [`docs/data/00-INDEX.md`](data/00-INDEX.md) for the complete data dictionary, conversion guide, and CSV templates for onboarding real data.

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
- **detect-secrets** (Yelp baseline scan — a new secret-looking string fails the commit
  until it is removed or audited into `.secrets.baseline`)

### PR Requirements

1. The two required status checks must pass: **Backend CI Success** and **Tier 1-5 agent harness**
2. **0 approvals required** (solo-dev repo) - there is no CODEOWNERS gate and stale reviews are not auto-dismissed
3. Merge policy: **always preserve history** via `--merge` merge-commits, **never squash** (branch protection keeps `required_linear_history=false` so merge commits stay legal)

> **Not a footgun any more**: the two required workflows are **deliberately not**
> path-filtered — both `backend-tests.yml` and `tier1-5-test.yml` carry a
> `DELIBERATELY NOT path-filtered` banner, and the path gating moved *inside* a
> `changes` job. The required contexts therefore always report, and a docs-only or
> frontend-only PR merges normally. `enforce_admins=false` remains as an escape hatch.
> See `scripts/setup_branch_protection.sh` for the full applied policy and rationale.

### Code Style

| Tool | Purpose | Config Location |
|------|---------|----------------|
| **Ruff** | Linting, formatting, and the CI formatter gate (`ruff format --check src/ tests/`) | `pyproject.toml` [tool.ruff] |
| **Black** | Python formatting — run by `make format` only; CI checks Ruff's formatter, not Black | `pyproject.toml` [tool.black] |
| **MyPy** | Type checking — **enforced** in CI against a `MYPY_CEILING` inside the required `Backend CI Success` gate | `pyproject.toml` [tool.mypy] |
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
make lint           # ruff check src/ tests/ + whole-tree mypy src/  (see caveat below)
make format         # black src/ tests/ + ruff check --fix src/ tests/
make generate-types # Export OpenAPI + regenerate frontend/src/types/generated/api.ts
make docker-up      # Start all Docker services
make docker-down    # Stop all Docker services
make docker-logs    # Tail API + frontend logs
make api-docs       # Generate OpenAPI spec + Redoc HTML
make clean          # Remove build artifacts
```

> ⚠️ **`make lint` runs whole-tree `mypy src/`, which spikes ~1.6 GiB.** Do not run it
> on the droplet (dev and prod are the same box — see [section 8](#8-deployment)).
> Scope local type checks to the files you changed (`mypy <changed_file.py>`) and let
> CI's `Type Check (MyPy)` gate be the arbiter; read its `mypy-report` artifact for the
> actual errors.

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
| **RAGAS (fixture)** | OpenAI + RAGAS | `scripts/run_ragas_eval.py` | Sequential | ~10 min |
| **RAGAS (real pipeline)** | Live API + OpenAI | `scripts/run_real_pipeline_ragas.py` | Sequential | ~20 min |
| **Tier 0** | Custom runner | `scripts/run_tier0_test.py` | Sequential | ~20 min |
| **Tier 1-5** | Custom runner | `scripts/run_tier1_5_test.py` | Sequential | ~15 min |

> **The two RAGAS runs measure different things (#1485).**
> `run_ragas_eval.py` never invokes the RAG pipeline — it scores the golden
> set's hardcoded answers over contexts byte-identical to the reference, so its
> context metrics are 1.0-by-construction and its faithfulness/answer-relevancy
> score the fixture's own prose. Treat it as a judge-drift sentinel on frozen
> input, **not** a quality gate.
> `run_real_pipeline_ragas.py` judges what the pipeline actually generated over
> what it actually retrieved. It needs a host that can reach the live API, and
> runs at n≈10–15 on demand (#504 throughput constraint):
>
> ```bash
> .venv/bin/python scripts/replay_golden_set.py --limit 12 \
>     --record-out /tmp/goldset_records.json
> .venv/bin/python scripts/run_real_pipeline_ragas.py \
>     --records /tmp/goldset_records.json --fail-on-threshold
> ```

### Running Tests

```bash
# Standard test suite (4 parallel workers, 30s timeout per test)
.venv/bin/pytest tests/

# With coverage (backend gate: pyproject.toml fail_under)
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

Available markers (authoritative list: the `markers = [...]` block in `pyproject.toml`):
`unit`, `integration`, `e2e`, `slow`, `requires_redis`, `requires_falkordb`,
`requires_supabase`, `heavy_ml`, `xdist_group`, `real_data` (needs real CSU/Optum cohort
data on disk), `benchmark` (retrieval Recall@10 / MRR — excluded from the default sweep),
`live_llm` and `live_lm` (need a live LM key; CI-skipped without one), and
`real_supabase` (opt out of the unit-tree dead-Supabase pin — read-only checks only).

### Test Environment Guarantees

**Unit tests can never touch a real Supabase.** An autouse fixture in
`tests/unit/conftest.py` pins the whole unit tree to dead credentials —
`SUPABASE_URL=http://127.0.0.1:1` (a reserved port that is never listening, so any
attempt fails immediately with ECONNREFUSED) plus placeholder keys. This matters
specifically on the droplet, where dev and prod are the same box and CI's nominal
`localhost:54321` is the **live production** Supabase. The pin is locked by
`tests/unit/test_utils/test_unit_tree_dead_supabase_1420.py`. Two deliberate escape
hatches exist: `@pytest.mark.real_supabase` (read-only, reachability-gated checks only)
and a per-test `monkeypatch.setenv`.

**Stall and crash guards.** `--timeout` only arms a timer *inside* an xdist worker's
runtest protocol, so a stall in the controller — or in a worker holding the GIL in
native code — has no timer at all and burns the job's whole timeout budget with no
diagnosis. Two controller-side guards in `tests/conftest.py` cover that:

- a session inactivity watchdog, inert unless `E2I_PYTEST_STALL_TIMEOUT` is set (opt-in
  per lane, because a safe window depends on that lane's longest per-test budget, which
  ranges from 30s to 2700s here);
- an xdist crash guard that refuses a green exit when a worker dies during collection —
  xdist synthesises no failure for a worker that was running no item, so pytest would
  otherwise return 0 with nothing run.

### Tier Tests (ML Pipeline Validation)

```bash
# Tier 0: Full ML pipeline (generates 1500 patients, caches output)
.venv/bin/python scripts/run_tier0_test.py

# Tier 1-5: Test all 13 agents using cached Tier 0 output
.venv/bin/python scripts/run_tier1_5_test.py

# Run specific tiers
.venv/bin/python scripts/run_tier1_5_test.py --tiers 2,3

# Run specific agents
.venv/bin/python scripts/run_tier1_5_test.py --agents causal_impact,explainer
```

### Coverage Thresholds

| Component | Lines | Branches | Functions | Statements | Config |
|-----------|-------|----------|-----------|------------|--------|
| **Backend** | `fail_under = 20` | — | — | — | `pyproject.toml` |
| **Frontend** | 62% | 55% | 54% | 62% | `frontend/vitest.config.ts` |

> The backend has a **single line-coverage gate**, re-baselined down to 20 in April 2026
> so the gate reflects the tree instead of blocking every PR; 70% remains the
> aspiration, not the enforced number. Check the live value with
> `grep -n '^fail_under' pyproject.toml`.

---

## 8. Deployment

### Architecture

Dev and prod are the **same machine** — a single DigitalOcean droplet
(8 vCPU / 16 GB RAM). All services run via Docker Compose. Host nginx handles SSL
termination.

> The box runs under memory pressure. **Do not run whole-tree `mypy` or the full
> `pytest` suite on it** — a whole-tree mypy spikes ~1.6 GiB. Scope local checks to the
> files you changed; CI is the arbiter (see `CLAUDE.md`).

### What production actually runs

**Production is the base `docker/docker-compose.yml` alone — no overlay.** The deploy
workflow's `pick_overlay()` returns an empty overlay because
`docker/frontend/Dockerfile` has a `FROM nginx:alpine AS production` stage; the two
other branches (`docker-compose.frontend-dev.yml`, `docker-compose.dev.yml`) are
rollback targets from earlier eras. Live containers are `e2i_api` (gunicorn with
UvicornWorker, `read_only: true`, GHCR image tagged by commit sha, `8000:8000`) and
`e2i_frontend` (nginx serving the pre-built bundle, `3002:80`).

**Local development** is base + `docker-compose.dev.yml`: `e2i_*_dev` container names,
`uvicorn --reload`, and Vite HMR on `3002:5173`.

There are seven compose files in `docker/` (`docker-compose.yml`, `.dev`,
`.frontend-dev`, `.monitoring`, `.opik`, `.rxnav`, `.secure`); only the base one is
deployed.

### Deploy Process

**Deploys happen only by merging to `main`**, which runs `.github/workflows/deploy.yml`
(build + push images to GHCR, apply DB migrations, flip the app services, health-check,
roll back to the previous sha on failure). To redeploy the current `main` without a new
commit:

```bash
gh workflow run deploy.yml

# Verify (read-only, safe on the droplet)
./scripts/health_check.sh
```

> ⚠️ **`make deploy` / `make deploy-build` / `./scripts/deploy.sh` is a legacy
> local-dev path — never run it on the droplet.** It starts the **dev overlay**
> (`docker-compose.dev.yml`), skips the feast/bentoml gates, does
> `git reset --hard origin/main` and `git checkout <sha>` in the checkout it runs from
> — and the droplet checkout is the shared deploy target. Use `deploy.yml`.

In production nothing hot-reloads: the API and frontend images are rebuilt and the
containers replaced by the workflow. `uvicorn --reload` and Vite HMR only exist in the
dev overlay, where Celery workers still need an explicit restart (Celery does not
auto-reload).

### CI/CD Pipeline

Push to `main` triggers the deploy workflow:

1. **Backend Tests** (lint, type-check, unit tests, integration tests)
2. **Build & Push** API image to GHCR
3. **Build & Push** Frontend image to GHCR
4. **SSH Deploy** to droplet: apply DB migrations (`scripts/run_migrations.sh`), pull
   the new sha-tagged images, recreate the app services, run the health check, and roll
   back to the previous sha if any gate fails

### Accessing Services (via SSH Tunnel)

```bash
# Start all tunnels
bash scripts/ssh-tunnels/tunnels.sh

# Or minimal tunnel for frontend
ssh -N -L 8443:localhost:443 enunez@138.197.4.36
```

| Service | Local URL | Started by |
|---------|-----------|------------|
| Frontend | https://localhost:8443 | default `up` |
| API Docs | https://localhost:8443/api/docs | default `up` |
| MLflow | http://localhost:5000 | default `up` (nginx `auth_basic` in front) |
| Supabase Studio | http://localhost:3001 | the separate self-hosted Supabase stack |
| Grafana | http://localhost:3200 | `monitoring` profile |
| Prometheus | http://localhost:9091 | `monitoring` profile |
| Alertmanager | http://localhost:9093 | `monitoring` profile |
| FalkorDB Browser | http://localhost:3030 | `debug` profile |
| Flower / Redis Commander | — | `dev-tools` profile (dev overlay only) |
| Opik (stopped May 2026 — only if the legacy overlay is manually started) | http://localhost:5173 | not started |

All management ports bind to `127.0.0.1` on the droplet, so they are reachable only
through the tunnel. Profile-gated services are **not running unless you started their
profile** — a refused connection on 3200/9091/9093/3030 usually means the profile is
off, not that the service is broken.

---

## 9. Security & Compliance

### Authentication

- **JWT-based** via Supabase Auth
- Tokens are verified by calling Supabase — `verify_supabase_token` →
  `client.auth.get_user()`, which needs `SUPABASE_URL` + `SUPABASE_ANON_KEY`.
  `SUPABASE_JWT_SECRET` is **not** used on this path; missing URL/anon-key is the real
  "auth disabled" condition (`src/api/dependencies/auth.py`)
- Testing mode auto-bypasses auth (`E2I_TESTING_MODE=true`), and only when
  `ENVIRONMENT != production`

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
| Calculations (`/calculate`, and any POST) | 30 req | 60s |
| Batch operations | 10 req | 60s |
| Health checks | 300 req | 60s |
| CopilotKit chat | 30 req | 1 hour |
| CopilotKit status/info | 100 req | 60s |
| CopilotKit other (analytics, feedback) | 60 req | 60s |

Source of truth: `DEFAULT_LIMITS` in `src/api/middleware/rate_limit_middleware.py`
(`/health`, `/healthz`, `/ready`, `/metrics` and `/api/kpis/health` are exempt).

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
| API routes | `src/api/routes/` (220+ endpoints) |
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
| CI/CD workflows | `.github/workflows/` (20+ YAML files) |

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

router = APIRouter(prefix="/<domain>", tags=["domain"])

@router.get("/endpoint")
async def get_data(user=Depends(require_analyst)):
    ...
```

The public base path is **`/api`**, not `/api/v1`: `src/api/main.py` mounts routers with
`include_router(..., prefix="/api")`. Declare the domain segment on the router and let
`main.py` supply `/api`. The one historical exception is `src/api/routes/rag.py`, which
declares its own `/api/v1/rag`. Confirm with:

```bash
grep -rhoE 'prefix="/api[^"]*"' src/api/routes/ src/api/main.py | sort | uniq -c
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
   Understand what each reported service does and verify they're running. The probe
   set is derived from `docker compose config --services`; profile-gated services show
   as SKIPPED and Opik shows as UNHEALTHY by design.

2. **Explore the API docs**
   Open http://localhost:8000/api/docs (Swagger UI) and browse the 220+ endpoints.

3. **Run the test suite**
   ```bash
   make test-fast
   ```
   Understand what's being tested and the test organization.

4. **Read the architecture docs**
   - `docs/ARCHITECTURE.md` - System architecture with C4 diagrams
   - `config/agent_config.yaml` - All 22 agent definitions
   - `config/kpi_definitions.yaml` - the KPI registry (45 KPIs today; `summary.total_kpis` is authoritative)

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
   Check GitHub Issues labeled `good first issue` (with spaces — `gh issue list --label "good first issue"`).

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
12. **Monitor Grafana dashboards** at http://localhost:3200 — start the stack first:
    `COMPOSE_PROFILES=monitoring docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d`
13. **Explore the knowledge graph** via FalkorDB Browser at http://localhost:3030 —
    behind the `debug` profile (`COMPOSE_PROFILES=debug … up -d`)

---

## 12. Tools & Access

### Required Accounts

| Tool | Purpose | Access |
|------|---------|--------|
| **GitHub** | Code, PRs, CI/CD | Repo collaborator access |
| **Supabase** | Database, auth | **Self-hosted** on the droplet — there is no hosted project to be added to. Credentials come from `/opt/supabase/docker/.env`; Studio is reachable at http://localhost:3001 through the SSH tunnel |
| **OpenAI Platform** | Default LLM provider + RAG embeddings | API key |
| **Anthropic Console** | Claude API (`LLM_PROVIDER=anthropic`, routing judge, adaptive-validity evaluator) | API key |
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
A: Create a route in `src/api/routes/` with `APIRouter(prefix="/<domain>")` — **not** `/api/v1/...`; `src/api/main.py` supplies the `/api` mount when it registers the router. Add auth requirements via `Depends(require_analyst)`, write tests in `tests/unit/test_api/`, and if the change alters a response model, run `make generate-types` and commit the regenerated `frontend/src/types/generated/api.ts` in the same PR.

**Q: How do I add a new agent?**
A: Create a directory in `src/agents/<agent_name>/` with `graph.py` (LangGraph state machine), node functions, and tools. Add the agent definition to `config/agent_config.yaml`. Write tests in `tests/unit/test_agents/`.

**Q: Is MyPy blocking?**
A: Yes, as a *ceiling*. The codebase has `ignore_missing_imports = true` because many ML libraries lack type stubs, so instead of demanding zero errors, `backend-tests.yml` pins a `MYPY_CEILING` and fails the required `Backend CI Success` gate if the error count exceeds it. Read the `mypy-report` artifact for the actual errors, and check the current ceiling with `grep -n 'MYPY_CEILING=' .github/workflows/backend-tests.yml`.

**Q: How do I run just the causal engine tests?**
A: `pytest tests/unit/test_causal_engine/ -v` for unit tests, or `pytest tests/synthetic/ -v` for validation benchmarks.

**Q: What's the difference between worker_light, worker_medium, and worker_heavy?**
A: Light workers (2 replicas, concurrency 2) handle the `default`/`quick`/`api` queues. Medium (1 replica, concurrency 2) handles `analytics`/`reports`/`aggregations`. Heavy (concurrency 1) handles `shap`/`causal`/`ml`/`twins` with a 3600s hard time limit; it starts at **0 replicas** and scales up on demand (see the `HEAVY_OFFLOAD_ENABLED` note in `docker/docker-compose.yml`).

**Q: How do I view agent traces / LLM usage?**
A: Per-call LLM usage (model, tokens, cost, latency, agent) is recorded in the `llm_usage_events` table and surfaced in the frontend at `/admin` → Observability. (Opik, the former tracing stack, was intentionally stopped in May 2026; its overlay remains in `docker/docker-compose.opik.yml` for reference only.)

**Q: How do I regenerate TypeScript types from the API?**
A: `make generate-types`. It exports the OpenAPI spec statically (`python -m scripts.export_openapi --output openapi.json`) and regenerates `frontend/src/types/generated/api.ts` — no running API needed. This is what CI's verify-types gate runs, so **commit the regenerated `api.ts` in the same PR** as the response-model change that caused it.

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

# Deploy (merge to main; or re-run the workflow on the current main)
gh workflow run deploy.yml
# NEVER `make deploy` on the droplet — legacy dev-overlay path, see section 8

# API docs
open http://localhost:8000/api/docs

# Monitoring (start it first: COMPOSE_PROFILES=monitoring docker compose ... up -d)
open http://localhost:3200  # Grafana (via tunnel)
open http://localhost:9091  # Prometheus (via tunnel)
```

---

*Last Updated: 2026-09-07*
