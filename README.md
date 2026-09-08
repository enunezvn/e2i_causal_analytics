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
[![OpenAPI Types](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/verify-types.yml/badge.svg)](https://github.com/enunezvn/e2i_causal_analytics/actions/workflows/verify-types.yml)

## Overview

E2I Causal Analytics is a sophisticated 22-agent, 6-tier agentic system designed for pharmaceutical companies to understand and optimize drug adoption through causal inference and natural language querying.

### Key Features

- **22 AI Agents** across 6 tiers (ML Foundation, Coordination, Causal Analytics, Monitoring, Predictions, Self-Improvement)
- **Tri-Memory Architecture** (Working, Episodic, Procedural, Semantic)
- **Causal Validation** with 5 DoWhy refutation tests
- **MLOps Integration** (MLflow, Feast, Great Expectations, Optuna, SHAP, BentoML)
- **Tiered LLM Factory** — provider-switchable (OpenAI default, Anthropic alternative) fast/standard/reasoning tiers with per-call usage metering — see [`docs/LLM_CONFIGURATION.md`](docs/LLM_CONFIGURATION.md)
- **Real-Time Model Interpretability** (v4.1) - SHAP explanations in 50-500ms via REST API
- **Digital Twin Engine** (v4.2) - A/B test pre-screening with ML-based simulations
- **Tool Composer** (v4.2) - Multi-faceted query decomposition & dynamic tool orchestration
- **Natural Language Interface** with typo-tolerant query processing
- **140+ Database Tables** across core, ML, memory, chat, audit, and RAG schemas
- **Hybrid RAG System** with vector + full-text + graph search
- **Full-Stack Dashboard** — React 18 + TypeScript + Vite with 31 pages
- **Production Observability** — LLM usage/cost tracking in the `/admin` Observability tab, always on; the Prometheus/Grafana/Loki/Alertmanager stack is **opt-in** behind the `monitoring` compose profile
- **In-app "How E2I Works"** — the platform explains itself at `/documentation` (predictive cohorts, intervention channels, agent tiers)

### Analyzed Brands

- **Remibrutinib** - BTK inhibitor for chronic spontaneous urticaria (CSU)
- **Fabhalta** - Factor B inhibitor for paroxysmal nocturnal hemoglobinuria (PNH)
- **Kisqali** - CDK4/6 inhibitor (ribociclib) for breast cancer

## Architecture

### 6-Tier Agent System

**TIER 0: ML FOUNDATION** (9 agents)
- scope_definer, cohort_constructor, cohort_profiler, data_preparer, feature_analyzer, model_selector, model_trainer, model_deployer, observability_connector

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
├── config/                    # YAML configurations (27 files)
│   ├── agent_config.yaml      # Agent definitions
│   ├── domain_vocabulary.yaml # Consolidated NLP vocabulary (v5.x)
│   ├── kpi_definitions.yaml   # 45 KPIs
│   └── ...
│
├── database/                  # SQL schemas (140+ tables)
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
│   ├── rwd/                   # Real-world data (CSU, Optum) — git-ignored
│   ├── training/              # fastText corpus
│   └── kg_cache/              # Committed Layer-2 KG cache, packaged into the API image
│                              # (rebuild via docs/runbooks/kg_cache.md)
│                              # Synthetic population data is generated on-demand (not stored)
│
├── src/                       # Main source code — 27 top-level packages
│   │                          # (orientation map: docs/ARCHITECTURE.md §3.0)
│   ├── agents/                # 22 agent implementations (6 tiers)
│   │   ├── orchestrator/      # Tier 1 coordination (4-stage classifier + router)
│   │   ├── tool_composer/     # Multi-faceted query decomposition & orchestration
│   │   ├── experiment_designer/ # Experiment design with Digital Twin pre-screening
│   │   ├── ml_foundation/     # 7 Tier 0 agents (scope, data, features, models)
│   │   └── ...                # 12 more agents (causal, monitoring, predictions, etc.)
│   ├── api/                   # FastAPI app, middleware stack, dependencies, route modules
│   ├── causal/                # Small shared statistics helpers (z-scores) — NOT the engine
│   ├── causal_engine/         # The causal engine: discovery, hierarchical CATE, IV, uplift,
│   │                          # refutation, energy-score validation, expert-review gate
│   ├── data/                  # Data access over the analytic tables (adaptive validity,
│   │                          # audit sidecars, causal-role classification, leakage checks)
│   ├── digital_twin/          # A/B test pre-screening with ML-based simulations
│   ├── etl/                   # Scheduled rollups feeding the analytic marts
│   ├── feature_store/         # Feast client + the lightweight Redis feature cache
│   ├── insights/              # Insight generation: causal/clinical context, narratives, labels
│   ├── kpi/                   # KPI registry calculators, cache, history backfill
│   ├── lifecycle/             # Gate lifecycle state machine shared by the quality gates
│   ├── memory/                # Tri-memory backends (working, episodic, procedural, semantic)
│   ├── ml/                    # Synthetic generators (synthetic/, synthetic_v2/, data_generator.py)
│   ├── mlops/                 # MLflow, Opik, Feast, BentoML, SHAP connectors
│   ├── nlp/                   # Query processing, entity extraction, typo handling
│   ├── ontology/              # Ontology YAML compilation, validation & inference
│   ├── optimization/          # DSPy prompt optimization, GEPA, lane A/B, shared DSPy LM config
│   ├── rag/                   # Hybrid RAG (vector + full-text + graph via FalkorDB)
│   ├── repositories/          # Supabase data-access layer, one repository per table family
│   ├── security/              # PHI scanning
│   ├── services/              # Cross-cutting app services (admin users, alert routing,
│   │                          # chat capability catalog, clinical context, cohort resolution)
│   ├── skills/                # Skill loading & matching for the agent skill packs
│   ├── tasks/                 # Celery task bodies — every beat_schedule entry lands here
│   ├── testing/               # In-tree quality gates and contract validators
│   ├── tool_registry/         # Tool discovery & management
│   ├── utils/                 # Shared primitives (audit chain, circuit breaker, env diagnostics)
│   └── workers/               # Celery app, beat_schedule SSOT, worker monitoring, consumers
│
├── tests/                     # unit, integration, e2e, api, rag, ml, performance, security,
│                              # stress, synthetic, benchmarks, ci, configs, fixtures, insights
├── scripts/                   # Utility scripts (deploy, health check, backups, migrations)
├── frontend/                  # React 18 + TypeScript + Vite dashboard (31 pages)
├── docs/                      # Comprehensive documentation
│   ├── ARCHITECTURE.md        # C4-model architecture documentation
│   ├── ONBOARDING.md          # Developer onboarding guide
│   ├── SYNTHETIC_DATA.md      # Synthetic data generation & validation reference
│   ├── api/                   # Hand-written API notes (chat, crystal digests); the
│   │                          # OpenAPI spec itself is generated on demand, not tracked
│   └── data/                  # Data dictionary & conversion docs
│       ├── 00-INDEX.md        # Master index & quick-start
│       ├── 01-08 *.md         # Schema docs (core, ML, graph, Feast, KPIs, leakage)
│       └── templates/         # CSV templates with example rows
└── docker/                    # Container configurations
```

## Recent Highlights (June–September 2026)

> Ongoing change tracking lives in [`CHANGELOG.md`](CHANGELOG.md), which is current through September 2026, and the decision log in [`docs/decisions/`](docs/decisions/README.md). The bullets below are a curated skim; the CHANGELOG is the record.

- **Observability is opt-in** — the Prometheus/Grafana/Loki/Alertmanager stack sits behind the `monitoring` compose profile (#1806), so a plain `up -d` starts the platform without it.
- **Maintenance-cron freshness alarm** — after the droplet's cron layer stopped silently for eight weeks, a freshness check keyed on completed-run stamps now runs unattended from GitHub Actions and files a tracking issue (#1799–#1807, #1810).
- **Guided causal discovery hardened** — prior-asserted DAGs are no longer reported as discovered, the gate scores bootstrap edge stability, FCI latent diagnostics annotate without gating, and the discovery UI gained question multiselect plus a cooperative Cancel (#1879, #1883, #1886, #1898, #1899).
- **Sampling-aware model-performance trend** — analytic standard errors and a skipped open walk-forward month, so a partial-month fold no longer reads as "degrading" (#1916).
- **Copilot suggestion pills from a capability catalog** — pills are generated from declared capabilities and filtered by a validator, so the chat stops offering what the platform cannot serve (#1900–#1919).

- **LLM model refresh + tier factory** — provider-switchable fast/standard/reasoning tiers (OpenAI `gpt-5.6-luna`/`gpt-5.6-terra` default, Anthropic `claude-haiku-4-5`/`claude-sonnet-5` alternative), `LLM_MODEL` deployment override, DSPy default `openai/gpt-5.6-terra`. See [`docs/LLM_CONFIGURATION.md`](docs/LLM_CONFIGURATION.md).
- **Admin LLM observability** — every factory LLM call meters tokens into `llm_usage_events`, priced at read time in the `/admin` Observability tab.
- **Feedback-learning loop live end-to-end** — golden-set replay → learning signals → pattern detection → gated prompt-update proposals.
- **Feature-importance stability gating** — adaptive SHAP sample sizing with a statistical stability criterion certifying the displayed covariate ranking.
- **KPI engine depth** — conversion-rate brand/segment/line-of-therapy/window routing, TRx share windows, clinical segment breakdowns, trend charting.
- **Causal analysis hardening** — RCT-aware estimator selection (ANCOVA), fail-closed refutation gating, E-value gate scoped to observational designs.
- **GHCR-based zero-on-box-build deploy** — images built in CI and pulled by the droplet, with feast-freshness and BentoML-readiness gates (see `DEPLOYMENT.md`).

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
- **Domain Vocabulary**: `config/domain_vocabulary.yaml` carries the Tool Composer vocabularies (`routing_patterns`, `dependency_types`, `tool_categories`) as documentation; the enum the classifier actually enforces is `RoutingPattern` in `src/agents/orchestrator/classifier/schemas.py`
- **Enhanced Orchestrator**: 4-stage classifier for intelligent query routing
- **Enhanced Experiment Designer**: Digital twin pre-screening tools integrated

## Quick Start

### Prerequisites

- Docker Engine 24+ and Docker Compose v2
- Supabase account (or self-hosted Supabase)
- OpenAI API key (the default LLM provider; Anthropic is the optional alternative — see [`docs/LLM_CONFIGURATION.md`](docs/LLM_CONFIGURATION.md))

All services (API, frontend, workers, Redis, FalkorDB, MLflow) run in Docker containers via Docker Compose. The observability stack (Prometheus, Grafana, Loki, Promtail, Alertmanager, node/postgres exporters) is **opt-in** behind the `monitoring` compose profile; `falkordb-browser` is behind `debug`.

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

   To bring up the observability stack as well:
   ```bash
   COMPOSE_PROFILES=monitoring docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
   ```

   Optional overlays (Opik observability — currently not run in production — and debug tools) are described in `DEPLOYMENT.md`.

3. **Verify services are running**
   ```bash
   docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml ps
   curl http://localhost:8000/health
   ```

4. **Initialize database**
   ```bash
   # Applies every pending file under database/ (migrations, memory, core, ml,
   # causal, chat, rag, audit) in order, tracked in public.schema_migrations.
   # Connection auto-detects: SUPABASE_DB_URL if set, else docker-exec into
   # the supabase-db container. See docs/runbooks/migrations.md.
   ./scripts/run_migrations.sh --dry-run   # list pending
   ./scripts/run_migrations.sh             # apply
   ```

5. **Generate synthetic data**
   ```bash
   make data-generate
   # Or: python src/ml/data_generator.py
   ```

See `docker/README.md` for Docker Compose configuration and `DEPLOYMENT.md` for setup instructions.

## CI/CD Workflows

All workflows live in `.github/workflows/` and run on GitHub Actions. The main ones:

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| Backend Tests | `backend-tests.yml` | Push/PR | pytest with coverage gate + MyPy error-count ceiling |
| Frontend Tests | `frontend-tests.yml` | Push/PR | Vitest + coverage thresholds |
| Tier 1-5 Agent Harness | `tier1-5-test.yml` | Every PR + weekly Monday cron | Agent-tier integration harness (required check). Path gating lives **inside** a `changes` job so the required context always reports and never sits Pending (#1445) |
| Deploy | `deploy.yml` | Push to main (path-filtered) | CI image build+push to GHCR, then gated droplet deploy with auto-rollback |
| Security | `security.yml` | Push/PR + daily cron | Bandit, pip-audit, Semgrep, gitleaks secrets scan, `npm audit` (frontend) |
| Verify OpenAPI Types | `verify-types.yml` | Push/PR (path-filtered) | Regenerates the OpenAPI spec, Spectral lint, frontend type-drift check |
| RAGAS Fixture Regression | `ragas-evaluation.yml` | **Manual only** | Judge-drift sentinel on FROZEN input — scores a static fixture, **not** production quality (#1485). gpt-4o judge; CI-key throughput-bound, see #504 |
| Synthetic Benchmarks | `synthetic-benchmarks.yml` | Push/PR | Causal engine benchmark suite |
| Slow Tests | `slow-tests.yml` | Nightly 05:00 UTC + manual | `pytest -m slow` and the heavy e2e suites excluded from the PR run; upstream-transient reds are routed to a rolling issue rather than the red alarm (#1816, #1823) |
| Maintenance Freshness | `maintenance-freshness.yml` | Nightly 07:30 UTC + manual | Unattended audit of the droplet's `e2i-maintenance` cron layer from its success stamps; files/updates a tracking issue on failure (#1807) |
| Deploy Currency | `deploy-currency.yml` | Every 3h + manual | Is production actually running `main`? Reads the running container (not job conclusions) and alarms when a deploy-triggering push is past its grace window and still not live — the case a skipped deploy chain makes invisible (#1962) |

That table is the operator-facing subset. `ls .github/workflows/*.yml` is the full list (24 today) and also covers the guard and benchmark workflows: feature contract, lifecycle state, RPC DDL, methodology sign-off (two workflows), G3 wiring, lockfile resolution, retrieval benchmarks, general benchmarks, Feast apply, RAGAS smoke, and the tier-1b B2 diagnostic/experiment pair.

> **Measuring real RAG quality.** `ragas-evaluation.yml` never invokes the RAG
> pipeline — it judges the golden set's hardcoded answers over contexts that are
> byte-identical to the reference, so its context metrics are 1.0-by-construction
> and its faithfulness/answer-relevancy score the fixture's own prose (#1485). It
> is useful only as a judge-drift signal on frozen input. For production quality,
> replay through the live pipeline and judge the real output:
>
> ```bash
> .venv/bin/python scripts/replay_golden_set.py --limit 12 \
>     --record-out /tmp/goldset_records.json
> .venv/bin/python scripts/run_real_pipeline_ragas.py \
>     --records /tmp/goldset_records.json --fail-on-threshold
> ```
>
> Run it on a host that can reach the pipeline (GitHub's runners cannot), at
> n≈10–15 per the #504 throughput constraint.

## Operational Scripts

Key scripts in `scripts/`:

**Core Operations**
- `deploy.sh` — Manual deploy path (git pull, restart workers, seed FalkorDB, health check); the normal production deploy is CI's `deploy.yml`, which does NOT invoke this script
- `health_check.sh` — Probes every service a default `up` starts, deriving the set from `docker compose config --services`; profile-gated services (monitoring, debug, dev-tools) report SKIPPED and Opik reports UNHEALTHY by design. Also reports the maintenance-cron freshness verdict
- `run_migrations.sh` — Ledger-tracked migration runner over all `database/` dirs; auto-detects `SUPABASE_DB_URL` vs docker-exec into `supabase-db`; run unconditionally by every deploy
- `backup_data_stores.sh` — Backup Redis, FalkorDB, MLflow artifacts
- `backup_cron.sh` — Scheduled backup wrapper

**Testing**
- `run_tests_batched.sh` — Full test suite in 43 batches (~20 min)
- `run_frontend_tests_batched.sh` — Frontend test suite in batches

**Maintenance (`scripts/maintenance/`, driven by the droplet's `e2i-maintenance` crontab)**
- `check_maintenance_freshness.sh` — Are the cron jobs actually running? Answers from their success stamps, with each interval derived from the crontab itself. Deliberately **not** in the crontab it audits — see [`docs/runbooks/maintenance-cron.md`](docs/runbooks/maintenance-cron.md)
- `memory_monitor.sh` — Memory relief valve (`--auto-cleanup`) on a box that runs prod and dev together
- `cleanup_orphans.sh`, `docker_cleanup.sh` — Orphaned-process and Docker-artifact reaping
- `setup_cron.sh`, `resize_swap.sh`, `harden_ssh.sh` — One-shot box setup helpers

**Data & models**
- `reseed_synthetic.sh` — Staged synthetic reseed (stages in `scripts/lib/reseed_stages.sh`) — see [`docs/runbooks/synthetic_reseed.md`](docs/runbooks/synthetic_reseed.md)
- `promote_hcp_adoption_champions.py` — Calibrate and promote the goldstd hcp_adoption models to champion
- `rag/ingest_chunk_corpus.py` — Populate the chat-RAG chunk corpus in the embedding space the retriever queries
- `gen_kpi_catalog.py` — Regenerate the KPI catalog consumed by chat charting
- `build_kg_cache.py` — Rebuild the committed Layer-2 KG cache — see [`docs/runbooks/kg_cache.md`](docs/runbooks/kg_cache.md)

**Evaluation**
- `replay_golden_set.py` + `run_real_pipeline_ragas.py` — Replay the golden set through the live pipeline and judge the real output (the honest RAG-quality measurement; see the note above)

**Infrastructure**
- `deploy/check_image_drift.py` — Fail-loud drift check between the built image and what the droplet runs — see [`docs/runbooks/deploy-operations.md`](docs/runbooks/deploy-operations.md)
- `opik-manager.sh` — Start/stop/status for the Opik overlay (intentionally stopped in production)
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
make lint           # ruff check src/ tests/ + whole-tree mypy
make format         # black src/ tests/ + ruff check --fix
make generate-types # Export openapi.json, regenerate frontend/src/types/generated/api.ts
make clean          # Clean build artifacts
```

> `make lint` runs whole-tree mypy, which peaks around 1.6 GiB — do not run it on the
> droplet, where production and development share one box. Scope local checks to the files
> you changed (`mypy <file>`) and let CI's `Type Check (MyPy)` job be the arbiter. CI's
> formatter gate is `ruff format --check`, which is separate from `make format`'s `black`.
>
> Run `make generate-types` whenever a response model changes — the frontend type-drift
> check in `verify-types.yml` fails otherwise.

### Running Tests

```bash
pytest tests/ -v --cov=src
```

## Key Components

### Real-Time Model Interpretability (v4.1)

**SHAP Explanations API**
- 7 REST endpoints under `/api/explain`: `/predict`, `/predict/batch`, `/history/{patient_id}`, `/models`, `/sample-entities`, `/global`, `/health`
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
# The public API base is /api, not /api/v1 (src/api/main.py)
app.include_router(explain_router, prefix="/api")

# Import chat tools
from src.agents.orchestrator.tools.explain_tool import ExplainIntentHandler

# Import SHAP engine
from src.mlops.shap_explainer_realtime import RealTimeSHAPExplainer
```

See `docs/ARCHITECTURE.md` for SHAP integration details.

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

See `docs/data/05-FEATURE-STORE-REFERENCE.md` for feature definitions and `docs/ARCHITECTURE.md` for integration details.

### LLM Configuration & Observability

**Tiered LLM Factory** (`src/utils/llm_factory.py`)
- Three tiers — `fast` (classification/routing), `standard` (chat/synthesis), `reasoning` (complex analysis)
- Provider-switchable via `LLM_PROVIDER`: **OpenAI is the default** (`gpt-5.6-luna` / `gpt-5.6-terra`), Anthropic the alternative (`claude-haiku-4-5` / `claude-sonnet-5`)
- Deployment-level model pinning via `LLM_MODEL` (no code change)
- DSPy paths resolve separately through `src/optimization/dspy_lm.py` (default `openai/gpt-5.6-terra`)

**LLM Usage Observability**
- Every factory-built model meters tokens into the `llm_usage_events` table, attributed to the authenticated user
- Costs computed at read time (`src/services/llm_pricing.py`) so pricing corrections apply retroactively; unknown models render "unpriced", never a silent default
- Surfaced in the `/admin` page's **Observability** tab (`/api/admin/observability/llm-usage`)

See [`docs/LLM_CONFIGURATION.md`](docs/LLM_CONFIGURATION.md) for the full reference.

**Note on Opik**: the codebase retains an Opik tracing integration (`src/mlops/opik_connector.py`, `docker/docker-compose.opik.yml`), but the Opik stack is **intentionally stopped in production** (2026-05-29). Agent/LLM observability in production is the `llm_usage_events` path above plus Prometheus/Grafana/Loki.

### Tri-Memory System

**Working Memory** (Redis)
- Session state, messages, evidence board
- TTL: 86400 seconds (24 hours)

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

Each test carries a three-state verdict — **passed** | **warning** | **failed** (#1869) — and
tests that could not run are reported in `skipped_tests` with a reason rather than silently
counted as passes.

Gate decisions: **proceed** | **review** | **block**

### Query Robustness

3-layer natural language processing:
1. **fastText normalization** - Handle typos via subword embeddings
2. **rapidfuzz matching** - Fuzzy match against domain vocabulary
3. **LLM disambiguation** - Resolve complex/ambiguous queries

## Database

140+ tables across 8 categories:
- **Core Data**: patient_journeys, hcp_profiles, treatment_events, triggers, business_metrics, agent_tier_mapping, etc.
- **ML Pipeline** (60+): experiments, model registry, digital twins, causal validation, A/B testing, GEPA, cohort constructor, etc.
- **Memory** (7): episodic_memories, procedural_memories, semantic_memory_cache, cognitive_cycles, etc.
- **RAG** (2): rag_document_chunks (HNSW), rag_search_logs
- **Chat** (10+): chat_threads, chat_messages, user_preferences (RLS), chatbot analytics, feedback, training signals
- **Audit** (3): audit_chain_entries (SHA-256 hash chain), audit_chain_verification_log, security_audit_log (partitioned)
- **FalkorDB Graph**: 10 entity types, 11 relationship types (`E2IEntityType` / `E2IRelationshipType` in `src/memory/graphiti_config.py`)
- **Feast Feature Store**: see [`docs/data/05-FEATURE-STORE-REFERENCE.md`](docs/data/05-FEATURE-STORE-REFERENCE.md) for the current feature views and fields (derive with `grep -c 'FeatureView(' feature_repo/features/*.py`)

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
  - [KPI Reference](docs/data/06-KPI-REFERENCE.md) — All KPIs with formulas & thresholds
  - [Supporting Schemas](docs/data/07-SUPPORTING-SCHEMAS.md) — Memory, RAG, Chat, Audit
  - [Leakage Detection Contract](docs/data/08-LEAKAGE-DETECTION-CONTRACT.md) — What counts as leakage and how it is checked
  - [CSV Templates](docs/data/templates/) — Ready-to-use templates with example rows
- **LLM Configuration**: [`docs/LLM_CONFIGURATION.md`](docs/LLM_CONFIGURATION.md) — Provider default, model tiers, overrides, usage metering & pricing
- **API Reference**: the OpenAPI 3.0 spec is generated on demand (`make api-docs` / `make generate-types`) and per-PR in CI (`verify-types.yml`) — the generated spec is not tracked in git, but `docs/api/` does track hand-written notes ([`chat.md`](docs/api/chat.md), [`crystal_digests.md`](docs/api/crystal_digests.md))
- **Runbooks**: [`docs/runbooks/`](docs/runbooks/) — operator procedures:
  - [`migrations.md`](docs/runbooks/migrations.md) — how migrations apply (auto on deploy + manual path)
  - [`deploy-operations.md`](docs/runbooks/deploy-operations.md) — the CI deploy path, sha selection, image drift, rollback
  - [`maintenance-cron.md`](docs/runbooks/maintenance-cron.md) — the droplet's `e2i-maintenance` crontab and its freshness alarm
  - [`synthetic_reseed.md`](docs/runbooks/synthetic_reseed.md) — staged synthetic reseed
  - [`kg_cache.md`](docs/runbooks/kg_cache.md) — rebuilding the committed Layer-2 knowledge-graph cache
  - [`sentinels.md`](docs/runbooks/sentinels.md) — graph-emptiness and related self-healing sentinels
  - [`frontend-env-and-csp.md`](docs/runbooks/frontend-env-and-csp.md) / [`frontend-serving-flip.md`](docs/runbooks/frontend-serving-flip.md) — frontend build env, CSP, and serving mode
  - [`gotrue-smtp.md`](docs/runbooks/gotrue-smtp.md) / [`reviewer-provisioning.md`](docs/runbooks/reviewer-provisioning.md) — auth mail and reviewer accounts
- **Developer Reference**: `CLAUDE.md` — Quick reference for AI-assisted development

## Tech Stack

| Category | Technologies |
|----------|-------------|
| AI/ML | LangGraph, LangChain, DSPy, OpenAI GPT-5.x (default), Claude (Anthropic, alternative) |
| Causal | DoWhy, EconML, NetworkX |
| MLOps | MLflow, Optuna, SHAP, BentoML, Great Expectations |
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

**Version**: 4.2.1 (`pyproject.toml`; there is no tagged release — see [`CHANGELOG.md`](CHANGELOG.md))
**Last Updated**: September 2026
**Recent**: opt-in observability profile, maintenance-cron freshness alarm, guided causal-discovery hardening + discovery UX, sampling-aware model-performance trend, capability-catalog copilot pills (see Recent Highlights above)
