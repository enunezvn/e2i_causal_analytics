# E2I Causal Analytics - Deployment Guide

How to run the full stack locally using Docker Compose with the dev overlay, plus
how the real production deploy works (see [Production Deploy (CI/CD)](#production-deploy-cicd)).

Production runs the **base compose file alone** and is deployed **only** by merging
to `main`. The `_dev` container names and dev-overlay commands below are the local
ones — see [What production actually runs](#what-production-actually-runs) for the
deployed shape.

---

## Prerequisites

- **Docker Engine 24+** with Docker Compose v2+
- **Git**
- **8 GB+ RAM** recommended (PyTorch + ML dependencies are heavy)

## Quick Start

```bash
# 1. Clone the repository
git clone git@github.com:enunezvn/e2i_causal_analytics.git
cd e2i_causal_analytics

# 2. Create environment file from template
cp .env.example .env
# Edit .env — fill in required keys (see Environment Variables below)

# 3. Start all services
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# 4. Verify services are running
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml ps

# 5. Check API health
curl -s http://localhost:8000/health | python3 -m json.tool
```

First build pulls PyTorch + ML dependencies — expect 10-15 minutes. Subsequent starts use cached layers.

---

## Environment Variables

### Required (must set in `.env`)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key — the **default** LLM provider (`gpt-5.6-terra` standard/reasoning, `gpt-5.6-luna` fast). RAG embeddings (`text-embedding-3-small`) are OpenAI *regardless* of `LLM_PROVIDER`, so this key is always required |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anonymous key |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |
| `SUPABASE_POSTGRES_PASSWORD` | Password of the self-hosted `supabase-db` container — mirrors `POSTGRES_PASSWORD` in `/opt/supabase/docker/.env`. Compose derives the container-internal `SUPABASE_DB_URL` from it and **refuses to start without it** |
| `REDIS_PASSWORD` | Redis authentication password |
| `FALKORDB_PASSWORD` | FalkorDB authentication password |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password — still required with the `monitoring` profile **off** (compose interpolates every service before it filters by profile) |
| `SUPABASE_DB_URL` | Host-side PostgreSQL connection string — **not** forwarded into containers (see below) |

Four of these are `:?`-enforced in `docker/docker-compose.yml`, so compose exits
before starting anything if they are unset: `REDIS_PASSWORD`, `FALKORDB_PASSWORD`,
`SUPABASE_POSTGRES_PASSWORD` and `GRAFANA_ADMIN_PASSWORD`. Everything else fails at
runtime instead. Validate a candidate `.env` without starting anything:

```bash
# rc 0 = every :?-enforced variable has a value
docker compose --env-file .env -f docker/docker-compose.yml config -q

# the enforced set itself, derived rather than remembered
grep -o '\${[A-Z_]*:?' docker/docker-compose.yml | sort -u
```

### Optional LLM configuration

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `openai` (code default, `src/utils/llm_factory.py`) or `anthropic` |
| `ANTHROPIC_API_KEY` | Required only with `LLM_PROVIDER=anthropic`. It also gates two paths that are Anthropic-only whatever the provider, both fail-open: the nightly routing-label judge (`src/tasks/routing_label_tasks.py`) and the Layer-4 adaptive-validity evaluator (`src/data/causal_role_evaluator.py`) |
| `LLM_MODEL` | Pin the OpenAI standard/reasoning model without a code change |
| `DSPY_LM_MODEL` | Verbatim litellm model string for the DSPy/GEPA lane; takes precedence over `LLM_PROVIDER` there |

**As deployed on the droplet** the two are deliberately split — the factory lane
runs Anthropic while the DSPy/GEPA lane stays pinned to OpenAI (ADR-010):

```
LLM_PROVIDER=anthropic
DSPY_LM_MODEL=openai/gpt-5.6-terra
```

So production needs **both** keys, and "`LLM_PROVIDER=anthropic`, therefore no
OpenAI key" is wrong twice over — the DSPy pin and the embeddings both need it.

See `docs/LLM_CONFIGURATION.md` for tiers, model mappings, and overrides.

### How a `.env` value reaches the containers

**Compose forwards no `.env` file wholesale.** The `x-common-env` anchor in
`docker/docker-compose.yml` is a *whitelist*: a host variable reaches `api`,
`worker_*` and `scheduler` only if that anchor names it. Setting anything else in
`.env` is a silent no-op inside the containers — the in-code default governs and
nothing warns you. This has bitten the platform repeatedly (`OPIK_ENABLED`,
`OPENAI_API_KEY`, the three biomedical keys), and the compose comments say so at
each entry.

Derive the current list rather than trusting any copy of it, this one included:

```bash
# every host variable the anchor forwards
sed -n '/^x-common-env:/,/^x-common-worker:/p' docker/docker-compose.yml \
  | grep -o '\${[A-Z_0-9]*' | tr -d '${' | sort -u
```

At the time of writing that yields 39 variables.

#### Derived inside the network — do not set these

These are computed by compose from the passwords above; a host-side value for the
same name is ignored.

| Variable | Docker value |
|----------|--------------|
| `SUPABASE_DB_URL` | `postgresql://postgres:${SUPABASE_POSTGRES_PASSWORD}@supabase-db:5432/postgres` |
| `REDIS_URL` | `redis://:${REDIS_PASSWORD}@redis:6379/0` |
| `FALKORDB_URL` | `redis://:${FALKORDB_PASSWORD}@falkordb:6379/0` |
| `CELERY_BROKER_URL` | `redis://:${REDIS_PASSWORD}@redis:6379/1` |
| `CELERY_RESULT_BACKEND` | `redis://:${REDIS_PASSWORD}@redis:6379/2` |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` |
| `BENTOML_SERVICE_URL` | `http://bentoml:3000` |
| `FEAST_URL` | `http://feast:6566` |
| `OPIK_URL` | `http://opik-backend:8080` |
| `ENVIRONMENT` / `LOG_LEVEL` | hardcoded `production` / `INFO` — the host values feed only `config-check` |
| `ADAPTIVE_VALIDITY_ARTIFACTS_DIR` | `/app/data/audit_artifacts` |
| `SUPABASE_ANON_KEY` | alias of `SUPABASE_KEY` |

Note the `SUPABASE_DB_URL` split: the host `.env` value points at the Supavisor
pooler and is used by host-side tools (`scripts/run_migrations.sh` psql mode);
inside the network compose connects to `supabase-db` directly as `postgres`.

#### Runtime knobs forwarded with a compose default

Every one of these is optional — the compose default is what production runs
unless the host `.env` overrides it. `Ref` is the issue/PR the compose comment
cites; read that comment for the full rationale.

| Variable | Compose default | Purpose | Ref |
|----------|-----------------|---------|-----|
| `OPIK_ENABLED` | `true` | Opt-**out** switch for Opik tracing. Production keeps `false` in the host `.env` so the tracers skip client construction against the intentionally-stopped `opik-backend` | #952 |
| `ORCHESTRATOR_CLASSIFIER_MODE` | `shadow` | 4-stage query classifier: `off` \| `shadow` (classify + log, legacy routing) \| `active` (classifier routes when confident) | — |
| `E2I_REQUIRE_FULL_AGENT_REGISTRY` | `false` | Arm to turn a partial agent registry into a raised `PartialAgentRegistryError`. A one-boot post-deploy verification, not a steady-state setting | #1448 |
| `GUNICORN_PRELOAD` | `true` | Gunicorn preload — the stream-tear fix (workers fork warm instead of each re-importing the app on its event loop). Kill switch: `false`. Only the `api` service reads it | #1560 |
| `CHATBOT_STARTUP_WARM_ENABLED` | `true` | Pre-build the DSPy LM config, retrieval clients, orchestrator registry and intent classifier per worker at boot, fail-open | #1454 |
| `CHATBOT_STARTUP_WARM_LLM_ENABLED` | `true` | The warm's two synthetic-LLM legs (classify + RAG rewriter); ~2 small completions per worker per boot. `false` keeps the construction-only warm | #1475 |
| `CHATBOT_RAG_LLM_TIMEOUT_S` | `20` | Fail-open ceiling per `retrieve_rag` chain LLM call (rewrite / score / hop-decider) | #1484 |
| `CHATBOT_RAG_DRY_HOP_LIMIT` | `2` | Consecutive zero-new-keep hops before the multi-hop loop stops; `0` = legacy run-to-max | #1484 |
| `CHATBOT_RAG_REWRITE_COT` | *(empty)* | Empty = Predict-only rewriter; `true` restores ChainOfThought | #1518 |
| `CHATBOT_RAG_SKIP_EMPTY_DECIDER` | *(empty)* | Empty/`true` = skip the decider LLM call when hop-1 kept nothing; `false` restores the LLM decider | #1518 |
| `ROUTING_LABEL_MIN_NEW_ROWS` | `10` | Skip the nightly routing labeler below this many unlabeled `classification_logs` rows | #1341 |
| `ROUTING_LABEL_JUDGE_CAP` | `50` | Max LLM-judge calls per labeler run (token-spend bound) | #1341 |
| `ROUTING_LABEL_JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Labeler judge model (needs `ANTHROPIC_API_KEY`; fail-open) | #1341 |
| `ROUTING_LABEL_LOOKBACK_DAYS` | `30` | How far back the labeler looks for unlabeled rows | #1341 |
| `DSPY_RAG_RECORDS_PATH` | *(empty)* | Records file for the nightly RAG-prompt GEPA leg. **Resolved inside the container** — must be a path under a named volume the service mounts read-write (`/app/optimized_modules` is the natural pick); a host path skips forever while looking configured | #1486 |
| `DSPY_RAG_MAX_METRIC_CALLS` | *(empty)* | Judge-call budget for that leg. Empty on purpose: the in-code default (40) is the single source of truth | #1486 |
| `DSPY_RAG_DB_FEEDSTOCK_ENABLED` | *(empty)* | Live-traffic feedstock — the only way the nightly cycle runs unattended. Off by default because enabling it spends judge budget; parsed fail-closed | #1489 |
| `DSPY_RAG_DB_LOOKBACK_DAYS` | *(empty)* | Read window in days for that feedstock (in-code default 30) | #1489 |
| `CHATBOT_OPT_DRAIN_ENABLED` | *(empty)* | Gate for the nightly chatbot DSPy optimization queue drainer; unset = drain skipped. Enabling it spends real GEPA budget | #1515 |
| `CHATBOT_OPT_DRAIN_MAX_PER_CYCLE` | *(empty)* | GEPA executions per drain cycle (in-code default 1) | #1515 |
| `CHATBOT_OPT_STALE_HOURS` | *(empty)* | Staleness bound for the drainer (in-code default 168) | #1515 |
| `CHATBOT_OPT_ZOMBIE_HOURS` | *(empty)* | Zombie bound for the drainer (in-code default 12) | #1515 |
| `CHATBOT_OPT_MIN_SIGNALS` | *(empty)* | Producer minimum signals (in-code default 50) | #1515 |
| `E2I_KPI_INCLUDE_SYNTHETIC` | `0` | KPI reads swap to the `*_include_synthetic` twins and the frontend badges the figures as synthetic | — |
| `E2I_INCLUDE_SYNTHETIC` | `0` | Deployment-wide synthetic showcase switch: every read-path chokepoint (`apply_provenance_filter`) and the orchestrator resolver include synthetic. Reversible — unset restores the strict gate | — |
| `LLM_PROVIDER` | `openai` | Factory-lane provider (see Optional LLM configuration above) | — |
| `LLM_MODEL` | *(empty)* | OpenAI workhorse pin | — |
| `DSPY_LM_MODEL` | *(empty)* | Verbatim litellm model string for the DSPy lane; takes precedence there | — |

#### Optional external biomedical API credentials (soft-degrade)

All three use `:-`, so an unset host value never blocks container start — it
buys a quieter, smaller degradation instead.

| Variable | Absent | Present |
|----------|--------|---------|
| `NCBI_API_KEY` | PubMed E-utilities on the anonymous ~3 req/s tier. Measured over 8 rapid `esearch` calls, same host and params: 5 of 8 throttled (HTTP 429) | 0 of 8 throttled |
| `UMLS_UTS_API_KEY` | `src/data/kg/umls_uts.py` raises `UMLSAuthError`; `CitationResolver` catches it and degrades to `umls=None`, disabling synonym expansion — genuine supporting citations can score as unverified | Synonym expansion on |
| `OPENFDA_API_KEY` | Unauthenticated client, which returns no rate-limit headers at all | openFDA reports `x-ratelimit-limit: 240` |

#### Read by code but NOT forwarded — setting these in `.env` is inert in containers

Same whitelist rule, other direction. Each of these is read via `os.environ` in
application code but does not appear in `x-common-env`, so a host `.env` value
never reaches the containers and the in-code default governs. Verify with
`grep -c '<VAR>' docker/docker-compose.yml` (comment-only hits do not count).

| Variable | In-code default | Where it is read |
|----------|-----------------|------------------|
| `SEGMENT_ANALYSIS_BUDGET_SECONDS` | `900.0` | `src/api/routes/segments.py` |
| `AGENT_COMPUTE_EXECUTOR_WORKERS` | `1` | `src/api/dependencies/compute.py` |
| `HEAVY_COMPUTE_EXECUTOR_WORKERS`, `HEAVY_COMPUTE_MAX_CONCURRENCY`, `HEAVY_OFFLOAD_ENABLED` | see module | `src/api/dependencies/compute.py` |
| `ADAPTIVE_CRITERIA` | `true` | ML training path — so the documented rollback switch is inert in containers |
| `ADAPTIVE_VALIDITY_EVALUATOR_ENABLED`, `ADAPTIVE_VALIDITY_EVALUATOR_MODEL` | off / Haiku 4.5 | Layer-4 evaluator |
| `ANTHROPIC_MODEL` | in-code default | Deliberately not forwarded — the host `.env` pins an id meant for interactive use, not the platform lanes |
| `CORS_ORIGINS` | — | No reader in `src/` at all |
| `SUPABASE_JWT_SECRET` | — | `src/api/dependencies/auth.py` logs about it, but verification goes through `client.auth.get_user()`; currently optional and unused |

Changing any of those on the droplet needs a compose change, not an `.env` change.

---

## Service Map

| Service | Port | URL | Notes |
|---------|------|-----|-------|
| API (FastAPI) | 8000 | http://localhost:8000 | Auto-reloads under the dev overlay only |
| API Docs | 8000 | http://localhost:8000/docs | Swagger UI |
| Frontend | 3002 | http://localhost:3002 | Vite HMR under the dev overlay; nginx on the built bundle in production |
| MLflow | 5000 | http://localhost:5000 | 127.0.0.1 only |
| Redis | 6382 | redis://localhost:6382 | 127.0.0.1 only |
| FalkorDB | 6381 | redis://localhost:6381 | 127.0.0.1 only |
| BentoML | 3000 | http://localhost:3000 | 127.0.0.1 only |
| Feast | 6567 | http://localhost:6567 | 127.0.0.1 only |
| Grafana* | 3200 | http://localhost:3200 | `monitoring` profile; 127.0.0.1 only |
| Prometheus* | 9091 | http://localhost:9091 | `monitoring` profile; 127.0.0.1 only |
| Loki* | 3101 | http://localhost:3101 | `monitoring` profile; 127.0.0.1 only |
| Alertmanager* | 9093 | http://localhost:9093 | `monitoring` profile; 127.0.0.1 only |
| Flower* | 5555 | http://localhost:5555 | `dev-tools` profile (dev overlay); 127.0.0.1 only |
| FalkorDB Browser* | 3030 | http://localhost:3030 | `debug` profile; 127.0.0.1 only |
| Redis Commander* | 8081 | http://localhost:8081 | `dev-tools` profile (dev overlay); 127.0.0.1 only |

\* Requires `--profile`. Management ports (127.0.0.1 only) need SSH tunnels for remote access — see `scripts/ssh-tunnels/`.

**The observability stack is opt-in.** Since #1806 prometheus, alertmanager,
node-exporter, postgres-exporter, loki, promtail and grafana sit behind the
`monitoring` profile, so a plain `up -d` does not start them and **no deploy step
does either**. Start them deliberately:

```bash
COMPOSE_PROFILES=monitoring docker compose -f docker/docker-compose.yml up -d
```

`scripts/health_check.sh` derives its probe set from `docker compose config
--services`, so it reports profile-gated services as SKIPPED while the profile is
off and re-arms automatically when it is on — nothing to toggle in two places.

---

## Production Deploy (CI/CD)

### What production actually runs

Production is the **base `docker/docker-compose.yml` alone — no overlay**.
`deploy.yml`'s `pick_overlay()` returns the empty string because
`docker/frontend/Dockerfile:106` is `FROM nginx:alpine AS production`; the
`docker-compose.frontend-dev.yml` branch below it belongs to the #528-A rollback
era and the `docker-compose.dev.yml` branch to the pre-flip #527 dev-in-prod era.
Neither is reachable today.

The live app containers are `e2i_api` (gunicorn, `--workers 2` with
`uvicorn.workers.UvicornWorker`, `read_only: true`, a GHCR image tagged by commit
sha, `8000:8000`) and `e2i_frontend` (nginx serving the built bundle, `3002:80`).
**Local dev** is base + `docker-compose.dev.yml`: `e2i_*_dev` container names,
`uvicorn --reload`, Vite HMR on `3002:5173`. The names in the Service Map and
Common Commands sections above are the dev-overlay ones; on the droplet drop the
`_dev` suffix.

```bash
# what is actually running right now
docker ps --format '{{.Names}} {{.Image}} {{.Ports}}' | grep -E '^e2i_(api|frontend) '
sed -n '419,427p' .github/workflows/deploy.yml    # pick_overlay()
```

### How it is deployed

Only by merging to `main`, which runs `.github/workflows/deploy.yml` — the
workflow file is the source of truth; what follows is the operator-level summary.

`scripts/deploy.sh`, and therefore `make deploy` / `make deploy-build`, is the
**legacy local-dev path**: it composes with the dev overlay
(`scripts/deploy.sh:25`), skips the feast / health / bentoml gates, does its own
`git reset --hard origin/main` (`:77`) and rolls back with `git checkout <sha>`
(`:136`) inside the checkout. **Never run it on the droplet** — the droplet
checkout is the deploy target, and both of those git operations move it. To
redeploy the current `main`, re-run the workflow instead:

```bash
gh workflow run deploy.yml
```

**Trigger**: push to `main`, path-filtered to deploy inputs (`src/`, `config/`,
compose files, `frontend/`, `requirements*`/`pyproject.toml`/`requirements.lock`,
`patches/`, BentoML serving inputs). A docs-only merge does NOT deploy.

**Pipeline** (`test` → `build-and-push` + `build-and-push-frontend` → `deploy`):

1. **Images build in CI, not on the droplet.** The app and frontend images are
   built and pushed to GHCR, tagged with the commit SHA. The droplet pulls them
   (`--no-build`); a local build happens only as a fallback when the GHCR pull
   fails. This keeps the OOM-prone React production build off the box.
2. **Hard sync**: the droplet checkout is `git reset --hard origin/main`. The
   deploy **aborts** if any *tracked* file has uncommitted changes (a live
   hot-patch it would clobber); untracked files never block it. The droplet is
   a deploy target, not a dev box — don't leave tracked edits on it.
3. **Migrations apply automatically**: `scripts/run_migrations.sh` runs
   **unconditionally** on every deploy. It auto-detects the connection
   (`SUPABASE_DB_URL` if set, else docker-exec into the `supabase-db`
   container), covers every `database/` schema dir, and tracks applied files in
   `public.schema_migrations`. See `docs/runbooks/migrations.md`.
4. **Ordered rollout with gates**:
   - `feast` + `feast-materializer` recreate first; the deploy waits (up to
     10 min) for a **fresh materialize heartbeat** before the app is allowed
     to flip — the API must never serve against a stale/empty online store.
   - The app tier (`api`, `frontend`, `worker_*`, `scheduler`) then flips to
     the GHCR-pulled images, followed by a 30-attempt `/health` check loop.
   - When a serving input changed, the `bentoml` container is recreated and
     gated on `POST /model_info` reporting a non-empty `available_models` —
     proving the cohort bundles actually loaded, not just that the server is up.
5. **Automatic rollback**: every gate failure rolls the affected services back
   to the pre-deploy SHA (app tier re-pulled from GHCR at the old SHA — no
   local rebuild) and fails the deploy loudly.
6. **Post-deploy prune**: unreferenced images + build cache are pruned
   (historically grew to 100% disk without this).

**Not part of the CI deploy**: FalkorDB seeding (manual, see below) and
synthetic data reseeds.

---

## Development Workflow

### Hot Reload

| Component | Auto-reloads? | How |
|-----------|---------------|-----|
| API (Python) | Yes | `uvicorn --reload` watches `/app/src` |
| Frontend (React) | Yes | Vite HMR via bind-mounted `frontend/src` |
| Workers (Celery) | **No** | Restart manually (see below) |
| Scheduler (Beat) | **No** | Restart manually |

### Restarting Workers

After changing Python code that runs in workers:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart worker_light worker_medium scheduler
```

### FalkorDB Seeding

`scripts/seed_falkordb_all.sh` seeds the FalkorDB knowledge graph from Supabase
core tables. **The CI/CD deploy does NOT run it** — `deploy.yml` never invokes
`scripts/deploy.sh`, the legacy local-dev path that chains into the seeder (see
[How it is deployed](#how-it-is-deployed)). Reseed the graph manually when needed:

```bash
./scripts/seed_falkordb_all.sh
```

### docker/.env Symlink

Docker Compose at `docker/docker-compose.yml` does not auto-find the root `.env` file. A symlink is required:

```bash
cd docker && ln -sf ../.env .env
```

This is already set up on the droplet. If you get empty variable errors when starting containers, check that this symlink exists.

### Running Tests

```bash
# Full suite (4 parallel workers)
.venv/bin/pytest tests/

# With coverage
.venv/bin/pytest tests/ --cov --cov-report=term-missing

# Single file
.venv/bin/pytest tests/unit/test_agents/test_orchestrator.py -v

# Sequential (for debugging)
.venv/bin/pytest tests/unit/test_some_test.py -n 0 -v -s
```

### Linting and Formatting

```bash
make lint           # Ruff check + mypy
make format         # Ruff format
```

---

## Common Commands

```bash
# Start / stop
make docker-up              # Start all services
make docker-down            # Stop all services (keeps volumes)

# Logs
make docker-logs            # Tail API + frontend logs
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs -f worker_light worker_medium

# Shell into containers
docker exec -it e2i_api_dev bash
docker exec -it e2i_frontend_dev sh

# Rebuild after dependency changes (requirements.txt, package.json)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d --build

# Full teardown (removes volumes — data loss!)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml down -v
```

---

## Optional Stacks

### Opik (LLM observability) — intentionally stopped in production

The Opik overlay exists but is **not run in production** (intentionally stopped
2026-05-29); production LLM observability is the `llm_usage_events` path (see
`docs/LLM_CONFIGURATION.md` §4). To run Opik locally anyway:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml -f docker/docker-compose.opik.yml up -d
```

Adds 10 services (MySQL, ClickHouse, ZooKeeper, MinIO, Opik backend/frontend). Access at http://localhost:5173.

### Debug Tools (Redis Commander, FalkorDB Browser)

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml --profile dev-tools --profile debug up -d
```

| Tool | Port | URL |
|------|------|-----|
| Redis Commander | 8081 | http://localhost:8081 |
| FalkorDB Browser | 3030 | http://localhost:3030 |

---

## Troubleshooting

### Port conflicts

If a port is already in use, stop the conflicting service or change the port mapping in `docker-compose.dev.yml`. Common conflicts: port 3000 (BentoML), port 5000 (MLflow / macOS AirPlay), port 3001 (Supabase Studio).

### First build is slow

Normal — the Dockerfile installs PyTorch, scikit-learn, and other ML dependencies. Subsequent builds use Docker layer caching.

### Redis / FalkorDB authentication errors

Ensure `REDIS_PASSWORD` and `FALKORDB_PASSWORD` are set in your `.env`. All connection URLs include authentication — empty passwords will cause startup failures.

### Container can't reach Redis / FalkorDB

Make sure you're using **both** compose files:

```bash
# Correct (both files)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# Wrong (missing dev overrides)
docker compose up
```

### Hot reload not working

Source code is bind-mounted into containers by the dev overlay. If reload stops working, restart the affected service:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart api
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart frontend
```

### API returns 502 Bad Gateway

The API container is not running or still starting:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs api
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart api
```

---

## File Reference

| File | Purpose |
|------|---------|
| `docker/docker-compose.yml` | Base service definitions (shared across environments) |
| `docker/docker-compose.dev.yml` | Dev overlay: volume mounts, hot reload, debug settings |
| `docker/docker-compose.opik.yml` | Opik LLM observability overlay (10 services) |
| `docker/Dockerfile` | Multi-stage build for API + workers |
| `docker/frontend/Dockerfile` | Multi-stage build for React app |
| `docker/Dockerfile.feast` | Feast feature server |
| `docker/nginx/nginx.conf` | Docker nginx reverse proxy |

---

*Last Updated: 2026-07-18*
