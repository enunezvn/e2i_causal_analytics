# E2I Causal Analytics - Deployment Guide

How to run the full stack locally using Docker Compose with the dev overlay,
plus how the real production deploy works (see [Production Deploy (CI/CD)](#production-deploy-cicd)).

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
| `OPENAI_API_KEY` | OpenAI API key — the **default** LLM provider (`gpt-5.6-terra`/`gpt-5.6-luna` tiers) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anonymous key |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |
| `REDIS_PASSWORD` | Redis authentication password |
| `FALKORDB_PASSWORD` | FalkorDB authentication password |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password |
| `SUPABASE_DB_URL` | Supabase PostgreSQL connection string |

### Optional LLM configuration

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic key — only needed with `LLM_PROVIDER=anthropic` |
| `LLM_PROVIDER` | `openai` (default) or `anthropic` |
| `LLM_MODEL` | Pin the OpenAI standard/reasoning model without a code change |

See `docs/LLM_CONFIGURATION.md` for tiers, model mappings, and overrides.

### Auto-configured (set by compose, no action needed)

These are defined in `docker-compose.yml` via the `x-common-env` anchor:

| Variable | Docker Value | Purpose |
|----------|-------------|---------|
| `REDIS_URL` | `redis://:${REDIS_PASSWORD}@redis:6379/0` | Authenticated container networking |
| `FALKORDB_URL` | `redis://:${FALKORDB_PASSWORD}@falkordb:6379/0` | Authenticated container networking |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | Docker DNS resolution |
| `CELERY_BROKER_URL` | `redis://:${REDIS_PASSWORD}@redis:6379/1` | Task queue |
| `CELERY_RESULT_BACKEND` | `redis://:${REDIS_PASSWORD}@redis:6379/2` | Results store |

---

## Service Map

| Service | Port | URL | Notes |
|---------|------|-----|-------|
| API (FastAPI) | 8000 | http://localhost:8000 | Auto-reloads |
| API Docs | 8000 | http://localhost:8000/docs | Swagger UI |
| Frontend (Vite) | 3002 | http://localhost:3002 | HMR |
| MLflow | 5000 | http://localhost:5000 | 127.0.0.1 only |
| Redis | 6382 | redis://localhost:6382 | |
| FalkorDB | 6381 | redis://localhost:6381 | |
| BentoML | 3000 | http://localhost:3000 | 127.0.0.1 only |
| Feast | 6567 | http://localhost:6567 | 127.0.0.1 only |
| Grafana | 3200 | http://localhost:3200 | 127.0.0.1 only |
| Prometheus | 9091 | http://localhost:9091 | 127.0.0.1 only |
| Loki | 3101 | http://localhost:3101 | 127.0.0.1 only |
| Alertmanager | 9093 | http://localhost:9093 | 127.0.0.1 only |
| Flower* | 5555 | http://localhost:5555 | debug profile |
| FalkorDB Browser* | 3030 | http://localhost:3030 | debug profile |
| Redis Commander* | 8081 | http://localhost:8081 | dev-tools profile |

\* Requires `--profile`. Management ports (127.0.0.1 only) need SSH tunnels for remote access — see `scripts/ssh-tunnels/`.

---

## Production Deploy (CI/CD)

The production deploy is `.github/workflows/deploy.yml` — the workflow file is
the source of truth; this is the operator-level summary (verified 2026-07-18).

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
`scripts/deploy.sh` (that script is a separate, manual deploy path which does
chain into the seeder). Reseed the graph manually when needed:

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
