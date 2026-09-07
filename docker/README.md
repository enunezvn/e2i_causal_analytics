# E2I Causal Analytics - Docker Development Setup

How to run the full stack using the compose files with the dev overlay.

> **Note**: Both local development and the production droplet use the same Docker Compose setup (base + dev overlay). The dev overlay provides volume mounts for hot-reloading and debug settings.

## Prerequisites

- Docker Engine 24+
- Docker Compose v2+
- Git

## Quick Start

```bash
# 1. Clone
git clone git@github.com:enunezvn/e2i_causal_analytics.git
cd e2i_causal_analytics

# 2. Create env file from template
cp .env.example .env
# Edit .env — fill in required keys (see Environment Variables below)

# 3. Start everything
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# 4. Verify
curl -s http://localhost:8000/health | python3 -m json.tool
```

First build pulls PyTorch + ML dependencies — subsequent starts use cached layers.

## Services

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
| Grafana* | 3200 | http://localhost:3200 | monitoring profile; 127.0.0.1 only |
| Prometheus* | 9091 | http://localhost:9091 | monitoring profile; 127.0.0.1 only |
| Loki* | 3101 | http://localhost:3101 | monitoring profile; 127.0.0.1 only |
| Alertmanager* | 9093 | http://localhost:9093 | monitoring profile; 127.0.0.1 only |
| Flower* | 5555 | http://localhost:5555 | debug profile |
| FalkorDB Browser* | 3030 | http://localhost:3030 | debug profile |
| Redis Commander* | 8081 | http://localhost:8081 | dev-tools profile |

\* Requires `--profile`. Management ports (127.0.0.1 only) need SSH tunnels for remote access — see `scripts/ssh-tunnels/`.

## Environment Variables

### Required (must set in `.env`)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key — the **default** LLM provider (`gpt-5.6-terra` standard/reasoning, `gpt-5.6-luna` fast), and RAG embeddings (`text-embedding-3-small`) are OpenAI *regardless* of `LLM_PROVIDER`, so this key is always required |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anonymous key |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |
| `SUPABASE_POSTGRES_PASSWORD` | Password of the self-hosted `supabase-db` container — mirrors `POSTGRES_PASSWORD` in `/opt/supabase/docker/.env`. Compose derives the container-internal `SUPABASE_DB_URL` from it and **refuses to start without it** |
| `REDIS_PASSWORD` | Redis authentication password |
| `FALKORDB_PASSWORD` | FalkorDB authentication password |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password — still required with the `monitoring` profile **off** (compose interpolates every service before it filters by profile) |
| `SUPABASE_DB_URL` | Host-side PostgreSQL connection string — **not** forwarded into containers (see below) |

`REDIS_PASSWORD`, `FALKORDB_PASSWORD`, `SUPABASE_POSTGRES_PASSWORD` and
`GRAFANA_ADMIN_PASSWORD` are `:?`-enforced: compose exits before starting anything
if they are unset. Validate an `.env` without starting anything:

```bash
docker compose --env-file .env -f docker/docker-compose.yml config -q   # rc 0 = ok
```

### Optional LLM configuration

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `openai` (code default, `src/utils/llm_factory.py`) or `anthropic` |
| `ANTHROPIC_API_KEY` | Required only with `LLM_PROVIDER=anthropic`. It also gates two paths that are Anthropic-only whatever the provider, both fail-open: the nightly routing-label judge (`src/tasks/routing_label_tasks.py`) and the Layer-4 adaptive-validity evaluator (`src/data/causal_role_evaluator.py`) |
| `LLM_MODEL` | Pin the OpenAI standard/reasoning model without a code change |
| `DSPY_LM_MODEL` | Verbatim litellm model string for the DSPy/GEPA lane; takes precedence over `LLM_PROVIDER` there |

See `docs/LLM_CONFIGURATION.md` for tiers, model mappings and overrides.

### Auto-configured (set by compose, no action needed)

These are set in `docker-compose.yml` via the `x-common-env` anchor:

| Variable | Docker Value | Why |
|----------|-------------|-----|
| `REDIS_URL` | `redis://:${REDIS_PASSWORD}@redis:6379/0` | Authenticated container networking |
| `FALKORDB_URL` | `redis://:${FALKORDB_PASSWORD}@falkordb:6379/0` | Authenticated container networking |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | Docker DNS resolution |
| `CELERY_BROKER_URL` | `redis://:${REDIS_PASSWORD}@redis:6379/1` | Authenticated task queue |
| `CELERY_RESULT_BACKEND` | `redis://:${REDIS_PASSWORD}@redis:6379/2` | Authenticated results store |

## Common Commands

```bash
# Start all services
make docker-up

# Stop all services
make docker-down

# View logs
make docker-logs

# Deploy (git pull + restart workers)
make deploy

# Deploy with rebuild
make deploy-build

# Shell into API container
docker exec -it e2i_api_dev bash

# Start with debug tools (Redis Commander, FalkorDB Browser)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml --profile dev-tools --profile debug up -d

# Start the observability stack (Prometheus, Grafana, Loki, Alertmanager, Promtail, exporters)
# Off by default: no deploy step starts these, and they add ~1-1.5GB of RSS.
COMPOSE_PROFILES=monitoring docker compose -f docker/docker-compose.yml up -d
```

> **Observability is opt-in.** The monitoring services are gated behind the
> `monitoring` profile, so a plain `up -d` does not start them and no deployment
> can. `scripts/health_check.sh` derives its skip-set from
> `docker compose config --services`, so enabling the profile re-arms those probes
> automatically — nothing to toggle in two places.

## Troubleshooting

### Port conflicts

If a port is already in use, stop the conflicting service or change the port mapping in `docker-compose.dev.yml`.

Common conflicts: port 3000 (BentoML), port 5000 (MLflow/macOS AirPlay), port 3001 (Supabase Studio).

### First build is slow

Normal — the Dockerfile installs PyTorch, scikit-learn, and other ML dependencies. Subsequent builds use Docker layer caching.

### Redis/FalkorDB authentication errors

Ensure `REDIS_PASSWORD` and `FALKORDB_PASSWORD` are set in your `.env`. All connection URLs include authentication.

### Container can't reach Redis/FalkorDB

Make sure you're using both compose files:

```bash
# Correct (both files)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up

# Wrong (missing dev overrides)
docker compose up
```

### Hot reload not working

Source code is bind-mounted into containers. The uvicorn `--reload` flag watches for changes. If reload stops, restart the service:
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart api
```

## File Reference

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Base service definitions (shared across environments) |
| `docker-compose.dev.yml` | Dev overlay: volume mounts, hot reload, debug settings |
| `Dockerfile` | Multi-stage build for API + workers |
| `frontend/Dockerfile` | Multi-stage build for React app |
| `Dockerfile.feast` | Feast feature server |
| `nginx/nginx.conf` | Docker nginx (for full-Docker deployments) |
