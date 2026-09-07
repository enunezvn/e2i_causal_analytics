# E2I Causal Analytics - Docker Development Setup

How to run the full stack using the compose files with the dev overlay.

> **Production runs the base file alone — no overlay.** `deploy.yml`'s
> `pick_overlay()` returns the empty string because `docker/frontend/Dockerfile:106`
> is `FROM nginx:alpine AS production`, so the droplet runs `e2i_api` (gunicorn
> `--workers 2` / UvicornWorker, `read_only: true`, GHCR image tagged by commit sha,
> `8000:8000`) and `e2i_frontend` (nginx serving the built bundle, `3002:80`).
> **This file describes local development**: base + `docker-compose.dev.yml`, which
> adds the bind mounts, `uvicorn --reload`, Vite HMR on `3002:5173`, and the `_dev`
> container names used throughout below. Production is deployed *only* by merging to
> `main`; see `DEPLOYMENT.md` § Production Deploy.

## Prerequisites

- Docker Engine 24+
- Docker Compose v2+
- Git
- **A running self-hosted Supabase stack.** `docker-compose.yml` joins
  `supabase-network` as an `external: true` network and several services resolve
  the `supabase-db` container over it, so on a machine where that stack has never
  run `up` fails on the missing network before anything starts.

## Quick Start

```bash
# 0. Bring up self-hosted Supabase first (this creates the external supabase-network)
./docker/supabase/start.sh

# 1. Clone
git clone git@github.com:enunezvn/e2i_causal_analytics.git
cd e2i_causal_analytics

# 2. Create env file from template
cp .env.example .env
# Edit .env — fill in required keys (see Environment Variables below).
# SUPABASE_POSTGRES_PASSWORD is easy to miss and compose refuses to start
# without it; validate before starting anything:
docker compose --env-file .env -f docker/docker-compose.yml config -q

# 3. Start everything
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# 4. Verify
curl -s http://localhost:8000/health | python3 -m json.tool
```

First build pulls PyTorch + ML dependencies — subsequent starts use cached layers.

## Services

| Service | Port | URL | Notes |
|---------|------|-----|-------|
| API (FastAPI) | 8000 | http://localhost:8000 | Auto-reloads (dev overlay) |
| API Docs | 8000 | http://localhost:8000/docs | Swagger UI |
| Frontend (Vite) | 3002 | http://localhost:3002 | HMR (dev overlay); nginx on the built bundle in production |
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

### How a `.env` value reaches the containers

**Compose forwards no `.env` file wholesale.** The `x-common-env` anchor in
`docker-compose.yml` is a *whitelist*: a host variable reaches `api`, `worker_*`
and `scheduler` only if that anchor names it. Anything else you put in `.env` is a
silent no-op inside the containers — the in-code default governs and nothing warns
you. Derive the current list rather than trusting a copy of it:

```bash
sed -n '/^x-common-env:/,/^x-common-worker:/p' docker/docker-compose.yml \
  | grep -o '\${[A-Z_0-9]*' | tr -d '${' | sort -u
```

A few entries are computed by compose and must **not** be set by hand — a host
value for the same name is ignored:

| Variable | Docker value |
|----------|--------------|
| `SUPABASE_DB_URL` | `postgresql://postgres:${SUPABASE_POSTGRES_PASSWORD}@supabase-db:5432/postgres` |
| `REDIS_URL` | `redis://:${REDIS_PASSWORD}@redis:6379/0` |
| `FALKORDB_URL` | `redis://:${FALKORDB_PASSWORD}@falkordb:6379/0` |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | `redis://:${REDIS_PASSWORD}@redis:6379/1` and `/2` |
| `MLFLOW_TRACKING_URI` / `BENTOML_SERVICE_URL` / `FEAST_URL` / `OPIK_URL` | in-network service URLs |
| `ENVIRONMENT` / `LOG_LEVEL` | hardcoded `production` / `INFO` |

The remaining forwarded entries are optional runtime knobs (chatbot warm, RAG
chain, routing labeler, DSPy/GEPA legs, synthetic visibility) plus the optional
biomedical API keys. `DEPLOYMENT.md` § *Runtime knobs forwarded by compose*
carries the full derived table with each one's compose default, purpose and the
issue its rationale lives in — and the list of variables that application code
reads but compose does **not** forward, where an `.env` edit is inert.

## Common Commands

```bash
# Start all services
make docker-up

# Stop all services
make docker-down

# View logs
make docker-logs

# Shell into API container (dev overlay name; on the droplet it is e2i_api)
docker exec -it e2i_api_dev bash

# Start with debug tools (Redis Commander, FalkorDB Browser)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml --profile dev-tools --profile debug up -d

# Start the observability stack (Prometheus, Grafana, Loki, Alertmanager, Promtail, exporters)
# Off by default: no deploy step starts these, and they add ~1-1.5GB of RSS.
COMPOSE_PROFILES=monitoring docker compose -f docker/docker-compose.yml up -d
```

> **`make deploy` / `make deploy-build` are NOT the production deploy.** They run
> `scripts/deploy.sh`, a legacy local-dev path that composes with the dev overlay,
> skips the feast/health/bentoml gates, and does `git reset --hard origin/main` plus
> a `git checkout <sha>` rollback inside the checkout. Never run either on the
> droplet — production deploys only by merging to `main` (`.github/workflows/deploy.yml`),
> and a manual redeploy is `gh workflow run deploy.yml`.

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
| `docker-compose.yml` | Base service definitions. **This alone is production** |
| `docker-compose.dev.yml` | Dev overlay: bind mounts, hot reload, `_dev` names, `dev-tools` profile |
| `docker-compose.frontend-dev.yml` | Legacy (#528-A rollback era); `pick_overlay()` can no longer select it |
| `docker-compose.monitoring.yml` | Superseded by the `monitoring` profile in the base file |
| `docker-compose.opik.yml` | Opik LLM observability overlay (10 services) |
| `docker-compose.rxnav.yml` | RxNav stub |
| `docker-compose.secure.yml` | Hardened variant, not used by the deploy |
| `Dockerfile` | Multi-stage build for API + workers |
| `frontend/Dockerfile` | Multi-stage build for React app; `AS production` at L106 is what makes `pick_overlay()` return empty |
| `Dockerfile.feast` | Feast feature server |
| `frontend/nginx.conf` | **Baked into the frontend image** at `/etc/nginx/nginx.conf` (`frontend/Dockerfile:112`) — the config actually serving the app |
| `nginx/host-nginx.conf` | Canonical source for the **host** nginx site (`/etc/nginx/sites-available/e2i-analytics`) |
| `nginx/nginx.secure.conf` | Mounted only by `docker-compose.secure.yml` |
| `nginx/nginx.conf` | **No consumer** — not baked into any image, not mounted by any compose file |

See `../DEPLOYMENT.md` for the persistent-volume inventory and the production
deploy pipeline.
