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

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| API (FastAPI) | 8000 | http://localhost:8000 | Backend + agents (auto-reloads) |
| API Docs | 8000 | http://localhost:8000/docs | Swagger UI |
| Frontend (Vite) | 3002 | http://localhost:3002 | React app (HMR) |
| MLflow | 5000 | http://localhost:5000 | Experiment tracking |
| Redis | 6382 | redis://localhost:6382 | Cache + task queue |
| FalkorDB | 6381 | redis://localhost:6381 | Graph database |
| Grafana | 3200 | http://localhost:3200 | Dashboards |
| Prometheus | 9091 | http://localhost:9091 | Metrics |
| Redis Commander* | 8081 | http://localhost:8081 | Redis web UI |

\* Dev-tools profile only — start with `--profile dev-tools`

## Environment Variables

### Required (must set in `.env`)

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for agents |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anonymous key |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |
| `REDIS_PASSWORD` | Redis authentication password |
| `FALKORDB_PASSWORD` | FalkorDB authentication password |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password |
| `SUPABASE_DB_URL` | Supabase PostgreSQL connection string |

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
```

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
| `frontend/nginx.conf` | Frontend nginx (production build serving) |
