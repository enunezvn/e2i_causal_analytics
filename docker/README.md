# E2I Causal Analytics - Docker Development Setup

How to run the full stack locally using the **root-level** compose files.

> **Note**: Production runs on systemd (DigitalOcean droplet). These Docker files are for local development only.

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
cp .env.example .env.dev
# Edit .env.dev — fill in required keys (see Environment Variables below)

# 3. Start everything
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# 4. Verify
curl -s http://localhost:8000/health | python3 -m json.tool
```

First build pulls PyTorch + ML dependencies — subsequent starts use cached layers.

## Services

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| API (FastAPI) | 8000 | http://localhost:8000 | Backend + 21 agents |
| API Docs | 8000 | http://localhost:8000/docs | Swagger UI |
| Frontend | 3000 | http://localhost:3000 | React app |
| MLflow | 5000 | http://localhost:5000 | Experiment tracking |
| Redis | 6379 | redis://localhost:6379 | Cache + task queue |
| FalkorDB | 6381 | redis://localhost:6381 | Graph database |
| Grafana | 3100 | http://localhost:3100 | Dashboards |
| Prometheus | 9091 | http://localhost:9091 | Metrics |
| Redis Commander* | 8081 | http://localhost:8081 | Redis web UI |

\* Debug profile only — start with `--profile debug`

## Environment Variables

### Required (must set in `.env.dev`)

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for agents |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anonymous key |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |

### Auto-configured (set by compose, no action needed)

These are overridden in `docker-compose.dev.yml` to use Docker-internal DNS names:

| Variable | Docker Value | Why |
|----------|-------------|-----|
| `REDIS_URL` | `redis://redis:6379` | Container-to-container networking |
| `FALKORDB_HOST` | `falkordb` | Docker DNS resolution |
| `FALKORDB_URL` | `redis://falkordb:6379` | Docker DNS resolution |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | Docker DNS resolution |
| `CELERY_BROKER_URL` | `redis://redis:6379/1` | Task queue broker |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/2` | Task results store |

> The `.env.example` file has `localhost` values which work for host-based development. The compose file overrides these with Docker DNS names so containers can talk to each other.

## Common Commands

```bash
# Start in foreground (see logs)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Start detached
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

# View logs
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f fastapi
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f agent-worker

# Rebuild a single service
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build fastapi

# Shell into API container
docker exec -it e2i-fastapi bash

# Stop everything
docker compose -f docker-compose.yml -f docker-compose.dev.yml down

# Stop and remove volumes (reset data)
docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v

# Start with debug tools (Redis Commander)
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile debug up
```

## Troubleshooting

### Port conflicts

If a port is already in use, stop the conflicting service or change the port mapping in `docker-compose.dev.yml`.

Common conflicts: port 3000 (React/BentoML), port 5000 (MLflow/macOS AirPlay).

### First build is slow

Normal — the Dockerfile installs PyTorch, scikit-learn, and other ML dependencies. Subsequent builds use Docker layer caching.

### Supabase connection errors

Verify your `.env.dev` has correct Supabase credentials. The database is hosted externally (not in Docker), so your machine needs internet access.

### Container can't reach Redis/FalkorDB

The dev compose sets Docker-internal URLs automatically. If you see connection errors, make sure you're using both compose files:

```bash
# Correct (both files)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Wrong (missing dev overrides)
docker compose up
```

### Hot reload not working

Source code is bind-mounted as read-only. The uvicorn `--reload` flag watches for changes. If reload stops working, restart the fastapi service.

## Architecture: Docker Dev vs Production

| Aspect | Docker Dev (this setup) | Production (droplet) |
|--------|------------------------|---------------------|
| Orchestration | Docker Compose | systemd |
| API server | Container (uvicorn --reload) | systemd service |
| Redis | Container (port 6379) | Docker container (port 6382) |
| FalkorDB | Container (port 6381) | Docker container (port 6381) |
| MLflow | Container (SQLite backend) | Docker container (PostgreSQL backend) |
| Frontend | Container (dev server) | nginx serving static build |
| Config | `.env.dev` + compose overrides | `.env` + systemd env files |

## File Reference

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Base service definitions (shared across environments) |
| `docker-compose.dev.yml` | Dev overrides: ports, hot reload, debug settings, Docker DNS URLs |
| `docker/fastapi/Dockerfile` | Multi-stage build for API + worker |
| `docker/frontend/Dockerfile` | Multi-stage build for React app |
| `docker/mlflow/Dockerfile` | MLflow tracking server |
| `.env.example` | Template for environment variables |
