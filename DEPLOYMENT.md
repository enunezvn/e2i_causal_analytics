# E2I Causal Analytics - Deployment Guide

How to run the full stack locally using Docker Compose with the dev overlay.

> **Admin?** See [`DEPLOYMENT_ADMIN.md`](DEPLOYMENT_ADMIN.md) for droplet-specific operations (SSH access, nginx, remote deploy, tunnels).

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
| `ANTHROPIC_API_KEY` | Claude API key for LLM-powered agents |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anonymous key |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |
| `REDIS_PASSWORD` | Redis authentication password |
| `FALKORDB_PASSWORD` | FalkorDB authentication password |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password |
| `SUPABASE_DB_URL` | Supabase PostgreSQL connection string |

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

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| API (FastAPI) | 8000 | http://localhost:8000 | Backend + agents (auto-reloads) |
| API Docs | 8000 | http://localhost:8000/docs | Swagger UI |
| Frontend (Vite) | 3002 | http://localhost:3002 | React app (HMR) |
| MLflow | 5000 | http://localhost:5000 | Experiment tracking |
| Redis | 6382 | `redis://localhost:6382` | Cache + task queue |
| FalkorDB | 6381 | `redis://localhost:6381` | Graph database |
| Grafana | 3200 | http://localhost:3200 | Dashboards |
| Prometheus | 9091 | http://localhost:9091 | Metrics |
| BentoML | 3000 | http://localhost:3000 | Model serving |
| Feast | 6567 | http://localhost:6567 | Feature store |

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

### Opik (LLM observability)

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
| `docker/frontend/nginx.conf` | Frontend nginx (production build serving) |

---

*Last Updated: 2026-02-08*
