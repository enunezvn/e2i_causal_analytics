# E2I Causal Analytics - Docker Setup

## Directory Structure

```
docker/
├── docker-compose.yml          # Production compose (no bind mounts)
├── docker-compose.dev.yml      # Development overrides (bind mounts + dev tools)
├── Dockerfile                  # Multi-stage: API + Worker
├── Dockerfile.feast            # Feast feature store
├── .env.example                # Environment template
├── Makefile                    # Convenience commands
│
└── frontend/
    ├── Dockerfile              # Multi-stage: React dev + prod
    └── nginx.conf              # Production nginx config
```

## Quick Start

### 1. Setup Environment

```bash
cd docker
cp .env.example .env
# Edit .env with your Supabase and Anthropic credentials
```

### 2. Development Mode

```bash
# Start with hot-reloading (bind mounts enabled)
make dev

# Or in detached mode
make dev-d

# With dev tools (Flower, Redis Commander)
make dev-tools
```

### 3. Production Mode

```bash
# Build and start production containers
make prod

# View logs
make prod-logs
```

## Volume Strategy

### Bind Mounts (Development Only)
Hot-reload code changes without rebuilding:

| Local Path | Container Path | Purpose |
|------------|----------------|---------|
| `./src` | `/app/src` | Python source |
| `./config` | `/app/config` | YAML configs |
| `./frontend/src` | `/app/src` | React source |

### Shared Volumes (Data Exchange)
Containers passing data to each other:

| Volume | Producers | Consumers | Data |
|--------|-----------|-----------|------|
| `ml_artifacts` | worker | mlflow | Trained models, metrics |
| `model_registry` | mlflow | bentoml (ro) | Registered models |
| `causal_outputs` | api, worker | api | DAGs, effect estimates |
| `feature_cache` | feast | api (ro), worker (ro) | Materialized features |

### Persistent Volumes (State)
Data that survives container restarts:

| Volume | Container | Data |
|--------|-----------|------|
| `redis_data` | redis | Working memory, LangGraph checkpoints |
| `falkordb_data` | falkordb | Semantic memory graph |
| `mlflow_db` | mlflow | Experiment metadata |
| `opik_data` | opik | LLM observability traces |

## Service Ports

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| Frontend | 3001 | http://localhost:3001 |
| MLflow | 5000 | http://localhost:5000 |
| BentoML | 3000 | http://localhost:3000 |
| Feast | 6566 | http://localhost:6566 |
| Opik | 5173 | http://localhost:5173 |
| Redis | 6379 | redis://localhost:6379 |
| FalkorDB | 6380 | redis://localhost:6380 |
| Flower* | 5555 | http://localhost:5555 |
| Redis Commander* | 8081 | http://localhost:8081 |

*Dev tools only (use `make dev-tools`)

## Common Commands

```bash
# View all commands
make help

# Check health of all services
make health

# View logs
make logs          # All containers
make logs-api      # API only
make logs-worker   # Worker only

# Shell access
make shell-api     # Bash in API container
make shell-worker  # Bash in Worker container

# Testing
make test          # Run all tests
make test-unit     # Unit tests only
make test-cov      # With coverage report

# Cleanup
make down          # Stop containers
make down-v        # Stop + remove volumes
make clean         # Full cleanup
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
│                      http://localhost:3001                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API (FastAPI + 18 Agents)                  │
│                      http://localhost:8000                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Orchestrator │  │Causal Impact │  │ Experiment Designer  │  │
│  │   (Tier 1)   │  │   (Tier 2)   │  │      (Tier 3)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         │ Celery Queue       │ Shared Volumes     │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────────────────────────────────┐
│     WORKER      │  │              SHARED VOLUMES                 │
│  (Long Tasks)   │  │  ml_artifacts │ causal_outputs │ features  │
└─────────────────┘  └─────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MLOPS SERVICES                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐│
│  │ MLflow  │  │ BentoML │  │  Feast  │  │        Opik         ││
│  │ :5000   │  │  :3000  │  │  :6566  │  │       :5173         ││
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INFRASTRUCTURE                             │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐│
│  │        Redis         │  │           FalkorDB               ││
│  │   Working Memory     │  │       Semantic Memory            ││
│  │   :6379              │  │       :6380                      ││
│  └──────────────────────┘  └──────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SUPABASE CLOUD                               │
│               PostgreSQL + pgvector (28 tables)                 │
│              Episodic + Procedural Memory                       │
└─────────────────────────────────────────────────────────────────┘
```

## Debugging

### VS Code Remote Debugging

1. Start dev environment: `make dev`
2. Attach VS Code debugger to port 5678
3. Set breakpoints in your code

### Celery Task Debugging

```bash
# Start with dev tools to access Flower
make dev-tools

# Open Flower UI
make flower-ui
```

### Redis Inspection

```bash
# Connect to Redis CLI
docker exec -it e2i_redis_dev redis-cli

# Or use Redis Commander
make dev-tools
# Open http://localhost:8081
```

## Troubleshooting

### "Permission denied" on bind mounts
```bash
# Fix ownership
sudo chown -R $USER:$USER ./src ./config
```

### Container won't start
```bash
# Check logs
docker logs e2i_api_dev

# Rebuild from scratch
make clean
make dev
```

### Database connection issues
```bash
# Verify Supabase credentials in .env
# Check if services are healthy
make health
```
