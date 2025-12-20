# Docker Setup Assessment & Action Plan
**E2I Causal Analytics**
**Date:** 2025-12-17
**Reviewer:** Claude Code

---

## Executive Summary

**Overall Score: 8.5/10**

The Docker setup demonstrates a production-grade, well-architected multi-service deployment with excellent separation of concerns. The architecture supports an 18-agent, 6-tier system with a tri-memory architecture (Redis, FalkorDB, Supabase). The volume strategy is particularly well-designed with clear separation between development bind mounts, inter-service data exchange volumes, and persistent state volumes.

**Key Strengths:**
- ✅ Multi-stage Dockerfile with dev/prod targets
- ✅ Intelligent volume strategy (bind mounts, shared volumes, persistent volumes)
- ✅ Two-network architecture for service isolation
- ✅ Profile-based dev tools (Flower, Redis Commander)
- ✅ Comprehensive Makefile with excellent DX
- ✅ Hot-reloading support for development
- ✅ VS Code remote debugging integration
- ✅ Clear environment separation

**Critical Gaps:**
- ❌ Missing Dockerfile.feast
- ❌ Missing frontend/Dockerfile
- ❌ Missing uv.lock dependency file
- ⚠️ Worker container runs as root (security risk)
- ⚠️ Some health check endpoints may not exist

---

## Detailed Assessment

### Architecture Review

#### 1. Multi-Stage Dockerfile ✅ EXCELLENT
**Location:** `docker/Dockerfile`

**Stages:**
1. **base** - Python 3.11 + system dependencies + uv
2. **dependencies** - Virtual environment + production dependencies
3. **development** - Dev tools + debugpy + all dependencies
4. **production** - Slim image with runtime-only dependencies + non-root user

**Strengths:**
- Clear stage separation minimizes production image size
- Uses `uv` for fast package management
- Includes all necessary scientific computing libraries (OpenBLAS, LAPACK, GraphViz)
- Non-root user (e2i:e2i) in production for security
- Health check configured
- Gunicorn with uvicorn workers for production

**Issues:**
- References `uv.lock` (lines 59, 72) which may not exist yet

#### 2. Volume Strategy ✅ EXCELLENT
**Classification:**

| Type | Purpose | Lifecycle | Examples |
|------|---------|-----------|----------|
| **Bind Mounts** | Hot-reload (dev only) | Temporary | `../src:/app/src` |
| **Shared Volumes** | Inter-container data | Per-deployment | `ml_artifacts`, `causal_outputs` |
| **Persistent Volumes** | Service state | Survives restarts | `redis_data`, `falkordb_data` |

**Strengths:**
- Clear data flow between services
- Read-only mounts where appropriate (`feature_cache:ro`)
- tmpfs for worker scratch space
- Proper volume exclusions to prevent package overwrites

**Issues:**
- Bind mount permissions may cause issues on Linux/WSL
- No resource limits on tmpfs (could grow unbounded in error scenarios)

#### 3. Network Architecture ✅ GOOD
**Networks:**
- `e2i_network` - Main application network
- `mlops_network` - Isolated MLOps services

**Strengths:**
- Service isolation between app and MLOps layers
- MLOps services can communicate without exposing to main network

**Issues:**
- Frontend on `e2i_network` when it only needs API access
- Could further isolate with dedicated frontend network

#### 4. Service Health Checks ⚠️ NEEDS VERIFICATION
**Configured Health Checks:**
- ✅ redis: `redis-cli ping`
- ✅ falkordb: `redis-cli ping`
- ✅ mlflow: `curl /health`
- ✅ bentoml: `curl /healthz`
- ❓ feast: `curl /health` (verify endpoint exists)
- ❓ opik: `curl /health` (verify endpoint exists)
- ❓ frontend: `curl /health` (Vite/React doesn't have this by default)

#### 5. Development Experience ✅ EXCELLENT
**Features:**
- Hot-reloading via bind mounts
- VS Code remote debugging (port 5678)
- Separate dev compose file with overrides
- Dev-only tools via profiles (Flower, Redis Commander)
- Relaxed healthcheck timeouts for development
- Debug logging enabled

**Strengths:**
- Clear separation between dev and prod configurations
- Minimal rebuild requirements during development
- Comprehensive Makefile shortcuts

---

## Action Plan

### Priority 1: Critical Issues (Must Fix Before First Run)

#### [ ] 1.1 Create Missing Dockerfile.feast
**Location:** `docker/Dockerfile.feast`
**Status:** MISSING
**Impact:** Production compose will fail to start

**Options:**
A. Create custom Dockerfile
B. Use official Feast image (recommended)

**Recommendation:** Use official image - less maintenance overhead

---

#### [ ] 1.2 Create Missing frontend/Dockerfile
**Location:** `docker/frontend/Dockerfile`
**Status:** MISSING
**Impact:** Frontend service won't build

**Requirements:**
- Multi-stage: development (Vite) + production (nginx)
- Development: Node 20 + Vite dev server
- Production: Static build served via nginx
- Health check endpoint (add to Vite config or nginx)

---

#### [ ] 1.3 Create frontend/nginx.conf
**Location:** `docker/frontend/nginx.conf`
**Status:** MISSING (referenced in README)
**Impact:** Production frontend won't serve correctly

**Requirements:**
- Serve static React build
- Reverse proxy API calls
- SPA routing (fallback to index.html)
- CORS configuration
- Gzip compression

---

#### [ ] 1.4 Handle uv.lock Dependency
**Location:** `docker/Dockerfile` lines 59, 72
**Status:** File may not exist
**Impact:** Docker build will fail

**Options:**
A. Generate `uv.lock`: Run `uv lock` at project root
B. Switch to requirements.txt: Modify Dockerfile to use `pip install -r requirements.txt`

**Recommendation:** Option A if using uv, Option B for simplicity

---

### Priority 2: Security Issues (Fix Before Production Deploy)

#### [ ] 2.1 Remove C_FORCE_ROOT from Worker Container
**Location:** `docker-compose.yml:396`
**Current:** Worker runs as root
**Risk:** Security vulnerability
**Impact:** Container compromise could affect host

**Fix:**
```yaml
worker:
  # In production stage of Dockerfile
  user: e2i:e2i
  environment:
    # Remove C_FORCE_ROOT=1
```

**Testing Required:** Verify Celery works with non-root user

---

#### [ ] 2.2 Add Resource Limits to All Services
**Location:** `docker-compose.yml`
**Current:** No resource constraints
**Risk:** Resource exhaustion, OOM kills

**Fix:** Add deploy limits to each service
```yaml
services:
  worker:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

**Recommended Limits:**
- API: 2 CPUs, 4GB RAM
- Worker: 2 CPUs, 4GB RAM
- MLflow: 1 CPU, 2GB RAM
- Redis: 1 CPU, 512MB RAM
- FalkorDB: 1 CPU, 1GB RAM

---

### Priority 3: Reliability Issues (Fix Before Production Deploy)

#### [ ] 3.1 Fix Service Dependency Race Conditions
**Location:** `docker-compose.yml:415`
**Current:** `worker depends_on api: service_started`
**Issue:** Worker may start before API is ready

**Fix:**
```yaml
worker:
  depends_on:
    redis:
      condition: service_healthy
    falkordb:
      condition: service_healthy
    api:
      condition: service_healthy  # Change from service_started
```

---

#### [ ] 3.2 Verify Health Check Endpoints
**Services to Verify:**
- feast: `http://localhost:6566/health`
- opik: `http://localhost:5173/health`
- frontend: `http://localhost:80/health` or `http://localhost:5173/` (dev)

**Actions:**
1. Start each service individually
2. Test health endpoint with curl
3. Update health check command if endpoint differs
4. Add health endpoint to frontend if missing

---

#### [ ] 3.3 Add Startup Validation Script
**Location:** `scripts/docker_healthcheck.sh`
**Purpose:** Pre-flight checks before container startup

**Checks:**
- ✅ Required environment variables are set
- ✅ Supabase URL is reachable
- ✅ Anthropic API key is valid format
- ✅ Volume mounts are writable
- ✅ Redis connection works

**Integration:** Add to Dockerfile as entrypoint wrapper

---

### Priority 4: Configuration Improvements (Should Fix)

#### [ ] 4.1 Reduce Environment Variable Duplication
**Location:** `docker-compose.yml:302-327, 379-393`
**Issue:** API and Worker have duplicate env vars

**Fix:** Use YAML anchors
```yaml
x-common-env: &common-env
  SUPABASE_URL: ${SUPABASE_URL}
  SUPABASE_KEY: ${SUPABASE_KEY}
  SUPABASE_SERVICE_KEY: ${SUPABASE_SERVICE_KEY}
  REDIS_URL: redis://redis:6379/0
  FALKORDB_URL: redis://falkordb:6380/0
  MLFLOW_TRACKING_URI: http://mlflow:5000
  BENTOML_URL: http://bentoml:3000
  FEAST_URL: http://feast:6566
  OPIK_URL: http://opik:5174
  CELERY_BROKER_URL: redis://redis:6379/1
  CELERY_RESULT_BACKEND: redis://redis:6379/2
  ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}

services:
  api:
    environment:
      <<: *common-env
      ENVIRONMENT: production
      LOG_LEVEL: INFO
      WORKERS: 4

  worker:
    environment:
      <<: *common-env
      ENVIRONMENT: production
      LOG_LEVEL: INFO
```

**Benefits:**
- Single source of truth
- Easier maintenance
- Reduced risk of configuration drift

---

#### [ ] 4.2 Add Logging Configuration
**Location:** All services in docker-compose.yml
**Issue:** Unbounded log growth

**Fix:** Add to each service
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

**Custom for high-volume services:**
```yaml
api:
  logging:
    driver: "json-file"
    options:
      max-size: "50m"
      max-file: "5"
```

---

#### [ ] 4.3 Improve Network Isolation
**Location:** `docker-compose.yml:474`
**Issue:** Frontend on main network when it only needs API

**Current:**
```yaml
frontend:
  networks:
    - e2i_network
```

**Better:**
```yaml
networks:
  frontend_network:
    driver: bridge
    name: e2i_frontend_network

services:
  frontend:
    networks:
      - frontend_network

  api:
    networks:
      - e2i_network
      - frontend_network  # Only API bridges to frontend
```

---

### Priority 5: Nice to Have (Future Enhancements)

#### [ ] 5.1 Add .dockerignore File
**Location:** Project root and docker/
**Purpose:** Speed up builds, reduce context size

**Contents:**
```
# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build
.venv
venv

# IDE
.vscode
.idea
*.swp
*.swo

# Docker
docker-compose*.yml
Dockerfile*

# Documentation
*.md
docs/

# Tests
tests/
.pytest_cache
.coverage
htmlcov

# Data
data/
*.csv
*.parquet
```

---

#### [ ] 5.2 Add Docker BuildKit Features
**Location:** `docker/Dockerfile`
**Benefits:** Faster builds, better caching

**Changes:**
```dockerfile
# syntax=docker/dockerfile:1.4

# Use BuildKit cache mounts
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

---

#### [ ] 5.3 Create Docker Healthcheck Dashboard
**Location:** `scripts/health_dashboard.sh`
**Purpose:** Single command to check all services

**Features:**
- Color-coded status (green/red)
- Service uptime
- Container resource usage
- Recent logs summary

---

#### [ ] 5.4 Add Backup & Restore Scripts
**Location:** `scripts/backup_volumes.sh`, `scripts/restore_volumes.sh`
**Purpose:** Persistent volume backup

**Volumes to Backup:**
- redis_data (working memory snapshots)
- falkordb_data (semantic memory graph)
- mlflow_db (experiment metadata)
- opik_data (observability traces)

---

#### [ ] 5.5 Create Production Deployment Guide
**Location:** `docker/PRODUCTION_DEPLOYMENT.md`
**Contents:**
- Pre-deployment checklist
- Secrets management (not in .env)
- SSL/TLS setup with nginx
- Monitoring setup
- Backup procedures
- Rollback procedures
- Scaling guidelines

---

## Testing Checklist

### Pre-Deployment Testing

#### Development Environment
- [ ] `make dev` starts all services successfully
- [ ] API accessible at http://localhost:8000
- [ ] API health check returns 200 OK
- [ ] Frontend accessible at http://localhost:3001
- [ ] Hot-reload works (change src file, see update)
- [ ] VS Code debugger attaches successfully (port 5678)
- [ ] Redis accessible and storing data
- [ ] FalkorDB accessible and storing graphs
- [ ] MLflow UI accessible at http://localhost:5000
- [ ] BentoML accessible at http://localhost:3000
- [ ] Opik UI accessible at http://localhost:5173
- [ ] Flower UI accessible with `make dev-tools`
- [ ] Redis Commander accessible with `make dev-tools`
- [ ] Worker processes Celery tasks
- [ ] Logs viewable with `make logs`
- [ ] Shell access works with `make shell-api`
- [ ] Tests run with `make test`

#### Production Environment
- [ ] `make prod` starts all services successfully
- [ ] All health checks pass within start_period
- [ ] No containers running as root (except Redis/FalkorDB)
- [ ] Resource limits enforced
- [ ] Volumes persist data after restart
- [ ] API serves requests under load
- [ ] Worker processes long-running tasks
- [ ] MLOps services integrate correctly
- [ ] Logging works and rotates
- [ ] Backup script creates valid backups
- [ ] Restore script recovers from backup

#### Integration Testing
- [ ] API can write to Supabase
- [ ] API can read from Supabase
- [ ] Workers can access shared volumes
- [ ] MLflow tracks experiments
- [ ] BentoML serves models
- [ ] Feast serves features
- [ ] Opik traces LLM calls
- [ ] Redis stores LangGraph checkpoints
- [ ] FalkorDB stores semantic graphs
- [ ] Frontend communicates with API
- [ ] WebSocket connections work

---

## File Creation Summary

### Files to Create
1. `docker/Dockerfile.feast` - Feast feature store container
2. `docker/frontend/Dockerfile` - Multi-stage React/nginx container
3. `docker/frontend/nginx.conf` - Production nginx configuration
4. `scripts/docker_healthcheck.sh` - Pre-flight validation script
5. `.dockerignore` - Build context optimization
6. `docker/PRODUCTION_DEPLOYMENT.md` - Production deployment guide
7. `scripts/backup_volumes.sh` - Volume backup automation
8. `scripts/restore_volumes.sh` - Volume restore automation
9. `scripts/health_dashboard.sh` - Service health monitoring

### Files to Modify
1. `docker-compose.yml` - Add resource limits, logging, fix dependencies
2. `docker-compose.dev.yml` - Update health checks
3. `docker/Dockerfile` - Add BuildKit optimizations, fix uv.lock handling

---

## Estimated Effort

| Priority | Tasks | Estimated Time |
|----------|-------|----------------|
| P1 - Critical | 4 tasks | 4-6 hours |
| P2 - Security | 2 tasks | 2-3 hours |
| P3 - Reliability | 3 tasks | 3-4 hours |
| P4 - Configuration | 3 tasks | 2-3 hours |
| P5 - Nice to Have | 5 tasks | 4-6 hours |
| **Total** | **17 tasks** | **15-22 hours** |

---

## Next Steps

### Recommended Order
1. **Session 1:** P1.1, P1.2, P1.3 (Create missing Dockerfiles)
2. **Session 2:** P1.4 (Fix dependency management)
3. **Session 3:** P2.1, P3.1, P3.2 (Security & reliability)
4. **Session 4:** P4.1, P4.2 (Configuration improvements)
5. **Session 5:** P3.3, P5.1-P5.5 (Testing & documentation)

### Quick Wins (Start Here)
- [x] Create docker/Dockerfile.feast (15 min)
- [ ] Add .dockerignore (10 min)
- [ ] Fix uv.lock issue (20 min)
- [ ] Add YAML anchors for env vars (30 min)

---

## Conclusion

Your Docker setup is architecturally sound and demonstrates sophisticated understanding of containerization best practices. The main gaps are missing files and some security hardening needed for production deployment. Once the Priority 1 and Priority 2 items are addressed, this will be a robust, production-ready deployment system.

The volume strategy is particularly well-designed and the separation between development and production environments is exemplary. The comprehensive Makefile and documentation make this highly maintainable.

**Recommendation:** Address P1 and P2 items before any production deployment. P3-P5 items can be addressed iteratively based on operational needs.
