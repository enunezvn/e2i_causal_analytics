#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Health Check Script
# =============================================================================
# Tests all service health endpoints and reports status
#
# Usage:
#   ./scripts/health_check.sh                    # Run once
#   watch -n 2 './scripts/health_check.sh'       # Continuous monitoring
#
# Author: E2I Causal Analytics Team
# Version: 1.0.0
# =============================================================================

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
HEALTHY=0
UNHEALTHY=0
SKIPPED=0

echo "=========================================="
echo "E2I Causal Analytics - Health Check"
echo "=========================================="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# =============================================================================
# HTTP SERVICES
# =============================================================================

check_http() {
  local url=$1
  local name=$2
  local timeout=${3:-5}

  if curl -sfk --max-time "$timeout" "$url" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ $name - HEALTHY${NC}"
    ((HEALTHY++)) || true
    return 0
  else
    echo -e "${RED}❌ $name - UNHEALTHY${NC}"
    ((UNHEALTHY++)) || true
    return 1
  fi
}

echo "--- HTTP Services ---"
check_http "http://localhost:8000/health" "API (FastAPI)" || true
check_http "http://localhost:5000/health" "MLflow" || true

check_http "http://localhost:3000/healthz" "BentoML" 10 || true
check_http "http://localhost:6566/health" "Feast" || true
check_http "http://localhost:5173/" "Opik (UI)" 5 || true
check_http "http://localhost:8084/health-check" "Opik (Backend)" 10 || true

check_http "http://localhost:3002" "Frontend (Vite Dev)" || true

# =============================================================================
# SUPABASE SERVICES
# =============================================================================

echo ""
echo "--- Supabase ---"

# Supabase Kong requires apikey header; read from env or supabase .env
SUPABASE_ANON_KEY="${SUPABASE_ANON_KEY:-$(grep ANON_KEY /opt/supabase/docker/.env 2>/dev/null | head -1 | cut -d= -f2)}"

check_supabase() {
  local url=$1
  local name=$2
  local timeout=${3:-5}
  if curl -sfk --max-time "$timeout" -H "apikey: ${SUPABASE_ANON_KEY}" "$url" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ $name - HEALTHY${NC}"
    ((HEALTHY++)) || true
    return 0
  else
    echo -e "${RED}❌ $name - UNHEALTHY${NC}"
    ((UNHEALTHY++)) || true
    return 1
  fi
}

check_supabase "http://localhost:54321/rest/v1/" "Supabase REST (PostgREST)" || true
check_supabase "http://localhost:54321/auth/v1/health" "Supabase Auth (GoTrue)" || true
check_http "http://localhost:54321/storage/v1/status" "Supabase Storage" || true
check_http "http://localhost:3001" "Supabase Studio" || true

# =============================================================================
# MEMORY SYSTEMS
# =============================================================================

echo ""
echo "--- Memory Systems ---"

# Redis
REDIS_CONTAINER=$(docker ps --filter "name=redis" --filter "label=e2i.service=infrastructure" --filter "label=e2i.memory=working" --format "{{.Names}}" | head -1)
REDIS_CONTAINER="${REDIS_CONTAINER:-e2i_redis}"
if docker exec "$REDIS_CONTAINER" redis-cli -a "${REDIS_PASSWORD:-changeme}" ping 2>/dev/null | grep -q PONG; then
  echo -e "${GREEN}✅ Redis (Working Memory) - HEALTHY${NC}"
  ((HEALTHY++)) || true
else
  echo -e "${RED}❌ Redis (Working Memory) - UNHEALTHY${NC}"
  ((UNHEALTHY++)) || true
fi

# FalkorDB
FALKOR_CONTAINER=$(docker ps --filter "name=falkordb" --filter "label=e2i.service=infrastructure" --filter "label=e2i.memory=semantic" --format "{{.Names}}" | head -1)
FALKOR_CONTAINER="${FALKOR_CONTAINER:-e2i_falkordb}"
if docker exec "$FALKOR_CONTAINER" redis-cli -a "${FALKORDB_PASSWORD:-changeme}" ping 2>/dev/null | grep -q PONG; then
  echo -e "${GREEN}✅ FalkorDB (Semantic Memory) - HEALTHY${NC}"
  ((HEALTHY++)) || true
else
  echo -e "${RED}❌ FalkorDB (Semantic Memory) - UNHEALTHY${NC}"
  ((UNHEALTHY++)) || true
fi

# =============================================================================
# CELERY WORKERS
# =============================================================================

echo ""
echo "--- Celery Workers ---"

check_worker() {
  local worker_name=$1
  local container_pattern=$2

  # Check if any containers matching the pattern are running
  local running_count=$(docker ps --filter "name=${container_pattern}" --filter "status=running" -q | wc -l)

  if [ "$running_count" -gt 0 ]; then
    # Try to ping the first container
    local container=$(docker ps --filter "name=${container_pattern}" --filter "status=running" --format "{{.Names}}" | head -1)

    if docker exec "$container" celery -A src.workers.celery_app inspect ping 2>/dev/null | grep -q "pong"; then
      echo -e "${GREEN}✅ $worker_name - HEALTHY ($running_count replicas)${NC}"
      ((HEALTHY++)) || true
    else
      echo -e "${RED}❌ $worker_name - UNHEALTHY (container running but not responding)${NC}"
      ((UNHEALTHY++)) || true
    fi
  else
    echo -e "${YELLOW}⚠️  $worker_name - NOT RUNNING (scaled to 0)${NC}"
    ((SKIPPED++)) || true
  fi
}

check_worker "Worker Light" "worker_light"
check_worker "Worker Medium" "worker_medium"
check_worker "Worker Heavy" "worker_heavy"

# Scheduler
if docker ps --filter "name=scheduler" --filter "status=running" -q | grep -q .; then
  echo -e "${GREEN}✅ Celery Beat (Scheduler) - RUNNING${NC}"
  ((HEALTHY++)) || true
else
  echo -e "${RED}❌ Celery Beat (Scheduler) - NOT RUNNING${NC}"
  ((UNHEALTHY++)) || true
fi

# =============================================================================
# OBSERVABILITY
# =============================================================================

echo ""
echo "--- Observability ---"
check_http "http://localhost:9091/-/healthy" "Prometheus" || true
check_http "http://localhost:3200/api/health" "Grafana" || true
check_http "http://localhost:3101/ready" "Loki" || true

check_container() {
  local name=$1
  local container=$2
  if docker ps --filter "name=${container}" --filter "status=running" -q | grep -q .; then
    echo -e "${GREEN}✅ $name - RUNNING${NC}"
    ((HEALTHY++)) || true
  else
    echo -e "${RED}❌ $name - NOT RUNNING${NC}"
    ((UNHEALTHY++)) || true
  fi
}

check_container "Alertmanager" "e2i_alertmanager"
check_container "Promtail" "e2i_promtail"
check_container "Node Exporter" "e2i_node_exporter"
check_container "Postgres Exporter" "e2i_postgres_exporter"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
TOTAL=$((HEALTHY + UNHEALTHY + SKIPPED))
echo "Total Services: $TOTAL"
echo -e "${GREEN}Healthy: $HEALTHY${NC}"
echo -e "${RED}Unhealthy: $UNHEALTHY${NC}"
echo -e "${YELLOW}Skipped (scaled to 0): $SKIPPED${NC}"
echo ""

# Exit with error if any services are unhealthy
if [ "$UNHEALTHY" -gt 0 ]; then
  echo -e "${RED}❌ SYSTEM STATUS: DEGRADED${NC}"
  exit 1
else
  echo -e "${GREEN}✅ SYSTEM STATUS: HEALTHY${NC}"
  exit 0
fi
