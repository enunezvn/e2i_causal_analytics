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

set -e

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

  if curl -sf --max-time "$timeout" "$url" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ $name - HEALTHY${NC}"
    ((HEALTHY++))
    return 0
  else
    echo -e "${RED}❌ $name - UNHEALTHY${NC}"
    ((UNHEALTHY++))
    return 1
  fi
}

echo "--- HTTP Services ---"
check_http "http://localhost:8000/health" "API (FastAPI)"
check_http "http://localhost:5000/health" "MLflow"
check_http "http://localhost:3000/healthz" "BentoML" 10
check_http "http://localhost:6566/health" "Feast"
check_http "http://localhost:5173/health" "Opik (UI)" 5 || \
  check_http "http://localhost:5174/health" "Opik (API)" 5
check_http "http://localhost:3002/health" "Frontend"

# =============================================================================
# REDIS SERVICES
# =============================================================================

echo ""
echo "--- Memory Systems ---"

# Redis
if docker exec e2i_redis redis-cli ping 2>/dev/null | grep -q PONG; then
  echo -e "${GREEN}✅ Redis (Working Memory) - HEALTHY${NC}"
  ((HEALTHY++))
else
  echo -e "${RED}❌ Redis (Working Memory) - UNHEALTHY${NC}"
  ((UNHEALTHY++))
fi

# FalkorDB
if docker exec e2i_falkordb redis-cli ping 2>/dev/null | grep -q PONG; then
  echo -e "${GREEN}✅ FalkorDB (Semantic Memory) - HEALTHY${NC}"
  ((HEALTHY++))
else
  echo -e "${RED}❌ FalkorDB (Semantic Memory) - UNHEALTHY${NC}"
  ((UNHEALTHY++))
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
      ((HEALTHY++))
    else
      echo -e "${RED}❌ $worker_name - UNHEALTHY (container running but not responding)${NC}"
      ((UNHEALTHY++))
    fi
  else
    echo -e "${YELLOW}⚠️  $worker_name - NOT RUNNING (scaled to 0)${NC}"
    ((SKIPPED++))
  fi
}

check_worker "Worker Light" "worker_light"
check_worker "Worker Medium" "worker_medium"
check_worker "Worker Heavy" "worker_heavy"

# Scheduler
if docker ps --filter "name=e2i_scheduler" --filter "status=running" -q | grep -q .; then
  echo -e "${GREEN}✅ Celery Beat (Scheduler) - RUNNING${NC}"
  ((HEALTHY++))
else
  echo -e "${RED}❌ Celery Beat (Scheduler) - NOT RUNNING${NC}"
  ((UNHEALTHY++))
fi

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
