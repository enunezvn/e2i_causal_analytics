#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Droplet Health Report
# =============================================================================
# Comprehensive health report: resource usage, response times, Prometheus
# targets, disk, and container stats. Complementary to health_check.sh
# (which is lightweight and CI-friendly).
#
# Usage:
#   ./scripts/droplet_report.sh              # Full report
#   ./scripts/droplet_report.sh 2>&1 | tee /tmp/report.txt   # Save to file
#
# Author: E2I Causal Analytics Team
# Version: 1.0.0
# =============================================================================

# NOTE: No set -e — arithmetic (( 0 )) returns exit code 1, which would kill
# the script. All (( )) expressions have || true to prevent premature exit.

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Counters
HEALTHY=0
UNHEALTHY=0
SKIPPED=0

# Width for section headers
WIDTH=72

# =============================================================================
# UTILITIES
# =============================================================================

section() {
  echo ""
  echo -e "${BOLD}$(printf '=%.0s' $(seq 1 $WIDTH))${NC}"
  echo -e "${BOLD}  $1${NC}"
  echo -e "${BOLD}$(printf '=%.0s' $(seq 1 $WIDTH))${NC}"
}

subsection() {
  echo ""
  echo -e "${CYAN}--- $1 ---${NC}"
}

# Check an HTTP endpoint and print status code + response time
check_endpoint() {
  local url=$1
  local name=$2
  local timeout=${3:-5}

  local result
  result=$(curl -sk --max-time "$timeout" -o /dev/null -w "%{http_code} %{time_total}" "$url" 2>/dev/null)
  local exit_code=$?

  if [ $exit_code -ne 0 ]; then
    printf "  %-40s ${RED}%-8s${NC}  %s\n" "$name" "DOWN" "--"
    ((UNHEALTHY++)) || true
    return 1
  fi

  local code
  code=$(echo "$result" | awk '{print $1}')
  local time_s
  time_s=$(echo "$result" | awk '{print $2}')
  local time_ms
  time_ms=$(awk "BEGIN {printf \"%.0f\", $time_s * 1000}")

  if [[ "$code" =~ ^2 ]]; then
    printf "  %-40s ${GREEN}%-8s${NC}  %sms\n" "$name" "$code" "$time_ms"
    ((HEALTHY++)) || true
    return 0
  elif [[ "$code" =~ ^3 ]]; then
    printf "  %-40s ${YELLOW}%-8s${NC}  %sms\n" "$name" "$code" "$time_ms"
    ((HEALTHY++)) || true
    return 0
  else
    printf "  %-40s ${RED}%-8s${NC}  %sms\n" "$name" "$code" "$time_ms"
    ((UNHEALTHY++)) || true
    return 1
  fi
}

# Check an HTTP endpoint with an apikey header (Supabase)
check_endpoint_apikey() {
  local url=$1
  local name=$2
  local apikey=$3
  local timeout=${4:-5}

  local result
  result=$(curl -sk --max-time "$timeout" -H "apikey: ${apikey}" -o /dev/null -w "%{http_code} %{time_total}" "$url" 2>/dev/null)
  local exit_code=$?

  if [ $exit_code -ne 0 ]; then
    printf "  %-40s ${RED}%-8s${NC}  %s\n" "$name" "DOWN" "--"
    ((UNHEALTHY++)) || true
    return 1
  fi

  local code
  code=$(echo "$result" | awk '{print $1}')
  local time_s
  time_s=$(echo "$result" | awk '{print $2}')
  local time_ms
  time_ms=$(awk "BEGIN {printf \"%.0f\", $time_s * 1000}")

  if [[ "$code" =~ ^2 ]]; then
    printf "  %-40s ${GREEN}%-8s${NC}  %sms\n" "$name" "$code" "$time_ms"
    ((HEALTHY++)) || true
    return 0
  else
    printf "  %-40s ${RED}%-8s${NC}  %sms\n" "$name" "$code" "$time_ms"
    ((UNHEALTHY++)) || true
    return 1
  fi
}

# =============================================================================
# 1. HEADER
# =============================================================================

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║              E2I Causal Analytics — Droplet Health Report          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  Timestamp : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Hostname  : $(hostname)"
echo "  Uptime    : $(uptime -p 2>/dev/null || uptime)"

# =============================================================================
# 2. SYSTEM RESOURCES
# =============================================================================

section "System Resources"

CPU_COUNT=$(nproc 2>/dev/null || echo "?")
LOAD=$(cat /proc/loadavg 2>/dev/null | awk '{print $1, $2, $3}')

echo "  CPUs       : $CPU_COUNT"
echo "  Load (1/5/15) : $LOAD"
echo ""

# RAM
if command -v free &>/dev/null; then
  echo "  Memory:"
  free -h | awk '
    NR==1 { printf "    %-12s %10s %10s %10s\n", "", $1, $2, $3 }
    NR==2 { printf "    %-12s %10s %10s %10s\n", $1, $2, $3, $4 }
    NR==3 { printf "    %-12s %10s %10s %10s\n", $1, $2, $3, "" }
  '
fi
echo ""

# Swap
SWAP_TOTAL=$(free -m 2>/dev/null | awk '/Swap/ {print $2}')
SWAP_USED=$(free -m 2>/dev/null | awk '/Swap/ {print $3}')
if [ "${SWAP_TOTAL:-0}" -gt 0 ] 2>/dev/null; then
  echo "  Swap       : ${SWAP_USED}M / ${SWAP_TOTAL}M"
else
  echo "  Swap       : none"
fi

# =============================================================================
# 3. SERVICE HEALTH + RESPONSE TIMES
# =============================================================================

section "Service Health + Response Times"
printf "  %-40s %-8s  %s\n" "SERVICE" "STATUS" "LATENCY"
printf "  %-40s %-8s  %s\n" "-------" "------" "-------"

subsection "E2I Application"
check_endpoint "http://localhost:8000/health" "API (FastAPI :8000)" || true
check_endpoint "http://localhost:3002" "Frontend (Vite :3002)" || true

subsection "MLOps"
check_endpoint "http://localhost:5000/health" "MLflow (:5000)" || true
check_endpoint "http://localhost:3000/healthz" "BentoML (:3000 /healthz)" 10 || true
check_endpoint "http://localhost:6567/health" "Feast (:6567)" || true

subsection "Opik"
check_endpoint "http://localhost:5173/" "Opik UI (:5173)" || true
check_endpoint "http://localhost:8084/health-check" "Opik Backend (:8084)" 10 || true
check_endpoint "http://localhost:8001/healthcheck" "Opik Python Backend (:8001)" || true
check_endpoint "http://localhost:9090" "Opik MinIO Console (:9090)" || true

subsection "Memory Systems"

# Redis PING
REDIS_CONTAINER=$(docker ps --filter "name=redis" --filter "label=e2i.service=infrastructure" --filter "label=e2i.memory=working" --format "{{.Names}}" 2>/dev/null | head -1)
REDIS_CONTAINER="${REDIS_CONTAINER:-e2i_redis}"
REDIS_START=$(date +%s%N)
if docker exec "$REDIS_CONTAINER" redis-cli --no-auth-warning -a "${REDIS_PASSWORD:-changeme}" ping 2>/dev/null | grep -q PONG; then
  REDIS_MS=$(( ($(date +%s%N) - REDIS_START) / 1000000 ))
  printf "  %-40s ${GREEN}%-8s${NC}  %sms\n" "Redis PING" "PONG" "$REDIS_MS"
  ((HEALTHY++)) || true
else
  printf "  %-40s ${RED}%-8s${NC}  %s\n" "Redis PING" "FAIL" "--"
  ((UNHEALTHY++)) || true
fi

# FalkorDB PING
FALKOR_CONTAINER=$(docker ps --filter "name=falkordb" --filter "label=e2i.service=infrastructure" --filter "label=e2i.memory=semantic" --format "{{.Names}}" 2>/dev/null | head -1)
FALKOR_CONTAINER="${FALKOR_CONTAINER:-e2i_falkordb}"
FALKOR_START=$(date +%s%N)
if docker exec "$FALKOR_CONTAINER" redis-cli --no-auth-warning -a "${FALKORDB_PASSWORD:-changeme}" ping 2>/dev/null | grep -q PONG; then
  FALKOR_MS=$(( ($(date +%s%N) - FALKOR_START) / 1000000 ))
  printf "  %-40s ${GREEN}%-8s${NC}  %sms\n" "FalkorDB PING" "PONG" "$FALKOR_MS"
  ((HEALTHY++)) || true
else
  printf "  %-40s ${RED}%-8s${NC}  %s\n" "FalkorDB PING" "FAIL" "--"
  ((UNHEALTHY++)) || true
fi

check_endpoint "http://localhost:3030" "FalkorDB Browser (:3030)" || true

subsection "Supabase"
SUPABASE_ANON_KEY="${SUPABASE_ANON_KEY:-$(grep ANON_KEY /opt/supabase/docker/.env 2>/dev/null | head -1 | cut -d= -f2)}"
check_endpoint_apikey "http://localhost:54321/rest/v1/" "Supabase REST (PostgREST)" "$SUPABASE_ANON_KEY" || true
check_endpoint_apikey "http://localhost:54321/auth/v1/health" "Supabase Auth (GoTrue)" "$SUPABASE_ANON_KEY" || true
check_endpoint "http://localhost:54321/storage/v1/status" "Supabase Storage" || true
check_endpoint "http://localhost:3001" "Supabase Studio (:3001)" || true

subsection "Observability"
check_endpoint "http://localhost:9091/-/healthy" "Prometheus (:9091)" || true
check_endpoint "http://localhost:3200/api/health" "Grafana (:3200)" || true
check_endpoint "http://localhost:3101/ready" "Loki (:3101)" || true
check_endpoint "http://localhost:9093/-/healthy" "Alertmanager (:9093)" || true

# =============================================================================
# 4. CELERY WORKERS
# =============================================================================

section "Celery Workers"

check_celery_worker() {
  local worker_name=$1
  local container_pattern=$2

  local running_count
  running_count=$(docker ps --filter "name=${container_pattern}" --filter "status=running" -q 2>/dev/null | wc -l)

  if [ "$running_count" -gt 0 ]; then
    local container
    container=$(docker ps --filter "name=${container_pattern}" --filter "status=running" --format "{{.Names}}" | head -1)

    if docker exec "$container" celery -A src.workers.celery_app inspect ping 2>/dev/null | grep -q "pong"; then
      echo -e "  ${GREEN}$worker_name — HEALTHY ($running_count replicas)${NC}"
      ((HEALTHY++)) || true
    else
      echo -e "  ${RED}$worker_name — NOT RESPONDING ($running_count replicas running)${NC}"
      ((UNHEALTHY++)) || true
    fi
  else
    echo -e "  ${YELLOW}$worker_name — NOT RUNNING (scaled to 0)${NC}"
    ((SKIPPED++)) || true
  fi
}

check_celery_worker "Worker Light" "worker_light"
check_celery_worker "Worker Medium" "worker_medium"
check_celery_worker "Worker Heavy" "worker_heavy"

# Scheduler
if docker ps --filter "name=scheduler" --filter "status=running" -q 2>/dev/null | grep -q .; then
  echo -e "  ${GREEN}Celery Beat (Scheduler) — RUNNING${NC}"
  ((HEALTHY++)) || true
else
  echo -e "  ${RED}Celery Beat (Scheduler) — NOT RUNNING${NC}"
  ((UNHEALTHY++)) || true
fi

# =============================================================================
# 5. CONTAINER RESOURCE USAGE
# =============================================================================

section "Container Resource Usage (sorted by memory)"

# Header
printf "  ${BOLD}%-35s %7s %22s %6s %15s %5s${NC}\n" \
  "CONTAINER" "CPU%" "MEM USAGE/LIMIT" "MEM%" "NET I/O" "PIDs"
printf "  %-35s %7s %22s %6s %15s %5s\n" \
  "---------" "----" "---------------" "----" "-------" "----"

# docker stats --no-stream, sorted by memory usage descending
docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.PIDs}}" 2>/dev/null | \
  sort -t$'\t' -k3 -h -r | \
  while IFS=$'\t' read -r name cpu mem_usage mem_pct net_io pids; do
    printf "  %-35s %7s %22s %6s %15s %5s\n" \
      "$name" "$cpu" "$mem_usage" "$mem_pct" "$net_io" "$pids"
  done

# =============================================================================
# 6. PROMETHEUS SCRAPE TARGETS
# =============================================================================

section "Prometheus Scrape Targets"

PROM_TARGETS_FILE=$(mktemp /tmp/prom_targets_XXXXXX.json 2>/dev/null || echo "/tmp/prom_targets_$$.json")
if curl -sf --max-time 5 "http://localhost:9091/api/v1/targets" -o "$PROM_TARGETS_FILE" 2>/dev/null; then
  printf "  ${BOLD}%-25s %-8s  %s${NC}\n" "JOB" "HEALTH" "SCRAPE URL"
  printf "  %-25s %-8s  %s\n" "---" "------" "----------"

  python3 -c "
import json, sys
with open('$PROM_TARGETS_FILE') as f:
    data = json.load(f)
targets = data.get('data', {}).get('activeTargets', [])
targets.sort(key=lambda t: t.get('labels', {}).get('job', ''))
for t in targets:
    job = t.get('labels', {}).get('job', '?')
    health = t.get('health', '?')
    url = t.get('scrapeUrl', '?')
    color = '\033[0;32m' if health == 'up' else '\033[0;31m'
    nc = '\033[0m'
    print(f'  {job:<25s} {color}{health:<8s}{nc}  {url}')
" 2>/dev/null || echo "  (failed to parse targets JSON)"

  rm -f "$PROM_TARGETS_FILE"
else
  echo "  Prometheus not reachable — skipping targets"
  rm -f "$PROM_TARGETS_FILE"
fi

# =============================================================================
# 7. DISK USAGE
# =============================================================================

section "Disk Usage"

subsection "Filesystem"
df -h / 2>/dev/null | awk '
  NR==1 { printf "  %-20s %8s %8s %8s %6s %s\n", $1, $2, $3, $4, $5, $6 }
  NR==2 { printf "  %-20s %8s %8s %8s %6s %s\n", $1, $2, $3, $4, $5, $6 }
'

subsection "Docker Disk"
docker system df 2>/dev/null | while IFS= read -r line; do
  echo "  $line"
done

# =============================================================================
# 8. CONTAINER STATUS
# =============================================================================

section "Container Status (all)"

docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}" 2>/dev/null | \
  (read -r header; echo "  $header"; sort -t$'\t' -k1) | \
  while IFS= read -r line; do
    if echo "$line" | grep -q "(unhealthy)"; then
      echo -e "  ${RED}${line}${NC}"
    elif echo "$line" | grep -q "(healthy)"; then
      echo -e "  ${GREEN}${line}${NC}"
    elif echo "$line" | grep -q "Exited"; then
      echo -e "  ${RED}${line}${NC}"
    else
      echo "  $line"
    fi
  done

# =============================================================================
# 9. SUMMARY
# =============================================================================

section "Summary"

TOTAL_CHECKED=$((HEALTHY + UNHEALTHY + SKIPPED))

# Container counts
TOTAL_CONTAINERS=$(docker ps -q 2>/dev/null | wc -l)
HEALTHY_CONTAINERS=$(docker ps --filter "health=healthy" -q 2>/dev/null | wc -l)
UNHEALTHY_CONTAINERS=$(docker ps --filter "health=unhealthy" -q 2>/dev/null | wc -l)
NO_HC_CONTAINERS=$((TOTAL_CONTAINERS - HEALTHY_CONTAINERS - UNHEALTHY_CONTAINERS))

echo "  Service checks:"
echo -e "    ${GREEN}Healthy    : $HEALTHY${NC}"
echo -e "    ${RED}Unhealthy  : $UNHEALTHY${NC}"
echo -e "    ${YELLOW}Skipped    : $SKIPPED${NC}"
echo "    Total      : $TOTAL_CHECKED"
echo ""
echo "  Containers:"
echo "    Running           : $TOTAL_CONTAINERS"
echo -e "    Healthy (hc)      : ${GREEN}$HEALTHY_CONTAINERS${NC}"
echo -e "    Unhealthy (hc)    : ${RED}$UNHEALTHY_CONTAINERS${NC}"
echo "    No healthcheck    : $NO_HC_CONTAINERS"
echo ""

if [ "$UNHEALTHY" -gt 0 ] || [ "$UNHEALTHY_CONTAINERS" -gt 0 ]; then
  echo -e "  ${RED}SYSTEM STATUS: DEGRADED${NC}"
  echo ""
  exit 1
else
  echo -e "  ${GREEN}SYSTEM STATUS: ALL CLEAR${NC}"
  echo ""
  exit 0
fi
