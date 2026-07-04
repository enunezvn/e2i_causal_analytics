#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Redis & FalkorDB Backup
# =============================================================================
# Creates RDB snapshots from Redis and FalkorDB docker volumes.
# Keeps 7 daily snapshots.
#
# Backup Strategy:
#   - Primary DR: DigitalOcean droplet image backups (full-system recovery)
#   - This script: Supplementary point-in-time snapshots for Redis/FalkorDB
#   - Use DO image backups for full disaster recovery
#   - Use this script for granular data store rollback
#
# Usage:
#   ./scripts/backup_data_stores.sh
#
# Environment:
#   REDIS_PASSWORD     - Redis auth password
#   FALKORDB_PASSWORD  - FalkorDB auth password
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_BASE="$PROJECT_ROOT/backups/data_stores"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_BASE/$TIMESTAMP"
RETENTION_DAYS=7

# Source .env if available
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

mkdir -p "$BACKUP_DIR"

echo -e "${GREEN}=== Data Store Backup - $TIMESTAMP ===${NC}"

# Container names differ between the prod compose (docker/docker-compose.yml:
# e2i_redis, e2i_falkordb) and the dev compose actually running on this
# droplet (docker/docker-compose.dev.yml: e2i_redis_dev, e2i_falkordb_dev).
# Resolve at runtime so the script works against either.
resolve_container() {
  local name
  for name in "$@"; do
    if docker ps --format '{{.Names}}' | grep -qx "$name"; then
      echo "$name"
      return 0
    fi
  done
  return 1
}

# Snapshot one redis-protocol store: BGSAVE, then copy the RDB out of the
# container from wherever the server says it lives (redis uses /data,
# falkordb /var/lib/falkordb/data — ask CONFIG GET dir instead of guessing).
snapshot_rdb() {
  local container="$1" password="$2" dest="$3" rdb_dir
  if docker exec "$container" redis-cli -a "$password" --no-auth-warning BGSAVE; then
    echo "BGSAVE triggered, waiting for completion..."
    sleep 5
    rdb_dir=$(docker exec "$container" redis-cli -a "$password" --no-auth-warning CONFIG GET dir | tail -1)
    if docker cp "$container:$rdb_dir/dump.rdb" "$dest"; then
      echo -e "${GREEN}Backup saved: $dest${NC}"
    else
      echo -e "${YELLOW}WARNING: Could not copy RDB file from $container:$rdb_dir${NC}"
    fi
  else
    echo -e "${YELLOW}WARNING: BGSAVE failed on $container${NC}"
  fi
}

# --- Redis Backup ---
echo ""
echo "--- Redis RDB Snapshot ---"
REDIS_CONTAINER="$(resolve_container e2i_redis e2i_redis_dev || true)"
if [ -z "$REDIS_CONTAINER" ]; then
  echo -e "${YELLOW}WARNING: no Redis container running (e2i_redis / e2i_redis_dev), skipping${NC}"
elif [ -n "${REDIS_PASSWORD:-}" ]; then
  snapshot_rdb "$REDIS_CONTAINER" "$REDIS_PASSWORD" "$BACKUP_DIR/redis_dump.rdb"
else
  echo -e "${YELLOW}WARNING: REDIS_PASSWORD not set, skipping Redis backup${NC}"
fi

# --- FalkorDB Backup ---
echo ""
echo "--- FalkorDB RDB Snapshot ---"
FALKORDB_CONTAINER="$(resolve_container e2i_falkordb e2i_falkordb_dev || true)"
if [ -z "$FALKORDB_CONTAINER" ]; then
  echo -e "${YELLOW}WARNING: no FalkorDB container running (e2i_falkordb / e2i_falkordb_dev), skipping${NC}"
elif [ -n "${FALKORDB_PASSWORD:-}" ]; then
  snapshot_rdb "$FALKORDB_CONTAINER" "$FALKORDB_PASSWORD" "$BACKUP_DIR/falkordb_dump.rdb"
else
  echo -e "${YELLOW}WARNING: FALKORDB_PASSWORD not set, skipping FalkorDB backup${NC}"
fi

# --- Layer-4 Audit Artifacts Backup (named-volume tarball) ---
# Plan: .claude/plans/layer4_evaluator_audit_consumer.md.
# The audit_artifacts named volume holds adaptive-validity sidecar JSONs
# produced by src/agents/ml_foundation/data_preparer/graph.py:
# write_adaptive_verdicts_sidecar. Treat as supplementary forensic data
# (the canonical persistence is the JSON files themselves; this tar is a
# rollback escape hatch).
echo ""
echo "--- Layer-4 Audit Artifacts Tarball ---"
if docker volume inspect e2i_audit_artifacts >/dev/null 2>&1; then
  if docker run --rm \
      -v e2i_audit_artifacts:/data:ro \
      -v "$BACKUP_DIR":/backup \
      alpine:3.20 \
      tar czf /backup/audit_artifacts.tar.gz -C /data . 2>/dev/null; then
    echo -e "${GREEN}Audit artifacts saved: $BACKUP_DIR/audit_artifacts.tar.gz${NC}"
  else
    echo -e "${YELLOW}WARNING: audit_artifacts tar failed${NC}"
  fi
else
  echo -e "${YELLOW}NOTICE: e2i_audit_artifacts volume not present, skipping${NC}"
fi

# --- Retention ---
echo ""
echo "--- Cleanup (retention: ${RETENTION_DAYS} days) ---"
if [ -d "$BACKUP_BASE" ]; then
  DELETED=$(find "$BACKUP_BASE" -maxdepth 1 -type d -not -name "data_stores" -mtime +$RETENTION_DAYS -print -exec rm -rf {} \; 2>/dev/null | wc -l)
  echo "Deleted $DELETED old backup(s)"
fi

echo ""
echo -e "${GREEN}Data store backup complete: $BACKUP_DIR${NC}"
