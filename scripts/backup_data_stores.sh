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

# --- Redis Backup ---
echo ""
echo "--- Redis RDB Snapshot ---"
if [ -n "${REDIS_PASSWORD:-}" ]; then
  # Trigger BGSAVE
  if docker exec e2i_redis redis-cli -a "$REDIS_PASSWORD" BGSAVE 2>/dev/null; then
    echo "BGSAVE triggered, waiting for completion..."
    sleep 5
    # Copy RDB from volume
    if docker cp e2i_redis:/data/dump.rdb "$BACKUP_DIR/redis_dump.rdb" 2>/dev/null; then
      echo -e "${GREEN}Redis backup saved: $BACKUP_DIR/redis_dump.rdb${NC}"
    else
      echo -e "${YELLOW}WARNING: Could not copy Redis RDB file${NC}"
    fi
  else
    echo -e "${YELLOW}WARNING: Redis BGSAVE failed (container may not be running)${NC}"
  fi
else
  echo -e "${YELLOW}WARNING: REDIS_PASSWORD not set, skipping Redis backup${NC}"
fi

# --- FalkorDB Backup ---
echo ""
echo "--- FalkorDB RDB Snapshot ---"
if [ -n "${FALKORDB_PASSWORD:-}" ]; then
  # Trigger BGSAVE
  if docker exec e2i_falkordb redis-cli -a "$FALKORDB_PASSWORD" BGSAVE 2>/dev/null; then
    echo "BGSAVE triggered, waiting for completion..."
    sleep 5
    # Copy RDB from volume
    if docker cp e2i_falkordb:/data/dump.rdb "$BACKUP_DIR/falkordb_dump.rdb" 2>/dev/null; then
      echo -e "${GREEN}FalkorDB backup saved: $BACKUP_DIR/falkordb_dump.rdb${NC}"
    else
      echo -e "${YELLOW}WARNING: Could not copy FalkorDB RDB file${NC}"
    fi
  else
    echo -e "${YELLOW}WARNING: FalkorDB BGSAVE failed (container may not be running)${NC}"
  fi
else
  echo -e "${YELLOW}WARNING: FALKORDB_PASSWORD not set, skipping FalkorDB backup${NC}"
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
