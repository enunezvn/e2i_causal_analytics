#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Automated Backup (Cron Wrapper)
# =============================================================================
# Dumps the local Supabase Postgres (supabase-db container) with retention
# and logging. Also invokes data store backups (Redis + FalkorDB).
#
# Crontab entry (daily at 2 AM). Log under $HOME — /var/log is NOT writable
# by the cron user on this host (a root-owned dir kills the redirect with
# Permission denied BEFORE the script runs, silently no-oping the job):
#   0 2 * * * /home/enunez/Projects/e2i_causal_analytics/scripts/backup_cron.sh >> /home/enunez/logs/e2i-backup.log 2>&1
#
# Environment (set in crontab or sourced from .env):
#   REDIS_PASSWORD        - Redis auth password
#   FALKORDB_PASSWORD     - FalkorDB auth password
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_BASE="$PROJECT_ROOT/backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "E2I Backup - $TIMESTAMP"
echo "============================================"

# Source .env if available
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

mkdir -p "$BACKUP_BASE"

# --- Supabase Database Backup ---
# Prod Postgres is the LOCAL supabase-db container (this droplet is prod);
# dump in-container so pg_dump always matches the server version. The old
# path via scripts/supabase/export_cloud_db.sh is a one-time cloud->self-hosted
# migration tool that requires the Supabase CLI (not installed here) and
# never produced a backup on this host.
echo ""
echo "--- Supabase Database Backup ---"
DB_CONTAINER="supabase-db"
if docker ps --format '{{.Names}}' | grep -qx "$DB_CONTAINER"; then
  BACKUP_DIR="$BACKUP_BASE/supabase_export_$TIMESTAMP"
  mkdir -p "$BACKUP_DIR"
  if docker exec "$DB_CONTAINER" pg_dump -U postgres -d postgres --format=custom \
       > "$BACKUP_DIR/full_backup.dump" \
     && docker exec "$DB_CONTAINER" pg_dumpall -U postgres --globals-only \
       > "$BACKUP_DIR/roles.sql"; then
    echo "Database backup succeeded: $BACKUP_DIR ($(du -sh "$BACKUP_DIR" | cut -f1))"
  else
    echo "ERROR: Database backup failed — removing partial $BACKUP_DIR"
    rm -rf "$BACKUP_DIR"
  fi
else
  echo "ERROR: $DB_CONTAINER container not running — skipping database backup"
fi

# --- Data Store Backups (Redis + FalkorDB) ---
echo ""
echo "--- Data Store Backups ---"
DATA_BACKUP_SCRIPT="$SCRIPT_DIR/backup_data_stores.sh"
if [ -x "$DATA_BACKUP_SCRIPT" ]; then
  if "$DATA_BACKUP_SCRIPT"; then
    echo "Data store backup succeeded"
  else
    echo "ERROR: Data store backup failed (exit code $?)"
  fi
else
  echo "WARNING: Data store backup script not found: $DATA_BACKUP_SCRIPT"
fi

# --- Retention: delete backups older than $RETENTION_DAYS days ---
echo ""
echo "--- Cleanup (retention: ${RETENTION_DAYS} days) ---"
DELETED=$(find "$BACKUP_BASE" -maxdepth 1 -type d -name "supabase_export_*" -mtime +$RETENTION_DAYS -print -exec rm -rf {} \; 2>/dev/null | wc -l)
echo "Deleted $DELETED old database backup(s)"

DELETED_RDB=$(find "$BACKUP_BASE/data_stores" -maxdepth 1 -type d -mtime +7 -print -exec rm -rf {} \; 2>/dev/null | wc -l)
echo "Deleted $DELETED_RDB old data store backup(s)"

echo ""
echo "Backup completed at $(date)"
