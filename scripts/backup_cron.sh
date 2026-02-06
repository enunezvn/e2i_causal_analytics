#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Automated Backup (Cron Wrapper)
# =============================================================================
# Wraps the Supabase export script with retention and logging.
# Also invokes data store backups (Redis + FalkorDB).
#
# Crontab entry (daily at 2 AM):
#   0 2 * * * /home/enunez/Projects/e2i_causal_analytics/scripts/backup_cron.sh >> /var/log/e2i-backup.log 2>&1
#
# Environment (set in crontab or sourced from .env):
#   SUPABASE_DB_URL       - PostgreSQL connection string (for DB backup)
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
echo ""
echo "--- Supabase Database Backup ---"
EXPORT_SCRIPT="$SCRIPT_DIR/supabase/export_cloud_db.sh"
if [ -x "$EXPORT_SCRIPT" ]; then
  BACKUP_DIR="$BACKUP_BASE/supabase_export_$TIMESTAMP"
  if "$EXPORT_SCRIPT" "$BACKUP_DIR"; then
    echo "Database backup succeeded: $BACKUP_DIR"
  else
    echo "ERROR: Database backup failed (exit code $?)"
  fi
else
  echo "WARNING: Export script not found or not executable: $EXPORT_SCRIPT"
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
