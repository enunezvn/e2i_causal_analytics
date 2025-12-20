---
allowed-tools: Read, Write, Edit, Bash
argument-hint: [operation] | --backup | --restore | --schedule | --validate | --cleanup
description: Manage Supabase database backups with automated scheduling and recovery procedures
---

# Supabase Backup Manager

Manage comprehensive Supabase database backups with automated scheduling and recovery validation: **$ARGUMENTS**

---

## Input Validation

Before proceeding, validate and parse the operation argument.

```bash
# Optional operation with specific flags
if [ -z "$ARGUMENTS" ]; then
  echo "ℹ️  No operation specified, will show backup status and options"
  OPERATION=""
elif [[ "$ARGUMENTS" == "--backup" ]]; then
  echo "✓ Operation: Create new backup"
  OPERATION="backup"
elif [[ "$ARGUMENTS" == "--restore" ]]; then
  echo "✓ Operation: Restore from backup"
  OPERATION="restore"
elif [[ "$ARGUMENTS" == "--schedule" ]]; then
  echo "✓ Operation: Manage backup schedule"
  OPERATION="schedule"
elif [[ "$ARGUMENTS" == "--validate" ]]; then
  echo "✓ Operation: Validate existing backups"
  OPERATION="validate"
elif [[ "$ARGUMENTS" == "--cleanup" ]]; then
  echo "✓ Operation: Cleanup old backups"
  OPERATION="cleanup"
else
  echo "❌ Error: Invalid operation '$ARGUMENTS'"
  echo ""
  echo "Valid operations:"
  echo "  --backup     Create new database backup"
  echo "  --restore    Restore from existing backup"
  echo "  --schedule   Manage automated backup schedules"
  echo "  --validate   Validate backup integrity"
  echo "  --cleanup    Remove old or unused backups"
  echo ""
  echo "Usage: /supabase-backup-manager [operation]"
  echo "Example: /supabase-backup-manager --backup"
  exit 1
fi
```

---

## Current Backup Context

- Supabase project: MCP integration for backup operations and status monitoring
- Backup storage: Current backup configuration and storage capacity
- Recovery testing: Last backup validation and recovery procedure verification
- Automation status: !`find . -name "*.yml" -o -name "*.json" | xargs grep -l "backup\|cron" 2>/dev/null | head -3` scheduled backup configuration

## Task

Execute comprehensive backup management with automated procedures and recovery validation:

**Backup Operation**: Use $ARGUMENTS to specify backup creation, data restoration, schedule management, backup validation, or cleanup procedures

**Backup Management Framework**:
1. **Backup Strategy** - Design backup schedules, implement retention policies, configure incremental backups, optimize storage usage
2. **Automated Backup** - Create database snapshots, export schema and data, validate backup integrity, monitor backup completion
3. **Recovery Procedures** - Test restore processes, validate data integrity, implement point-in-time recovery, optimize recovery time
4. **Schedule Management** - Configure automated backup schedules, implement backup monitoring, setup failure notifications, optimize backup windows
5. **Storage Optimization** - Manage backup storage, implement compression strategies, archive old backups, monitor storage costs
6. **Disaster Recovery** - Plan disaster recovery procedures, test recovery scenarios, document recovery processes, validate business continuity

**Advanced Features**: Automated backup validation, recovery time optimization, cross-region backup replication, backup encryption, compliance reporting.

**Monitoring Integration**: Backup success monitoring, failure alerting, storage usage tracking, recovery time measurement, compliance reporting.

**Output**: Complete backup management system with automated schedules, recovery procedures, validation reports, and disaster recovery planning.