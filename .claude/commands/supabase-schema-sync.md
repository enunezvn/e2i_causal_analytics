---
allowed-tools: Read, Write, Edit, Bash
argument-hint: [action] | --pull | --push | --diff | --validate
description: Synchronize database schema with Supabase using MCP integration
---

# Supabase Schema Sync

Synchronize database schema between local and Supabase with comprehensive validation: **$ARGUMENTS**

---

## Input Validation

Before proceeding, validate and parse the sync action argument.

```bash
# Optional sync action with specific flags
if [ -z "$ARGUMENTS" ]; then
  echo "ℹ️  No action specified, will show schema diff"
  ACTION="diff"
elif [[ "$ARGUMENTS" == "--pull" ]]; then
  echo "✓ Action: Pull schema from Supabase"
  ACTION="pull"
elif [[ "$ARGUMENTS" == "--push" ]]; then
  echo "✓ Action: Push schema to Supabase"
  ACTION="push"
elif [[ "$ARGUMENTS" == "--diff" ]]; then
  echo "✓ Action: Compare local and remote schemas"
  ACTION="diff"
elif [[ "$ARGUMENTS" == "--validate" ]]; then
  echo "✓ Action: Validate schema consistency"
  ACTION="validate"
else
  echo "❌ Error: Invalid sync action '$ARGUMENTS'"
  echo ""
  echo "Valid sync actions:"
  echo "  --pull       Pull schema from Supabase to local"
  echo "  --push       Push local schema to Supabase"
  echo "  --diff       Show differences between local and remote"
  echo "  --validate   Validate schema consistency"
  echo ""
  echo "Usage: /supabase-schema-sync [action]"
  echo "Example: /supabase-schema-sync --pull"
  exit 1
fi
```

---

## Current Supabase Context

- MCP connection: Supabase MCP server with read-only access configured
- Local schema: !`find . -name "schema.sql" -o -name "migrations" -type d | head -3` local database files
- Project config: !`find . -name "supabase" -type d -o -name ".env*" | grep -v node_modules | head -3` configuration files
- Git status: !`git status --porcelain | grep -E "\\.sql$|\\.ts$" | head -5` database-related changes

## Task

Execute comprehensive schema synchronization with Supabase integration:

**Sync Action**: Use $ARGUMENTS to specify pull from remote, push to remote, diff comparison, or schema validation

**Schema Synchronization Framework**:
1. **MCP Integration** - Connect to Supabase via MCP server, authenticate with project credentials, validate connection status
2. **Schema Analysis** - Compare local vs remote schema, identify structural differences, analyze migration requirements, assess breaking changes
3. **Sync Operations** - Execute pull/push operations, apply schema migrations, handle conflict resolution, validate data integrity
4. **Validation Process** - Verify schema consistency, validate foreign key constraints, check index performance, test query compatibility
5. **Migration Management** - Generate migration scripts, track version history, implement rollback procedures, optimize execution order
6. **Safety Checks** - Backup critical data, validate permissions, check production impact, implement dry-run mode

**Advanced Features**: Automated conflict resolution, schema version control, performance impact analysis, team collaboration workflows, CI/CD integration.

**Quality Assurance**: Schema validation, data integrity checks, performance optimization, rollback readiness, team synchronization.

**Output**: Complete schema sync with validation reports, migration scripts, conflict resolution, and team collaboration updates.