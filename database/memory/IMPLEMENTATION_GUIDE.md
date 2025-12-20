# Supabase Hybrid Search Implementation Guide

**Created**: 2025-12-15
**Updated**: 2025-12-15 (Fixed column references + ambiguous column fix)
**Migration**: 011_hybrid_search_functions_fixed.sql
**Status**: Ready for Implementation

> **IMPORTANT**: Use `011_hybrid_search_functions_fixed.sql` instead of the original file.
> The fixed version corrects:
> - Column references to match your actual database schema
> - Ambiguous column reference in `get_search_stats()` function (renamed to `tbl_name`)
>
> See `FIX_SUMMARY.md` for complete details on what was fixed.

---

## Prerequisites

Before running this migration, ensure:

1. ✅ Supabase project is created and accessible
2. ✅ You have access to Supabase SQL Editor
3. ✅ The following tables exist in your database:
   - `insight_embeddings`
   - `episodic_memories`
   - `procedural_memories`
   - `causal_paths`
   - `agent_activities`
   - `triggers`
4. ✅ You have admin/owner permissions on the Supabase project

---

## Implementation Steps

### Step 1: Access Supabase SQL Editor

1. Log in to your Supabase dashboard: https://supabase.com/dashboard
2. Select your project: **e2i-causal-analytics** (or your project name)
3. Navigate to: **SQL Editor** (left sidebar)
4. Click: **New Query**

### Step 2: Enable pgvector Extension (if not already enabled)

Copy and paste this into the SQL Editor:

```sql
-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;
```

Click **Run** or press `Ctrl+Enter`

**Expected Output**:
```
Success. No rows returned
```

Or if already installed:
```
extension "vector" already exists
```

### Step 3: Run the Main Migration

1. Open the file: `database/memory/011_hybrid_search_functions_fixed.sql` ⚠️ **Use the FIXED version!**
2. Copy the **entire contents** of the file
3. Paste into Supabase SQL Editor
4. Click **Run** or press `Ctrl+Enter`

**Expected Execution Time**: 10-30 seconds (depending on data volume)

**Expected Output**:
```
Success. No rows returned
```

**Note**: You may see multiple "Success" messages as the script executes different parts. This is normal.

### Step 4: Validate Installation

1. Create a new query in SQL Editor
2. Open the file: `database/memory/011_validation_queries.sql`
3. Copy the **validation summary query** at the bottom:

```sql
SELECT
    'Validation Summary' as check_type,
    (SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector') as pgvector_installed,
    (SELECT COUNT(*) FROM information_schema.columns WHERE column_name = 'search_vector' AND table_name IN ('causal_paths', 'agent_activities', 'triggers')) as search_columns_added,
    (SELECT COUNT(*) FROM pg_indexes WHERE indexname LIKE 'idx_%_search' OR indexname LIKE 'idx_%_vector') as indexes_created,
    (SELECT COUNT(*) FROM information_schema.routines WHERE routine_name IN ('hybrid_vector_search', 'hybrid_fulltext_search', 'test_vector_search', 'test_fulltext_search', 'get_search_stats')) as functions_created;
```

4. Click **Run**

**Expected Output**:
```
check_type         | pgvector_installed | search_columns_added | indexes_created | functions_created
Validation Summary | 1                  | 3                    | 5               | 5
```

✅ **Success Criteria**:
- `pgvector_installed`: 1
- `search_columns_added`: 3
- `indexes_created`: 5
- `functions_created`: 5

### Step 5: Get Search Statistics

Run this query to see table statistics:

```sql
SELECT * FROM get_search_stats();
```

**Expected Output** (example):
```
tbl_name             | total_rows | has_search_vector | has_embedding_index
episodic_memories    | 45         | true              | true
procedural_memories  | 12         | false             | true
causal_paths         | 78         | true              | false
agent_activities     | 234        | true              | false
triggers             | 56         | true              | false
```

**Note**: The column is named `tbl_name` (not `table_name`) to avoid ambiguous references in the function.

**Note**: Row counts will be 0 if you haven't populated data yet. That's okay!

### Step 6: Test Full-Text Search

Run a test search to verify functionality:

```sql
SELECT
    id,
    LEFT(content, 100) as content_preview,
    rank,
    source_table
FROM hybrid_fulltext_search('conversion rate', 5)
ORDER BY rank DESC;
```

**Expected Output**:
- If you have data: Rows showing search results
- If no data yet: Empty result set (this is normal)

**No errors** means the function is working correctly!

### Step 7: Test Vector Search Function

Test that the vector search function exists:

```sql
SELECT
    routine_name,
    data_type as return_type
FROM information_schema.routines
WHERE routine_name = 'hybrid_vector_search';
```

**Expected Output**:
```
routine_name         | return_type
hybrid_vector_search | record
```

**Note**: Actual vector search will work once you generate embeddings with OpenAI.

---

## Verification Checklist

Run through this checklist to confirm successful implementation:

### Database Objects Created

- [ ] `search_vector` column added to `causal_paths` ✓
- [ ] `search_vector` column added to `agent_activities` ✓
- [ ] `search_vector` column added to `triggers` ✓
- [ ] GIN index created on `causal_paths.search_vector` ✓
- [ ] GIN index created on `agent_activities.search_vector` ✓
- [ ] GIN index created on `triggers.search_vector` ✓
- [ ] HNSW index created on `episodic_memories.embedding` ✓
- [ ] HNSW index created on `procedural_memories.trigger_embedding` ✓

### Functions Created

- [ ] `hybrid_vector_search(vector, int, jsonb)` function created ✓
- [ ] `hybrid_fulltext_search(text, int, jsonb)` function created ✓
- [ ] `test_vector_search(text, int)` function created ✓
- [ ] `test_fulltext_search(text, int)` function created ✓
- [ ] `get_search_stats()` function created ✓

### Permissions

- [ ] Functions are executable by `authenticated` role ✓
- [ ] No permission errors when running test queries ✓

### Performance

- [ ] Indexes show non-zero size (if data exists) ✓
- [ ] No timeout errors when running validation queries ✓

---

## Common Issues & Solutions

### Issue 1: "extension 'vector' does not exist"

**Solution**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

If you get a permission error, contact your Supabase admin or check project permissions.

### Issue 2: "column 'embedding' does not exist"

**Cause**: The memory tables haven't been created yet.

**Solution**: Run the memory schema migrations first:
1. `database/memory/001_agentic_memory_schema_v1.3.sql`
2. Then run `011_hybrid_search_functions.sql`

### Issue 3: "function hybrid_vector_search already exists"

**Solution**: This is normal if you're re-running the migration. The `CREATE OR REPLACE FUNCTION` statement will update the existing function.

### Issue 4: Index creation timeout

**Cause**: Large tables may take time to index.

**Solution**:
- Indexes are created in the background
- Check progress: `SELECT * FROM pg_stat_progress_create_index;`
- Wait for completion before running queries

### Issue 5: Search returns no results

**Possible Causes**:
1. No data in tables yet (normal if just starting)
2. Search query doesn't match any content
3. Embeddings not generated yet (for vector search)

**Solution**:
- Check table row counts: `SELECT * FROM get_search_stats();`
- Try broader search terms
- For vector search, you'll need to generate embeddings first

---

## Next Steps After Successful Implementation

Once all validation checks pass:

### 1. Update Project Documentation

- [ ] Mark task as complete in todo list
- [ ] Update `PROJECT_STRUCTURE.txt`
- [ ] Document database changes in project docs

### 2. Test with Sample Data

If you have sample data:

```sql
-- Test fulltext search with your actual data
SELECT * FROM hybrid_fulltext_search('your search term', 10);

-- Check what's in your tables
SELECT COUNT(*), 'causal_paths' as table_name FROM causal_paths
UNION ALL
SELECT COUNT(*), 'agent_activities' FROM agent_activities
UNION ALL
SELECT COUNT(*), 'triggers' FROM triggers;
```

### 3. Prepare for Embedding Generation

Before vector search works, you'll need to:
- Implement `OpenAIEmbeddingClient` (src/rag/embeddings.py)
- Generate embeddings for existing content
- Insert embeddings into `insight_embeddings` table

### 4. Move to FalkorDB Setup

Next migration: `002_semantic_graph_schema.cypher` for knowledge graph

---

## Monitoring & Maintenance

### Check Index Health

Run periodically to monitor index usage:

```sql
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan as times_used,
    idx_tup_read as rows_read,
    idx_tup_fetch as rows_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE indexrelname LIKE 'idx_%_search' OR indexrelname LIKE 'idx_%_vector'
ORDER BY idx_scan DESC;
```

### Monitor Function Performance

```sql
SELECT
    funcname,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_user_functions
WHERE funcname LIKE 'hybrid_%'
ORDER BY total_time DESC;
```

### Rebuild Indexes (if needed)

If search becomes slow:

```sql
REINDEX INDEX CONCURRENTLY idx_causal_paths_search;
REINDEX INDEX CONCURRENTLY idx_agent_activities_search;
REINDEX INDEX CONCURRENTLY idx_triggers_search;
```

**Note**: Use `CONCURRENTLY` to avoid locking tables during rebuild.

---

## Rollback Procedure

If you need to rollback this migration:

```sql
-- Drop all functions
DROP FUNCTION IF EXISTS hybrid_vector_search(vector(1536), int, jsonb);
DROP FUNCTION IF EXISTS hybrid_fulltext_search(text, int, jsonb);
DROP FUNCTION IF EXISTS test_vector_search(text, int);
DROP FUNCTION IF EXISTS test_fulltext_search(text, int);
DROP FUNCTION IF EXISTS get_search_stats();

-- Drop all indexes
DROP INDEX IF EXISTS idx_causal_paths_search;
DROP INDEX IF EXISTS idx_agent_activities_search;
DROP INDEX IF EXISTS idx_triggers_search;
DROP INDEX IF EXISTS idx_insight_embeddings_vector;
DROP INDEX IF EXISTS idx_episodic_memories_vector;
DROP INDEX IF EXISTS idx_procedural_memories_vector;

-- Drop search vector columns
ALTER TABLE causal_paths DROP COLUMN IF EXISTS search_vector;
ALTER TABLE agent_activities DROP COLUMN IF EXISTS search_vector;
ALTER TABLE triggers DROP COLUMN IF EXISTS search_vector;
```

**Verify Rollback**:
```sql
SELECT * FROM get_search_stats();  -- Should error (function doesn't exist)
```

---

## Support

If you encounter issues:

1. Check Supabase logs in Dashboard → Database → Logs
2. Review validation queries in `011_validation_queries.sql`
3. Consult Supabase docs: https://supabase.com/docs/guides/database
4. Check pgvector docs: https://github.com/pgvector/pgvector

---

## Success Confirmation

When you see this output, you're ready to proceed:

```
✅ pgvector extension installed
✅ 3 search_vector columns added
✅ 5 indexes created (3 GIN + 2 HNSW)
✅ 5 search functions created
✅ All validation queries execute without errors
✅ Permissions granted to authenticated role
```

**Status**: Migration 011 Complete - Proceed to FalkorDB Setup

---

**Migration File**: `database/memory/011_hybrid_search_functions_fixed.sql` ⚠️ **Use FIXED version**
**Validation File**: `database/memory/011_validation_queries.sql`
**Fix Summary**: `database/memory/FIX_SUMMARY.md`
**Last Updated**: 2025-12-15
