# Hybrid Search SQL Script Fix Summary

**Date**: 2025-12-15
**Issue**: Column reference errors when executing `011_hybrid_search_functions.sql`
**Status**: ✅ FIXED in `011_hybrid_search_functions_fixed.sql`

---

## Problem

When executing the original `011_hybrid_search_functions.sql` script in Supabase, you encountered this error:

```
ERROR: 42703: column "description" does not exist
```

This occurred because the SQL script referenced columns that don't exist in the actual database schema.

---

## Root Cause Analysis

### Tables Referenced That Don't Exist:
1. **`insight_embeddings`** - This table was referenced in the vector search function but doesn't exist in the current schema

### Column Mismatches in Existing Tables:

#### `causal_paths` table:
- **Referenced**: `description` column
- **Actual schema**: Has NO `description` column
- **Available columns**: `causal_chain` (JSONB), `start_node`, `end_node`, `method_used`, `intermediate_nodes`, etc.

#### `agent_activities` table:
- **Referenced**: `analysis_results` as text
- **Actual schema**: `analysis_results` is JSONB type (needs casting to text)

#### `triggers` table:
- **Referenced**: `trigger_reason` ✅ (This was correct!)

---

## Solution Applied

### Fixed File: `011_hybrid_search_functions_fixed.sql`

### Changes Made:

#### 1. Vector Search Function (`hybrid_vector_search`)
**BEFORE** (referenced non-existent tables):
```sql
-- Search insight_embeddings (doesn't exist!)
-- Search episodic_memories
-- Search procedural_memories
```

**AFTER** (uses actual tables):
```sql
-- Search episodic_memories (uses em.description, em.embedding)
-- Search procedural_memories (uses pm.procedure_name, pm.trigger_embedding)
```

**Tables now searched**:
- `episodic_memories` - conversation history, user queries, agent actions
- `procedural_memories` - successful patterns, tool sequences

#### 2. Full-Text Search Function (`hybrid_fulltext_search`)

**BEFORE** (causal_paths):
```sql
setweight(to_tsvector('english', coalesce(description, '')), 'A') -- ERROR: doesn't exist
```

**AFTER** (causal_paths):
```sql
setweight(to_tsvector('english', coalesce(start_node, '')), 'A') ||
setweight(to_tsvector('english', coalesce(end_node, '')), 'A') ||
setweight(to_tsvector('english', coalesce(method_used, '')), 'B') ||
setweight(to_tsvector('english', coalesce(causal_chain::text, '')), 'C')
```

**Content field in SELECT**:
```sql
-- BEFORE:
cp.description as content  -- ERROR!

-- AFTER:
COALESCE(cp.start_node, '') || ' → ' || COALESCE(cp.end_node, '') || ': ' ||
COALESCE(cp.method_used, '') as content
```

#### 3. Indexes Updated

**BEFORE**: 6 indexes (3 GIN + 3 HNSW)
- Including indexes on non-existent `insight_embeddings` table

**AFTER**: 5 indexes (3 GIN + 2 HNSW)
- `idx_causal_paths_search` (GIN on search_vector)
- `idx_agent_activities_search` (GIN on search_vector)
- `idx_triggers_search` (GIN on search_vector)
- `idx_episodic_memories_vector_hnsw` (HNSW on embedding)
- `idx_procedural_memories_vector_hnsw` (HNSW on trigger_embedding)

---

## What You Need to Do

### Step 1: Use the Fixed Script
Run `011_hybrid_search_functions_fixed.sql` instead of the original `011_hybrid_search_functions.sql`

### Step 2: Verify the Fix
After running the fixed script, execute this validation query:

```sql
SELECT * FROM get_search_stats();
```

**Expected Output**:
```
table_name           | total_rows | has_search_vector | has_embedding_index
episodic_memories    | N          | true              | true
procedural_memories  | N          | false             | true
causal_paths         | N          | true              | false
agent_activities     | N          | true              | false
triggers             | N          | true              | false
```

### Step 3: Test Full-Text Search
```sql
SELECT
    id,
    LEFT(content, 100) as content_preview,
    rank,
    source_table
FROM hybrid_fulltext_search('causal effect', 5)
ORDER BY rank DESC;
```

**Expected**: No errors, returns results if data exists

### Step 4: Test Vector Search Function
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

---

## Database Objects Created

### ✅ 3 Search Vector Columns (Generated Columns)
1. `causal_paths.search_vector` - Auto-updates from start_node, end_node, method_used, causal_chain
2. `agent_activities.search_vector` - Auto-updates from agent_name, activity_type, analysis_results
3. `triggers.search_vector` - Auto-updates from trigger_reason, trigger_type, recommended_action

### ✅ 5 Indexes
1. `idx_causal_paths_search` (GIN)
2. `idx_agent_activities_search` (GIN)
3. `idx_triggers_search` (GIN)
4. `idx_episodic_memories_vector_hnsw` (HNSW for vector search)
5. `idx_procedural_memories_vector_hnsw` (HNSW for vector search)

### ✅ 5 Functions
1. `hybrid_vector_search(vector, int, jsonb)` - Semantic search across episodic & procedural memories
2. `hybrid_fulltext_search(text, int, jsonb)` - Full-text search across causal_paths, agent_activities, triggers
3. `test_vector_search(text, int)` - Test helper for vector search
4. `test_fulltext_search(text, int)` - Test helper for full-text search
5. `get_search_stats()` - Returns statistics on searchable tables

---

## Key Differences from Original

| Aspect | Original (Broken) | Fixed Version |
|--------|------------------|---------------|
| Vector search tables | 3 tables (incl. non-existent `insight_embeddings`) | 2 tables (episodic_memories, procedural_memories) |
| `causal_paths` columns | Used non-existent `description` | Uses `start_node`, `end_node`, `method_used`, `causal_chain` |
| Total indexes | 6 (3 broken) | 5 (all working) |
| Return type IDs | uuid | uuid (correct) |
| Metadata fields | Generic | Specific to each table's schema |

---

## Future Considerations

### If You Want to Add `insight_embeddings` Table Later

You can create it with this structure:
```sql
CREATE TABLE insight_embeddings (
    insight_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(1536),
    source_type VARCHAR(50),
    agent_name VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_insight_embeddings_vector
    ON insight_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

Then add it to `hybrid_vector_search` function:
```sql
UNION ALL
SELECT
    ie.insight_id as id,
    ie.content,
    1 - (ie.embedding <=> query_embedding) as similarity,
    ie.metadata,
    'insight_embeddings'::text as source_table
FROM insight_embeddings ie
WHERE (1 - (ie.embedding <=> query_embedding)) > 0.5
```

---

## 🔄 Additional Fix Applied (2025-12-15)

### Issue: Ambiguous Column Reference in `get_search_stats()`

**Error Message**:
```
ERROR: 42702: column reference "table_name" is ambiguous
DETAIL: It could refer to either a PL/pgSQL variable or a table column.
```

**Root Cause**:
The function's RETURN TABLE defined an output column named `table_name`, and the SQL query also referenced `information_schema.columns.table_name`, creating ambiguity for PostgreSQL.

**Fix Applied**:
1. **Renamed output column**: `table_name` → `tbl_name`
2. **Added table aliases** in EXISTS subqueries:
   - `information_schema.columns c` → reference as `c.table_name`
   - `pg_indexes idx` → reference as `idx.tablename`

**Before (Broken)**:
```sql
RETURNS TABLE (
    table_name text,  -- Conflicts with column references below!
    ...
)
...
WHERE table_name = 'episodic_memories'  -- Ambiguous!
```

**After (Fixed)**:
```sql
RETURNS TABLE (
    tbl_name text,  -- Renamed to avoid conflict
    ...
)
...
WHERE c.table_name = 'episodic_memories'  -- Qualified with alias
```

This ensures PostgreSQL knows we're referring to the actual table columns in information_schema, not the output column.

---

## 🔄 Additional Fix Applied (2025-12-15 - Final)

### Issue 3: ID Type Mismatch

**Error Message**:
```
ERROR: 22P02: invalid input syntax for type uuid: "AA_000000"
```

**Root Cause**:
The hybrid search functions declared ID columns as `uuid`, but actual tables use mixed ID types:
- `episodic_memories.memory_id`: UUID ✓
- `procedural_memories.procedure_id`: UUID ✓
- `causal_paths.path_id`: VARCHAR (not UUID!)
- `agent_activities.activity_id`: VARCHAR (not UUID!)
- `triggers.trigger_id`: VARCHAR (not UUID!)

**Fix Applied**:
1. Changed return type in `hybrid_vector_search()`: `id uuid` → `id text`
2. Changed return type in `hybrid_fulltext_search()`: `id uuid` → `id text`
3. Added `::text` casts to all ID columns:
   - `em.memory_id::text`
   - `pm.procedure_id::text`
   - `cp.path_id::text`
   - `aa.activity_id::text`
   - `t.trigger_id::text`
4. Updated helper functions to return `id text`

### Issue 4: Type Mismatch in Full-Text Search Rank

**Error Message**:
```
ERROR: 42804: structure of query does not match function result type
DETAIL: Returned type real does not match expected type double precision in column 3
```

**Root Cause**:
- `ts_rank_cd()` returns `real` type (4-byte float)
- Function declared `rank float` which PostgreSQL interprets as `double precision` (8-byte float)
- Type mismatch in UNION ALL branches

**Fix Applied**:
1. Changed return type in `hybrid_fulltext_search()`: `rank float` → `rank double precision`
2. Added explicit cast to all `ts_rank_cd()` calls: `ts_rank_cd(...)::double precision`
3. Updated `test_fulltext_search()` helper function to return `rank double precision`

---

## Validation Checklist

After running the fixed script:

- [x] pgvector extension is enabled
- [x] 3 search_vector columns exist (causal_paths, agent_activities, triggers)
- [x] 5 indexes created successfully
- [x] 5 functions created and executable
- [x] `get_search_stats()` returns data without errors (column now named `tbl_name`)
- [x] `hybrid_fulltext_search()` executes without column errors
- [x] `hybrid_vector_search()` function exists and is callable
- [x] No permission errors when running test queries
- [x] All type casting issues resolved (uuid → text, real → double precision)

---

## Files to Use

✅ **Use these**:
- `database/memory/011_hybrid_search_functions_fixed.sql` - Fully corrected migration (all 4 issues fixed)
- `database/memory/IMPLEMENTATION_GUIDE.md` - Step-by-step setup guide
- `database/memory/011_validation_queries.sql` - Validation queries
- `database/memory/FIX_SUMMARY.md` - This document (complete fix history)

❌ **Don't use**:
- `database/memory/011_hybrid_search_functions.sql` - Original version with multiple issues

---

## Success Criteria

Your migration is successful when:

1. ✅ All SQL runs without errors
2. ✅ `SELECT * FROM get_search_stats();` returns 5 rows with `tbl_name` column
3. ✅ Full-text search works: `SELECT * FROM hybrid_fulltext_search('causal', 5);`
4. ✅ Vector search function exists and is callable (actual search requires embeddings)
5. ✅ All 5 indexes are visible in pg_indexes
6. ✅ No "column does not exist" errors
7. ✅ No "invalid input syntax for type uuid" errors
8. ✅ No "type mismatch" errors for rank values

---

## Execution Results (2025-12-15)

Successfully executed via Supabase MCP:

**Database Statistics**:
- episodic_memories: 0 rows (has search_vector & embedding index)
- procedural_memories: 0 rows (has embedding index)
- causal_paths: 50 rows (has search_vector)
- agent_activities: 2,323 rows (has search_vector)
- triggers: 606 rows (has search_vector)

**Test Results**:
- ✅ Full-text search for "causal": 5 results returned from agent_activities
- ✅ All functions created and permissions granted
- ✅ All validation queries execute successfully

---

---

## 🔄 Additional Fix Applied (2025-12-15 - Validation Query)

### Issue 5: Validation Query Counting Wrong Number of Indexes

**Problem**:
The validation summary query was counting 4 indexes instead of 5 because:
- Pattern `indexname LIKE 'idx_%_vector'` didn't match `idx_episodic_memories_vector_hnsw` (has `_hnsw` suffix)
- Pattern would match pre-existing `idx_episodic_search` if broadened to `%vector%`

**Root Cause**:
- Our HNSW indexes are named `idx_%_vector_hnsw`, not `idx_%_vector`
- There's a pre-existing index `idx_episodic_search` that shouldn't be counted

**Fix Applied**:
Updated validation query pattern from:
```sql
indexname LIKE 'idx_%_search' OR indexname LIKE 'idx_%_vector'
```

To:
```sql
(indexname LIKE 'idx_%_search' AND indexname != 'idx_episodic_search')
OR indexname LIKE 'idx_%vector_hnsw'
```

This correctly counts our 5 new indexes:
1. `idx_causal_paths_search`
2. `idx_agent_activities_search`
3. `idx_triggers_search`
4. `idx_episodic_memories_vector_hnsw`
5. `idx_procedural_memories_vector_hnsw`

And excludes the pre-existing `idx_episodic_search`.

---

**Status**: ✅ **FULLY IMPLEMENTED, TESTED, AND VALIDATED IN SUPABASE**
**Next Step**: Implement Python RAG module (src/rag/) to use these search functions
