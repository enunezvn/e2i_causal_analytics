-- ============================================================================
-- HYBRID SEARCH VALIDATION QUERIES
-- ============================================================================
-- Purpose: Verify that hybrid search functions are installed correctly
-- Run these queries in Supabase SQL Editor after running 011_hybrid_search_functions.sql
-- ============================================================================

-- ============================================================================
-- STEP 1: VERIFY PGVECTOR EXTENSION
-- ============================================================================

-- Check if pgvector extension is installed
SELECT
    extname,
    extversion,
    'Extension installed' as status
FROM pg_extension
WHERE extname = 'vector';

-- Expected output: One row showing 'vector' extension
-- If empty, run: CREATE EXTENSION vector;

-- ============================================================================
-- STEP 2: VERIFY SEARCH VECTOR COLUMNS
-- ============================================================================

-- Check that search_vector columns were added
SELECT
    table_name,
    column_name,
    data_type,
    'Column exists' as status
FROM information_schema.columns
WHERE table_schema = 'public'
  AND column_name = 'search_vector'
  AND table_name IN ('causal_paths', 'agent_activities', 'triggers')
ORDER BY table_name;

-- Expected output: 3 rows (one for each table)

-- ============================================================================
-- STEP 3: VERIFY INDEXES
-- ============================================================================

-- Check that all indexes were created
SELECT
    tablename,
    indexname,
    indexdef,
    'Index exists' as status
FROM pg_indexes
WHERE schemaname = 'public'
  AND (
    indexname LIKE 'idx_%_search' OR
    indexname LIKE 'idx_%_vector'
  )
ORDER BY tablename, indexname;

-- Expected output: 5 indexes
-- - idx_causal_paths_search (GIN)
-- - idx_agent_activities_search (GIN)
-- - idx_triggers_search (GIN)
-- - idx_episodic_memories_vector_hnsw (HNSW)
-- - idx_procedural_memories_vector_hnsw (HNSW)

-- ============================================================================
-- STEP 4: VERIFY FUNCTIONS
-- ============================================================================

-- Check that all functions were created
SELECT
    routine_name,
    routine_type,
    data_type as return_type,
    'Function exists' as status
FROM information_schema.routines
WHERE routine_schema = 'public'
  AND routine_name IN (
    'hybrid_vector_search',
    'hybrid_fulltext_search',
    'test_vector_search',
    'test_fulltext_search',
    'get_search_stats'
  )
ORDER BY routine_name;

-- Expected output: 5 rows (one for each function)

-- ============================================================================
-- STEP 5: GET SEARCH STATISTICS
-- ============================================================================

-- Run the stats function to see table info
SELECT * FROM get_search_stats();

-- Expected output: 6 rows showing:
-- - Table name
-- - Row count
-- - Whether search_vector column exists (for fulltext tables)
-- - Whether embedding index exists (for vector tables)

-- ============================================================================
-- STEP 6: TEST FULL-TEXT SEARCH
-- ============================================================================

-- Test fulltext search with a sample query
SELECT
    id,
    LEFT(content, 100) as content_preview,
    rank,
    source_table
FROM hybrid_fulltext_search('conversion rate', 5)
ORDER BY rank DESC;

-- Expected: Results from causal_paths, agent_activities, or triggers
-- If no results, it means no data matches 'conversion rate'

-- Try another search
SELECT
    id,
    LEFT(content, 100) as content_preview,
    rank,
    source_table
FROM hybrid_fulltext_search('Remibrutinib', 5)
ORDER BY rank DESC;

-- Test with filters
SELECT
    id,
    LEFT(content, 100) as content_preview,
    rank,
    metadata->>'brand' as brand,
    source_table
FROM hybrid_fulltext_search(
    'coverage gap',
    10,
    '{"brand": "Remibrutinib"}'::jsonb
)
ORDER BY rank DESC;

-- ============================================================================
-- STEP 7: TEST VECTOR SEARCH (requires actual embeddings)
-- ============================================================================

-- First, check if we have any embeddings
SELECT
    'insight_embeddings' as table_name,
    COUNT(*) as embedding_count,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as non_null_embeddings
FROM insight_embeddings

UNION ALL

SELECT
    'episodic_memories' as table_name,
    COUNT(*) as embedding_count,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as non_null_embeddings
FROM episodic_memories

UNION ALL

SELECT
    'procedural_memories' as table_name,
    COUNT(*) as embedding_count,
    COUNT(*) FILTER (WHERE trigger_embedding IS NOT NULL) as non_null_embeddings
FROM procedural_memories;

-- If you have embeddings, test vector search
-- Note: Replace the zero vector with an actual embedding from OpenAI
-- For now, test with a zero vector (won't return meaningful results)
SELECT
    id,
    LEFT(content, 100) as content_preview,
    similarity,
    source_table
FROM hybrid_vector_search(
    array_fill(0.0, ARRAY[1536])::vector(1536),
    5
)
ORDER BY similarity DESC;

-- ============================================================================
-- STEP 8: VERIFY FUNCTION PERMISSIONS
-- ============================================================================

-- Check that functions are executable by authenticated role
SELECT
    routine_name,
    privilege_type,
    grantee
FROM information_schema.routine_privileges
WHERE routine_schema = 'public'
  AND routine_name IN (
    'hybrid_vector_search',
    'hybrid_fulltext_search'
  )
  AND grantee = 'authenticated';

-- Expected output: EXECUTE privileges for authenticated role

-- ============================================================================
-- STEP 9: PERFORMANCE CHECK
-- ============================================================================

-- Check index sizes (helps identify if indexes are built)
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND (
    indexrelname LIKE 'idx_%_search' OR
    indexrelname LIKE 'idx_%_vector'
  )
ORDER BY pg_relation_size(indexrelid) DESC;

-- Expected: Non-zero sizes for all indexes (if data exists)

-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as indexes_size
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename IN (
    'insight_embeddings',
    'episodic_memories',
    'procedural_memories',
    'causal_paths',
    'agent_activities',
    'triggers'
  )
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- ============================================================================
-- STEP 10: TEST EDGE CASES
-- ============================================================================

-- Test with empty query (should handle gracefully)
SELECT * FROM hybrid_fulltext_search('', 5);
-- Expected: No results or empty set

-- Test with very long query
SELECT * FROM hybrid_fulltext_search(
    'conversion rate drop in South region due to coverage gaps and HCP engagement issues',
    5
);
-- Expected: Relevant results if data exists

-- Test with special characters
SELECT * FROM hybrid_fulltext_search('Remibrutinib & Fabhalta', 5);
-- Expected: Results for both brands

-- Test with filters that have no matches
SELECT * FROM hybrid_fulltext_search(
    'coverage',
    5,
    '{"brand": "NonExistentBrand"}'::jsonb
);
-- Expected: No results

-- ============================================================================
-- VALIDATION SUMMARY
-- ============================================================================

-- Run this summary query to get overall status
SELECT
    'Validation Summary' as check_type,
    (SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector') as pgvector_installed,
    (SELECT COUNT(*) FROM information_schema.columns WHERE column_name = 'search_vector' AND table_name IN ('causal_paths', 'agent_activities', 'triggers')) as search_columns_added,
    (SELECT COUNT(*) FROM pg_indexes WHERE (indexname LIKE 'idx_%_search' AND indexname != 'idx_episodic_search') OR indexname LIKE 'idx_%vector_hnsw') as indexes_created,
    (SELECT COUNT(*) FROM information_schema.routines WHERE routine_name IN ('hybrid_vector_search', 'hybrid_fulltext_search', 'test_vector_search', 'test_fulltext_search', 'get_search_stats')) as functions_created;

-- Expected output:
-- pgvector_installed: 1
-- search_columns_added: 3
-- indexes_created: 5 (excludes pre-existing idx_episodic_search)
-- functions_created: 5

-- ============================================================================
-- TROUBLESHOOTING
-- ============================================================================

-- If pgvector extension is missing:
-- CREATE EXTENSION IF NOT EXISTS vector;

-- If functions are missing, check for errors:
-- SELECT * FROM pg_stat_user_functions WHERE schemaname = 'public';

-- To see function definitions:
-- SELECT routine_name, routine_definition
-- FROM information_schema.routines
-- WHERE routine_schema = 'public'
--   AND routine_name LIKE 'hybrid_%';

-- To check for any errors in function creation:
-- \df hybrid_*

-- ============================================================================
-- SUCCESS CRITERIA
-- ============================================================================

/*
Your implementation is successful if:

1. ✅ pgvector extension is installed
2. ✅ All 3 search_vector columns exist (causal_paths, agent_activities, triggers)
3. ✅ All 5 indexes are created (3 GIN, 2 HNSW)
4. ✅ All 5 functions are created and executable
5. ✅ get_search_stats() returns data
6. ✅ hybrid_fulltext_search() returns results (if data exists)
7. ✅ No errors when running validation queries
8. ✅ Index sizes are non-zero (if data exists)

If all checks pass, proceed to:
- Update PROJECT_STRUCTURE.txt
- Mark tasks as complete in todo list
- Begin FalkorDB graph schema implementation
*/
