-- ============================================================================
-- RAG SCHEMA VALIDATION QUERIES
-- ============================================================================
-- Purpose: Verify that RAG schema is installed correctly
-- Run after 001_rag_schema.sql
-- ============================================================================

-- ============================================================================
-- STEP 1: VERIFY TABLES EXIST
-- ============================================================================

SELECT
    table_name,
    'Table exists' as status
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('rag_document_chunks', 'rag_search_logs')
ORDER BY table_name;

-- Expected: 2 rows

-- ============================================================================
-- STEP 2: VERIFY INDEXES EXIST
-- ============================================================================

SELECT
    tablename,
    indexname,
    indexdef,
    'Index exists' as status
FROM pg_indexes
WHERE schemaname = 'public'
  AND indexname LIKE 'idx_rag_%'
ORDER BY tablename, indexname;

-- Expected: Multiple indexes for rag_document_chunks and rag_search_logs

-- ============================================================================
-- STEP 3: VERIFY FUNCTIONS EXIST
-- ============================================================================

SELECT
    routine_name,
    routine_type,
    data_type as return_type,
    'Function exists' as status
FROM information_schema.routines
WHERE routine_schema = 'public'
  AND routine_name IN (
    'rag_vector_search',
    'rag_fulltext_search',
    'log_rag_search'
  )
ORDER BY routine_name;

-- Expected: 3 rows

-- ============================================================================
-- STEP 4: VERIFY VIEWS EXIST
-- ============================================================================

SELECT
    table_name as view_name,
    'View exists' as status
FROM information_schema.views
WHERE table_schema = 'public'
  AND table_name IN ('rag_slow_queries', 'rag_search_stats')
ORDER BY table_name;

-- Expected: 2 rows

-- ============================================================================
-- STEP 5: TEST RAG FULLTEXT SEARCH
-- ============================================================================

-- Test with sample query (may return no results if no data)
SELECT
    id,
    LEFT(content, 80) as content_preview,
    rank,
    source_table
FROM rag_fulltext_search('conversion rate', 5)
ORDER BY rank DESC;

-- ============================================================================
-- STEP 6: TEST LOG FUNCTION
-- ============================================================================

-- Insert a test log entry
SELECT log_rag_search(
    p_query := 'test query for validation',
    p_vector_count := 5,
    p_fulltext_count := 3,
    p_graph_count := 2,
    p_fused_count := 8,
    p_total_latency_ms := 150.5,
    p_vector_latency_ms := 80.0,
    p_fulltext_latency_ms := 30.0,
    p_graph_latency_ms := 40.0,
    p_sources_used := '{"vector": true, "fulltext": true, "graph": true}'::jsonb,
    p_errors := '[]'::jsonb
) as test_log_id;

-- Verify the log was created
SELECT
    log_id,
    query,
    total_latency_ms,
    fused_count,
    sources_used,
    created_at
FROM rag_search_logs
WHERE query = 'test query for validation'
ORDER BY created_at DESC
LIMIT 1;

-- ============================================================================
-- STEP 7: VERIFY COLUMN STRUCTURE
-- ============================================================================

-- Check rag_document_chunks columns
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'rag_document_chunks'
ORDER BY ordinal_position;

-- Check rag_search_logs columns
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'rag_search_logs'
ORDER BY ordinal_position;

-- ============================================================================
-- STEP 8: TEST INSERT INTO rag_document_chunks
-- ============================================================================

-- Insert a test document chunk
INSERT INTO rag_document_chunks (
    document_id,
    document_type,
    chunk_index,
    content,
    brand,
    region,
    metadata
) VALUES (
    'test-doc-001',
    'test_document',
    0,
    'This is a test document chunk for validation purposes.',
    'Remibrutinib',
    'West',
    '{"test": true}'::jsonb
)
ON CONFLICT (document_id, chunk_index) DO NOTHING
RETURNING chunk_id, document_id, content;

-- ============================================================================
-- STEP 9: VALIDATION SUMMARY
-- ============================================================================

SELECT
    'RAG Schema Validation' as check_type,
    (SELECT COUNT(*) FROM information_schema.tables
     WHERE table_name IN ('rag_document_chunks', 'rag_search_logs')) as tables_created,
    (SELECT COUNT(*) FROM pg_indexes WHERE indexname LIKE 'idx_rag_%') as indexes_created,
    (SELECT COUNT(*) FROM information_schema.routines
     WHERE routine_name IN ('rag_vector_search', 'rag_fulltext_search', 'log_rag_search')) as functions_created,
    (SELECT COUNT(*) FROM information_schema.views
     WHERE table_name IN ('rag_slow_queries', 'rag_search_stats')) as views_created;

-- Expected:
-- tables_created: 2
-- indexes_created: ~7 (depends on existing indexes)
-- functions_created: 3
-- views_created: 2

-- ============================================================================
-- CLEANUP TEST DATA (optional)
-- ============================================================================

/*
-- Remove test entries if desired
DELETE FROM rag_search_logs WHERE query = 'test query for validation';
DELETE FROM rag_document_chunks WHERE document_id = 'test-doc-001';
*/

-- ============================================================================
-- SUCCESS CRITERIA
-- ============================================================================

/*
Your RAG schema installation is successful if:

1. ✅ Both tables exist (rag_document_chunks, rag_search_logs)
2. ✅ All indexes are created
3. ✅ All 3 functions exist (rag_vector_search, rag_fulltext_search, log_rag_search)
4. ✅ Both views exist (rag_slow_queries, rag_search_stats)
5. ✅ log_rag_search returns a UUID
6. ✅ Test document chunk can be inserted
7. ✅ rag_fulltext_search runs without errors

If all checks pass, proceed to:
- Update HybridRetriever to use logging
- Implement embedding generation for document chunks
- Test integration with existing hybrid search
*/
