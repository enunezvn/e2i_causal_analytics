-- Create the get_rag_search_stats function
-- This function provides RAG usage statistics for the /api/v1/rag/stats endpoint

CREATE OR REPLACE FUNCTION get_rag_search_stats(
    hours_lookback int DEFAULT 24
)
RETURNS jsonb
LANGUAGE plpgsql
AS $func$
DECLARE
    result jsonb;
    cutoff_time timestamptz;
BEGIN
    cutoff_time := NOW() - (hours_lookback || ' hours')::interval;

    SELECT jsonb_build_object(
        'period_hours', hours_lookback,
        'total_searches', COALESCE(COUNT(*), 0),
        'avg_latency_ms', COALESCE(ROUND(AVG(total_latency_ms)::numeric, 2), 0),
        'p95_latency_ms', COALESCE(
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms)::numeric, 2),
            0
        ),
        'avg_results', COALESCE(ROUND(AVG(fused_count)::numeric, 2), 0),
        'error_rate', COALESCE(
            ROUND(
                (SUM(CASE WHEN errors != '[]' THEN 1 ELSE 0 END)::float /
                 NULLIF(COUNT(*), 0) * 100)::numeric,
                2
            ),
            0
        ),
        'backend_usage', jsonb_build_object(
            'vector', COALESCE(SUM(vector_count), 0),
            'fulltext', COALESCE(SUM(fulltext_count), 0),
            'graph', COALESCE(SUM(graph_count), 0)
        ),
        'top_queries', COALESCE(
            (SELECT jsonb_agg(q.query_info)
             FROM (
                 SELECT jsonb_build_object(
                     'query', query,
                     'count', COUNT(*),
                     'avg_latency_ms', ROUND(AVG(total_latency_ms)::numeric, 2)
                 ) as query_info
                 FROM rag_search_logs
                 WHERE created_at >= cutoff_time
                 GROUP BY query
                 ORDER BY COUNT(*) DESC
                 LIMIT 10
             ) q),
            '[]'::jsonb
        )
    ) INTO result
    FROM rag_search_logs
    WHERE created_at >= cutoff_time;

    RETURN result;
END;
$func$;

COMMENT ON FUNCTION get_rag_search_stats IS
'Get aggregated search statistics for the specified time period.
 Returns: JSONB with total_searches, avg_latency_ms, p95_latency_ms, error_rate, backend_usage, top_queries';

-- Grant permissions
GRANT EXECUTE ON FUNCTION get_rag_search_stats TO authenticated;
GRANT EXECUTE ON FUNCTION get_rag_search_stats TO anon;
GRANT EXECUTE ON FUNCTION get_rag_search_stats TO service_role;
