"""
Phase 6: pgvector Extension Validation Tests
=============================================
Tests for pgvector extension functionality in self-hosted Supabase.

Tests validate:
- pgvector extension installed and version
- Vector columns exist with correct dimensions
- HNSW and IVFFlat indexes created
- Vector search functions work correctly
- Fulltext search functions work correctly
- Search logging functions work correctly

Run with:
    ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
        source .env && SUPABASE_DB_HOST=172.22.0.4 \
        .venv/bin/pytest tests/integration/test_migration/test_phase6_pgvector.py -v"
"""

import uuid

import pytest


class TestPgvectorExtension:
    """Test pgvector extension installation."""

    def test_pgvector_installed(self, pg_connection):
        """pgvector extension should be installed."""
        with pg_connection.cursor() as cur:
            cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
            result = cur.fetchone()

        assert result is not None, "pgvector extension not found"
        assert result[0] == "vector"

    def test_pgvector_version(self, pg_connection):
        """pgvector should be at least version 0.5.0."""
        with pg_connection.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            result = cur.fetchone()

        assert result is not None
        version = result[0]
        major, minor = map(int, version.split(".")[:2])
        # pgvector 0.5.0+ has HNSW support
        assert (major, minor) >= (0, 5), f"pgvector {version} too old, need >= 0.5.0"


class TestVectorColumns:
    """Test vector columns exist with correct dimensions."""

    @pytest.mark.parametrize(
        "table_name,column_name,expected_dims",
        [
            ("episodic_memories", "embedding", 1536),
            ("procedural_memories", "trigger_embedding", 1536),
            ("cognitive_cycles", "query_embedding", 1536),
            ("rag_document_chunks", "embedding", 1536),
            ("experiment_knowledge_store", "embedding", 1536),
        ],
    )
    def test_vector_column_exists(self, pg_connection, table_name, column_name, expected_dims):
        """Vector columns should exist with 1536 dimensions."""
        with pg_connection.cursor() as cur:
            cur.execute(
                """
                SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type
                FROM pg_catalog.pg_attribute a
                JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
                JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'public'
                AND c.relname = %s
                AND a.attname = %s
                AND a.attnum > 0
                """,
                (table_name, column_name),
            )
            result = cur.fetchone()

        assert result is not None, f"Column {table_name}.{column_name} not found"
        assert result[0] == f"vector({expected_dims})", (
            f"Expected vector({expected_dims}), got {result[0]}"
        )


class TestVectorIndexes:
    """Test vector indexes are created correctly."""

    @pytest.mark.parametrize(
        "table_name,index_type",
        [
            ("episodic_memories", "hnsw"),
            ("procedural_memories", "hnsw"),
            ("rag_document_chunks", "hnsw"),
            ("episodic_memories", "ivfflat"),
            ("procedural_memories", "ivfflat"),
            ("cognitive_cycles", "ivfflat"),
        ],
    )
    def test_vector_index_exists(self, pg_connection, table_name, index_type):
        """Vector indexes should exist for key tables."""
        with pg_connection.cursor() as cur:
            cur.execute(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                AND tablename = %s
                AND indexdef LIKE %s
                """,
                (table_name, f"%{index_type}%"),
            )
            result = cur.fetchone()

        assert result is not None, f"No {index_type} index found for {table_name}"
        assert "vector_cosine_ops" in result[1], "Index should use cosine distance"


class TestRagTables:
    """Test RAG-specific tables exist."""

    def test_rag_document_chunks_exists(self, pg_connection):
        """rag_document_chunks table should exist."""
        with pg_connection.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'rag_document_chunks'
                )
                """
            )
            result = cur.fetchone()

        assert result[0] is True, "rag_document_chunks table not found"

    def test_rag_search_logs_exists(self, pg_connection):
        """rag_search_logs table should exist."""
        with pg_connection.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'rag_search_logs'
                )
                """
            )
            result = cur.fetchone()

        assert result[0] is True, "rag_search_logs table not found"


class TestVectorSearchFunctions:
    """Test vector search functions work correctly."""

    def test_rag_vector_search_function_exists(self, pg_connection):
        """rag_vector_search function should exist."""
        with pg_connection.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.routines
                    WHERE routine_schema = 'public'
                    AND routine_name = 'rag_vector_search'
                )
                """
            )
            result = cur.fetchone()

        assert result[0] is True, "rag_vector_search function not found"

    def test_rag_fulltext_search_function_exists(self, pg_connection):
        """rag_fulltext_search function should exist."""
        with pg_connection.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.routines
                    WHERE routine_schema = 'public'
                    AND routine_name = 'rag_fulltext_search'
                )
                """
            )
            result = cur.fetchone()

        assert result[0] is True, "rag_fulltext_search function not found"

    def test_vector_search_returns_results(self, pg_connection):
        """Vector search should work with a test document."""
        test_doc_id = f"test_pgvector_{uuid.uuid4().hex[:8]}"

        try:
            # Insert test document
            with pg_connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_document_chunks
                    (document_id, document_type, content, brand, embedding)
                    VALUES (%s, 'test_document', 'Test TRx prescription data', 'TestBrand',
                    ('[' || '0.1, 0.2, 0.15, 0.05, 0.3, 0.1, 0.08, 0.12, 0.05, 0.1' ||
                    repeat(', 0.0', 1526) || ']')::vector(1536))
                    RETURNING chunk_id
                    """,
                    (test_doc_id,),
                )
                chunk_id = cur.fetchone()[0]
                pg_connection.commit()

            # Search for it
            with pg_connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, similarity, source_table
                    FROM rag_vector_search(
                        ('[' || '0.11, 0.19, 0.16, 0.04, 0.29, 0.11, 0.07, 0.13, 0.06, 0.09' ||
                        repeat(', 0.0', 1526) || ']')::vector(1536),
                        5,
                        '{"brand": "TestBrand"}'::jsonb
                    )
                    """
                )
                results = cur.fetchall()

            assert len(results) >= 1, "Vector search should return at least 1 result"
            assert str(chunk_id) in [r[0] for r in results], "Test document should be in results"
            # Similarity should be high (> 0.9) for similar vectors
            assert results[0][1] > 0.9, f"Similarity too low: {results[0][1]}"

        finally:
            # Cleanup
            with pg_connection.cursor() as cur:
                cur.execute(
                    "DELETE FROM rag_document_chunks WHERE document_id = %s",
                    (test_doc_id,),
                )
                pg_connection.commit()

    def test_fulltext_search_returns_results(self, pg_connection):
        """Fulltext search should work with a test document."""
        test_doc_id = f"test_pgvector_ft_{uuid.uuid4().hex[:8]}"

        try:
            # Insert test document
            with pg_connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_document_chunks
                    (document_id, document_type, content, brand)
                    VALUES (%s, 'test_document',
                    'Remibrutinib prescription TRx volume analysis for Northeast region',
                    'Remibrutinib')
                    RETURNING chunk_id
                    """,
                    (test_doc_id,),
                )
                chunk_id = cur.fetchone()[0]
                pg_connection.commit()

            # Search for it
            with pg_connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, rank, source_table
                    FROM rag_fulltext_search('Remibrutinib TRx prescription', 5)
                    WHERE source_table = 'rag_document_chunks'
                    """
                )
                results = cur.fetchall()

            assert len(results) >= 1, "Fulltext search should return at least 1 result"
            assert str(chunk_id) in [r[0] for r in results], "Test document should be in results"

        finally:
            # Cleanup
            with pg_connection.cursor() as cur:
                cur.execute(
                    "DELETE FROM rag_document_chunks WHERE document_id = %s",
                    (test_doc_id,),
                )
                pg_connection.commit()


class TestSearchLogging:
    """Test search logging functionality."""

    def test_log_rag_search_function_exists(self, pg_connection):
        """log_rag_search function should exist."""
        with pg_connection.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.routines
                    WHERE routine_schema = 'public'
                    AND routine_name = 'log_rag_search'
                )
                """
            )
            result = cur.fetchone()

        assert result[0] is True, "log_rag_search function not found"

    def test_search_logging_works(self, pg_connection):
        """Search logging should work correctly."""
        test_query = f"test_logging_query_{uuid.uuid4().hex[:8]}"

        try:
            with pg_connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT log_rag_search(
                        %s,
                        NULL,
                        'test_user',
                        5, 3, 0, 8,
                        150.5,
                        50.2, 30.1, NULL, 70.2,
                        '{"vector": true, "fulltext": true}'::jsonb,
                        '[]'::jsonb,
                        '{"rrf_k": 60}'::jsonb,
                        '{"brands": ["TestBrand"]}'::jsonb
                    ) as log_id
                    """,
                    (test_query,),
                )
                log_id = cur.fetchone()[0]
                pg_connection.commit()

            # Verify log was created
            with pg_connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT query, total_latency_ms, vector_count, fused_count
                    FROM rag_search_logs WHERE log_id = %s
                    """,
                    (log_id,),
                )
                result = cur.fetchone()

            assert result is not None, "Search log not found"
            assert result[0] == test_query
            assert result[1] == 150.5
            assert result[2] == 5
            assert result[3] == 8

        finally:
            # Cleanup
            with pg_connection.cursor() as cur:
                cur.execute(
                    "DELETE FROM rag_search_logs WHERE query = %s",
                    (test_query,),
                )
                pg_connection.commit()


class TestVectorViews:
    """Test RAG analytics views exist."""

    def test_rag_slow_queries_view_exists(self, pg_connection):
        """rag_slow_queries view should exist."""
        with pg_connection.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.views
                    WHERE table_schema = 'public'
                    AND table_name = 'rag_slow_queries'
                )
                """
            )
            result = cur.fetchone()

        assert result[0] is True, "rag_slow_queries view not found"

    def test_rag_search_stats_view_exists(self, pg_connection):
        """rag_search_stats view should exist."""
        with pg_connection.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.views
                    WHERE table_schema = 'public'
                    AND table_name = 'rag_search_stats'
                )
                """
            )
            result = cur.fetchone()

        assert result[0] is True, "rag_search_stats view not found"
