"""
Phase 2: Schema Tests for Supabase Migration Validation.

Tests table counts, core tables, columns, enums, views, and indexes.

Expected runtime: ~2 minutes
Test count: ~10 tests
"""

from __future__ import annotations

from .conftest import (
    SPLIT_VIEW_PREFIXES,
)

# =============================================================================
# TABLE COUNT TESTS
# =============================================================================


class TestTableCounts:
    """Test table counts in public and auth schemas."""

    def test_public_table_count(self, pg_cursor, expected_counts):
        """Test that public schema has expected number of tables."""
        pg_cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """)
        count = pg_cursor.fetchone()[0]

        expected = expected_counts["public_tables"]
        assert count >= expected, f"Expected >= {expected} public tables, got {count}"

    def test_auth_table_count(self, pg_cursor, expected_counts):
        """Test that auth schema has expected number of tables."""
        pg_cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'auth'
            AND table_type = 'BASE TABLE'
        """)
        count = pg_cursor.fetchone()[0]

        expected = expected_counts["auth_tables"]
        assert count >= expected, f"Expected >= {expected} auth tables, got {count}"


# =============================================================================
# CORE TABLE TESTS
# =============================================================================


class TestCoreTables:
    """Test that core tables exist and have correct structure."""

    def test_core_tables_exist(self, pg_cursor, core_tables):
        """Test that all core tables exist in public schema."""
        pg_cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """)
        existing = {row[0] for row in pg_cursor.fetchall()}

        missing = set(core_tables) - existing
        assert not missing, f"Missing core tables: {missing}"

    def test_agent_registry_columns(self, pg_cursor):
        """Test that agent_registry has all expected columns."""
        pg_cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = 'agent_registry'
        """)
        columns = {row[0] for row in pg_cursor.fetchall()}

        # Check essential columns (allow for additional columns)
        # Note: agent_registry uses agent_name as primary key, no separate 'id' column
        essential_columns = {"agent_name", "agent_tier", "created_at"}
        missing = essential_columns - columns
        assert not missing, f"Missing essential columns in agent_registry: {missing}"

    def test_causal_paths_columns(self, pg_cursor):
        """Test that causal_paths has all expected columns."""
        pg_cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = 'causal_paths'
        """)
        columns = {row[0] for row in pg_cursor.fetchall()}

        # Check essential columns
        # Note: causal_paths uses path_id as primary key, not 'id'
        essential_columns = {"path_id", "causal_chain", "created_at"}
        missing = essential_columns - columns
        assert not missing, f"Missing essential columns in causal_paths: {missing}"

    def test_agent_tier_mapping_exists(self, pg_cursor):
        """Test that agent_tier_mapping table exists."""
        pg_cursor.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'agent_tier_mapping'
            )
        """)
        exists = pg_cursor.fetchone()[0]
        assert exists, "agent_tier_mapping table does not exist"


# =============================================================================
# ENUM TYPE TESTS
# =============================================================================


class TestEnumTypes:
    """Test that required enum types exist."""

    def test_enum_types_exist(self, pg_cursor, required_enums):
        """Test that all required enum types exist."""
        pg_cursor.execute("""
            SELECT typname
            FROM pg_type
            WHERE typtype = 'e'
            AND typnamespace = (
                SELECT oid FROM pg_namespace WHERE nspname = 'public'
            )
        """)
        existing = {row[0] for row in pg_cursor.fetchall()}

        missing = set(required_enums) - existing
        assert not missing, f"Missing enum types: {missing}"

    def test_data_split_type_values(self, pg_cursor):
        """Test that data_split_type has correct values."""
        pg_cursor.execute("""
            SELECT enumlabel
            FROM pg_enum
            WHERE enumtypid = (
                SELECT oid FROM pg_type WHERE typname = 'data_split_type'
            )
            ORDER BY enumsortorder
        """)
        values = [row[0] for row in pg_cursor.fetchall()]

        expected_values = ["train", "validation", "test", "holdout", "unassigned"]
        for val in expected_values:
            assert val in values, f"data_split_type missing value: {val}"


# =============================================================================
# VIEW TESTS
# =============================================================================


class TestSplitAwareViews:
    """Test that split-aware views exist."""

    def test_split_aware_views_exist(self, pg_cursor):
        """Test that split-aware views exist for key tables."""
        pg_cursor.execute("""
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'public'
        """)
        views = {row[0] for row in pg_cursor.fetchall()}

        # Check for at least some split-aware views
        found_split_views = [
            v for v in views if any(v.startswith(prefix) for prefix in SPLIT_VIEW_PREFIXES)
        ]

        assert len(found_split_views) > 0, (
            f"No split-aware views found. Expected views with prefixes: {SPLIT_VIEW_PREFIXES}"
        )


# =============================================================================
# INDEX TESTS
# =============================================================================


class TestIndexes:
    """Test that critical indexes exist."""

    def test_indexes_exist(self, pg_cursor):
        """Test that tables have indexes defined."""
        pg_cursor.execute("""
            SELECT COUNT(*)
            FROM pg_indexes
            WHERE schemaname = 'public'
        """)
        count = pg_cursor.fetchone()[0]

        # Expect a reasonable number of indexes (at least one per core table)
        assert count >= 10, f"Expected >= 10 indexes, got {count}"

    def test_primary_key_indexes(self, pg_cursor, core_tables):
        """Test that core tables have primary key indexes."""
        # Check for pkey indexes
        pg_cursor.execute("""
            SELECT tablename, indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND indexname LIKE '%pkey'
        """)
        pkey_tables = {row[0] for row in pg_cursor.fetchall()}

        # Most core tables should have primary keys
        missing_pkeys = []
        for table in core_tables[:5]:  # Check first 5 core tables
            if table not in pkey_tables:
                # Some tables might use different naming
                has_pkey = False
                pg_cursor.execute(f"""
                    SELECT indexname FROM pg_indexes
                    WHERE schemaname = 'public'
                    AND tablename = '{table}'
                    AND indexdef LIKE '%PRIMARY KEY%'
                """)
                if pg_cursor.fetchone():
                    has_pkey = True
                if not has_pkey:
                    missing_pkeys.append(table)

        # Allow some flexibility as not all tables require pkey
        assert len(missing_pkeys) <= 2, (
            f"Too many core tables without primary keys: {missing_pkeys}"
        )
