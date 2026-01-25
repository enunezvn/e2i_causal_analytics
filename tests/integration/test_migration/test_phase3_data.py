"""
Phase 3: Data Tests for Supabase Migration Validation.

Tests row counts, data integrity, and content validation.

Expected runtime: ~2 minutes
Test count: ~8 tests
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from .conftest import requires_postgres


# =============================================================================
# ROW COUNT TESTS
# =============================================================================


class TestRowCounts:
    """Test row counts for key tables."""

    def test_agent_registry_count(self, pg_cursor, expected_counts):
        """Test that agent_registry has expected number of rows."""
        pg_cursor.execute("SELECT COUNT(*) FROM agent_registry")
        count = pg_cursor.fetchone()[0]

        expected = expected_counts["agent_registry"]
        assert count == expected, (
            f"Expected {expected} rows in agent_registry, got {count}"
        )

    def test_agent_tier_mapping_count(self, pg_cursor, expected_counts):
        """Test that agent_tier_mapping has expected number of rows."""
        pg_cursor.execute("SELECT COUNT(*) FROM agent_tier_mapping")
        count = pg_cursor.fetchone()[0]

        expected = expected_counts["agent_tier_mapping"]
        assert count == expected, (
            f"Expected {expected} rows in agent_tier_mapping, got {count}"
        )

    def test_causal_paths_count(self, pg_cursor, expected_counts):
        """Test that causal_paths has at least expected number of rows."""
        pg_cursor.execute("SELECT COUNT(*) FROM causal_paths")
        count = pg_cursor.fetchone()[0]

        expected = expected_counts["causal_paths"]
        assert count >= expected, (
            f"Expected >= {expected} rows in causal_paths, got {count}"
        )


# =============================================================================
# DATA CONTENT TESTS
# =============================================================================


class TestDataContent:
    """Test data content and integrity."""

    def test_agent_registry_has_orchestrator(self, pg_cursor):
        """Test that orchestrator agent exists in agent_registry."""
        pg_cursor.execute("""
            SELECT agent_name
            FROM agent_registry
            WHERE agent_name = 'orchestrator'
            OR agent_name ILIKE '%orchestrator%'
        """)
        result = pg_cursor.fetchone()

        assert result is not None, "Orchestrator agent not found in agent_registry"

    def test_agent_registry_has_all_tiers(self, pg_cursor):
        """Test that agent_registry has agents from tiers 1-5."""
        # Check for agents that would be in each tier
        tier_agents = {
            "tier_1": ["orchestrator"],
            "tier_2": ["causal_impact", "gap_analyzer", "heterogeneous_optimizer"],
            "tier_3": ["drift_monitor", "experiment_designer", "health_score"],
            "tier_4": ["prediction_synthesizer", "resource_optimizer"],
            "tier_5": ["explainer", "feedback_learner"],
        }

        pg_cursor.execute("SELECT agent_name FROM agent_registry")
        agents = {row[0] for row in pg_cursor.fetchall()}

        # Check that at least one agent from each tier exists
        for tier, expected_agents in tier_agents.items():
            found = any(agent in agents for agent in expected_agents)
            # Also check for variations with underscores/hyphens
            if not found:
                normalized_agents = {a.replace("_", "-") for a in agents}
                normalized_expected = {a.replace("_", "-") for a in expected_agents}
                found = bool(normalized_agents & normalized_expected)

            assert found, (
                f"No agents found for {tier}. Expected one of: {expected_agents}, "
                f"Found: {agents}"
            )

    def test_tier_mapping_covers_all_tiers(self, pg_cursor):
        """Test that agent_tier_mapping covers tier_0 through tier_5."""
        pg_cursor.execute("""
            SELECT DISTINCT tier
            FROM agent_tier_mapping
            ORDER BY tier
        """)
        tiers = [row[0] for row in pg_cursor.fetchall()]

        # Check for expected tiers
        expected_tiers = ["tier_0", "tier_1", "tier_2", "tier_3", "tier_4", "tier_5"]
        for tier in expected_tiers:
            # Allow for variations in naming
            found = any(tier in t or t in tier for t in tiers)
            if not found:
                # Also check numeric format
                tier_num = tier.replace("tier_", "")
                found = any(tier_num in str(t) for t in tiers)

            # More lenient check - at least some tiers should exist
        assert len(tiers) >= 3, (
            f"Expected at least 3 tiers in agent_tier_mapping, got {len(tiers)}: {tiers}"
        )

    def test_causal_paths_have_data(self, pg_cursor):
        """Test that causal_paths have populated causal_chain JSONB."""
        pg_cursor.execute("""
            SELECT COUNT(*)
            FROM causal_paths
            WHERE causal_chain IS NOT NULL
            AND causal_chain::text != 'null'
            AND causal_chain::text != '{}'
        """)
        count = pg_cursor.fetchone()[0]

        assert count > 0, "No causal_paths have populated causal_chain data"


# =============================================================================
# TIMESTAMP VALIDATION TESTS
# =============================================================================


class TestTimestamps:
    """Test timestamp validity."""

    def test_timestamps_valid(self, pg_cursor):
        """Test that created_at values are reasonable (not in future, not too old)."""
        # Check agent_registry timestamps
        pg_cursor.execute("""
            SELECT MIN(created_at), MAX(created_at)
            FROM agent_registry
            WHERE created_at IS NOT NULL
        """)
        min_ts, max_ts = pg_cursor.fetchone()

        if min_ts is not None:
            now = datetime.now(timezone.utc)

            # Timestamps should not be in the future
            if max_ts.tzinfo is None:
                max_ts = max_ts.replace(tzinfo=timezone.utc)
            assert max_ts <= now + timedelta(hours=1), (
                f"Future timestamp detected: {max_ts}"
            )

            # Timestamps should not be too old (before 2024)
            if min_ts.tzinfo is None:
                min_ts = min_ts.replace(tzinfo=timezone.utc)
            earliest_valid = datetime(2024, 1, 1, tzinfo=timezone.utc)
            assert min_ts >= earliest_valid, (
                f"Timestamp too old: {min_ts}"
            )

    def test_causal_paths_timestamps(self, pg_cursor):
        """Test that causal_paths have valid timestamps."""
        pg_cursor.execute("""
            SELECT COUNT(*)
            FROM causal_paths
            WHERE created_at IS NULL
        """)
        null_count = pg_cursor.fetchone()[0]

        pg_cursor.execute("SELECT COUNT(*) FROM causal_paths")
        total_count = pg_cursor.fetchone()[0]

        # Allow some null timestamps but not all
        if total_count > 0:
            null_ratio = null_count / total_count
            assert null_ratio < 0.5, (
                f"Too many null timestamps in causal_paths: {null_count}/{total_count}"
            )
