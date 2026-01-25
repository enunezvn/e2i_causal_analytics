"""
Phase 1: Connectivity Tests for Supabase Migration Validation.

Tests database connectivity, PostgreSQL version, and required extensions.

Expected runtime: ~1 minute
Test count: 5 tests
"""

from __future__ import annotations

import os
import re

import pytest
import requests

# =============================================================================
# DATABASE CONNECTION TESTS
# =============================================================================


class TestDatabaseConnectivity:
    """Test database connection and basic operations."""

    def test_database_connection(self, pg_connection):
        """Test that PostgreSQL connection succeeds."""
        assert pg_connection is not None
        assert not pg_connection.closed

        # Verify connection is functional
        with pg_connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result == (1,)

    def test_database_version(self, pg_connection):
        """Test that PostgreSQL version is >= 15.0."""
        with pg_connection.cursor() as cursor:
            cursor.execute("SELECT version()")
            version_string = cursor.fetchone()[0]

            # Extract version number (e.g., "PostgreSQL 15.8")
            match = re.search(r"PostgreSQL (\d+)\.(\d+)", version_string)
            assert match, f"Could not parse version from: {version_string}"

            major = int(match.group(1))
            minor = int(match.group(2))

            assert major >= 15, f"PostgreSQL version must be >= 15.0, got {major}.{minor}"


# =============================================================================
# EXTENSION TESTS
# =============================================================================


class TestPostgreSQLExtensions:
    """Test that required PostgreSQL extensions are installed."""

    def test_extension_uuid_ossp(self, pg_cursor):
        """Test that uuid-ossp extension is installed."""
        pg_cursor.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'uuid-ossp'"
        )
        result = pg_cursor.fetchone()
        assert result is not None, "uuid-ossp extension not installed"
        assert result[0] == "uuid-ossp"

    def test_extension_pgcrypto(self, pg_cursor):
        """Test that pgcrypto extension is installed."""
        pg_cursor.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'pgcrypto'"
        )
        result = pg_cursor.fetchone()
        assert result is not None, "pgcrypto extension not installed"
        assert result[0] == "pgcrypto"

    def test_all_required_extensions(self, pg_cursor, required_extensions):
        """Test that all required extensions are installed."""
        pg_cursor.execute("SELECT extname FROM pg_extension")
        installed = {row[0] for row in pg_cursor.fetchall()}

        missing = set(required_extensions) - installed
        assert not missing, f"Missing required extensions: {missing}"


# =============================================================================
# API CONNECTIVITY TESTS
# =============================================================================


class TestSupabaseAPIConnectivity:
    """Test Supabase API (Kong) connectivity."""

    def test_supabase_api_reachable(self, supabase_config):
        """Test that Supabase Kong API responds."""
        url = supabase_config.get("url")
        if not url:
            pytest.skip("SUPABASE_URL not configured")

        # Kong health endpoint
        # Try the REST API endpoint
        key = supabase_config.get("anon_key") or supabase_config.get("service_key")
        if not key:
            pytest.skip("No Supabase API key configured")

        try:
            response = requests.get(
                f"{url}/rest/v1/",
                headers={"apikey": key, "Authorization": f"Bearer {key}"},
                timeout=10,
            )
            # Kong/PostgREST typically returns 200 OK or specific status
            assert response.status_code in [200, 400, 401, 404], (
                f"Unexpected status code: {response.status_code}"
            )
        except requests.exceptions.ConnectionError as e:
            pytest.fail(f"Could not connect to Supabase API at {url}: {e}")
        except requests.exceptions.Timeout:
            pytest.fail(f"Timeout connecting to Supabase API at {url}")
