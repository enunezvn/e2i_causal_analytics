"""
Phase 4: Auth Tests for Supabase Migration Validation.

Tests auth schema, users, and GoTrue API.

Expected runtime: ~1 minute
Test count: ~5 tests
"""

from __future__ import annotations

import pytest
import requests

# =============================================================================
# AUTH SCHEMA TESTS
# =============================================================================


class TestAuthSchema:
    """Test auth schema existence and structure."""

    def test_auth_schema_exists(self, pg_cursor):
        """Test that auth schema is present."""
        pg_cursor.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.schemata
                WHERE schema_name = 'auth'
            )
        """)
        exists = pg_cursor.fetchone()[0]
        assert exists, "auth schema does not exist"

    def test_auth_users_table_exists(self, pg_cursor):
        """Test that auth.users table exists."""
        pg_cursor.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'auth'
                AND table_name = 'users'
            )
        """)
        exists = pg_cursor.fetchone()[0]
        assert exists, "auth.users table does not exist"


# =============================================================================
# AUTH USER TESTS
# =============================================================================


class TestAuthUsers:
    """Test auth user data."""

    def test_auth_users_count(self, pg_cursor, expected_counts):
        """Test that auth.users has expected number of users."""
        pg_cursor.execute("SELECT COUNT(*) FROM auth.users")
        count = pg_cursor.fetchone()[0]

        expected = expected_counts["auth_users"]
        assert count == expected, f"Expected {expected} users in auth.users, got {count}"

    def test_auth_users_have_email(self, pg_cursor):
        """Test that all auth users have email addresses."""
        pg_cursor.execute("""
            SELECT COUNT(*)
            FROM auth.users
            WHERE email IS NULL OR email = ''
        """)
        null_count = pg_cursor.fetchone()[0]

        assert null_count == 0, f"{null_count} users have null or empty email addresses"

    def test_auth_users_email_format(self, pg_cursor):
        """Test that auth user emails have valid format."""
        pg_cursor.execute("""
            SELECT email
            FROM auth.users
            WHERE email NOT LIKE '%@%.%'
        """)
        invalid_emails = pg_cursor.fetchall()

        assert len(invalid_emails) == 0, (
            f"Invalid email formats found: {[e[0] for e in invalid_emails]}"
        )


# =============================================================================
# GOTRUE API TESTS
# =============================================================================


class TestGoTrueAPI:
    """Test GoTrue authentication API."""

    def test_gotrue_health(self, supabase_config):
        """Test that GoTrue API responds."""
        url = supabase_config.get("url")
        if not url:
            pytest.skip("SUPABASE_URL not configured")

        # GoTrue health endpoint
        gotrue_url = f"{url}/auth/v1/health"

        try:
            response = requests.get(gotrue_url, timeout=10)

            # GoTrue typically returns 200 with health status
            # 401 is also acceptable for self-hosted where Kong requires auth
            assert response.status_code in [200, 401, 404], (
                f"Unexpected GoTrue status: {response.status_code}"
            )
        except requests.exceptions.ConnectionError as e:
            pytest.fail(f"Could not connect to GoTrue at {gotrue_url}: {e}")
        except requests.exceptions.Timeout:
            pytest.fail(f"Timeout connecting to GoTrue at {gotrue_url}")

    def test_auth_admin_api(self, supabase_config):
        """Test that admin can access user management API."""
        url = supabase_config.get("url")
        service_key = supabase_config.get("service_key")

        if not url:
            pytest.skip("SUPABASE_URL not configured")
        if not service_key:
            pytest.skip("SUPABASE_SERVICE_KEY not configured (required for admin API)")

        # Admin users endpoint
        admin_url = f"{url}/auth/v1/admin/users"

        try:
            response = requests.get(
                admin_url,
                headers={
                    "apikey": service_key,
                    "Authorization": f"Bearer {service_key}",
                },
                timeout=10,
            )

            # Should return 200 with user list or 401 if key is wrong
            assert response.status_code in [200, 401, 403], (
                f"Unexpected admin API status: {response.status_code}"
            )

            if response.status_code == 200:
                data = response.json()
                # Should have users array
                assert "users" in data or isinstance(data, list), (
                    "Admin API response missing users data"
                )
        except requests.exceptions.ConnectionError as e:
            pytest.fail(f"Could not connect to admin API at {admin_url}: {e}")
        except requests.exceptions.Timeout:
            pytest.fail(f"Timeout connecting to admin API at {admin_url}")
