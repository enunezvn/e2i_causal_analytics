"""
Phase 5: API Tests for Supabase Migration Validation.

Tests PostgREST, E2I API health, and key endpoints.

Expected runtime: ~2 minutes
Test count: ~5 tests
"""

from __future__ import annotations

import os

import pytest
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

# E2I API configuration
E2I_API_URL = os.getenv("E2I_API_URL", "http://localhost:8000")


# =============================================================================
# POSTGREST API TESTS
# =============================================================================


class TestPostgRESTAPI:
    """Test PostgREST (Supabase REST API)."""

    def test_postgrest_health(self, supabase_config):
        """Test that PostgREST API responds."""
        url = supabase_config.get("url")
        if not url:
            pytest.skip("SUPABASE_URL not configured")

        key = supabase_config.get("anon_key") or supabase_config.get("service_key")
        if not key:
            pytest.skip("No Supabase API key configured")

        # REST API base endpoint
        rest_url = f"{url}/rest/v1/"

        try:
            response = requests.get(
                rest_url,
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                },
                timeout=10,
            )

            # PostgREST returns 200 with OpenAPI spec or table list
            # 401 is also acceptable for self-hosted where Kong requires auth
            assert response.status_code in [200, 400, 401], (
                f"Unexpected PostgREST status: {response.status_code}"
            )
        except requests.exceptions.ConnectionError as e:
            pytest.fail(f"Could not connect to PostgREST at {rest_url}: {e}")
        except requests.exceptions.Timeout:
            pytest.fail(f"Timeout connecting to PostgREST at {rest_url}")

    def test_query_agent_registry_via_api(self, supabase_client):
        """Test that agent_registry can be queried via Supabase client."""
        try:
            result = supabase_client.table("agent_registry").select("*").limit(5).execute()

            # Should return data array
            assert hasattr(result, "data"), "Response missing data attribute"
            assert isinstance(result.data, list), "Data is not a list"

            # Should have some data
            assert len(result.data) > 0, "No data returned from agent_registry"

            # Each row should have expected fields
            if result.data:
                first_row = result.data[0]
                assert "id" in first_row or "agent_name" in first_row, (
                    f"Unexpected row structure: {first_row.keys()}"
                )
        except Exception as e:
            error_str = str(e).lower()
            # May fail with RLS policies or authentication - this is OK for self-hosted
            if any(
                keyword in error_str
                for keyword in ["policy", "permission", "authentication", "credentials", "401"]
            ):
                pytest.skip(f"Access blocked (RLS/auth): {e}")
            raise


# =============================================================================
# E2I API TESTS
# =============================================================================


class TestE2IAPI:
    """Test E2I FastAPI application endpoints."""

    def test_e2i_api_health(self):
        """Test that E2I FastAPI health endpoint responds."""
        health_url = f"{E2I_API_URL}/health"

        try:
            response = requests.get(health_url, timeout=10)

            # Health endpoint should return 200
            assert response.status_code == 200, (
                f"E2I API health check failed with status: {response.status_code}"
            )

            # Should return JSON with status
            data = response.json()
            assert "status" in data or "healthy" in data or "ok" in str(data).lower(), (
                f"Unexpected health response: {data}"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip(f"E2I API not available at {E2I_API_URL}")
        except requests.exceptions.Timeout:
            pytest.fail(f"Timeout connecting to E2I API at {health_url}")

    def test_e2i_kpi_endpoint(self):
        """Test that /api/kpis endpoint returns data."""
        kpi_url = f"{E2I_API_URL}/api/kpis"

        try:
            response = requests.get(kpi_url, timeout=10)

            # Should return 200 or 401 (if auth required)
            assert response.status_code in [200, 401, 403], (
                f"Unexpected KPI endpoint status: {response.status_code}"
            )

            if response.status_code == 200:
                data = response.json()
                # Should return array or object with KPI data
                assert data is not None, "KPI endpoint returned null"
        except requests.exceptions.ConnectionError:
            pytest.skip(f"E2I API not available at {E2I_API_URL}")
        except requests.exceptions.Timeout:
            pytest.fail(f"Timeout connecting to KPI endpoint at {kpi_url}")

    def test_copilotkit_endpoint(self):
        """Test that /api/copilotkit endpoint responds."""
        copilotkit_url = f"{E2I_API_URL}/api/copilotkit"

        try:
            # CopilotKit endpoint typically accepts POST
            response = requests.post(
                copilotkit_url,
                json={},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            # Should return response (even if error due to missing data)
            assert response.status_code in [200, 400, 401, 403, 422, 405], (
                f"Unexpected CopilotKit endpoint status: {response.status_code}"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip(f"E2I API not available at {E2I_API_URL}")
        except requests.exceptions.Timeout:
            pytest.fail(f"Timeout connecting to CopilotKit endpoint at {copilotkit_url}")


# =============================================================================
# API DOCUMENTATION TESTS
# =============================================================================


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_docs_available(self):
        """Test that OpenAPI documentation is accessible."""
        docs_url = f"{E2I_API_URL}/docs"

        try:
            response = requests.get(docs_url, timeout=10)

            # Swagger UI should be available (401 acceptable if auth required)
            assert response.status_code in [200, 401], (
                f"OpenAPI docs not available: {response.status_code}"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip(f"E2I API not available at {E2I_API_URL}")
        except requests.exceptions.Timeout:
            pytest.fail(f"Timeout connecting to docs at {docs_url}")

    def test_openapi_json_available(self):
        """Test that OpenAPI JSON spec is accessible."""
        openapi_url = f"{E2I_API_URL}/openapi.json"

        try:
            response = requests.get(openapi_url, timeout=10)

            # OpenAPI JSON should be available (401 acceptable if auth required)
            assert response.status_code in [200, 401], (
                f"OpenAPI JSON not available: {response.status_code}"
            )

            if response.status_code == 200:
                # Should be valid JSON
                data = response.json()
                assert "openapi" in data or "swagger" in data, "Response is not OpenAPI spec"
        except requests.exceptions.ConnectionError:
            pytest.skip(f"E2I API not available at {E2I_API_URL}")
        except requests.exceptions.Timeout:
            pytest.fail(f"Timeout connecting to OpenAPI JSON at {openapi_url}")
