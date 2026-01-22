"""
Integration tests for RBAC (Role-Based Access Control) on API endpoints.

Tests verify that:
- Protected endpoints reject insufficient roles with 403
- Protected endpoints accept sufficient roles
- Testing mode bypasses authentication

Author: E2I Causal Analytics Team
"""

import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


# Mock user fixtures for different roles
VIEWER_USER = {
    "id": "viewer-user-id",
    "email": "viewer@e2i-analytics.com",
    "role": "authenticated",
    "aud": "authenticated",
    "app_metadata": {"role": "viewer"},
    "user_metadata": {"name": "Viewer User"},
}

ANALYST_USER = {
    "id": "analyst-user-id",
    "email": "analyst@e2i-analytics.com",
    "role": "authenticated",
    "aud": "authenticated",
    "app_metadata": {"role": "analyst"},
    "user_metadata": {"name": "Analyst User"},
}

OPERATOR_USER = {
    "id": "operator-user-id",
    "email": "operator@e2i-analytics.com",
    "role": "authenticated",
    "aud": "authenticated",
    "app_metadata": {"role": "operator"},
    "user_metadata": {"name": "Operator User"},
}

ADMIN_USER = {
    "id": "admin-user-id",
    "email": "admin@e2i-analytics.com",
    "role": "authenticated",
    "aud": "authenticated",
    "app_metadata": {"role": "admin"},
    "user_metadata": {"name": "Admin User"},
}


def create_mock_token_verifier(user_to_return):
    """Create a mock token verifier that returns the specified user."""
    async def mock_verify(token: str):
        if token == "invalid":
            return None
        return user_to_return
    return mock_verify


@pytest.fixture
def app_client():
    """Create a test client with mocked auth."""
    # Import here to avoid circular imports and allow patching
    from src.api.main import app
    return TestClient(app)


class TestAnalystEndpoints:
    """Test endpoints that require analyst role."""

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/causal/hierarchical/analyze", "post", {"treatment": "test", "outcome": "test", "data_source": "test"}),
        ("/causal/route", "post", {"query": "test query"}),
        ("/causal/pipeline/sequential", "post", {"stages": [], "treatment": "test", "outcome": "test"}),
        ("/causal/pipeline/parallel", "post", {"libraries": [], "treatment": "test", "outcome": "test"}),
        ("/causal/validate", "post", {"treatment": "test", "outcome": "test", "libraries": []}),
        ("/gaps/analyze", "post", {"query": "test", "brand": "Kisqali"}),
        ("/segments/analyze", "post", {"query": "test", "brand": "Kisqali"}),
    ])
    def test_viewer_denied_analyst_endpoints(self, app_client, endpoint, method, body):
        """Test that viewer role is denied access to analyst endpoints."""
        with patch("src.api.dependencies.auth.verify_supabase_token", create_mock_token_verifier(VIEWER_USER)):
            headers = {"Authorization": "Bearer valid-token"}
            if method == "post":
                response = app_client.post(endpoint, json=body, headers=headers)
            else:
                response = app_client.get(endpoint, headers=headers)

            assert response.status_code == 403, f"Expected 403 for {endpoint}, got {response.status_code}"
            assert "Analyst privileges required" in response.text

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/causal/hierarchical/analyze", "post", {"treatment": "test", "outcome": "test", "data_source": "test"}),
        ("/causal/route", "post", {"query": "test query"}),
    ])
    def test_analyst_allowed_analyst_endpoints(self, app_client, endpoint, method, body):
        """Test that analyst role is allowed access to analyst endpoints."""
        with patch("src.api.dependencies.auth.verify_supabase_token", create_mock_token_verifier(ANALYST_USER)):
            headers = {"Authorization": "Bearer valid-token"}
            if method == "post":
                response = app_client.post(endpoint, json=body, headers=headers)
            else:
                response = app_client.get(endpoint, headers=headers)

            # Should not get 401 or 403 - may get other errors due to missing data
            assert response.status_code not in [401, 403], f"Unexpected auth error for {endpoint}"


class TestOperatorEndpoints:
    """Test endpoints that require operator role."""

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/experiments/exp-123/randomize", "post", {"unit_ids": ["u1", "u2"]}),
        ("/experiments/exp-123/enroll", "post", {"unit_id": "u1", "attributes": {}}),
        ("/experiments/exp-123/interim-analysis", "post", {}),
        ("/feedback/learn", "post", {"query": "test"}),
        ("/feedback/process", "post", {"feedback_ids": ["f1"]}),
        ("/digital-twin/simulate", "post", {"intervention": "test", "brand": "Kisqali"}),
        ("/digital-twin/validate", "post", {"simulation_id": "sim-123", "actual_ate": 0.5}),
    ])
    def test_analyst_denied_operator_endpoints(self, app_client, endpoint, method, body):
        """Test that analyst role is denied access to operator endpoints."""
        with patch("src.api.dependencies.auth.verify_supabase_token", create_mock_token_verifier(ANALYST_USER)):
            headers = {"Authorization": "Bearer valid-token"}
            if method == "post":
                response = app_client.post(endpoint, json=body, headers=headers)
            else:
                response = app_client.get(endpoint, headers=headers)

            assert response.status_code == 403, f"Expected 403 for {endpoint}, got {response.status_code}"
            assert "Operator privileges required" in response.text

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/feedback/learn", "post", {"query": "test"}),
        ("/digital-twin/simulate", "post", {"intervention": "test", "brand": "Kisqali"}),
    ])
    def test_operator_allowed_operator_endpoints(self, app_client, endpoint, method, body):
        """Test that operator role is allowed access to operator endpoints."""
        with patch("src.api.dependencies.auth.verify_supabase_token", create_mock_token_verifier(OPERATOR_USER)):
            headers = {"Authorization": "Bearer valid-token"}
            if method == "post":
                response = app_client.post(endpoint, json=body, headers=headers)
            else:
                response = app_client.get(endpoint, headers=headers)

            # Should not get 401 or 403 - may get other errors due to missing data
            assert response.status_code not in [401, 403], f"Unexpected auth error for {endpoint}"


class TestAdminEndpoints:
    """Test endpoints that require admin role."""

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/monitoring/retraining/trigger/model-123", "post", {}),
        ("/monitoring/retraining/sweep", "post", {}),
        ("/monitoring/retraining/job-123/rollback", "post", {}),
    ])
    def test_operator_denied_admin_endpoints(self, app_client, endpoint, method, body):
        """Test that operator role is denied access to admin endpoints."""
        with patch("src.api.dependencies.auth.verify_supabase_token", create_mock_token_verifier(OPERATOR_USER)):
            headers = {"Authorization": "Bearer valid-token"}
            if method == "post":
                response = app_client.post(endpoint, json=body, headers=headers)
            else:
                response = app_client.get(endpoint, headers=headers)

            assert response.status_code == 403, f"Expected 403 for {endpoint}, got {response.status_code}"
            assert "Admin privileges required" in response.text

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/monitoring/retraining/trigger/model-123", "post", {}),
        ("/monitoring/retraining/sweep", "post", {}),
    ])
    def test_admin_allowed_admin_endpoints(self, app_client, endpoint, method, body):
        """Test that admin role is allowed access to admin endpoints."""
        with patch("src.api.dependencies.auth.verify_supabase_token", create_mock_token_verifier(ADMIN_USER)):
            headers = {"Authorization": "Bearer valid-token"}
            if method == "post":
                response = app_client.post(endpoint, json=body, headers=headers)
            else:
                response = app_client.get(endpoint, headers=headers)

            # Should not get 401 or 403 - may get other errors due to missing data
            assert response.status_code not in [401, 403], f"Unexpected auth error for {endpoint}"


class TestUnauthenticatedAccess:
    """Test endpoints reject unauthenticated requests."""

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/causal/hierarchical/analyze", "post", {"treatment": "test", "outcome": "test"}),
        ("/experiments/exp-123/randomize", "post", {"unit_ids": []}),
        ("/monitoring/retraining/trigger/model-123", "post", {}),
    ])
    def test_no_token_returns_401(self, app_client, endpoint, method, body):
        """Test that requests without token return 401."""
        # No Authorization header
        if method == "post":
            response = app_client.post(endpoint, json=body)
        else:
            response = app_client.get(endpoint)

        assert response.status_code == 401, f"Expected 401 for {endpoint}, got {response.status_code}"

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/causal/hierarchical/analyze", "post", {"treatment": "test", "outcome": "test"}),
    ])
    def test_invalid_token_returns_401(self, app_client, endpoint, method, body):
        """Test that requests with invalid token return 401."""
        with patch("src.api.dependencies.auth.verify_supabase_token", create_mock_token_verifier(None)):
            headers = {"Authorization": "Bearer invalid"}
            if method == "post":
                response = app_client.post(endpoint, json=body, headers=headers)
            else:
                response = app_client.get(endpoint, headers=headers)

            assert response.status_code == 401, f"Expected 401 for {endpoint}, got {response.status_code}"


class TestRoleHierarchy:
    """Test that higher roles have access to lower-role endpoints."""

    def test_admin_has_operator_access(self, app_client):
        """Test that admin can access operator endpoints."""
        with patch("src.api.dependencies.auth.verify_supabase_token", create_mock_token_verifier(ADMIN_USER)):
            headers = {"Authorization": "Bearer valid-token"}
            response = app_client.post(
                "/feedback/learn",
                json={"query": "test"},
                headers=headers
            )
            # Should not get 403
            assert response.status_code != 403

    def test_admin_has_analyst_access(self, app_client):
        """Test that admin can access analyst endpoints."""
        with patch("src.api.dependencies.auth.verify_supabase_token", create_mock_token_verifier(ADMIN_USER)):
            headers = {"Authorization": "Bearer valid-token"}
            response = app_client.post(
                "/causal/route",
                json={"query": "test query"},
                headers=headers
            )
            # Should not get 403
            assert response.status_code != 403

    def test_operator_has_analyst_access(self, app_client):
        """Test that operator can access analyst endpoints."""
        with patch("src.api.dependencies.auth.verify_supabase_token", create_mock_token_verifier(OPERATOR_USER)):
            headers = {"Authorization": "Bearer valid-token"}
            response = app_client.post(
                "/causal/route",
                json={"query": "test query"},
                headers=headers
            )
            # Should not get 403
            assert response.status_code != 403
