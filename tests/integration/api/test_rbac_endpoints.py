"""
Integration tests for RBAC (Role-Based Access Control) on API endpoints.

Tests verify that:
- Protected endpoints reject insufficient roles with 403
- Protected endpoints accept sufficient roles
- Unauthenticated requests return 401

Uses FastAPI's dependency_overrides pattern to inject mock users.

Author: E2I Causal Analytics Team
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.dependencies.auth import (
    require_auth,
    require_viewer,
    require_analyst,
    require_operator,
    require_admin,
    AuthError,
)


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


def make_user_dependency(user_to_return):
    """Create a dependency that returns a specific user."""
    async def override():
        return user_to_return
    return override


def make_auth_failure_dependency():
    """Create a dependency that simulates authentication failure."""
    async def override():
        raise AuthError("Invalid or expired token")
    return override


@pytest.fixture
def app_client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Clean up dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


class TestAnalystEndpoints:
    """Test endpoints that require analyst role."""

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/api/causal/hierarchical/analyze", "post", {"treatment": "test", "outcome": "test", "data_source": "test"}),
        ("/api/causal/route", "post", {"query": "test query"}),
        ("/api/causal/pipeline/sequential", "post", {"stages": [], "treatment": "test", "outcome": "test"}),
        ("/api/causal/pipeline/parallel", "post", {"libraries": [], "treatment": "test", "outcome": "test"}),
        ("/api/causal/validate", "post", {"treatment": "test", "outcome": "test", "libraries": []}),
        ("/api/gaps/analyze", "post", {"query": "test", "brand": "Kisqali"}),
        ("/api/segments/analyze", "post", {"query": "test", "brand": "Kisqali"}),
    ])
    def test_viewer_denied_analyst_endpoints(self, app_client, endpoint, method, body):
        """Test that viewer role is denied access to analyst endpoints."""
        # Override require_analyst to return viewer user (should fail role check)
        # We need to override require_auth to return viewer, then require_analyst will check role
        app.dependency_overrides[require_auth] = make_user_dependency(VIEWER_USER)

        if method == "post":
            response = app_client.post(endpoint, json=body)
        else:
            response = app_client.get(endpoint)

        assert response.status_code == 403, f"Expected 403 for {endpoint}, got {response.status_code}: {response.text}"
        assert "Analyst privileges required" in response.text

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/api/causal/hierarchical/analyze", "post", {"treatment": "test", "outcome": "test", "data_source": "test"}),
        ("/api/causal/route", "post", {"query": "test query"}),
    ])
    def test_analyst_allowed_analyst_endpoints(self, app_client, endpoint, method, body):
        """Test that analyst role is allowed access to analyst endpoints."""
        # Override require_auth to return analyst user
        app.dependency_overrides[require_auth] = make_user_dependency(ANALYST_USER)

        if method == "post":
            response = app_client.post(endpoint, json=body)
        else:
            response = app_client.get(endpoint)

        # Should not get 401 or 403 - may get other errors due to missing data
        assert response.status_code not in [401, 403], f"Unexpected auth error for {endpoint}: {response.text}"


class TestOperatorEndpoints:
    """Test endpoints that require operator role."""

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/api/experiments/exp-123/randomize", "post", {"unit_ids": ["u1", "u2"]}),
        ("/api/experiments/exp-123/enroll", "post", {"unit_id": "u1", "attributes": {}}),
        ("/api/experiments/exp-123/interim-analysis", "post", {}),
        ("/api/feedback/learn", "post", {"query": "test"}),
        ("/api/feedback/process", "post", {"feedback_ids": ["f1"]}),
        ("/api/digital-twin/simulate", "post", {"intervention": "test", "brand": "Kisqali"}),
        ("/api/digital-twin/validate", "post", {"simulation_id": "sim-123", "actual_ate": 0.5}),
    ])
    def test_analyst_denied_operator_endpoints(self, app_client, endpoint, method, body):
        """Test that analyst role is denied access to operator endpoints."""
        app.dependency_overrides[require_auth] = make_user_dependency(ANALYST_USER)

        if method == "post":
            response = app_client.post(endpoint, json=body)
        else:
            response = app_client.get(endpoint)

        assert response.status_code == 403, f"Expected 403 for {endpoint}, got {response.status_code}: {response.text}"
        assert "Operator privileges required" in response.text

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/api/feedback/learn", "post", {"query": "test"}),
        ("/api/digital-twin/simulate", "post", {"intervention": "test", "brand": "Kisqali"}),
    ])
    def test_operator_allowed_operator_endpoints(self, app_client, endpoint, method, body):
        """Test that operator role is allowed access to operator endpoints."""
        app.dependency_overrides[require_auth] = make_user_dependency(OPERATOR_USER)

        if method == "post":
            response = app_client.post(endpoint, json=body)
        else:
            response = app_client.get(endpoint)

        # Should not get 401 or 403 - may get other errors due to missing data
        assert response.status_code not in [401, 403], f"Unexpected auth error for {endpoint}: {response.text}"


class TestAdminEndpoints:
    """Test endpoints that require admin role."""

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/api/monitoring/retraining/trigger/model-123", "post", {}),
        ("/api/monitoring/retraining/sweep", "post", {}),
        ("/api/monitoring/retraining/job-123/rollback", "post", {}),
    ])
    def test_operator_denied_admin_endpoints(self, app_client, endpoint, method, body):
        """Test that operator role is denied access to admin endpoints."""
        app.dependency_overrides[require_auth] = make_user_dependency(OPERATOR_USER)

        if method == "post":
            response = app_client.post(endpoint, json=body)
        else:
            response = app_client.get(endpoint)

        assert response.status_code == 403, f"Expected 403 for {endpoint}, got {response.status_code}: {response.text}"
        assert "Admin privileges required" in response.text

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/api/monitoring/retraining/trigger/model-123", "post", {}),
        ("/api/monitoring/retraining/sweep", "post", {}),
    ])
    def test_admin_allowed_admin_endpoints(self, app_client, endpoint, method, body):
        """Test that admin role is allowed access to admin endpoints."""
        app.dependency_overrides[require_auth] = make_user_dependency(ADMIN_USER)

        if method == "post":
            response = app_client.post(endpoint, json=body)
        else:
            response = app_client.get(endpoint)

        # Should not get 401 or 403 - may get other errors due to missing data
        assert response.status_code not in [401, 403], f"Unexpected auth error for {endpoint}: {response.text}"


class TestUnauthenticatedAccess:
    """Test endpoints reject unauthenticated requests."""

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/api/causal/hierarchical/analyze", "post", {"treatment": "test", "outcome": "test"}),
        ("/api/experiments/exp-123/randomize", "post", {"unit_ids": []}),
        ("/api/monitoring/retraining/trigger/model-123", "post", {}),
    ])
    def test_no_auth_returns_401(self, app_client, endpoint, method, body):
        """Test that requests without valid auth return 401."""
        # Override require_auth to raise AuthError (simulating no token)
        app.dependency_overrides[require_auth] = make_auth_failure_dependency()

        if method == "post":
            response = app_client.post(endpoint, json=body)
        else:
            response = app_client.get(endpoint)

        assert response.status_code == 401, f"Expected 401 for {endpoint}, got {response.status_code}: {response.text}"


class TestRoleHierarchy:
    """Test that higher roles have access to lower-role endpoints."""

    def test_admin_has_operator_access(self, app_client):
        """Test that admin can access operator endpoints."""
        app.dependency_overrides[require_auth] = make_user_dependency(ADMIN_USER)

        response = app_client.post(
            "/api/feedback/learn",
            json={"query": "test"},
        )
        # Should not get 403
        assert response.status_code != 403, f"Admin should have operator access: {response.text}"

    def test_admin_has_analyst_access(self, app_client):
        """Test that admin can access analyst endpoints."""
        app.dependency_overrides[require_auth] = make_user_dependency(ADMIN_USER)

        response = app_client.post(
            "/api/causal/route",
            json={"query": "test query"},
        )
        # Should not get 403
        assert response.status_code != 403, f"Admin should have analyst access: {response.text}"

    def test_operator_has_analyst_access(self, app_client):
        """Test that operator can access analyst endpoints."""
        app.dependency_overrides[require_auth] = make_user_dependency(OPERATOR_USER)

        response = app_client.post(
            "/api/causal/route",
            json={"query": "test query"},
        )
        # Should not get 403
        assert response.status_code != 403, f"Operator should have analyst access: {response.text}"
