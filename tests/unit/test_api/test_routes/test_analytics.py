"""Unit tests for the analytics dashboard route.

Focus: C23 (error-handling) honest-degradation fix.

The ``GET /api/analytics/dashboard`` endpoint is intentionally PUBLIC
(owner decision: "public for dashboard widgets"). This test suite verifies
that when the underlying audit-metrics FETCH fails, the endpoint surfaces an
HONEST degraded signal instead of silently returning a fully-zeroed dashboard
that a client would render as if it were real measurements.

The endpoint must STAY public; only the error-handling on a fetch failure
changes.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def app():
    """Create a FastAPI app with the analytics router mounted."""
    from src.api.routes.analytics import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class _FakeDB:
    """Stand-in Supabase client; presence only is required for these tests."""


# =============================================================================
# HONEST-DEGRADATION TESTS (C23)
# =============================================================================


class TestDashboardFetchFailureIsHonest:
    """A metrics-fetch failure must NOT be presented as real zeroed data."""

    async def _failing_fetch(self, *args, **kwargs):
        # Mirrors the shape returned by _fetch_audit_metrics on error.
        return {"success": False, "data": [], "error": "connection reset by peer"}

    def test_fetch_failure_does_not_return_zeroed_200(self, client):
        """On a fetch failure the endpoint must surface a degraded signal.

        Before the fix this returned HTTP 200 with a fully-zeroed dashboard
        (total_queries=0, success_rate=0.0, all latencies 0.0) which a client
        renders as real measurements. The honest behavior is to fail the
        request (HTTP 503) so the fabricated zeros are never presented as real.
        """
        with (
            patch(
                "src.api.routes.analytics._get_supabase_client",
                return_value=_FakeDB(),
            ),
            patch(
                "src.api.routes.analytics._fetch_audit_metrics",
                side_effect=self._failing_fetch,
            ),
        ):
            response = client.get("/analytics/dashboard?period=7d")

        # The response must NOT be a 200 carrying zeros-as-real.
        if response.status_code == 200:
            body = response.json()
            summary = body.get("summary", {})
            zeroed = (
                summary.get("total_queries") == 0
                and summary.get("successful_queries") == 0
                and summary.get("success_rate") == 0.0
                and summary.get("avg_latency_ms") == 0.0
            )
            assert not zeroed, (
                "Fetch failure was swallowed and a fully-zeroed dashboard was "
                "returned as a 200 OK; fabricated zeros are presented as real "
                "measurements."
            )
        else:
            assert response.status_code == 503

    def test_fetch_failure_returns_503(self, client):
        """The chosen honest signal for a fetch failure is HTTP 503.

        This mirrors the existing 503 returned when the DB client itself is
        unavailable, and drives the frontend's existing error-render path.
        """
        with (
            patch(
                "src.api.routes.analytics._get_supabase_client",
                return_value=_FakeDB(),
            ),
            patch(
                "src.api.routes.analytics._fetch_audit_metrics",
                side_effect=self._failing_fetch,
            ),
        ):
            response = client.get("/analytics/dashboard?period=7d")

        assert response.status_code == 503
        assert "detail" in response.json()


class TestDashboardStaysPublicAndWorks:
    """The endpoint stays public and returns real data on the happy path."""

    async def _ok_fetch(self, *args, **kwargs):
        return {
            "success": True,
            "data": [
                {
                    "agent_name": "orchestrator",
                    "agent_tier": 1,
                    "duration_ms": 120.0,
                    "validation_passed": True,
                    "confidence_score": 0.9,
                    "created_at": "2026-06-01T12:00:00Z",
                    "action_type": "route",
                }
            ],
        }

    def test_happy_path_no_auth_returns_200(self, client):
        """No Authorization header (public) + successful fetch => 200 with data."""
        with (
            patch(
                "src.api.routes.analytics._get_supabase_client",
                return_value=_FakeDB(),
            ),
            patch(
                "src.api.routes.analytics._fetch_audit_metrics",
                side_effect=self._ok_fetch,
            ),
        ):
            response = client.get("/analytics/dashboard?period=7d")

        assert response.status_code == 200
        body = response.json()
        assert body["summary"]["total_queries"] == 1
        assert body["summary"]["successful_queries"] == 1
        assert len(body["agent_metrics"]) == 1

    def test_client_unavailable_still_503(self, client):
        """Pre-existing behavior: DB client unavailable => 503 (unchanged)."""
        with patch(
            "src.api.routes.analytics._get_supabase_client",
            return_value=None,
        ):
            response = client.get("/analytics/dashboard?period=7d")

        assert response.status_code == 503
