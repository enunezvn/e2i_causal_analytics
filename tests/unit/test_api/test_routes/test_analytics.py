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


class TestLatencyHonestyNullNotZero:
    """avg_latency_ms must be null (UNMEASURED), not a fabricated 0.0, when no
    audit entry in the window carried a real duration_ms.

    Regression target: agent graphs that only emitted a genesis workflow_start
    entry (no duration_ms) left the latency list empty, and the endpoint
    reported avg_latency_ms = 0.0, which the dashboard rendered as a misleading
    "0ms / instant".
    """

    async def _untimed_fetch(self, *args, **kwargs):
        # Two real entries (queries happened) but NEITHER carries a duration_ms
        # -> latency is genuinely unmeasured.
        return {
            "success": True,
            "data": [
                {
                    "agent_name": "health_score",
                    "agent_tier": 3,
                    "duration_ms": None,
                    "validation_passed": True,
                    "created_at": "2026-06-01T12:00:00Z",
                    "action_type": "workflow_start",
                },
                {
                    "agent_name": "experiment_monitor",
                    "agent_tier": 3,
                    "duration_ms": None,
                    "validation_passed": True,
                    "created_at": "2026-06-01T12:05:00Z",
                    "action_type": "workflow_start",
                },
            ],
        }

    async def _timed_fetch(self, *args, **kwargs):
        return {
            "success": True,
            "data": [
                {
                    "agent_name": "orchestrator",
                    "agent_tier": 1,
                    "duration_ms": 140.0,
                    "validation_passed": True,
                    "created_at": "2026-06-01T12:00:00Z",
                    "action_type": "route",
                }
            ],
        }

    def test_untimed_window_reports_null_latency(self, client):
        with (
            patch(
                "src.api.routes.analytics._get_supabase_client",
                return_value=_FakeDB(),
            ),
            patch(
                "src.api.routes.analytics._fetch_audit_metrics",
                side_effect=self._untimed_fetch,
            ),
        ):
            response = client.get("/analytics/dashboard?period=7d")

        assert response.status_code == 200
        summary = response.json()["summary"]
        # Real queries happened ...
        assert summary["total_queries"] == 2
        # ... but latency is UNMEASURED -> null, NOT a fabricated 0.
        assert summary["avg_latency_ms"] is None
        assert summary["p50_latency_ms"] is None
        assert summary["p95_latency_ms"] is None
        assert summary["p99_latency_ms"] is None
        assert response.json()["latency_breakdown"]["total_ms"] is None

    def test_timed_window_reports_real_latency(self, client):
        with (
            patch(
                "src.api.routes.analytics._get_supabase_client",
                return_value=_FakeDB(),
            ),
            patch(
                "src.api.routes.analytics._fetch_audit_metrics",
                side_effect=self._timed_fetch,
            ),
        ):
            response = client.get("/analytics/dashboard?period=7d")

        assert response.status_code == 200
        summary = response.json()["summary"]
        assert summary["avg_latency_ms"] == 140.0


class TestQueryVolumeCountedPerWorkflowNotPerEntry:
    """total_queries must count WORKFLOW runs, not raw audit entries.

    Regression target: per-node instrumentation writes one workflow_start genesis
    entry plus N per-node entries per run. Counting len(entries) as queries would
    inflate the volume several-fold once agents emit timed nodes.
    """

    async def _one_workflow_many_nodes(self, *args, **kwargs):
        wid = "11111111-1111-1111-1111-111111111111"
        # One genesis + 4 timed node entries == ONE query/workflow.
        return {
            "success": True,
            "data": [
                {
                    "workflow_id": wid,
                    "agent_name": "gap_analyzer",
                    "agent_tier": 2,
                    "duration_ms": None,
                    "validation_passed": None,
                    "created_at": "2026-06-01T12:00:00Z",
                    "action_type": "workflow_start",
                },
                *[
                    {
                        "workflow_id": wid,
                        "agent_name": "gap_analyzer",
                        "agent_tier": 2,
                        "duration_ms": 50.0,
                        "validation_passed": True,
                        "created_at": "2026-06-01T12:00:01Z",
                        "action_type": node,
                    }
                    for node in ("gap_detector", "roi_calculator", "prioritizer", "formatter")
                ],
            ],
        }

    def test_one_workflow_with_many_nodes_is_one_query(self, client):
        with (
            patch(
                "src.api.routes.analytics._get_supabase_client",
                return_value=_FakeDB(),
            ),
            patch(
                "src.api.routes.analytics._fetch_audit_metrics",
                side_effect=self._one_workflow_many_nodes,
            ),
        ):
            response = client.get("/analytics/dashboard?period=7d")

        assert response.status_code == 200
        summary = response.json()["summary"]
        # 5 audit entries, but ONE workflow -> ONE query (not 5).
        assert summary["total_queries"] == 1
        assert summary["successful_queries"] == 1
        # Real per-node latency is still aggregated (50ms each).
        assert summary["avg_latency_ms"] == 50.0

    def test_volume_trend_counts_one_point_per_workflow(self, client):
        """query_volume_trend buckets each workflow once, not each audit entry."""
        with (
            patch(
                "src.api.routes.analytics._get_supabase_client",
                return_value=_FakeDB(),
            ),
            patch(
                "src.api.routes.analytics._fetch_audit_metrics",
                side_effect=self._one_workflow_many_nodes,
            ),
        ):
            response = client.get("/analytics/dashboard?period=7d")

        assert response.status_code == 200
        trend = response.json()["query_volume_trend"]
        # All 5 entries are in the same hour bucket, but it's ONE workflow.
        total_volume = sum(point["value"] for point in trend)
        assert total_volume == 1.0

    async def _mixed_window(self, *args, **kwargs):
        # A transition window: one instrumented run (workflow_id + genesis + node)
        # PLUS one legacy genesis row that predates instrumentation (no workflow_id).
        wid = "22222222-2222-2222-2222-222222222222"
        return {
            "success": True,
            "data": [
                {
                    "workflow_id": wid,
                    "agent_name": "orchestrator",
                    "agent_tier": 1,
                    "duration_ms": None,
                    "validation_passed": None,
                    "created_at": "2026-06-01T12:00:00Z",
                    "action_type": "workflow_start",
                },
                {
                    "workflow_id": wid,
                    "agent_name": "orchestrator",
                    "agent_tier": 1,
                    "duration_ms": 30.0,
                    "validation_passed": True,
                    "created_at": "2026-06-01T12:00:01Z",
                    "action_type": "classify",
                },
                {
                    # Legacy row: no workflow_id.
                    "agent_name": "health_score",
                    "agent_tier": 3,
                    "duration_ms": None,
                    "validation_passed": True,
                    "created_at": "2026-06-01T13:00:00Z",
                    "action_type": "workflow_start",
                },
            ],
        }

    def test_mixed_window_counts_both_partitions(self, client):
        """A window mixing workflow-id rows and legacy no-id rows counts BOTH.

        The instrumented run (1 workflow via 2 entries) + the legacy genesis row
        (1 query) == 2 queries. The legacy row must NOT be dropped just because
        another row carries a workflow_id.
        """
        with (
            patch(
                "src.api.routes.analytics._get_supabase_client",
                return_value=_FakeDB(),
            ),
            patch(
                "src.api.routes.analytics._fetch_audit_metrics",
                side_effect=self._mixed_window,
            ),
        ):
            response = client.get("/analytics/dashboard?period=7d")

        assert response.status_code == 200
        summary = response.json()["summary"]
        assert summary["total_queries"] == 2
        trend_total = sum(p["value"] for p in response.json()["query_volume_trend"])
        assert trend_total == 2.0

    async def _malformed_ts(self, *args, **kwargs):
        # One row has a garbage created_at; it must drop that point, not 500.
        return {
            "success": True,
            "data": [
                {
                    "workflow_id": "33333333-3333-3333-3333-333333333333",
                    "agent_name": "orchestrator",
                    "agent_tier": 1,
                    "duration_ms": 100.0,
                    "validation_passed": True,
                    "created_at": "not-a-timestamp",
                    "action_type": "classify",
                },
                {
                    "workflow_id": "44444444-4444-4444-4444-444444444444",
                    "agent_name": "orchestrator",
                    "agent_tier": 1,
                    "duration_ms": 200.0,
                    "validation_passed": True,
                    "created_at": "2026-06-01T12:00:00Z",
                    "action_type": "classify",
                },
            ],
        }

    def test_malformed_created_at_does_not_500(self, client):
        """A single unparseable created_at drops that trend point, not the request."""
        with (
            patch(
                "src.api.routes.analytics._get_supabase_client",
                return_value=_FakeDB(),
            ),
            patch(
                "src.api.routes.analytics._fetch_audit_metrics",
                side_effect=self._malformed_ts,
            ),
        ):
            response = client.get("/analytics/dashboard?period=7d")

        # Must not crash; the well-formed row still produces a trend point.
        assert response.status_code == 200
        # Both workflows still counted (counting doesn't depend on ts parse).
        assert response.json()["summary"]["total_queries"] == 2
