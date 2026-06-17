"""
Tests for Memory API endpoints.

Tests the memory system endpoints:
- POST /memory/search
- POST /memory/episodic
- GET /memory/episodic/{id}
- POST /memory/procedural/feedback
- GET /memory/semantic/paths
- GET /memory/stats
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.rag.models.retrieval_models import RetrievalResult

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_hybrid_search():
    """Mock hybrid_search function."""
    with patch("src.api.routes.memory.hybrid_search") as mock:
        mock.return_value = [
            RetrievalResult(
                content="TRx dropped due to HCP engagement decline",
                source="episodic_memories",
                source_id="mem_1",
                score=0.85,
                retrieval_method="dense",
                metadata={"brand": "Kisqali"},
            ),
            RetrievalResult(
                content="Causal path: HCP visits → Script volume → TRx",
                source="causal_paths",
                source_id="path_1",
                score=0.75,
                retrieval_method="sparse",
                metadata={},
            ),
        ]
        yield mock


@pytest.fixture
def mock_episodic_memory_functions():
    """Mock episodic memory functions."""
    with (
        patch("src.api.routes.memory.insert_episodic_memory_with_text") as mock_insert,
        patch("src.api.routes.memory.get_memory_by_id") as mock_get,
    ):
        mock_insert.return_value = "mem_123"
        mock_get.return_value = {
            "memory_id": "mem_123",
            "description": "Test memory content",
            "event_type": "query",
            "session_id": "sess_abc",
            "agent_name": "orchestrator",
            "brand": "Kisqali",
            "region": "northeast",
            "raw_content": {},
            "occurred_at": "2025-01-01T00:00:00",
        }
        yield {"insert": mock_insert, "get": mock_get}


@pytest.fixture
def mock_procedural_memory_functions():
    """Mock procedural memory functions."""
    with (
        patch("src.api.routes.memory.update_procedure_outcome") as mock_update,
        patch("src.api.routes.memory.get_procedure_by_id") as mock_get,
        patch("src.api.routes.memory.record_learning_signal") as mock_signal,
    ):
        mock_update.return_value = None
        mock_get.return_value = {"procedure_id": "proc_001", "usage_count": 10, "success_count": 9}
        mock_signal.return_value = "signal_123"
        yield {"update": mock_update, "get": mock_get, "signal": mock_signal}


@pytest.fixture
def brand_scoped_viewer():
    """Override auth to a NON-admin viewer holding a single brand grant
    (``BrandA``).

    The default ``TEST_USER`` is a cross-brand admin, so it bypasses every
    per-tenant gate. To exercise the L13 procedural-feedback BOLA check we must
    authenticate as a non-admin whose grants we control.
    """
    from src.api.dependencies.auth import require_viewer

    user = {
        "id": "viewer-a",
        "email": "viewer-a@e2i-analytics.com",
        "app_metadata": {"role": "viewer", "brands": ["BrandA"]},
    }

    async def _fake_viewer():
        return user

    app.dependency_overrides[require_viewer] = _fake_viewer
    yield user
    app.dependency_overrides.pop(require_viewer, None)


@pytest.fixture
def mock_semantic_memory():
    """Mock semantic memory."""
    memory = MagicMock()
    memory.find_causal_paths_for_kpi = MagicMock(
        return_value=[
            {
                "nodes": ["HCP engagement", "Script volume", "TRx"],
                "confidence": 0.85,
                "path_id": "path_1",
            }
        ]
    )
    memory.traverse_causal_chain = MagicMock(
        return_value=[{"path": ["Entity A", "Entity B"], "confidence": 0.8}]
    )
    return memory


# =============================================================================
# SEARCH ENDPOINT TESTS
# =============================================================================


class TestMemorySearch:
    """Tests for POST /memory/search."""

    def test_search_returns_results(self, mock_hybrid_search):
        """search should return hybrid search results."""
        response = client.post(
            "/api/memory/search", json={"query": "Why did TRx drop in northeast?", "k": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 2
        assert len(data["results"]) == 2
        assert data["query"] == "Why did TRx drop in northeast?"

    def test_search_with_filters(self, mock_hybrid_search):
        """search should pass filters to hybrid_search."""
        response = client.post(
            "/api/memory/search",
            json={
                "query": "TRx trends",
                "k": 5,
                "filters": {"brand": "Kisqali", "region": "northeast"},
            },
        )

        assert response.status_code == 200
        mock_hybrid_search.assert_called_once()
        call_kwargs = mock_hybrid_search.call_args[1]
        assert call_kwargs["filters"] == {"brand": "Kisqali", "region": "northeast"}

    def test_search_with_kpi_name(self, mock_hybrid_search):
        """search should pass kpi_name for targeted retrieval."""
        response = client.post(
            "/api/memory/search", json={"query": "What impacts TRx?", "kpi_name": "TRx"}
        )

        assert response.status_code == 200
        call_kwargs = mock_hybrid_search.call_args[1]
        assert call_kwargs["kpi_name"] == "TRx"

    def test_search_filters_by_min_score(self, mock_hybrid_search):
        """search should filter results below min_score."""
        response = client.post(
            "/api/memory/search", json={"query": "TRx analysis", "min_score": 0.8}
        )

        assert response.status_code == 200
        data = response.json()
        # Only the first result (0.85) should pass the 0.8 threshold
        assert data["total_results"] == 1
        assert data["results"][0]["score"] >= 0.8

    def test_search_includes_latency(self, mock_hybrid_search):
        """search should include search latency in response."""
        response = client.post("/api/memory/search", json={"query": "test query"})

        assert response.status_code == 200
        data = response.json()
        assert "search_latency_ms" in data
        assert data["search_latency_ms"] >= 0

    def test_search_validates_query_length(self):
        """search should reject empty queries."""
        response = client.post("/api/memory/search", json={"query": ""})

        assert response.status_code == 422  # Validation error


# =============================================================================
# EPISODIC MEMORY TESTS
# =============================================================================


class TestEpisodicMemory:
    """Tests for episodic memory endpoints."""

    def test_create_episodic_memory(self, mock_episodic_memory_functions):
        """POST /memory/episodic should create a new memory."""
        response = client.post(
            "/api/memory/episodic",
            json={
                "content": "User asked about TRx trends",
                "event_type": "query",
                "session_id": "sess_123",
                "agent_name": "orchestrator",
                "brand": "Kisqali",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mem_123"
        assert data["content"] == "User asked about TRx trends"
        assert data["event_type"] == "query"

    def test_create_episodic_memory_minimal(self, mock_episodic_memory_functions):
        """POST /memory/episodic should work with minimal fields."""
        response = client.post(
            "/api/memory/episodic", json={"content": "Minimal memory", "event_type": "action"}
        )

        assert response.status_code == 200

    def test_get_episodic_memory_by_id(self, mock_episodic_memory_functions):
        """GET /memory/episodic/{id} should retrieve a memory."""
        response = client.get("/api/memory/episodic/mem_123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mem_123"
        assert data["content"] == "Test memory content"
        assert data["brand"] == "Kisqali"

    def test_get_episodic_memory_not_found(self):
        """GET /memory/episodic/{id} should return 404 for missing memory."""
        with patch("src.api.routes.memory.get_memory_by_id") as mock_get:
            mock_get.return_value = None
            response = client.get("/api/memory/episodic/nonexistent")

        assert response.status_code == 404


# =============================================================================
# PROCEDURAL MEMORY TESTS
# =============================================================================


class TestProceduralFeedback:
    """Tests for POST /memory/procedural/feedback."""

    def test_record_feedback_success(self, mock_procedural_memory_functions):
        """Should record feedback and return new success rate."""
        response = client.post(
            "/api/memory/procedural/feedback",
            json={
                "procedure_id": "proc_001",
                "outcome": "success",
                "score": 0.9,
                "feedback_text": "Analysis was accurate",
                "agent_name": "feedback_learner",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["procedure_id"] == "proc_001"
        assert data["feedback_recorded"] is True
        # 9 successes / 10 usage = 0.9 success rate
        assert data["new_success_rate"] == 0.9
        # Verify mocks were called
        mock_procedural_memory_functions["update"].assert_called_once()
        mock_procedural_memory_functions["signal"].assert_called_once()

    def test_record_feedback_minimal(self, mock_procedural_memory_functions):
        """Should work with minimal required fields."""
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "proc_002", "outcome": "partial", "score": 0.6},
        )

        assert response.status_code == 200

    def test_record_feedback_rejects_invalid_outcome(self, mock_procedural_memory_functions):
        """L13 (#694): an outcome outside {success, partial, failure} (incl.
        typos like 'sucess') must 422 via Pydantic rather than being silently
        recorded as a failure."""
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "proc_003", "outcome": "sucess", "score": 0.9},
        )

        assert response.status_code == 422
        # The procedure outcome must NOT have been updated for an invalid value.
        mock_procedural_memory_functions["update"].assert_not_called()

    def test_record_feedback_partial_is_not_full_success(self, mock_procedural_memory_functions):
        """L13 (#694): 'partial' is a distinct outcome — it does NOT count as
        a full success, so update_procedure_outcome is called with
        success=False (same as 'failure'), but it is an accepted value (200)."""
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "proc_004", "outcome": "partial", "score": 0.5},
        )

        assert response.status_code == 200
        mock_procedural_memory_functions["update"].assert_called_once()
        _, kwargs = mock_procedural_memory_functions["update"].call_args
        assert kwargs["success"] is False

    def test_record_feedback_failure_records_not_success(self, mock_procedural_memory_functions):
        """'failure' is accepted and recorded as success=False."""
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "proc_005", "outcome": "failure", "score": 0.1},
        )

        assert response.status_code == 200
        _, kwargs = mock_procedural_memory_functions["update"].call_args
        assert kwargs["success"] is False

    def test_record_feedback_success_records_success(self, mock_procedural_memory_functions):
        """'success' is accepted and recorded as success=True."""
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "proc_006", "outcome": "success", "score": 0.95},
        )

        assert response.status_code == 200
        _, kwargs = mock_procedural_memory_functions["update"].call_args
        assert kwargs["success"] is True

    # --- L13 (#694) BOLA: ownership/brand grant check ---------------------------

    def test_feedback_cross_tenant_is_denied_404(
        self, mock_procedural_memory_functions, brand_scoped_viewer
    ):
        """L13 (#694) BOLA: a non-admin viewer granted ``BrandA`` must NOT be
        able to mutate a procedure scoped to ``BrandB``. The endpoint returns
        404 (existence-hiding, matching the episodic gate) and performs NO
        success-count mutation or DSPy learning-signal injection."""
        mock_procedural_memory_functions["get"].return_value = {
            "procedure_id": "proc_b",
            "usage_count": 10,
            "success_count": 9,
            "applicable_brands": ["BrandB"],
        }
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "proc_b", "outcome": "success", "score": 0.9},
        )

        assert response.status_code == 404
        mock_procedural_memory_functions["update"].assert_not_called()
        mock_procedural_memory_functions["signal"].assert_not_called()

    def test_feedback_in_grant_is_allowed(
        self, mock_procedural_memory_functions, brand_scoped_viewer
    ):
        """A non-admin viewer may record feedback for a procedure scoped to a
        brand they are granted."""
        mock_procedural_memory_functions["get"].return_value = {
            "procedure_id": "proc_a",
            "usage_count": 10,
            "success_count": 9,
            "applicable_brands": ["BrandA"],
        }
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "proc_a", "outcome": "success", "score": 0.9},
        )

        assert response.status_code == 200
        mock_procedural_memory_functions["update"].assert_called_once()

    def test_feedback_partial_brand_overlap_is_allowed(
        self, mock_procedural_memory_functions, brand_scoped_viewer
    ):
        """A procedure scoped to MULTIPLE brands is ratable as long as the
        caller is granted at least one of them — exercises the ``any()`` overlap
        (the grant ``BrandA`` matches the procedure's second listed brand)."""
        mock_procedural_memory_functions["get"].return_value = {
            "procedure_id": "proc_multi",
            "usage_count": 8,
            "success_count": 4,
            "applicable_brands": ["BrandB", "BrandA"],
        }
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "proc_multi", "outcome": "success", "score": 0.7},
        )

        assert response.status_code == 200
        mock_procedural_memory_functions["update"].assert_called_once()

    def test_feedback_global_procedure_is_allowed(
        self, mock_procedural_memory_functions, brand_scoped_viewer
    ):
        """A procedure with ``applicable_brands=['all']`` is GLOBAL and ratable
        by any authenticated viewer, regardless of their brand grant."""
        mock_procedural_memory_functions["get"].return_value = {
            "procedure_id": "proc_all",
            "usage_count": 4,
            "success_count": 2,
            "applicable_brands": ["all"],
        }
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "proc_all", "outcome": "failure", "score": 0.1},
        )

        assert response.status_code == 200
        mock_procedural_memory_functions["update"].assert_called_once()

    def test_feedback_nonexistent_procedure_is_404(
        self, mock_procedural_memory_functions, brand_scoped_viewer
    ):
        """A feedback POST for a procedure that does not exist returns 404 and
        performs no mutation (previously it silently 200'd with a null rate)."""
        mock_procedural_memory_functions["get"].return_value = None
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "ghost", "outcome": "success", "score": 0.9},
        )

        assert response.status_code == 404
        mock_procedural_memory_functions["update"].assert_not_called()
        mock_procedural_memory_functions["signal"].assert_not_called()

    def test_feedback_admin_bypasses_brand_scope(self, mock_procedural_memory_functions):
        """A cross-brand admin (the default ``TEST_USER``) may record feedback
        for any procedure regardless of its brand scope."""
        mock_procedural_memory_functions["get"].return_value = {
            "procedure_id": "proc_b",
            "usage_count": 10,
            "success_count": 9,
            "applicable_brands": ["BrandB"],
        }
        response = client.post(
            "/api/memory/procedural/feedback",
            json={"procedure_id": "proc_b", "outcome": "success", "score": 0.9},
        )

        assert response.status_code == 200
        mock_procedural_memory_functions["update"].assert_called_once()


# =============================================================================
# SEMANTIC PATH TESTS
# =============================================================================


class TestSemanticPaths:
    """Tests for GET /memory/semantic/paths."""

    def test_query_paths_by_kpi(self, mock_semantic_memory):
        """Should find paths for a given KPI."""
        with patch("src.api.routes.memory.get_semantic_memory", return_value=mock_semantic_memory):
            response = client.get(
                "/api/memory/semantic/paths", params={"kpi_name": "TRx", "min_confidence": 0.6}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_paths"] == 1
        assert "query_latency_ms" in data
        mock_semantic_memory.find_causal_paths_for_kpi.assert_called_once_with(
            kpi_name="TRx", min_confidence=0.6
        )

    def test_query_paths_by_entity(self, mock_semantic_memory):
        """Should traverse from a starting entity."""
        with patch("src.api.routes.memory.get_semantic_memory", return_value=mock_semantic_memory):
            response = client.get(
                "/api/memory/semantic/paths", params={"start_entity_id": "ent_001", "max_depth": 2}
            )

        assert response.status_code == 200
        mock_semantic_memory.traverse_causal_chain.assert_called_once()


# =============================================================================
# STATS ENDPOINT TESTS
# =============================================================================


class TestMemoryStats:
    """Tests for GET /memory/stats."""

    def test_get_stats_returns_structure(self):
        """Should return stats for all memory types."""
        response = client.get("/api/memory/stats")

        assert response.status_code == 200
        data = response.json()
        assert "working" in data
        assert "episodic" in data
        assert "procedural" in data
        assert "semantic" in data
        assert "last_updated" in data

    def test_get_stats_includes_live_working_block(self):
        """Working-memory card is wired: live active-session count + status."""
        wm = MagicMock()
        wm.count_active_sessions = AsyncMock(return_value=7)
        wm.ttl_seconds = 86400
        with patch("src.memory.working_memory.get_working_memory", return_value=wm):
            response = client.get("/api/memory/stats")

        assert response.status_code == 200
        working = response.json()["working"]
        assert working["active_sessions"] == 7
        assert working["status"] == "healthy"
        assert working["ttl_hours"] == 24.0


# =============================================================================
# LIST EPISODIC MEMORIES (PR #293)
# =============================================================================


class TestListEpisodicMemories:
    """Tests for GET /memory/episodic (PR #293)."""

    def test_returns_array_with_default_limit(self):
        with patch("src.api.routes.memory.get_recent_memories") as mock_recent:
            mock_recent.return_value = [
                {
                    "memory_id": "mem_1",
                    "description": "Engagement dropped",
                    "event_type": "query",
                    "session_id": "s1",
                    "agent_name": "orchestrator",
                    "brand": "Kisqali",
                    "region": "northeast",
                    "occurred_at": "2025-01-01T00:00:00",
                    "raw_content": {"foo": "bar"},
                },
                {
                    "memory_id": "mem_2",
                    "description": "Result",
                    "event_type": "result",
                    "occurred_at": "2025-01-02T00:00:00",
                },
            ]
            response = client.get("/api/memory/episodic")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["id"] == "mem_1"
        assert data[0]["event_type"] == "query"
        assert data[0]["metadata"] == {"foo": "bar"}
        # Default kwargs forwarded
        mock_recent.assert_called_once_with(limit=20, event_types=None, agent_name=None, brand=None)

    def test_forwards_filter_kwargs(self):
        with patch("src.api.routes.memory.get_recent_memories") as mock_recent:
            mock_recent.return_value = []
            response = client.get(
                "/api/memory/episodic?event_type=query&agent_name=orch&brand=K&limit=5"
            )

        assert response.status_code == 200
        mock_recent.assert_called_once_with(
            limit=5, event_types=["query"], agent_name="orch", brand="K"
        )

    def test_session_id_filter_is_applied_post_query(self):
        with patch("src.api.routes.memory.get_recent_memories") as mock_recent:
            mock_recent.return_value = [
                {
                    "memory_id": "mem_1",
                    "session_id": "sess_A",
                    "description": "a",
                    "event_type": "q",
                },
                {
                    "memory_id": "mem_2",
                    "session_id": "sess_B",
                    "description": "b",
                    "event_type": "q",
                },
            ]
            response = client.get("/api/memory/episodic?session_id=sess_A")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "mem_1"

    def test_handles_recent_memories_error(self):
        with patch(
            "src.api.routes.memory.get_recent_memories",
            side_effect=Exception("supabase down"),
        ):
            response = client.get("/api/memory/episodic")

        assert response.status_code == 500
        body = response.json()
        # E2I error envelope: 'message' is the user-facing field; 'detail' is FastAPI's
        # raw default but the global handler wraps it.
        assert "Failed to list memories" in body.get("message", body.get("detail", ""))
