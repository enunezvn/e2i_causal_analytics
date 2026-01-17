"""Unit tests for Audit Chain API routes.

Tests cover:
- GET /audit/workflow/{workflow_id}: Get workflow entries
- GET /audit/workflow/{workflow_id}/verify: Verify chain integrity
- GET /audit/workflow/{workflow_id}/summary: Get workflow summary
- GET /audit/recent: Get recent workflows

Version: 1.0.0
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.audit import router, get_audit_service


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def workflow_id():
    """Create a test workflow ID."""
    return uuid4()


@pytest.fixture
def entry_id():
    """Create a test entry ID."""
    return uuid4()


@pytest.fixture
def mock_audit_entry_row(workflow_id, entry_id):
    """Create a mock database row for an audit entry."""
    return {
        "entry_id": str(entry_id),
        "workflow_id": str(workflow_id),
        "sequence_number": 1,
        "agent_name": "orchestrator",
        "agent_tier": 1,
        "action_type": "route_query",
        "created_at": "2025-01-15T10:30:00Z",
        "duration_ms": 150,
        "validation_passed": True,
        "confidence_score": 0.95,
        "refutation_results": None,
        "previous_entry_id": None,
        "previous_hash": None,
        "entry_hash": "abc123hash",
        "user_id": "user-123",
        "session_id": str(uuid4()),
        "brand": "remibrutinib",
    }


@pytest.fixture
def mock_audit_entries(workflow_id):
    """Create multiple mock audit entries for a workflow."""
    entry1_id = uuid4()
    entry2_id = uuid4()
    entry3_id = uuid4()

    return [
        {
            "entry_id": str(entry1_id),
            "workflow_id": str(workflow_id),
            "sequence_number": 1,
            "agent_name": "orchestrator",
            "agent_tier": 1,
            "action_type": "route_query",
            "created_at": "2025-01-15T10:30:00Z",
            "duration_ms": 100,
            "validation_passed": True,
            "confidence_score": 0.92,
            "refutation_results": None,
            "previous_entry_id": None,
            "previous_hash": None,
            "entry_hash": "hash1",
            "user_id": "user-123",
            "session_id": None,
            "brand": "remibrutinib",
        },
        {
            "entry_id": str(entry2_id),
            "workflow_id": str(workflow_id),
            "sequence_number": 2,
            "agent_name": "causal_impact",
            "agent_tier": 2,
            "action_type": "analyze",
            "created_at": "2025-01-15T10:30:05Z",
            "duration_ms": 500,
            "validation_passed": True,
            "confidence_score": 0.88,
            "refutation_results": {"test": "passed"},
            "previous_entry_id": str(entry1_id),
            "previous_hash": "hash1",
            "entry_hash": "hash2",
            "user_id": "user-123",
            "session_id": None,
            "brand": "remibrutinib",
        },
        {
            "entry_id": str(entry3_id),
            "workflow_id": str(workflow_id),
            "sequence_number": 3,
            "agent_name": "explainer",
            "agent_tier": 5,
            "action_type": "explain",
            "created_at": "2025-01-15T10:30:10Z",
            "duration_ms": 200,
            "validation_passed": False,
            "confidence_score": 0.75,
            "refutation_results": None,
            "previous_entry_id": str(entry2_id),
            "previous_hash": "hash2",
            "entry_hash": "hash3",
            "user_id": "user-123",
            "session_id": None,
            "brand": "remibrutinib",
        },
    ]


@pytest.fixture
def mock_verification_result(workflow_id):
    """Create a mock chain verification result."""
    result = MagicMock()
    result.is_valid = True
    result.entries_checked = 3
    result.first_invalid_entry = None
    result.error_message = None
    result.verified_at = datetime.now(timezone.utc)
    return result


@pytest.fixture
def mock_audit_service(mock_audit_entry_row, mock_verification_result):
    """Create a mock AuditChainService."""
    service = MagicMock()

    # Mock database query builder pattern
    mock_query = MagicMock()
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.range.return_value = mock_query
    mock_query.limit.return_value = mock_query

    # Mock execute to return data
    mock_result = MagicMock()
    mock_result.data = [mock_audit_entry_row]
    mock_query.execute.return_value = mock_result

    # Mock table method
    service.db.table.return_value = mock_query

    # Mock verify_workflow
    service.verify_workflow.return_value = mock_verification_result

    # Mock RPC for recent workflows
    mock_rpc = MagicMock()
    mock_rpc.execute.return_value = MagicMock(data=[])
    service.db.rpc.return_value = mock_rpc

    return service


@pytest.fixture
def app(mock_audit_service):
    """Create a FastAPI app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)

    # Patch get_audit_service to return our mock
    with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
        yield app


@pytest.fixture
def client(mock_audit_service):
    """Create a test client with mocked service."""
    app = FastAPI()
    app.include_router(router)

    with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
        with TestClient(app) as client:
            yield client


# =============================================================================
# GET WORKFLOW ENTRIES TESTS
# =============================================================================


class TestGetWorkflowEntries:
    """Tests for GET /audit/workflow/{workflow_id} endpoint."""

    def test_get_entries_success(self, mock_audit_service, workflow_id, mock_audit_entry_row):
        """Test successful retrieval of workflow entries."""
        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["agent_name"] == "orchestrator"
        assert data[0]["sequence_number"] == 1

    def test_get_entries_empty(self, mock_audit_service, workflow_id):
        """Test retrieval when workflow has no entries."""
        # Configure mock to return empty data
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.range.return_value = mock_query
        mock_result = MagicMock()
        mock_result.data = []
        mock_query.execute.return_value = mock_result
        mock_audit_service.db.table.return_value = mock_query

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_get_entries_with_pagination(self, mock_audit_service, workflow_id):
        """Test pagination parameters are passed correctly."""
        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}?limit=50&offset=10")

        assert response.status_code == 200
        # Verify range was called with pagination params
        mock_audit_service.db.table.return_value.range.assert_called_with(10, 59)

    def test_get_entries_service_unavailable(self, workflow_id):
        """Test error when audit service is unavailable."""
        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=None):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}")

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()

    def test_get_entries_database_error(self, mock_audit_service, workflow_id):
        """Test error handling on database failure."""
        mock_audit_service.db.table.return_value.execute.side_effect = Exception(
            "Database connection failed"
        )

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}")

        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


# =============================================================================
# VERIFY WORKFLOW CHAIN TESTS
# =============================================================================


class TestVerifyWorkflowChain:
    """Tests for GET /audit/workflow/{workflow_id}/verify endpoint."""

    def test_verify_chain_valid(self, mock_audit_service, workflow_id, mock_verification_result):
        """Test verification of a valid chain."""
        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/verify")

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["entries_checked"] == 3
        assert data["first_invalid_entry"] is None
        assert data["error_message"] is None

    def test_verify_chain_invalid(self, mock_audit_service, workflow_id):
        """Test verification of an invalid chain."""
        invalid_entry_id = uuid4()
        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.entries_checked = 2
        mock_result.first_invalid_entry = invalid_entry_id
        mock_result.error_message = "Hash mismatch at entry 2"
        mock_result.verified_at = datetime.now(timezone.utc)
        mock_audit_service.verify_workflow.return_value = mock_result

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/verify")

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert data["entries_checked"] == 2
        assert data["first_invalid_entry"] == str(invalid_entry_id)
        assert "hash mismatch" in data["error_message"].lower()

    def test_verify_chain_service_unavailable(self, workflow_id):
        """Test error when audit service is unavailable."""
        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=None):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/verify")

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()

    def test_verify_chain_verification_error(self, mock_audit_service, workflow_id):
        """Test error handling when verification fails."""
        mock_audit_service.verify_workflow.side_effect = Exception("Verification failed")

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/verify")

        assert response.status_code == 500
        assert "verification failed" in response.json()["detail"].lower()


# =============================================================================
# GET WORKFLOW SUMMARY TESTS
# =============================================================================


class TestGetWorkflowSummary:
    """Tests for GET /audit/workflow/{workflow_id}/summary endpoint."""

    def test_get_summary_success(self, mock_audit_service, workflow_id, mock_audit_entries):
        """Test successful retrieval of workflow summary."""
        # Configure mock to return multiple entries
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_result = MagicMock()
        mock_result.data = mock_audit_entries
        mock_query.execute.return_value = mock_result
        mock_audit_service.db.table.return_value = mock_query

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["total_entries"] == 3
        assert set(data["agents_involved"]) == {"causal_impact", "explainer", "orchestrator"}
        assert set(data["tiers_involved"]) == {1, 2, 5}
        assert data["brand"] == "remibrutinib"
        assert data["total_duration_ms"] == 800  # 100 + 500 + 200
        assert data["validation_passed_count"] == 2
        assert data["validation_failed_count"] == 1

    def test_get_summary_workflow_not_found(self, mock_audit_service, workflow_id):
        """Test summary for non-existent workflow."""
        # Configure mock to return empty data
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_result = MagicMock()
        mock_result.data = []
        mock_query.execute.return_value = mock_result
        mock_audit_service.db.table.return_value = mock_query

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/summary")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_summary_with_avg_confidence(self, mock_audit_service, workflow_id, mock_audit_entries):
        """Test that average confidence score is calculated correctly."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_result = MagicMock()
        mock_result.data = mock_audit_entries
        mock_query.execute.return_value = mock_result
        mock_audit_service.db.table.return_value = mock_query

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/summary")

        assert response.status_code == 200
        data = response.json()
        # (0.92 + 0.88 + 0.75) / 3 = 0.85
        expected_avg = (0.92 + 0.88 + 0.75) / 3
        assert abs(data["avg_confidence_score"] - expected_avg) < 0.01

    def test_get_summary_service_unavailable(self, workflow_id):
        """Test error when audit service is unavailable."""
        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=None):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/summary")

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()

    def test_get_summary_chain_verification_status(
        self, mock_audit_service, workflow_id, mock_audit_entries
    ):
        """Test that chain verification status is included in summary."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_result = MagicMock()
        mock_result.data = mock_audit_entries
        mock_query.execute.return_value = mock_result
        mock_audit_service.db.table.return_value = mock_query

        # Mock verification to return valid chain
        mock_verification = MagicMock()
        mock_verification.is_valid = True
        mock_audit_service.verify_workflow.return_value = mock_verification

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["chain_verified"] is True


# =============================================================================
# GET RECENT WORKFLOWS TESTS
# =============================================================================


class TestGetRecentWorkflows:
    """Tests for GET /audit/recent endpoint."""

    def test_get_recent_workflows_via_rpc(self, mock_audit_service):
        """Test successful retrieval via RPC."""
        workflow_id = uuid4()
        mock_rpc_data = [
            {
                "workflow_id": str(workflow_id),
                "started_at": "2025-01-15T10:30:00Z",
                "entry_count": 5,
                "first_agent": "orchestrator",
                "last_agent": "explainer",
                "brand": "remibrutinib",
            }
        ]
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = MagicMock(data=mock_rpc_data)
        mock_audit_service.db.rpc.return_value = mock_rpc

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get("/audit/recent")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["entry_count"] == 5
        assert data[0]["first_agent"] == "orchestrator"
        assert data[0]["last_agent"] == "explainer"

    def test_get_recent_workflows_with_filters(self, mock_audit_service):
        """Test recent workflows with brand and agent filters."""
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = MagicMock(data=[])
        mock_audit_service.db.rpc.return_value = mock_rpc

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get("/audit/recent?limit=10&brand=kisqali&agent_name=causal_impact")

        assert response.status_code == 200
        # Verify RPC was called with correct params
        mock_audit_service.db.rpc.assert_called_with(
            "get_recent_audit_workflows",
            {"p_limit": 10, "p_brand": "kisqali", "p_agent_name": "causal_impact"},
        )

    def test_get_recent_workflows_fallback(self, mock_audit_service):
        """Test fallback to direct query when RPC fails."""
        # Make RPC fail
        mock_audit_service.db.rpc.side_effect = Exception("RPC not available")

        # Set up fallback query mock
        workflow_id = uuid4()
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query

        # First query returns workflow with sequence 1
        first_result = MagicMock()
        first_result.data = [
            {
                "workflow_id": str(workflow_id),
                "created_at": "2025-01-15T10:30:00Z",
                "agent_name": "orchestrator",
                "brand": "remibrutinib",
            }
        ]
        # Second query returns count info
        count_result = MagicMock()
        count_result.data = [{"agent_name": "explainer", "sequence_number": 3}]

        mock_query.execute.side_effect = [first_result, count_result]
        mock_audit_service.db.table.return_value = mock_query

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get("/audit/recent")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["first_agent"] == "orchestrator"
        assert data[0]["last_agent"] == "explainer"
        assert data[0]["entry_count"] == 3

    def test_get_recent_workflows_empty(self):
        """Test when no recent workflows exist."""
        # Create a fresh mock service with empty data
        mock_service = MagicMock()

        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = MagicMock(data=[])
        mock_service.db.rpc.return_value = mock_rpc

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_service):
            client = TestClient(app)
            response = client.get("/audit/recent")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_get_recent_workflows_service_unavailable(self):
        """Test error when audit service is unavailable."""
        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=None):
            client = TestClient(app)
            response = client.get("/audit/recent")

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()

    def test_get_recent_workflows_limit_bounds(self, mock_audit_service):
        """Test limit parameter bounds validation."""
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = MagicMock(data=[])
        mock_audit_service.db.rpc.return_value = mock_rpc

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)

            # Test limit too high
            response = client.get("/audit/recent?limit=200")
            assert response.status_code == 422  # Validation error

            # Test limit too low
            response = client.get("/audit/recent?limit=0")
            assert response.status_code == 422

            # Test valid limit
            response = client.get("/audit/recent?limit=50")
            assert response.status_code == 200


# =============================================================================
# RESPONSE MODEL TESTS
# =============================================================================


class TestResponseModels:
    """Tests for response model serialization."""

    def test_audit_entry_response_serialization(
        self, mock_audit_service, workflow_id, mock_audit_entry_row
    ):
        """Test that audit entry response serializes correctly."""
        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}")

        assert response.status_code == 200
        data = response.json()[0]

        # Verify all required fields are present
        assert "entry_id" in data
        assert "workflow_id" in data
        assert "sequence_number" in data
        assert "agent_name" in data
        assert "agent_tier" in data
        assert "action_type" in data
        assert "created_at" in data
        assert "entry_hash" in data

    def test_verification_response_serialization(
        self, mock_audit_service, workflow_id, mock_verification_result
    ):
        """Test that verification response serializes correctly."""
        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/verify")

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields
        assert "workflow_id" in data
        assert "is_valid" in data
        assert "entries_checked" in data
        assert "verified_at" in data

    def test_summary_response_serialization(
        self, mock_audit_service, workflow_id, mock_audit_entries
    ):
        """Test that summary response serializes correctly."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_result = MagicMock()
        mock_result.data = mock_audit_entries
        mock_query.execute.return_value = mock_result
        mock_audit_service.db.table.return_value = mock_query

        app = FastAPI()
        app.include_router(router)

        with patch("src.api.routes.audit.get_audit_service", return_value=mock_audit_service):
            client = TestClient(app)
            response = client.get(f"/audit/workflow/{workflow_id}/summary")

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields
        assert "workflow_id" in data
        assert "total_entries" in data
        assert "agents_involved" in data
        assert "tiers_involved" in data
        assert "chain_verified" in data
        assert "total_duration_ms" in data
