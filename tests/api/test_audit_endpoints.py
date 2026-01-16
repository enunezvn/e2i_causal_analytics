"""
Tests for Audit Chain API endpoints.

Phase 3A of API Audit - Audit Chain API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 3A.1: Chain Verification (GET /workflow/{id}, GET /workflow/{id}/verify,
  GET /workflow/{id}/summary, GET /recent)
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.audit_chain import ChainVerificationResult

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_workflow_id():
    """Generate a consistent workflow ID for tests."""
    return uuid4()


@pytest.fixture
def mock_entry_id():
    """Generate a consistent entry ID for tests."""
    return uuid4()


@pytest.fixture
def mock_audit_entry_row(mock_workflow_id, mock_entry_id):
    """Mock database row for audit chain entry."""
    return {
        "entry_id": str(mock_entry_id),
        "workflow_id": str(mock_workflow_id),
        "sequence_number": 1,
        "agent_name": "orchestrator",
        "agent_tier": 1,
        "action_type": "query_routing",
        "created_at": "2024-01-15T10:30:00Z",
        "duration_ms": 125,
        "validation_passed": True,
        "confidence_score": 0.92,
        "refutation_results": None,
        "previous_entry_id": None,
        "previous_hash": None,
        "entry_hash": "abc123def456",
        "user_id": "analyst@example.com",
        "session_id": str(uuid4()),
        "brand": "Kisqali",
    }


@pytest.fixture
def mock_verification_result(mock_workflow_id):
    """Mock chain verification result."""
    return ChainVerificationResult(
        is_valid=True,
        entries_checked=5,
        first_invalid_entry=None,
        error_message=None,
        verified_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_recent_workflow_row(mock_workflow_id):
    """Mock recent workflow data from RPC."""
    return {
        "workflow_id": str(mock_workflow_id),
        "started_at": "2024-01-15T10:30:00Z",
        "entry_count": 5,
        "first_agent": "orchestrator",
        "last_agent": "explainer",
        "brand": "Remibrutinib",
    }


@pytest.fixture
def mock_audit_service(mock_audit_entry_row, mock_verification_result):
    """Mock AuditChainService instance."""
    service = MagicMock()

    # Mock db attribute with table queries
    mock_db = MagicMock()
    service.db = mock_db

    # Mock table query chain
    mock_table = MagicMock()
    mock_select = MagicMock()
    mock_eq = MagicMock()
    mock_order = MagicMock()
    mock_range = MagicMock()
    mock_limit = MagicMock()
    mock_execute = MagicMock()

    mock_db.table.return_value = mock_table
    mock_table.select.return_value = mock_select
    mock_select.eq.return_value = mock_eq
    mock_eq.order.return_value = mock_order
    mock_order.range.return_value = mock_range
    mock_range.execute.return_value = MagicMock(data=[mock_audit_entry_row])
    mock_order.execute.return_value = MagicMock(data=[mock_audit_entry_row])
    mock_order.limit.return_value = mock_limit
    mock_limit.execute.return_value = MagicMock(data=[mock_audit_entry_row])

    # Mock RPC calls
    mock_rpc = MagicMock()
    mock_db.rpc.return_value = mock_rpc
    mock_rpc.execute.return_value = MagicMock(data=[])

    # Mock verify_workflow method
    service.verify_workflow = MagicMock(return_value=mock_verification_result)

    return service


# =============================================================================
# BATCH 3A.1 - AUDIT CHAIN TESTS
# =============================================================================


class TestGetWorkflowEntries:
    """Tests for GET /audit/workflow/{workflow_id}."""

    def test_get_entries_success(self, mock_audit_service, mock_workflow_id):
        """Should return audit entries for a workflow."""
        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get(f"/api/audit/workflow/{mock_workflow_id}")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        # Check entry structure
        entry = data[0]
        assert "entry_id" in entry
        assert "workflow_id" in entry
        assert "sequence_number" in entry
        assert "agent_name" in entry
        assert "entry_hash" in entry

    def test_get_entries_with_pagination(self, mock_audit_service, mock_workflow_id):
        """Should respect pagination parameters."""
        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get(
                f"/api/audit/workflow/{mock_workflow_id}",
                params={"limit": 10, "offset": 5},
            )

        assert response.status_code == 200
        # Verify pagination was applied
        mock_audit_service.db.table.assert_called_with("audit_chain_entries")

    def test_get_entries_empty_workflow(self, mock_audit_service, mock_workflow_id):
        """Should return empty list for workflow with no entries."""
        # Override to return empty data
        mock_audit_service.db.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value = MagicMock(
            data=[]
        )

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get(f"/api/audit/workflow/{mock_workflow_id}")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_get_entries_service_unavailable(self, mock_workflow_id):
        """Should return 503 when service is unavailable."""
        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=None,
        ):
            response = client.get(f"/api/audit/workflow/{mock_workflow_id}")

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()


class TestVerifyWorkflowChain:
    """Tests for GET /audit/workflow/{workflow_id}/verify."""

    def test_verify_chain_valid(self, mock_audit_service, mock_workflow_id):
        """Should return valid chain verification result."""
        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get(f"/api/audit/workflow/{mock_workflow_id}/verify")

        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == str(mock_workflow_id)
        assert data["is_valid"] is True
        assert data["entries_checked"] == 5
        assert data["first_invalid_entry"] is None
        assert "verified_at" in data

    def test_verify_chain_invalid(self, mock_audit_service, mock_workflow_id, mock_entry_id):
        """Should report invalid chain with details."""
        # Override with invalid verification
        invalid_result = ChainVerificationResult(
            is_valid=False,
            entries_checked=3,
            first_invalid_entry=mock_entry_id,
            error_message="Hash mismatch at sequence 3",
            verified_at=datetime.now(timezone.utc),
        )
        mock_audit_service.verify_workflow.return_value = invalid_result

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get(f"/api/audit/workflow/{mock_workflow_id}/verify")

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert data["entries_checked"] == 3
        assert data["first_invalid_entry"] == str(mock_entry_id)
        assert "Hash mismatch" in data["error_message"]

    def test_verify_chain_service_unavailable(self, mock_workflow_id):
        """Should return 503 when service is unavailable."""
        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=None,
        ):
            response = client.get(f"/api/audit/workflow/{mock_workflow_id}/verify")

        assert response.status_code == 503


class TestGetWorkflowSummary:
    """Tests for GET /audit/workflow/{workflow_id}/summary."""

    def test_get_summary_success(self, mock_audit_service, mock_workflow_id, mock_audit_entry_row):
        """Should return workflow summary with aggregated metrics."""
        # Add multiple entries for aggregation
        entries = [
            mock_audit_entry_row,
            {
                **mock_audit_entry_row,
                "entry_id": str(uuid4()),
                "sequence_number": 2,
                "agent_name": "causal_impact",
                "agent_tier": 2,
                "confidence_score": 0.88,
                "duration_ms": 350,
            },
        ]
        mock_audit_service.db.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = MagicMock(
            data=entries
        )

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get(f"/api/audit/workflow/{mock_workflow_id}/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == str(mock_workflow_id)
        assert data["total_entries"] == 2
        assert "agents_involved" in data
        assert "tiers_involved" in data
        assert "chain_verified" in data
        assert "total_duration_ms" in data
        assert "avg_confidence_score" in data

    def test_get_summary_not_found(self, mock_audit_service, mock_workflow_id):
        """Should return 404 for non-existent workflow."""
        mock_audit_service.db.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = MagicMock(
            data=[]
        )

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get(f"/api/audit/workflow/{mock_workflow_id}/summary")

        assert response.status_code == 404
        # Handle different response structures
        data = response.json()
        error_text = data.get("detail", str(data)).lower()
        assert "not found" in error_text

    def test_get_summary_includes_brand(self, mock_audit_service, mock_workflow_id, mock_audit_entry_row):
        """Should include brand from first entry."""
        mock_audit_entry_row["brand"] = "Fabhalta"
        mock_audit_service.db.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = MagicMock(
            data=[mock_audit_entry_row]
        )

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get(f"/api/audit/workflow/{mock_workflow_id}/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["brand"] == "Fabhalta"


class TestGetRecentWorkflows:
    """Tests for GET /audit/recent."""

    def test_get_recent_success(self, mock_audit_service, mock_recent_workflow_row):
        """Should return recent workflows."""
        mock_audit_service.db.rpc.return_value.execute.return_value = MagicMock(
            data=[mock_recent_workflow_row]
        )

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get("/api/audit/recent")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        workflow = data[0]
        assert "workflow_id" in workflow
        assert "started_at" in workflow
        assert "entry_count" in workflow
        assert "first_agent" in workflow
        assert "last_agent" in workflow

    def test_get_recent_with_limit(self, mock_audit_service, mock_recent_workflow_row):
        """Should respect limit parameter."""
        mock_audit_service.db.rpc.return_value.execute.return_value = MagicMock(
            data=[mock_recent_workflow_row]
        )

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get("/api/audit/recent", params={"limit": 5})

        assert response.status_code == 200
        # Verify RPC was called with limit param
        mock_audit_service.db.rpc.assert_called()

    def test_get_recent_with_brand_filter(self, mock_audit_service, mock_recent_workflow_row):
        """Should filter by brand."""
        mock_audit_service.db.rpc.return_value.execute.return_value = MagicMock(
            data=[mock_recent_workflow_row]
        )

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get("/api/audit/recent", params={"brand": "Kisqali"})

        assert response.status_code == 200
        # RPC should have been called with brand filter
        call_args = mock_audit_service.db.rpc.call_args
        assert call_args[0][1]["p_brand"] == "Kisqali"

    def test_get_recent_with_agent_filter(self, mock_audit_service, mock_recent_workflow_row):
        """Should filter by agent name."""
        mock_audit_service.db.rpc.return_value.execute.return_value = MagicMock(
            data=[mock_recent_workflow_row]
        )

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get("/api/audit/recent", params={"agent_name": "causal_impact"})

        assert response.status_code == 200
        call_args = mock_audit_service.db.rpc.call_args
        assert call_args[0][1]["p_agent_name"] == "causal_impact"

    def test_get_recent_fallback_on_rpc_failure(self, mock_audit_service, mock_audit_entry_row):
        """Should use fallback when RPC fails."""
        # Make RPC fail
        mock_audit_service.db.rpc.return_value.execute.side_effect = Exception("RPC unavailable")

        # Setup fallback query chain
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_eq = MagicMock()
        mock_order = MagicMock()
        mock_limit = MagicMock()

        mock_audit_service.db.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        mock_eq.order.return_value = mock_order
        mock_order.limit.return_value = mock_limit
        mock_limit.execute.return_value = MagicMock(data=[mock_audit_entry_row])

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get("/api/audit/recent")

        assert response.status_code == 200

    def test_get_recent_service_unavailable(self):
        """Should return 503 when service is unavailable."""
        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=None,
        ):
            response = client.get("/api/audit/recent")

        assert response.status_code == 503

    def test_get_recent_empty_list(self, mock_audit_service):
        """Should return empty list when no recent workflows."""
        mock_audit_service.db.rpc.return_value.execute.return_value = MagicMock(data=[])
        # Also setup fallback to return empty
        mock_audit_service.db.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[]
        )

        with patch(
            "src.api.routes.audit.get_audit_service",
            return_value=mock_audit_service,
        ):
            response = client.get("/api/audit/recent")

        assert response.status_code == 200
        data = response.json()
        assert data == []
