"""Tests for ExpertReviewGate.

Version: 4.3
Tests the expert review gate workflow decisions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import date, timedelta

from src.causal_engine import (
    ExpertReviewGate,
    ReviewGateDecision,
    ReviewGateResult,
    check_dag_approval,
)


class TestExpertReviewGate:
    """Test ExpertReviewGate."""

    @pytest.fixture
    def mock_repo(self):
        """Create mock ExpertReviewRepository."""
        repo = MagicMock()
        return repo

    @pytest.fixture
    def gate(self, mock_repo):
        """Create gate with mock repository."""
        return ExpertReviewGate(repository=mock_repo)

    @pytest.mark.asyncio
    async def test_check_approval_approved_dag(self, gate, mock_repo):
        """Test check_approval returns PROCEED for approved DAG."""
        mock_repo.get_dag_approval = AsyncMock(return_value={
            "review_id": "rev-123",
            "approved_at": "2024-01-01T00:00:00Z",
            "valid_until": (date.today() + timedelta(days=60)).isoformat(),
            "reviewer_name": "Dr. Expert",
        })

        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.PROCEED
        assert result.is_approved is True
        assert result.review_id == "rev-123"
        assert result.requires_action is False

    @pytest.mark.asyncio
    async def test_check_approval_expiring_dag(self, gate, mock_repo):
        """Test check_approval returns RENEWAL_REQUIRED for expiring DAG."""
        mock_repo.get_dag_approval = AsyncMock(return_value={
            "review_id": "rev-123",
            "approved_at": "2024-01-01T00:00:00Z",
            "valid_until": (date.today() + timedelta(days=7)).isoformat(),
            "reviewer_name": "Dr. Expert",
        })

        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.RENEWAL_REQUIRED
        assert result.is_approved is True
        assert result.days_until_expiry == 7
        assert result.requires_action is True
        assert "expiring" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_approval_pending_review(self, gate, mock_repo):
        """Test check_approval returns PENDING_REVIEW for DAG with pending review."""
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=[
            {"review_id": "rev-pending", "approval_status": "pending"},
        ])

        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert result.is_approved is False
        assert result.review_id == "rev-pending"
        assert result.requires_action is True

    @pytest.mark.asyncio
    async def test_check_approval_auto_create_review(self, gate, mock_repo):
        """Test check_approval auto-creates review for new DAG."""
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=[])
        mock_repo.create_review = AsyncMock(return_value="rev-new")

        result = await gate.check_approval(
            "abc123",
            requester_id="user-1",
            treatment="engagement",
            outcome="conversions",
        )

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert result.review_id == "rev-new"
        mock_repo.create_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_approval_blocked(self, gate, mock_repo):
        """Test check_approval returns BLOCKED when no review and no auto-create."""
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=[])

        # No requester_id, so can't auto-create
        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.BLOCKED
        assert result.is_approved is False
        assert result.requires_action is True

    @pytest.mark.asyncio
    async def test_check_approval_without_repository(self):
        """Test check_approval bypasses gate without repository."""
        gate = ExpertReviewGate(repository=None)

        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.PROCEED
        assert result.is_approved is True
        assert "bypassed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_approval_with_brand_filter(self, gate, mock_repo):
        """Test check_approval passes brand filter."""
        mock_repo.get_dag_approval = AsyncMock(return_value={
            "review_id": "rev-123",
            "valid_until": (date.today() + timedelta(days=60)).isoformat(),
        })

        await gate.check_approval("abc123", brand="TestBrand")

        mock_repo.get_dag_approval.assert_called_with("abc123", "TestBrand")


class TestExpertReviewGateCanProceed:
    """Test can_proceed convenience method."""

    @pytest.fixture
    def mock_repo(self):
        """Create mock ExpertReviewRepository."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_can_proceed_approved(self, mock_repo):
        """Test can_proceed returns True for approved DAG."""
        mock_repo.get_dag_approval = AsyncMock(return_value={
            "review_id": "rev-123",
            "valid_until": (date.today() + timedelta(days=60)).isoformat(),
        })

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123")

        assert result is True

    @pytest.mark.asyncio
    async def test_can_proceed_expiring_allowed(self, mock_repo):
        """Test can_proceed with expiring approval allowed."""
        mock_repo.get_dag_approval = AsyncMock(return_value={
            "review_id": "rev-123",
            "valid_until": (date.today() + timedelta(days=7)).isoformat(),
        })

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123", allow_expiring=True)

        assert result is True

    @pytest.mark.asyncio
    async def test_can_proceed_expiring_not_allowed(self, mock_repo):
        """Test can_proceed with expiring approval not allowed."""
        mock_repo.get_dag_approval = AsyncMock(return_value={
            "review_id": "rev-123",
            "valid_until": (date.today() + timedelta(days=7)).isoformat(),
        })

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123", allow_expiring=False)

        # RENEWAL_REQUIRED is returned, which should be False if expiring not allowed
        assert result is False

    @pytest.mark.asyncio
    async def test_can_proceed_pending_allowed(self, mock_repo):
        """Test can_proceed with pending review allowed."""
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=[
            {"review_id": "rev-pending", "approval_status": "pending"},
        ])

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123", allow_pending=True)

        assert result is True

    @pytest.mark.asyncio
    async def test_can_proceed_blocked(self, mock_repo):
        """Test can_proceed returns False for blocked DAG."""
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=[])

        gate = ExpertReviewGate(repository=mock_repo, auto_create_review=False)
        result = await gate.can_proceed("abc123")

        assert result is False


class TestExpertReviewGateRenewal:
    """Test renewal functionality."""

    @pytest.fixture
    def mock_repo(self):
        """Create mock ExpertReviewRepository."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_request_renewal(self, mock_repo):
        """Test request_renewal creates renewal review."""
        mock_repo.get_dag_approval = AsyncMock(return_value={
            "review_id": "rev-old",
        })
        mock_repo.renew_review = AsyncMock(return_value="rev-new")

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.request_renewal(
            dag_hash="abc123",
            requester_id="user-1",
            requester_name="Test User",
        )

        assert result == "rev-new"
        mock_repo.renew_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_renewal_no_existing_approval(self, mock_repo):
        """Test request_renewal fails without existing approval."""
        mock_repo.get_dag_approval = AsyncMock(return_value=None)

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.request_renewal(
            dag_hash="abc123",
            requester_id="user-1",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_request_renewal_without_repository(self):
        """Test request_renewal returns None without repository."""
        gate = ExpertReviewGate(repository=None)

        result = await gate.request_renewal(
            dag_hash="abc123",
            requester_id="user-1",
        )

        assert result is None


class TestExpertReviewGateStatus:
    """Test gate status and monitoring methods."""

    @pytest.fixture
    def mock_repo(self):
        """Create mock ExpertReviewRepository."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_get_pending_review_count(self, mock_repo):
        """Test get_pending_review_count."""
        mock_repo.get_pending_reviews = AsyncMock(return_value=[
            {"review_id": "rev-1"},
            {"review_id": "rev-2"},
        ])

        gate = ExpertReviewGate(repository=mock_repo)
        count = await gate.get_pending_review_count()

        assert count == 2

    @pytest.mark.asyncio
    async def test_get_expiring_dag_count(self, mock_repo):
        """Test get_expiring_dag_count."""
        mock_repo.get_expiring_reviews = AsyncMock(return_value=[
            {"review_id": "rev-1"},
        ])

        gate = ExpertReviewGate(repository=mock_repo)
        count = await gate.get_expiring_dag_count(days=14)

        assert count == 1
        mock_repo.get_expiring_reviews.assert_called_with(14, None)

    @pytest.mark.asyncio
    async def test_get_gate_status_healthy(self, mock_repo):
        """Test get_gate_status for healthy gate."""
        mock_repo.get_review_summary = AsyncMock(return_value={
            "pending": 2,
            "approved": 10,
            "rejected": 1,
            "expired": 0,
            "expiring_soon": 1,
        })

        gate = ExpertReviewGate(repository=mock_repo)
        status = await gate.get_gate_status()

        assert status["healthy"] is True
        assert status["pending_reviews"] == 2
        assert status["expiring_soon"] == 1
        assert status["total_approved"] == 10

    @pytest.mark.asyncio
    async def test_get_gate_status_unhealthy(self, mock_repo):
        """Test get_gate_status for unhealthy gate."""
        mock_repo.get_review_summary = AsyncMock(return_value={
            "pending": 10,  # Too many pending
            "approved": 5,
            "rejected": 0,
            "expired": 2,
            "expiring_soon": 5,  # Too many expiring
        })

        gate = ExpertReviewGate(repository=mock_repo)
        status = await gate.get_gate_status()

        assert status["healthy"] is False
        assert "attention" in status["message"].lower()

    @pytest.mark.asyncio
    async def test_get_gate_status_without_repository(self):
        """Test get_gate_status without repository."""
        gate = ExpertReviewGate(repository=None)
        status = await gate.get_gate_status()

        assert status["healthy"] is True
        assert "not configured" in status["message"].lower()


class TestReviewGateResult:
    """Test ReviewGateResult dataclass."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = ReviewGateResult(
            decision=ReviewGateDecision.PROCEED,
            dag_hash="abc123",
            is_approved=True,
            review_id="rev-123",
            message="Test message",
        )

        d = result.to_dict()

        assert d["decision"] == "proceed"
        assert d["dag_hash"] == "abc123"
        assert d["is_approved"] is True
        assert d["review_id"] == "rev-123"

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields populated."""
        result = ReviewGateResult(
            decision=ReviewGateDecision.RENEWAL_REQUIRED,
            dag_hash="abc123",
            is_approved=True,
            review_id="rev-123",
            approved_at="2024-01-01T00:00:00Z",
            valid_until="2024-04-01",
            days_until_expiry=7,
            reviewer_name="Dr. Expert",
            message="Expiring soon",
            requires_action=True,
        )

        d = result.to_dict()

        assert d["days_until_expiry"] == 7
        assert d["reviewer_name"] == "Dr. Expert"
        assert d["requires_action"] is True


class TestCheckDagApprovalFunction:
    """Test standalone check_dag_approval function."""

    @pytest.mark.asyncio
    async def test_standalone_function(self):
        """Test check_dag_approval standalone function."""
        mock_repo = MagicMock()
        mock_repo.get_dag_approval = AsyncMock(return_value={
            "review_id": "rev-123",
            "valid_until": (date.today() + timedelta(days=60)).isoformat(),
        })

        result = await check_dag_approval("abc123", repository=mock_repo)

        assert result.decision == ReviewGateDecision.PROCEED
        assert result.is_approved is True

    @pytest.mark.asyncio
    async def test_standalone_function_without_repo(self):
        """Test check_dag_approval without repository."""
        result = await check_dag_approval("abc123")

        assert result.decision == ReviewGateDecision.PROCEED
        assert "bypassed" in result.message.lower()
