"""Tests for ExpertReviewRepository.

Version: 4.3
Tests the expert review repository CRUD operations.
"""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.expert_review import ExpertReviewRepository


class TestExpertReviewRepository:
    """Test ExpertReviewRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        repo = ExpertReviewRepository()
        repo.client = mock_client
        return repo

    @pytest.mark.asyncio
    async def test_create_review(self, repo, mock_client):
        """Test creating a new expert review."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-123"}]))
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        review_id = await repo.create_review(
            reviewer_id="user-1",
            review_type="initial_dag",
            dag_version_hash="abc123",
            brand="TestBrand",
            treatment_variable="hcp_engagement",
            outcome_variable="prescription_volume",
        )

        assert review_id == "rev-123"
        mock_client.table.assert_called_with("expert_reviews")

    @pytest.mark.asyncio
    async def test_create_review_without_client(self):
        """Test create_review returns None without client."""
        repo = ExpertReviewRepository()
        repo.client = None

        result = await repo.create_review(
            reviewer_id="user-1",
            review_type="initial_dag",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_submit_review_approved(self, repo, mock_client):
        """Test submitting an approved review."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-123"}]))
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.submit_review(
            review_id="rev-123",
            approval_status="approved",
            checklist={"confounder_check": True, "edge_direction": True},
        )

        assert result is True
        mock_client.table.assert_called_with("expert_reviews")

    @pytest.mark.asyncio
    async def test_submit_review_rejected(self, repo, mock_client):
        """Test submitting a rejected review."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-123"}]))
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.submit_review(
            review_id="rev-123",
            approval_status="rejected",
            checklist={"confounder_check": False},
            concerns_raised=["Missing confounders"],
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_submit_review_invalid_status(self, repo):
        """Test submit_review rejects invalid status."""
        result = await repo.submit_review(
            review_id="rev-123",
            approval_status="invalid_status",
            checklist={},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_is_dag_approved_true(self, repo, mock_client):
        """Test is_dag_approved returns True for approved DAG."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-123"}]))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value = mock_query

        result = await repo.is_dag_approved("abc123")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_dag_approved_false(self, repo, mock_client):
        """Test is_dag_approved returns False for unapproved DAG."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[]))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value = mock_query

        result = await repo.is_dag_approved("abc123")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_dag_approved_without_client(self):
        """Test is_dag_approved returns True without client (dev mode)."""
        repo = ExpertReviewRepository()
        repo.client = None

        result = await repo.is_dag_approved("abc123")

        # Default to True in dev mode
        assert result is True

    @pytest.mark.asyncio
    async def test_get_dag_approval(self, repo, mock_client):
        """Test getting DAG approval record."""
        approval_data = {
            "review_id": "rev-123",
            "approval_status": "approved",
            "reviewer_name": "Dr. Expert",
            "valid_until": (date.today() + timedelta(days=30)).isoformat(),
        }
        mock_execute = AsyncMock(return_value=MagicMock(data=[approval_data]))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.limit.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.order.return_value = mock_query

        result = await repo.get_dag_approval("abc123")

        assert result["review_id"] == "rev-123"
        assert result["approval_status"] == "approved"

    @pytest.mark.asyncio
    async def test_get_pending_reviews(self, repo, mock_client):
        """Test getting pending reviews."""
        pending_data = [
            {"review_id": "rev-1", "approval_status": "pending"},
            {"review_id": "rev-2", "approval_status": "pending"},
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=pending_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.limit.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value = (
            mock_query
        )

        result = await repo.get_pending_reviews()

        assert len(result) == 2
        assert result[0]["review_id"] == "rev-1"

    @pytest.mark.asyncio
    async def test_get_pending_reviews_with_brand_filter(self, repo, mock_client):
        """Test getting pending reviews with brand filter."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[]))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.limit.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value = (
            mock_query
        )

        await repo.get_pending_reviews(brand="TestBrand")

        # Verify brand filter was applied
        mock_query.eq.assert_called()

    @pytest.mark.asyncio
    async def test_get_expiring_reviews(self, repo, mock_client):
        """Test getting expiring reviews."""
        expiring_data = [
            {
                "review_id": "rev-1",
                "valid_until": (date.today() + timedelta(days=7)).isoformat(),
            },
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=expiring_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_client.table.return_value.select.return_value.eq.return_value.gte.return_value.lte.return_value.order.return_value = mock_query

        result = await repo.get_expiring_reviews(days_until_expiry=14)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_reviews_for_dag(self, repo, mock_client):
        """Test getting reviews for specific DAG."""
        reviews_data = [
            {"review_id": "rev-1", "dag_version_hash": "abc123"},
            {"review_id": "rev-2", "dag_version_hash": "abc123"},
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=reviews_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.or_.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value = (
            mock_query
        )

        result = await repo.get_reviews_for_dag("abc123")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_renew_review(self, repo, mock_client):
        """Test renewing an existing review."""
        # Mock get_by_id for original review
        original_review = {
            "review_id": "rev-old",
            "dag_version_hash": "abc123",
            "brand": "TestBrand",
            "treatment_variable": "engagement",
            "outcome_variable": "conversions",
        }

        async def mock_get_by_id(review_id):
            return original_review

        repo.get_by_id = mock_get_by_id

        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-new"}]))
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        result = await repo.renew_review(
            original_review_id="rev-old",
            reviewer_id="user-2",
            reviewer_name="New Reviewer",
        )

        assert result == "rev-new"

    @pytest.mark.asyncio
    async def test_renew_review_not_found(self, repo, mock_client):
        """Test renewing non-existent review."""

        async def mock_get_by_id(review_id):
            return None

        repo.get_by_id = mock_get_by_id

        result = await repo.renew_review(
            original_review_id="non-existent",
            reviewer_id="user-2",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_review_summary(self, repo, mock_client):
        """Test getting review summary."""
        reviews_data = [
            {"approval_status": "pending", "valid_until": None},
            {
                "approval_status": "approved",
                "valid_until": (date.today() + timedelta(days=30)).isoformat(),
            },
            {
                "approval_status": "approved",
                "valid_until": (date.today() + timedelta(days=7)).isoformat(),
            },
            {"approval_status": "rejected", "valid_until": None},
            {
                "approval_status": "approved",
                "valid_until": (date.today() - timedelta(days=7)).isoformat(),
            },
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=reviews_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_client.table.return_value.select.return_value = mock_query

        result = await repo.get_review_summary()

        assert result["pending"] == 1
        assert result["rejected"] == 1
        assert result["expired"] == 1
        assert result["expiring_soon"] == 1
        # One approved (day 30), one expiring (day 7 - counts as both approved and expiring_soon)
        assert result["approved"] == 2

    @pytest.mark.asyncio
    async def test_get_review_summary_without_client(self):
        """Test get_review_summary returns defaults without client."""
        repo = ExpertReviewRepository()
        repo.client = None

        result = await repo.get_review_summary()

        assert result["pending"] == 0
        assert result["approved"] == 0
        assert result["rejected"] == 0
        assert result["expired"] == 0
        assert result["expiring_soon"] == 0


class TestExpertReviewRepositoryErrorHandling:
    """Test error handling in ExpertReviewRepository."""

    @pytest.fixture
    def repo_with_failing_client(self):
        """Create repository with client that raises exceptions."""
        repo = ExpertReviewRepository()
        client = MagicMock()

        # Make execute raise exceptions
        async def failing_execute():
            raise Exception("Database error")

        mock_query = MagicMock()
        mock_query.execute = failing_execute
        client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value = mock_query
        client.table.return_value.insert.return_value.execute = failing_execute

        repo.client = client
        return repo

    @pytest.mark.asyncio
    async def test_create_review_handles_error(self, repo_with_failing_client):
        """Test create_review handles database errors gracefully."""
        result = await repo_with_failing_client.create_review(
            reviewer_id="user-1",
            review_type="initial_dag",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_is_dag_approved_handles_error(self, repo_with_failing_client):
        """Test is_dag_approved handles database errors gracefully."""
        result = await repo_with_failing_client.is_dag_approved("abc123")

        # Should return False on error
        assert result is False
