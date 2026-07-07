"""Tests for ExpertReviewRepository.

Version: 4.3
Tests the expert review repository CRUD operations.
"""

import json
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
    async def test_create_review_recovers_existing_pending_on_unique_violation(
        self, repo, mock_client
    ):
        """M-reach1: when a concurrent INSERT loses the uq_er_pending_dag_brand race,
        create_review returns the EXISTING pending review_id, not None."""
        # The INSERT raises (simulated 23505 unique-constraint violation).
        mock_client.table.return_value.insert.return_value.execute = AsyncMock(
            side_effect=Exception(
                'duplicate key value violates unique constraint "uq_er_pending_dag_brand"'
            )
        )
        # The recovery lookup (_find_pending_review_id, brand set) finds the winner's row:
        # .select().eq().eq().eq().limit().execute()
        recovery_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-winner"}]))
        (
            mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.limit.return_value.execute
        ) = recovery_execute

        review_id = await repo.create_review(
            reviewer_id="user-1",
            review_type="dag_approval",
            dag_version_hash="abc123",
            brand="Kisqali",
        )

        assert review_id == "rev-winner"

    @pytest.mark.asyncio
    async def test_create_review_does_not_recover_on_non_unique_error(self, repo, mock_client):
        """A transient (non-unique-constraint) insert failure must NOT trigger the
        pending-row recovery — it could mask the real error behind a stale row."""
        mock_client.table.return_value.insert.return_value.execute = AsyncMock(
            side_effect=Exception("connection reset by peer")
        )
        # If recovery were (wrongly) attempted, it would find this row and return it.
        leak_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-stale"}]))
        (
            mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.limit.return_value.execute
        ) = leak_execute

        review_id = await repo.create_review(
            reviewer_id="user-1",
            review_type="dag_approval",
            dag_version_hash="abc123",
            brand="Kisqali",
        )

        # Non-unique error: do NOT recover; surface as None (logged), never "rev-stale".
        assert review_id is None

    @pytest.mark.asyncio
    async def test_create_review_returns_none_on_unique_violation_when_no_pending_found(
        self, repo, mock_client
    ):
        """A unique violation whose winner row is already gone/resolved → None."""
        mock_client.table.return_value.insert.return_value.execute = AsyncMock(
            side_effect=Exception(
                'duplicate key value violates unique constraint "uq_er_pending_dag_brand"'
            )
        )
        recovery_execute = AsyncMock(return_value=MagicMock(data=[]))
        (
            mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.limit.return_value.execute
        ) = recovery_execute

        review_id = await repo.create_review(
            reviewer_id="user-1",
            review_type="dag_approval",
            dag_version_hash="abc123",
            brand="Kisqali",
        )

        assert review_id is None

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
    async def test_create_review_persists_dag_structure_json(self, repo, mock_client):
        """The DAG snapshot rides the insert as serialized dag_structure_json so
        the review UI can render the graph under review (mig 097)."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-dag"}]))
        mock_client.table.return_value.insert.return_value.execute = mock_execute
        structure = {
            "nodes": ["T", "O", "C"],
            "edges": [["T", "O"], ["C", "T"], ["C", "O"]],
            "treatment_nodes": ["T"],
            "outcome_nodes": ["O"],
        }

        review_id = await repo.create_review(
            reviewer_id="user-1",
            review_type="dag_approval",
            dag_version_hash="abc123",
            dag_structure=structure,
        )

        assert review_id == "rev-dag"
        inserted_row = mock_client.table.return_value.insert.call_args[0][0]
        assert json.loads(inserted_row["dag_structure_json"]) == structure

    @pytest.mark.asyncio
    async def test_create_review_omits_dag_structure_when_absent(self, repo, mock_client):
        """No structure provided -> no dag_structure_json key (column stays NULL)."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-x"}]))
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        await repo.create_review(reviewer_id="user-1", review_type="dag_approval")

        inserted_row = mock_client.table.return_value.insert.call_args[0][0]
        assert "dag_structure_json" not in inserted_row

    @pytest.mark.asyncio
    async def test_update_agent_assessment_persists_json(self, repo, mock_client):
        """update_agent_assessment writes agent_assessment_json for the row."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-123"}]))
        (mock_client.table.return_value.update.return_value.eq.return_value.execute) = mock_execute
        assessment = {"items": [{"id": "conf_complete", "verdict": "supports"}]}

        ok = await repo.update_agent_assessment("rev-123", assessment)

        assert ok is True
        updated = mock_client.table.return_value.update.call_args[0][0]
        assert json.loads(updated["agent_assessment_json"]) == assessment

    @pytest.mark.asyncio
    async def test_update_agent_assessment_zero_rows_returns_false(self, repo, mock_client):
        """A zero-row update (nonexistent review) must NOT report success."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[]))
        (mock_client.table.return_value.update.return_value.eq.return_value.execute) = mock_execute

        ok = await repo.update_agent_assessment("rev-missing", {"items": []})

        assert ok is False

    @pytest.mark.asyncio
    async def test_update_agent_assessment_without_client(self):
        """No client -> False (fail-closed), never a fabricated success."""
        repo = ExpertReviewRepository()
        repo.client = None

        ok = await repo.update_agent_assessment("rev-123", {"items": []})

        assert ok is False

    @pytest.mark.asyncio
    async def test_update_dag_structure_backfills_row(self, repo, mock_client):
        """Backfill (097): a pre-097 pending row gains the snapshot + evidence
        ids when the same DAG is re-encountered."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[{"review_id": "rev-old"}]))
        (mock_client.table.return_value.update.return_value.eq.return_value.execute) = mock_execute
        structure = {"nodes": ["T", "O"], "edges": [["T", "O"]]}

        ok = await repo.update_dag_structure("rev-old", structure, related_validation_ids=["val-1"])

        assert ok is True
        updated = mock_client.table.return_value.update.call_args[0][0]
        assert json.loads(updated["dag_structure_json"]) == structure
        assert updated["related_validation_ids"] == ["val-1"]

    @pytest.mark.asyncio
    async def test_update_dag_structure_zero_rows_returns_false(self, repo, mock_client):
        mock_execute = AsyncMock(return_value=MagicMock(data=[]))
        (mock_client.table.return_value.update.return_value.eq.return_value.execute) = mock_execute

        ok = await repo.update_dag_structure("rev-missing", {"nodes": ["T"], "edges": []})

        assert ok is False

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
    async def test_submit_review_zero_rows_returns_false(self, repo, mock_client):
        """FIX B (codex HIGH): a ZERO-ROW update must return False, not True.

        Resolving a NONEXISTENT or already-resolved review_id does an update that
        matches no rows; supabase-py returns the updated rows in ``result.data``,
        so an empty ``data`` means nothing was touched. Returning True there is a
        fabricated success (the route would 200 a record it never changed).
        """
        mock_execute = AsyncMock(return_value=MagicMock(data=[]))
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.submit_review(
            review_id="does-not-exist",
            approval_status="approved",
            checklist={"confounder_check": True},
        )

        assert result is False

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
    async def test_get_reviews_for_dag_filters_by_brand(self, repo, mock_client):
        """get_reviews_for_dag must apply an .eq('brand', brand) filter when a
        brand is supplied, so cross-brand reviews are not returned."""
        reviews_data = [{"review_id": "rev-1", "dag_version_hash": "abc123", "brand": "BrandX"}]
        mock_execute = AsyncMock(return_value=MagicMock(data=reviews_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.or_.return_value = mock_query
        mock_query.eq.return_value = mock_query
        # base chain: table().select().eq(dag_hash).order() -> mock_query
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value = (
            mock_query
        )

        result = await repo.get_reviews_for_dag("abc123", brand="BrandX")

        assert len(result) == 1
        # The brand filter was applied on the post-order query object.
        mock_query.eq.assert_any_call("brand", "BrandX")

    @pytest.mark.asyncio
    async def test_get_reviews_for_dag_no_brand_no_filter(self, repo, mock_client):
        """When brand is None, no brand .eq filter is applied (back-compat)."""
        reviews_data = [{"review_id": "rev-1", "dag_version_hash": "abc123"}]
        mock_execute = AsyncMock(return_value=MagicMock(data=reviews_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.or_.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value = (
            mock_query
        )

        result = await repo.get_reviews_for_dag("abc123")

        assert len(result) == 1
        for call in mock_query.eq.call_args_list:
            assert call.args[:1] != ("brand",)

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
