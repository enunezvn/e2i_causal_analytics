"""Tests for ExpertReviewGate.

Version: 4.3
Tests the expert review gate workflow decisions.
"""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

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
        mock_repo.get_dag_approval = AsyncMock(
            return_value={
                "review_id": "rev-123",
                "approved_at": "2024-01-01T00:00:00Z",
                "valid_until": (date.today() + timedelta(days=60)).isoformat(),
                "reviewer_name": "Dr. Expert",
            }
        )

        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.PROCEED
        assert result.is_approved is True
        assert result.review_id == "rev-123"
        assert result.requires_action is False

    @pytest.mark.asyncio
    async def test_check_approval_expiring_dag(self, gate, mock_repo):
        """Test check_approval returns RENEWAL_REQUIRED for expiring DAG."""
        mock_repo.get_dag_approval = AsyncMock(
            return_value={
                "review_id": "rev-123",
                "approved_at": "2024-01-01T00:00:00Z",
                "valid_until": (date.today() + timedelta(days=7)).isoformat(),
                "reviewer_name": "Dr. Expert",
            }
        )

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
        mock_repo.get_reviews_for_dag = AsyncMock(
            return_value=[
                {"review_id": "rev-pending", "approval_status": "pending"},
            ]
        )

        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert result.is_approved is False
        assert result.review_id == "rev-pending"
        assert result.requires_action is True

    @pytest.mark.asyncio
    async def test_check_approval_auto_create_review(self, mock_repo):
        """Test check_approval auto-creates review for new DAG.

        M-reach1: auto_create_review now defaults False (fail-closed); a caller that
        wants the producer must opt in explicitly, so this test constructs the gate
        with auto_create_review=True rather than relying on the old default.
        """
        gate = ExpertReviewGate(repository=mock_repo, auto_create_review=True)
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
        mock_repo.get_dag_approval = AsyncMock(
            return_value={
                "review_id": "rev-123",
                "valid_until": (date.today() + timedelta(days=60)).isoformat(),
            }
        )

        await gate.check_approval("abc123", brand="TestBrand")

        mock_repo.get_dag_approval.assert_called_with("abc123", "TestBrand")

    @pytest.mark.asyncio
    async def test_pending_lookup_passes_brand(self, gate, mock_repo):
        """check_approval must forward the brand to get_reviews_for_dag so a
        pending review from a different brand cannot gate this analysis."""
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=[])
        mock_repo.create_review = AsyncMock(return_value="rev-new")

        await gate.check_approval("abc123", brand="BrandX", requester_id="user-1")

        # brand must be forwarded to the pending-review lookup.
        _, kwargs = mock_repo.get_reviews_for_dag.call_args
        assert kwargs.get("brand") == "BrandX"


class _CapturingRepo:
    """Fake repo capturing create_review kwargs (NO live PostgREST insert).

    The async repo speaks PostgREST (HTTP) so BEGIN..ROLLBACK is unavailable and
    a real insert would pollute the live ``expert_reviews`` table. This fake
    records the exact kwargs the gate passes to ``create_review`` so we can assert
    the review_type WITHOUT touching the DB. The DB cast-disproof (a read-only
    ``SELECT 'initial_dag'::expert_review_type`` ERRORs against the live enum,
    confirmed faithfully against the droplet) covers the other half: with
    ``initial_dag`` the real INSERT would fail the enum cast.
    """

    def __init__(self) -> None:
        self.create_kwargs: dict | None = None

    async def get_dag_approval(self, dag_hash, brand=None):
        return None

    async def get_reviews_for_dag(self, dag_hash, include_expired=False, brand=None):
        return []

    async def create_review(self, **kwargs):
        self.create_kwargs = kwargs
        return "rev-captured"


class TestAutoCreateReviewTypeEnum:
    """C1: the auto-created review MUST use a VALID expert_review_type member.

    The live ``expert_review_type`` ENUM (010 :53-58) is
    {dag_approval, methodology_review, quarterly_audit, ad_hoc_validation} —
    there is NO ``initial_dag`` member. While ``auto_create_review`` defaulted
    False (R5) the call site never ran, so passing ``review_type="initial_dag"``
    was a LATENT bug. F2 wiring sets ``auto_create_review=True``, which ACTIVATES
    it: the INSERT would fail the enum cast -> create_review returns None -> the
    gate falls through to BLOCKED instead of PENDING_REVIEW (a silent hard-block,
    zero rows). The fix is ``review_type="dag_approval"``.
    """

    @pytest.mark.asyncio
    async def test_auto_create_uses_valid_dag_approval_enum(self):
        repo = _CapturingRepo()
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)

        result = await gate.check_approval(
            "deadbeef",
            requester_id="causal_impact_agent",
            treatment="email_frequency",
            outcome="trx",
            analysis_context="confidence=0.60, gate=review",
        )

        assert repo.create_kwargs is not None, "create_review was not called"
        # The load-bearing assertion: a VALID enum member, NOT the latent
        # 'initial_dag' that would fail the Postgres enum cast.
        assert repo.create_kwargs["review_type"] == "dag_approval"
        assert repo.create_kwargs["review_type"] != "initial_dag"
        # And the gate returns PENDING_REVIEW (not BLOCKED) on a successful create.
        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert result.review_id == "rev-captured"


class TestSanitizeDagStructure:
    """The persisted snapshot must be a bounded, JSON-serializable subset of the
    rich in-state CausalGraph — render/assessment keys only, no dag_dot blob."""

    def test_keeps_render_keys_and_coerces_edge_tuples(self):
        from src.causal_engine.expert_review_gate import sanitize_dag_structure

        rich_graph = {
            "nodes": ["T", "O", "C"],
            "edges": [("T", "O"), ("C", "T"), ("C", "O")],  # tuples from nx
            "treatment_nodes": ["T"],
            "outcome_nodes": ["O"],
            "adjustment_sets": [["C"]],
            "dag_dot": "digraph { ... }",  # must be dropped (redundant blob)
            "confidence": 0.85,
            "augmented_edges": [("X", "O")],
            "discovery_gate_decision": "augment",
            "dag_version_hash": "deadbeef",
        }

        structure = sanitize_dag_structure(rich_graph)

        assert structure is not None
        assert structure["nodes"] == ["T", "O", "C"]
        assert structure["edges"] == [["T", "O"], ["C", "T"], ["C", "O"]]
        assert structure["treatment_nodes"] == ["T"]
        assert structure["outcome_nodes"] == ["O"]
        assert structure["adjustment_sets"] == [["C"]]
        assert structure["augmented_edges"] == [["X", "O"]]
        assert structure["discovery_gate_decision"] == "augment"
        assert structure["dag_version_hash"] == "deadbeef"
        assert "dag_dot" not in structure
        # Must survive JSON round-trip (it is persisted as JSONB).
        import json

        assert json.loads(json.dumps(structure)) == structure

    def test_returns_none_without_nodes(self):
        from src.causal_engine.expert_review_gate import sanitize_dag_structure

        assert sanitize_dag_structure(None) is None
        assert sanitize_dag_structure({}) is None
        assert sanitize_dag_structure({"edges": [["A", "B"]]}) is None


class TestAutoCreateDagStructureCapture:
    """Mig 097: the auto-created review row must carry the sanitized DAG
    snapshot and the refutation validation-row ids, so the review UI can render
    the graph and link its evidence."""

    @pytest.mark.asyncio
    async def test_auto_create_forwards_structure_and_validation_ids(self):
        repo = _CapturingRepo()
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        rich_graph = {
            "nodes": ["T", "O", "C"],
            "edges": [("T", "O"), ("C", "T"), ("C", "O")],
            "treatment_nodes": ["T"],
            "outcome_nodes": ["O"],
            "dag_dot": "digraph { ... }",
        }

        result = await gate.check_approval(
            "deadbeef",
            requester_id="causal_impact_agent",
            treatment="email_frequency",
            outcome="trx",
            dag_structure=rich_graph,
            related_validation_ids=["val-1", "val-2"],
        )

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert repo.create_kwargs is not None
        structure = repo.create_kwargs["dag_structure"]
        assert structure["edges"] == [["T", "O"], ["C", "T"], ["C", "O"]]
        assert "dag_dot" not in structure
        assert repo.create_kwargs["related_validation_ids"] == ["val-1", "val-2"]

    @pytest.mark.asyncio
    async def test_auto_create_without_structure_still_creates(self):
        """No graph in scope (defensive) -> review still created, structure None."""
        repo = _CapturingRepo()
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)

        result = await gate.check_approval("deadbeef", requester_id="agent")

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert repo.create_kwargs is not None
        assert repo.create_kwargs.get("dag_structure") is None


class _PendingRepo(_CapturingRepo):
    """Repo whose queue already holds a pending row for the DAG (pre-097 rows
    lack dag_structure_json). Captures update_dag_structure calls."""

    def __init__(self, pending_row: dict) -> None:
        super().__init__()
        self._pending_row = pending_row
        self.structure_updates: list[tuple] = []

    async def get_reviews_for_dag(self, dag_hash, include_expired=False, brand=None):
        return [self._pending_row]

    async def update_dag_structure(self, review_id, dag_structure, related_validation_ids=None):
        self.structure_updates.append((review_id, dag_structure, related_validation_ids))
        return True


class TestPendingRowStructureBackfill:
    """Backfill-on-encounter (097): the queue's pre-097 rows carry only the
    one-way hash. When the SAME DAG is re-analyzed, the gate short-circuits on
    the existing pending row — so that consult is the ONLY chance to attach the
    renderable snapshot. It must backfill a structure-less pending row and
    leave a row that already has one untouched."""

    @pytest.mark.asyncio
    async def test_backfills_structureless_pending_row(self):
        repo = _PendingRepo(
            {"review_id": "rev-old", "approval_status": "pending", "dag_structure_json": None}
        )
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        graph = {
            "nodes": ["t", "y"],
            "edges": [("t", "y")],
            "treatment_nodes": ["t"],
            "outcome_nodes": ["y"],
        }

        result = await gate.check_approval(
            "deadbeef",
            requester_id="agent",
            dag_structure=graph,
            related_validation_ids=["val-1"],
        )

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert result.review_id == "rev-old"
        assert len(repo.structure_updates) == 1
        review_id, structure, val_ids = repo.structure_updates[0]
        assert review_id == "rev-old"
        assert structure["edges"] == [["t", "y"]]
        assert val_ids == ["val-1"]
        # The short-circuit must NOT create a duplicate row.
        assert repo.create_kwargs is None

    @pytest.mark.asyncio
    async def test_row_with_structure_is_left_untouched(self):
        repo = _PendingRepo(
            {
                "review_id": "rev-has",
                "approval_status": "pending",
                "dag_structure_json": {"nodes": ["t"], "edges": []},
            }
        )
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)

        result = await gate.check_approval(
            "deadbeef",
            requester_id="agent",
            dag_structure={"nodes": ["t", "y"], "edges": [("t", "y")]},
        )

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert repo.structure_updates == []

    @pytest.mark.asyncio
    async def test_backfill_error_never_breaks_the_gate(self):
        class _BoomRepo(_PendingRepo):
            async def update_dag_structure(self, *a, **k):
                raise RuntimeError("db down")

        repo = _BoomRepo(
            {"review_id": "rev-old", "approval_status": "pending", "dag_structure_json": None}
        )
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)

        result = await gate.check_approval(
            "deadbeef",
            requester_id="agent",
            dag_structure={"nodes": ["t", "y"], "edges": [("t", "y")]},
        )

        assert result.decision == ReviewGateDecision.PENDING_REVIEW


class TestExpertReviewGateCanProceed:
    """Test can_proceed convenience method."""

    @pytest.fixture
    def mock_repo(self):
        """Create mock ExpertReviewRepository."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_can_proceed_approved(self, mock_repo):
        """Test can_proceed returns True for approved DAG."""
        mock_repo.get_dag_approval = AsyncMock(
            return_value={
                "review_id": "rev-123",
                "valid_until": (date.today() + timedelta(days=60)).isoformat(),
            }
        )

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123")

        assert result is True

    @pytest.mark.asyncio
    async def test_can_proceed_expiring_allowed(self, mock_repo):
        """Test can_proceed with expiring approval allowed."""
        mock_repo.get_dag_approval = AsyncMock(
            return_value={
                "review_id": "rev-123",
                "valid_until": (date.today() + timedelta(days=7)).isoformat(),
            }
        )

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123", allow_expiring=True)

        assert result is True

    @pytest.mark.asyncio
    async def test_can_proceed_expiring_not_allowed(self, mock_repo):
        """Test can_proceed with expiring approval not allowed."""
        mock_repo.get_dag_approval = AsyncMock(
            return_value={
                "review_id": "rev-123",
                "valid_until": (date.today() + timedelta(days=7)).isoformat(),
            }
        )

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123", allow_expiring=False)

        # RENEWAL_REQUIRED is returned, which should be False if expiring not allowed
        assert result is False

    @pytest.mark.asyncio
    async def test_can_proceed_pending_allowed(self, mock_repo):
        """Test can_proceed with pending review allowed."""
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(
            return_value=[
                {"review_id": "rev-pending", "approval_status": "pending"},
            ]
        )

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
        mock_repo.get_dag_approval = AsyncMock(
            return_value={
                "review_id": "rev-old",
            }
        )
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
        mock_repo.get_pending_reviews = AsyncMock(
            return_value=[
                {"review_id": "rev-1"},
                {"review_id": "rev-2"},
            ]
        )

        gate = ExpertReviewGate(repository=mock_repo)
        count = await gate.get_pending_review_count()

        assert count == 2

    @pytest.mark.asyncio
    async def test_get_expiring_dag_count(self, mock_repo):
        """Test get_expiring_dag_count."""
        mock_repo.get_expiring_reviews = AsyncMock(
            return_value=[
                {"review_id": "rev-1"},
            ]
        )

        gate = ExpertReviewGate(repository=mock_repo)
        count = await gate.get_expiring_dag_count(days=14)

        assert count == 1
        mock_repo.get_expiring_reviews.assert_called_with(14, None)

    @pytest.mark.asyncio
    async def test_get_gate_status_healthy(self, mock_repo):
        """Test get_gate_status for healthy gate."""
        mock_repo.get_review_summary = AsyncMock(
            return_value={
                "pending": 2,
                "approved": 10,
                "rejected": 1,
                "expired": 0,
                "expiring_soon": 1,
            }
        )

        gate = ExpertReviewGate(repository=mock_repo)
        status = await gate.get_gate_status()

        assert status["healthy"] is True
        assert status["pending_reviews"] == 2
        assert status["expiring_soon"] == 1
        assert status["total_approved"] == 10

    @pytest.mark.asyncio
    async def test_get_gate_status_unhealthy(self, mock_repo):
        """Test get_gate_status for unhealthy gate."""
        mock_repo.get_review_summary = AsyncMock(
            return_value={
                "pending": 10,  # Too many pending
                "approved": 5,
                "rejected": 0,
                "expired": 2,
                "expiring_soon": 5,  # Too many expiring
            }
        )

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
        mock_repo.get_dag_approval = AsyncMock(
            return_value={
                "review_id": "rev-123",
                "valid_until": (date.today() + timedelta(days=60)).isoformat(),
            }
        )

        result = await check_dag_approval("abc123", repository=mock_repo)

        assert result.decision == ReviewGateDecision.PROCEED
        assert result.is_approved is True

    @pytest.mark.asyncio
    async def test_standalone_function_without_repo(self):
        """Test check_dag_approval without repository."""
        result = await check_dag_approval("abc123")

        assert result.decision == ReviewGateDecision.PROCEED
        assert "bypassed" in result.message.lower()


def test_auto_create_review_defaults_false_failclosed():
    """M-reach1 (DEFER hardening): until a review-queue consumer/admin-UI exists, a
    repo-backed gate must NOT silently create orphan `pending` rows. The default is
    therefore fail-closed (False); callers that have a human-in-the-loop consumer
    opt in explicitly with auto_create_review=True."""
    from src.causal_engine.expert_review_gate import ExpertReviewGate

    gate = ExpertReviewGate()  # no repository, default flags
    assert gate.auto_create_review is False, (
        "auto_create_review must default False until the admin-UI consumer (R6-F2) exists; "
        "True would let a future repo-backed wire create pending rows no human can clear"
    )
    # Opt-in still honored:
    assert ExpertReviewGate(auto_create_review=True).auto_create_review is True
