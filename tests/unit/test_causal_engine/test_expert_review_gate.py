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
from src.causal_engine.dag_hash import compute_adjustment_set_hash

#: What a run whose structure carries NO adjustment sets computes -- the
#: canonical EMPTY set sha256("[]"), a different fact from NULL (UNKNOWN). The
#: backfill rows below state it so their version identity MATCHES the run's and
#: the consult reaches the backfill short-circuit; a row left at NULL is the
#: separate ADVANCE case (codex round 2), pinned in its own test below.
_EMPTY_ADJUSTMENT = compute_adjustment_set_hash([])


class TestExpertReviewGate:
    """Test ExpertReviewGate."""

    @pytest.fixture
    def mock_repo(self):
        """Create mock ExpertReviewRepository."""
        repo = MagicMock()
        repo.get_reviews_for_dag = AsyncMock(return_value=[])
        # #1991 debt 3: check_approval reads the ESTIMAND's history (mig 140)
        # and records structure versions (mig 141). check_rejection still reads
        # get_reviews_for_dag -- a rejection is recorded against a hash.
        repo.get_reviews_for_estimand = AsyncMock(return_value=[])
        repo.append_version = AsyncMock(return_value=True)
        # #1991 debt 3 (codex round-1): before appending, the gate reads the
        # review's LAST recorded version and appends only when this run's
        # (hash, adjustment-set) pair differs from it. None = no timeline yet,
        # which is every mint in this class. A MagicMock here would make the
        # read UNREADABLE, which the gate deliberately treats as "already
        # recorded" -- so the default must be an explicit empty answer.
        repo.get_latest_version = AsyncMock(return_value=None)
        # NOT_RECORDED is now the RECORD-ONLY branch when the review already
        # carries this run's pair (the timeline needs version 1, the review does
        # not). A bare MagicMock is not awaitable, so the call would raise inside
        # the gate rather than be recorded.
        repo.record_version = AsyncMock(return_value=True)
        return repo

    @pytest.fixture
    def gate(self, mock_repo):
        """Create gate with mock repository."""
        return ExpertReviewGate(repository=mock_repo)

    @pytest.mark.asyncio
    async def test_check_approval_approved_dag(self, gate, mock_repo):
        """Test check_approval returns PROCEED for approved DAG.

        The approval is read from the estimand's history and is an approval OF
        this hash (#1991 debt 3) -- an approved row for a DIFFERENT structure
        does not clear this one.
        """
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-123",
                    "approval_status": "approved",
                    "dag_version_hash": "abc123",
                    "approved_at": "2024-01-01T00:00:00Z",
                    "valid_until": (date.today() + timedelta(days=60)).isoformat(),
                    "reviewer_name": "Dr. Expert",
                }
            ]
        )

        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.PROCEED
        assert result.is_approved is True
        assert result.review_id == "rev-123"
        assert result.requires_action is False

    @pytest.mark.asyncio
    async def test_check_approval_expiring_dag(self, gate, mock_repo):
        """Test check_approval returns RENEWAL_REQUIRED for expiring DAG."""
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-123",
                    "approval_status": "approved",
                    "dag_version_hash": "abc123",
                    "approved_at": "2024-01-01T00:00:00Z",
                    "valid_until": (date.today() + timedelta(days=7)).isoformat(),
                    "reviewer_name": "Dr. Expert",
                }
            ]
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
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-pending",
                    "approval_status": "pending",
                    "dag_version_hash": "abc123",
                },
            ]
        )

        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert result.is_approved is False
        assert result.review_id == "rev-pending"
        assert result.requires_action is True
        # Unchanged structure -> nothing appended to the version timeline.
        mock_repo.append_version.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_check_approval_auto_create_review(self, mock_repo):
        """Test check_approval auto-creates review for new DAG.

        M-reach1: auto_create_review now defaults False (fail-closed); a caller that
        wants the producer must opt in explicitly, so this test constructs the gate
        with auto_create_review=True rather than relying on the old default.
        """
        gate = ExpertReviewGate(repository=mock_repo, auto_create_review=True)
        mock_repo.get_reviews_for_estimand = AsyncMock(return_value=[])
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
        # Version 1 of the new review's timeline (mig 141).
        mock_repo.append_version.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_approval_blocked(self, gate, mock_repo):
        """Test check_approval returns BLOCKED when no review and no auto-create."""
        mock_repo.get_reviews_for_estimand = AsyncMock(return_value=[])

        # No requester_id, so can't auto-create
        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.BLOCKED
        assert result.is_approved is False
        assert result.requires_action is True

    @pytest.mark.asyncio
    async def test_check_approval_without_repository_is_unavailable(self):
        """#1971: a gate with no repository cannot check anything, so it must say
        so -- never PROCEED / is_approved=True. That bypass is exactly what a
        prod ServiceConnectionError degraded to (#1969)."""
        gate = ExpertReviewGate(repository=None)

        result = await gate.check_approval("abc123")

        assert result.decision == ReviewGateDecision.UNAVAILABLE
        assert result.decision.value == "unavailable"
        assert result.is_approved is False
        assert result.review_id is None
        assert result.requires_action is True
        assert "could not be consulted" in result.message.lower()
        assert "bypass" not in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_approval_keys_the_history_read_on_the_estimand(self, gate, mock_repo):
        """#1991 debt 3: the brand is no longer a FILTER on a hash-keyed read --
        it is part of the estimand key the history read is made with, so a
        review of another brand's estimand is a different row entirely."""
        mock_repo.get_reviews_for_estimand = AsyncMock(return_value=[])

        await gate.check_approval("abc123", brand="TestBrand", treatment="T", outcome="Y")

        mock_repo.get_reviews_for_estimand.assert_awaited_once_with(
            "testbrand:t:y", include_expired=True
        )
        # The hash-keyed active-approval query is not consulted any more.
        mock_repo.get_dag_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_another_brands_review_cannot_gate_this_analysis(self, gate, mock_repo):
        """Same guarantee the old brand filter gave, now structural: BrandX and
        BrandY are different estimands, so BrandY's pending row is not in
        BrandX's history at all."""
        mock_repo.get_reviews_for_estimand = AsyncMock(
            side_effect=lambda key, include_expired=True: (
                [{"review_id": "rev-other", "approval_status": "pending"}]
                if key == "brandy:t:y"
                else []
            )
        )
        mock_repo.create_review = AsyncMock(return_value="rev-new")
        gate = ExpertReviewGate(repository=mock_repo, auto_create_review=True)

        result = await gate.check_approval(
            "abc123", brand="BrandX", treatment="T", outcome="Y", requester_id="user-1"
        )

        assert result.review_id == "rev-new"
        mock_repo.get_reviews_for_estimand.assert_awaited_once_with(
            "brandx:t:y", include_expired=True
        )


class _CapturingRepo:
    """Fake repo capturing create_review kwargs (NO live PostgREST insert).

    The async repo speaks PostgREST (HTTP) so BEGIN..ROLLBACK is unavailable and
    a real insert would pollute the live ``expert_reviews`` table. This fake
    records the exact kwargs the gate passes to ``create_review`` so we can assert
    the review_type WITHOUT touching the DB. When this fix landed, the DB
    cast-disproof (a read-only ``SELECT 'initial_dag'::expert_review_type``
    ERRORed against the live enum, confirmed faithfully against the droplet)
    covered the other half. Since migration 152 ``initial_dag`` IS a member --
    the Lane B structural-AUTHOR review type -- so the reason the gate must not
    write it is now semantic: its loader would read a gate consult back as an
    authored prior.
    """

    def __init__(self) -> None:
        self.create_kwargs: dict | None = None
        self.appended: list[tuple] = []
        self.recorded: list[tuple] = []

    async def get_dag_approval(self, dag_hash, brand=None):
        return None

    async def get_reviews_for_dag(self, dag_hash, include_expired=False, brand=None):
        return []

    async def get_reviews_for_estimand(self, estimand_key, include_expired=True):
        return []

    async def create_review(self, **kwargs):
        self.create_kwargs = kwargs
        return "rev-captured"

    async def append_version(self, review_id, **kwargs):
        self.appended.append((review_id, kwargs.get("dag_version_hash")))
        return True

    async def record_version(self, review_id, **kwargs):
        """The timeline write alone, for a review already on this run's pair."""
        self.recorded.append((review_id, kwargs.get("dag_version_hash")))
        return True

    async def get_latest_version(self, review_id):
        """No timeline yet (#1991 debt 3). Without this the gate's version read
        raises AttributeError, which it swallows as UNKNOWN -- so this mint would
        log a warning and take the OUTAGE path instead of the normal one."""
        return None


class TestAutoCreateReviewTypeEnum:
    """C1: the auto-created review MUST use a VALID expert_review_type member.

    The ``expert_review_type`` ENUM (010 :53-58) is
    {dag_approval, methodology_review, quarterly_audit, ad_hoc_validation};
    ``initial_dag`` was added by migration 152 for the Lane B structural-AUTHOR
    review and is NOT the gate's type. While ``auto_create_review`` defaulted
    False (R5) the call site never ran, so passing ``review_type="initial_dag"``
    was a LATENT bug. F2 wiring sets ``auto_create_review=True``, which ACTIVATES
    it: before migration 152 the INSERT failed the enum cast -> create_review
    returned None -> the gate fell through to BLOCKED instead of PENDING_REVIEW
    (a silent hard-block, zero rows); since 152 the value casts, but it is the
    structural-author type and the loader would adopt the consult as an authored
    prior. Either way the gate writes ``review_type="dag_approval"``.
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
        # The load-bearing assertion: the gate's own type, NOT 'initial_dag'
        # (pre-152: failed the enum cast; post-152: the structural-author type).
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

    def __init__(self, pending_row: dict, latest_version: dict | str = "__own_pair__") -> None:
        super().__init__()
        self._pending_row = pending_row
        self.structure_updates: list[tuple] = []
        self.advances: list[tuple] = []
        self.append_kwargs: list[dict] = []
        # The review's TIMELINE. Default: one row carrying the pending row's own
        # pair, so the gate answers SAME and the consult reaches the backfill
        # short-circuit these tests exist for. An EMPTY timeline is a different
        # case (NOT_RECORDED: the gate records version 1 there), reachable by
        # passing latest_version=None.
        if latest_version == "__own_pair__":
            latest_version = {
                "version_id": "v1",
                "review_id": pending_row.get("review_id"),
                "dag_version_hash": pending_row.get("dag_version_hash"),
                "adjustment_set_hash": pending_row.get("adjustment_set_hash"),
                "dag_structure_json": None,
            }
        self._latest_version = latest_version

    async def get_reviews_for_dag(self, dag_hash, include_expired=False, brand=None):
        return [self._pending_row]

    async def get_reviews_for_estimand(self, estimand_key, include_expired=True):
        return [self._pending_row]

    async def update_dag_structure(self, review_id, dag_structure, related_validation_ids=None):
        self.structure_updates.append((review_id, dag_structure, related_validation_ids))
        return True

    async def advance_review(self, review_id, **kwargs):
        """The gate advances a review whose own version identity differs from
        the run's. Recorded so the backfill tests can assert they did NOT take
        that path."""
        self.advances.append((review_id, kwargs))
        return True

    async def append_version(self, review_id, **kwargs):
        self.appended.append((review_id, kwargs.get("dag_version_hash")))
        self.append_kwargs.append({"review_id": review_id, **kwargs})
        return True

    async def record_version(self, review_id, **kwargs):
        self.recorded.append((review_id, kwargs.get("dag_version_hash")))
        return True

    async def get_latest_version(self, review_id):
        return dict(self._latest_version) if self._latest_version is not None else None


class TestPendingRowStructureBackfill:
    """Backfill-on-encounter (097): the queue's pre-097 rows carry only the
    one-way hash. When the SAME DAG is re-analyzed, the gate short-circuits on
    the existing pending row — so that consult is the ONLY chance to attach the
    renderable snapshot. It must backfill a structure-less pending row and
    leave a row that already has one untouched."""

    @pytest.mark.asyncio
    async def test_backfills_structureless_pending_row(self):
        repo = _PendingRepo(
            {
                "review_id": "rev-old",
                "approval_status": "pending",
                # Pending on THIS structure: the backfill path. A DIFFERING hash
                # appends a version instead (#1991 debt 3).
                "dag_version_hash": "deadbeef",
                "adjustment_set_hash": _EMPTY_ADJUSTMENT,
                "dag_structure_json": None,
            }
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
                "dag_version_hash": "deadbeef",
                "adjustment_set_hash": _EMPTY_ADJUSTMENT,
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
            {
                "review_id": "rev-old",
                "approval_status": "pending",
                "dag_version_hash": "deadbeef",
                "adjustment_set_hash": _EMPTY_ADJUSTMENT,
                "dag_structure_json": None,
            }
        )
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)

        result = await gate.check_approval(
            "deadbeef",
            requester_id="agent",
            dag_structure={"nodes": ["t", "y"], "edges": [("t", "y")]},
        )

        assert result.decision == ReviewGateDecision.PENDING_REVIEW

    @pytest.mark.asyncio
    async def test_a_row_whose_adjustment_half_is_unknown_is_versioned_not_backfilled(self):
        """A pre-142 row carries NULL -- UNKNOWN, not "no adjustment set". The run
        computed one, so the row's version identity genuinely differs AND its
        timeline is empty: both decisions fire, which is ``append_version``.

        That is not a lost backfill. The append records the structure on the
        timeline and its advance writes the snapshot AND the adjustment half, so
        it does strictly more than ``update_dag_structure``. It has to happen, or
        the row never learns its second half and every guard keyed on it
        (resolution, the assessment persist, the compare-and-set) keeps matching
        half an identity -- the defect codex round 2 found.
        """
        repo = _PendingRepo(
            {
                "review_id": "rev-legacy",
                "approval_status": "pending",
                "dag_version_hash": "deadbeef",
                "adjustment_set_hash": None,
                "dag_structure_json": None,
            },
            # No timeline either: a pre-141 row the backfill skipped.
            latest_version=None,
        )
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)

        result = await gate.check_approval(
            "deadbeef",
            requester_id="agent",
            dag_structure={"nodes": ["t", "y"], "edges": [("t", "y")]},
            related_validation_ids=["val-1"],
        )

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert repo.structure_updates == [], "the append supersedes the backfill here"
        # Both decisions -> append_version, which advances as part of the same
        # call; a separate advance would be a second write of the same move.
        assert repo.appended == [("rev-legacy", "deadbeef")]
        assert repo.advances == []
        kwargs = repo.append_kwargs[0]
        assert kwargs["adjustment_set_hash"] == _EMPTY_ADJUSTMENT
        # The compare-and-set expects the UNKNOWN it read, matched IS NULL.
        assert kwargs["expected_current_adjustment_hash"] is None
        assert kwargs["expected_current_hash"] == "deadbeef"
        # The renderable snapshot still lands -- the backfill's whole purpose.
        assert kwargs["dag_structure"]["edges"] == [["t", "y"]]
        assert repo.create_kwargs is None


class TestExpertReviewGateCanProceed:
    """Test can_proceed convenience method."""

    @pytest.fixture
    def mock_repo(self):
        """Create mock ExpertReviewRepository."""
        repo = MagicMock()
        # #1991 debt 3: the gate reads the review's last recorded version before
        # appending. A bare MagicMock is not awaitable, which the gate swallows as
        # UNKNOWN -- the OUTAGE path, not the one these tests mean to exercise.
        repo.get_latest_version = AsyncMock(return_value=None)
        # NOT_RECORDED is now the RECORD-ONLY branch when the review already
        # carries this run's pair (the timeline needs version 1, the review does
        # not). A bare MagicMock is not awaitable, so the call would raise inside
        # the gate rather than be recorded.
        repo.record_version = AsyncMock(return_value=True)
        return repo

    @pytest.mark.asyncio
    async def test_can_proceed_approved(self, mock_repo):
        """Test can_proceed returns True for approved DAG."""
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-123",
                    "approval_status": "approved",
                    "dag_version_hash": "abc123",
                    "valid_until": (date.today() + timedelta(days=60)).isoformat(),
                }
            ]
        )

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123")

        assert result is True

    @pytest.mark.asyncio
    async def test_can_proceed_expiring_allowed(self, mock_repo):
        """Test can_proceed with expiring approval allowed."""
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-123",
                    "approval_status": "approved",
                    "dag_version_hash": "abc123",
                    "valid_until": (date.today() + timedelta(days=7)).isoformat(),
                }
            ]
        )

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123", allow_expiring=True)

        assert result is True

    @pytest.mark.asyncio
    async def test_can_proceed_expiring_not_allowed(self, mock_repo):
        """Test can_proceed with expiring approval not allowed."""
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-123",
                    "approval_status": "approved",
                    "dag_version_hash": "abc123",
                    "valid_until": (date.today() + timedelta(days=7)).isoformat(),
                }
            ]
        )

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123", allow_expiring=False)

        # RENEWAL_REQUIRED is returned, which should be False if expiring not allowed
        assert result is False

    @pytest.mark.asyncio
    async def test_can_proceed_pending_allowed(self, mock_repo):
        """Test can_proceed with pending review allowed."""
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-pending",
                    "approval_status": "pending",
                    "dag_version_hash": "abc123",
                },
            ]
        )

        gate = ExpertReviewGate(repository=mock_repo)
        result = await gate.can_proceed("abc123", allow_pending=True)

        assert result is True

    @pytest.mark.asyncio
    async def test_can_proceed_blocked(self, mock_repo):
        """Test can_proceed returns False for blocked DAG."""
        mock_repo.get_reviews_for_estimand = AsyncMock(return_value=[])

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
        assert d["rejection_reason"] is None

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
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-123",
                    "approval_status": "approved",
                    "dag_version_hash": "abc123",
                    "valid_until": (date.today() + timedelta(days=60)).isoformat(),
                }
            ]
        )

        result = await check_dag_approval("abc123", repository=mock_repo)

        assert result.decision == ReviewGateDecision.PROCEED
        assert result.is_approved is True

    @pytest.mark.asyncio
    async def test_standalone_function_without_repo(self):
        """#1971: no repository -> unavailable (never a PROCEED for a DAG nobody
        looked at)."""
        result = await check_dag_approval("abc123", repository=None)

        assert result.decision == ReviewGateDecision.UNAVAILABLE
        assert result.is_approved is False
        assert "could not be consulted" in result.message.lower()


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


class TestRejectedVerdictIsDurable:
    """#1970: a rejected review must not be re-queued as a fresh pending row."""

    @pytest.fixture
    def mock_repo(self):
        repo = MagicMock()
        repo.append_version = AsyncMock(return_value=True)
        # #1991 debt 3: the gate reads the review's last recorded version before
        # appending. A bare MagicMock is not awaitable, which the gate swallows as
        # UNKNOWN -- the OUTAGE path, not the one these tests mean to exercise.
        repo.get_latest_version = AsyncMock(return_value=None)
        # NOT_RECORDED is now the RECORD-ONLY branch when the review already
        # carries this run's pair (the timeline needs version 1, the review does
        # not). A bare MagicMock is not awaitable, so the call would raise inside
        # the gate rather than be recorded.
        repo.record_version = AsyncMock(return_value=True)
        return repo

    @pytest.mark.asyncio
    async def test_rejected_latest_row_is_rejected_without_auto_create(self, mock_repo):
        """#1971 refines the #1970 verdict label: a human rejection is its own
        decision (``rejected``), distinct from ``blocked`` (no approval and no
        review could be queued), so API consumers never have to guess which."""
        gate = ExpertReviewGate(repository=mock_repo, auto_create_review=True)
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-rejected",
                    "approval_status": "rejected",
                    # A rejection is recorded against the hash it was given on,
                    # and only covers THAT structure (#1991 debt 3).
                    "dag_version_hash": "abc123",
                    "reviewer_name": "Dr. No",
                    "concerns_raised": ["formulary_status is a collider"],
                },
                {
                    "review_id": "rev-old-pending-resolved",
                    "approval_status": "approved",
                    "valid_until": "2020-01-01",
                },
            ]
        )
        mock_repo.create_review = AsyncMock(return_value="rev-should-not-exist")

        result = await gate.check_approval("abc123", requester_id="user-1")

        assert result.decision == ReviewGateDecision.REJECTED
        assert result.decision.value == "rejected"
        assert result.is_approved is False
        assert result.review_id == "rev-rejected"
        assert result.reviewer_name == "Dr. No"
        assert result.rejection_reason == "formulary_status is a collider"
        assert "rejected" in result.message.lower()
        mock_repo.create_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_newer_pending_row_still_wins_over_older_rejection(self, mock_repo):
        """Ordering: rows come back created_at DESC; a pending row newer than a
        rejection means a reviewer re-opened the structure."""
        gate = ExpertReviewGate(repository=mock_repo, auto_create_review=True)
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-reopened",
                    "approval_status": "pending",
                    "dag_version_hash": "abc123",
                },
                {"review_id": "rev-rejected", "approval_status": "rejected"},
            ]
        )
        mock_repo.create_review = AsyncMock(return_value="rev-new")

        result = await gate.check_approval("abc123", requester_id="user-1")

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert result.review_id == "rev-reopened"
        mock_repo.create_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_older_rejection_below_a_newer_non_pending_row_does_not_block(self, mock_repo):
        """Only the MOST RECENT verdict is durable; an old rejection under a
        newer expired approval falls through to auto-create as before."""
        gate = ExpertReviewGate(repository=mock_repo, auto_create_review=True)
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-expired",
                    "approval_status": "approved",
                    "dag_version_hash": "abc123",
                    "valid_until": "2020-01-01",
                },
                {"review_id": "rev-rejected", "approval_status": "rejected"},
            ]
        )
        mock_repo.create_review = AsyncMock(return_value="rev-new")

        result = await gate.check_approval("abc123", requester_id="user-1")

        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert result.review_id == "rev-new"
        mock_repo.create_review.assert_called_once()


class TestCheckRejection:
    """#1971: the READ-ONLY rejection probe consulted on every refutation band.

    ``check_rejection`` answers one question -- "did a human reject this
    structure?" -- with the same precedence as ``check_approval`` (an active
    approval wins, then a pending row, then the latest verdict) and NEVER
    creates a review row, so a PROCEED band can ask it without queueing
    anything.
    """

    @pytest.fixture
    def mock_repo(self):
        repo = MagicMock()
        repo.get_reviews_for_dag = AsyncMock(return_value=[])
        repo.get_reviews_for_estimand = AsyncMock(return_value=[])
        repo.append_version = AsyncMock(return_value=True)
        repo.create_review = AsyncMock(return_value="rev-should-not-exist")
        # #1991 debt 3: the gate reads the review's last recorded version before
        # appending. A bare MagicMock is not awaitable, which the gate swallows as
        # UNKNOWN -- the OUTAGE path, not the one these tests mean to exercise.
        repo.get_latest_version = AsyncMock(return_value=None)
        # NOT_RECORDED is now the RECORD-ONLY branch when the review already
        # carries this run's pair (the timeline needs version 1, the review does
        # not). A bare MagicMock is not awaitable, so the call would raise inside
        # the gate rather than be recorded.
        repo.record_version = AsyncMock(return_value=True)
        return repo

    def _rejected(self, **extra):
        row = {
            "review_id": "rev-rejected",
            "approval_status": "rejected",
            # A rejection covers the hash it was recorded against.
            "dag_version_hash": "abc123",
            "reviewer_name": "Dr. No",
            "concerns_raised": ["formulary_status is a collider"],
        }
        row.update(extra)
        return row

    @pytest.mark.asyncio
    async def test_no_repository_cannot_tell_and_says_none(self):
        assert await ExpertReviewGate(repository=None).check_rejection("abc123") is None

    @pytest.mark.asyncio
    async def test_latest_rejected_row_is_reported(self, mock_repo):
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=[self._rejected()])

        result = await ExpertReviewGate(repository=mock_repo).check_rejection(
            "abc123", brand="Kisqali"
        )

        assert result is not None
        assert result.decision == ReviewGateDecision.REJECTED
        assert result.is_approved is False
        assert result.review_id == "rev-rejected"
        assert result.reviewer_name == "Dr. No"
        assert result.rejection_reason == "formulary_status is a collider"
        mock_repo.create_review.assert_not_called()
        mock_repo.get_reviews_for_dag.assert_awaited_once_with(
            "abc123", include_expired=True, brand="Kisqali"
        )

    @pytest.mark.asyncio
    async def test_halt_message_names_the_reviewer_only_when_one_was_recorded(self, mock_repo):
        """Codex iter-2 HIGH F1: the halt reason names whoever ``reviewer_name``
        holds. A resolution now writes the RESOLVER's name (NULL when unknown),
        so a None must read as no one -- never a requester, never a placeholder
        -- and a recorded name must be named (positive control)."""
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=[self._rejected(reviewer_name=None)])
        anonymous = await ExpertReviewGate(repository=mock_repo).check_rejection("abc123")
        assert anonymous is not None
        assert anonymous.reviewer_name is None
        assert anonymous.message == (
            "DAG structure was rejected by expert review: "
            "formulary_status is a collider; not re-queued"
        )

        mock_repo.get_reviews_for_dag = AsyncMock(return_value=[self._rejected()])
        named = await ExpertReviewGate(repository=mock_repo).check_rejection("abc123")
        assert named is not None
        assert named.message == (
            "DAG structure was rejected by expert review by Dr. No: "
            "formulary_status is a collider; not re-queued"
        )

    @pytest.mark.asyncio
    async def test_newer_active_approval_wins_over_an_older_rejection(self, mock_repo):
        """Chronology (rows newest first): approval after rejection -> cleared."""
        mock_repo.get_dag_approval = AsyncMock(
            return_value={"review_id": "rev-approved", "valid_until": "2099-01-01"}
        )
        mock_repo.get_reviews_for_dag = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-approved",
                    "approval_status": "approved",
                    "valid_until": "2099-01-01",
                },
                self._rejected(),
            ]
        )

        assert await ExpertReviewGate(repository=mock_repo).check_rejection("abc123") is None

    @pytest.mark.asyncio
    async def test_newer_rejection_is_not_masked_by_an_older_active_approval(self, mock_repo):
        """codex iter-3 HIGH: approve A (90-day validity), renew as B, reject B
        while A is still unexpired. get_dag_approval still returns A, but the
        most recent adjudication of this structure is a rejection -- the
        probe must say so, and never clear the structure on A."""
        mock_repo.get_dag_approval = AsyncMock(
            return_value={"review_id": "rev-a", "valid_until": "2099-01-01"}
        )
        mock_repo.get_reviews_for_dag = AsyncMock(
            return_value=[
                self._rejected(review_id="rev-b"),
                {"review_id": "rev-a", "approval_status": "approved", "valid_until": "2099-01-01"},
            ]
        )

        result = await ExpertReviewGate(repository=mock_repo).check_rejection("abc123")

        assert result is not None
        assert result.decision == ReviewGateDecision.REJECTED
        assert result.review_id == "rev-b"
        # Chronology needs the WHOLE history, expired approvals included.
        mock_repo.get_reviews_for_dag.assert_awaited_once_with(
            "abc123", include_expired=True, brand=None
        )

    @pytest.mark.asyncio
    async def test_reopened_after_rejection_is_not_cleared_by_an_older_approval(self, mock_repo):
        """[pending C, rejected B, approved A(active)]: the rejection superseded
        A; C re-opened the structure. check_rejection says "not rejected"
        (re-opened), and check_approval must report the pending re-review --
        not PROCEED on A."""
        rows = [
            {"review_id": "rev-c", "approval_status": "pending", "dag_version_hash": "abc123"},
            self._rejected(review_id="rev-b"),
            {
                "review_id": "rev-a",
                "approval_status": "approved",
                "dag_version_hash": "abc123",
                "valid_until": "2099-01-01",
            },
        ]
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=rows)
        mock_repo.get_reviews_for_estimand = AsyncMock(return_value=rows)
        gate = ExpertReviewGate(repository=mock_repo)

        assert await gate.check_rejection("abc123") is None
        result = await gate.check_approval("abc123")
        assert result.decision == ReviewGateDecision.PENDING_REVIEW
        assert result.review_id == "rev-c"
        assert result.is_approved is False

    @pytest.mark.asyncio
    async def test_pending_row_means_reopened_not_rejected(self, mock_repo):
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(
            return_value=[
                {"review_id": "rev-reopened", "approval_status": "pending"},
                self._rejected(),
            ]
        )

        assert await ExpertReviewGate(repository=mock_repo).check_rejection("abc123") is None

    @pytest.mark.asyncio
    async def test_pending_row_tied_with_the_rejection_is_not_a_reopen(self, mock_repo):
        """Migration 134 reads a pending row with the SAME created_at as the
        rejection as NOT newer (strict >). The probe must read the tie the same
        way whatever order the repository returns it in (lane 1, Task 3b); a
        genuinely newer pending row still reopens."""
        ts = "2026-09-08T12:00:00+00:00"
        rejected = self._rejected(created_at=ts)
        tie = {
            "review_id": "rev-tie",
            "approval_status": "pending",
            "dag_version_hash": "abc123",
            "created_at": ts,
        }
        for rows in ([tie, rejected], [rejected, tie]):
            mock_repo.get_reviews_for_dag = AsyncMock(return_value=rows)
            mock_repo.get_reviews_for_estimand = AsyncMock(return_value=rows)
            result = await ExpertReviewGate(repository=mock_repo).check_rejection("abc123")
            assert result is not None and result.decision == ReviewGateDecision.REJECTED
            # Both readers, one rule: check_approval must not read the tied
            # pending row as PENDING_REVIEW while check_rejection says REJECTED.
            approval = await ExpertReviewGate(repository=mock_repo).check_approval("abc123")
            assert approval.decision == ReviewGateDecision.REJECTED

        newer = [
            {
                "review_id": "rev-new",
                "approval_status": "pending",
                "dag_version_hash": "abc123",
                "created_at": "2026-09-09T00:00:00+00:00",
            },
            rejected,
        ]
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=newer)
        mock_repo.get_reviews_for_estimand = AsyncMock(return_value=newer)
        assert await ExpertReviewGate(repository=mock_repo).check_rejection("abc123") is None
        # The moved durable-rejection block must not swallow a genuinely newer
        # pending row: ``reopened`` skips it and the pending branch answers.
        approval = await ExpertReviewGate(repository=mock_repo).check_approval("abc123")
        assert approval.decision == ReviewGateDecision.PENDING_REVIEW

    @pytest.mark.asyncio
    async def test_no_rows_is_not_a_rejection(self, mock_repo):
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(return_value=[])

        assert await ExpertReviewGate(repository=mock_repo).check_rejection("abc123") is None
        mock_repo.create_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_older_rejection_under_a_newer_expired_approval_is_not_durable(self, mock_repo):
        """Same rule as check_approval: only the MOST RECENT verdict counts."""
        mock_repo.get_dag_approval = AsyncMock(return_value=None)
        mock_repo.get_reviews_for_dag = AsyncMock(
            return_value=[
                {"review_id": "rev-expired", "approval_status": "approved"},
                self._rejected(),
            ]
        )

        assert await ExpertReviewGate(repository=mock_repo).check_rejection("abc123") is None


class TestRejectionReason:
    """The reviewer's stated reason is read from the row, bounded, never invented."""

    def test_concerns_raised_wins(self):
        from src.causal_engine.expert_review_gate import rejection_reason_from_row

        row = {
            "concerns_raised": ["a collider", "missing tier"],
            "comments_json": {"note": "ignored"},
        }
        assert rejection_reason_from_row(row) == "a collider; missing tier"

    def test_comments_dict_then_conditions(self):
        from src.causal_engine.expert_review_gate import rejection_reason_from_row

        assert (
            rejection_reason_from_row({"comments_json": {"note": "bad edge"}}) == "note: bad edge"
        )
        assert rejection_reason_from_row({"comments_json": "plain text"}) == "plain text"
        assert rejection_reason_from_row({"conditions": "needs tier"}) == "needs tier"

    def test_nothing_recorded_is_none_not_a_placeholder(self):
        from src.causal_engine.expert_review_gate import rejection_reason_from_row

        assert rejection_reason_from_row({}) is None
        assert rejection_reason_from_row({"concerns_raised": [], "comments_json": None}) is None

    def test_reason_is_bounded(self):
        from src.causal_engine.expert_review_gate import rejection_reason_from_row

        reason = rejection_reason_from_row({"conditions": "x" * 1000})
        assert reason is not None
        assert len(reason) <= 300
        assert reason.endswith("...")


class TestUnavailableNeverProceeds:
    @pytest.mark.asyncio
    async def test_can_proceed_is_false_without_repository(self):
        assert await ExpertReviewGate(repository=None).can_proceed("abc123") is False
        assert (
            await ExpertReviewGate(repository=None).can_proceed("abc123", allow_pending=True)
            is False
        )


class TestApprovalPrecedenceIsChronological:
    """codex iter-3 HIGH: check_approval must apply the same chronology as
    check_rejection -- the most recent adjudication wins; an older active
    approval never masks a newer rejection; a newer pending row re-opens."""

    @pytest.fixture
    def mock_repo(self):
        repo = MagicMock()
        repo.get_reviews_for_dag = AsyncMock(return_value=[])
        repo.get_reviews_for_estimand = AsyncMock(return_value=[])
        repo.append_version = AsyncMock(return_value=True)
        repo.create_review = AsyncMock(return_value="rev-should-not-exist")
        # #1991 debt 3: the gate reads the review's last recorded version before
        # appending. A bare MagicMock is not awaitable, which the gate swallows as
        # UNKNOWN -- the OUTAGE path, not the one these tests mean to exercise.
        repo.get_latest_version = AsyncMock(return_value=None)
        # NOT_RECORDED is now the RECORD-ONLY branch when the review already
        # carries this run's pair (the timeline needs version 1, the review does
        # not). A bare MagicMock is not awaitable, so the call would raise inside
        # the gate rather than be recorded.
        repo.record_version = AsyncMock(return_value=True)
        return repo

    @pytest.mark.asyncio
    async def test_newer_rejection_beats_older_active_approval(self, mock_repo):
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-b",
                    "approval_status": "rejected",
                    "dag_version_hash": "abc123",
                    "reviewer_name": "Dr. No",
                    "concerns_raised": ["renewal found a collider"],
                },
                {
                    "review_id": "rev-a",
                    "approval_status": "approved",
                    "dag_version_hash": "abc123",
                    "valid_until": "2099-01-01",
                },
            ]
        )
        gate = ExpertReviewGate(repository=mock_repo, auto_create_review=True)

        result = await gate.check_approval("abc123", requester_id="user-1")

        assert result.decision == ReviewGateDecision.REJECTED
        assert result.is_approved is False
        assert result.review_id == "rev-b"
        assert result.reviewer_name == "Dr. No"
        mock_repo.create_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_newer_active_approval_still_beats_older_rejection(self, mock_repo):
        """#1970 direction, kept."""
        mock_repo.get_reviews_for_estimand = AsyncMock(
            return_value=[
                {
                    "review_id": "rev-a",
                    "approval_status": "approved",
                    "dag_version_hash": "abc123",
                    "valid_until": "2099-01-01",
                },
                {"review_id": "rev-old", "approval_status": "rejected"},
            ]
        )
        gate = ExpertReviewGate(repository=mock_repo, auto_create_review=True)

        result = await gate.check_approval("abc123", requester_id="user-1")

        assert result.decision == ReviewGateDecision.PROCEED
        assert result.review_id == "rev-a"
        mock_repo.create_review.assert_not_called()
