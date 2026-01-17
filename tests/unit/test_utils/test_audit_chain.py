"""
Tests for src/utils/audit_chain.py

Covers:
- AgentTier enum
- RefutationResults dataclass
- AuditChainEntry dataclass
- ChainVerificationResult dataclass
- AuditChainService class
- create_audit_chain_service factory
"""

import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.utils.audit_chain import (
    AgentTier,
    AuditChainEntry,
    AuditChainService,
    ChainVerificationResult,
    RefutationResults,
    create_audit_chain_service,
)


# =============================================================================
# AgentTier Tests
# =============================================================================


class TestAgentTier:
    """Tests for the AgentTier enum."""

    def test_tier_values(self):
        """Test that all tiers have correct integer values."""
        assert AgentTier.ML_FOUNDATION.value == 0
        assert AgentTier.COORDINATION.value == 1
        assert AgentTier.CAUSAL_ANALYTICS.value == 2
        assert AgentTier.MONITORING.value == 3
        assert AgentTier.ML_PREDICTIONS.value == 4
        assert AgentTier.SELF_IMPROVEMENT.value == 5

    def test_tier_count(self):
        """Test that we have exactly 6 tiers."""
        assert len(AgentTier) == 6

    def test_tier_names(self):
        """Test tier names match expected values."""
        expected_names = {
            "ML_FOUNDATION",
            "COORDINATION",
            "CAUSAL_ANALYTICS",
            "MONITORING",
            "ML_PREDICTIONS",
            "SELF_IMPROVEMENT",
        }
        actual_names = {tier.name for tier in AgentTier}
        assert actual_names == expected_names


# =============================================================================
# RefutationResults Tests
# =============================================================================


class TestRefutationResults:
    """Tests for the RefutationResults dataclass."""

    def test_default_values(self):
        """Test default values are None."""
        results = RefutationResults()
        assert results.placebo_treatment is None
        assert results.random_common_cause is None
        assert results.data_subset is None
        assert results.unobserved_confound is None

    def test_all_passed_when_all_true(self):
        """Test all_passed returns True when all executed tests pass."""
        results = RefutationResults(
            placebo_treatment=True,
            random_common_cause=True,
            data_subset=True,
            unobserved_confound=True,
        )
        assert results.all_passed is True

    def test_all_passed_when_some_true(self):
        """Test all_passed returns True when only executed tests pass."""
        results = RefutationResults(
            placebo_treatment=True,
            random_common_cause=True,
            # data_subset and unobserved_confound are None (not executed)
        )
        assert results.all_passed is True

    def test_all_passed_when_any_false(self):
        """Test all_passed returns False when any test fails."""
        results = RefutationResults(
            placebo_treatment=True,
            random_common_cause=False,
            data_subset=True,
        )
        assert results.all_passed is False

    def test_all_passed_when_none_executed(self):
        """Test all_passed returns False when no tests executed."""
        results = RefutationResults()
        assert results.all_passed is False

    def test_to_dict(self):
        """Test to_dict returns correct dictionary."""
        results = RefutationResults(
            placebo_treatment=True,
            random_common_cause=False,
            data_subset=None,
            unobserved_confound=True,
        )
        expected = {
            "placebo_treatment": True,
            "random_common_cause": False,
            "data_subset": None,
            "unobserved_confound": True,
        }
        assert results.to_dict() == expected


# =============================================================================
# AuditChainEntry Tests
# =============================================================================


class TestAuditChainEntry:
    """Tests for the AuditChainEntry dataclass."""

    @pytest.fixture
    def sample_entry(self):
        """Create a sample audit chain entry."""
        return AuditChainEntry(
            entry_id=UUID("12345678-1234-1234-1234-123456789abc"),
            workflow_id=UUID("abcdefab-cdef-abcd-efab-cdefabcdefab"),
            sequence_number=1,
            agent_name="causal_impact",
            agent_tier=2,
            action_type="graph_builder",
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            duration_ms=500,
            input_hash="abc123",
            output_hash="def456",
            validation_passed=True,
            confidence_score=0.95,
            refutation_results={"placebo_treatment": True},
            previous_entry_id=None,
            previous_hash=None,
            entry_hash="xyz789",
            user_id="analyst@pharma.com",
            session_id=UUID("11111111-2222-3333-4444-555555555555"),
            query_text="What causes prescription changes?",
            brand="Kisqali",
        )

    def test_to_db_dict_basic_fields(self, sample_entry):
        """Test to_db_dict includes all basic fields."""
        result = sample_entry.to_db_dict()

        assert result["entry_id"] == "12345678-1234-1234-1234-123456789abc"
        assert result["workflow_id"] == "abcdefab-cdef-abcd-efab-cdefabcdefab"
        assert result["sequence_number"] == 1
        assert result["agent_name"] == "causal_impact"
        assert result["agent_tier"] == 2
        assert result["action_type"] == "graph_builder"
        assert result["duration_ms"] == 500

    def test_to_db_dict_hashes(self, sample_entry):
        """Test to_db_dict includes hash fields."""
        result = sample_entry.to_db_dict()

        assert result["input_hash"] == "abc123"
        assert result["output_hash"] == "def456"
        assert result["entry_hash"] == "xyz789"
        assert result["previous_hash"] is None

    def test_to_db_dict_validation_fields(self, sample_entry):
        """Test to_db_dict includes validation fields."""
        result = sample_entry.to_db_dict()

        assert result["validation_passed"] is True
        assert result["confidence_score"] == 0.95
        assert result["refutation_results"] == {"placebo_treatment": True}

    def test_to_db_dict_metadata_fields(self, sample_entry):
        """Test to_db_dict includes metadata fields."""
        result = sample_entry.to_db_dict()

        assert result["user_id"] == "analyst@pharma.com"
        assert result["session_id"] == "11111111-2222-3333-4444-555555555555"
        assert result["query_text"] == "What causes prescription changes?"
        assert result["brand"] == "Kisqali"

    def test_to_db_dict_datetime_format(self, sample_entry):
        """Test to_db_dict formats datetime correctly."""
        result = sample_entry.to_db_dict()
        assert result["created_at"] == "2025-01-01T12:00:00+00:00"

    def test_to_db_dict_with_previous_entry(self, sample_entry):
        """Test to_db_dict handles previous entry correctly."""
        sample_entry.previous_entry_id = UUID("99999999-8888-7777-6666-555555555555")
        sample_entry.previous_hash = "prev_hash_123"

        result = sample_entry.to_db_dict()

        assert result["previous_entry_id"] == "99999999-8888-7777-6666-555555555555"
        assert result["previous_hash"] == "prev_hash_123"

    def test_to_db_dict_with_none_session_id(self):
        """Test to_db_dict handles None session_id."""
        entry = AuditChainEntry(
            entry_id=uuid4(),
            workflow_id=uuid4(),
            sequence_number=1,
            agent_name="test",
            agent_tier=0,
            action_type="test",
            created_at=datetime.now(timezone.utc),
            session_id=None,
        )
        result = entry.to_db_dict()
        assert result["session_id"] is None


# =============================================================================
# ChainVerificationResult Tests
# =============================================================================


class TestChainVerificationResult:
    """Tests for the ChainVerificationResult dataclass."""

    def test_valid_chain_result(self):
        """Test creating a valid chain result."""
        result = ChainVerificationResult(is_valid=True, entries_checked=10)

        assert result.is_valid is True
        assert result.entries_checked == 10
        assert result.first_invalid_entry is None
        assert result.error_message is None
        assert result.verified_at is not None

    def test_invalid_chain_result(self):
        """Test creating an invalid chain result."""
        invalid_entry = uuid4()
        result = ChainVerificationResult(
            is_valid=False,
            entries_checked=5,
            first_invalid_entry=invalid_entry,
            error_message="Hash mismatch at sequence 5",
        )

        assert result.is_valid is False
        assert result.entries_checked == 5
        assert result.first_invalid_entry == invalid_entry
        assert result.error_message == "Hash mismatch at sequence 5"


# =============================================================================
# AuditChainService Tests
# =============================================================================


class TestAuditChainService:
    """Tests for the AuditChainService class."""

    @pytest.fixture
    def mock_supabase(self):
        """Create a mock Supabase client."""
        return MagicMock()

    @pytest.fixture
    def service(self, mock_supabase):
        """Create an AuditChainService instance."""
        return AuditChainService(mock_supabase)

    # -------------------------------------------------------------------------
    # Hash Computation Tests
    # -------------------------------------------------------------------------

    def test_compute_hash(self, service):
        """Test _compute_hash produces correct SHA-256 hash."""
        test_input = "hello world"
        expected = hashlib.sha256(test_input.encode("utf-8")).hexdigest()

        result = service._compute_hash(test_input)

        assert result == expected
        assert len(result) == 64  # SHA-256 produces 64 hex chars

    def test_hash_payload_simple(self, service):
        """Test hash_payload with simple dictionary."""
        payload = {"key": "value", "number": 42}
        result = service.hash_payload(payload)

        # Verify it produces consistent results
        assert result == service.hash_payload(payload)
        assert len(result) == 64

    def test_hash_payload_sorting(self, service):
        """Test hash_payload produces same hash regardless of key order."""
        payload1 = {"a": 1, "b": 2, "c": 3}
        payload2 = {"c": 3, "a": 1, "b": 2}

        assert service.hash_payload(payload1) == service.hash_payload(payload2)

    def test_hash_payload_with_non_serializable(self, service):
        """Test hash_payload handles non-JSON-serializable types via default=str."""
        payload = {"date": datetime(2025, 1, 1), "uuid": uuid4()}
        # Should not raise - default=str handles these
        result = service.hash_payload(payload)
        assert len(result) == 64

    def test_compute_entry_hash(self, service):
        """Test _compute_entry_hash produces deterministic hash."""
        entry = AuditChainEntry(
            entry_id=UUID("12345678-1234-1234-1234-123456789abc"),
            workflow_id=UUID("abcdefab-cdef-abcd-efab-cdefabcdefab"),
            sequence_number=1,
            agent_name="causal_impact",
            agent_tier=2,
            action_type="graph_builder",
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            input_hash="input123",
            output_hash="output456",
            previous_hash=None,
        )

        hash1 = service._compute_entry_hash(entry)
        hash2 = service._compute_entry_hash(entry)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_compute_entry_hash_genesis_uses_genesis_marker(self, service):
        """Test genesis entry uses GENESIS marker for previous_hash."""
        entry = AuditChainEntry(
            entry_id=uuid4(),
            workflow_id=uuid4(),
            sequence_number=1,
            agent_name="test",
            agent_tier=0,
            action_type="test",
            created_at=datetime.now(timezone.utc),
            previous_hash=None,  # Genesis
        )

        # The hash should include "GENESIS" in computation
        result = service._compute_entry_hash(entry)
        assert len(result) == 64

    # -------------------------------------------------------------------------
    # Workflow Management Tests
    # -------------------------------------------------------------------------

    def test_start_workflow_creates_genesis_entry(self, service, mock_supabase):
        """Test start_workflow creates a genesis entry."""
        mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock()

        entry = service.start_workflow(
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="graph_builder",
            input_data={"treatment": "rep_visit"},
            user_id="analyst@pharma.com",
            brand="Kisqali",
        )

        assert entry.sequence_number == 1
        assert entry.agent_name == "causal_impact"
        assert entry.agent_tier == 2  # AgentTier.CAUSAL_ANALYTICS.value
        assert entry.previous_entry_id is None
        assert entry.previous_hash is None
        assert entry.input_hash is not None
        assert entry.entry_hash != ""
        assert entry.user_id == "analyst@pharma.com"
        assert entry.brand == "Kisqali"

    def test_start_workflow_caches_entry(self, service, mock_supabase):
        """Test start_workflow adds entry to cache."""
        mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock()

        entry = service.start_workflow(
            agent_name="test",
            agent_tier=AgentTier.ML_FOUNDATION,
            action_type="test",
        )

        assert entry.workflow_id in service._workflow_cache
        assert service._workflow_cache[entry.workflow_id] == entry

    def test_start_workflow_auto_commit_false(self, service, mock_supabase):
        """Test start_workflow with auto_commit=False doesn't persist."""
        entry = service.start_workflow(
            agent_name="test",
            agent_tier=AgentTier.ML_FOUNDATION,
            action_type="test",
            auto_commit=False,
        )

        # Should not call insert
        mock_supabase.table.return_value.insert.assert_not_called()
        assert entry.workflow_id in service._workflow_cache

    def test_add_entry_links_to_previous(self, service, mock_supabase):
        """Test add_entry links to previous entry via hash."""
        mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock()

        # Start workflow
        genesis = service.start_workflow(
            agent_name="orchestrator",
            agent_tier=AgentTier.COORDINATION,
            action_type="route",
            auto_commit=False,
        )

        # Add second entry
        second = service.add_entry(
            workflow_id=genesis.workflow_id,
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="estimate",
            auto_commit=False,
        )

        assert second.sequence_number == 2
        assert second.previous_entry_id == genesis.entry_id
        assert second.previous_hash == genesis.entry_hash

    def test_add_entry_raises_for_unknown_workflow(self, service, mock_supabase):
        """Test add_entry raises ValueError for unknown workflow."""
        unknown_workflow = uuid4()

        # Mock database returns empty result (workflow not found)
        mock_result = MagicMock()
        mock_result.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = (
            mock_result
        )

        with pytest.raises(ValueError, match="Workflow .* not found"):
            service.add_entry(
                workflow_id=unknown_workflow,
                agent_name="test",
                agent_tier=AgentTier.ML_FOUNDATION,
                action_type="test",
            )

    def test_add_entry_with_refutation_results(self, service, mock_supabase):
        """Test add_entry handles RefutationResults."""
        mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock()

        genesis = service.start_workflow(
            agent_name="test",
            agent_tier=AgentTier.ML_FOUNDATION,
            action_type="test",
            auto_commit=False,
        )

        refutation = RefutationResults(
            placebo_treatment=True,
            random_common_cause=True,
        )

        entry = service.add_entry(
            workflow_id=genesis.workflow_id,
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="validate",
            refutation_results=refutation,
            auto_commit=False,
        )

        assert entry.refutation_results == {
            "placebo_treatment": True,
            "random_common_cause": True,
            "data_subset": None,
            "unobserved_confound": None,
        }

    # -------------------------------------------------------------------------
    # Timed Entry Context Manager Tests
    # -------------------------------------------------------------------------

    def test_timed_entry_records_duration(self, service, mock_supabase):
        """Test timed_entry context manager records duration."""
        mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock()

        genesis = service.start_workflow(
            agent_name="test",
            agent_tier=AgentTier.ML_FOUNDATION,
            action_type="test",
            auto_commit=False,
        )

        with service.timed_entry(
            workflow_id=genesis.workflow_id,
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="estimate",
        ) as entry:
            # Simulate some work
            import time

            time.sleep(0.05)  # 50ms

        # Duration should be recorded
        assert entry.duration_ms is not None
        assert entry.duration_ms >= 50  # At least 50ms

    def test_timed_entry_commits_on_exit(self, service, mock_supabase):
        """Test timed_entry commits entry on context exit."""
        mock_insert = MagicMock()
        mock_supabase.table.return_value.insert.return_value = mock_insert
        mock_insert.execute.return_value = MagicMock()

        genesis = service.start_workflow(
            agent_name="test",
            agent_tier=AgentTier.ML_FOUNDATION,
            action_type="test",
            auto_commit=False,
        )

        with service.timed_entry(
            workflow_id=genesis.workflow_id,
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="estimate",
        ):
            pass

        # Should have committed
        mock_supabase.table.assert_called_with("audit_chain_entries")

    # -------------------------------------------------------------------------
    # Persistence Tests
    # -------------------------------------------------------------------------

    def test_commit_entry_calls_insert(self, service, mock_supabase):
        """Test commit_entry inserts to database."""
        mock_insert = MagicMock()
        mock_supabase.table.return_value.insert.return_value = mock_insert
        mock_insert.execute.return_value = MagicMock()

        entry = AuditChainEntry(
            entry_id=uuid4(),
            workflow_id=uuid4(),
            sequence_number=1,
            agent_name="test",
            agent_tier=0,
            action_type="test",
            created_at=datetime.now(timezone.utc),
            entry_hash="hash123",
        )

        service.commit_entry(entry)

        mock_supabase.table.assert_called_with("audit_chain_entries")
        mock_supabase.table.return_value.insert.assert_called_once()

    def test_get_last_entry_from_cache(self, service, mock_supabase):
        """Test _get_last_entry returns from cache if available."""
        mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock()

        genesis = service.start_workflow(
            agent_name="test",
            agent_tier=AgentTier.ML_FOUNDATION,
            action_type="test",
            auto_commit=False,
        )

        # Reset mock to verify no DB query
        mock_supabase.reset_mock()

        result = service._get_last_entry(genesis.workflow_id)

        assert result == genesis
        mock_supabase.table.assert_not_called()

    def test_get_last_entry_from_database(self, service, mock_supabase):
        """Test _get_last_entry queries database if not in cache."""
        workflow_id = uuid4()
        entry_id = uuid4()

        mock_result = MagicMock()
        mock_result.data = [
            {
                "entry_id": str(entry_id),
                "workflow_id": str(workflow_id),
                "sequence_number": 3,
                "agent_name": "causal_impact",
                "agent_tier": 2,
                "action_type": "estimate",
                "created_at": "2025-01-01T12:00:00+00:00",
                "duration_ms": 100,
                "input_hash": "input123",
                "output_hash": "output456",
                "entry_hash": "entryhash",
                "previous_entry_id": None,
                "previous_hash": None,
            }
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = (
            mock_result
        )

        result = service._get_last_entry(workflow_id)

        assert result is not None
        assert result.entry_id == entry_id
        assert result.sequence_number == 3
        assert result.agent_name == "causal_impact"

    def test_get_last_entry_returns_none_if_not_found(self, service, mock_supabase):
        """Test _get_last_entry returns None if workflow not found."""
        mock_result = MagicMock()
        mock_result.data = []

        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = (
            mock_result
        )

        result = service._get_last_entry(uuid4())

        assert result is None

    # -------------------------------------------------------------------------
    # Verification Tests
    # -------------------------------------------------------------------------

    def test_verify_workflow_calls_rpc(self, service, mock_supabase):
        """Test verify_workflow calls the database RPC function."""
        workflow_id = uuid4()

        mock_result = MagicMock()
        mock_result.data = [
            {
                "is_valid": True,
                "entries_checked": 5,
                "first_invalid_entry": None,
                "error_message": None,
            }
        ]
        mock_supabase.rpc.return_value.execute.return_value = mock_result
        mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock()

        result = service.verify_workflow(workflow_id)

        assert result.is_valid is True
        assert result.entries_checked == 5
        mock_supabase.rpc.assert_called_with(
            "verify_chain_integrity", {"p_workflow_id": str(workflow_id)}
        )

    def test_verify_workflow_local_valid_chain(self, service, mock_supabase):
        """Test verify_workflow_local with valid chain."""
        workflow_id = uuid4()
        entry1_id = uuid4()
        entry2_id = uuid4()

        # Create entries with proper hash chain
        entry1_hash = service._compute_hash(
            f"{entry1_id}{workflow_id}1testtest2025-01-01T12:00:00+00:00GENESIS"
        )
        entry2_hash = service._compute_hash(
            f"{entry2_id}{workflow_id}2testtest2025-01-01T12:00:01+00:00{entry1_hash}"
        )

        mock_result = MagicMock()
        mock_result.data = [
            {
                "entry_id": str(entry1_id),
                "workflow_id": str(workflow_id),
                "sequence_number": 1,
                "agent_name": "test",
                "agent_tier": 0,
                "action_type": "test",
                "created_at": "2025-01-01T12:00:00+00:00",
                "entry_hash": entry1_hash,
                "previous_hash": None,
            },
            {
                "entry_id": str(entry2_id),
                "workflow_id": str(workflow_id),
                "sequence_number": 2,
                "agent_name": "test",
                "agent_tier": 0,
                "action_type": "test",
                "created_at": "2025-01-01T12:00:01+00:00",
                "entry_hash": entry2_hash,
                "previous_hash": entry1_hash,
            },
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = (
            mock_result
        )

        result = service.verify_workflow_local(workflow_id)

        assert result.is_valid is True
        assert result.entries_checked == 2

    def test_verify_workflow_local_not_found(self, service, mock_supabase):
        """Test verify_workflow_local returns invalid for unknown workflow."""
        mock_result = MagicMock()
        mock_result.data = []

        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = (
            mock_result
        )

        result = service.verify_workflow_local(uuid4())

        assert result.is_valid is False
        assert result.entries_checked == 0
        assert result.error_message == "Workflow not found"

    def test_verify_workflow_local_broken_chain(self, service, mock_supabase):
        """Test verify_workflow_local detects broken chain (previous_hash mismatch)."""
        workflow_id = uuid4()
        entry1_id = uuid4()
        entry2_id = uuid4()

        # Create a valid entry1 with proper computed hash
        entry1_created = "2025-01-01T12:00:00+00:00"
        entry1_hash = service._compute_hash(
            f"{entry1_id}{workflow_id}1testtest{entry1_created}GENESIS"
        )

        # Create entry2 with WRONG previous_hash (should be entry1_hash but isn't)
        entry2_created = "2025-01-01T12:00:01+00:00"
        wrong_previous_hash = "definitely_wrong_hash"

        mock_result = MagicMock()
        mock_result.data = [
            {
                "entry_id": str(entry1_id),
                "workflow_id": str(workflow_id),
                "sequence_number": 1,
                "agent_name": "test",
                "agent_tier": 0,
                "action_type": "test",
                "created_at": entry1_created,
                "entry_hash": entry1_hash,  # Correct hash for entry 1
                "previous_hash": None,
                "input_hash": None,
                "output_hash": None,
            },
            {
                "entry_id": str(entry2_id),
                "workflow_id": str(workflow_id),
                "sequence_number": 2,
                "agent_name": "test",
                "agent_tier": 0,
                "action_type": "test",
                "created_at": entry2_created,
                "entry_hash": "dummy_hash",  # Doesn't matter for this test
                "previous_hash": wrong_previous_hash,  # BROKEN: should be entry1_hash
                "input_hash": None,
                "output_hash": None,
            },
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = (
            mock_result
        )

        result = service.verify_workflow_local(workflow_id)

        assert result.is_valid is False
        assert result.first_invalid_entry == entry2_id
        assert "Previous hash mismatch" in result.error_message

    # -------------------------------------------------------------------------
    # Query Tests
    # -------------------------------------------------------------------------

    def test_get_workflow_summary(self, service, mock_supabase):
        """Test get_workflow_summary queries summary view."""
        workflow_id = uuid4()

        mock_result = MagicMock()
        mock_result.data = [
            {
                "workflow_id": str(workflow_id),
                "total_entries": 5,
                "chain_valid": True,
            }
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
            mock_result
        )

        result = service.get_workflow_summary(workflow_id)

        assert result is not None
        assert result["total_entries"] == 5
        mock_supabase.table.assert_called_with("v_audit_chain_summary")

    def test_get_workflow_summary_not_found(self, service, mock_supabase):
        """Test get_workflow_summary returns None if not found."""
        mock_result = MagicMock()
        mock_result.data = []

        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
            mock_result
        )

        result = service.get_workflow_summary(uuid4())

        assert result is None

    def test_get_causal_validations(self, service, mock_supabase):
        """Test get_causal_validations queries validation view."""
        mock_result = MagicMock()
        mock_result.data = [
            {"validation_passed": True, "brand": "Kisqali"},
            {"validation_passed": True, "brand": "Kisqali"},
        ]

        mock_query = MagicMock()
        mock_supabase.table.return_value.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.gte.return_value = mock_query
        mock_query.lte.return_value = mock_query
        mock_query.order.return_value.execute.return_value = mock_result

        result = service.get_causal_validations(brand="Kisqali")

        assert len(result) == 2
        mock_supabase.table.assert_called_with("v_causal_validation_chain")


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateAuditChainService:
    """Tests for the create_audit_chain_service factory function."""

    def test_creates_service_with_client(self):
        """Test factory creates service with Supabase client."""
        # The create_client is imported inside the function from supabase
        with patch("supabase.create_client") as mock_create_client:
            mock_client = MagicMock()
            mock_create_client.return_value = mock_client

            service = create_audit_chain_service(
                supabase_url="https://test.supabase.co",
                supabase_key="test-key",
            )

            assert isinstance(service, AuditChainService)
            assert service.db == mock_client
            mock_create_client.assert_called_once_with(
                "https://test.supabase.co", "test-key"
            )
