"""Unit tests for AuditChainMixin and related utilities.

Tests the audit chain integration layer for LangGraph agents,
including the mixin class, decorators, and helper functions.
"""

import pytest
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from src.agents.base.audit_chain_mixin import (
    AuditChainMixin,
    audited_traced_node,
    create_workflow_initializer,
    get_audit_chain_service,
    set_audit_chain_service,
    _row_to_entry,
)
from src.utils.audit_chain import (
    AgentTier,
    AuditChainEntry,
    AuditChainService,
    ChainVerificationResult,
    RefutationResults,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_audit_service():
    """Create a mock AuditChainService."""
    service = MagicMock(spec=AuditChainService)
    # Add db attribute for get_workflow_entries which accesses service.db directly
    service.db = MagicMock()
    return service


@pytest.fixture
def sample_workflow_id():
    """Return a sample workflow ID."""
    return uuid4()


@pytest.fixture
def sample_entry(sample_workflow_id):
    """Create a sample AuditChainEntry."""
    return AuditChainEntry(
        entry_id=uuid4(),
        workflow_id=sample_workflow_id,
        sequence_number=1,
        agent_name="test_agent",
        agent_tier=AgentTier.CAUSAL_ANALYTICS.value,
        action_type="test_action",
        created_at=datetime.now(),
        entry_hash="abc123",
    )


@pytest.fixture
def sample_state(sample_workflow_id):
    """Create a sample state dict for testing."""
    return {
        "query": "What is the impact of treatment?",
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "session_id": "test-session-123",
        "audit_workflow_id": sample_workflow_id,
        "query_id": "trace-123",
        "span_id": "span-456",
    }


@pytest.fixture
def mixin_instance():
    """Create an AuditChainMixin instance for testing."""
    return AuditChainMixin()


@pytest.fixture(autouse=True)
def reset_global_service():
    """Reset the global audit service before each test."""
    set_audit_chain_service(None)
    yield
    set_audit_chain_service(None)


# =============================================================================
# Tests for get/set_audit_chain_service
# =============================================================================


class TestServiceManagement:
    """Tests for global service management functions."""

    def test_get_returns_none_when_not_set(self):
        """get_audit_chain_service returns None when not initialized."""
        result = get_audit_chain_service()
        assert result is None

    def test_set_and_get_service(self, mock_audit_service):
        """set_audit_chain_service sets the global service."""
        set_audit_chain_service(mock_audit_service)
        result = get_audit_chain_service()
        assert result is mock_audit_service

    def test_set_none_clears_service(self, mock_audit_service):
        """Setting None clears the service."""
        set_audit_chain_service(mock_audit_service)
        set_audit_chain_service(None)
        assert get_audit_chain_service() is None


# =============================================================================
# Tests for AuditChainMixin
# =============================================================================


class TestAuditChainMixin:
    """Tests for AuditChainMixin class methods."""

    @pytest.mark.asyncio
    async def test_start_audit_workflow_success(
        self, mixin_instance, mock_audit_service, sample_entry
    ):
        """start_audit_workflow returns workflow_id on success."""
        mock_audit_service.start_workflow.return_value = sample_entry
        set_audit_chain_service(mock_audit_service)

        result = await mixin_instance.start_audit_workflow(
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="initialization",
            input_data={"query": "test"},
        )

        assert result == sample_entry.workflow_id
        mock_audit_service.start_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_audit_workflow_service_unavailable(self, mixin_instance):
        """start_audit_workflow returns None when service is unavailable."""
        # Service is None by default

        result = await mixin_instance.start_audit_workflow(
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="initialization",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_start_audit_workflow_handles_exception(
        self, mixin_instance, mock_audit_service
    ):
        """start_audit_workflow returns None on exception."""
        mock_audit_service.start_workflow.side_effect = Exception("DB error")
        set_audit_chain_service(mock_audit_service)

        result = await mixin_instance.start_audit_workflow(
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="initialization",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_add_audit_entry_success(
        self, mixin_instance, mock_audit_service, sample_entry, sample_workflow_id
    ):
        """add_audit_entry returns entry on success."""
        mock_audit_service.add_entry.return_value = sample_entry
        set_audit_chain_service(mock_audit_service)

        result = await mixin_instance.add_audit_entry(
            workflow_id=sample_workflow_id,
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="estimation",
            duration_ms=150,
        )

        assert result == sample_entry
        mock_audit_service.add_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_audit_entry_service_unavailable(
        self, mixin_instance, sample_workflow_id
    ):
        """add_audit_entry returns None when service is unavailable."""
        result = await mixin_instance.add_audit_entry(
            workflow_id=sample_workflow_id,
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="estimation",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_add_audit_entry_with_refutation_results(
        self, mixin_instance, mock_audit_service, sample_entry, sample_workflow_id
    ):
        """add_audit_entry handles refutation results correctly."""
        mock_audit_service.add_entry.return_value = sample_entry
        set_audit_chain_service(mock_audit_service)

        refutation = RefutationResults(
            placebo_treatment=True,
            random_common_cause=True,
            data_subset=False,
            unobserved_confound=True,
        )

        await mixin_instance.add_audit_entry(
            workflow_id=sample_workflow_id,
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="refutation",
            refutation_results=refutation,
            validation_passed=True,
            confidence_score=0.85,
        )

        # Verify the call included the refutation results
        call_kwargs = mock_audit_service.add_entry.call_args.kwargs
        assert call_kwargs["refutation_results"] == refutation
        assert call_kwargs["validation_passed"] is True
        assert call_kwargs["confidence_score"] == 0.85

    @pytest.mark.asyncio
    async def test_verify_audit_workflow_success(
        self, mixin_instance, mock_audit_service, sample_workflow_id
    ):
        """verify_audit_workflow returns result on success."""
        verification = ChainVerificationResult(
            is_valid=True,
            entries_checked=5,
        )
        mock_audit_service.verify_workflow.return_value = verification
        set_audit_chain_service(mock_audit_service)

        result = await mixin_instance.verify_audit_workflow(sample_workflow_id)

        assert result == verification
        assert result.is_valid is True
        assert result.entries_checked == 5

    @pytest.mark.asyncio
    async def test_verify_audit_workflow_service_unavailable(
        self, mixin_instance, sample_workflow_id
    ):
        """verify_audit_workflow returns None when service unavailable."""
        result = await mixin_instance.verify_audit_workflow(sample_workflow_id)
        assert result is None

    def test_get_workflow_entries_success(
        self, mixin_instance, mock_audit_service, sample_workflow_id
    ):
        """get_workflow_entries returns list of entries."""
        # Mock database response
        mock_table = MagicMock()
        mock_audit_service.db.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.order.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[
            {
                "entry_id": str(uuid4()),
                "workflow_id": str(sample_workflow_id),
                "sequence_number": 1,
                "agent_name": "test",
                "agent_tier": 2,
                "action_type": "test",
                "created_at": "2025-12-30T00:00:00Z",
                "entry_hash": "abc123",
            }
        ])
        set_audit_chain_service(mock_audit_service)

        result = mixin_instance.get_workflow_entries(sample_workflow_id)

        assert len(result) == 1
        assert result[0].agent_name == "test"

    def test_get_workflow_entries_service_unavailable(
        self, mixin_instance, sample_workflow_id
    ):
        """get_workflow_entries returns empty list when service unavailable."""
        result = mixin_instance.get_workflow_entries(sample_workflow_id)
        assert result == []


# =============================================================================
# Tests for audited_traced_node decorator
# =============================================================================


class TestAuditedTracedNodeDecorator:
    """Tests for the audited_traced_node decorator."""

    @pytest.mark.asyncio
    async def test_decorator_calls_function(self, sample_state):
        """Decorator calls the wrapped function."""
        call_count = 0

        @audited_traced_node("estimation", "causal_impact", AgentTier.CAUSAL_ANALYTICS)
        async def test_node(state: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"status": "completed", "current_phase": "done"}

        # Mock opik connector
        with patch("src.agents.base.audit_chain_mixin.get_opik_connector") as mock_opik:
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_opik.return_value.trace_agent.return_value = mock_ctx

            result = await test_node(sample_state)

        assert call_count == 1
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_decorator_records_audit_entry(
        self, mock_audit_service, sample_state
    ):
        """Decorator records audit entry when service is available."""
        set_audit_chain_service(mock_audit_service)

        @audited_traced_node("estimation", "causal_impact", AgentTier.CAUSAL_ANALYTICS)
        async def test_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "completed", "estimation_result": {"energy_score": 0.9}}

        with patch("src.agents.base.audit_chain_mixin.get_opik_connector") as mock_opik:
            mock_span = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_span)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_opik.return_value.trace_agent.return_value = mock_ctx

            await test_node(sample_state)

        # Verify audit entry was recorded
        mock_audit_service.add_entry.assert_called_once()
        call_kwargs = mock_audit_service.add_entry.call_args.kwargs
        assert call_kwargs["action_type"] == "estimation"
        assert call_kwargs["agent_name"] == "causal_impact"

    @pytest.mark.asyncio
    async def test_decorator_handles_node_error(self, mock_audit_service, sample_state):
        """Decorator records error entry when node raises exception."""
        set_audit_chain_service(mock_audit_service)

        @audited_traced_node("estimation", "causal_impact", AgentTier.CAUSAL_ANALYTICS)
        async def failing_node(state: Dict[str, Any]) -> Dict[str, Any]:
            raise ValueError("Test error")

        with patch("src.agents.base.audit_chain_mixin.get_opik_connector") as mock_opik:
            mock_span = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_span)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_opik.return_value.trace_agent.return_value = mock_ctx

            with pytest.raises(ValueError, match="Test error"):
                await failing_node(sample_state)

        # Verify error entry was recorded
        mock_audit_service.add_entry.assert_called_once()
        call_kwargs = mock_audit_service.add_entry.call_args.kwargs
        assert call_kwargs["action_type"] == "estimation_error"
        assert call_kwargs["validation_passed"] is False

    @pytest.mark.asyncio
    async def test_decorator_works_without_audit_service(self, sample_state):
        """Decorator works when audit service is unavailable."""
        # Clear workflow_id to simulate no audit
        state_no_audit = {**sample_state, "audit_workflow_id": None}

        @audited_traced_node("estimation", "causal_impact", AgentTier.CAUSAL_ANALYTICS)
        async def test_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "completed"}

        with patch("src.agents.base.audit_chain_mixin.get_opik_connector") as mock_opik:
            mock_span = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_span)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_opik.return_value.trace_agent.return_value = mock_ctx

            result = await test_node(state_no_audit)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_decorator_extracts_refutation_results(
        self, mock_audit_service, sample_state
    ):
        """Decorator extracts refutation results for refutation node."""
        set_audit_chain_service(mock_audit_service)

        @audited_traced_node("refutation", "causal_impact", AgentTier.CAUSAL_ANALYTICS)
        async def refutation_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "status": "completed",
                "refutation_results": {
                    "overall_robust": True,
                    "individual_tests": {
                        "placebo_treatment": {"passed": True},
                        "random_common_cause": {"passed": True},
                        "data_subset": {"passed": False},
                        "unobserved_common_cause": {"passed": True},
                    },
                },
            }

        with patch("src.agents.base.audit_chain_mixin.get_opik_connector") as mock_opik:
            mock_span = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_span)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_opik.return_value.trace_agent.return_value = mock_ctx

            await refutation_node(sample_state)

        # Verify refutation results were included
        call_kwargs = mock_audit_service.add_entry.call_args.kwargs
        assert call_kwargs["validation_passed"] is True
        assert call_kwargs["refutation_results"] is not None


# =============================================================================
# Tests for create_workflow_initializer
# =============================================================================


class TestCreateWorkflowInitializer:
    """Tests for the create_workflow_initializer function."""

    def test_initializer_adds_workflow_id(self, mock_audit_service, sample_entry):
        """Initializer adds workflow_id to state."""
        mock_audit_service.start_workflow.return_value = sample_entry
        set_audit_chain_service(mock_audit_service)

        initializer = create_workflow_initializer(
            "causal_impact", AgentTier.CAUSAL_ANALYTICS
        )
        state = {"query": "test", "treatment_var": "x", "outcome_var": "y"}
        result = initializer(state)

        assert "audit_workflow_id" in result
        assert result["audit_workflow_id"] == sample_entry.workflow_id

    def test_initializer_preserves_existing_state(
        self, mock_audit_service, sample_entry
    ):
        """Initializer preserves existing state fields."""
        mock_audit_service.start_workflow.return_value = sample_entry
        set_audit_chain_service(mock_audit_service)

        initializer = create_workflow_initializer(
            "causal_impact", AgentTier.CAUSAL_ANALYTICS
        )
        state = {
            "query": "test",
            "treatment_var": "x",
            "outcome_var": "y",
            "extra_field": "preserved",
        }
        result = initializer(state)

        assert result["extra_field"] == "preserved"
        assert result["query"] == "test"

    def test_initializer_returns_state_when_service_unavailable(self):
        """Initializer returns original state when service unavailable."""
        initializer = create_workflow_initializer(
            "causal_impact", AgentTier.CAUSAL_ANALYTICS
        )
        state = {"query": "test"}
        result = initializer(state)

        assert result == state
        assert "audit_workflow_id" not in result

    def test_initializer_handles_exception(self, mock_audit_service):
        """Initializer returns original state on exception."""
        mock_audit_service.start_workflow.side_effect = Exception("DB error")
        set_audit_chain_service(mock_audit_service)

        initializer = create_workflow_initializer(
            "causal_impact", AgentTier.CAUSAL_ANALYTICS
        )
        state = {"query": "test"}
        result = initializer(state)

        assert result == state


# =============================================================================
# Tests for _row_to_entry helper
# =============================================================================


class TestRowToEntry:
    """Tests for the _row_to_entry helper function."""

    def test_converts_minimal_row(self):
        """Converts row with minimal required fields."""
        entry_id = uuid4()
        workflow_id = uuid4()
        row = {
            "entry_id": str(entry_id),
            "workflow_id": str(workflow_id),
            "sequence_number": 1,
            "agent_name": "test",
            "agent_tier": 2,
            "action_type": "test",
            "created_at": "2025-12-30T00:00:00Z",
            "entry_hash": "abc123",
        }

        result = _row_to_entry(row)

        assert result.entry_id == entry_id
        assert result.workflow_id == workflow_id
        assert result.agent_name == "test"
        assert result.entry_hash == "abc123"

    def test_converts_row_with_all_fields(self):
        """Converts row with all optional fields."""
        entry_id = uuid4()
        workflow_id = uuid4()
        previous_id = uuid4()
        session_id = uuid4()

        row = {
            "entry_id": str(entry_id),
            "workflow_id": str(workflow_id),
            "sequence_number": 3,
            "agent_name": "causal_impact",
            "agent_tier": 2,
            "action_type": "estimation",
            "created_at": "2025-12-30T12:30:00Z",
            "duration_ms": 250,
            "input_hash": "input123",
            "output_hash": "output456",
            "validation_passed": True,
            "confidence_score": 0.92,
            "refutation_results": {"placebo": True},
            "previous_entry_id": str(previous_id),
            "previous_hash": "prev789",
            "entry_hash": "abc123",
            "user_id": "user-123",
            "session_id": str(session_id),
            "brand": "Remibrutinib",
        }

        result = _row_to_entry(row)

        assert result.duration_ms == 250
        assert result.input_hash == "input123"
        assert result.validation_passed is True
        assert result.confidence_score == 0.92
        assert result.previous_entry_id == previous_id
        assert result.session_id == session_id
        assert result.brand == "Remibrutinib"

    def test_handles_none_optional_fields(self):
        """Handles None values for optional fields."""
        row = {
            "entry_id": str(uuid4()),
            "workflow_id": str(uuid4()),
            "sequence_number": 1,
            "agent_name": "test",
            "agent_tier": 2,
            "action_type": "test",
            "created_at": "2025-12-30T00:00:00Z",
            "entry_hash": "abc123",
            "duration_ms": None,
            "previous_entry_id": None,
            "session_id": None,
        }

        result = _row_to_entry(row)

        assert result.duration_ms is None
        assert result.previous_entry_id is None
        assert result.session_id is None
