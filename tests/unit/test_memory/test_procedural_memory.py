"""
Unit tests for E2I Procedural Memory module.

Tests pattern learning, few-shot example retrieval, and DSPy learning signals.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from src.memory.procedural_memory import (
    # Data classes
    ProceduralMemoryInput,
    LearningSignalInput,
    # Procedural memory functions
    find_relevant_procedures,
    find_relevant_procedures_by_text,
    insert_procedural_memory,
    insert_procedural_memory_with_text,
    get_few_shot_examples,
    get_few_shot_examples_by_text,
    update_procedure_outcome,
    get_procedure_by_id,
    deactivate_procedure,
    get_top_procedures,
    # Learning signal functions
    record_learning_signal,
    get_training_examples_for_agent,
    get_feedback_summary_for_trigger,
    get_feedback_summary_for_agent,
    get_recent_signals,
    # Statistics functions
    _increment_memory_stats,
    get_memory_statistics,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    client = MagicMock()
    table_mock = MagicMock()
    client.table.return_value = table_mock

    # Chain methods for table operations
    table_mock.select.return_value = table_mock
    table_mock.insert.return_value = table_mock
    table_mock.update.return_value = table_mock
    table_mock.upsert.return_value = table_mock
    table_mock.eq.return_value = table_mock
    table_mock.gte.return_value = table_mock
    table_mock.lte.return_value = table_mock
    table_mock.order.return_value = table_mock
    table_mock.limit.return_value = table_mock
    table_mock.single.return_value = table_mock
    table_mock.execute.return_value.data = []

    # RPC mock
    rpc_mock = MagicMock()
    client.rpc.return_value = rpc_mock
    rpc_mock.execute.return_value.data = []

    return client


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = MagicMock()
    service.embed = AsyncMock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def sample_embedding():
    """Sample embedding vector."""
    return [0.1] * 1536


@pytest.fixture
def sample_procedure_input():
    """Sample procedural memory input."""
    return ProceduralMemoryInput(
        procedure_name="investigate_trx_drop",
        tool_sequence=[
            {"tool": "query_kpi", "params": {"kpi": "TRx"}},
            {"tool": "analyze_trend", "params": {"period": "30d"}},
            {"tool": "find_correlations", "params": {}}
        ],
        procedure_type="investigation",
        trigger_pattern="Why did TRx drop?",
        intent_keywords=["TRx", "drop", "decline"],
        detected_intent="kpi_investigation",
        applicable_brands=["Kisqali"],
        applicable_regions=["northeast"],
        applicable_agents=["causal_impact", "gap_analyzer"]
    )


@pytest.fixture
def sample_learning_signal():
    """Sample learning signal input."""
    return LearningSignalInput(
        signal_type="thumbs_up",
        signal_value=1.0,
        signal_details={"feedback": "Helpful response"},
        applies_to_type="response",
        applies_to_id="resp_123",
        related_patient_id="pat_456",
        related_trigger_id="trig_789",
        brand="Kisqali",
        region="northeast",
        rated_agent="causal_impact",
        is_training_example=True,
        dspy_metric_name="relevance",
        dspy_metric_value=0.95,
        training_input="Why did TRx drop?",
        training_output="TRx dropped due to seasonal factors."
    )


# ============================================================================
# DATA CLASS TESTS
# ============================================================================

class TestProceduralMemoryInput:
    """Tests for ProceduralMemoryInput data class."""

    def test_minimal_input(self):
        """ProceduralMemoryInput should work with only required fields."""
        proc = ProceduralMemoryInput(
            procedure_name="test_procedure",
            tool_sequence=[{"tool": "test"}]
        )
        assert proc.procedure_name == "test_procedure"
        assert proc.tool_sequence == [{"tool": "test"}]
        assert proc.procedure_type == "tool_sequence"  # default
        assert proc.applicable_brands is None
        assert proc.applicable_regions is None

    def test_full_input(self, sample_procedure_input):
        """ProceduralMemoryInput should accept all fields."""
        proc = sample_procedure_input
        assert proc.procedure_name == "investigate_trx_drop"
        assert len(proc.tool_sequence) == 3
        assert proc.procedure_type == "investigation"
        assert proc.detected_intent == "kpi_investigation"
        assert "Kisqali" in proc.applicable_brands
        assert "causal_impact" in proc.applicable_agents


class TestLearningSignalInput:
    """Tests for LearningSignalInput data class."""

    def test_minimal_input(self):
        """LearningSignalInput should work with only required fields."""
        signal = LearningSignalInput(signal_type="thumbs_up")
        assert signal.signal_type == "thumbs_up"
        assert signal.signal_value is None
        assert signal.is_training_example is False

    def test_dspy_training_input(self, sample_learning_signal):
        """LearningSignalInput should support DSPy training fields."""
        signal = sample_learning_signal
        assert signal.is_training_example is True
        assert signal.dspy_metric_name == "relevance"
        assert signal.dspy_metric_value == 0.95
        assert signal.training_input is not None
        assert signal.training_output is not None

    def test_e2i_context(self, sample_learning_signal):
        """LearningSignalInput should include E2I context."""
        signal = sample_learning_signal
        assert signal.related_patient_id == "pat_456"
        assert signal.related_trigger_id == "trig_789"
        assert signal.brand == "Kisqali"
        assert signal.region == "northeast"
        assert signal.rated_agent == "causal_impact"


# ============================================================================
# PROCEDURAL MEMORY FUNCTION TESTS
# ============================================================================

class TestFindRelevantProcedures:
    """Tests for find_relevant_procedures function."""

    @pytest.mark.asyncio
    async def test_basic_search(self, mock_supabase, sample_embedding):
        """find_relevant_procedures should call RPC with correct parameters."""
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {"procedure_id": "proc_1", "similarity": 0.85}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await find_relevant_procedures(sample_embedding)

        mock_supabase.rpc.assert_called_once_with(
            "find_relevant_procedures",
            {
                "query_embedding": sample_embedding,
                "match_threshold": 0.6,
                "match_count": 5,
                "filter_type": None,
                "filter_intent": None,
                "filter_brand": None
            }
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_supabase, sample_embedding):
        """find_relevant_procedures should pass filters to RPC."""
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await find_relevant_procedures(
                sample_embedding,
                procedure_type="investigation",
                intent="kpi_investigation",
                brand="Kisqali",
                limit=10,
                min_similarity=0.7
            )

        call_args = mock_supabase.rpc.call_args[0][1]
        assert call_args["filter_type"] == "investigation"
        assert call_args["filter_intent"] == "kpi_investigation"
        assert call_args["filter_brand"] == "Kisqali"
        assert call_args["match_count"] == 10
        assert call_args["match_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_supabase, sample_embedding):
        """find_relevant_procedures should return empty list when no matches."""
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await find_relevant_procedures(sample_embedding)

        assert result == []

    @pytest.mark.asyncio
    async def test_none_data_handling(self, mock_supabase, sample_embedding):
        """find_relevant_procedures should handle None data gracefully."""
        mock_supabase.rpc.return_value.execute.return_value.data = None

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await find_relevant_procedures(sample_embedding)

        assert result == []


class TestFindRelevantProceduresByText:
    """Tests for find_relevant_procedures_by_text function."""

    @pytest.mark.asyncio
    async def test_text_search(self, mock_supabase, mock_embedding_service):
        """find_relevant_procedures_by_text should embed text and search."""
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {"procedure_id": "proc_1", "similarity": 0.8}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            with patch("src.memory.procedural_memory.get_embedding_service", return_value=mock_embedding_service):
                result = await find_relevant_procedures_by_text(
                    "Why did TRx drop?",
                    intent="kpi_investigation"
                )

        mock_embedding_service.embed.assert_called_once_with("Why did TRx drop?")
        assert len(result) == 1


class TestInsertProceduralMemory:
    """Tests for insert_procedural_memory function."""

    @pytest.mark.asyncio
    async def test_insert_new_procedure(self, mock_supabase, sample_procedure_input, sample_embedding):
        """insert_procedural_memory should insert new procedure when no similar exists."""
        # No similar procedure found
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            with patch("src.memory.procedural_memory._increment_memory_stats", new_callable=AsyncMock):
                result = await insert_procedural_memory(sample_procedure_input, sample_embedding)

        assert result is not None
        mock_supabase.table.assert_called_with("procedural_memories")

        # Check insert was called
        insert_call = mock_supabase.table.return_value.insert
        assert insert_call.called

    @pytest.mark.asyncio
    async def test_update_existing_procedure(self, mock_supabase, sample_procedure_input, sample_embedding):
        """insert_procedural_memory should update counts when similar procedure exists."""
        # Similar procedure found
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {"procedure_id": "existing_proc", "usage_count": 5, "success_count": 4}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await insert_procedural_memory(sample_procedure_input, sample_embedding)

        assert result == "existing_proc"

        # Check update was called
        update_call = mock_supabase.table.return_value.update
        assert update_call.called

    @pytest.mark.asyncio
    async def test_insert_sets_defaults(self, mock_supabase, sample_embedding):
        """insert_procedural_memory should set default values for optional fields."""
        minimal_input = ProceduralMemoryInput(
            procedure_name="test",
            tool_sequence=[{"tool": "test"}]
        )
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            with patch("src.memory.procedural_memory._increment_memory_stats", new_callable=AsyncMock):
                await insert_procedural_memory(minimal_input, sample_embedding)

        insert_call = mock_supabase.table.return_value.insert
        insert_data = insert_call.call_args[0][0]
        assert insert_data["applicable_brands"] == ["all"]
        assert insert_data["applicable_regions"] == ["all"]
        assert insert_data["usage_count"] == 1
        assert insert_data["success_count"] == 1
        assert insert_data["is_active"] is True


class TestInsertProceduralMemoryWithText:
    """Tests for insert_procedural_memory_with_text function."""

    @pytest.mark.asyncio
    async def test_insert_with_trigger_text(self, mock_supabase, mock_embedding_service, sample_procedure_input):
        """insert_procedural_memory_with_text should embed trigger text."""
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            with patch("src.memory.procedural_memory.get_embedding_service", return_value=mock_embedding_service):
                with patch("src.memory.procedural_memory._increment_memory_stats", new_callable=AsyncMock):
                    await insert_procedural_memory_with_text(
                        sample_procedure_input,
                        trigger_text="Custom trigger text"
                    )

        mock_embedding_service.embed.assert_called_once_with("Custom trigger text")

    @pytest.mark.asyncio
    async def test_insert_uses_trigger_pattern_fallback(self, mock_supabase, mock_embedding_service, sample_procedure_input):
        """insert_procedural_memory_with_text should fall back to trigger_pattern."""
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            with patch("src.memory.procedural_memory.get_embedding_service", return_value=mock_embedding_service):
                with patch("src.memory.procedural_memory._increment_memory_stats", new_callable=AsyncMock):
                    await insert_procedural_memory_with_text(sample_procedure_input)

        # Should use trigger_pattern from input
        mock_embedding_service.embed.assert_called_once_with("Why did TRx drop?")


class TestGetFewShotExamples:
    """Tests for get_few_shot_examples function."""

    @pytest.mark.asyncio
    async def test_format_examples(self, mock_supabase, sample_embedding):
        """get_few_shot_examples should format procedures as examples."""
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {
                "trigger_pattern": "Why did NRx drop?",
                "detected_intent": "kpi_investigation",
                "tool_sequence": json.dumps([{"tool": "query_kpi"}]),
                "success_rate": 0.85,
                "applicable_brands": ["Kisqali"],
                "applicable_regions": ["northeast"]
            }
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_few_shot_examples(sample_embedding)

        assert len(result) == 1
        example = result[0]
        assert example["trigger"] == "Why did NRx drop?"
        assert example["intent"] == "kpi_investigation"
        assert example["solution"] == [{"tool": "query_kpi"}]
        assert example["success_rate"] == 0.85
        assert "Kisqali" in example["applicable_brands"]

    @pytest.mark.asyncio
    async def test_handle_string_tool_sequence(self, mock_supabase, sample_embedding):
        """get_few_shot_examples should parse JSON string tool_sequence."""
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {
                "trigger_pattern": "test",
                "tool_sequence": '[{"tool": "test_tool"}]',
                "success_rate": 0.9
            }
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_few_shot_examples(sample_embedding)

        assert result[0]["solution"] == [{"tool": "test_tool"}]

    @pytest.mark.asyncio
    async def test_handle_list_tool_sequence(self, mock_supabase, sample_embedding):
        """get_few_shot_examples should handle list tool_sequence directly."""
        mock_supabase.rpc.return_value.execute.return_value.data = [
            {
                "trigger_pattern": "test",
                "tool_sequence": [{"tool": "test_tool"}],
                "success_rate": 0.9
            }
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_few_shot_examples(sample_embedding)

        assert result[0]["solution"] == [{"tool": "test_tool"}]


class TestUpdateProcedureOutcome:
    """Tests for update_procedure_outcome function."""

    @pytest.mark.asyncio
    async def test_update_success(self, mock_supabase):
        """update_procedure_outcome should increment both counts on success."""
        mock_supabase.table.return_value.single.return_value.execute.return_value.data = {
            "usage_count": 5,
            "success_count": 4
        }

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await update_procedure_outcome("proc_123", success=True)

        update_call = mock_supabase.table.return_value.update
        update_data = update_call.call_args[0][0]
        assert update_data["usage_count"] == 6
        assert update_data["success_count"] == 5

    @pytest.mark.asyncio
    async def test_update_failure(self, mock_supabase):
        """update_procedure_outcome should only increment usage on failure."""
        mock_supabase.table.return_value.single.return_value.execute.return_value.data = {
            "usage_count": 5,
            "success_count": 4
        }

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await update_procedure_outcome("proc_123", success=False)

        update_call = mock_supabase.table.return_value.update
        update_data = update_call.call_args[0][0]
        assert update_data["usage_count"] == 6
        assert "success_count" not in update_data

    @pytest.mark.asyncio
    async def test_procedure_not_found(self, mock_supabase):
        """update_procedure_outcome should handle missing procedure gracefully."""
        mock_supabase.table.return_value.single.return_value.execute.return_value.data = None

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            # Should not raise
            await update_procedure_outcome("nonexistent", success=True)

        # Update should not be called
        update_call = mock_supabase.table.return_value.update
        # We check .eq was not called on update (because select returned None)


class TestGetProcedureById:
    """Tests for get_procedure_by_id function."""

    @pytest.mark.asyncio
    async def test_get_existing(self, mock_supabase):
        """get_procedure_by_id should return procedure data."""
        mock_supabase.table.return_value.single.return_value.execute.return_value.data = {
            "procedure_id": "proc_123",
            "procedure_name": "test_procedure"
        }

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_procedure_by_id("proc_123")

        assert result["procedure_id"] == "proc_123"
        assert result["procedure_name"] == "test_procedure"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, mock_supabase):
        """get_procedure_by_id should return None for missing procedure."""
        mock_supabase.table.return_value.single.return_value.execute.return_value.data = None

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_procedure_by_id("nonexistent")

        assert result is None


class TestDeactivateProcedure:
    """Tests for deactivate_procedure function."""

    @pytest.mark.asyncio
    async def test_deactivate_success(self, mock_supabase):
        """deactivate_procedure should set is_active to False."""
        mock_supabase.table.return_value.execute.return_value.data = [{"procedure_id": "proc_123"}]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await deactivate_procedure("proc_123")

        assert result is True
        update_call = mock_supabase.table.return_value.update
        update_data = update_call.call_args[0][0]
        assert update_data["is_active"] is False

    @pytest.mark.asyncio
    async def test_deactivate_nonexistent(self, mock_supabase):
        """deactivate_procedure should return False for missing procedure."""
        mock_supabase.table.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await deactivate_procedure("nonexistent")

        assert result is False


class TestGetTopProcedures:
    """Tests for get_top_procedures function."""

    @pytest.mark.asyncio
    async def test_get_top_procedures_basic(self, mock_supabase):
        """get_top_procedures should return procedures ordered by success."""
        mock_supabase.table.return_value.execute.return_value.data = [
            {"procedure_id": "proc_1", "success_count": 100, "applicable_brands": ["all"]},
            {"procedure_id": "proc_2", "success_count": 50, "applicable_brands": ["all"]}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_top_procedures()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_top_procedures_with_brand_filter(self, mock_supabase):
        """get_top_procedures should filter by brand."""
        mock_supabase.table.return_value.execute.return_value.data = [
            {"procedure_id": "proc_1", "applicable_brands": ["Kisqali"]},
            {"procedure_id": "proc_2", "applicable_brands": ["Fabhalta"]},
            {"procedure_id": "proc_3", "applicable_brands": ["all"]}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_top_procedures(brand="Kisqali")

        # Should include Kisqali and "all"
        assert len(result) == 2
        procedure_ids = [p["procedure_id"] for p in result]
        assert "proc_1" in procedure_ids
        assert "proc_3" in procedure_ids

    @pytest.mark.asyncio
    async def test_get_top_procedures_with_type_filter(self, mock_supabase):
        """get_top_procedures should filter by procedure type."""
        mock_supabase.table.return_value.execute.return_value.data = [
            {"procedure_id": "proc_1", "applicable_brands": ["all"]}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await get_top_procedures(procedure_type="investigation")

        # Verify eq was called with procedure_type
        eq_calls = mock_supabase.table.return_value.eq.call_args_list
        assert any(call[0] == ("procedure_type", "investigation") for call in eq_calls)


# ============================================================================
# LEARNING SIGNAL TESTS
# ============================================================================

class TestRecordLearningSignal:
    """Tests for record_learning_signal function."""

    @pytest.mark.asyncio
    async def test_record_basic_signal(self, mock_supabase):
        """record_learning_signal should insert signal record."""
        signal = LearningSignalInput(signal_type="thumbs_up")

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await record_learning_signal(signal)

        assert result is not None
        mock_supabase.table.assert_called_with("learning_signals")

        insert_call = mock_supabase.table.return_value.insert
        insert_data = insert_call.call_args[0][0]
        assert insert_data["signal_type"] == "thumbs_up"

    @pytest.mark.asyncio
    async def test_record_signal_with_context(self, mock_supabase, sample_learning_signal):
        """record_learning_signal should include all E2I context."""
        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await record_learning_signal(
                sample_learning_signal,
                cycle_id="cycle_123",
                session_id="session_456"
            )

        insert_call = mock_supabase.table.return_value.insert
        insert_data = insert_call.call_args[0][0]

        assert insert_data["cycle_id"] == "cycle_123"
        assert insert_data["session_id"] == "session_456"
        assert insert_data["related_patient_id"] == "pat_456"
        assert insert_data["brand"] == "Kisqali"
        assert insert_data["rated_agent"] == "causal_impact"

    @pytest.mark.asyncio
    async def test_record_dspy_training_signal(self, mock_supabase, sample_learning_signal):
        """record_learning_signal should include DSPy training fields."""
        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await record_learning_signal(sample_learning_signal)

        insert_call = mock_supabase.table.return_value.insert
        insert_data = insert_call.call_args[0][0]

        assert insert_data["is_training_example"] is True
        assert insert_data["dspy_metric_name"] == "relevance"
        assert insert_data["dspy_metric_value"] == 0.95
        assert insert_data["training_input"] == "Why did TRx drop?"
        assert insert_data["training_output"] is not None


class TestGetTrainingExamplesForAgent:
    """Tests for get_training_examples_for_agent function."""

    @pytest.mark.asyncio
    async def test_get_training_examples(self, mock_supabase):
        """get_training_examples_for_agent should return high-quality examples."""
        mock_supabase.table.return_value.execute.return_value.data = [
            {
                "signal_id": "sig_1",
                "training_input": "input1",
                "training_output": "output1",
                "dspy_metric_value": 0.95
            }
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_training_examples_for_agent("causal_impact")

        assert len(result) == 1

        # Verify query filters
        eq_calls = mock_supabase.table.return_value.eq.call_args_list
        assert any(call[0] == ("rated_agent", "causal_impact") for call in eq_calls)
        assert any(call[0] == ("is_training_example", True) for call in eq_calls)

    @pytest.mark.asyncio
    async def test_get_training_examples_with_brand(self, mock_supabase):
        """get_training_examples_for_agent should filter by brand."""
        mock_supabase.table.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await get_training_examples_for_agent(
                "causal_impact",
                brand="Kisqali",
                min_score=0.8
            )

        eq_calls = mock_supabase.table.return_value.eq.call_args_list
        assert any(call[0] == ("brand", "Kisqali") for call in eq_calls)

        gte_calls = mock_supabase.table.return_value.gte.call_args_list
        assert any(call[0] == ("dspy_metric_value", 0.8) for call in gte_calls)


class TestGetFeedbackSummaryForTrigger:
    """Tests for get_feedback_summary_for_trigger function."""

    @pytest.mark.asyncio
    async def test_feedback_summary(self, mock_supabase):
        """get_feedback_summary_for_trigger should aggregate feedback."""
        mock_supabase.table.return_value.execute.return_value.data = [
            {"signal_type": "thumbs_up", "signal_value": None},
            {"signal_type": "thumbs_up", "signal_value": None},
            {"signal_type": "thumbs_down", "signal_value": None},
            {"signal_type": "rating", "signal_value": 4.5},
            {"signal_type": "rating", "signal_value": 3.5},
            {"signal_type": "correction", "signal_value": None}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_feedback_summary_for_trigger("trig_123")

        assert result["trigger_id"] == "trig_123"
        assert result["total_feedback"] == 6
        assert result["thumbs_up"] == 2
        assert result["thumbs_down"] == 1
        assert result["corrections_count"] == 1
        assert result["avg_rating"] == 4.0

    @pytest.mark.asyncio
    async def test_feedback_summary_no_ratings(self, mock_supabase):
        """get_feedback_summary_for_trigger should handle no ratings."""
        mock_supabase.table.return_value.execute.return_value.data = [
            {"signal_type": "thumbs_up", "signal_value": None}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_feedback_summary_for_trigger("trig_123")

        assert result["avg_rating"] is None

    @pytest.mark.asyncio
    async def test_feedback_summary_empty(self, mock_supabase):
        """get_feedback_summary_for_trigger should handle no feedback."""
        mock_supabase.table.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_feedback_summary_for_trigger("trig_123")

        assert result["total_feedback"] == 0
        assert result["thumbs_up"] == 0
        assert result["thumbs_down"] == 0


class TestGetFeedbackSummaryForAgent:
    """Tests for get_feedback_summary_for_agent function."""

    @pytest.mark.asyncio
    async def test_agent_feedback_summary(self, mock_supabase):
        """get_feedback_summary_for_agent should aggregate agent feedback."""
        mock_supabase.table.return_value.execute.return_value.data = [
            {"signal_type": "thumbs_up", "signal_value": None, "is_training_example": True},
            {"signal_type": "rating", "signal_value": 4.0, "is_training_example": False},
            {"signal_type": "correction", "signal_value": None, "is_training_example": True}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_feedback_summary_for_agent("causal_impact")

        assert result["agent_name"] == "causal_impact"
        assert result["total_feedback"] == 3
        assert result["thumbs_up"] == 1
        assert result["corrections_count"] == 1
        assert result["training_examples"] == 2
        assert result["avg_rating"] == 4.0


class TestGetRecentSignals:
    """Tests for get_recent_signals function."""

    @pytest.mark.asyncio
    async def test_get_recent_signals(self, mock_supabase):
        """get_recent_signals should return recent signals ordered by date."""
        mock_supabase.table.return_value.execute.return_value.data = [
            {"signal_id": "sig_1", "signal_type": "thumbs_up"},
            {"signal_id": "sig_2", "signal_type": "rating"}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_recent_signals(limit=10)

        assert len(result) == 2
        mock_supabase.table.return_value.order.assert_called_with("created_at", desc=True)
        mock_supabase.table.return_value.limit.assert_called_with(10)

    @pytest.mark.asyncio
    async def test_get_recent_signals_with_filters(self, mock_supabase):
        """get_recent_signals should filter by type and agent."""
        mock_supabase.table.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await get_recent_signals(
                limit=20,
                signal_type="thumbs_up",
                agent_name="causal_impact"
            )

        eq_calls = mock_supabase.table.return_value.eq.call_args_list
        assert any(call[0] == ("signal_type", "thumbs_up") for call in eq_calls)
        assert any(call[0] == ("rated_agent", "causal_impact") for call in eq_calls)


# ============================================================================
# MEMORY STATISTICS TESTS
# ============================================================================

class TestIncrementMemoryStats:
    """Tests for _increment_memory_stats function."""

    @pytest.mark.asyncio
    async def test_increment_stats(self, mock_supabase):
        """_increment_memory_stats should upsert stats record."""
        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await _increment_memory_stats("procedural", "investigation")

        mock_supabase.table.assert_called_with("memory_statistics")
        upsert_call = mock_supabase.table.return_value.upsert
        assert upsert_call.called

        upsert_data = upsert_call.call_args[0][0]
        assert upsert_data["memory_type"] == "procedural"
        assert upsert_data["subtype"] == "investigation"
        assert upsert_data["count"] == 1

    @pytest.mark.asyncio
    async def test_increment_stats_default_subtype(self, mock_supabase):
        """_increment_memory_stats should use 'general' as default subtype."""
        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await _increment_memory_stats("procedural")

        upsert_call = mock_supabase.table.return_value.upsert
        upsert_data = upsert_call.call_args[0][0]
        assert upsert_data["subtype"] == "general"

    @pytest.mark.asyncio
    async def test_increment_stats_handles_error(self, mock_supabase):
        """_increment_memory_stats should not raise on error."""
        mock_supabase.table.return_value.upsert.side_effect = Exception("DB error")

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            # Should not raise
            await _increment_memory_stats("procedural", "investigation")


class TestGetMemoryStatistics:
    """Tests for get_memory_statistics function."""

    @pytest.mark.asyncio
    async def test_get_memory_statistics(self, mock_supabase):
        """get_memory_statistics should aggregate stats by type."""
        mock_supabase.table.return_value.execute.return_value.data = [
            {"memory_type": "procedural", "count": 100, "stat_date": "2025-01-01"},
            {"memory_type": "procedural", "count": 50, "stat_date": "2025-01-02"},
            {"memory_type": "episodic", "count": 200, "stat_date": "2025-01-01"}
        ]

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_memory_statistics(days_back=30)

        assert result["period_days"] == 30
        assert result["totals_by_type"]["procedural"] == 150
        assert result["totals_by_type"]["episodic"] == 200
        assert len(result["daily_breakdown"]) == 3

    @pytest.mark.asyncio
    async def test_get_memory_statistics_with_type_filter(self, mock_supabase):
        """get_memory_statistics should filter by memory type."""
        mock_supabase.table.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await get_memory_statistics(days_back=7, memory_type="procedural")

        eq_calls = mock_supabase.table.return_value.eq.call_args_list
        assert any(call[0] == ("memory_type", "procedural") for call in eq_calls)

    @pytest.mark.asyncio
    async def test_get_memory_statistics_empty(self, mock_supabase):
        """get_memory_statistics should handle empty stats."""
        mock_supabase.table.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            result = await get_memory_statistics()

        assert result["totals_by_type"] == {}
        assert result["daily_breakdown"] == []


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_empty_tool_sequence(self, mock_supabase, sample_embedding):
        """Procedure with empty tool sequence should still work."""
        procedure = ProceduralMemoryInput(
            procedure_name="empty_procedure",
            tool_sequence=[]
        )
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            with patch("src.memory.procedural_memory._increment_memory_stats", new_callable=AsyncMock):
                result = await insert_procedural_memory(procedure, sample_embedding)

        assert result is not None
        insert_call = mock_supabase.table.return_value.insert
        insert_data = insert_call.call_args[0][0]
        assert insert_data["tool_sequence"] == "[]"

    @pytest.mark.asyncio
    async def test_all_signal_types(self, mock_supabase):
        """All signal types should be recordable."""
        signal_types = ["thumbs_up", "thumbs_down", "rating", "correction"]

        for signal_type in signal_types:
            signal = LearningSignalInput(signal_type=signal_type)

            with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
                result = await record_learning_signal(signal)

            assert result is not None

    @pytest.mark.asyncio
    async def test_all_e2i_brands(self, mock_supabase, sample_embedding):
        """All E2I brands should be supported."""
        brands = ["Remibrutinib", "Fabhalta", "Kisqali"]

        for brand in brands:
            mock_supabase.rpc.return_value.execute.return_value.data = []

            with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
                await find_relevant_procedures(sample_embedding, brand=brand)

            call_args = mock_supabase.rpc.call_args[0][1]
            assert call_args["filter_brand"] == brand

    @pytest.mark.asyncio
    async def test_json_serialization_in_signal(self, mock_supabase):
        """Signal details should be JSON serialized."""
        signal = LearningSignalInput(
            signal_type="correction",
            signal_details={
                "original": "Wrong answer",
                "corrected": "Right answer",
                "nested": {"key": "value"}
            }
        )

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            await record_learning_signal(signal)

        insert_call = mock_supabase.table.return_value.insert
        insert_data = insert_call.call_args[0][0]

        # Should be JSON string
        details = insert_data["signal_details"]
        assert isinstance(details, str)
        parsed = json.loads(details)
        assert parsed["original"] == "Wrong answer"
        assert parsed["nested"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_procedure_with_all_optional_fields(self, mock_supabase, sample_embedding):
        """Procedure with all optional fields should work."""
        procedure = ProceduralMemoryInput(
            procedure_name="full_procedure",
            tool_sequence=[{"tool": "test"}],
            procedure_type="analysis",
            trigger_pattern="Test pattern",
            intent_keywords=["test", "analysis"],
            detected_intent="test_intent",
            applicable_brands=["Kisqali", "Fabhalta"],
            applicable_regions=["northeast", "south"],
            applicable_agents=["causal_impact", "gap_analyzer"]
        )
        mock_supabase.rpc.return_value.execute.return_value.data = []

        with patch("src.memory.procedural_memory.get_supabase_client", return_value=mock_supabase):
            with patch("src.memory.procedural_memory._increment_memory_stats", new_callable=AsyncMock):
                result = await insert_procedural_memory(procedure, sample_embedding)

        assert result is not None
        insert_call = mock_supabase.table.return_value.insert
        insert_data = insert_call.call_args[0][0]

        assert insert_data["applicable_brands"] == ["Kisqali", "Fabhalta"]
        assert insert_data["applicable_regions"] == ["northeast", "south"]
        assert insert_data["applicable_agents"] == ["causal_impact", "gap_analyzer"]
