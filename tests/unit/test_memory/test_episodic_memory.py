"""
Unit tests for E2I Episodic Memory (Supabase + pgvector).

Tests the episodic memory module with mocked Supabase client.
"""

from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.episodic_memory import (
    AgentActivityContext,
    E2IAgentName,
    E2IBrand,
    # Data classes
    E2IEntityContext,
    E2IEntityReferences,
    # Enums
    E2IEntityType,
    E2IRegion,
    EnrichedEpisodicMemory,
    EpisodicMemoryInput,
    EpisodicSearchFilters,
    bulk_insert_episodic_memories,
    count_memories_by_type,
    delete_memory,
    get_agent_activity_with_context,
    get_causal_path_context,
    get_enriched_episodic_memory,
    get_memory_by_id,
    get_memory_entity_context,
    get_recent_memories,
    insert_episodic_memory,
    insert_episodic_memory_with_text,
    search_episodic_by_e2i_entity,
    search_episodic_by_text,
    # Functions
    search_episodic_memory,
    sync_treatment_relationships_to_cache,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    client = MagicMock()

    # Mock table operations
    table_mock = MagicMock()
    client.table.return_value = table_mock

    # Chain methods
    table_mock.select.return_value = table_mock
    table_mock.eq.return_value = table_mock
    table_mock.in_.return_value = table_mock
    table_mock.order.return_value = table_mock
    table_mock.limit.return_value = table_mock
    table_mock.single.return_value = table_mock
    table_mock.insert.return_value = table_mock
    table_mock.delete.return_value = table_mock

    # Default execute result
    result = MagicMock()
    result.data = []
    result.count = 0
    table_mock.execute.return_value = result

    # Mock RPC
    rpc_mock = MagicMock()
    client.rpc.return_value = rpc_mock
    rpc_result = MagicMock()
    rpc_result.data = []
    rpc_mock.execute.return_value = rpc_result

    return client


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = AsyncMock()
    service.embed.return_value = [0.1] * 1536  # OpenAI-style embedding
    return service


@pytest.fixture
def sample_memory_input():
    """Create a sample episodic memory input."""
    return EpisodicMemoryInput(
        event_type="query_answer",
        description="Answered question about TRx drop in Northeast region",
        event_subtype="kpi_investigation",
        raw_content={"query": "Why did TRx drop?", "response": "Analysis shows..."},
        entities={"kpis": ["TRx"], "regions": ["northeast"]},
        outcome_type="success",
        outcome_details={"confidence": 0.85},
        user_satisfaction_score=4,
        agent_name="causal_impact",
        importance_score=0.75,
        e2i_refs=E2IEntityReferences(brand="Kisqali", region="northeast", hcp_id="hcp_123"),
    )


@pytest.fixture
def sample_search_filters():
    """Create sample search filters."""
    return EpisodicSearchFilters(
        event_type="query_answer", agent_name="causal_impact", brand="Kisqali", region="northeast"
    )


@pytest.fixture
def sample_memory_record():
    """Create a sample memory record from database."""
    return {
        "memory_id": "mem_12345678",
        "session_id": "sess_123",
        "cycle_id": "cycle_123",
        "event_type": "query_answer",
        "event_subtype": "kpi_investigation",
        "description": "Answered question about TRx drop",
        "raw_content": '{"query": "Why?"}',
        "entities": '{"kpis": ["TRx"]}',
        "outcome_type": "success",
        "outcome_details": '{"confidence": 0.85}',
        "user_satisfaction_score": 4,
        "agent_name": "causal_impact",
        "importance_score": 0.75,
        "brand": "Kisqali",
        "region": "northeast",
        "hcp_id": "hcp_123",
        "occurred_at": "2025-12-20T10:00:00+00:00",
    }


# ============================================================================
# DATA CLASS TESTS
# ============================================================================


class TestEnums:
    """Test E2I enum values."""

    def test_entity_type_values(self):
        """E2IEntityType should have all expected values."""
        assert E2IEntityType.PATIENT.value == "patient"
        assert E2IEntityType.HCP.value == "hcp"
        assert E2IEntityType.TRIGGER.value == "trigger"
        assert E2IEntityType.CAUSAL_PATH.value == "causal_path"
        assert E2IEntityType.PREDICTION.value == "prediction"
        assert E2IEntityType.TREATMENT.value == "treatment"
        assert E2IEntityType.EXPERIMENT.value == "experiment"
        assert E2IEntityType.AGENT_ACTIVITY.value == "agent_activity"

    def test_brand_values(self):
        """E2IBrand should have all expected values."""
        assert E2IBrand.REMIBRUTINIB.value == "Remibrutinib"
        assert E2IBrand.FABHALTA.value == "Fabhalta"
        assert E2IBrand.KISQALI.value == "Kisqali"
        assert E2IBrand.ALL.value == "all"

    def test_region_values(self):
        """E2IRegion should have all expected values."""
        assert E2IRegion.NORTHEAST.value == "northeast"
        assert E2IRegion.SOUTH.value == "south"
        assert E2IRegion.MIDWEST.value == "midwest"
        assert E2IRegion.WEST.value == "west"

    def test_agent_names(self):
        """E2IAgentName should have all expected values."""
        assert E2IAgentName.ORCHESTRATOR.value == "orchestrator"
        assert E2IAgentName.CAUSAL_IMPACT.value == "causal_impact"
        assert E2IAgentName.FEEDBACK_LEARNER.value == "feedback_learner"


class TestDataClasses:
    """Test data class creation and defaults."""

    def test_e2i_entity_context_defaults(self):
        """E2IEntityContext should have None defaults."""
        context = E2IEntityContext()
        assert context.patient is None
        assert context.hcp is None
        assert context.trigger is None
        assert context.causal_path is None

    def test_e2i_entity_context_with_values(self):
        """E2IEntityContext should accept values."""
        context = E2IEntityContext(
            patient={"id": "p1", "name": "Patient 1"}, hcp={"id": "h1", "name": "Dr. Smith"}
        )
        assert context.patient["id"] == "p1"
        assert context.hcp["name"] == "Dr. Smith"

    def test_e2i_entity_references_defaults(self):
        """E2IEntityReferences should have None defaults."""
        refs = E2IEntityReferences()
        assert refs.patient_journey_id is None
        assert refs.hcp_id is None
        assert refs.brand is None

    def test_episodic_memory_input_required_fields(self, sample_memory_input):
        """EpisodicMemoryInput should have required fields."""
        assert sample_memory_input.event_type == "query_answer"
        assert sample_memory_input.description is not None

    def test_episodic_memory_input_default_importance(self):
        """EpisodicMemoryInput should have default importance_score."""
        memory = EpisodicMemoryInput(event_type="test", description="Test memory")
        assert memory.importance_score == 0.5

    def test_episodic_search_filters_all_optional(self):
        """EpisodicSearchFilters should have all optional fields."""
        filters = EpisodicSearchFilters()
        assert filters.event_type is None
        assert filters.agent_name is None
        assert filters.brand is None

    def test_enriched_episodic_memory(self):
        """EnrichedEpisodicMemory should store context."""
        memory = EnrichedEpisodicMemory(
            memory_id="mem_123",
            event_type="query_answer",
            description="Test",
            occurred_at="2025-12-20T10:00:00",
            patient_context={"id": "p1"},
        )
        assert memory.memory_id == "mem_123"
        assert memory.patient_context["id"] == "p1"

    def test_agent_activity_context(self):
        """AgentActivityContext should store activity details."""
        activity = AgentActivityContext(
            activity_id="act_123",
            agent_name="causal_impact",
            action_type="analyze",
            started_at="2025-12-20T10:00:00",
            status="completed",
            duration_ms=1500,
        )
        assert activity.activity_id == "act_123"
        assert activity.duration_ms == 1500


# ============================================================================
# SEARCH FUNCTION TESTS
# ============================================================================


class TestSearchEpisodicMemory:
    """Test episodic memory search functions."""

    @pytest.mark.asyncio
    async def test_search_episodic_memory_basic(self, mock_supabase):
        """search_episodic_memory should call RPC with parameters."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            embedding = [0.1] * 1536

            mock_supabase.rpc.return_value.execute.return_value.data = [
                {"memory_id": "mem_1", "similarity": 0.85}
            ]

            results = await search_episodic_memory(embedding=embedding, limit=10)

            mock_supabase.rpc.assert_called_once()
            call_args = mock_supabase.rpc.call_args
            assert call_args[0][0] == "search_episodic_memory"
            assert len(results) == 1
            assert results[0]["memory_id"] == "mem_1"

    @pytest.mark.asyncio
    async def test_search_episodic_memory_with_filters(self, mock_supabase, sample_search_filters):
        """search_episodic_memory should pass filters to RPC."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            embedding = [0.1] * 1536

            mock_supabase.rpc.return_value.execute.return_value.data = []

            await search_episodic_memory(
                embedding=embedding, filters=sample_search_filters, min_similarity=0.7
            )

            call_args = mock_supabase.rpc.call_args
            filter_params = call_args[0][1]
            assert filter_params["filter_event_type"] == "query_answer"
            assert filter_params["filter_brand"] == "Kisqali"
            assert filter_params["match_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_search_episodic_memory_empty_results(self, mock_supabase):
        """search_episodic_memory should handle empty results."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.rpc.return_value.execute.return_value.data = None

            results = await search_episodic_memory(embedding=[0.1] * 1536)

            assert results == []

    @pytest.mark.asyncio
    async def test_search_episodic_by_text(self, mock_supabase, mock_embedding_service):
        """search_episodic_by_text should generate embedding and search."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            with patch(
                "src.memory.episodic_memory.get_embedding_service",
                return_value=mock_embedding_service,
            ):
                mock_supabase.rpc.return_value.execute.return_value.data = [{"memory_id": "mem_1"}]

                results = await search_episodic_by_text("Why did TRx drop?")

                mock_embedding_service.embed.assert_called_once_with("Why did TRx drop?")
                assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_episodic_by_e2i_entity(self, mock_supabase):
        """search_episodic_by_e2i_entity should query by entity column."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = [{"memory_id": "mem_1"}]

            results = await search_episodic_by_e2i_entity(
                entity_type=E2IEntityType.HCP, entity_id="hcp_123"
            )

            mock_supabase.table.assert_called_with("episodic_memories")
            mock_supabase.table.return_value.eq.assert_called_with("hcp_id", "hcp_123")
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_episodic_by_e2i_entity_with_event_types(self, mock_supabase):
        """search_episodic_by_e2i_entity should filter by event types."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = []

            await search_episodic_by_e2i_entity(
                entity_type=E2IEntityType.PATIENT,
                entity_id="patient_123",
                event_types=["query_answer", "investigation"],
            )

            mock_supabase.table.return_value.in_.assert_called_with(
                "event_type", ["query_answer", "investigation"]
            )

    @pytest.mark.asyncio
    async def test_search_episodic_by_e2i_entity_all_types(self, mock_supabase):
        """search_episodic_by_e2i_entity should handle all entity types."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = []

            # Test each entity type maps to correct column
            entity_column_map = {
                E2IEntityType.PATIENT: "patient_journey_id",
                E2IEntityType.HCP: "hcp_id",
                E2IEntityType.TRIGGER: "trigger_id",
                E2IEntityType.PREDICTION: "prediction_id",
                E2IEntityType.CAUSAL_PATH: "causal_path_id",
                E2IEntityType.EXPERIMENT: "experiment_id",
                E2IEntityType.TREATMENT: "treatment_event_id",
                E2IEntityType.AGENT_ACTIVITY: "agent_activity_id",
            }

            for entity_type, expected_column in entity_column_map.items():
                mock_supabase.reset_mock()
                # Reset chain
                mock_supabase.table.return_value.select.return_value = (
                    mock_supabase.table.return_value
                )
                mock_supabase.table.return_value.eq.return_value = mock_supabase.table.return_value
                mock_supabase.table.return_value.order.return_value = (
                    mock_supabase.table.return_value
                )
                mock_supabase.table.return_value.limit.return_value = (
                    mock_supabase.table.return_value
                )
                mock_supabase.table.return_value.execute.return_value.data = []

                await search_episodic_by_e2i_entity(entity_type, "id_123")

                mock_supabase.table.return_value.eq.assert_called_with(expected_column, "id_123")


# ============================================================================
# INSERT FUNCTION TESTS
# ============================================================================


class TestInsertEpisodicMemory:
    """Test episodic memory insert functions."""

    @pytest.mark.asyncio
    async def test_insert_episodic_memory_basic(self, mock_supabase, sample_memory_input):
        """insert_episodic_memory should insert with all fields."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            embedding = [0.1] * 1536

            memory_id = await insert_episodic_memory(
                memory=sample_memory_input,
                embedding=embedding,
                session_id="sess_123",
                cycle_id="cycle_456",
            )

            assert memory_id is not None
            mock_supabase.table.assert_called_with("episodic_memories")
            mock_supabase.table.return_value.insert.assert_called_once()

            # Check insert arguments
            insert_call = mock_supabase.table.return_value.insert.call_args
            record = insert_call[0][0]
            assert record["event_type"] == "query_answer"
            assert record["description"] == sample_memory_input.description
            assert record["session_id"] == "sess_123"
            assert record["cycle_id"] == "cycle_456"
            assert record["brand"] == "Kisqali"
            assert record["hcp_id"] == "hcp_123"

    @pytest.mark.asyncio
    async def test_insert_episodic_memory_minimal(self, mock_supabase):
        """insert_episodic_memory should work with minimal input."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            memory = EpisodicMemoryInput(event_type="test", description="Test memory")
            embedding = [0.1] * 1536

            memory_id = await insert_episodic_memory(memory=memory, embedding=embedding)

            assert memory_id is not None
            insert_call = mock_supabase.table.return_value.insert.call_args
            record = insert_call[0][0]
            assert record["event_type"] == "test"
            assert "session_id" not in record or record.get("session_id") is None

    @pytest.mark.asyncio
    async def test_insert_episodic_memory_with_text(
        self, mock_supabase, mock_embedding_service, sample_memory_input
    ):
        """insert_episodic_memory_with_text should auto-generate embedding."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            with patch(
                "src.memory.episodic_memory.get_embedding_service",
                return_value=mock_embedding_service,
            ):
                memory_id = await insert_episodic_memory_with_text(memory=sample_memory_input)

                mock_embedding_service.embed.assert_called_once_with(
                    sample_memory_input.description
                )
                assert memory_id is not None

    @pytest.mark.asyncio
    async def test_insert_episodic_memory_with_custom_text(
        self, mock_supabase, mock_embedding_service, sample_memory_input
    ):
        """insert_episodic_memory_with_text should use custom text for embedding."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            with patch(
                "src.memory.episodic_memory.get_embedding_service",
                return_value=mock_embedding_service,
            ):
                await insert_episodic_memory_with_text(
                    memory=sample_memory_input, text_to_embed="Custom embedding text"
                )

                mock_embedding_service.embed.assert_called_once_with("Custom embedding text")

    @pytest.mark.asyncio
    async def test_bulk_insert_episodic_memories(self, mock_supabase):
        """bulk_insert_episodic_memories should insert multiple records."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            memories = [
                (EpisodicMemoryInput(event_type="test1", description="Test 1"), [0.1] * 1536),
                (EpisodicMemoryInput(event_type="test2", description="Test 2"), [0.2] * 1536),
                (EpisodicMemoryInput(event_type="test3", description="Test 3"), [0.3] * 1536),
            ]

            memory_ids = await bulk_insert_episodic_memories(
                memories=memories, session_id="sess_123"
            )

            assert len(memory_ids) == 3
            mock_supabase.table.return_value.insert.assert_called_once()
            insert_call = mock_supabase.table.return_value.insert.call_args
            records = insert_call[0][0]
            assert len(records) == 3
            assert records[0]["event_type"] == "test1"
            assert records[1]["event_type"] == "test2"
            assert records[2]["event_type"] == "test3"

    @pytest.mark.asyncio
    async def test_bulk_insert_empty_list(self, mock_supabase):
        """bulk_insert_episodic_memories should handle empty list."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            memory_ids = await bulk_insert_episodic_memories(memories=[])

            assert memory_ids == []
            mock_supabase.table.return_value.insert.assert_not_called()


# ============================================================================
# CONTEXT FUNCTION TESTS
# ============================================================================


class TestContextFunctions:
    """Test entity context retrieval functions."""

    @pytest.mark.asyncio
    async def test_get_memory_entity_context(self, mock_supabase):
        """get_memory_entity_context should return entity details."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.rpc.return_value.execute.return_value.data = [
                {
                    "entity_type": "patient",
                    "entity_id": "p1",
                    "entity_name": "Patient 1",
                    "entity_details": {},
                },
                {
                    "entity_type": "hcp",
                    "entity_id": "h1",
                    "entity_name": "Dr. Smith",
                    "entity_details": {},
                },
            ]

            context = await get_memory_entity_context("mem_123")

            mock_supabase.rpc.assert_called_with(
                "get_memory_entity_context", {"p_memory_id": "mem_123"}
            )
            assert context.patient["id"] == "p1"
            assert context.hcp["name"] == "Dr. Smith"

    @pytest.mark.asyncio
    async def test_get_memory_entity_context_empty(self, mock_supabase):
        """get_memory_entity_context should return empty context for no results."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.rpc.return_value.execute.return_value.data = []

            context = await get_memory_entity_context("mem_123")

            assert context.patient is None
            assert context.hcp is None

    @pytest.mark.asyncio
    async def test_get_memory_entity_context_error_handling(self, mock_supabase):
        """get_memory_entity_context should handle errors gracefully."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.rpc.return_value.execute.side_effect = Exception("RPC failed")

            context = await get_memory_entity_context("mem_123")

            # Should return empty context, not raise
            assert context.patient is None

    @pytest.mark.asyncio
    async def test_get_enriched_episodic_memory(self, mock_supabase, sample_memory_record):
        """get_enriched_episodic_memory should return memory with context."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = sample_memory_record
            mock_supabase.rpc.return_value.execute.return_value.data = [
                {
                    "entity_type": "hcp",
                    "entity_id": "h1",
                    "entity_name": "Dr. Smith",
                    "entity_details": {},
                }
            ]

            enriched = await get_enriched_episodic_memory("mem_12345678")

            assert enriched is not None
            assert enriched.memory_id == "mem_12345678"
            assert enriched.event_type == "query_answer"
            assert enriched.hcp_context is not None

    @pytest.mark.asyncio
    async def test_get_enriched_episodic_memory_not_found(self, mock_supabase):
        """get_enriched_episodic_memory should return None if not found."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = None

            enriched = await get_enriched_episodic_memory("nonexistent")

            assert enriched is None

    @pytest.mark.asyncio
    async def test_get_agent_activity_with_context(self, mock_supabase):
        """get_agent_activity_with_context should return activity details."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.rpc.return_value.execute.return_value.data = [
                {
                    "activity_id": "act_123",
                    "agent_name": "causal_impact",
                    "action_type": "analyze",
                    "started_at": "2025-12-20T10:00:00",
                    "completed_at": "2025-12-20T10:00:05",
                    "status": "completed",
                    "trigger": {"id": "t1"},
                    "causal_paths": [{"id": "cp1"}],
                    "predictions": None,
                    "duration_ms": 5000,
                    "tokens_used": 1500,
                }
            ]

            activity = await get_agent_activity_with_context("act_123")

            assert activity is not None
            assert activity.agent_name == "causal_impact"
            assert activity.duration_ms == 5000

    @pytest.mark.asyncio
    async def test_get_agent_activity_with_context_not_found(self, mock_supabase):
        """get_agent_activity_with_context should return None if not found."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.rpc.return_value.execute.return_value.data = []

            activity = await get_agent_activity_with_context("nonexistent")

            assert activity is None

    @pytest.mark.asyncio
    async def test_get_causal_path_context(self, mock_supabase):
        """get_causal_path_context should return path with related memories."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            # First call - get causal path
            path_result = MagicMock()
            path_result.data = {
                "path_id": "cp_123",
                "source_entity": "TRx",
                "target_entity": "Revenue",
                "effect_size": 0.35,
                "confidence": 0.89,
                "method_used": "dowhy",
                "created_at": "2025-12-20",
            }

            # Second call - get related memories
            memories_result = MagicMock()
            memories_result.data = [{"memory_id": "mem_1"}]

            mock_supabase.table.return_value.execute.side_effect = [path_result, memories_result]

            context = await get_causal_path_context("cp_123")

            assert context is not None
            assert context["path_id"] == "cp_123"
            assert context["effect_size"] == 0.35
            assert len(context["related_memories"]) == 1

    @pytest.mark.asyncio
    async def test_get_causal_path_context_not_found(self, mock_supabase):
        """get_causal_path_context should return None if path not found."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = None

            context = await get_causal_path_context("nonexistent")

            assert context is None


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================


class TestUtilityFunctions:
    """Test episodic memory utility functions."""

    @pytest.mark.asyncio
    async def test_get_recent_memories(self, mock_supabase):
        """get_recent_memories should return memories ordered by time."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = [
                {"memory_id": "mem_1"},
                {"memory_id": "mem_2"},
            ]

            memories = await get_recent_memories(limit=10)

            mock_supabase.table.return_value.order.assert_called_with("occurred_at", desc=True)
            assert len(memories) == 2

    @pytest.mark.asyncio
    async def test_get_recent_memories_with_filters(self, mock_supabase):
        """get_recent_memories should apply filters."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = []

            await get_recent_memories(
                limit=10, event_types=["query_answer"], agent_name="causal_impact", brand="Kisqali"
            )

            mock_supabase.table.return_value.in_.assert_called_with("event_type", ["query_answer"])
            # eq is called multiple times for different filters

    @pytest.mark.asyncio
    async def test_get_memory_by_id(self, mock_supabase, sample_memory_record):
        """get_memory_by_id should return single memory."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = sample_memory_record

            memory = await get_memory_by_id("mem_12345678")

            mock_supabase.table.return_value.eq.assert_called_with("memory_id", "mem_12345678")
            assert memory is not None

    @pytest.mark.asyncio
    async def test_get_memory_by_id_not_found(self, mock_supabase):
        """get_memory_by_id should return None if not found."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = None

            memory = await get_memory_by_id("nonexistent")

            assert memory is None

    @pytest.mark.asyncio
    async def test_delete_memory(self, mock_supabase):
        """delete_memory should delete and return True."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = [{"memory_id": "mem_123"}]

            result = await delete_memory("mem_123")

            mock_supabase.table.return_value.delete.assert_called_once()
            mock_supabase.table.return_value.eq.assert_called_with("memory_id", "mem_123")
            assert result is True

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, mock_supabase):
        """delete_memory should return False if not found."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = []

            result = await delete_memory("nonexistent")

            assert result is False

    @pytest.mark.asyncio
    async def test_count_memories_by_type(self, mock_supabase):
        """count_memories_by_type should return count."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.count = 42

            count = await count_memories_by_type(event_type="query_answer")

            assert count == 42

    @pytest.mark.asyncio
    async def test_count_memories_by_type_zero(self, mock_supabase):
        """count_memories_by_type should return 0 for no matches."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.count = None

            count = await count_memories_by_type()

            assert count == 0

    @pytest.mark.asyncio
    async def test_sync_treatment_relationships_to_cache(self, mock_supabase):
        """sync_treatment_relationships_to_cache should call RPC and return count."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.rpc.return_value.execute.return_value.data = 150

            count = await sync_treatment_relationships_to_cache()

            mock_supabase.rpc.assert_called_with("sync_hcp_patient_relationships_to_cache", {})
            assert count == 150

    @pytest.mark.asyncio
    async def test_sync_treatment_relationships_error(self, mock_supabase):
        """sync_treatment_relationships_to_cache should handle errors."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.rpc.return_value.execute.side_effect = Exception("RPC failed")

            count = await sync_treatment_relationships_to_cache()

            assert count == 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_insert_memory_filters_none_values(self, mock_supabase):
        """insert_episodic_memory should filter out None values from record."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            memory = EpisodicMemoryInput(
                event_type="test", description="Test", event_subtype=None, outcome_type=None
            )

            await insert_episodic_memory(memory=memory, embedding=[0.1] * 1536)

            insert_call = mock_supabase.table.return_value.insert.call_args
            record = insert_call[0][0]
            # None values should be filtered out
            for key, value in record.items():
                assert value is not None, f"Key {key} should not have None value"

    @pytest.mark.asyncio
    async def test_search_with_include_entity_context(self, mock_supabase):
        """search_episodic_memory with include_entity_context should enrich results."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            # First call - search
            mock_supabase.rpc.return_value.execute.return_value.data = [
                {"memory_id": "mem_1", "similarity": 0.9}
            ]

            results = await search_episodic_memory(
                embedding=[0.1] * 1536, include_entity_context=True
            )

            # Should have called RPC twice (search + context)
            assert mock_supabase.rpc.call_count >= 1
            assert len(results) == 1
            # e2i_context should be added
            assert "e2i_context" in results[0]

    @pytest.mark.asyncio
    async def test_all_valid_entity_types_work(self, mock_supabase):
        """search_episodic_by_e2i_entity should handle all valid entity types."""
        with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
            mock_supabase.table.return_value.execute.return_value.data = []

            # Test that all valid entity types work without error
            for entity_type in E2IEntityType:
                # Reset mock for each call
                mock_supabase.reset_mock()
                mock_supabase.table.return_value.select.return_value = (
                    mock_supabase.table.return_value
                )
                mock_supabase.table.return_value.eq.return_value = mock_supabase.table.return_value
                mock_supabase.table.return_value.order.return_value = (
                    mock_supabase.table.return_value
                )
                mock_supabase.table.return_value.limit.return_value = (
                    mock_supabase.table.return_value
                )
                mock_supabase.table.return_value.execute.return_value.data = []

                result = await search_episodic_by_e2i_entity(entity_type, "id_123")
                assert result == []  # Should return empty list without error

    def test_memory_input_with_all_e2i_refs(self):
        """EpisodicMemoryInput should accept all E2I reference types."""
        refs = E2IEntityReferences(
            patient_journey_id="pj_1",
            patient_id="p_1",
            hcp_id="h_1",
            treatment_event_id="te_1",
            trigger_id="t_1",
            prediction_id="pr_1",
            causal_path_id="cp_1",
            experiment_id="e_1",
            agent_activity_id="aa_1",
            brand="Kisqali",
            region="northeast",
        )

        memory = EpisodicMemoryInput(
            event_type="test", description="Test with all refs", e2i_refs=refs
        )

        assert memory.e2i_refs.patient_journey_id == "pj_1"
        assert memory.e2i_refs.brand == "Kisqali"

    def test_enriched_memory_to_dict(self):
        """EnrichedEpisodicMemory should be convertible to dict."""
        memory = EnrichedEpisodicMemory(
            memory_id="mem_123",
            event_type="query_answer",
            description="Test",
            occurred_at="2025-12-20T10:00:00",
            patient_context={"id": "p1", "name": "Patient 1"},
        )

        memory_dict = asdict(memory)

        assert memory_dict["memory_id"] == "mem_123"
        assert memory_dict["patient_context"]["id"] == "p1"
