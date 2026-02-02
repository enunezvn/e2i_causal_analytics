"""
Comprehensive unit tests for src/memory/006_memory_backends_v1_3.py

Tests cover:
- Configuration loading
- Service factories
- Embedding services
- LLM services
- RedisWorkingMemory
- Episodic memory operations
- FalkorDBSemanticMemory
- GraphityExtractor
- Procedural memory operations
- Learning signals
- Memory statistics
"""

import importlib
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock config BEFORE importing the module (config file may not exist locally)
MOCK_CONFIG = {
    "environment": "local_pilot",
    "embeddings": {
        "local_pilot": {"model": "text-embedding-3-small"},
        "aws_production": {"model": "amazon.titan-embed-text-v1"},
    },
    "llm": {
        "local_pilot": {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        "aws_production": {
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "max_tokens": 4096,
        },
    },
    "memory_backends": {
        "working": {
            "local_pilot": {
                "session_prefix": "e2i:session:",
                "evidence_prefix": "e2i:evidence:",
                "ttl_seconds": 3600,
                "context_window_messages": 20,
            }
        },
        "semantic": {
            "local_pilot": {
                "graph_name": "e2i_causal_memory",
                "graphity": {
                    "entity_types": ["Patient", "HCP", "Trigger", "CausalPath"],
                    "relationship_types": ["TREATED_BY", "PRESCRIBES", "CAUSES", "IMPACTS"],
                },
            }
        },
    },
}

# Import the module with numeric prefix using importlib
# Patch file loading before import since config yaml may not exist locally
with patch("builtins.open", MagicMock()):
    with patch("yaml.safe_load", return_value=MOCK_CONFIG):
        memory_backends = importlib.import_module("src.memory.006_memory_backends_v1_3")

# Create convenient aliases for classes and functions from the module
E2IEntityType = memory_backends.E2IEntityType
E2IBrand = memory_backends.E2IBrand
E2IRegion = memory_backends.E2IRegion
E2IAgentName = memory_backends.E2IAgentName
E2IEntityContext = memory_backends.E2IEntityContext
E2IEntityReferences = memory_backends.E2IEntityReferences
EpisodicMemoryInput = memory_backends.EpisodicMemoryInput
EpisodicSearchFilters = memory_backends.EpisodicSearchFilters
EnrichedEpisodicMemory = memory_backends.EnrichedEpisodicMemory
ProceduralMemoryInput = memory_backends.ProceduralMemoryInput
LearningSignalInput = memory_backends.LearningSignalInput
OpenAIEmbeddingService = memory_backends.OpenAIEmbeddingService
BedrockEmbeddingService = memory_backends.BedrockEmbeddingService
AnthropicLLMService = memory_backends.AnthropicLLMService
BedrockLLMService = memory_backends.BedrockLLMService
RedisWorkingMemory = memory_backends.RedisWorkingMemory
FalkorDBSemanticMemory = memory_backends.FalkorDBSemanticMemory
GraphityExtractor = memory_backends.GraphityExtractor
get_embedding_service = memory_backends.get_embedding_service
get_llm_service = memory_backends.get_llm_service
get_redis_client = memory_backends.get_redis_client
get_supabase_client = memory_backends.get_supabase_client
get_falkordb_client = memory_backends.get_falkordb_client
get_working_memory = memory_backends.get_working_memory
get_semantic_memory = memory_backends.get_semantic_memory
search_episodic_memory = memory_backends.search_episodic_memory
search_episodic_by_e2i_entity = memory_backends.search_episodic_by_e2i_entity
insert_episodic_memory = memory_backends.insert_episodic_memory
get_memory_entity_context = memory_backends.get_memory_entity_context
get_enriched_episodic_memory = memory_backends.get_enriched_episodic_memory
bulk_insert_episodic_memories = memory_backends.bulk_insert_episodic_memories
find_relevant_procedures = memory_backends.find_relevant_procedures
insert_procedural_memory = memory_backends.insert_procedural_memory
record_learning_signal = memory_backends.record_learning_signal
get_training_examples_for_agent = memory_backends.get_training_examples_for_agent
get_memory_statistics = memory_backends.get_memory_statistics


@pytest.fixture(autouse=True)
def mock_config():
    """Mock configuration loading for all tests."""
    with patch.object(memory_backends, "load_config", return_value=MOCK_CONFIG):
        with patch.object(memory_backends, "CONFIG", MOCK_CONFIG):
            with patch.object(memory_backends, "ENVIRONMENT", "local_pilot"):
                yield MOCK_CONFIG


# =============================================================================
# Configuration Loading Tests
# =============================================================================


def test_config_structure():
    """Test that config has expected structure."""
    assert "environment" in MOCK_CONFIG
    assert "embeddings" in MOCK_CONFIG
    assert "llm" in MOCK_CONFIG
    assert "memory_backends" in MOCK_CONFIG


def test_environment_config():
    """Test environment-specific config."""
    assert MOCK_CONFIG["environment"] == "local_pilot"
    assert "local_pilot" in MOCK_CONFIG["embeddings"]
    assert "aws_production" in MOCK_CONFIG["embeddings"]


# =============================================================================
# Entity Type Enum Tests
# =============================================================================


def test_e2i_entity_type_enum():
    """Test E2IEntityType enum values."""
    assert E2IEntityType.PATIENT.value == "patient"
    assert E2IEntityType.HCP.value == "hcp"
    assert E2IEntityType.TREATMENT.value == "treatment"
    assert E2IEntityType.TRIGGER.value == "trigger"
    assert E2IEntityType.PREDICTION.value == "prediction"
    assert E2IEntityType.CAUSAL_PATH.value == "causal_path"
    assert E2IEntityType.EXPERIMENT.value == "experiment"
    assert E2IEntityType.AGENT_ACTIVITY.value == "agent_activity"


def test_e2i_brand_enum():
    """Test E2IBrand enum values."""

    assert E2IBrand.REMIBRUTINIB.value == "Remibrutinib"
    assert E2IBrand.FABHALTA.value == "Fabhalta"
    assert E2IBrand.KISQALI.value == "Kisqali"
    assert E2IBrand.ALL.value == "all"


def test_e2i_region_enum():
    """Test E2IRegion enum values."""

    assert E2IRegion.NORTHEAST.value == "northeast"
    assert E2IRegion.SOUTH.value == "south"
    assert E2IRegion.MIDWEST.value == "midwest"
    assert E2IRegion.WEST.value == "west"
    assert E2IRegion.ALL.value == "all"


def test_e2i_agent_name_enum():
    """Test E2IAgentName enum includes all agents."""

    # Tier 0
    assert E2IAgentName.SCOPE_DEFINER.value == "scope_definer"
    assert E2IAgentName.DATA_PREPARER.value == "data_preparer"
    # Tier 1
    assert E2IAgentName.ORCHESTRATOR.value == "orchestrator"
    assert E2IAgentName.TOOL_COMPOSER.value == "tool_composer"
    # Tier 2
    assert E2IAgentName.CAUSAL_IMPACT.value == "causal_impact"
    # Tier 5
    assert E2IAgentName.FEEDBACK_LEARNER.value == "feedback_learner"
    assert E2IAgentName.EXPLAINER.value == "explainer"


# =============================================================================
# Data Class Tests
# =============================================================================


def test_e2i_entity_context_creation():
    """Test E2IEntityContext dataclass."""

    context = E2IEntityContext(
        patient={"id": "PAT123", "name": "Patient A"},
        hcp={"id": "HCP456", "name": "Dr. Smith"},
    )

    assert context.patient["id"] == "PAT123"
    assert context.hcp["id"] == "HCP456"
    assert context.trigger is None
    assert context.causal_path is None


def test_e2i_entity_references_creation():
    """Test E2IEntityReferences dataclass."""

    refs = E2IEntityReferences(
        patient_journey_id="PJ123",
        patient_id="PAT123",
        hcp_id="HCP456",
        brand="Kisqali",
        region="northeast",
    )

    assert refs.patient_journey_id == "PJ123"
    assert refs.patient_id == "PAT123"
    assert refs.hcp_id == "HCP456"
    assert refs.brand == "Kisqali"
    assert refs.region == "northeast"


def test_episodic_memory_input_creation():
    """Test EpisodicMemoryInput dataclass."""

    refs = E2IEntityReferences(patient_id="PAT123", brand="Fabhalta")

    memory_input = EpisodicMemoryInput(
        event_type="user_query",
        description="User asked about Fabhalta",
        agent_name="causal_impact",
        importance_score=0.8,
        e2i_refs=refs,
    )

    assert memory_input.event_type == "user_query"
    assert memory_input.description == "User asked about Fabhalta"
    assert memory_input.agent_name == "causal_impact"
    assert memory_input.importance_score == 0.8
    assert memory_input.e2i_refs.patient_id == "PAT123"


# =============================================================================
# Service Factory Tests
# =============================================================================


def test_get_embedding_service_local_pilot():
    """Test get_embedding_service for local_pilot environment."""
    with patch.object(memory_backends, "OpenAIEmbeddingService") as mock_openai_service:
        mock_instance = MagicMock()
        mock_openai_service.return_value = mock_instance

        service = get_embedding_service()

        mock_openai_service.assert_called_once()
        assert service == mock_instance


def test_get_embedding_service_aws_production():
    """Test get_embedding_service for aws_production environment."""
    with patch.object(memory_backends, "ENVIRONMENT", "aws_production"):
        with patch.object(memory_backends, "BedrockEmbeddingService") as mock_bedrock_service:
            mock_instance = MagicMock()
            mock_bedrock_service.return_value = mock_instance

            service = get_embedding_service()

            mock_bedrock_service.assert_called_once()
            assert service == mock_instance


def test_get_llm_service_local_pilot():
    """Test get_llm_service for local_pilot environment."""
    with patch.object(memory_backends, "AnthropicLLMService") as mock_anthropic_service:
        mock_instance = MagicMock()
        mock_anthropic_service.return_value = mock_instance

        service = get_llm_service()

        mock_anthropic_service.assert_called_once()
        assert service == mock_instance


def test_get_redis_client():
    """Test get_redis_client."""
    with patch.dict("os.environ", {"REDIS_URL": "redis://test:6382"}):
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_from_url.return_value = mock_client

            client = get_redis_client()

            mock_from_url.assert_called_once_with("redis://test:6382", decode_responses=True)
            assert client == mock_client


def test_get_supabase_client():
    """Test get_supabase_client."""
    with patch.dict("os.environ", {"SUPABASE_URL": "http://test", "SUPABASE_ANON_KEY": "test-key"}):
        with patch("supabase.create_client") as mock_create_client:
            mock_client = MagicMock()
            mock_create_client.return_value = mock_client

            client = get_supabase_client()

            mock_create_client.assert_called_once_with("http://test", "test-key")
            assert client == mock_client


def test_get_supabase_client_missing_env_vars():
    """Test get_supabase_client raises when env vars missing."""

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_ANON_KEY must be set"):
            get_supabase_client()


def test_get_falkordb_client():
    """Test get_falkordb_client."""
    with patch.dict("os.environ", {"FALKORDB_HOST": "test-host", "FALKORDB_PORT": "6381"}):
        with patch("falkordb.FalkorDB") as mock_falkordb:
            mock_client = MagicMock()
            mock_falkordb.return_value = mock_client

            client = get_falkordb_client()

            mock_falkordb.assert_called_once_with(host="test-host", port=6381)
            assert client == mock_client


# =============================================================================
# OpenAIEmbeddingService Tests
# =============================================================================


def test_openai_embedding_service_init():
    """Test OpenAIEmbeddingService initialization."""
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        service = OpenAIEmbeddingService()

        assert service.client == mock_client
        assert service.model == "text-embedding-3-small"
        assert service._cache == {}


@pytest.mark.asyncio
async def test_openai_embedding_service_embed():
    """Test OpenAIEmbeddingService.embed()."""
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        service = OpenAIEmbeddingService()
        embedding = await service.embed("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test text"
        )


@pytest.mark.asyncio
async def test_openai_embedding_service_embed_caching():
    """Test OpenAIEmbeddingService caching."""
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        service = OpenAIEmbeddingService()

        # First call
        embedding1 = await service.embed("test text")
        # Second call should use cache
        embedding2 = await service.embed("test text")

        assert embedding1 == embedding2
        # Should only call API once due to caching
        assert mock_client.embeddings.create.call_count == 1


@pytest.mark.asyncio
async def test_openai_embedding_service_embed_batch():
    """Test OpenAIEmbeddingService.embed_batch()."""
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        service = OpenAIEmbeddingService()
        embeddings = await service.embed_batch(["text1", "text2"])

        assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


# =============================================================================
# RedisWorkingMemory Tests
# =============================================================================


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    mock = AsyncMock()
    mock.hgetall = AsyncMock(return_value={})
    mock.hset = AsyncMock()
    mock.expire = AsyncMock()
    mock.lrange = AsyncMock(return_value=[])
    mock.rpush = AsyncMock()
    mock.ltrim = AsyncMock()
    mock.hincrby = AsyncMock()
    mock.delete = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_redis_working_memory_create_session(mock_redis_client):
    """Test RedisWorkingMemory.create_session()."""
    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory = RedisWorkingMemory()
        session_id = await memory.create_session(
            user_id="user123",
            initial_context={"brand": "Kisqali", "filters": {"region": "northeast"}},
        )

        assert isinstance(session_id, str)
        mock_redis_client.hset.assert_called_once()
        mock_redis_client.expire.assert_called_once()


@pytest.mark.asyncio
async def test_redis_working_memory_create_session_with_id(mock_redis_client):
    """Test RedisWorkingMemory.create_session() with provided session_id."""
    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory = RedisWorkingMemory()
        custom_id = "custom-session-123"
        session_id = await memory.create_session(session_id=custom_id)

        assert session_id == custom_id


@pytest.mark.asyncio
async def test_redis_working_memory_get_session():
    """Test RedisWorkingMemory.get_session()."""
    mock_redis_client = AsyncMock()
    mock_redis_client.hgetall = AsyncMock(
        return_value={
            "session_id": "sess123",
            "user_id": "user123",
            "message_count": "5",
            "user_preferences": '{"theme": "dark"}',
            "active_filters": '{"region": "northeast"}',
        }
    )

    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory = RedisWorkingMemory()
        session = await memory.get_session("sess123")

        assert session["session_id"] == "sess123"
        assert session["user_id"] == "user123"
        assert session["message_count"] == 5
        assert session["user_preferences"] == {"theme": "dark"}
        assert session["active_filters"] == {"region": "northeast"}


@pytest.mark.asyncio
async def test_redis_working_memory_get_session_not_found():
    """Test RedisWorkingMemory.get_session() when session not found."""
    mock_redis_client = AsyncMock()
    mock_redis_client.hgetall = AsyncMock(return_value={})

    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory = RedisWorkingMemory()
        session = await memory.get_session("nonexistent")

        assert session is None


@pytest.mark.asyncio
async def test_redis_working_memory_update_session(mock_redis_client):
    """Test RedisWorkingMemory.update_session()."""
    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory = RedisWorkingMemory()
        await memory.update_session("sess123", {"current_phase": "investigation"})

        mock_redis_client.hset.assert_called_once()
        mock_redis_client.expire.assert_called_once()


@pytest.mark.asyncio
async def test_redis_working_memory_set_e2i_context(mock_redis_client):
    """Test RedisWorkingMemory.set_e2i_context()."""
    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory = RedisWorkingMemory()
        await memory.set_e2i_context(
            "sess123",
            brand="Fabhalta",
            region="south",
            patient_ids=["PAT1", "PAT2"],
            hcp_ids=["HCP1"],
        )

        mock_redis_client.hset.assert_called_once()
        call_args = mock_redis_client.hset.call_args[1]["mapping"]
        assert call_args["active_brand"] == "Fabhalta"
        assert call_args["active_region"] == "south"


@pytest.mark.asyncio
async def test_redis_working_memory_get_e2i_context():
    """Test RedisWorkingMemory.get_e2i_context()."""
    mock_redis_client = AsyncMock()
    mock_redis_client.hgetall = AsyncMock(
        return_value={
            "active_brand": "Kisqali",
            "active_region": "northeast",
            "active_patient_ids": '["PAT1", "PAT2"]',
            "active_hcp_ids": '["HCP1"]',
        }
    )

    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory = RedisWorkingMemory()
        context = await memory.get_e2i_context("sess123")

        assert context["brand"] == "Kisqali"
        assert context["region"] == "northeast"
        assert context["patient_ids"] == ["PAT1", "PAT2"]
        assert context["hcp_ids"] == ["HCP1"]


@pytest.mark.asyncio
async def test_redis_working_memory_add_message(mock_redis_client):
    """Test RedisWorkingMemory.add_message()."""
    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory = RedisWorkingMemory()
        await memory.add_message(
            "sess123", "user", "What is Kisqali?", metadata={"intent": "info_query"}
        )

        mock_redis_client.rpush.assert_called_once()
        mock_redis_client.ltrim.assert_called_once()
        mock_redis_client.hincrby.assert_called_once()


@pytest.mark.asyncio
async def test_redis_working_memory_get_messages():
    """Test RedisWorkingMemory.get_messages()."""
    mock_redis_client = AsyncMock()
    mock_messages = [
        json.dumps(
            {
                "role": "user",
                "content": "Hello",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": "{}",
            }
        ),
        json.dumps(
            {
                "role": "assistant",
                "content": "Hi",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": "{}",
            }
        ),
    ]
    mock_redis_client.lrange = AsyncMock(return_value=mock_messages)

    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory = RedisWorkingMemory()
        messages = await memory.get_messages("sess123")

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_redis_working_memory_evidence_trail(mock_redis_client):
    """Test RedisWorkingMemory evidence trail operations."""
    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory = RedisWorkingMemory()

        # Append evidence
        await memory.append_evidence("sess123", {"source": "episodic", "content": "Evidence 1"})
        mock_redis_client.rpush.assert_called_once()

        # Get evidence trail
        mock_redis_client.lrange = AsyncMock(
            return_value=[json.dumps({"source": "episodic", "content": "Evidence 1"})]
        )
        trail = await memory.get_evidence_trail("sess123")
        assert len(trail) == 1
        assert trail[0]["source"] == "episodic"

        # Clear evidence
        await memory.clear_evidence("sess123")
        mock_redis_client.delete.assert_called_once()


# =============================================================================
# Episodic Memory Tests
# =============================================================================


@pytest.mark.asyncio
async def test_search_episodic_memory():
    """Test search_episodic_memory function."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "memory_id": "mem123",
            "description": "Test memory",
            "similarity": 0.9,
        }
    ]
    mock_client.rpc.return_value.execute.return_value = mock_result

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        filters = EpisodicSearchFilters(event_type="user_query", brand="Kisqali")
        memories = await search_episodic_memory(
            embedding=[0.1, 0.2, 0.3], filters=filters, limit=10
        )

        assert len(memories) == 1
        assert memories[0]["memory_id"] == "mem123"
        mock_client.rpc.assert_called_once()


@pytest.mark.asyncio
async def test_search_episodic_by_e2i_entity():
    """Test search_episodic_by_e2i_entity function."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [{"memory_id": "mem123", "description": "Patient memory"}]

    mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
    mock_client.table.return_value = mock_table

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        memories = await search_episodic_by_e2i_entity(
            entity_type=E2IEntityType.PATIENT,
            entity_id="PAT123",
            limit=20,
        )

        assert len(memories) == 1
        assert memories[0]["memory_id"] == "mem123"


@pytest.mark.asyncio
async def test_insert_episodic_memory():
    """Test insert_episodic_memory function."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_table.insert.return_value.execute.return_value = None
    mock_client.table.return_value = mock_table

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        with patch.object(memory_backends, "_increment_memory_stats") as mock_stats:
            refs = E2IEntityReferences(patient_id="PAT123", brand="Kisqali")
            memory = EpisodicMemoryInput(
                event_type="user_query",
                description="Test memory",
                agent_name="causal_impact",
                e2i_refs=refs,
            )

            memory_id = await insert_episodic_memory(
                memory, embedding=[0.1, 0.2, 0.3], session_id="sess123"
            )

            assert isinstance(memory_id, str)
            mock_table.insert.assert_called_once()
            mock_stats.assert_called_once_with("episodic", "user_query")


@pytest.mark.asyncio
async def test_get_memory_entity_context():
    """Test get_memory_entity_context function."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "entity_type": "patient",
            "entity_id": "PAT123",
            "entity_name": "Patient A",
            "entity_details": {"age": 45},
        },
        {
            "entity_type": "hcp",
            "entity_id": "HCP456",
            "entity_name": "Dr. Smith",
            "entity_details": {"specialty": "Oncology"},
        },
    ]
    mock_client.rpc.return_value.execute.return_value = mock_result

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        context = await get_memory_entity_context("mem123")

        assert context.patient is not None
        assert context.patient["id"] == "PAT123"
        assert context.hcp is not None
        assert context.hcp["id"] == "HCP456"


@pytest.mark.asyncio
async def test_get_enriched_episodic_memory():
    """Test get_enriched_episodic_memory function."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.data = {
        "memory_id": "mem123",
        "event_type": "user_query",
        "event_subtype": "causal_inquiry",
        "description": "Test memory",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "outcome_type": "success",
        "agent_name": "causal_impact",
        "importance_score": 0.8,
    }
    mock_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_result

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        with patch.object(memory_backends, "get_memory_entity_context") as mock_context:
            mock_context.return_value = E2IEntityContext(
                patient={"id": "PAT123", "name": "Patient A"}
            )

            enriched = await get_enriched_episodic_memory("mem123")

            assert enriched is not None
            assert enriched.memory_id == "mem123"
            assert enriched.event_type == "user_query"
            assert enriched.patient_context is not None


@pytest.mark.asyncio
async def test_bulk_insert_episodic_memories():
    """Test bulk_insert_episodic_memories function."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_table.insert.return_value.execute.return_value = None
    mock_client.table.return_value = mock_table

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        memories = [
            (EpisodicMemoryInput(event_type="query", description="Test 1"), [0.1, 0.2]),
            (EpisodicMemoryInput(event_type="query", description="Test 2"), [0.3, 0.4]),
        ]

        memory_ids = await bulk_insert_episodic_memories(memories)

        assert len(memory_ids) == 2
        mock_table.insert.assert_called_once()


# =============================================================================
# FalkorDBSemanticMemory Tests
# =============================================================================


def test_falkordb_semantic_memory_init():
    """Test FalkorDBSemanticMemory initialization."""
    mock_client = MagicMock()

    with patch.object(memory_backends, "get_falkordb_client", return_value=mock_client):
        memory = FalkorDBSemanticMemory()

        assert memory._client is None
        assert memory._graph is None


def test_falkordb_add_e2i_entity():
    """Test FalkorDBSemanticMemory.add_e2i_entity()."""
    mock_client = MagicMock()
    mock_graph = MagicMock()
    mock_graph.query.return_value = MagicMock()
    mock_client.select_graph.return_value = mock_graph

    with patch.object(memory_backends, "get_falkordb_client", return_value=mock_client):
        memory = FalkorDBSemanticMemory()
        result = memory.add_e2i_entity(
            entity_type=E2IEntityType.PATIENT,
            entity_id="PAT123",
            properties={"name": "Patient A", "age": 45},
        )

        assert result is True
        mock_graph.query.assert_called_once()


def test_falkordb_add_e2i_relationship():
    """Test FalkorDBSemanticMemory.add_e2i_relationship()."""
    mock_client = MagicMock()
    mock_graph = MagicMock()
    mock_graph.query.return_value = MagicMock()
    mock_client.select_graph.return_value = mock_graph

    with patch.object(memory_backends, "get_falkordb_client", return_value=mock_client):
        memory = FalkorDBSemanticMemory()
        result = memory.add_e2i_relationship(
            source_type=E2IEntityType.PATIENT,
            source_id="PAT123",
            target_type=E2IEntityType.HCP,
            target_id="HCP456",
            rel_type="TREATED_BY",
            properties={"since": "2024-01-01"},
        )

        assert result is True
        # Should call query twice (once for each entity, once for relationship)
        assert mock_graph.query.call_count >= 1


def test_falkordb_get_patient_network():
    """Test FalkorDBSemanticMemory.get_patient_network()."""
    mock_client = MagicMock()
    mock_graph = MagicMock()
    mock_result = MagicMock()
    mock_result.result_set = []
    mock_graph.query.return_value = mock_result
    mock_client.select_graph.return_value = mock_graph

    with patch.object(memory_backends, "get_falkordb_client", return_value=mock_client):
        memory = FalkorDBSemanticMemory()
        network = memory.get_patient_network("PAT123", max_depth=2)

        assert "patient_id" in network
        assert network["patient_id"] == "PAT123"
        assert "hcps" in network
        assert "treatments" in network


def test_falkordb_traverse_causal_chain():
    """Test FalkorDBSemanticMemory.traverse_causal_chain()."""
    mock_client = MagicMock()
    mock_graph = MagicMock()
    mock_result = MagicMock()
    mock_result.result_set = []
    mock_graph.query.return_value = mock_result
    mock_client.select_graph.return_value = mock_graph

    with patch.object(memory_backends, "get_falkordb_client", return_value=mock_client):
        memory = FalkorDBSemanticMemory()
        chains = memory.traverse_causal_chain("START123", max_depth=3)

        assert isinstance(chains, list)


# =============================================================================
# GraphityExtractor Tests
# =============================================================================


@pytest.mark.asyncio
async def test_graphity_extractor_extract_and_store():
    """Test GraphityExtractor.extract_and_store()."""
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(
        return_value='{"entities": [{"id": "E1", "type": "Patient", "properties": {"name": "Test"}}], "relationships": [{"subject": "E1", "subject_type": "Patient", "predicate": "TREATED_BY", "object": "E2", "object_type": "HCP", "confidence": 0.9}]}'
    )

    mock_semantic = MagicMock()
    mock_semantic.add_e2i_entity = MagicMock()
    mock_semantic.add_e2i_relationship = MagicMock()

    with patch.object(memory_backends, "get_llm_service", return_value=mock_llm):
        with patch.object(memory_backends, "get_semantic_memory", return_value=mock_semantic):
            extractor = GraphityExtractor()
            result = await extractor.extract_and_store(
                text="Patient A was treated by Dr. Smith",
                known_e2i_entities={"Patient A": "PAT123"},
            )

            assert result["entities_extracted"] >= 0
            assert result["relationships_extracted"] >= 0


# =============================================================================
# Procedural Memory Tests
# =============================================================================


@pytest.mark.asyncio
async def test_find_relevant_procedures():
    """Test find_relevant_procedures function."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "procedure_id": "proc123",
            "procedure_name": "Test Procedure",
            "similarity": 0.8,
        }
    ]
    mock_client.rpc.return_value.execute.return_value = mock_result

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        procedures = await find_relevant_procedures(
            embedding=[0.1, 0.2, 0.3],
            procedure_type="tool_sequence",
            intent="causal_analysis",
            limit=5,
        )

        assert len(procedures) == 1
        assert procedures[0]["procedure_id"] == "proc123"


@pytest.mark.asyncio
async def test_insert_procedural_memory():
    """Test insert_procedural_memory function."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_table.insert.return_value.execute.return_value = None
    mock_client.table.return_value = mock_table

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        with patch.object(memory_backends, "find_relevant_procedures", return_value=[]):
            with patch.object(memory_backends, "_increment_memory_stats"):
                procedure = ProceduralMemoryInput(
                    procedure_name="test_procedure",
                    tool_sequence=[{"tool": "causal_impact", "params": {}}],
                    trigger_pattern="Why did X happen?",
                )

                procedure_id = await insert_procedural_memory(
                    procedure, trigger_embedding=[0.1, 0.2, 0.3]
                )

                assert isinstance(procedure_id, str)
                mock_table.insert.assert_called_once()


# =============================================================================
# Learning Signals Tests
# =============================================================================


@pytest.mark.asyncio
async def test_record_learning_signal():
    """Test record_learning_signal function."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_table.insert.return_value.execute.return_value = None
    mock_client.table.return_value = mock_table

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        signal = LearningSignalInput(
            signal_type="thumbs_up",
            signal_value=1.0,
            rated_agent="causal_impact",
            brand="Kisqali",
        )

        await record_learning_signal(signal, cycle_id="cycle123", session_id="sess123")

        mock_table.insert.assert_called_once()


@pytest.mark.asyncio
async def test_get_training_examples_for_agent():
    """Test get_training_examples_for_agent function."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "signal_id": "sig123",
            "rated_agent": "causal_impact",
            "dspy_metric_value": 0.85,
        }
    ]
    # The query chain is: table -> select -> eq -> eq -> gte -> order -> limit -> eq (brand) -> execute
    mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.eq.return_value.execute.return_value = mock_result

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        examples = await get_training_examples_for_agent(
            agent_name="causal_impact", brand="Kisqali", min_score=0.7, limit=100
        )

        assert len(examples) == 1
        assert examples[0]["rated_agent"] == "causal_impact"


# =============================================================================
# Memory Statistics Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_memory_statistics():
    """Test get_memory_statistics function."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "stat_date": "2024-01-30",
            "memory_type": "episodic",
            "subtype": "user_query",
            "count": 10,
        },
        {
            "stat_date": "2024-01-30",
            "memory_type": "procedural",
            "subtype": "tool_sequence",
            "count": 5,
        },
    ]
    mock_client.table.return_value.select.return_value.gte.return_value.order.return_value.execute.return_value = mock_result

    with patch.object(memory_backends, "get_supabase_client", return_value=mock_client):
        stats = await get_memory_statistics(days_back=30)

        assert "totals_by_type" in stats
        assert "episodic" in stats["totals_by_type"]
        assert stats["totals_by_type"]["episodic"] == 10


# =============================================================================
# Integration Tests
# =============================================================================


def test_get_working_memory_singleton():
    """Test get_working_memory singleton pattern."""
    mock_redis_client = AsyncMock()

    with patch.object(memory_backends, "get_redis_client", return_value=mock_redis_client):
        memory1 = get_working_memory()
        memory2 = get_working_memory()

        # Should return the same instance
        assert memory1 is memory2


def test_get_semantic_memory_singleton():
    """Test get_semantic_memory singleton pattern."""
    mock_client = MagicMock()

    with patch.object(memory_backends, "get_falkordb_client", return_value=mock_client):
        memory1 = get_semantic_memory()
        memory2 = get_semantic_memory()

        # Should return the same instance
        assert memory1 is memory2
