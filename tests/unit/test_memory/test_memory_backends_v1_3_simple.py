"""
Simple comprehensive unit tests for src/memory/006_memory_backends_v1_3.py

Tests the main functionality with mocked external dependencies.
"""

import importlib
import json
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock config BEFORE importing the module
MOCK_CONFIG = {
    "environment": "local_pilot",
    "embeddings": {
        "local_pilot": {"model": "text-embedding-3-small"},
    },
    "llm": {
        "local_pilot": {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4096,
            "temperature": 0.7,
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
                    "entity_types": ["Patient", "HCP"],
                    "relationship_types": ["TREATED_BY", "PRESCRIBES"],
                },
            }
        },
    },
}


# Patch the load_config function before importing
with patch("builtins.open", MagicMock()):
    with patch("yaml.safe_load", return_value=MOCK_CONFIG):
        # Import the module with numeric prefix using importlib
        mb = importlib.import_module("src.memory.006_memory_backends_v1_3")


@pytest.fixture(autouse=True)
def mock_config_fixture():
    """Mock configuration for all tests."""
    with patch.object(mb, "CONFIG", MOCK_CONFIG):
        with patch.object(mb, "ENVIRONMENT", "local_pilot"):
            yield


# Test enum values
def test_entity_types():
    """Test that entity type enums exist."""
    assert mb.E2IEntityType.PATIENT.value == "patient"
    assert mb.E2IEntityType.HCP.value == "hcp"
    assert mb.E2IBrand.KISQALI.value == "Kisqali"
    assert mb.E2IRegion.NORTHEAST.value == "northeast"


# Test dataclasses
def test_entity_context_creation():
    """Test E2IEntityContext creation."""
    context = mb.E2IEntityContext(
        patient={"id": "PAT123"},
        hcp={"id": "HCP456"},
    )
    assert context.patient["id"] == "PAT123"
    assert context.hcp["id"] == "HCP456"


def test_entity_references_creation():
    """Test E2IEntityReferences creation."""
    refs = mb.E2IEntityReferences(
        patient_id="PAT123",
        brand="Kisqali",
    )
    assert refs.patient_id == "PAT123"
    assert refs.brand == "Kisqali"


# Test service factories
@patch.object(mb, "OpenAIEmbeddingService")
def test_get_embedding_service(mock_service):
    """Test embedding service factory."""
    mock_instance = MagicMock()
    mock_service.return_value = mock_instance

    service = mb.get_embedding_service()

    mock_service.assert_called_once()
    assert service == mock_instance


@patch.dict("os.environ", {"REDIS_URL": "redis://test:6382"})
@patch("redis.asyncio.from_url")
def test_get_redis_client(mock_from_url):
    """Test Redis client factory."""
    mock_client = MagicMock()
    mock_from_url.return_value = mock_client

    client = mb.get_redis_client()

    mock_from_url.assert_called_once_with("redis://test:6382", decode_responses=True)
    assert client == mock_client


@patch.dict("os.environ", {"SUPABASE_URL": "http://test", "SUPABASE_ANON_KEY": "key"})
@patch("supabase.create_client")
def test_get_supabase_client(mock_create):
    """Test Supabase client factory."""
    mock_client = MagicMock()
    mock_create.return_value = mock_client

    client = mb.get_supabase_client()

    mock_create.assert_called_once_with("http://test", "key")
    assert client == mock_client


def test_get_supabase_client_no_env():
    """Test Supabase client fails without env vars."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="SUPABASE_URL"):
            mb.get_supabase_client()


# Test OpenAIEmbeddingService
@patch("openai.OpenAI")
def test_openai_service_init(mock_openai_class):
    """Test OpenAI service initialization."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    service = mb.OpenAIEmbeddingService()

    mock_openai_class.assert_called_once()
    assert service.client == mock_client
    assert service.model == "text-embedding-3-small"


@patch("openai.OpenAI")
@pytest.mark.asyncio
async def test_openai_embed(mock_openai_class):
    """Test OpenAI embedding generation."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_client.embeddings.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    service = mb.OpenAIEmbeddingService()
    embedding = await service.embed("test")

    assert embedding == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input="test",
    )


# Test RedisWorkingMemory
@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    client = AsyncMock()
    client.hgetall = AsyncMock(return_value={})
    client.hset = AsyncMock()
    client.expire = AsyncMock()
    return client


@patch.object(mb, "get_redis_client")
@pytest.mark.asyncio
async def test_working_memory_create_session(mock_get_redis, mock_redis_client):
    """Test session creation."""
    mock_get_redis.return_value = mock_redis_client

    memory = mb.RedisWorkingMemory()
    session_id = await memory.create_session(user_id="user123")

    assert isinstance(session_id, str)
    mock_redis_client.hset.assert_called_once()


@patch.object(mb, "get_redis_client")
@pytest.mark.asyncio
async def test_working_memory_get_session(mock_get_redis, mock_redis_client):
    """Test session retrieval."""
    mock_redis_client.hgetall = AsyncMock(return_value={
        "session_id": "sess123",
        "user_id": "user123",
        "message_count": "5",
    })
    mock_get_redis.return_value = mock_redis_client

    memory = mb.RedisWorkingMemory()
    session = await memory.get_session("sess123")

    assert session["session_id"] == "sess123"
    assert session["message_count"] == 5


# Test episodic memory
@patch.object(mb, "get_supabase_client")
@pytest.mark.asyncio
async def test_search_episodic_memory(mock_get_supabase):
    """Test episodic memory search."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [{"memory_id": "mem123", "similarity": 0.9}]
    mock_client.rpc.return_value.execute.return_value = mock_result
    mock_get_supabase.return_value = mock_client

    memories = await mb.search_episodic_memory(
        embedding=[0.1, 0.2],
        limit=10,
    )

    assert len(memories) == 1
    assert memories[0]["memory_id"] == "mem123"


@patch.object(mb, "get_supabase_client")
@patch.object(mb, "_increment_memory_stats")
@pytest.mark.asyncio
async def test_insert_episodic_memory(mock_stats, mock_get_supabase):
    """Test episodic memory insertion."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_table.insert.return_value.execute.return_value = None
    mock_client.table.return_value = mock_table
    mock_get_supabase.return_value = mock_client

    memory = mb.EpisodicMemoryInput(
        event_type="user_query",
        description="Test",
    )

    memory_id = await mb.insert_episodic_memory(memory, embedding=[0.1, 0.2])

    assert isinstance(memory_id, str)
    mock_table.insert.assert_called_once()


# Test semantic memory
@patch.object(mb, "get_falkordb_client")
def test_semantic_memory_add_entity(mock_get_client):
    """Test adding entity to semantic graph."""
    mock_client = MagicMock()
    mock_graph = MagicMock()
    mock_graph.query.return_value = MagicMock()
    mock_client.select_graph.return_value = mock_graph
    mock_get_client.return_value = mock_client

    memory = mb.FalkorDBSemanticMemory()
    result = memory.add_e2i_entity(
        entity_type=mb.E2IEntityType.PATIENT,
        entity_id="PAT123",
        properties={"name": "Test"},
    )

    assert result is True
    mock_graph.query.assert_called_once()


# Test procedural memory
@patch.object(mb, "get_supabase_client")
@pytest.mark.asyncio
async def test_find_procedures(mock_get_supabase):
    """Test finding relevant procedures."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [{"procedure_id": "proc123"}]
    mock_client.rpc.return_value.execute.return_value = mock_result
    mock_get_supabase.return_value = mock_client

    procedures = await mb.find_relevant_procedures(
        embedding=[0.1, 0.2],
        limit=5,
    )

    assert len(procedures) == 1


# Test learning signals
@patch.object(mb, "get_supabase_client")
@pytest.mark.asyncio
async def test_record_learning_signal(mock_get_supabase):
    """Test recording learning signal."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_table.insert.return_value.execute.return_value = None
    mock_client.table.return_value = mock_table
    mock_get_supabase.return_value = mock_client

    signal = mb.LearningSignalInput(
        signal_type="thumbs_up",
        signal_value=1.0,
    )

    await mb.record_learning_signal(signal)

    mock_table.insert.assert_called_once()


# Test statistics
@patch.object(mb, "get_supabase_client")
@pytest.mark.asyncio
async def test_get_memory_statistics(mock_get_supabase):
    """Test memory statistics retrieval."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "stat_date": "2024-01-30",
            "memory_type": "episodic",
            "count": 10,
        }
    ]
    mock_client.table.return_value.select.return_value.gte.return_value.order.return_value.execute.return_value = mock_result
    mock_get_supabase.return_value = mock_client

    stats = await mb.get_memory_statistics(days_back=7)

    assert "totals_by_type" in stats
    assert "episodic" in stats["totals_by_type"]
