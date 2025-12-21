"""
Unit tests for E2I Agentic Memory - Working Memory.

Tests focus on:
- Session management (create, get, update, delete)
- E2I context management (brand, region, patient/HCP IDs)
- Message history (add, get, clear)
- Evidence board (append, get trail, clear)
- Workflow phase management
- LangGraph checkpointer integration

All tests use mocked Redis client to avoid external dependencies.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.working_memory import (
    RedisWorkingMemory,
    get_langgraph_checkpointer,
    get_working_memory,
    reset_working_memory,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.hset = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hincrby = AsyncMock()
    redis.expire = AsyncMock()
    redis.delete = AsyncMock(return_value=1)
    redis.rpush = AsyncMock()
    redis.lrange = AsyncMock(return_value=[])
    redis.ltrim = AsyncMock()
    redis.ping = AsyncMock()
    return redis


@pytest.fixture
def working_memory(mock_redis):
    """Create a RedisWorkingMemory instance with mocked Redis."""
    reset_working_memory()
    wm = RedisWorkingMemory()
    wm._client = mock_redis
    return wm


# ============================================================================
# SESSION MANAGEMENT TESTS
# ============================================================================


class TestSessionManagement:
    """Tests for session create, get, update, delete operations."""

    @pytest.mark.asyncio
    async def test_create_session_returns_uuid(self, working_memory, mock_redis):
        """create_session should return a valid UUID string."""
        session_id = await working_memory.create_session(user_id="user123")

        assert session_id is not None
        assert len(session_id) == 36  # UUID format
        assert "-" in session_id

    @pytest.mark.asyncio
    async def test_create_session_stores_user_id(self, working_memory, mock_redis):
        """create_session should store the user_id in Redis."""
        await working_memory.create_session(user_id="user123")

        # Check that hset was called with user_id
        call_args = mock_redis.hset.call_args
        assert call_args is not None
        mapping = call_args.kwargs.get("mapping", {})
        assert mapping.get("user_id") == "user123"

    @pytest.mark.asyncio
    async def test_create_session_with_initial_context(self, working_memory, mock_redis):
        """create_session should store initial context."""
        context = {
            "preferences": {"theme": "dark"},
            "filters": {"date_range": "last_30_days"},
            "brand": "Remibrutinib",
            "region": "northeast",
        }
        await working_memory.create_session(user_id="user123", initial_context=context)

        call_args = mock_redis.hset.call_args
        mapping = call_args.kwargs.get("mapping", {})
        assert json.loads(mapping.get("user_preferences", "{}")) == {"theme": "dark"}
        assert json.loads(mapping.get("active_filters", "{}")) == {"date_range": "last_30_days"}
        assert mapping.get("active_brand") == "Remibrutinib"
        assert mapping.get("active_region") == "northeast"

    @pytest.mark.asyncio
    async def test_create_session_sets_ttl(self, working_memory, mock_redis):
        """create_session should set TTL on the session key."""
        await working_memory.create_session(user_id="user123")

        mock_redis.expire.assert_called()
        # Check that expire was called with TTL from config
        call_args = mock_redis.expire.call_args
        assert call_args[0][1] == working_memory.ttl_seconds

    @pytest.mark.asyncio
    async def test_create_session_anonymous_user(self, working_memory, mock_redis):
        """create_session without user_id should use 'anonymous'."""
        await working_memory.create_session()

        call_args = mock_redis.hset.call_args
        mapping = call_args.kwargs.get("mapping", {})
        assert mapping.get("user_id") == "anonymous"

    @pytest.mark.asyncio
    async def test_get_session_returns_data(self, working_memory, mock_redis):
        """get_session should return session data from Redis."""
        mock_redis.hgetall.return_value = {
            "session_id": "test-session-id",
            "user_id": "user123",
            "created_at": "2025-01-01T00:00:00",
            "message_count": "5",
            "user_preferences": json.dumps({"theme": "dark"}),
            "active_filters": json.dumps({"brand": "Remibrutinib"}),
        }

        session = await working_memory.get_session("test-session-id")

        assert session is not None
        assert session["session_id"] == "test-session-id"
        assert session["user_id"] == "user123"
        assert session["message_count"] == 5  # Should be converted to int
        assert session["user_preferences"] == {"theme": "dark"}  # Should be parsed

    @pytest.mark.asyncio
    async def test_get_session_returns_none_for_missing(self, working_memory, mock_redis):
        """get_session should return None for non-existent session."""
        mock_redis.hgetall.return_value = {}

        session = await working_memory.get_session("nonexistent")

        assert session is None

    @pytest.mark.asyncio
    async def test_update_session_sets_fields(self, working_memory, mock_redis):
        """update_session should update specified fields."""
        await working_memory.update_session(
            "test-session-id", {"current_phase": "investigator", "custom_field": "value"}
        )

        mock_redis.hset.assert_called()
        call_args = mock_redis.hset.call_args
        mapping = call_args.kwargs.get("mapping", {})
        assert mapping.get("current_phase") == "investigator"
        assert mapping.get("custom_field") == "value"
        assert "last_activity_at" in mapping  # Should be updated

    @pytest.mark.asyncio
    async def test_update_session_serializes_dicts(self, working_memory, mock_redis):
        """update_session should serialize dict fields to JSON."""
        await working_memory.update_session(
            "test-session-id", {"user_preferences": {"new_pref": True}}
        )

        call_args = mock_redis.hset.call_args
        mapping = call_args.kwargs.get("mapping", {})
        assert mapping.get("user_preferences") == json.dumps({"new_pref": True})

    @pytest.mark.asyncio
    async def test_update_session_refreshes_ttl(self, working_memory, mock_redis):
        """update_session should refresh the session TTL."""
        await working_memory.update_session("test-session-id", {"field": "value"})

        mock_redis.expire.assert_called()

    @pytest.mark.asyncio
    async def test_delete_session_removes_all_keys(self, working_memory, mock_redis):
        """delete_session should remove session and related keys."""
        mock_redis.delete.return_value = 1

        result = await working_memory.delete_session("test-session-id")

        assert result is True
        # Should delete session, messages, and evidence keys
        assert mock_redis.delete.call_count >= 1

    @pytest.mark.asyncio
    async def test_delete_session_returns_false_for_missing(self, working_memory, mock_redis):
        """delete_session should return False if nothing was deleted."""
        mock_redis.delete.return_value = 0

        result = await working_memory.delete_session("nonexistent")

        assert result is False


# ============================================================================
# E2I CONTEXT TESTS
# ============================================================================


class TestE2IContext:
    """Tests for E2I entity context management."""

    @pytest.mark.asyncio
    async def test_set_e2i_context_brand(self, working_memory, mock_redis):
        """set_e2i_context should store brand."""
        await working_memory.set_e2i_context("test-session-id", brand="Fabhalta")

        call_args = mock_redis.hset.call_args
        mapping = call_args.kwargs.get("mapping", {})
        assert mapping.get("active_brand") == "Fabhalta"

    @pytest.mark.asyncio
    async def test_set_e2i_context_region(self, working_memory, mock_redis):
        """set_e2i_context should store region."""
        await working_memory.set_e2i_context("test-session-id", region="midwest")

        call_args = mock_redis.hset.call_args
        mapping = call_args.kwargs.get("mapping", {})
        assert mapping.get("active_region") == "midwest"

    @pytest.mark.asyncio
    async def test_set_e2i_context_patient_ids(self, working_memory, mock_redis):
        """set_e2i_context should store patient IDs as JSON."""
        await working_memory.set_e2i_context(
            "test-session-id", patient_ids=["P001", "P002", "P003"]
        )

        call_args = mock_redis.hset.call_args
        mapping = call_args.kwargs.get("mapping", {})
        assert json.loads(mapping.get("active_patient_ids", "[]")) == ["P001", "P002", "P003"]

    @pytest.mark.asyncio
    async def test_set_e2i_context_hcp_ids(self, working_memory, mock_redis):
        """set_e2i_context should store HCP IDs as JSON."""
        await working_memory.set_e2i_context("test-session-id", hcp_ids=["HCP001", "HCP002"])

        call_args = mock_redis.hset.call_args
        mapping = call_args.kwargs.get("mapping", {})
        assert json.loads(mapping.get("active_hcp_ids", "[]")) == ["HCP001", "HCP002"]

    @pytest.mark.asyncio
    async def test_set_e2i_context_all_fields(self, working_memory, mock_redis):
        """set_e2i_context should store all fields together."""
        await working_memory.set_e2i_context(
            "test-session-id",
            brand="Kisqali",
            region="south",
            patient_ids=["P100"],
            hcp_ids=["HCP100"],
        )

        call_args = mock_redis.hset.call_args
        mapping = call_args.kwargs.get("mapping", {})
        assert mapping.get("active_brand") == "Kisqali"
        assert mapping.get("active_region") == "south"
        assert "active_patient_ids" in mapping
        assert "active_hcp_ids" in mapping

    @pytest.mark.asyncio
    async def test_set_e2i_context_no_update_when_empty(self, working_memory, mock_redis):
        """set_e2i_context should not call Redis if no updates."""
        await working_memory.set_e2i_context("test-session-id")

        # hset should not be called if no updates
        mock_redis.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_e2i_context_returns_context(self, working_memory, mock_redis):
        """get_e2i_context should return E2I context from session."""
        mock_redis.hgetall.return_value = {
            "active_brand": "Remibrutinib",
            "active_region": "northeast",
            "active_patient_ids": json.dumps(["P001", "P002"]),
            "active_hcp_ids": json.dumps(["HCP001"]),
        }

        context = await working_memory.get_e2i_context("test-session-id")

        assert context["brand"] == "Remibrutinib"
        assert context["region"] == "northeast"
        assert context["patient_ids"] == ["P001", "P002"]
        assert context["hcp_ids"] == ["HCP001"]

    @pytest.mark.asyncio
    async def test_get_e2i_context_returns_empty_for_missing(self, working_memory, mock_redis):
        """get_e2i_context should return empty dict for missing session."""
        mock_redis.hgetall.return_value = {}

        context = await working_memory.get_e2i_context("nonexistent")

        assert context == {}


# ============================================================================
# MESSAGE HISTORY TESTS
# ============================================================================


class TestMessageHistory:
    """Tests for message history operations."""

    @pytest.mark.asyncio
    async def test_add_message_stores_message(self, working_memory, mock_redis):
        """add_message should store message in Redis list."""
        await working_memory.add_message(
            "test-session-id", role="user", content="Why did TRx drop?"
        )

        mock_redis.rpush.assert_called()
        call_args = mock_redis.rpush.call_args
        message_json = call_args[0][1]
        message = json.loads(message_json)

        assert message["role"] == "user"
        assert message["content"] == "Why did TRx drop?"
        assert "timestamp" in message

    @pytest.mark.asyncio
    async def test_add_message_with_metadata(self, working_memory, mock_redis):
        """add_message should store metadata."""
        await working_memory.add_message(
            "test-session-id",
            role="assistant",
            content="The drop was caused by...",
            metadata={"agent": "causal_impact", "confidence": 0.85},
        )

        call_args = mock_redis.rpush.call_args
        message_json = call_args[0][1]
        message = json.loads(message_json)
        metadata = json.loads(message["metadata"])

        assert metadata["agent"] == "causal_impact"
        assert metadata["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_add_message_trims_to_window(self, working_memory, mock_redis):
        """add_message should trim list to context window size."""
        await working_memory.add_message("test-session-id", role="user", content="Test message")

        mock_redis.ltrim.assert_called()
        # Should trim to keep last N messages
        call_args = mock_redis.ltrim.call_args
        assert call_args[0][1] == -working_memory.context_window_messages
        assert call_args[0][2] == -1

    @pytest.mark.asyncio
    async def test_add_message_increments_count(self, working_memory, mock_redis):
        """add_message should increment message_count in session."""
        await working_memory.add_message("test-session-id", role="user", content="Test")

        mock_redis.hincrby.assert_called()
        call_args = mock_redis.hincrby.call_args
        assert call_args[0][1] == "message_count"
        assert call_args[0][2] == 1

    @pytest.mark.asyncio
    async def test_get_messages_returns_all(self, working_memory, mock_redis):
        """get_messages without limit should return all messages."""
        mock_redis.lrange.return_value = [
            json.dumps(
                {
                    "role": "user",
                    "content": "Q1",
                    "timestamp": "2025-01-01T00:00:00",
                    "metadata": "{}",
                }
            ),
            json.dumps(
                {
                    "role": "assistant",
                    "content": "A1",
                    "timestamp": "2025-01-01T00:00:01",
                    "metadata": "{}",
                }
            ),
        ]

        messages = await working_memory.get_messages("test-session-id")

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Q1"
        assert messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self, working_memory, mock_redis):
        """get_messages with limit should return limited messages."""
        mock_redis.lrange.return_value = [
            json.dumps(
                {
                    "role": "user",
                    "content": "Last Q",
                    "timestamp": "2025-01-01T00:00:00",
                    "metadata": "{}",
                }
            ),
        ]

        await working_memory.get_messages("test-session-id", limit=1)

        mock_redis.lrange.assert_called()
        call_args = mock_redis.lrange.call_args
        # Should use negative indices for last N messages
        assert call_args[0][1] == -1
        assert call_args[0][2] == -1

    @pytest.mark.asyncio
    async def test_get_messages_deserializes_metadata(self, working_memory, mock_redis):
        """get_messages should deserialize metadata JSON."""
        mock_redis.lrange.return_value = [
            json.dumps(
                {
                    "role": "assistant",
                    "content": "Answer",
                    "timestamp": "2025-01-01T00:00:00",
                    "metadata": json.dumps({"key": "value"}),
                }
            ),
        ]

        messages = await working_memory.get_messages("test-session-id")

        assert messages[0]["metadata"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_clear_messages_deletes_list(self, working_memory, mock_redis):
        """clear_messages should delete the messages list."""
        await working_memory.clear_messages("test-session-id")

        mock_redis.delete.assert_called()

    @pytest.mark.asyncio
    async def test_clear_messages_resets_count(self, working_memory, mock_redis):
        """clear_messages should reset message_count to 0."""
        await working_memory.clear_messages("test-session-id")

        mock_redis.hset.assert_called()
        call_args = mock_redis.hset.call_args
        assert call_args[0][1] == "message_count"
        assert call_args[0][2] == "0"


# ============================================================================
# EVIDENCE BOARD TESTS
# ============================================================================


class TestEvidenceBoard:
    """Tests for evidence board operations."""

    @pytest.mark.asyncio
    async def test_append_evidence_stores_item(self, working_memory, mock_redis):
        """append_evidence should store evidence in Redis list."""
        evidence = {
            "source": "causal_impact",
            "content": "TRx drop linked to competitor launch",
            "relevance": 0.92,
        }

        await working_memory.append_evidence("test-session-id", evidence)

        mock_redis.rpush.assert_called()
        call_args = mock_redis.rpush.call_args
        evidence_json = call_args[0][1]
        stored = json.loads(evidence_json)

        assert stored["source"] == "causal_impact"
        assert stored["relevance"] == 0.92
        assert "timestamp" in stored  # Should be added

    @pytest.mark.asyncio
    async def test_append_evidence_preserves_timestamp(self, working_memory, mock_redis):
        """append_evidence should preserve existing timestamp."""
        evidence = {
            "source": "gap_analyzer",
            "content": "Revenue gap identified",
            "timestamp": "2025-01-01T12:00:00",
        }

        await working_memory.append_evidence("test-session-id", evidence)

        call_args = mock_redis.rpush.call_args
        stored = json.loads(call_args[0][1])
        assert stored["timestamp"] == "2025-01-01T12:00:00"

    @pytest.mark.asyncio
    async def test_get_evidence_trail_returns_all(self, working_memory, mock_redis):
        """get_evidence_trail should return all evidence items."""
        mock_redis.lrange.return_value = [
            json.dumps({"source": "source1", "content": "Evidence 1"}),
            json.dumps({"source": "source2", "content": "Evidence 2"}),
        ]

        trail = await working_memory.get_evidence_trail("test-session-id")

        assert len(trail) == 2
        assert trail[0]["source"] == "source1"
        assert trail[1]["source"] == "source2"

    @pytest.mark.asyncio
    async def test_get_evidence_trail_empty(self, working_memory, mock_redis):
        """get_evidence_trail should return empty list for no evidence."""
        mock_redis.lrange.return_value = []

        trail = await working_memory.get_evidence_trail("test-session-id")

        assert trail == []

    @pytest.mark.asyncio
    async def test_clear_evidence_deletes_key(self, working_memory, mock_redis):
        """clear_evidence should delete the evidence key."""
        await working_memory.clear_evidence("test-session-id")

        mock_redis.delete.assert_called()

    @pytest.mark.asyncio
    async def test_get_evidence_summary_returns_stats(self, working_memory, mock_redis):
        """get_evidence_summary should return evidence statistics."""
        mock_redis.lrange.return_value = [
            json.dumps({"source": "causal_impact", "relevance": 0.9}),
            json.dumps({"source": "gap_analyzer", "relevance": 0.7}),
            json.dumps({"source": "causal_impact", "relevance": 0.8}),
        ]

        summary = await working_memory.get_evidence_summary("test-session-id")

        assert summary["count"] == 3
        assert set(summary["sources"]) == {"causal_impact", "gap_analyzer"}
        assert summary["max_relevance"] == 0.9
        assert abs(summary["avg_relevance"] - 0.8) < 0.01

    @pytest.mark.asyncio
    async def test_get_evidence_summary_empty(self, working_memory, mock_redis):
        """get_evidence_summary should handle empty evidence."""
        mock_redis.lrange.return_value = []

        summary = await working_memory.get_evidence_summary("test-session-id")

        assert summary["count"] == 0
        assert summary["sources"] == []
        assert summary["max_relevance"] == 0.0


# ============================================================================
# WORKFLOW PHASE TESTS
# ============================================================================


class TestWorkflowPhase:
    """Tests for workflow phase management."""

    @pytest.mark.asyncio
    async def test_set_workflow_phase(self, working_memory, mock_redis):
        """set_workflow_phase should update current_phase."""
        await working_memory.set_workflow_phase("test-session-id", "investigator")

        mock_redis.hset.assert_called()
        call_args = mock_redis.hset.call_args
        mapping = call_args.kwargs.get("mapping", {})
        assert mapping.get("current_phase") == "investigator"

    @pytest.mark.asyncio
    async def test_get_workflow_phase_returns_phase(self, working_memory, mock_redis):
        """get_workflow_phase should return current phase."""
        mock_redis.hgetall.return_value = {
            "current_phase": "agent",
        }

        phase = await working_memory.get_workflow_phase("test-session-id")

        assert phase == "agent"

    @pytest.mark.asyncio
    async def test_get_workflow_phase_returns_none_for_missing(self, working_memory, mock_redis):
        """get_workflow_phase should return None for missing session."""
        mock_redis.hgetall.return_value = {}

        phase = await working_memory.get_workflow_phase("nonexistent")

        assert phase is None


# ============================================================================
# LANGGRAPH CHECKPOINTER TESTS
# ============================================================================


class TestLangGraphCheckpointer:
    """Tests for LangGraph checkpointer integration."""

    def test_get_langgraph_checkpointer_returns_checkpointer(self, working_memory):
        """get_langgraph_checkpointer should return a checkpointer."""
        # Reset checkpointer cache
        working_memory._checkpointer = None

        checkpointer = working_memory.get_langgraph_checkpointer()

        # Should return something (either RedisSaver or MemorySaver fallback)
        assert checkpointer is not None

    def test_get_langgraph_checkpointer_caches_instance(self, working_memory):
        """get_langgraph_checkpointer should cache the checkpointer."""
        # Reset checkpointer cache
        working_memory._checkpointer = None

        checkpointer1 = working_memory.get_langgraph_checkpointer()
        checkpointer2 = working_memory.get_langgraph_checkpointer()

        assert checkpointer1 is checkpointer2

    def test_get_langgraph_checkpointer_fallback_to_memory(self, working_memory):
        """get_langgraph_checkpointer should fall back to MemorySaver when Redis unavailable."""
        # Reset checkpointer cache
        working_memory._checkpointer = None

        # Since langgraph-checkpoint-redis is likely not installed in test env,
        # the fallback to MemorySaver should happen automatically
        checkpointer = working_memory.get_langgraph_checkpointer()

        # Should be a MemorySaver instance (fallback)
        from langgraph.checkpoint.memory import MemorySaver

        assert isinstance(checkpointer, MemorySaver)


# ============================================================================
# SINGLETON AND FACTORY TESTS
# ============================================================================


class TestSingletonAndFactory:
    """Tests for singleton pattern and factory functions."""

    def test_get_working_memory_returns_singleton(self):
        """get_working_memory should return same instance."""
        reset_working_memory()

        wm1 = get_working_memory()
        wm2 = get_working_memory()

        assert wm1 is wm2

    def test_reset_working_memory_clears_singleton(self):
        """reset_working_memory should clear the singleton."""
        wm1 = get_working_memory()
        reset_working_memory()
        wm2 = get_working_memory()

        assert wm1 is not wm2

    def test_get_langgraph_checkpointer_uses_singleton(self):
        """get_langgraph_checkpointer should use working memory singleton."""
        reset_working_memory()

        with patch.object(RedisWorkingMemory, "get_langgraph_checkpointer") as mock_method:
            mock_method.return_value = MagicMock()

            get_langgraph_checkpointer()

            mock_method.assert_called_once()


# ============================================================================
# PROPERTY TESTS
# ============================================================================


class TestProperties:
    """Tests for configuration properties."""

    def test_session_prefix_from_config(self, working_memory):
        """session_prefix should come from config."""
        prefix = working_memory.session_prefix
        assert prefix is not None
        assert isinstance(prefix, str)

    def test_evidence_prefix_from_config(self, working_memory):
        """evidence_prefix should come from config."""
        prefix = working_memory.evidence_prefix
        assert prefix is not None
        assert isinstance(prefix, str)

    def test_ttl_seconds_from_config(self, working_memory):
        """ttl_seconds should come from config."""
        ttl = working_memory.ttl_seconds
        assert ttl is not None
        assert isinstance(ttl, int)
        assert ttl > 0

    def test_context_window_messages_from_config(self, working_memory):
        """context_window_messages should come from config."""
        window = working_memory.context_window_messages
        assert window is not None
        assert isinstance(window, int)
        assert window > 0
