"""
Integration tests for Redis-backed Working Memory.

These tests require a running Redis instance.
Skip if Redis is not available.

Run with: pytest tests/integration/test_memory/test_redis_integration.py -v

Environment variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
"""

import os
import json
import pytest
import asyncio
from datetime import datetime

# Check if Redis is available before running tests
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not REDIS_AVAILABLE,
    reason="redis package not installed"
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def redis_client():
    """Create a real Redis client for testing."""
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    client = redis.from_url(url, decode_responses=True)

    # Check if Redis is actually running
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis server not available")

    yield client

    # Cleanup: delete all test keys
    keys = await client.keys("e2i:test:*")
    if keys:
        await client.delete(*keys)
    await client.aclose()


@pytest.fixture
async def working_memory(redis_client):
    """Create a working memory instance with test prefixes."""
    from src.memory.working_memory import RedisWorkingMemory, reset_working_memory

    reset_working_memory()
    wm = RedisWorkingMemory()
    wm._client = redis_client

    # Use test prefixes to avoid conflicts
    wm._working_config.session_prefix = "e2i:test:session:"
    wm._working_config.evidence_prefix = "e2i:test:evidence:"

    yield wm

    # Cleanup sessions created during test
    keys = await redis_client.keys("e2i:test:*")
    if keys:
        await redis_client.delete(*keys)

    reset_working_memory()


# ============================================================================
# CONNECTION TESTS
# ============================================================================


class TestRedisConnection:
    """Tests for Redis connection health."""

    @pytest.mark.asyncio
    async def test_redis_ping(self, redis_client):
        """Redis should respond to ping."""
        result = await redis_client.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_set_get(self, redis_client):
        """Redis should set and get values."""
        await redis_client.set("e2i:test:key", "value")
        result = await redis_client.get("e2i:test:key")
        assert result == "value"
        await redis_client.delete("e2i:test:key")


# ============================================================================
# SESSION INTEGRATION TESTS
# ============================================================================


class TestSessionIntegration:
    """Integration tests for session management."""

    @pytest.mark.asyncio
    async def test_create_and_retrieve_session(self, working_memory):
        """Should create a session and retrieve it."""
        session_id = await working_memory.create_session(
            user_id="integration_test_user",
            initial_context={
                "preferences": {"language": "en"},
                "brand": "Remibrutinib",
                "region": "northeast"
            }
        )

        # Retrieve the session
        session = await working_memory.get_session(session_id)

        assert session is not None
        assert session["session_id"] == session_id
        assert session["user_id"] == "integration_test_user"
        assert session["user_preferences"]["language"] == "en"
        assert session.get("active_brand") == "Remibrutinib"
        assert session.get("active_region") == "northeast"

    @pytest.mark.asyncio
    async def test_session_ttl_is_set(self, working_memory, redis_client):
        """Session should have TTL set."""
        session_id = await working_memory.create_session(user_id="ttl_test")
        session_key = f"{working_memory.session_prefix}{session_id}"

        ttl = await redis_client.ttl(session_key)

        assert ttl > 0
        assert ttl <= working_memory.ttl_seconds

    @pytest.mark.asyncio
    async def test_update_session_preserves_data(self, working_memory):
        """Update should preserve existing session data."""
        session_id = await working_memory.create_session(
            user_id="update_test",
            initial_context={"brand": "Fabhalta"}
        )

        # Update with new field
        await working_memory.update_session(session_id, {
            "current_phase": "investigator"
        })

        # Verify both old and new data exist
        session = await working_memory.get_session(session_id)
        assert session.get("active_brand") == "Fabhalta"
        assert session["current_phase"] == "investigator"

    @pytest.mark.asyncio
    async def test_delete_session_removes_all_data(self, working_memory, redis_client):
        """Delete should remove session and related keys."""
        session_id = await working_memory.create_session(user_id="delete_test")

        # Add some messages and evidence
        await working_memory.add_message(session_id, "user", "Test message")
        await working_memory.append_evidence(session_id, {"content": "Test evidence"})

        # Delete session
        result = await working_memory.delete_session(session_id)
        assert result is True

        # Verify all keys are gone
        session = await working_memory.get_session(session_id)
        messages = await working_memory.get_messages(session_id)
        evidence = await working_memory.get_evidence_trail(session_id)

        assert session is None
        assert messages == []
        assert evidence == []


# ============================================================================
# E2I CONTEXT INTEGRATION TESTS
# ============================================================================


class TestE2IContextIntegration:
    """Integration tests for E2I context management."""

    @pytest.mark.asyncio
    async def test_set_and_get_e2i_context(self, working_memory):
        """Should set and retrieve E2I context."""
        session_id = await working_memory.create_session(user_id="e2i_test")

        await working_memory.set_e2i_context(
            session_id,
            brand="Kisqali",
            region="south",
            patient_ids=["P001", "P002"],
            hcp_ids=["HCP001"]
        )

        context = await working_memory.get_e2i_context(session_id)

        assert context["brand"] == "Kisqali"
        assert context["region"] == "south"
        assert context["patient_ids"] == ["P001", "P002"]
        assert context["hcp_ids"] == ["HCP001"]

    @pytest.mark.asyncio
    async def test_e2i_context_partial_update(self, working_memory):
        """Partial E2I context update should preserve existing values."""
        session_id = await working_memory.create_session(user_id="partial_test")

        # Set initial context
        await working_memory.set_e2i_context(
            session_id,
            brand="Remibrutinib",
            region="midwest"
        )

        # Partial update - only brand
        await working_memory.set_e2i_context(
            session_id,
            brand="Fabhalta"
        )

        context = await working_memory.get_e2i_context(session_id)

        assert context["brand"] == "Fabhalta"
        assert context["region"] == "midwest"  # Should be preserved


# ============================================================================
# MESSAGE HISTORY INTEGRATION TESTS
# ============================================================================


class TestMessageHistoryIntegration:
    """Integration tests for message history."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_messages(self, working_memory):
        """Should add and retrieve messages in order."""
        session_id = await working_memory.create_session(user_id="msg_test")

        await working_memory.add_message(session_id, "user", "Question 1")
        await working_memory.add_message(session_id, "assistant", "Answer 1")
        await working_memory.add_message(session_id, "user", "Question 2")

        messages = await working_memory.get_messages(session_id)

        assert len(messages) == 3
        assert messages[0]["content"] == "Question 1"
        assert messages[1]["content"] == "Answer 1"
        assert messages[2]["content"] == "Question 2"

    @pytest.mark.asyncio
    async def test_message_metadata_roundtrip(self, working_memory):
        """Message metadata should survive storage and retrieval."""
        session_id = await working_memory.create_session(user_id="meta_test")

        metadata = {
            "agent": "causal_impact",
            "confidence": 0.85,
            "sources": ["kpi_data", "causal_graph"]
        }
        await working_memory.add_message(
            session_id,
            "assistant",
            "Analysis result",
            metadata=metadata
        )

        messages = await working_memory.get_messages(session_id)

        assert len(messages) == 1
        assert messages[0]["metadata"]["agent"] == "causal_impact"
        assert messages[0]["metadata"]["confidence"] == 0.85
        assert "causal_graph" in messages[0]["metadata"]["sources"]

    @pytest.mark.asyncio
    async def test_message_limit(self, working_memory):
        """get_messages with limit should return last N messages."""
        session_id = await working_memory.create_session(user_id="limit_test")

        for i in range(5):
            await working_memory.add_message(session_id, "user", f"Message {i}")

        messages = await working_memory.get_messages(session_id, limit=2)

        assert len(messages) == 2
        assert messages[0]["content"] == "Message 3"
        assert messages[1]["content"] == "Message 4"

    @pytest.mark.asyncio
    async def test_clear_messages(self, working_memory):
        """clear_messages should remove all messages."""
        session_id = await working_memory.create_session(user_id="clear_test")

        await working_memory.add_message(session_id, "user", "Message 1")
        await working_memory.add_message(session_id, "user", "Message 2")
        await working_memory.clear_messages(session_id)

        messages = await working_memory.get_messages(session_id)
        session = await working_memory.get_session(session_id)

        assert messages == []
        assert session["message_count"] == 0

    @pytest.mark.asyncio
    async def test_message_count_increments(self, working_memory):
        """Adding messages should increment message_count."""
        session_id = await working_memory.create_session(user_id="count_test")

        session = await working_memory.get_session(session_id)
        assert session["message_count"] == 0

        await working_memory.add_message(session_id, "user", "Msg 1")
        await working_memory.add_message(session_id, "user", "Msg 2")
        await working_memory.add_message(session_id, "user", "Msg 3")

        session = await working_memory.get_session(session_id)
        assert session["message_count"] == 3


# ============================================================================
# EVIDENCE BOARD INTEGRATION TESTS
# ============================================================================


class TestEvidenceBoardIntegration:
    """Integration tests for evidence board."""

    @pytest.mark.asyncio
    async def test_append_and_retrieve_evidence(self, working_memory):
        """Should append and retrieve evidence items."""
        session_id = await working_memory.create_session(user_id="evidence_test")

        await working_memory.append_evidence(session_id, {
            "source": "causal_impact",
            "content": "TRx drop linked to competitor launch",
            "relevance": 0.92
        })
        await working_memory.append_evidence(session_id, {
            "source": "gap_analyzer",
            "content": "Revenue gap in Q3",
            "relevance": 0.78
        })

        trail = await working_memory.get_evidence_trail(session_id)

        assert len(trail) == 2
        assert trail[0]["source"] == "causal_impact"
        assert trail[1]["source"] == "gap_analyzer"

    @pytest.mark.asyncio
    async def test_evidence_summary_calculation(self, working_memory):
        """Evidence summary should calculate correct statistics."""
        session_id = await working_memory.create_session(user_id="summary_test")

        await working_memory.append_evidence(session_id, {
            "source": "agent_a",
            "relevance": 0.9
        })
        await working_memory.append_evidence(session_id, {
            "source": "agent_b",
            "relevance": 0.7
        })
        await working_memory.append_evidence(session_id, {
            "source": "agent_a",
            "relevance": 0.8
        })

        summary = await working_memory.get_evidence_summary(session_id)

        assert summary["count"] == 3
        assert set(summary["sources"]) == {"agent_a", "agent_b"}
        assert summary["max_relevance"] == 0.9
        assert abs(summary["avg_relevance"] - 0.8) < 0.01

    @pytest.mark.asyncio
    async def test_clear_evidence(self, working_memory):
        """clear_evidence should remove all evidence."""
        session_id = await working_memory.create_session(user_id="clear_ev_test")

        await working_memory.append_evidence(session_id, {"content": "Evidence 1"})
        await working_memory.append_evidence(session_id, {"content": "Evidence 2"})
        await working_memory.clear_evidence(session_id)

        trail = await working_memory.get_evidence_trail(session_id)

        assert trail == []


# ============================================================================
# WORKFLOW PHASE INTEGRATION TESTS
# ============================================================================


class TestWorkflowPhaseIntegration:
    """Integration tests for workflow phase management."""

    @pytest.mark.asyncio
    async def test_set_and_get_workflow_phase(self, working_memory):
        """Should set and retrieve workflow phase."""
        session_id = await working_memory.create_session(user_id="phase_test")

        # Initial phase from session creation
        phase = await working_memory.get_workflow_phase(session_id)
        assert phase == "init"

        # Update phase
        await working_memory.set_workflow_phase(session_id, "summarizer")
        phase = await working_memory.get_workflow_phase(session_id)
        assert phase == "summarizer"

        # Update to another phase
        await working_memory.set_workflow_phase(session_id, "investigator")
        phase = await working_memory.get_workflow_phase(session_id)
        assert phase == "investigator"

    @pytest.mark.asyncio
    async def test_workflow_phase_progression(self, working_memory):
        """Should track full workflow phase progression."""
        session_id = await working_memory.create_session(user_id="progression_test")

        phases = ["summarizer", "investigator", "agent", "reflector"]
        for phase in phases:
            await working_memory.set_workflow_phase(session_id, phase)
            current = await working_memory.get_workflow_phase(session_id)
            assert current == phase


# ============================================================================
# CONCURRENT ACCESS TESTS
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_message_adds(self, working_memory):
        """Multiple concurrent message adds should work correctly."""
        session_id = await working_memory.create_session(user_id="concurrent_test")

        # Add 10 messages concurrently
        tasks = [
            working_memory.add_message(session_id, "user", f"Message {i}")
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        messages = await working_memory.get_messages(session_id)
        session = await working_memory.get_session(session_id)

        assert len(messages) == 10
        assert session["message_count"] == 10

    @pytest.mark.asyncio
    async def test_concurrent_evidence_appends(self, working_memory):
        """Multiple concurrent evidence appends should work correctly."""
        session_id = await working_memory.create_session(user_id="concurrent_ev_test")

        # Add 10 evidence items concurrently
        tasks = [
            working_memory.append_evidence(session_id, {"content": f"Evidence {i}"})
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        trail = await working_memory.get_evidence_trail(session_id)

        assert len(trail) == 10


# ============================================================================
# LANGGRAPH CHECKPOINTER INTEGRATION TESTS
# ============================================================================


class TestLangGraphCheckpointerIntegration:
    """Integration tests for LangGraph checkpointer."""

    @pytest.mark.asyncio
    async def test_checkpointer_creation(self, working_memory):
        """Should create a working checkpointer."""
        checkpointer = working_memory.get_langgraph_checkpointer()

        # Should return something (either RedisSaver or MemorySaver)
        assert checkpointer is not None

    @pytest.mark.asyncio
    async def test_checkpointer_is_cached(self, working_memory):
        """Checkpointer should be cached."""
        cp1 = working_memory.get_langgraph_checkpointer()
        cp2 = working_memory.get_langgraph_checkpointer()

        assert cp1 is cp2


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Performance tests for working memory operations."""

    @pytest.mark.asyncio
    async def test_session_create_latency(self, working_memory):
        """Session creation should be fast (<50ms)."""
        import time

        start = time.perf_counter()
        await working_memory.create_session(user_id="perf_test")
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 50, f"Session creation took {elapsed:.2f}ms (limit: 50ms)"

    @pytest.mark.asyncio
    async def test_session_read_latency(self, working_memory):
        """Session read should be fast (<50ms)."""
        import time

        session_id = await working_memory.create_session(user_id="perf_read_test")

        start = time.perf_counter()
        await working_memory.get_session(session_id)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 50, f"Session read took {elapsed:.2f}ms (limit: 50ms)"

    @pytest.mark.asyncio
    async def test_message_add_latency(self, working_memory):
        """Message add should be fast (<50ms)."""
        import time

        session_id = await working_memory.create_session(user_id="perf_msg_test")

        start = time.perf_counter()
        await working_memory.add_message(session_id, "user", "Test message")
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 50, f"Message add took {elapsed:.2f}ms (limit: 50ms)"
