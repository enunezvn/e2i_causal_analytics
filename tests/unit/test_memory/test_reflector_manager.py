"""
Unit tests for ReflectorTaskManager.

Tests:
- Task submission and tracking
- Timeout handling
- Error logging and counting
- Graceful shutdown
- Statistics reporting
"""

import asyncio
from unittest.mock import patch

import pytest

from src.memory.cognitive_integration import ReflectorTaskManager


class TestReflectorTaskManager:
    """Tests for ReflectorTaskManager."""

    @pytest.mark.asyncio
    async def test_submit_tracks_task(self):
        """Test that submitted tasks are tracked."""
        manager = ReflectorTaskManager(timeout_seconds=5.0)

        async def simple_task():
            await asyncio.sleep(0.1)

        await manager.submit(simple_task(), cycle_id="test-1")

        # Task should be pending initially
        assert manager.pending_count >= 0  # May complete quickly

        # Wait for task to complete
        await asyncio.sleep(0.2)
        assert manager.pending_count == 0

    @pytest.mark.asyncio
    async def test_successful_task_increments_counter(self):
        """Test that successful tasks increment success counter."""
        manager = ReflectorTaskManager(timeout_seconds=5.0)

        async def simple_task():
            await asyncio.sleep(0.01)

        await manager.submit(simple_task(), cycle_id="test-1")
        await asyncio.sleep(0.1)

        stats = manager.stats
        assert stats["succeeded"] == 1
        assert stats["failed"] == 0

    @pytest.mark.asyncio
    async def test_timeout_increments_failed_counter(self):
        """Test that timeouts increment failed and timeout counters."""
        manager = ReflectorTaskManager(timeout_seconds=0.05)

        async def slow_task():
            await asyncio.sleep(1.0)  # Will timeout

        await manager.submit(slow_task(), cycle_id="test-1")
        await asyncio.sleep(0.2)

        stats = manager.stats
        assert stats["failed"] == 1
        assert stats["timeouts"] == 1

    @pytest.mark.asyncio
    async def test_exception_increments_failed_counter(self):
        """Test that exceptions increment failed counter."""
        manager = ReflectorTaskManager(timeout_seconds=5.0)

        async def failing_task():
            raise ValueError("Test error")

        await manager.submit(failing_task(), cycle_id="test-1")
        await asyncio.sleep(0.1)

        stats = manager.stats
        assert stats["failed"] == 1
        assert stats["timeouts"] == 0

    @pytest.mark.asyncio
    async def test_wait_all_completes_tasks(self):
        """Test that wait_all waits for pending tasks."""
        manager = ReflectorTaskManager(timeout_seconds=5.0)

        async def task_with_delay():
            await asyncio.sleep(0.1)

        await manager.submit(task_with_delay(), cycle_id="test-1")
        await manager.submit(task_with_delay(), cycle_id="test-2")

        result = await manager.wait_all(timeout=5.0)

        assert result["completed"] == 2
        assert result["cancelled"] == 0
        assert result["pending"] == 0

    @pytest.mark.asyncio
    async def test_wait_all_cancels_slow_tasks(self):
        """Test that wait_all cancels tasks that don't complete in time."""
        manager = ReflectorTaskManager(timeout_seconds=10.0)  # Long timeout

        async def very_slow_task():
            await asyncio.sleep(10.0)

        await manager.submit(very_slow_task(), cycle_id="test-1")

        # Short wait timeout should cancel the slow task
        result = await manager.wait_all(timeout=0.1)

        assert result["cancelled"] == 1
        assert result["completed"] == 0

    @pytest.mark.asyncio
    async def test_stats_returns_all_fields(self):
        """Test that stats includes all expected fields."""
        manager = ReflectorTaskManager(timeout_seconds=30.0)

        stats = manager.stats

        assert "pending" in stats
        assert "succeeded" in stats
        assert "failed" in stats
        assert "timeouts" in stats
        assert "total_processed" in stats
        assert "timeout_seconds" in stats
        assert stats["timeout_seconds"] == 30.0

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self):
        """Test handling of multiple concurrent tasks."""
        manager = ReflectorTaskManager(timeout_seconds=5.0)

        async def quick_task():
            await asyncio.sleep(0.01)

        # Submit 10 tasks
        for i in range(10):
            await manager.submit(quick_task(), cycle_id=f"test-{i}")

        # Wait for all to complete
        await asyncio.sleep(0.2)

        stats = manager.stats
        assert stats["succeeded"] == 10
        assert stats["failed"] == 0
        assert stats["pending"] == 0

    @pytest.mark.asyncio
    async def test_env_var_timeout_configuration(self):
        """Test that timeout can be configured via environment variable."""
        with patch.dict("os.environ", {"E2I_REFLECTOR_TIMEOUT": "45.0"}):
            manager = ReflectorTaskManager()
            assert manager._timeout == 45.0

    @pytest.mark.asyncio
    async def test_task_cleanup_on_completion(self):
        """Test that tasks are removed from pending set on completion."""
        manager = ReflectorTaskManager(timeout_seconds=5.0)

        async def quick_task():
            await asyncio.sleep(0.01)

        await manager.submit(quick_task(), cycle_id="test-1")

        # Initially might be pending
        initial_count = manager.pending_count

        # Wait for completion
        await asyncio.sleep(0.1)

        # Should be cleaned up
        assert manager.pending_count == 0
        assert manager.pending_count <= initial_count
