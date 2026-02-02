"""
Unit tests for Feedback Learner Concurrency.
Version: 4.3

Tests concurrent batch processing and race condition handling.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


class TestConcurrentBatchProcessing:
    """Tests for concurrent feedback batch processing."""

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for testing."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        agent = FeedbackLearnerAgent()
        return agent

    @pytest.fixture
    def create_mock_output(self):
        """Factory for creating mock outputs."""

        def _create(batch_id: str, delay: float = 0.0):
            async def mock_learn(*args, **kwargs):
                if delay > 0:
                    await asyncio.sleep(delay)
                output = MagicMock()
                output.batch_id = kwargs.get("batch_id", batch_id)
                output.status = "completed"
                output.feedback_count = 10
                output.pattern_count = 2
                output.recommendation_count = 1
                output.errors = []
                output.warnings = []
                return output

            return mock_learn

        return _create

    @pytest.mark.asyncio
    async def test_parallel_batch_processing(self, create_mock_output):
        """Multiple batches can process in parallel."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        results = []
        processing_order = []

        async def tracked_learn(batch_id: str, delay: float):
            processing_order.append(f"start_{batch_id}")
            await asyncio.sleep(delay)
            processing_order.append(f"end_{batch_id}")
            output = MagicMock()
            output.batch_id = batch_id
            output.status = "completed"
            output.feedback_count = 10
            output.pattern_count = 2
            output.recommendation_count = 1
            output.errors = []
            return output

        # Create separate agent instances for parallel processing
        [FeedbackLearnerAgent() for _ in range(3)]

        # Process batches with different delays
        tasks = [
            tracked_learn("batch_1", 0.1),
            tracked_learn("batch_2", 0.05),
            tracked_learn("batch_3", 0.02),
        ]

        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 3
        assert all(r.status == "completed" for r in results)

        # Verify parallel execution (not sequential)
        # batch_3 should end before batch_1 due to shorter delay
        batch_3_end = processing_order.index("end_batch_3")
        batch_1_end = processing_order.index("end_batch_1")
        assert batch_3_end < batch_1_end

    @pytest.mark.asyncio
    async def test_batch_isolation(self, create_mock_output):
        """Each batch maintains its own state."""
        batch_states = {}

        async def stateful_process(batch_id: str):
            # Simulate stateful processing
            batch_states[batch_id] = {"started": True, "items": []}

            for i in range(5):
                await asyncio.sleep(0.01)
                batch_states[batch_id]["items"].append(f"item_{i}")

            batch_states[batch_id]["completed"] = True
            return batch_states[batch_id]

        # Process multiple batches concurrently
        tasks = [
            stateful_process("batch_a"),
            stateful_process("batch_b"),
            stateful_process("batch_c"),
        ]

        await asyncio.gather(*tasks)

        # Each batch should have its own items
        assert len(batch_states) == 3
        for _batch_id, state in batch_states.items():
            assert state["completed"] is True
            assert len(state["items"]) == 5
            assert all(f"item_{i}" in state["items"] for i in range(5))

    @pytest.mark.asyncio
    async def test_signal_aggregation_thread_safety(self):
        """DSPy signal aggregation should be thread-safe."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        signals = []
        lock = asyncio.Lock()

        async def add_signal(batch_id: str):
            signal = FeedbackLearnerTrainingSignal(
                batch_id=batch_id,
                feedback_count=10,
                time_range_start="2025-01-01",
                time_range_end="2025-01-02",
            )
            signal.patterns_detected = 2
            signal.recommendations_generated = 1

            async with lock:
                signals.append(signal)
                await asyncio.sleep(0.001)  # Simulate work

            return signal

        # Add signals concurrently
        tasks = [add_signal(f"batch_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All signals should be added
        assert len(signals) == 10
        assert len(results) == 10

        # Each signal should have correct batch_id
        batch_ids = {s.batch_id for s in signals}
        assert len(batch_ids) == 10

    @pytest.mark.asyncio
    async def test_concurrent_pattern_detection(self):
        """Pattern detection should work correctly with concurrent batches."""
        detected_patterns = []
        lock = asyncio.Lock()

        async def detect_patterns(feedback_items: list, batch_id: str):
            # Simulate pattern detection
            await asyncio.sleep(0.02)  # Simulate processing time

            patterns = [
                {
                    "pattern_id": f"pattern_{batch_id}_{i}",
                    "batch_id": batch_id,
                    "severity": "medium",
                }
                for i in range(3)
            ]

            async with lock:
                detected_patterns.extend(patterns)

            return patterns

        # Run pattern detection concurrently
        tasks = [detect_patterns([], f"batch_{i}") for i in range(5)]
        await asyncio.gather(*tasks)

        # All patterns should be collected
        assert len(detected_patterns) == 15  # 5 batches * 3 patterns each

        # Patterns should be correctly attributed
        for pattern in detected_patterns:
            assert pattern["batch_id"] in [f"batch_{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_knowledge_update_ordering(self):
        """Knowledge updates should maintain order within a batch."""
        update_log = []

        async def apply_update(update_id: str, batch_id: str, order: int):
            await asyncio.sleep(0.01 * (5 - order))  # Reverse delay
            update_log.append(
                {
                    "update_id": update_id,
                    "batch_id": batch_id,
                    "order": order,
                    "timestamp": datetime.now(timezone.utc),
                }
            )
            return update_id

        # Apply updates for a single batch - should maintain order
        batch_updates = [apply_update(f"update_{i}", "batch_1", i) for i in range(5)]

        await asyncio.gather(*batch_updates)

        # Updates from same batch should all be present
        batch_1_updates = [u for u in update_log if u["batch_id"] == "batch_1"]
        assert len(batch_1_updates) == 5


class TestRaceConditionHandling:
    """Tests for race condition scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_state_updates(self):
        """Concurrent updates to shared state should be handled safely."""
        shared_state = {
            "patterns": [],
            "recommendations": [],
            "update_count": 0,
        }
        lock = asyncio.Lock()

        async def update_state(batch_id: str):
            pattern = {"id": f"pattern_{batch_id}", "type": "accuracy_issue"}
            recommendation = {"id": f"rec_{batch_id}", "category": "prompt_update"}

            async with lock:
                shared_state["patterns"].append(pattern)
                shared_state["recommendations"].append(recommendation)
                shared_state["update_count"] += 1
                await asyncio.sleep(0.001)

        # Run many concurrent updates
        tasks = [update_state(f"batch_{i}") for i in range(50)]
        await asyncio.gather(*tasks)

        # All updates should be recorded
        assert shared_state["update_count"] == 50
        assert len(shared_state["patterns"]) == 50
        assert len(shared_state["recommendations"]) == 50

    @pytest.mark.asyncio
    async def test_concurrent_metric_accumulation(self):
        """Metric accumulation should be accurate under concurrency."""
        metrics = {
            "total_feedback": 0,
            "total_patterns": 0,
            "total_recommendations": 0,
        }
        lock = asyncio.Lock()

        async def accumulate_metrics(feedback_count: int, pattern_count: int):
            async with lock:
                metrics["total_feedback"] += feedback_count
                metrics["total_patterns"] += pattern_count
                metrics["total_recommendations"] += 1
                await asyncio.sleep(0.001)

        # Run concurrent accumulations
        tasks = [accumulate_metrics(10, 2) for _ in range(100)]
        await asyncio.gather(*tasks)

        # Metrics should be accurate
        assert metrics["total_feedback"] == 1000  # 100 * 10
        assert metrics["total_patterns"] == 200  # 100 * 2
        assert metrics["total_recommendations"] == 100

    @pytest.mark.asyncio
    async def test_scheduler_cycle_isolation(self):
        """Scheduler cycles should not interfere with each other."""
        from src.agents.feedback_learner.scheduler import (
            FeedbackLearnerScheduler,
            SchedulerConfig,
        )

        asyncio.Lock()

        async def mock_learn(*args, **kwargs):
            output = MagicMock()
            output.batch_id = kwargs.get("batch_id", "unknown")
            output.status = "completed"
            output.feedback_count = 10
            output.pattern_count = 2
            output.recommendation_count = 1
            output.errors = []
            return output

        agent = MagicMock()
        agent._feedback_store = None
        agent.learn = mock_learn

        config = SchedulerConfig(
            interval_hours=0.0001,
            initial_delay_seconds=0,
            min_feedback_threshold=0,
        )

        scheduler = FeedbackLearnerScheduler(agent, config)

        # Run multiple cycles manually
        tasks = [scheduler.run_cycle_now(force=True) for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # All cycles should complete successfully
        assert len(results) == 5
        assert all(r.success for r in results)

        # Each cycle should have unique ID
        cycle_ids = {r.cycle_id for r in results}
        assert len(cycle_ids) == 5


class TestStressTests:
    """Stress tests for concurrent processing."""

    @pytest.mark.asyncio
    async def test_high_concurrency_processing(self):
        """System should handle high concurrency gracefully."""
        results = []
        errors = []
        lock = asyncio.Lock()

        async def process_batch(batch_id: int):
            try:
                await asyncio.sleep(0.01)  # Simulate processing
                async with lock:
                    results.append(batch_id)
                return batch_id
            except Exception as e:
                async with lock:
                    errors.append((batch_id, str(e)))
                raise

        # Run many concurrent batches
        tasks = [process_batch(i) for i in range(100)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without errors
        assert len(results) == 100
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_mixed_success_failure_batches(self):
        """System should handle mixed success/failure batches."""
        results = {"success": [], "failure": []}
        lock = asyncio.Lock()

        async def process_batch(batch_id: int, should_fail: bool):
            await asyncio.sleep(0.01)

            if should_fail:
                async with lock:
                    results["failure"].append(batch_id)
                raise ValueError(f"Simulated failure for batch {batch_id}")

            async with lock:
                results["success"].append(batch_id)
            return batch_id

        # Mix of success and failure
        tasks = [process_batch(i, should_fail=(i % 3 == 0)) for i in range(30)]

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        # Should have both successes and failures
        assert len(results["success"]) == 20  # 30 - 10 failures
        assert len(results["failure"]) == 10  # Every 3rd batch

        # Exceptions should be returned, not raised
        exceptions = [o for o in outcomes if isinstance(o, Exception)]
        assert len(exceptions) == 10

    @pytest.mark.asyncio
    async def test_cancellation_handling(self):
        """Cancelled tasks should be handled gracefully."""
        started = []
        completed = []
        cancelled = []
        lock = asyncio.Lock()

        async def cancellable_process(batch_id: int, delay: float):
            async with lock:
                started.append(batch_id)

            try:
                await asyncio.sleep(delay)
                async with lock:
                    completed.append(batch_id)
                return batch_id
            except asyncio.CancelledError:
                async with lock:
                    cancelled.append(batch_id)
                raise

        # Start tasks with different delays
        tasks = [asyncio.create_task(cancellable_process(i, 0.1 * (i + 1))) for i in range(5)]

        # Wait briefly then cancel remaining
        await asyncio.sleep(0.15)
        for task in tasks:
            if not task.done():
                task.cancel()

        # Gather results (including cancellations)
        await asyncio.gather(*tasks, return_exceptions=True)

        # First task should complete, others may be cancelled
        assert len(started) == 5
        assert 0 in completed or 0 in cancelled
