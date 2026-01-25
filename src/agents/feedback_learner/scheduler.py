"""
E2I Feedback Learner Agent - Async Scheduler
Version: 4.3
Purpose: Schedule and manage feedback learning cycles

The scheduler enables:
- Periodic feedback learning cycles (cron-based)
- Minimum feedback threshold before triggering
- Graceful shutdown with in-progress cycle completion
- Integration with MLflow for cycle tracking
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .agent import FeedbackLearnerAgent, FeedbackLearnerOutput

logger = logging.getLogger(__name__)


class SchedulerState(Enum):
    """State of the scheduler."""

    STOPPED = "stopped"
    RUNNING = "running"
    STOPPING = "stopping"
    PAUSED = "paused"


@dataclass
class SchedulerConfig:
    """Configuration for the feedback learning scheduler."""

    # Schedule configuration
    interval_hours: float = 6.0  # Run every N hours
    initial_delay_seconds: float = 60.0  # Wait before first run

    # Threshold configuration
    min_feedback_threshold: int = 10  # Minimum feedback items to trigger
    max_batch_size: int = 1000  # Maximum items per cycle

    # Timing configuration
    cycle_timeout_seconds: float = 300.0  # 5 minute timeout per cycle
    shutdown_timeout_seconds: float = 60.0  # Wait for in-progress cycle

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 30.0

    # Focus configuration
    focus_agents: Optional[List[str]] = None  # Agents to focus on


@dataclass
class SchedulerMetrics:
    """Metrics tracked by the scheduler."""

    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    skipped_cycles: int = 0  # Skipped due to threshold
    total_feedback_processed: int = 0
    total_patterns_detected: int = 0
    total_recommendations: int = 0
    last_cycle_time: Optional[datetime] = None
    last_cycle_duration_seconds: float = 0.0
    last_error: Optional[str] = None


@dataclass
class CycleResult:
    """Result of a single learning cycle."""

    cycle_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    skipped: bool = False
    skip_reason: Optional[str] = None
    feedback_count: int = 0
    pattern_count: int = 0
    recommendation_count: int = 0
    error: Optional[str] = None
    output: Optional["FeedbackLearnerOutput"] = None


class FeedbackLearnerScheduler:
    """
    Async scheduler for feedback learning cycles.

    Manages periodic execution of feedback learning with:
    - Configurable intervals
    - Minimum feedback thresholds
    - Graceful shutdown
    - Metrics tracking

    Usage:
        agent = FeedbackLearnerAgent(...)
        scheduler = FeedbackLearnerScheduler(agent)

        # Start in background
        await scheduler.start()

        # ... application runs ...

        # Graceful shutdown
        await scheduler.stop()
    """

    def __init__(
        self,
        agent: "FeedbackLearnerAgent",
        config: Optional[SchedulerConfig] = None,
        feedback_count_fn: Optional[Callable[[], int]] = None,
        on_cycle_complete: Optional[Callable[[CycleResult], None]] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            agent: The FeedbackLearnerAgent to run cycles on
            config: Optional scheduler configuration
            feedback_count_fn: Optional function to check pending feedback count
            on_cycle_complete: Optional callback when cycle completes
        """
        self._agent = agent
        self._config = config or SchedulerConfig()
        self._feedback_count_fn = feedback_count_fn
        self._on_cycle_complete = on_cycle_complete

        self._state = SchedulerState.STOPPED
        self._metrics = SchedulerMetrics()
        self._current_cycle: Optional[asyncio.Task] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._cycle_history: List[CycleResult] = []
        self._cycle_counter = 0

    @property
    def state(self) -> SchedulerState:
        """Get current scheduler state."""
        return self._state

    @property
    def metrics(self) -> SchedulerMetrics:
        """Get scheduler metrics."""
        return self._metrics

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._state == SchedulerState.RUNNING

    @property
    def cycle_history(self) -> List[CycleResult]:
        """Get history of cycle results."""
        return self._cycle_history.copy()

    async def start(self) -> None:
        """
        Start the scheduler.

        Begins the scheduling loop in the background.
        """
        if self._state != SchedulerState.STOPPED:
            logger.warning(f"Scheduler already in state: {self._state}")
            return

        logger.info(
            f"Starting feedback learner scheduler: "
            f"interval={self._config.interval_hours}h, "
            f"threshold={self._config.min_feedback_threshold}"
        )

        self._state = SchedulerState.RUNNING
        self._stop_event.clear()
        self._scheduler_task = asyncio.create_task(self._run_loop())

    async def stop(self, wait_for_current: bool = True) -> None:
        """
        Stop the scheduler gracefully.

        Args:
            wait_for_current: Whether to wait for in-progress cycle to complete
        """
        if self._state == SchedulerState.STOPPED:
            return

        logger.info("Stopping feedback learner scheduler...")
        self._state = SchedulerState.STOPPING
        self._stop_event.set()

        if wait_for_current and self._current_cycle is not None:
            logger.info("Waiting for in-progress cycle to complete...")
            try:
                await asyncio.wait_for(
                    self._current_cycle,
                    timeout=self._config.shutdown_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for cycle, cancelling...")
                self._current_cycle.cancel()
                try:
                    await self._current_cycle
                except asyncio.CancelledError:
                    pass

        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        self._state = SchedulerState.STOPPED
        logger.info("Scheduler stopped")

    async def pause(self) -> None:
        """Pause the scheduler (cycles won't run until resumed)."""
        if self._state == SchedulerState.RUNNING:
            self._state = SchedulerState.PAUSED
            logger.info("Scheduler paused")

    async def resume(self) -> None:
        """Resume a paused scheduler."""
        if self._state == SchedulerState.PAUSED:
            self._state = SchedulerState.RUNNING
            logger.info("Scheduler resumed")

    async def run_cycle_now(self, force: bool = False) -> CycleResult:
        """
        Run a learning cycle immediately.

        Args:
            force: If True, skip threshold check

        Returns:
            CycleResult with cycle outcome
        """
        return await self._execute_cycle(force=force)

    async def check_pending_feedback(self) -> int:
        """
        Check the number of pending feedback items.

        Returns:
            Number of pending feedback items
        """
        if self._feedback_count_fn is not None:
            try:
                count = self._feedback_count_fn()
                if asyncio.iscoroutine(count):
                    count = await count
                return count
            except Exception as e:
                logger.warning(f"Error checking feedback count: {e}")
                return 0

        # Default: check via agent's feedback store
        if hasattr(self._agent, "_feedback_store") and self._agent._feedback_store:
            try:
                # Try to get count from store
                store = self._agent._feedback_store
                if hasattr(store, "count_pending"):
                    count = store.count_pending()
                    if asyncio.iscoroutine(count):
                        count = await count
                    return count
                elif hasattr(store, "get_feedback"):
                    # Fallback: get all and count
                    items = await store.get_feedback()
                    return len(items) if items else 0
            except Exception as e:
                logger.warning(f"Error checking feedback store: {e}")

        # Default to threshold to allow cycles
        return self._config.min_feedback_threshold

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        # Initial delay
        if self._config.initial_delay_seconds > 0:
            logger.debug(
                f"Waiting {self._config.initial_delay_seconds}s before first cycle"
            )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._config.initial_delay_seconds,
                )
                return  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Continue to first cycle

        interval_seconds = self._config.interval_hours * 3600

        while not self._stop_event.is_set():
            if self._state == SchedulerState.PAUSED:
                # Wait a short time then check again
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                    break
                except asyncio.TimeoutError:
                    continue

            # Execute cycle
            try:
                self._current_cycle = asyncio.create_task(self._execute_cycle())
                await self._current_cycle
            except asyncio.CancelledError:
                logger.debug("Cycle cancelled")
            except Exception as e:
                logger.error(f"Unexpected error in cycle: {e}")
            finally:
                self._current_cycle = None

            # Wait for next interval
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval_seconds
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Continue to next cycle

    async def _execute_cycle(self, force: bool = False) -> CycleResult:
        """
        Execute a single learning cycle.

        Args:
            force: Skip threshold check if True

        Returns:
            CycleResult with outcome
        """
        self._cycle_counter += 1
        cycle_id = f"cycle_{self._cycle_counter}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now(timezone.utc)

        logger.info(f"Starting learning cycle: {cycle_id}")

        result = CycleResult(cycle_id=cycle_id, started_at=started_at)

        try:
            # Check threshold unless forced
            if not force:
                pending_count = await self.check_pending_feedback()
                if pending_count < self._config.min_feedback_threshold:
                    logger.info(
                        f"Skipping cycle: {pending_count} feedback < "
                        f"threshold {self._config.min_feedback_threshold}"
                    )
                    result.skipped = True
                    result.skip_reason = (
                        f"Insufficient feedback: {pending_count} < "
                        f"{self._config.min_feedback_threshold}"
                    )
                    result.completed_at = datetime.now(timezone.utc)
                    self._metrics.skipped_cycles += 1
                    self._record_cycle(result)
                    return result

            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=self._config.interval_hours)

            # Execute learning cycle with timeout
            output = await asyncio.wait_for(
                self._agent.learn(
                    time_range_start=start_time.isoformat(),
                    time_range_end=end_time.isoformat(),
                    batch_id=cycle_id,
                    focus_agents=self._config.focus_agents,
                ),
                timeout=self._config.cycle_timeout_seconds,
            )

            # Record results
            result.completed_at = datetime.now(timezone.utc)
            result.success = output.status == "completed"
            result.feedback_count = output.feedback_count
            result.pattern_count = output.pattern_count
            result.recommendation_count = output.recommendation_count
            result.output = output

            if result.success:
                self._metrics.successful_cycles += 1
            else:
                self._metrics.failed_cycles += 1
                result.error = "; ".join(str(e) for e in output.errors)

            logger.info(
                f"Cycle {cycle_id} completed: success={result.success}, "
                f"feedback={result.feedback_count}, patterns={result.pattern_count}"
            )

        except asyncio.TimeoutError:
            result.completed_at = datetime.now(timezone.utc)
            result.success = False
            result.error = f"Cycle timed out after {self._config.cycle_timeout_seconds}s"
            self._metrics.failed_cycles += 1
            logger.error(f"Cycle {cycle_id} timed out")

        except Exception as e:
            result.completed_at = datetime.now(timezone.utc)
            result.success = False
            result.error = str(e)
            self._metrics.failed_cycles += 1
            self._metrics.last_error = str(e)
            logger.error(f"Cycle {cycle_id} failed: {e}")

        finally:
            self._metrics.total_cycles += 1
            self._record_cycle(result)

        return result

    def _record_cycle(self, result: CycleResult) -> None:
        """Record cycle result in history and metrics."""
        self._cycle_history.append(result)

        # Update metrics
        if result.success:
            self._metrics.total_feedback_processed += result.feedback_count
            self._metrics.total_patterns_detected += result.pattern_count
            self._metrics.total_recommendations += result.recommendation_count

        if result.completed_at:
            self._metrics.last_cycle_time = result.completed_at
            self._metrics.last_cycle_duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()

        # Keep history bounded
        max_history = 100
        if len(self._cycle_history) > max_history:
            self._cycle_history = self._cycle_history[-max_history:]

        # Callback if registered
        if self._on_cycle_complete:
            try:
                self._on_cycle_complete(result)
            except Exception as e:
                logger.warning(f"Error in cycle complete callback: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status summary.

        Returns:
            Dictionary with status information
        """
        return {
            "state": self._state.value,
            "config": {
                "interval_hours": self._config.interval_hours,
                "min_feedback_threshold": self._config.min_feedback_threshold,
                "max_batch_size": self._config.max_batch_size,
            },
            "metrics": {
                "total_cycles": self._metrics.total_cycles,
                "successful_cycles": self._metrics.successful_cycles,
                "failed_cycles": self._metrics.failed_cycles,
                "skipped_cycles": self._metrics.skipped_cycles,
                "total_feedback_processed": self._metrics.total_feedback_processed,
                "total_patterns_detected": self._metrics.total_patterns_detected,
                "last_cycle_time": (
                    self._metrics.last_cycle_time.isoformat()
                    if self._metrics.last_cycle_time
                    else None
                ),
                "last_error": self._metrics.last_error,
            },
            "recent_cycles": [
                {
                    "cycle_id": c.cycle_id,
                    "success": c.success,
                    "skipped": c.skipped,
                    "feedback_count": c.feedback_count,
                }
                for c in self._cycle_history[-5:]
            ],
        }


# Factory function
def create_scheduler(
    agent: "FeedbackLearnerAgent",
    interval_hours: float = 6.0,
    min_feedback_threshold: int = 10,
    **kwargs,
) -> FeedbackLearnerScheduler:
    """
    Create a feedback learner scheduler with common configuration.

    Args:
        agent: The FeedbackLearnerAgent to schedule
        interval_hours: Hours between learning cycles
        min_feedback_threshold: Minimum feedback to trigger cycle
        **kwargs: Additional SchedulerConfig parameters

    Returns:
        Configured FeedbackLearnerScheduler
    """
    config = SchedulerConfig(
        interval_hours=interval_hours,
        min_feedback_threshold=min_feedback_threshold,
        **kwargs,
    )
    return FeedbackLearnerScheduler(agent, config)
