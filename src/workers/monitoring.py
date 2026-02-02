"""
Celery Queue Monitoring and Metrics.

G12/G19 from observability audit remediation plan:
- Queue depth monitoring for all Celery queues
- Worker health and availability tracking
- Redis broker statistics collection
- Autoscaler metrics support

Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from celery import Celery

logger = logging.getLogger(__name__)

# =============================================================================
# Prometheus Metrics Integration
# =============================================================================

try:
    from prometheus_client import (
        CollectorRegistry,
        Gauge,
        Info,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, queue metrics disabled")


@dataclass
class QueueMetrics:
    """Container for queue monitoring metrics."""

    registry: Optional[Any] = None
    queue_length: Optional[Any] = None
    queue_consumer_count: Optional[Any] = None
    broker_connection_count: Optional[Any] = None
    scheduled_tasks: Optional[Any] = None
    reserved_tasks: Optional[Any] = None
    worker_info: Optional[Any] = None

    _initialized: bool = False

    def initialize(self, registry: Optional[Any] = None) -> None:
        """Initialize queue monitoring metrics."""
        if self._initialized or not PROMETHEUS_AVAILABLE:
            return

        self.registry = registry or CollectorRegistry()

        # Queue depth gauge
        self.queue_length = Gauge(
            "celery_queue_length",
            "Number of messages waiting in queue",
            ["queue_name"],
            registry=self.registry,
        )

        # Consumer count per queue
        self.queue_consumer_count = Gauge(
            "celery_queue_consumers",
            "Number of consumers for each queue",
            ["queue_name"],
            registry=self.registry,
        )

        # Broker connections
        self.broker_connection_count = Gauge(
            "celery_broker_connections",
            "Number of active broker connections",
            registry=self.registry,
        )

        # Scheduled (ETA) tasks
        self.scheduled_tasks = Gauge(
            "celery_scheduled_tasks",
            "Number of tasks scheduled for future execution",
            registry=self.registry,
        )

        # Reserved tasks (prefetched by workers)
        self.reserved_tasks = Gauge(
            "celery_reserved_tasks",
            "Number of tasks reserved by workers",
            ["worker"],
            registry=self.registry,
        )

        # Worker info
        self.worker_info = Info(
            "celery_worker",
            "Celery worker information",
            ["hostname"],
            registry=self.registry,
        )

        self._initialized = True
        logger.info("Queue monitoring metrics initialized")


# Global metrics instance
queue_metrics = QueueMetrics()


# =============================================================================
# Queue Monitoring
# =============================================================================


class CeleryQueueMonitor:
    """
    Monitors Celery queue depths and broker health.

    Features:
    - Real-time queue depth monitoring via Redis
    - Worker availability tracking
    - Broker connection statistics
    - Autoscaler integration support

    Usage:
        monitor = CeleryQueueMonitor(celery_app)
        depths = monitor.get_queue_depths()
        monitor.update_metrics()
    """

    # Queue names from celery_app.py
    QUEUE_NAMES = [
        "default",
        "quick",
        "api",
        "analytics",
        "reports",
        "aggregations",
        "shap",
        "causal",
        "ml",
        "twins",
    ]

    # Queue tier mapping
    QUEUE_TIERS = {
        "light": ["default", "quick", "api"],
        "medium": ["analytics", "reports", "aggregations"],
        "heavy": ["shap", "causal", "ml", "twins"],
    }

    def __init__(
        self,
        app: Celery,
        registry: Optional[Any] = None,
    ):
        """
        Initialize the queue monitor.

        Args:
            app: Celery application instance
            registry: Optional Prometheus registry
        """
        self.app = app
        self._redis_client = None

        # Initialize metrics
        queue_metrics.initialize(registry)

    def _get_redis_client(self):
        """Get or create Redis client for direct queue inspection."""
        if self._redis_client is None:
            try:
                import redis

                broker_url = self.app.conf.broker_url
                self._redis_client = redis.from_url(broker_url)
            except ImportError:
                logger.warning("redis package not installed, using Celery inspect")
                return None
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                return None

        return self._redis_client

    def get_queue_depths(self) -> Dict[str, int]:
        """
        Get current depth of all monitored queues.

        Returns:
            Dictionary mapping queue names to message counts
        """
        depths = {}

        # Try direct Redis access first (faster)
        redis_client = self._get_redis_client()
        if redis_client:
            try:
                for queue_name in self.QUEUE_NAMES:
                    # Celery uses list with queue name as key
                    depth = redis_client.llen(queue_name)
                    depths[queue_name] = depth
                return depths
            except Exception as e:
                logger.warning(f"Redis queue depth check failed: {e}")

        # Fallback to Celery inspect (slower but more portable)
        try:
            inspect = self.app.control.inspect()
            inspect.active() or {}
            reserved = inspect.reserved() or {}
            inspect.scheduled() or {}

            # Aggregate by queue (approximation)
            for queue_name in self.QUEUE_NAMES:
                depths[queue_name] = 0

            # Count reserved tasks as queue depth estimate
            for _worker, tasks in reserved.items():
                for task in tasks:
                    queue = task.get("delivery_info", {}).get("routing_key", "default")
                    if queue in depths:
                        depths[queue] += 1

        except Exception as e:
            logger.warning(f"Celery inspect failed: {e}")
            # Return zeros if all methods fail
            for queue_name in self.QUEUE_NAMES:
                depths[queue_name] = 0

        return depths

    def get_queue_depths_by_tier(self) -> Dict[str, int]:
        """
        Get aggregated queue depths by worker tier.

        Returns:
            Dictionary mapping tier names to total message counts
        """
        depths = self.get_queue_depths()
        tier_depths = {}

        for tier, queues in self.QUEUE_TIERS.items():
            tier_depths[tier] = sum(depths.get(q, 0) for q in queues)

        return tier_depths

    def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active workers.

        Returns:
            Dictionary with worker statistics
        """
        stats = {
            "total_workers": 0,
            "workers_by_tier": {"light": 0, "medium": 0, "heavy": 0, "unknown": 0},
            "workers": [],
        }

        try:
            inspect = self.app.control.inspect()
            ping_response = inspect.ping() or {}

            for hostname, response in ping_response.items():
                if response:
                    stats["total_workers"] += 1

                    # Infer tier from hostname
                    tier = "unknown"
                    hostname_lower = hostname.lower()
                    if "light" in hostname_lower:
                        tier = "light"
                    elif "medium" in hostname_lower:
                        tier = "medium"
                    elif "heavy" in hostname_lower:
                        tier = "heavy"

                    stats["workers_by_tier"][tier] += 1
                    stats["workers"].append(
                        {
                            "hostname": hostname,
                            "tier": tier,
                            "status": "online",
                        }
                    )

        except Exception as e:
            logger.warning(f"Failed to get worker stats: {e}")

        return stats

    def get_scheduled_task_count(self) -> int:
        """
        Get count of scheduled (ETA/countdown) tasks.

        Returns:
            Number of scheduled tasks
        """
        try:
            inspect = self.app.control.inspect()
            scheduled = inspect.scheduled() or {}

            count = 0
            for _worker, tasks in scheduled.items():
                count += len(tasks)

            return count

        except Exception as e:
            logger.warning(f"Failed to get scheduled tasks: {e}")
            return 0

    def get_reserved_tasks(self) -> Dict[str, int]:
        """
        Get count of reserved (prefetched) tasks per worker.

        Returns:
            Dictionary mapping worker hostnames to reserved task counts
        """
        reserved_counts = {}

        try:
            inspect = self.app.control.inspect()
            reserved = inspect.reserved() or {}

            for worker, tasks in reserved.items():
                reserved_counts[worker] = len(tasks)

        except Exception as e:
            logger.warning(f"Failed to get reserved tasks: {e}")

        return reserved_counts

    def update_metrics(self) -> None:
        """Update all Prometheus metrics with current values."""
        if not queue_metrics._initialized:
            return

        # Update queue depths
        depths = self.get_queue_depths()
        for queue_name, depth in depths.items():
            queue_metrics.queue_length.labels(queue_name=queue_name).set(depth)

        # Update scheduled tasks
        scheduled_count = self.get_scheduled_task_count()
        queue_metrics.scheduled_tasks.set(scheduled_count)

        # Update reserved tasks per worker
        reserved = self.get_reserved_tasks()
        for worker, count in reserved.items():
            queue_metrics.reserved_tasks.labels(worker=worker).set(count)

        logger.debug(f"Updated queue metrics: depths={depths}, scheduled={scheduled_count}")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary.

        Returns:
            Dictionary with all monitoring data
        """
        depths = self.get_queue_depths()
        tier_depths = self.get_queue_depths_by_tier()
        worker_stats = self.get_worker_stats()
        scheduled = self.get_scheduled_task_count()
        reserved = self.get_reserved_tasks()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "queues": {
                "depths": depths,
                "by_tier": tier_depths,
                "total": sum(depths.values()),
            },
            "workers": worker_stats,
            "scheduled_tasks": scheduled,
            "reserved_tasks": {
                "by_worker": reserved,
                "total": sum(reserved.values()),
            },
            "health": {
                "status": "healthy" if worker_stats["total_workers"] > 0 else "degraded",
                "workers_online": worker_stats["total_workers"],
                "queue_backlog": sum(depths.values()),
            },
        }


# =============================================================================
# Celery Tasks
# =============================================================================


def register_monitoring_tasks(app: Celery) -> None:
    """
    Register monitoring tasks with Celery app.

    Args:
        app: Celery application instance
    """

    @app.task(name="src.tasks.collect_queue_metrics", bind=True)
    def collect_queue_metrics(self) -> Dict[str, Any]:
        """
        Collect and record queue metrics.

        This task is scheduled every 5 minutes by Celery Beat.
        Updates Prometheus metrics and returns summary.
        """
        monitor = CeleryQueueMonitor(app)
        monitor.update_metrics()
        return monitor.get_monitoring_summary()

    @app.task(name="src.tasks.get_queue_status", bind=True)
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status (on-demand).

        Returns summary without updating metrics.
        """
        monitor = CeleryQueueMonitor(app)
        return monitor.get_monitoring_summary()

    logger.info("Registered Celery monitoring tasks")


# =============================================================================
# Autoscaler Support
# =============================================================================


@dataclass
class AutoscalerRecommendation:
    """Recommendation for worker scaling."""

    tier: str
    current_workers: int
    recommended_workers: int
    reason: str
    queue_depth: int
    timestamp: str


class AutoscalerMetricsProvider:
    """
    Provides metrics for Celery autoscaler integration.

    Calculates recommended worker counts based on:
    - Queue depths
    - Task processing rates
    - Worker utilization
    """

    # Scaling thresholds
    SCALE_UP_THRESHOLD = 10  # Messages per worker
    SCALE_DOWN_THRESHOLD = 2  # Messages per worker
    MIN_WORKERS = {"light": 1, "medium": 1, "heavy": 0}
    MAX_WORKERS = {"light": 4, "medium": 4, "heavy": 8}

    def __init__(self, app: Celery):
        """Initialize autoscaler provider."""
        self.monitor = CeleryQueueMonitor(app)

    def get_scaling_recommendations(self) -> List[AutoscalerRecommendation]:
        """
        Calculate scaling recommendations for each tier.

        Returns:
            List of scaling recommendations
        """
        recommendations = []
        tier_depths = self.monitor.get_queue_depths_by_tier()
        worker_stats = self.monitor.get_worker_stats()
        timestamp = datetime.now(timezone.utc).isoformat()

        for tier, depth in tier_depths.items():
            current_workers = worker_stats["workers_by_tier"].get(tier, 0)
            min_workers = self.MIN_WORKERS.get(tier, 0)
            max_workers = self.MAX_WORKERS.get(tier, 4)

            # Calculate recommended workers
            if current_workers == 0:
                # Need at least min workers if there's work
                if depth > 0:
                    recommended = min_workers
                    reason = f"Queue has {depth} tasks but no workers"
                else:
                    recommended = 0
                    reason = "No tasks in queue"
            else:
                tasks_per_worker = depth / current_workers

                if tasks_per_worker > self.SCALE_UP_THRESHOLD:
                    # Scale up
                    recommended = min(current_workers + 1, max_workers)
                    reason = f"High load: {tasks_per_worker:.1f} tasks/worker"
                elif tasks_per_worker < self.SCALE_DOWN_THRESHOLD and current_workers > min_workers:
                    # Scale down
                    recommended = max(current_workers - 1, min_workers)
                    reason = f"Low load: {tasks_per_worker:.1f} tasks/worker"
                else:
                    # Maintain
                    recommended = current_workers
                    reason = f"Load balanced: {tasks_per_worker:.1f} tasks/worker"

            recommendations.append(
                AutoscalerRecommendation(
                    tier=tier,
                    current_workers=current_workers,
                    recommended_workers=recommended,
                    reason=reason,
                    queue_depth=depth,
                    timestamp=timestamp,
                )
            )

        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================


def get_queue_monitor(app: Optional[Celery] = None) -> CeleryQueueMonitor:
    """
    Get a queue monitor instance.

    Args:
        app: Optional Celery app (uses default if not provided)

    Returns:
        CeleryQueueMonitor instance
    """
    if app is None:
        from .celery_app import celery_app

        app = celery_app

    return CeleryQueueMonitor(app)


def get_queue_depths(app: Optional[Celery] = None) -> Dict[str, int]:
    """
    Convenience function to get queue depths.

    Args:
        app: Optional Celery app

    Returns:
        Dictionary of queue depths
    """
    return get_queue_monitor(app).get_queue_depths()


def get_monitoring_summary(app: Optional[Celery] = None) -> Dict[str, Any]:
    """
    Convenience function to get monitoring summary.

    Args:
        app: Optional Celery app

    Returns:
        Monitoring summary dictionary
    """
    return get_queue_monitor(app).get_monitoring_summary()
