"""
E2I Health Score Agent - Component Health Node
Version: 4.2
Purpose: Check health of system components with parallel execution
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Protocol, cast

from ..metrics import DEFAULT_THRESHOLDS
from ..state import ComponentStatus, HealthScoreState

logger = logging.getLogger(__name__)


class HealthClient(Protocol):
    """Protocol for health check client"""

    async def check(self, endpoint: str) -> Dict[str, Any]:
        """Check health of an endpoint"""
        ...


class ComponentHealthNode:
    """
    Check health of system components.
    Fast parallel health checks with configurable timeout.
    """

    # Default components to check
    DEFAULT_COMPONENTS = [
        {"name": "database", "endpoint": "/health/db"},
        {"name": "cache", "endpoint": "/health/cache"},
        {"name": "vector_store", "endpoint": "/health/vectors"},
        {"name": "api_gateway", "endpoint": "/health/api"},
        {"name": "message_queue", "endpoint": "/health/queue"},
    ]

    def __init__(
        self,
        health_client: Optional[HealthClient] = None,
        components: Optional[List[Dict[str, str]]] = None,
        timeout_ms: int = DEFAULT_THRESHOLDS.health_check_timeout_ms,
    ):
        """
        Initialize component health node.

        Args:
            health_client: Client for making health check requests
            components: List of components to check (uses defaults if None)
            timeout_ms: Timeout for each health check in milliseconds
        """
        self.health_client = health_client
        self.components = components or self.DEFAULT_COMPONENTS
        self.timeout_ms = timeout_ms

    async def execute(self, state: HealthScoreState) -> HealthScoreState:
        """Execute component health checks."""
        start_time = time.time()

        # Skip if quick check that doesn't need components
        if state.get("check_scope") == "quick":
            logger.debug("Skipping component health for quick check")
            return {
                **state,
                "component_statuses": [],
                "component_health_score": 1.0,
                "status": "checking",
            }

        try:
            # Run parallel health checks
            if self.health_client:
                tasks = [self._check_component(comp) for comp in self.components]
                statuses = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # No client - simulate healthy for testing
                statuses = [self._create_mock_status(comp) for comp in self.components]

            # Process results
            component_statuses = []
            for comp, status in zip(self.components, statuses, strict=False):
                if isinstance(status, Exception):
                    component_statuses.append(
                        ComponentStatus(
                            component_name=comp["name"],
                            status="unknown",
                            latency_ms=None,
                            last_check=datetime.now(timezone.utc).isoformat(),
                            error_message=str(status),
                        )
                    )
                elif isinstance(status, dict):
                    component_statuses.append(status)

            # Calculate component health score
            if component_statuses:
                healthy_count = sum(1 for s in component_statuses if s["status"] == "healthy")
                degraded_count = sum(1 for s in component_statuses if s["status"] == "degraded")
                # Healthy = 1.0, Degraded = 0.5, Others = 0.0
                total_score = healthy_count + (degraded_count * 0.5)
                health_score = total_score / len(component_statuses)
            else:
                health_score = 1.0

            check_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Component health check complete: score={health_score:.2f}, "
                f"duration={check_time}ms"
            )

            return {
                **state,
                "component_statuses": component_statuses,
                "component_health_score": health_score,
                "total_latency_ms": check_time,
                "status": "checking",
            }

        except Exception as e:
            logger.error(f"Component health check failed: {e}")
            return {
                **state,
                "errors": [{"node": "component_health", "error": str(e)}],
                "component_health_score": 0.0,
                "component_statuses": [],
                "status": "checking",
            }

    async def _check_component(self, component: Dict[str, str]) -> ComponentStatus:
        """Check single component health."""
        start = time.time()

        try:
            assert self.health_client is not None
            result = await asyncio.wait_for(
                self.health_client.check(component["endpoint"]),
                timeout=self.timeout_ms / 1000,
            )

            latency = int((time.time() - start) * 1000)

            # Determine status based on response
            if result.get("ok"):
                status = "healthy"
            elif result.get("degraded"):
                status = "degraded"
            else:
                status = "unhealthy"

            return ComponentStatus(
                component_name=component["name"],
                status=cast(Literal["healthy", "degraded", "unhealthy", "unknown"], status),
                latency_ms=latency,
                last_check=datetime.now(timezone.utc).isoformat(),
                error_message=result.get("error"),
            )

        except asyncio.TimeoutError:
            return ComponentStatus(
                component_name=component["name"],
                status="unhealthy",
                latency_ms=self.timeout_ms,
                last_check=datetime.now(timezone.utc).isoformat(),
                error_message="Health check timed out",
            )

        except Exception as e:
            return ComponentStatus(
                component_name=component["name"],
                status="unknown",
                latency_ms=int((time.time() - start) * 1000),
                last_check=datetime.now(timezone.utc).isoformat(),
                error_message=str(e),
            )

    def _create_mock_status(self, component: Dict[str, str]) -> ComponentStatus:
        """Create mock healthy status for testing."""
        return ComponentStatus(
            component_name=component["name"],
            status="healthy",
            latency_ms=10,
            last_check=datetime.now(timezone.utc).isoformat(),
            error_message=None,
        )
