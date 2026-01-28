"""Opik Trace Verifier for Agent Observability Testing.

Verifies that Opik traces are created correctly for agent executions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class TraceVerificationResult:
    """Result of verifying an Opik trace."""

    trace_exists: bool
    trace_id: str | None = None
    trace_url: str | None = None
    metadata_valid: bool = False
    expected_metadata: dict[str, Any] = field(default_factory=dict)
    actual_metadata: dict[str, Any] = field(default_factory=dict)
    metadata_errors: list[str] = field(default_factory=list)
    span_count: int = 0
    span_names: list[str] = field(default_factory=list)
    duration_ms: float | None = None
    error_captured: bool = False
    error: str | None = None

    @property
    def summary(self) -> str:
        """Get a summary string of the verification result."""
        if not self.trace_exists:
            return f"TRACE NOT FOUND: {self.error or 'Unknown error'}"
        status = "VALID" if self.metadata_valid else "INVALID"
        return f"{status}: {self.span_count} spans, {self.duration_ms or 0:.1f}ms"


class OpikTraceVerifier:
    """Verifies Opik traces for agent executions.

    Usage:
        verifier = OpikTraceVerifier()
        result = await verifier.verify_trace_exists("trace_123")
        result = await verifier.verify_agent_trace(
            agent_name="causal_impact",
            trace_id="trace_123",
            tier=2,
        )
    """

    def __init__(
        self,
        opik_base_url: str = "http://localhost:5173",
        timeout: float = 10.0,
    ):
        """Initialize the verifier.

        Args:
            opik_base_url: Base URL for Opik API
            timeout: HTTP request timeout in seconds
        """
        self.base_url = opik_base_url.rstrip("/")
        self.timeout = timeout

    async def verify_trace_exists(self, trace_id: str) -> bool:
        """Check if a trace exists in Opik.

        Args:
            trace_id: The trace ID to check

        Returns:
            True if trace exists, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Try the Opik API endpoint
                resp = await client.get(
                    f"{self.base_url}/api/v1/private/traces/{trace_id}"
                )
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
        except Exception:
            return False

    async def get_trace_details(self, trace_id: str) -> dict[str, Any] | None:
        """Get full trace details from Opik.

        Args:
            trace_id: The trace ID to fetch

        Returns:
            Trace details dict or None if not found
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/api/v1/private/traces/{trace_id}"
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
        return None

    async def get_trace_spans(self, trace_id: str) -> list[dict[str, Any]]:
        """Get all spans for a trace.

        Args:
            trace_id: The trace ID

        Returns:
            List of span details
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/api/v1/private/spans",
                    params={"trace_id": trace_id},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("content", data.get("spans", []))
        except Exception:
            pass
        return []

    async def verify_agent_trace(
        self,
        agent_name: str,
        trace_id: str,
        tier: int,
        expected_metadata: dict[str, Any] | None = None,
    ) -> TraceVerificationResult:
        """Verify a trace for a specific agent execution.

        Args:
            agent_name: Name of the agent (e.g., 'causal_impact')
            trace_id: The trace ID to verify
            tier: Agent tier number (1-5)
            expected_metadata: Optional additional metadata to verify

        Returns:
            TraceVerificationResult with verification details
        """
        result = TraceVerificationResult(
            trace_exists=False,
            trace_id=trace_id,
            trace_url=f"{self.base_url}/traces/{trace_id}",
            expected_metadata={
                "agent_name": agent_name,
                "tier": tier,
                "framework": "langgraph",
                **(expected_metadata or {}),
            },
        )

        # Check if trace exists
        trace_details = await self.get_trace_details(trace_id)
        if trace_details is None:
            result.error = "Trace not found"
            return result

        result.trace_exists = True
        result.actual_metadata = trace_details.get("metadata", {})

        # Get spans
        spans = await self.get_trace_spans(trace_id)
        result.span_count = len(spans)
        result.span_names = [s.get("name", "unknown") for s in spans]

        # Calculate duration
        if trace_details.get("start_time") and trace_details.get("end_time"):
            try:
                # Parse ISO timestamps and calculate duration
                from datetime import datetime

                start = datetime.fromisoformat(
                    trace_details["start_time"].replace("Z", "+00:00")
                )
                end = datetime.fromisoformat(
                    trace_details["end_time"].replace("Z", "+00:00")
                )
                result.duration_ms = (end - start).total_seconds() * 1000
            except Exception:
                pass

        # Check for errors in trace
        if trace_details.get("error") or trace_details.get("exception"):
            result.error_captured = True

        # Verify metadata
        metadata_errors = []
        for key, expected_value in result.expected_metadata.items():
            actual_value = result.actual_metadata.get(key)
            if actual_value is None:
                metadata_errors.append(f"Missing metadata key: {key}")
            elif actual_value != expected_value:
                metadata_errors.append(
                    f"Metadata mismatch for {key}: expected {expected_value}, got {actual_value}"
                )

        result.metadata_errors = metadata_errors
        result.metadata_valid = len(metadata_errors) == 0

        return result

    async def verify_trace_metadata(
        self,
        trace_id: str,
        expected_metadata: dict[str, Any],
    ) -> TraceVerificationResult:
        """Verify that a trace has expected metadata.

        Args:
            trace_id: The trace ID
            expected_metadata: Dict of expected metadata key-value pairs

        Returns:
            TraceVerificationResult
        """
        result = TraceVerificationResult(
            trace_exists=False,
            trace_id=trace_id,
            trace_url=f"{self.base_url}/traces/{trace_id}",
            expected_metadata=expected_metadata,
        )

        trace_details = await self.get_trace_details(trace_id)
        if trace_details is None:
            result.error = "Trace not found"
            return result

        result.trace_exists = True
        result.actual_metadata = trace_details.get("metadata", {})

        # Verify metadata
        metadata_errors = []
        for key, expected_value in expected_metadata.items():
            actual_value = result.actual_metadata.get(key)
            if actual_value is None:
                metadata_errors.append(f"Missing: {key}")
            elif actual_value != expected_value:
                metadata_errors.append(
                    f"{key}: expected {expected_value!r}, got {actual_value!r}"
                )

        result.metadata_errors = metadata_errors
        result.metadata_valid = len(metadata_errors) == 0

        return result

    async def check_opik_health(self) -> dict[str, Any]:
        """Check if Opik service is healthy and accessible.

        Returns:
            Dict with health status and details
        """
        result = {
            "healthy": False,
            "base_url": self.base_url,
            "error": None,
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try the health endpoint
                resp = await client.get(f"{self.base_url}/api/v1/private/is-alive/ping")
                if resp.status_code == 200:
                    result["healthy"] = True
                    return result

                # Try alternative health check
                resp = await client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    result["healthy"] = True
                    return result

                result["error"] = f"HTTP {resp.status_code}"
        except httpx.ConnectError:
            result["error"] = "Connection refused - is Opik running?"
        except httpx.TimeoutException:
            result["error"] = "Connection timeout"
        except Exception as e:
            result["error"] = str(e)

        return result

    async def list_recent_traces(
        self,
        project_name: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """List recent traces, optionally filtered by project.

        Args:
            project_name: Optional project name filter
            limit: Maximum number of traces to return

        Returns:
            List of trace summaries
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {"page_size": limit}
                if project_name:
                    params["project_name"] = project_name

                resp = await client.get(
                    f"{self.base_url}/api/v1/private/traces",
                    params=params,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("content", data.get("traces", []))
        except Exception:
            pass
        return []


# Convenience function for quick checks
async def verify_opik_available(base_url: str = "http://localhost:5173") -> bool:
    """Quick check if Opik is available.

    Args:
        base_url: Opik base URL

    Returns:
        True if Opik is responding, False otherwise
    """
    verifier = OpikTraceVerifier(opik_base_url=base_url)
    health = await verifier.check_opik_health()
    return health.get("healthy", False)
