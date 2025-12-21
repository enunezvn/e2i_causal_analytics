"""Span Emitter Node - Emit spans to Opik and database.

This module provides real integration with:
- OpikConnector for SDK-based observability
- ObservabilitySpanRepository for database persistence

Version: 2.0.0 (Phase 2 Integration)
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

# Lazy imports to avoid circular dependencies
# These are imported inside functions that need them:
# - OpikConnector from src.mlops.opik_connector
# - ObservabilitySpanRepository from src.repositories.observability_span
# - Model classes from src.agents.ml_foundation.observability_connector.models

if TYPE_CHECKING:
    from src.mlops.opik_connector import OpikConnector
    from src.repositories.observability_span import ObservabilitySpanRepository

logger = logging.getLogger(__name__)


# Module-level singletons (initialized lazily)
_opik_connector: Optional["OpikConnector"] = None
_span_repository: Optional["ObservabilitySpanRepository"] = None


def get_opik_connector() -> "OpikConnector":
    """Get or create the OpikConnector singleton."""
    global _opik_connector
    if _opik_connector is None:
        from src.mlops.opik_connector import OpikConnector

        _opik_connector = OpikConnector()
    return _opik_connector


def get_span_repository() -> Optional["ObservabilitySpanRepository"]:
    """Get or create the ObservabilitySpanRepository.

    Returns None if Supabase client is not available.
    """
    global _span_repository
    if _span_repository is None:
        try:
            # Import here to avoid circular dependencies
            from src.repositories import get_supabase_client
            from src.repositories.observability_span import ObservabilitySpanRepository

            client = get_supabase_client()
            if client:
                _span_repository = ObservabilitySpanRepository(client=client)
        except Exception as e:
            logger.warning(f"Failed to initialize span repository: {e}")
    return _span_repository


def _parse_agent_name(name: str):
    """Parse agent name string to enum."""
    from src.agents.ml_foundation.observability_connector.models import AgentNameEnum

    try:
        return AgentNameEnum(name) if name else None
    except ValueError:
        return None


def _parse_agent_tier(tier: Any):
    """Parse agent tier to enum."""
    from src.agents.ml_foundation.observability_connector.models import AgentTierEnum

    tier_mapping = {
        0: "ml_foundation",
        1: "coordination",
        2: "causal_analytics",
        3: "monitoring",
        4: "ml_predictions",
        5: "self_improvement",
    }

    if isinstance(tier, int) and tier in tier_mapping:
        try:
            return AgentTierEnum(tier_mapping[tier])
        except ValueError:
            return None
    elif isinstance(tier, str):
        try:
            return AgentTierEnum(tier)
        except ValueError:
            return None
    return None


def _parse_status(status: str):
    """Parse status string to enum."""
    from src.agents.ml_foundation.observability_connector.models import SpanStatusEnum

    status_map = {
        "ok": SpanStatusEnum.SUCCESS,
        "success": SpanStatusEnum.SUCCESS,
        "completed": SpanStatusEnum.SUCCESS,
        "error": SpanStatusEnum.ERROR,
        "timeout": SpanStatusEnum.TIMEOUT,
    }
    return status_map.get(status.lower(), SpanStatusEnum.SUCCESS)


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse datetime from various formats."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            # Handle ISO format
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


async def emit_spans(state: Dict[str, Any]) -> Dict[str, Any]:
    """Emit observability spans to Opik and persist to database.

    This function performs dual emission:
    1. Opik SDK - for real-time observability dashboards
    2. Database - for persistent storage and analytics

    Args:
        state: Current agent state with events_to_log

    Returns:
        State updates with emission results
    """
    try:
        events_to_log = state.get("events_to_log", [])

        if not events_to_log:
            return {
                "span_ids_logged": [],
                "trace_ids_logged": [],
                "events_logged": 0,
                "emission_successful": True,
                "emission_errors": [],
                "opik_project": "e2i-causal-analytics",
                "opik_workspace": "default",
                "db_writes_successful": True,
                "db_write_count": 0,
            }

        # Get connectors
        opik = get_opik_connector()
        repository = get_span_repository()

        span_ids = []
        trace_ids = []
        emission_errors = []
        db_write_count = 0
        opik_emit_count = 0

        # Process each event
        for event in events_to_log:
            try:
                # Extract event fields
                span_id = event.get("span_id", str(uuid.uuid4()))
                trace_id = event.get("trace_id", str(uuid.uuid4()))

                span_ids.append(span_id)
                if trace_id not in trace_ids:
                    trace_ids.append(trace_id)

                # Emit to Opik (non-blocking via log_metric for now)
                # Full trace integration happens via trace_agent context manager
                try:
                    if opik.is_enabled:
                        opik.log_metric(
                            name="span_emitted",
                            value=1.0,
                            trace_id=trace_id,
                            metadata={
                                "span_id": span_id,
                                "agent_name": event.get("agent_name"),
                                "operation": event.get("operation"),
                                "status": event.get("status"),
                                "duration_ms": event.get("duration_ms"),
                            },
                        )
                        opik_emit_count += 1
                except Exception as e:
                    logger.debug(f"Opik emission failed (graceful): {e}")
                    # Continue with database write - don't fail the whole operation

                # Persist to database
                if repository:
                    try:
                        # Lazy import to avoid circular dependencies
                        from src.agents.ml_foundation.observability_connector.models import (
                            AgentNameEnum,
                            AgentTierEnum,
                            ObservabilitySpan,
                        )

                        # Parse agent name and tier
                        agent_name_enum = _parse_agent_name(event.get("agent_name", ""))
                        agent_tier_enum = _parse_agent_tier(event.get("agent_tier", 0))

                        # Default to orchestrator if unknown agent
                        if agent_name_enum is None:
                            agent_name_enum = AgentNameEnum.ORCHESTRATOR
                        if agent_tier_enum is None:
                            agent_tier_enum = AgentTierEnum.COORDINATION

                        # Parse timestamps
                        started_at = _parse_datetime(event.get("started_at"))
                        ended_at = _parse_datetime(event.get("completed_at"))

                        if started_at is None:
                            started_at = datetime.now(timezone.utc)

                        # Create ObservabilitySpan model
                        span = ObservabilitySpan(
                            trace_id=trace_id,
                            span_id=span_id,
                            parent_span_id=event.get("parent_span_id"),
                            agent_name=agent_name_enum,
                            agent_tier=agent_tier_enum,
                            operation_type=event.get("operation", "unknown"),
                            started_at=started_at,
                            ended_at=ended_at,
                            duration_ms=event.get("duration_ms"),
                            model_name=event.get("model_used"),
                            input_tokens=event.get("input_tokens"),
                            output_tokens=event.get("output_tokens"),
                            total_tokens=event.get("tokens_used"),
                            status=_parse_status(event.get("status", "success")),
                            error_type=event.get("error_type"),
                            error_message=event.get("error"),
                            fallback_used=event.get("metadata", {}).get(
                                "fallback_used", False
                            ),
                            attributes=event.get("metadata", {}),
                        )

                        # Insert to database
                        result = await repository.insert_span(span)
                        if result:
                            db_write_count += 1
                        else:
                            logger.debug(f"DB insert returned None for span {span_id}")

                    except Exception as e:
                        logger.warning(f"Database write failed for span {span_id}: {e}")
                        emission_errors.append(
                            f"DB write failed for span {span_id}: {str(e)}"
                        )

            except Exception as e:
                emission_errors.append(f"Failed to emit span {span_id}: {str(e)}")

        # Determine overall success (allow partial success)
        emission_successful = len(span_ids) > 0 and (
            opik_emit_count > 0 or db_write_count > 0
        )
        db_writes_successful = db_write_count > 0 or repository is None

        return {
            "span_ids_logged": span_ids,
            "trace_ids_logged": trace_ids,
            "events_logged": len(span_ids),
            "emission_successful": emission_successful,
            "emission_errors": emission_errors,
            "opik_project": opik.config.project_name,
            "opik_workspace": opik.config.workspace,
            "opik_url": "https://www.comet.com/opik",
            "opik_emit_count": opik_emit_count,
            "db_writes_successful": db_writes_successful,
            "db_write_count": db_write_count,
        }

    except Exception as e:
        logger.error(f"Span emission failed: {e}")
        return {
            "error": f"Span emission failed: {str(e)}",
            "error_type": "span_emission_error",
            "error_details": {"exception": str(e)},
            "emission_successful": False,
            "span_ids_logged": [],
            "trace_ids_logged": [],
            "events_logged": 0,
        }
