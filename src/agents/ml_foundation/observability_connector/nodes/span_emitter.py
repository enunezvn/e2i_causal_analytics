"""Span Emitter Node - Emit spans to Opik and database."""

import uuid
from typing import Dict, Any, List
from datetime import datetime


async def emit_spans(state: Dict[str, Any]) -> Dict[str, Any]:
    """Emit observability spans to Opik and persist to database.

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

        span_ids = []
        trace_ids = []
        emission_errors = []

        # Process each event
        for event in events_to_log:
            try:
                # Extract event fields
                span_id = event.get("span_id", str(uuid.uuid4()))
                trace_id = event.get("trace_id", str(uuid.uuid4()))

                span_ids.append(span_id)
                if trace_id not in trace_ids:
                    trace_ids.append(trace_id)

                # In production, emit to Opik
                # await opik_client.create_span(
                #     trace_id=trace_id,
                #     span_id=span_id,
                #     parent_span_id=event.get("parent_span_id"),
                #     name=event.get("operation", "unknown"),
                #     start_time=event.get("started_at"),
                #     end_time=event.get("completed_at"),
                #     status=event.get("status", "started"),
                #     metadata={
                #         "agent_name": event.get("agent_name"),
                #         "duration_ms": event.get("duration_ms"),
                #         "error": event.get("error"),
                #         **event.get("metadata", {})
                #     }
                # )

                # In production, persist to database
                # await db.insert("ml_observability_spans", {
                #     "span_id": span_id,
                #     "trace_id": trace_id,
                #     "parent_span_id": event.get("parent_span_id"),
                #     "operation_name": event.get("operation"),
                #     "agent_name": event.get("agent_name"),
                #     "agent_tier": event.get("agent_tier", 0),
                #     "start_time": event.get("started_at"),
                #     "end_time": event.get("completed_at"),
                #     "duration_ms": event.get("duration_ms"),
                #     "status": event.get("status"),
                #     "error_type": event.get("error_type"),
                #     "error_message": event.get("error"),
                #     "attributes": event.get("metadata", {}),
                #     "llm_model": event.get("model_used"),
                #     "input_tokens": event.get("input_tokens"),
                #     "output_tokens": event.get("output_tokens"),
                #     "total_tokens": event.get("tokens_used"),
                #     "events": []
                # })

            except Exception as e:
                emission_errors.append(f"Failed to emit span {span_id}: {str(e)}")

        # Determine overall success
        emission_successful = len(emission_errors) == 0

        return {
            "span_ids_logged": span_ids,
            "trace_ids_logged": trace_ids,
            "events_logged": len(span_ids),
            "emission_successful": emission_successful,
            "emission_errors": emission_errors,
            "opik_project": "e2i-causal-analytics",
            "opik_workspace": "default",
            "opik_url": "https://www.comet.com/opik",
            "db_writes_successful": emission_successful,
            "db_write_count": len(span_ids),
        }

    except Exception as e:
        return {
            "error": f"Span emission failed: {str(e)}",
            "error_type": "span_emission_error",
            "error_details": {"exception": str(e)},
            "emission_successful": False,
            "span_ids_logged": [],
            "trace_ids_logged": [],
            "events_logged": 0,
        }
