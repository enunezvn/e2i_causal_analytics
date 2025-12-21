"""Context Manager Node - Manage observability context (trace_id, span_id)."""

import uuid
from typing import Any, Dict


async def create_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create new observability context for a request.

    Args:
        state: Current agent state with request metadata

    Returns:
        State updates with new context
    """
    try:
        # Generate new trace and span IDs
        trace_id = _generate_trace_id()
        span_id = _generate_span_id()

        # Extract request metadata
        request_id = state.get("request_id", str(uuid.uuid4()))
        experiment_id = state.get("experiment_id")
        user_id = state.get("user_id")

        # Determine sampling
        sample_rate = state.get("sample_rate", 1.0)  # Default: sample everything
        sampled = _should_sample(sample_rate)

        return {
            "current_trace_id": trace_id,
            "current_span_id": span_id,
            "current_parent_span_id": None,  # Root span
            "request_id": request_id,
            "experiment_id": experiment_id,
            "user_id": user_id,
            "sampled": sampled,
            "sample_rate": sample_rate,
        }

    except Exception as e:
        return {
            "error": f"Context creation failed: {str(e)}",
            "error_type": "context_creation_error",
            "error_details": {"exception": str(e)},
        }


async def extract_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract observability context from headers.

    Args:
        state: Current agent state with headers

    Returns:
        State updates with extracted context
    """
    try:
        headers = state.get("headers", {})

        # Extract trace context from headers (W3C Trace Context format)
        # traceparent: 00-{trace_id}-{span_id}-{flags}
        traceparent = headers.get("traceparent", "")

        if traceparent:
            parts = traceparent.split("-")
            if len(parts) == 4:
                trace_id = parts[1]
                parent_span_id = parts[2]
                # Generate new span_id for this operation
                span_id = _generate_span_id()
            else:
                # Invalid traceparent, create new context
                trace_id = _generate_trace_id()
                span_id = _generate_span_id()
                parent_span_id = None
        else:
            # No traceparent, create new context
            trace_id = _generate_trace_id()
            span_id = _generate_span_id()
            parent_span_id = None

        # Extract baggage (metadata propagated with trace)
        # tracestate: key1=value1,key2=value2
        tracestate = headers.get("tracestate", "")
        baggage = _parse_tracestate(tracestate)

        return {
            "current_trace_id": trace_id,
            "current_span_id": span_id,
            "current_parent_span_id": parent_span_id,
            "request_id": baggage.get("request_id", str(uuid.uuid4())),
            "experiment_id": baggage.get("experiment_id"),
            "user_id": baggage.get("user_id"),
            "sampled": baggage.get("sampled", "1") == "1",
            "sample_rate": float(baggage.get("sample_rate", "1.0")),
        }

    except Exception as e:
        return {
            "error": f"Context extraction failed: {str(e)}",
            "error_type": "context_extraction_error",
            "error_details": {"exception": str(e)},
        }


async def inject_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Inject observability context into headers for outgoing requests.

    Args:
        state: Current agent state with context

    Returns:
        State updates with headers
    """
    try:
        trace_id = state.get("current_trace_id")
        span_id = state.get("current_span_id")
        sampled = state.get("sampled", True)

        if not trace_id or not span_id:
            return {
                "error": "Missing trace_id or span_id for context injection",
                "error_type": "missing_context",
            }

        # Create traceparent header (W3C Trace Context format)
        # version-trace_id-span_id-flags
        # flags: 01 = sampled, 00 = not sampled
        flags = "01" if sampled else "00"
        traceparent = f"00-{trace_id}-{span_id}-{flags}"

        # Create tracestate header (baggage)
        baggage = {
            "request_id": state.get("request_id", ""),
            "experiment_id": state.get("experiment_id", ""),
            "user_id": state.get("user_id", ""),
            "sampled": "1" if sampled else "0",
            "sample_rate": str(state.get("sample_rate", 1.0)),
        }
        tracestate = _format_tracestate(baggage)

        # Build headers dict
        headers = {
            "traceparent": traceparent,
            "tracestate": tracestate,
        }

        return {"headers": headers}

    except Exception as e:
        return {
            "error": f"Context injection failed: {str(e)}",
            "error_type": "context_injection_error",
            "error_details": {"exception": str(e)},
        }


def _generate_trace_id() -> str:
    """Generate a new trace ID.

    Returns:
        32-character hex trace ID
    """
    return uuid.uuid4().hex


def _generate_span_id() -> str:
    """Generate a new span ID.

    Returns:
        16-character hex span ID
    """
    return uuid.uuid4().hex[:16]


def _should_sample(sample_rate: float) -> bool:
    """Determine if this trace should be sampled.

    Args:
        sample_rate: Sampling rate (0.0-1.0)

    Returns:
        True if should sample, False otherwise
    """
    import random

    return random.random() < sample_rate


def _parse_tracestate(tracestate: str) -> Dict[str, str]:
    """Parse tracestate header into baggage dict.

    Args:
        tracestate: Tracestate header value (key1=value1,key2=value2)

    Returns:
        Dict of baggage items
    """
    if not tracestate:
        return {}

    baggage = {}
    for item in tracestate.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            baggage[key.strip()] = value.strip()

    return baggage


def _format_tracestate(baggage: Dict[str, str]) -> str:
    """Format baggage dict into tracestate header.

    Args:
        baggage: Dict of baggage items

    Returns:
        Tracestate header value (key1=value1,key2=value2)
    """
    items = []
    for key, value in baggage.items():
        if value:  # Only include non-empty values
            items.append(f"{key}={value}")

    return ",".join(items)
