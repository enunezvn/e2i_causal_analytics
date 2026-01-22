"""BentoML Prediction Audit Trail Module.

Provides utility functions for logging prediction audit trails
to Opik from BentoML service templates.

Phase 1 G07 from observability audit remediation plan.

This module enables:
- Input/output logging for model predictions
- Request ID propagation for distributed tracing
- Automatic metadata extraction (latency, model info)
- Graceful degradation when Opik is unavailable

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Track Opik availability
_OPIK_AVAILABLE = None


def _check_opik_available() -> bool:
    """Check if OpikConnector is available."""
    global _OPIK_AVAILABLE
    if _OPIK_AVAILABLE is None:
        try:
            from src.mlops.opik_connector import OpikConnector

            _OPIK_AVAILABLE = True
        except ImportError:
            _OPIK_AVAILABLE = False
            logger.warning(
                "OpikConnector not available - prediction audit trail disabled"
            )
    return _OPIK_AVAILABLE


async def log_prediction_audit(
    model_name: str,
    model_tag: str,
    service_type: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    latency_ms: float,
    request_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Log a prediction to Opik for audit trail.

    This function creates a trace in Opik capturing:
    - Model inputs and outputs
    - Prediction latency
    - Model and service metadata
    - Request ID for correlation

    Args:
        model_name: Name of the model (e.g., "churn_classifier")
        model_tag: BentoML model tag (e.g., "churn_classifier:v1")
        service_type: Type of service ("classification", "regression", "causal")
        input_data: Input features/data sent to the model
        output_data: Model predictions/outputs
        latency_ms: Prediction latency in milliseconds
        request_id: Optional request ID for distributed tracing
        metadata: Additional metadata to log

    Returns:
        Trace ID if successful, None otherwise

    Example:
        trace_id = await log_prediction_audit(
            model_name="churn_classifier",
            model_tag="churn_classifier:v1",
            service_type="classification",
            input_data={"features": [[0.1, 0.2, 0.3]]},
            output_data={"predictions": [1], "probabilities": [0.85]},
            latency_ms=45.2,
            request_id="req-123",
        )
    """
    if not _check_opik_available():
        return None

    try:
        from src.mlops.opik_connector import OpikConnector

        connector = OpikConnector()

        # Build metadata
        full_metadata = {
            "model_tag": model_tag,
            "service_type": service_type,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if request_id:
            full_metadata["request_id"] = request_id
        if metadata:
            full_metadata.update(metadata)

        # Log to Opik
        trace_id = await connector.log_model_prediction(
            model_name=model_name,
            input_data=input_data,
            output_data=output_data,
            metadata=full_metadata,
        )

        logger.debug(
            f"Logged prediction audit: model={model_name}, "
            f"latency={latency_ms:.2f}ms, trace_id={trace_id}"
        )

        return trace_id

    except Exception as e:
        logger.warning(f"Failed to log prediction audit: {e}")
        return None


@asynccontextmanager
async def prediction_audit_context(
    model_name: str,
    model_tag: str,
    service_type: str,
    request_id: Optional[str] = None,
):
    """Context manager for prediction audit with automatic timing.

    Usage:
        async with prediction_audit_context(
            model_name="churn_classifier",
            model_tag="churn_classifier:v1",
            service_type="classification",
        ) as ctx:
            # Do prediction
            result = model.predict(features)
            # Set input/output for audit
            ctx.set_input({"features": features.tolist()})
            ctx.set_output({"predictions": result.tolist()})

    Args:
        model_name: Name of the model
        model_tag: BentoML model tag
        service_type: Service type
        request_id: Optional request ID

    Yields:
        PredictionAuditContext for setting input/output
    """
    ctx = PredictionAuditContext(model_name, model_tag, service_type, request_id)
    ctx.start_time = time.time()
    try:
        yield ctx
    finally:
        ctx.end_time = time.time()
        latency_ms = (ctx.end_time - ctx.start_time) * 1000

        # Only log if we have both input and output
        if ctx.input_data is not None and ctx.output_data is not None:
            # Fire and forget - don't block on audit logging
            asyncio.create_task(
                log_prediction_audit(
                    model_name=model_name,
                    model_tag=model_tag,
                    service_type=service_type,
                    input_data=ctx.input_data,
                    output_data=ctx.output_data,
                    latency_ms=latency_ms,
                    request_id=request_id,
                    metadata=ctx.metadata,
                )
            )


class PredictionAuditContext:
    """Context holder for prediction audit data."""

    def __init__(
        self,
        model_name: str,
        model_tag: str,
        service_type: str,
        request_id: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_tag = model_tag
        self.service_type = service_type
        self.request_id = request_id
        self.input_data: Optional[Dict[str, Any]] = None
        self.output_data: Optional[Dict[str, Any]] = None
        self.metadata: Dict[str, Any] = {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def set_input(self, data: Dict[str, Any]) -> None:
        """Set the input data for audit."""
        self.input_data = data

    def set_output(self, data: Dict[str, Any]) -> None:
        """Set the output data for audit."""
        self.output_data = data

    def add_metadata(self, key: str, value: Any) -> None:
        """Add additional metadata."""
        self.metadata[key] = value


# =============================================================================
# Synchronous API (for non-async contexts)
# =============================================================================


def log_prediction_audit_sync(
    model_name: str,
    model_tag: str,
    service_type: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    latency_ms: float,
    request_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Synchronous version of log_prediction_audit.

    Schedules the audit logging in the background without blocking.
    Use this when you need to log from synchronous code.

    Args:
        Same as log_prediction_audit
    """
    if not _check_opik_available():
        return

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule as background task
            asyncio.create_task(
                log_prediction_audit(
                    model_name=model_name,
                    model_tag=model_tag,
                    service_type=service_type,
                    input_data=input_data,
                    output_data=output_data,
                    latency_ms=latency_ms,
                    request_id=request_id,
                    metadata=metadata,
                )
            )
        else:
            # Run synchronously if no event loop
            loop.run_until_complete(
                log_prediction_audit(
                    model_name=model_name,
                    model_tag=model_tag,
                    service_type=service_type,
                    input_data=input_data,
                    output_data=output_data,
                    latency_ms=latency_ms,
                    request_id=request_id,
                    metadata=metadata,
                )
            )
    except RuntimeError:
        # No event loop - create new one (shouldn't happen in BentoML)
        asyncio.run(
            log_prediction_audit(
                model_name=model_name,
                model_tag=model_tag,
                service_type=service_type,
                input_data=input_data,
                output_data=output_data,
                latency_ms=latency_ms,
                request_id=request_id,
                metadata=metadata,
            )
        )
