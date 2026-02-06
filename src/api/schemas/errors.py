"""
Error Response Schemas
======================

Pydantic models for structured API error responses.  These mirror the
``E2IError.to_dict()`` output from ``src/api/errors.py`` so that every
router can reference them in ``responses={}``.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SchemaError(BaseModel):
    """A single field-level validation error."""

    field: str = Field(..., description="Dot-delimited field path (e.g. 'body.treatment_var')")
    message: str = Field(..., description="Human-readable validation message")
    type: str = Field(..., description="Pydantic error type code")


class ErrorResponse(BaseModel):
    """Standard structured error returned by all API endpoints."""

    error: str = Field(..., description="Error class name (e.g. 'NotFoundError')")
    error_id: str = Field(..., description="Short unique ID for support reference")
    category: str = Field(
        ...,
        description="Error category",
        examples=["validation", "authentication", "not_found", "internal", "agent_error"],
    )
    message: str = Field(..., description="Human-readable error message")
    timestamp: datetime = Field(..., description="UTC timestamp of the error")
    suggested_action: Optional[str] = Field(
        None, description="Actionable guidance for resolving the error"
    )
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "NotFoundError",
                "error_id": "a1b2c3d4",
                "category": "not_found",
                "message": "Endpoint '/api/nonexistent' not found",
                "timestamp": "2026-02-06T12:00:00Z",
                "suggested_action": "Check the API documentation at /api/docs for available endpoints.",
            }
        }
    )


class ValidationErrorResponse(ErrorResponse):
    """Structured error returned when request validation fails (HTTP 422)."""

    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Validation details including schema_errors list",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "SchemaValidationError",
                "error_id": "e5f6g7h8",
                "category": "validation",
                "message": "Request validation failed",
                "timestamp": "2026-02-06T12:00:00Z",
                "suggested_action": "Review the schema errors and correct your request.",
                "details": {
                    "schema_errors": [
                        {
                            "field": "body.treatment_var",
                            "message": "Field required",
                            "type": "missing",
                        }
                    ]
                },
            }
        }
    )
