"""
E2I Audit Chain API
====================

FastAPI endpoints for audit chain verification and inspection.

Endpoints:
- GET /audit/workflow/{workflow_id}: Get all entries for a workflow
- GET /audit/workflow/{workflow_id}/verify: Verify chain integrity
- GET /audit/workflow/{workflow_id}/summary: Get workflow summary
- GET /audit/recent: Get recent audit workflows

Integration Points:
- AuditChainService for chain operations
- Supabase for persistence
- All 18 agents emit audit entries

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from src.api.dependencies.auth import require_auth
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.utils.audit_chain import (
    AuditChainService,
)
from src.utils.type_helpers import parse_supabase_row, parse_supabase_rows

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/audit",
    tags=["Audit Chain"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class AuditEntryResponse(BaseModel):
    """Response model for a single audit chain entry."""

    entry_id: UUID
    workflow_id: UUID
    sequence_number: int
    agent_name: str
    agent_tier: int
    action_type: str
    created_at: datetime
    duration_ms: Optional[int] = None

    # Validation & confidence
    validation_passed: Optional[bool] = None
    confidence_score: Optional[float] = None
    refutation_results: Optional[Dict[str, Any]] = None

    # Hash chain fields
    previous_entry_id: Optional[UUID] = None
    previous_hash: Optional[str] = None
    entry_hash: str

    # Context
    user_id: Optional[str] = None
    session_id: Optional[UUID] = None
    brand: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ChainVerificationResponse(BaseModel):
    """Response model for chain verification result."""

    workflow_id: UUID
    is_valid: bool
    entries_checked: int
    first_invalid_entry: Optional[UUID] = None
    error_message: Optional[str] = None
    verified_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WorkflowSummaryResponse(BaseModel):
    """Response model for workflow summary."""

    workflow_id: UUID
    total_entries: int
    first_entry_at: Optional[datetime] = None
    last_entry_at: Optional[datetime] = None
    agents_involved: List[str]
    tiers_involved: List[int]
    chain_verified: bool
    brand: Optional[str] = None

    # Aggregated metrics
    total_duration_ms: int = 0
    avg_confidence_score: Optional[float] = None
    validation_passed_count: int = 0
    validation_failed_count: int = 0


class RecentWorkflowResponse(BaseModel):
    """Response model for recent workflow listing."""

    workflow_id: UUID
    started_at: datetime
    entry_count: int
    first_agent: str
    last_agent: str
    brand: Optional[str] = None


# =============================================================================
# DEPENDENCY - Get Audit Service
# =============================================================================


def get_audit_service() -> Optional[AuditChainService]:
    """
    Get the global AuditChainService instance.

    Returns None if service is not initialized (graceful degradation).
    """
    from src.agents.base.audit_chain_mixin import get_audit_chain_service

    return get_audit_chain_service()


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get(
    "/workflow/{workflow_id}",
    response_model=List[AuditEntryResponse],
    summary="Get workflow audit entries",
    operation_id="get_workflow_entries",
    description="Retrieve all audit chain entries for a specific workflow, ordered by sequence number.",
)
async def get_workflow_entries(
    workflow_id: UUID,
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum entries to return"),
    offset: int = Query(default=0, ge=0, description="Number of entries to skip"),
    user: Dict[str, Any] = Depends(require_auth),
) -> List[AuditEntryResponse]:
    """Get all audit entries for a workflow."""
    service = get_audit_service()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Audit chain service unavailable. Check service configuration.",
        )

    try:
        # Query entries from database
        result = (
            service.db.table("audit_chain_entries")
            .select("*")
            .eq("workflow_id", str(workflow_id))
            .order("sequence_number")
            .range(offset, offset + limit - 1)
            .execute()
        )

        if not result.data:
            return []

        # Convert to response models
        entries = []
        for row in parse_supabase_rows(result.data):
            entries.append(_row_to_response(row))

        return entries

    except Exception as e:
        logger.error(f"Failed to get workflow entries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve entries: {str(e)}")


@router.get(
    "/workflow/{workflow_id}/verify",
    response_model=ChainVerificationResponse,
    summary="Verify chain integrity",
    operation_id="verify_workflow_chain",
    description="Verify the cryptographic integrity of a workflow's audit chain.",
)
async def verify_workflow_chain(
    workflow_id: UUID,
    user: Dict[str, Any] = Depends(require_auth),
) -> ChainVerificationResponse:
    """Verify the integrity of a workflow's audit chain."""
    service = get_audit_service()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Audit chain service unavailable. Check service configuration.",
        )

    try:
        verification = service.verify_workflow(workflow_id)

        return ChainVerificationResponse(
            workflow_id=workflow_id,
            is_valid=verification.is_valid,
            entries_checked=verification.entries_checked,
            first_invalid_entry=verification.first_invalid_entry,
            error_message=verification.error_message,
            verified_at=verification.verified_at,
        )

    except Exception as e:
        logger.error(f"Failed to verify workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.get(
    "/workflow/{workflow_id}/summary",
    response_model=WorkflowSummaryResponse,
    summary="Get workflow summary",
    operation_id="get_workflow_summary",
    description="Get a summary of a workflow including agents involved and aggregated metrics.",
)
async def get_workflow_summary(
    workflow_id: UUID,
    user: Dict[str, Any] = Depends(require_auth),
) -> WorkflowSummaryResponse:
    """Get a summary of a workflow's audit chain."""
    service = get_audit_service()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Audit chain service unavailable. Check service configuration.",
        )

    try:
        # Get all entries for the workflow
        result = (
            service.db.table("audit_chain_entries")
            .select("*")
            .eq("workflow_id", str(workflow_id))
            .order("sequence_number")
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        entries = parse_supabase_rows(result.data)

        # Aggregate data
        agents = list({e["agent_name"] for e in entries})
        tiers = list({e["agent_tier"] for e in entries})

        total_duration = sum(e.get("duration_ms") or 0 for e in entries)

        confidence_scores = [
            e["confidence_score"] for e in entries if e.get("confidence_score") is not None
        ]
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else None
        )

        validation_passed = sum(1 for e in entries if e.get("validation_passed") is True)
        validation_failed = sum(1 for e in entries if e.get("validation_passed") is False)

        # Check chain validity
        try:
            verification = service.verify_workflow(workflow_id)
            chain_verified = verification.is_valid
        except Exception:
            chain_verified = False

        # Get brand from first entry
        brand = entries[0].get("brand") if entries else None

        return WorkflowSummaryResponse(
            workflow_id=workflow_id,
            total_entries=len(entries),
            first_entry_at=datetime.fromisoformat(entries[0]["created_at"].replace("Z", "+00:00")),
            last_entry_at=datetime.fromisoformat(entries[-1]["created_at"].replace("Z", "+00:00")),
            agents_involved=sorted(agents),
            tiers_involved=sorted(tiers),
            chain_verified=chain_verified,
            brand=brand,
            total_duration_ms=total_duration,
            avg_confidence_score=avg_confidence,
            validation_passed_count=validation_passed,
            validation_failed_count=validation_failed,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@router.get(
    "/recent",
    response_model=List[RecentWorkflowResponse],
    summary="Get recent workflows",
    operation_id="get_recent_workflows",
    description="Get a list of recent audit workflows with basic info.",
)
async def get_recent_workflows(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum workflows to return"),
    brand: Optional[str] = Query(default=None, description="Filter by brand"),
    agent_name: Optional[str] = Query(default=None, description="Filter by agent name"),
    user: Dict[str, Any] = Depends(require_auth),
) -> List[RecentWorkflowResponse]:
    """Get recent audit workflows."""
    service = get_audit_service()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Audit chain service unavailable. Check service configuration.",
        )

    try:
        # Use SQL view for efficient aggregation
        query = service.db.rpc(
            "get_recent_audit_workflows",
            {"p_limit": limit, "p_brand": brand, "p_agent_name": agent_name},
        )
        result = query.execute()

        if not result.data:
            # Fallback: direct query if RPC not available
            return await _get_recent_workflows_fallback(service, limit, brand, agent_name)

        return [
            RecentWorkflowResponse(
                workflow_id=UUID(row["workflow_id"]),
                started_at=datetime.fromisoformat(row["started_at"].replace("Z", "+00:00")),
                entry_count=row["entry_count"],
                first_agent=row["first_agent"],
                last_agent=row["last_agent"],
                brand=row.get("brand"),
            )
            for row in parse_supabase_rows(result.data)
        ]

    except Exception as e:
        logger.warning(f"RPC failed, using fallback: {e}")
        return await _get_recent_workflows_fallback(service, limit, brand, agent_name)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _row_to_response(row: Dict[str, Any]) -> AuditEntryResponse:
    """Convert a database row to AuditEntryResponse."""
    return AuditEntryResponse(
        entry_id=UUID(row["entry_id"]),
        workflow_id=UUID(row["workflow_id"]),
        sequence_number=row["sequence_number"],
        agent_name=row["agent_name"],
        agent_tier=row["agent_tier"],
        action_type=row["action_type"],
        created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
        duration_ms=row.get("duration_ms"),
        validation_passed=row.get("validation_passed"),
        confidence_score=row.get("confidence_score"),
        refutation_results=row.get("refutation_results"),
        previous_entry_id=(
            UUID(row["previous_entry_id"]) if row.get("previous_entry_id") else None
        ),
        previous_hash=row.get("previous_hash"),
        entry_hash=row["entry_hash"],
        user_id=row.get("user_id"),
        session_id=UUID(row["session_id"]) if row.get("session_id") else None,
        brand=row.get("brand"),
    )


async def _get_recent_workflows_fallback(
    service: AuditChainService,
    limit: int,
    brand: Optional[str],
    agent_name: Optional[str],
) -> List[RecentWorkflowResponse]:
    """Fallback method to get recent workflows without RPC."""
    try:
        # Get distinct workflow_ids with first entry
        query = service.db.table("audit_chain_entries").select(
            "workflow_id, created_at, agent_name, brand"
        )

        if brand:
            query = query.eq("brand", brand)
        if agent_name:
            query = query.eq("agent_name", agent_name)

        query = query.eq("sequence_number", 1).order("created_at", desc=True).limit(limit)

        result = query.execute()

        if not result.data:
            return []

        workflows = []
        for row in parse_supabase_rows(result.data):
            workflow_id = UUID(row["workflow_id"])

            # Get entry count and last agent
            count_result = (
                service.db.table("audit_chain_entries")
                .select("agent_name, sequence_number")
                .eq("workflow_id", str(workflow_id))
                .order("sequence_number", desc=True)
                .limit(1)
                .execute()
            )

            entry_count = 1
            last_agent = row["agent_name"]
            if count_result.data:
                count_row = parse_supabase_row(count_result.data[0])
                entry_count = count_row["sequence_number"]
                last_agent = count_row["agent_name"]

            workflows.append(
                RecentWorkflowResponse(
                    workflow_id=workflow_id,
                    started_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
                    entry_count=entry_count,
                    first_agent=row["agent_name"],
                    last_agent=last_agent,
                    brand=row.get("brand"),
                )
            )

        return workflows

    except Exception as e:
        logger.error(f"Fallback query failed: {e}")
        return []
