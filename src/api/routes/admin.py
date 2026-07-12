"""Admin user management endpoints (spec 2026-07-11). ALL endpoints require
the admin role via require_admin; the router is NOT in PUBLIC_PATHS, so the
JWT middleware gates it before RBAC even runs. Every mutation is audited to
security_audit_log via the (now-fixed) SecurityAuditService."""

import asyncio
import ipaddress
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from src.api.dependencies.auth import require_admin
from src.services.admin_user_service import (
    AdminConflictError,
    AdminGuardError,
    AdminNotFoundError,
    AdminServiceError,
    AdminUserService,
    AdminValidationError,
)
from src.services.llm_observability_service import LLMObservabilityService
from src.utils.security_audit import (
    SecurityAuditEvent,
    SecurityEventSeverity,
    SecurityEventType,
    get_security_audit_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

_service: Optional[AdminUserService] = None


def get_admin_service() -> AdminUserService:
    global _service
    if _service is None:
        _service = AdminUserService()
    return _service


_obs_service: Optional[LLMObservabilityService] = None


def get_llm_observability_service() -> LLMObservabilityService:
    global _obs_service
    if _obs_service is None:
        _obs_service = LLMObservabilityService()
    return _obs_service


def _audit(
    event_type: SecurityEventType,
    admin: Dict[str, Any],
    request: Request,
    message: str,
    target_user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Audit-log an admin action. Never blocks the action (existing convention)."""
    try:
        # security_audit_log.client_ip is INET — non-IP hosts (e.g. the test
        # client's literal "testclient") would fail the insert silently.
        raw_ip = request.headers.get("x-real-ip") or (
            request.client.host if request.client else None
        )
        try:
            client_ip = str(ipaddress.ip_address(raw_ip)) if raw_ip else None
        except ValueError:
            client_ip = None
        get_security_audit_service().log_event(
            SecurityAuditEvent(
                event_id=uuid4(),
                event_type=event_type,
                severity=SecurityEventSeverity.WARNING,
                timestamp=datetime.now(timezone.utc),
                message=message,
                user_id=str(admin.get("id")),
                user_email=admin.get("email"),
                client_ip=client_ip,
                endpoint=str(request.url.path),
                http_method=request.method,
                resource_type="auth_user",
                resource_id=target_user_id,
                action_result="success",
                metadata=metadata or {},
            )
        )
    except Exception:
        logger.warning("admin audit logging failed (non-blocking)", exc_info=True)


def _map_error(e: AdminServiceError) -> HTTPException:
    if isinstance(e, AdminValidationError):
        return HTTPException(status_code=422, detail=str(e))
    if isinstance(e, AdminConflictError):
        return HTTPException(status_code=409, detail=str(e))
    if isinstance(e, AdminGuardError):
        return HTTPException(status_code=403, detail=str(e))
    if isinstance(e, AdminNotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    return HTTPException(status_code=502, detail=f"auth service error: {e}")


# ------------------------------------------------------------------- schemas


_EMAIL_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class InviteRequest(BaseModel):
    # plain str + shape check: EmailStr would pull in the email-validator
    # package (not pinned here); GoTrue validates authoritatively on invite.
    email: str
    role: str = "viewer"
    brands: List[str] = Field(default_factory=lambda: ["all"])
    full_name: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _email_shape(cls, v: str) -> str:
        v = v.strip()
        if not _EMAIL_SHAPE.match(v):
            raise ValueError("invalid email address")
        return v


class UpdateUserRequest(BaseModel):
    role: Optional[str] = None
    brands: Optional[List[str]] = None
    full_name: Optional[str] = None


class UsersResponse(BaseModel):
    users: List[Dict[str, Any]]


class LinkResponse(BaseModel):
    user_id: str
    email: str
    invite_link: str
    link_type: str


# ----------------------------------------------------------------- endpoints


@router.get("/users", response_model=UsersResponse)
async def list_users(
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> UsersResponse:
    users = await asyncio.to_thread(service.list_users)
    return UsersResponse(users=users)


@router.post("/users/invite", response_model=LinkResponse)
async def invite_user(
    body: InviteRequest,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> LinkResponse:
    try:
        result = await asyncio.to_thread(
            service.invite_user, body.email, body.role, body.brands, body.full_name
        )
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.ADMIN_USER_MODIFIED,
        admin,
        request,
        f"Invited {body.email} as {body.role}",
        target_user_id=result["user_id"],
        metadata={"action": "invite", "role": body.role, "brands": body.brands},
    )
    return LinkResponse(**result)


@router.post("/users/{user_id}/reinvite", response_model=LinkResponse)
async def reinvite_user(
    user_id: str,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> LinkResponse:
    try:
        result = await asyncio.to_thread(service.reinvite_user, user_id)
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.ADMIN_USER_MODIFIED,
        admin,
        request,
        f"Re-invited {result['email']} ({result['link_type']} link)",
        target_user_id=user_id,
        metadata={"action": "reinvite", "link_type": result["link_type"]},
    )
    return LinkResponse(**result)


@router.post("/users/{user_id}/recovery-link", response_model=LinkResponse)
async def recovery_link(
    user_id: str,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> LinkResponse:
    try:
        result = await asyncio.to_thread(service.recovery_link, user_id)
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.ADMIN_USER_MODIFIED,
        admin,
        request,
        f"Generated recovery link for {result['email']}",
        target_user_id=user_id,
        metadata={"action": "recovery_link"},
    )
    return LinkResponse(**result)


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    body: UpdateUserRequest,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    try:
        result = await asyncio.to_thread(
            service.update_user,
            user_id,
            str(admin.get("id")),
            body.role,
            body.brands,
            body.full_name,
        )
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.AUTHZ_ROLE_CHANGE,
        admin,
        request,
        f"Updated user {user_id}: role={result['role']} brands={result['brands']}",
        target_user_id=user_id,
        metadata={"action": "update", **result},
    )
    return result


@router.post("/users/{user_id}/disable")
async def disable_user(
    user_id: str,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    try:
        result = await asyncio.to_thread(service.disable_user, user_id, str(admin.get("id")))
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.AUTHZ_PERMISSION_REVOKED,
        admin,
        request,
        f"Disabled user {user_id}",
        target_user_id=user_id,
        metadata={"action": "disable"},
    )
    return result


@router.post("/users/{user_id}/enable")
async def enable_user(
    user_id: str,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    try:
        result = await asyncio.to_thread(service.enable_user, user_id)
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.AUTHZ_PERMISSION_GRANTED,
        admin,
        request,
        f"Enabled user {user_id}",
        target_user_id=user_id,
        metadata={"action": "enable"},
    )
    return result


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    try:
        result = await asyncio.to_thread(service.delete_user, user_id, str(admin.get("id")))
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.ADMIN_USER_MODIFIED,
        admin,
        request,
        f"DELETED user {result['email']}",
        target_user_id=user_id,
        metadata={"action": "delete", "email": result["email"]},
    )
    return result


@router.get("/users/{user_id}/activity")
async def user_activity(
    user_id: str,
    days: int = Query(default=90, ge=1, le=365),
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(service.user_activity, user_id, days)
    except AdminServiceError as e:
        raise _map_error(e) from e


@router.get("/activity/overview")
async def activity_overview(
    days: int = Query(default=30, ge=1, le=365),
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    return await asyncio.to_thread(service.platform_activity, days)


@router.get("/observability/llm-usage")
async def llm_usage_overview(
    days: int = Query(default=30, ge=1, le=365),
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
    obs: LLMObservabilityService = Depends(get_llm_observability_service),
) -> Dict[str, Any]:
    """LLM usage/tokens/cost: per-user + per-session (chat) and platform
    aggregates (spec 2026-07-12). Cost computed at read time from the
    pricing table; unpriced models surface in unpriced_models."""

    def _query() -> Dict[str, Any]:
        users = service.list_users()
        return obs.llm_usage(days, users)

    return await asyncio.to_thread(_query)


@router.get("/audit")
async def admin_audit_feed(
    days: int = Query(default=30, ge=1, le=365),
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    def _query() -> List[Dict[str, Any]]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = (
            service.admin_client.table("security_audit_log")
            .select(
                "event_id, event_type, severity, timestamp, message, "
                "user_email, resource_id, metadata"
            )
            .gte("timestamp", since)
            .order("timestamp", desc=True)
            .limit(200)
            .execute()
        )
        return cast(List[Dict[str, Any]], rows.data or [])

    return {"events": await asyncio.to_thread(_query)}
