"""Admin user management over Supabase GoTrue (spec 2026-07-11).

Server-only: holds a SERVICE-ROLE client. Never import from frontend-reachable
code paths without require_admin in front.

Design facts this code encodes (all live-verified 2026-07-11 — see spec):
- Invite = generate_link(type='invite') -> hashed_token -> our own URL on
  E2I_PUBLIC_APP_URL (GoTrue SMTP is fake; GOTRUE_URI_ALLOW_LIST excludes the
  site, so we never use action_link).
- Reinvite: pending -> invite-type reissues; active -> GoTrue rejects ->
  recovery-type link.
- Disable = ban (blocks sign-in/refresh) + app_metadata.disabled=true
  (immediate API lockout via verify_supabase_token). banned_until is not
  readable through gotrue-py, so app_metadata.disabled IS the status flag.
- Role dual-write: app_metadata (API-authoritative) + chatbot_user_profiles
  (RLS). user_profiles does not exist in prod.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from supabase import create_client

logger = logging.getLogger(__name__)

VALID_ROLES = ("viewer", "analyst", "operator", "admin")
VALID_BRANDS = ("Kisqali", "Fabhalta", "Remibrutinib", "all")
BAN_DURATION = "876000h"  # ~100 years


class AdminServiceError(Exception):
    """Base error; routes map subclasses to HTTP statuses."""


class AdminValidationError(AdminServiceError):
    """Invalid role/brand/input -> 422."""


class AdminConflictError(AdminServiceError):
    """Duplicate email / GoTrue state conflict -> 409."""


class AdminGuardError(AdminServiceError):
    """Self-targeting or last-admin protection -> 403."""


class AdminNotFoundError(AdminServiceError):
    """Unknown user id -> 404."""


def _validate(role: str, brands: List[str]) -> None:
    if role not in VALID_ROLES:
        raise AdminValidationError(f"invalid role {role!r}; must be one of {VALID_ROLES}")
    if not brands:
        raise AdminValidationError("brands must be non-empty (use ['all'] for cross-brand)")
    for b in brands:
        if b not in VALID_BRANDS:
            raise AdminValidationError(f"invalid brand {b!r}; must be one of {VALID_BRANDS}")


class AdminUserService:
    """All GoTrue admin + profile dual-write operations."""

    def __init__(self) -> None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "") or os.environ.get(
            "SUPABASE_SERVICE_KEY", ""
        )
        if not url or not key:
            raise AdminServiceError("SUPABASE_URL / SUPABASE_SERVICE_KEY not configured")
        self.admin_client = create_client(url, key)
        self.public_app_url = os.environ.get("E2I_PUBLIC_APP_URL", "https://eznomics.site")

    # ------------------------------------------------------------------ users

    def _get_auth_user(self, user_id: str):
        try:
            resp = self.admin_client.auth.admin.get_user_by_id(user_id)
        except Exception as e:
            raise AdminNotFoundError(f"user {user_id} not found: {e}") from e
        if resp is None or resp.user is None:
            raise AdminNotFoundError(f"user {user_id} not found")
        return resp.user

    @staticmethod
    def _status(user) -> str:
        meta = user.app_metadata or {}
        if meta.get("disabled"):
            return "disabled"
        if user.last_sign_in_at is None:
            return "invited"
        return "active"

    def _list_all_auth_users(self) -> List[Any]:
        """Every auth user, across pages. list_users caps at per_page; the
        last-admin guard and invite dedup must never undercount past 1000."""
        out: List[Any] = []
        page = 1
        while True:
            batch = self.admin_client.auth.admin.list_users(page=page, per_page=1000)
            out.extend(batch)
            if len(batch) < 1000:
                return out
            page += 1

    def list_users(self) -> List[Dict[str, Any]]:
        auth_users = self._list_all_auth_users()
        profiles = (
            self.admin_client.table("chatbot_user_profiles")
            .select("id, role, total_conversations, total_messages, last_active_at")
            .execute()
        )
        by_id = {p["id"]: p for p in profiles.data}
        out = []
        for u in auth_users:
            meta = u.app_metadata or {}
            p = by_id.get(u.id, {})
            out.append(
                {
                    "id": u.id,
                    "email": u.email,
                    "full_name": (u.user_metadata or {}).get("full_name"),
                    "role": meta.get("role") or p.get("role") or "viewer",
                    "brands": meta.get("brands") or [],
                    "status": self._status(u),
                    "created_at": str(u.created_at) if u.created_at else None,
                    "last_sign_in_at": str(u.last_sign_in_at) if u.last_sign_in_at else None,
                    "total_conversations": p.get("total_conversations") or 0,
                    "total_messages": p.get("total_messages") or 0,
                    "last_active_at": p.get("last_active_at"),
                }
            )
        return sorted(out, key=lambda x: x["created_at"] or "", reverse=True)

    # ---------------------------------------------------------------- profile

    def _upsert_profile(self, user_id: str, email: str, role: str) -> None:
        self.admin_client.table("chatbot_user_profiles").upsert(
            {
                "id": user_id,
                "email": email,
                "role": role,
                "is_admin": role == "admin",
            },
            on_conflict="id",
        ).execute()

    # ----------------------------------------------------------------- invite

    def _accept_link(self, hashed_token: str) -> str:
        return f"{self.public_app_url}/accept-invite?token_hash={quote(hashed_token)}"

    def invite_user(
        self,
        email: str,
        role: str,
        brands: List[str],
        full_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        _validate(role, brands)
        # GoTrue re-issues invite links for PENDING duplicates instead of
        # erroring (verified 2026-07-11), so existence is checked explicitly:
        # inviting an existing user (any state) is a 409 — reinvite is the
        # intentional path for a fresh link.
        lowered = email.lower()
        for existing in self._list_all_auth_users():
            if existing.email and existing.email.lower() == lowered:
                raise AdminConflictError(f"{email} is already registered")
        try:
            resp = self.admin_client.auth.admin.generate_link(
                {"type": "invite", "email": email}
            )
        except Exception as e:
            if "already been registered" in str(e):
                raise AdminConflictError(f"{email} is already registered") from e
            raise
        user = resp.user
        attrs: Dict[str, Any] = {"app_metadata": {"role": role, "brands": brands}}
        if full_name:
            attrs["user_metadata"] = {"full_name": full_name}
        self.admin_client.auth.admin.update_user_by_id(user.id, attrs)
        self._upsert_profile(user.id, email, role)
        return {
            "user_id": user.id,
            "email": email,
            "invite_link": self._accept_link(resp.properties.hashed_token),
            "link_type": "invite",
        }

    def reinvite_user(self, user_id: str) -> Dict[str, Any]:
        user = self._get_auth_user(user_id)
        # A fresh link would let a disabled user set a password while banned,
        # primed to sign in the moment they're re-enabled. Refuse instead.
        self._guard_not_disabled(user, "reinvite")
        try:
            resp = self.admin_client.auth.admin.generate_link(
                {"type": "invite", "email": user.email}
            )
            link_type = "invite"
        except Exception as e:
            if "already been registered" not in str(e):
                raise
            resp = self.admin_client.auth.admin.generate_link(
                {"type": "recovery", "email": user.email}
            )
            link_type = "recovery"
        return {
            "user_id": user.id,
            "email": user.email,
            "invite_link": self._accept_link(resp.properties.hashed_token),
            "link_type": link_type,
        }

    def recovery_link(self, user_id: str) -> Dict[str, Any]:
        user = self._get_auth_user(user_id)
        self._guard_not_disabled(user, "issue a recovery link for")
        resp = self.admin_client.auth.admin.generate_link(
            {"type": "recovery", "email": user.email}
        )
        return {
            "user_id": user.id,
            "email": user.email,
            "invite_link": self._accept_link(resp.properties.hashed_token),
            "link_type": "recovery",
        }

    # ----------------------------------------------------------------- guards

    def _enabled_admin_ids(self) -> List[str]:
        return [
            u.id
            for u in self._list_all_auth_users()
            if (u.app_metadata or {}).get("role") == "admin"
            and not (u.app_metadata or {}).get("disabled")
        ]

    def _guard_not_self(self, user_id: str, acting_admin_id: str, action: str) -> None:
        if user_id == acting_admin_id:
            raise AdminGuardError(f"admins cannot {action} their own account")

    def _guard_not_last_admin(self, user_id: str, action: str) -> None:
        admins = self._enabled_admin_ids()
        if user_id in admins and len(admins) <= 1:
            raise AdminGuardError(f"cannot {action} the last enabled admin")

    def _guard_not_disabled(self, user, action: str) -> None:
        if self._status(user) == "disabled":
            raise AdminGuardError(f"cannot {action} a disabled user — enable them first")

    def _verify_admins_remain(self, compensate, action: str) -> None:
        """Close _guard_not_last_admin's TOCTOU window. GoTrue writes can't be
        wrapped in a transaction with the guard's read, so two concurrent
        requests targeting the last two admins can both pass the pre-write
        check. Re-count AFTER the write; if the platform is left with zero
        enabled admins, run the compensating write and refuse."""
        if not self._enabled_admin_ids():
            compensate()
            raise AdminGuardError(
                f"{action} reverted: a concurrent change would have left no enabled admins"
            )

    # ------------------------------------------------------------------ write

    def update_user(
        self,
        user_id: str,
        acting_admin_id: str,
        role: Optional[str] = None,
        brands: Optional[List[str]] = None,
        full_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        user = self._get_auth_user(user_id)
        meta = dict(user.app_metadata or {})
        new_role = role or meta.get("role") or "viewer"
        new_brands = brands if brands is not None else (meta.get("brands") or ["all"])
        _validate(new_role, new_brands)
        demoting_admin = (
            meta.get("role") == "admin" and new_role != "admin" and not meta.get("disabled")
        )
        if meta.get("role") == "admin" and new_role != "admin":
            self._guard_not_last_admin(user_id, "demote")
        attrs: Dict[str, Any] = {
            "app_metadata": {**meta, "role": new_role, "brands": new_brands}
        }
        if full_name is not None:
            attrs["user_metadata"] = {**(user.user_metadata or {}), "full_name": full_name}
        self.admin_client.auth.admin.update_user_by_id(user_id, attrs)
        self._upsert_profile(user_id, user.email, new_role)
        if demoting_admin:

            def _repromote() -> None:
                self.admin_client.auth.admin.update_user_by_id(
                    user_id, {"app_metadata": meta}
                )
                self._upsert_profile(user_id, user.email, "admin")

            self._verify_admins_remain(_repromote, "demote")
        return {"user_id": user_id, "role": new_role, "brands": new_brands}

    def disable_user(self, user_id: str, acting_admin_id: str) -> Dict[str, Any]:
        self._guard_not_self(user_id, acting_admin_id, "disable")
        self._guard_not_last_admin(user_id, "disable")
        user = self._get_auth_user(user_id)
        meta = dict(user.app_metadata or {})
        was_enabled_admin = meta.get("role") == "admin" and not meta.get("disabled")
        meta["disabled"] = True
        self.admin_client.auth.admin.update_user_by_id(
            user_id, {"app_metadata": meta, "ban_duration": BAN_DURATION}
        )
        if was_enabled_admin:

            def _reenable() -> None:
                self.admin_client.auth.admin.update_user_by_id(
                    user_id,
                    {"app_metadata": {**meta, "disabled": False}, "ban_duration": "none"},
                )

            self._verify_admins_remain(_reenable, "disable")
        return {"user_id": user_id, "status": "disabled"}

    def enable_user(self, user_id: str) -> Dict[str, Any]:
        user = self._get_auth_user(user_id)
        meta = dict(user.app_metadata or {})
        meta["disabled"] = False
        self.admin_client.auth.admin.update_user_by_id(
            user_id, {"app_metadata": meta, "ban_duration": "none"}
        )
        return {"user_id": user_id, "status": "active" if user.last_sign_in_at else "invited"}

    def delete_user(self, user_id: str, acting_admin_id: str) -> Dict[str, Any]:
        self._guard_not_self(user_id, acting_admin_id, "delete")
        self._guard_not_last_admin(user_id, "delete")
        user = self._get_auth_user(user_id)  # 404 before delete
        meta = dict(user.app_metadata or {})
        if meta.get("role") == "admin" and not meta.get("disabled"):
            # Deletion is not compensable, so an admin target is demoted
            # first (compensable); once the demote survives the post-write
            # re-count, the delete can no longer remove the last admin.
            self.admin_client.auth.admin.update_user_by_id(
                user_id, {"app_metadata": {**meta, "role": "viewer"}}
            )

            def _restore_role() -> None:
                self.admin_client.auth.admin.update_user_by_id(
                    user_id, {"app_metadata": meta}
                )

            self._verify_admins_remain(_restore_role, "delete")
        self.admin_client.auth.admin.delete_user(user_id)
        # Prod chatbot_user_profiles has NO FK to auth.users (intentional: the
        # anonymous@copilotkit.system profile has no auth row), so the profile
        # is removed explicitly. Activity/audit history is kept on purpose.
        self.admin_client.table("chatbot_user_profiles").delete().eq("id", user_id).execute()
        return {"user_id": user_id, "email": user.email, "deleted": True}

    # --------------------------------------------------------------- activity

    def user_activity(self, user_id: str, days: int = 90) -> Dict[str, Any]:
        user = self._get_auth_user(user_id)
        auth_events = self.admin_client.rpc(
            "admin_get_login_activity", {"p_user_id": user_id, "p_days": days}
        ).execute()
        api_rows = (
            self.admin_client.table("user_activity_log")
            .select("endpoint_group, http_method, bucket_minute, request_count")
            .eq("user_id", user_id)
            .order("bucket_minute", desc=True)
            .limit(5000)
            .execute()
        )
        recent = self.admin_client.rpc(
            "admin_get_user_recent_events", {"p_user_id": user_id, "p_limit": 25}
        ).execute()
        profile = (
            self.admin_client.table("chatbot_user_profiles")
            .select("total_conversations, total_messages, last_active_at")
            .eq("id", user_id)
            .execute()
        )
        chat = (
            profile.data[0]
            if profile.data
            else {"total_conversations": 0, "total_messages": 0, "last_active_at": None}
        )
        return {
            "user_id": user_id,
            "email": user.email,
            "auth_events": auth_events.data or [],
            "api_activity": api_rows.data or [],
            "recent_events": recent.data or [],
            "chat": chat,
        }

    def platform_activity(self, days: int = 30) -> Dict[str, Any]:
        rows = self.admin_client.rpc(
            "admin_get_platform_activity", {"p_days": days}
        ).execute()
        return {"days": rows.data or []}

    # -------------------------------------------------------------- reconcile

    def reconcile_role_stores(self) -> List[Dict[str, Any]]:
        """One-time drift repair: users with NO jwt role but a profile role get
        app_metadata backfilled from the profile (migration 101 already synced
        the other direction, jwt -> profile)."""
        report: List[Dict[str, Any]] = []
        profiles = (
            self.admin_client.table("chatbot_user_profiles").select("id, role").execute()
        )
        by_id = {p["id"]: p.get("role") for p in profiles.data}
        for u in self._list_all_auth_users():
            meta = dict(u.app_metadata or {})
            if not meta.get("role") and by_id.get(u.id):
                meta["role"] = by_id[u.id]
                meta.setdefault("brands", ["all"])
                self.admin_client.auth.admin.update_user_by_id(u.id, {"app_metadata": meta})
                report.append(
                    {"user_id": u.id, "action": "app_metadata_backfilled", "role": meta["role"]}
                )
        return report
