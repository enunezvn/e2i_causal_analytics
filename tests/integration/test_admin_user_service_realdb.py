"""AdminUserService against the REAL local Supabase stack (no mocks).

Disposable users use the +admsvc email tag; every test cleans up after itself.

    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_admin_user_service_realdb.py -p no:cacheprovider -v
"""

import os
import secrets

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

TAG = "+admsvc"
# Runtime-randomized so no credential-shaped literal ever lands in git history
# (GitGuardian flags even throwaway test passwords, and its PR check re-scans
# all historical commits forever).
PW = "Adm!2" + secrets.token_urlsafe(12)


@pytest.fixture()
def svc():
    from src.services.admin_user_service import AdminUserService

    service = AdminUserService()
    yield service
    # cleanup ALL disposable users this file may have created — auth user AND
    # profile row (no FK in prod, the signup trigger recreates profiles)
    for u in service.admin_client.auth.admin.list_users():
        if u.email and TAG in u.email:
            service.admin_client.auth.admin.delete_user(u.id)
    service.admin_client.table("chatbot_user_profiles").delete().like("email", f"%{TAG}%").execute()


def test_list_users_merges_auth_and_profile(svc):
    users = svc.list_users()
    assert len(users) >= 8  # the 8 real users
    me = next(u for u in users if u["email"] == "etn3724@gmail.com")
    assert me["role"] == "admin"
    assert me["status"] == "active"
    assert me["last_sign_in_at"] is not None
    # profile join fields present (backfilled by migration 101)
    assert "total_messages" in me and "last_active_at" in me


def test_invite_creates_pending_user_with_role_and_link(svc):
    email = f"etn3724{TAG}-inv@gmail.com"
    result = svc.invite_user(email=email, role="analyst", brands=["Kisqali"], full_name="Inv Test")
    assert result["invite_link"].startswith("https://eznomics.site/accept-invite?token_hash=")
    # dual-write landed
    user = next(u for u in svc.list_users() if u["email"] == email)
    assert user["role"] == "analyst"
    assert user["brands"] == ["Kisqali"]
    assert user["status"] == "invited"  # never signed in
    profile = (
        svc.admin_client.table("chatbot_user_profiles")
        .select("role, is_admin")
        .eq("id", user["id"])
        .execute()
    )
    assert profile.data[0]["role"] == "analyst"
    assert profile.data[0]["is_admin"] is False


def test_invite_duplicate_email_raises_conflict(svc):
    from src.services.admin_user_service import AdminConflictError

    email = f"etn3724{TAG}-dup@gmail.com"
    svc.invite_user(email=email, role="viewer", brands=["all"])
    with pytest.raises(AdminConflictError):
        svc.invite_user(email=email, role="viewer", brands=["all"])


def test_invite_link_completes_verify_otp(svc):
    """The returned token_hash must actually work — the whole feature hinges
    on this (proven live 2026-07-11; this test pins it forever)."""
    from urllib.parse import parse_qs, urlparse

    from supabase import create_client

    email = f"etn3724{TAG}-otp@gmail.com"
    result = svc.invite_user(email=email, role="viewer", brands=["all"])
    token_hash = parse_qs(urlparse(result["invite_link"]).query)["token_hash"][0]
    anon = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"],
    )
    verified = anon.auth.verify_otp({"type": "invite", "token_hash": token_hash})
    assert verified.session is not None
    anon.auth.update_user({"password": PW})
    anon.auth.sign_out()
    fresh = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"],
    )
    signed = fresh.auth.sign_in_with_password({"email": email, "password": PW})
    assert signed.session is not None


def test_reinvite_pending_reissues_and_active_falls_back_to_recovery(svc):
    email = f"etn3724{TAG}-re@gmail.com"
    first = svc.invite_user(email=email, role="viewer", brands=["all"])
    user_id = next(u["id"] for u in svc.list_users() if u["email"] == email)

    second = svc.reinvite_user(user_id)
    assert second["invite_link"] != first["invite_link"]
    assert second["link_type"] == "invite"

    # activate the user, then reinvite must fall back to recovery
    from urllib.parse import parse_qs, urlparse

    from supabase import create_client

    token_hash = parse_qs(urlparse(second["invite_link"]).query)["token_hash"][0]
    anon = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"],
    )
    anon.auth.verify_otp({"type": "invite", "token_hash": token_hash})
    anon.auth.update_user({"password": PW})

    third = svc.reinvite_user(user_id)
    assert third["link_type"] == "recovery"


def test_recovery_link_for_active_user(svc):
    me = next(u for u in svc.list_users() if u["email"] == "etn3724@gmail.com")
    link = svc.recovery_link(me["id"])
    assert link["invite_link"].startswith("https://eznomics.site/accept-invite?token_hash=")
    assert link["link_type"] == "recovery"


def test_invalid_role_and_brand_rejected(svc):
    from src.services.admin_user_service import AdminValidationError

    with pytest.raises(AdminValidationError):
        svc.invite_user(email=f"etn3724{TAG}-bad@gmail.com", role="superuser", brands=["all"])
    with pytest.raises(AdminValidationError):
        svc.invite_user(email=f"etn3724{TAG}-bad@gmail.com", role="viewer", brands=["Humira"])


def _mk(svc, suffix, role="viewer", password=PW):
    """Create + activate a disposable user, return (id, email)."""
    email = f"etn3724{TAG}-{suffix}@gmail.com"
    created = svc.admin_client.auth.admin.create_user(
        {
            "email": email,
            "password": password,
            "email_confirm": True,
            "app_metadata": {"role": role, "brands": ["all"]},
        }
    )
    svc._upsert_profile(created.user.id, email, role)
    return created.user.id, email


def test_update_user_dual_writes_role(svc):
    uid, email = _mk(svc, "upd")
    svc.update_user(uid, role="operator", brands=["Fabhalta"], acting_admin_id="not-the-target")
    u = svc._get_auth_user(uid)
    assert (u.app_metadata or {}).get("role") == "operator"
    assert (u.app_metadata or {}).get("brands") == ["Fabhalta"]
    p = (
        svc.admin_client.table("chatbot_user_profiles")
        .select("role, is_admin")
        .eq("id", uid)
        .execute()
    )
    assert p.data[0]["role"] == "operator" and p.data[0]["is_admin"] is False


def test_disable_sets_flag_and_blocks_signin_enable_reverses(svc):
    from supabase import create_client

    uid, email = _mk(svc, "dis")
    svc.disable_user(uid, acting_admin_id="not-the-target")
    u = svc._get_auth_user(uid)
    assert (u.app_metadata or {}).get("disabled") is True
    anon = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"],
    )
    with pytest.raises(Exception, match="[Bb]anned"):
        anon.auth.sign_in_with_password({"email": email, "password": PW})

    svc.enable_user(uid)
    u = svc._get_auth_user(uid)
    assert not (u.app_metadata or {}).get("disabled")
    signed = anon.auth.sign_in_with_password({"email": email, "password": PW})
    assert signed.session is not None


def test_delete_removes_user_and_cascades_profile(svc):
    uid, _ = _mk(svc, "del")
    svc.delete_user(uid, acting_admin_id="not-the-target")
    from src.services.admin_user_service import AdminNotFoundError

    with pytest.raises(AdminNotFoundError):
        svc._get_auth_user(uid)
    p = svc.admin_client.table("chatbot_user_profiles").select("id").eq("id", uid).execute()
    assert p.data == []  # service deletes the profile explicitly (no FK in prod)


def test_self_targeting_guards(svc):
    from src.services.admin_user_service import AdminGuardError

    uid, _ = _mk(svc, "self", role="admin")
    with pytest.raises(AdminGuardError):
        svc.delete_user(uid, acting_admin_id=uid)
    with pytest.raises(AdminGuardError):
        svc.disable_user(uid, acting_admin_id=uid)


def test_last_admin_guards(svc):
    """With >=2 enabled admins (real etn3724 + a disposable one), deleting the
    disposable admin is allowed — exercising the demote-then-delete path end
    to end. The guard's REFUSAL path would require exactly one enabled admin
    on the live instance (never true here), so it is pinned deterministically
    in tests/unit/test_admin_guards.py instead of being environment-gated."""
    uid, _ = _mk(svc, "lastadm", role="admin")
    svc.delete_user(uid, acting_admin_id="not-the-target")

    admins = svc._enabled_admin_ids()
    assert len(admins) >= 1
    assert uid not in admins


def test_disabled_user_cannot_get_fresh_links(svc):
    """Reinvite/recovery for a disabled user is refused server-side (the UI
    hides the buttons, but a direct API call must not mint a link that lets
    a banned user set a password while disabled)."""
    from src.services.admin_user_service import AdminGuardError

    uid, _ = _mk(svc, "dislink")
    svc.disable_user(uid, acting_admin_id="not-the-target")
    with pytest.raises(AdminGuardError, match="disabled"):
        svc.reinvite_user(uid)
    with pytest.raises(AdminGuardError, match="disabled"):
        svc.recovery_link(uid)


def test_activity_readers_return_real_history(svc):
    # Platform: real auth.audit_log_entries exist since 2026-02 (login events)
    platform = svc.platform_activity(days=365)
    assert platform["days"]  # non-empty
    assert any(d["logins"] > 0 for d in platform["days"])

    # Per-user: the real admin has login history
    me = next(u for u in svc.list_users() if u["email"] == "etn3724@gmail.com")
    activity = svc.user_activity(me["id"], days=365)
    assert any(d["event_type"] == "login" and d["event_count"] > 0 for d in activity["auth_events"])
    assert isinstance(activity["api_activity"], list)
    assert isinstance(activity["recent_events"], list)
    assert activity["chat"]["total_messages"] >= 0


def test_reconcile_role_stores(svc):
    """Users with NULL jwt role but a profile role get app_metadata backfilled."""
    uid, email = _mk(svc, "recon")
    # strip the jwt role to simulate legacy drift
    svc.admin_client.auth.admin.update_user_by_id(
        uid, {"app_metadata": {"role": None, "brands": None}}
    )
    svc.admin_client.table("chatbot_user_profiles").update({"role": "analyst"}).eq(
        "id", uid
    ).execute()

    report = svc.reconcile_role_stores()
    assert any(r["user_id"] == uid and r["action"] == "app_metadata_backfilled" for r in report)
    u = svc._get_auth_user(uid)
    assert (u.app_metadata or {}).get("role") == "analyst"
