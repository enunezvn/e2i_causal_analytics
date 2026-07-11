"""AdminUserService against the REAL local Supabase stack (no mocks).

Disposable users use the +admsvc email tag; every test cleans up after itself.

    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_admin_user_service_realdb.py -p no:cacheprovider -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

TAG = "+admsvc"


@pytest.fixture()
def svc():
    from src.services.admin_user_service import AdminUserService

    service = AdminUserService()
    yield service
    # cleanup ALL disposable users this file may have created
    for u in service.admin_client.auth.admin.list_users():
        if u.email and TAG in u.email:
            service.admin_client.auth.admin.delete_user(u.id)


def test_list_users_merges_auth_and_profile(svc):
    users = svc.list_users()
    assert len(users) >= 8  # the 8 real users
    me = next(u for u in users if u["email"] == "etn3724@gmail.com")
    assert me["role"] == "admin"
    assert me["status"] == "active"
    assert me["last_sign_in_at"] is not None
    # profile join fields present (backfilled by migration 100)
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
    anon.auth.update_user({"password": "AdmSvc#2026-otp"})
    anon.auth.sign_out()
    fresh = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"],
    )
    signed = fresh.auth.sign_in_with_password({"email": email, "password": "AdmSvc#2026-otp"})
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
    anon.auth.update_user({"password": "AdmSvc#2026-re"})

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
