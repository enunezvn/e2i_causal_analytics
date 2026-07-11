"""Disable must lock out EXISTING tokens immediately (spec verified fact 6).

DISPROVED by live experiment 2026-07-11: a GoTrue ban does NOT invalidate an
existing access token (get_user still succeeds; gotrue-py's User model has no
banned_until). Ban alone leaves a <=1h window. The fix: disable sets
app_metadata.disabled=true and verify_supabase_token fails closed on it —
get_user returns FRESH app_metadata per request, so lockout is immediate.

    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_auth_disabled_flag_realdb.py -p no:cacheprovider -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

EMAIL = "etn3724+admtest-disabled@gmail.com"
PASSWORD = "AdmTest#2026-disabled"


@pytest.fixture()
def disposable_user():
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    admin = create_client(
        url,
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_SERVICE_KEY"],
    )
    anon = create_client(url, os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"])
    for u in admin.auth.admin.list_users():
        if u.email == EMAIL:
            admin.auth.admin.delete_user(u.id)
    created = admin.auth.admin.create_user(
        {"email": EMAIL, "password": PASSWORD, "email_confirm": True}
    )
    session = anon.auth.sign_in_with_password({"email": EMAIL, "password": PASSWORD})
    yield admin, created.user.id, session.session.access_token
    admin.auth.admin.delete_user(created.user.id)
    # the signup trigger creates a profile row and prod has no FK — clean it
    admin.table("chatbot_user_profiles").delete().eq("id", created.user.id).execute()


@pytest.mark.asyncio
async def test_disabled_flag_rejects_live_token(disposable_user):
    from src.api.dependencies.auth import verify_supabase_token

    admin, user_id, token = disposable_user

    # Token is valid before the flag
    user = await verify_supabase_token(token)
    assert user is not None and user["email"] == EMAIL

    # Set the disabled flag (what /disable will do) — same LIVE token must now fail
    admin.auth.admin.update_user_by_id(user_id, {"app_metadata": {"disabled": True}})
    assert await verify_supabase_token(token) is None, (
        "disabled user's existing token must be rejected immediately"
    )

    # Clearing the flag restores access (what /enable will do)
    admin.auth.admin.update_user_by_id(user_id, {"app_metadata": {"disabled": False}})
    user = await verify_supabase_token(token)
    assert user is not None
