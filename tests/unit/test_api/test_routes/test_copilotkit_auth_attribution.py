"""_require_auth_for_copilotkit_execution stashes the JWT-verified user id
for the capture hooks' attribution fallback. CopilotKit threadIds are bare
UUIDs (no ``user~`` prefix on any real chat session), so the verified token
identity is the only honest source of per-user chat attribution."""

from types import SimpleNamespace

import pytest

from src.utils.llm_attribution import (
    clear_attribution,
    get_attribution,
    set_authenticated_user,
    set_chat_attribution,
)

VERIFIED = "22222222-2222-2222-2222-222222222222"
BARE_SESSION = "014ad833-54ed-475e-801a-85c24432137b"


@pytest.fixture(autouse=True)
def _reset_context():
    clear_attribution()
    set_authenticated_user(None)
    yield
    clear_attribution()
    set_authenticated_user(None)


def _request_with_bearer(token: str):
    return SimpleNamespace(headers={"Authorization": f"Bearer {token}"}, state=SimpleNamespace())


async def test_auth_stashes_verified_user_for_attribution_fallback(monkeypatch):
    import src.api.routes.copilotkit as ck

    monkeypatch.setattr(ck, "TESTING_MODE", False)

    async def _verify(token):
        assert token == "tok"
        return {"id": VERIFIED, "email": "user@example.com"}

    monkeypatch.setattr(ck, "verify_supabase_token", _verify)

    request = _request_with_bearer("tok")
    user = await ck._require_auth_for_copilotkit_execution(request)
    assert user["id"] == VERIFIED
    assert request.state.user is user

    # Adapter path: bare-UUID CopilotKit threadId -> falls back to verified id
    set_chat_attribution(BARE_SESSION, request_id="run-1")
    assert get_attribution().user_id == VERIFIED


async def test_testing_mode_non_uuid_user_never_attributed(monkeypatch):
    import src.api.routes.copilotkit as ck

    monkeypatch.setattr(ck, "TESTING_MODE", True)

    request = SimpleNamespace(headers={}, state=SimpleNamespace())
    user = await ck._require_auth_for_copilotkit_execution(request)
    assert user["id"] == "test-user-id"

    # non-UUID test identity must not leak into the UUID-typed column
    set_chat_attribution(BARE_SESSION)
    assert get_attribution().user_id is None
