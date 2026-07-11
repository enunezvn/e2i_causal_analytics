"""Deterministic unit coverage for AdminUserService safety guards.

Why stubs here (exception to the real-DB convention): the refusal paths need
the instance to hold exactly one (or zero) enabled admins, which is
unreachable against the live DB without demoting/deleting the REAL admin
account. Happy paths and everything else run against the real DB in
tests/integration/test_admin_user_service_realdb.py; these tests pin the
guard logic itself:

- _guard_not_last_admin refuses when the target is the only enabled admin
  (codex finding: the realdb variant's raise branch was environment-gated
  and never executed).
- post-write re-verification closes the guard's TOCTOU window: if a
  concurrent request leaves zero enabled admins, the write is compensated
  (role restored / re-enabled / delete aborted) and AdminGuardError raised.
- _list_all_auth_users paginates past the 1000-user page cap so the guard
  and invite dedup never undercount.
"""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.services.admin_user_service import (
    AdminGuardError,
    AdminUserService,
)


def _user(uid: str, role: str = "viewer", disabled: bool = False, email: str | None = None):
    return SimpleNamespace(
        id=uid,
        email=email or f"{uid}@x.com",
        app_metadata={"role": role, "disabled": disabled, "brands": ["all"]},
        user_metadata={},
        last_sign_in_at="2026-07-01T00:00:00Z",
        created_at="2026-06-01T00:00:00Z",
    )


class _FakeTableQuery:
    """Chainable no-op stand-in for postgrest table queries."""

    def __getattr__(self, _name):
        def _chain(*_a, **_k):
            return self

        return _chain

    def execute(self):
        return SimpleNamespace(data=[])


class _FakeAuthAdmin:
    """Scripted GoTrue admin API: list_users pops pre-scripted responses in
    call order (so a test can present a different world before and after a
    write, simulating a concurrent admin action)."""

    def __init__(self, list_users_script: List[List[Any]], users_by_id: Dict[str, Any]):
        self._script = list(list_users_script)
        self._users_by_id = users_by_id
        self.update_calls: List[tuple] = []
        self.delete_calls: List[str] = []

    def list_users(self, page: int = 1, per_page: int = 1000):
        if not self._script:
            return []
        return self._script.pop(0)

    def get_user_by_id(self, uid: str):
        u = self._users_by_id.get(uid)
        if u is None:
            raise Exception(f"user {uid} not found")
        return SimpleNamespace(user=u)

    def update_user_by_id(self, uid: str, attrs: Dict[str, Any]):
        self.update_calls.append((uid, attrs))

    def delete_user(self, uid: str):
        self.delete_calls.append(uid)


def _service(list_users_script: List[List[Any]], users_by_id: Dict[str, Any]) -> AdminUserService:
    svc = AdminUserService.__new__(AdminUserService)
    fake_auth = _FakeAuthAdmin(list_users_script, users_by_id)
    svc.admin_client = SimpleNamespace(
        auth=SimpleNamespace(admin=fake_auth),
        table=lambda *_a, **_k: _FakeTableQuery(),
    )
    svc.public_app_url = "https://example.test"
    return svc


# --------------------------------------------------------------- guard raise


def test_guard_refuses_when_target_is_last_enabled_admin():
    only_admin = _user("a1", role="admin")
    svc = _service([[only_admin]], {"a1": only_admin})
    with pytest.raises(AdminGuardError, match="last enabled admin"):
        svc._guard_not_last_admin("a1", "demote")


def test_guard_passes_with_two_enabled_admins():
    admins = [_user("a1", role="admin"), _user("a2", role="admin")]
    svc = _service([admins], {u.id: u for u in admins})
    svc._guard_not_last_admin("a1", "demote")  # no raise


def test_disabled_admin_does_not_count_toward_quorum():
    target = _user("a1", role="admin")
    benched = _user("a2", role="admin", disabled=True)
    svc = _service([[target, benched]], {"a1": target, "a2": benched})
    with pytest.raises(AdminGuardError, match="last enabled admin"):
        svc._guard_not_last_admin("a1", "disable")


# ------------------------------------------------- TOCTOU: demote compensated


def test_concurrent_demote_race_is_reverted():
    """Both of the last two admins demoted concurrently: the pre-write guard
    passes (2 admins visible), but the post-write re-count sees zero — the
    write must be reverted and refused."""
    a1, a2 = _user("a1", role="admin"), _user("a2", role="admin")
    script = [
        [a1, a2],  # pre-write guard: two admins, demote allowed
        [],  # post-write re-count: other admin demoted concurrently
    ]
    svc = _service(script, {"a1": a1, "a2": a2})
    with pytest.raises(AdminGuardError, match="revert"):
        svc.update_user("a1", acting_admin_id="a2", role="viewer")
    fake = svc.admin_client.auth.admin
    # last update restores the original admin app_metadata
    restored = fake.update_calls[-1]
    assert restored[0] == "a1"
    assert restored[1]["app_metadata"]["role"] == "admin"


def test_demote_with_surviving_admin_commits():
    a1, a2 = _user("a1", role="admin"), _user("a2", role="admin")
    script = [
        [a1, a2],  # pre-write guard
        [a2],  # post-write: a2 still enabled admin
    ]
    svc = _service(script, {"a1": a1, "a2": a2})
    result = svc.update_user("a1", acting_admin_id="a2", role="viewer")
    assert result["role"] == "viewer"
    fake = svc.admin_client.auth.admin
    assert all(c[1].get("app_metadata", {}).get("role") != "admin" for c in fake.update_calls)


# ------------------------------------------------ TOCTOU: disable compensated


def test_concurrent_disable_race_is_reverted():
    a1, a2 = _user("a1", role="admin"), _user("a2", role="admin")
    script = [
        [a1, a2],  # pre-write guard
        [],  # post-write: other admin disabled concurrently
    ]
    svc = _service(script, {"a1": a1, "a2": a2})
    with pytest.raises(AdminGuardError, match="revert"):
        svc.disable_user("a1", acting_admin_id="a2")
    fake = svc.admin_client.auth.admin
    restored = fake.update_calls[-1]
    assert restored[0] == "a1"
    assert restored[1]["app_metadata"]["disabled"] is False
    assert restored[1]["ban_duration"] == "none"


# ------------------------------------------- TOCTOU: delete = demote-then-del


def test_concurrent_delete_race_aborts_before_deletion():
    """Deletion is not compensable, so an admin target is demoted first; if
    the post-demote re-count finds no admins, the role is restored and the
    delete never happens."""
    a1, a2 = _user("a1", role="admin"), _user("a2", role="admin")
    script = [
        [a1, a2],  # pre-write guard
        [],  # post-demote re-count: zero admins left
    ]
    svc = _service(script, {"a1": a1, "a2": a2})
    with pytest.raises(AdminGuardError, match="revert"):
        svc.delete_user("a1", acting_admin_id="a2")
    fake = svc.admin_client.auth.admin
    assert fake.delete_calls == []  # delete aborted
    restored = fake.update_calls[-1]
    assert restored[1]["app_metadata"]["role"] == "admin"


def test_delete_nonadmin_needs_no_demote_step():
    v1 = _user("v1", role="viewer")
    a1 = _user("a1", role="admin")
    svc = _service([[a1]], {"v1": v1, "a1": a1})
    result = svc.delete_user("v1", acting_admin_id="a1")
    assert result["deleted"] is True
    fake = svc.admin_client.auth.admin
    assert fake.delete_calls == ["v1"]
    assert fake.update_calls == []  # no demote round-trip for non-admins


# -------------------------------------------------- disabled-user link guards


def test_reinvite_refused_for_disabled_user():
    d1 = _user("d1", role="viewer", disabled=True)
    svc = _service([], {"d1": d1})
    with pytest.raises(AdminGuardError, match="disabled"):
        svc.reinvite_user("d1")


def test_recovery_link_refused_for_disabled_user():
    d1 = _user("d1", role="viewer", disabled=True)
    svc = _service([], {"d1": d1})
    with pytest.raises(AdminGuardError, match="disabled"):
        svc.recovery_link("d1")


# ------------------------------------------------------------- pagination


def test_list_all_auth_users_walks_past_page_cap():
    page1 = [_user(f"u{i}") for i in range(1000)]
    page2 = [_user("u1000"), _user("u1001")]
    svc = _service([page1, page2], {})
    users = svc._list_all_auth_users()
    assert len(users) == 1002


def test_list_all_auth_users_single_short_page():
    page1 = [_user("u0"), _user("u1")]
    svc = _service([page1], {})
    assert len(svc._list_all_auth_users()) == 2
