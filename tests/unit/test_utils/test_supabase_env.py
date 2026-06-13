"""Tests for Supabase service-key resolution.

Regression for the silent anon-downgrade bug: the drift_monitor and
heterogeneous_optimizer connectors resolved their key as
``SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY`` — skipping
``SUPABASE_SERVICE_KEY``, the name the deployment (docker-compose/.env)
actually provides. With only ``SUPABASE_SERVICE_KEY`` set they fell back to the
**anon** key, which lacks GRANTs on the service_role-only ml_* tables, so the
Celery drift sweep got ``42501 permission denied for table ml_model_registry``,
found zero models, and wrote no monitoring runs (a silent prod no-op).

The canonical resolvers (memory factories, feature_store) already chain
``SERVICE_ROLE_KEY -> SERVICE_KEY -> ...``; these tests lock that contract into a
single shared helper so the outliers cannot drift again.
"""

from __future__ import annotations

import pytest

from src.utils.supabase_env import resolve_supabase_service_key

_KEYS = ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_KEY", "SUPABASE_ANON_KEY")


@pytest.fixture(autouse=True)
def _clear_supabase_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


def test_explicit_key_wins(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "role")
    assert resolve_supabase_service_key("explicit") == "explicit"


def test_service_role_key_preferred_over_service_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "role")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    assert resolve_supabase_service_key() == "role"


def test_service_key_used_when_service_role_absent(monkeypatch):
    # THE BUG: SUPABASE_SERVICE_KEY is what the deployment provides.
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    assert resolve_supabase_service_key() == "service"


def test_service_key_preferred_over_anon(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    assert resolve_supabase_service_key() != "anon"


def test_anon_fallback_only_when_allowed(monkeypatch):
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    assert resolve_supabase_service_key(allow_anon=True) == "anon"


def test_no_anon_leak_when_disallowed(monkeypatch):
    # rag/config must NOT silently use anon: returns None so the caller raises.
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    assert resolve_supabase_service_key(allow_anon=False) is None


def test_service_key_resolves_even_when_anon_disallowed(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service")
    assert resolve_supabase_service_key(allow_anon=False) == "service"


def test_none_when_nothing_set():
    assert resolve_supabase_service_key() is None


def test_empty_string_explicit_is_skipped(monkeypatch):
    # An empty explicit arg should fall through to env resolution, not pin "".
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service")
    assert resolve_supabase_service_key("") == "service"
