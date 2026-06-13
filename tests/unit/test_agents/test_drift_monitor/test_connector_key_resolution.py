"""Wiring regression: the drift connector + factory must resolve the
service-role key from SUPABASE_SERVICE_KEY (the deployment env-var name), not
silently fall back to the anon key.

Root cause of the live ``42501 permission denied for table ml_model_registry``:
the connector resolved ``SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY`` and so
never read ``SUPABASE_SERVICE_KEY`` (the name compose/.env set) — downgrading to
the anon role, which has no GRANTs on the service_role-only ml_* tables.
"""

from __future__ import annotations

import pytest

from src.agents.drift_monitor.connectors.factory import _auto_detect_connector_type
from src.agents.drift_monitor.connectors.supabase_connector import SupabaseDataConnector

_KEYS = ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_KEY", "SUPABASE_ANON_KEY")


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SUPABASE_URL", "http://supabase.local")


def test_connector_uses_service_key_when_only_service_key_set(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-role-jwt")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-jwt")
    connector = SupabaseDataConnector()
    assert connector.supabase_key == "service-role-jwt"


def test_connector_prefers_service_role_name_when_both_set(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "role-jwt")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-jwt")
    connector = SupabaseDataConnector()
    assert connector.supabase_key == "role-jwt"


def test_connector_anon_only_when_no_service_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-jwt")
    connector = SupabaseDataConnector()
    assert connector.supabase_key == "anon-jwt"


def test_explicit_key_still_honored(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-jwt")
    connector = SupabaseDataConnector(supabase_key="explicit-jwt")
    assert connector.supabase_key == "explicit-jwt"


def test_auto_detect_supabase_with_service_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-role-jwt")
    assert _auto_detect_connector_type() == "supabase"


def test_auto_detect_mock_when_no_key(monkeypatch):
    # URL set (fixture) but no key of any kind → mock.
    assert _auto_detect_connector_type() == "mock"
