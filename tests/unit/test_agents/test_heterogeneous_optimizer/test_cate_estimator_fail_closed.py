"""Tests for F-013 (#431): cate_estimator fail-closed on MockDataConnector.

Previously ``_get_default_data_connector`` silently returned
``MockDataConnector`` whenever Supabase env vars were missing OR connector
init failed. Downstream consumers had no signal that they were receiving
synthetic ``np.random.seed(42)`` data.

These tests pin the new policy:

* Supabase env unset + ``E2I_ALLOW_MOCK_CONNECTOR`` unset + ENVIRONMENT=production
  → RuntimeError raised (NOT silent mock).
* Supabase env unset + ``E2I_ALLOW_MOCK_CONNECTOR=1`` → mock returned (dev mode).
* Supabase env unset + ENVIRONMENT=development → mock returned (default dev).
* Supabase env unset + ``E2I_ALLOW_MOCK_CONNECTOR=0`` → RuntimeError even in dev.
"""

from unittest.mock import patch

import pytest

from src.agents.heterogeneous_optimizer.connectors import MockDataConnector
from src.agents.heterogeneous_optimizer.nodes.cate_estimator import (
    _get_default_data_connector,
    _mock_connector_allowed,
)


class TestMockConnectorPolicy:
    """The env-gate decision matrix."""

    def test_explicit_allow_yields_true(self):
        with patch.dict("os.environ", {"E2I_ALLOW_MOCK_CONNECTOR": "1"}, clear=False):
            assert _mock_connector_allowed() is True

    def test_explicit_deny_yields_false(self):
        with patch.dict(
            "os.environ",
            {
                "E2I_ALLOW_MOCK_CONNECTOR": "0",
                "ENVIRONMENT": "development",
            },
            clear=False,
        ):
            assert _mock_connector_allowed() is False

    def test_production_default_denies(self):
        env = {"ENVIRONMENT": "production"}
        with patch.dict("os.environ", env, clear=False):
            import os as _os

            _os.environ.pop("E2I_ALLOW_MOCK_CONNECTOR", None)
            assert _mock_connector_allowed() is False

    def test_development_default_allows(self):
        env = {"ENVIRONMENT": "development"}
        with patch.dict("os.environ", env, clear=False):
            import os as _os

            _os.environ.pop("E2I_ALLOW_MOCK_CONNECTOR", None)
            assert _mock_connector_allowed() is True

    def test_unset_environment_allows(self):
        with patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop("ENVIRONMENT", None)
            _os.environ.pop("E2I_ALLOW_MOCK_CONNECTOR", None)
            assert _mock_connector_allowed() is True


class TestGetDefaultDataConnectorFailClosed:
    @staticmethod
    def _clear_supabase_env():
        import os as _os

        for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY"):
            _os.environ.pop(k, None)

    def test_raises_when_supabase_missing_and_mock_forbidden(self):
        """Production-style env: no Supabase creds, no mock opt-in → RuntimeError."""
        with patch.dict(
            "os.environ",
            {
                "ENVIRONMENT": "production",
            },
            clear=False,
        ):
            self._clear_supabase_env()
            import os as _os

            _os.environ.pop("E2I_ALLOW_MOCK_CONNECTOR", None)

            with pytest.raises(RuntimeError) as exc_info:
                _get_default_data_connector()
            msg = str(exc_info.value)
            assert "MockDataConnector fallback is disabled" in msg
            assert "SUPABASE" in msg

    def test_raises_when_explicit_deny_overrides_dev_environment(self):
        """Explicit E2I_ALLOW_MOCK_CONNECTOR=0 wins even in dev env."""
        with patch.dict(
            "os.environ",
            {
                "ENVIRONMENT": "development",
                "E2I_ALLOW_MOCK_CONNECTOR": "0",
            },
            clear=False,
        ):
            self._clear_supabase_env()
            with pytest.raises(RuntimeError):
                _get_default_data_connector()

    def test_returns_mock_when_explicitly_allowed(self):
        """E2I_ALLOW_MOCK_CONNECTOR=1 → MockDataConnector returned."""
        with patch.dict(
            "os.environ",
            {
                "E2I_ALLOW_MOCK_CONNECTOR": "1",
            },
            clear=False,
        ):
            self._clear_supabase_env()
            connector = _get_default_data_connector()
            assert isinstance(connector, MockDataConnector)

    def test_returns_mock_in_development_default(self):
        """ENVIRONMENT=development (default mock-allowed) → MockDataConnector."""
        with patch.dict(
            "os.environ",
            {"ENVIRONMENT": "development"},
            clear=False,
        ):
            self._clear_supabase_env()
            import os as _os

            _os.environ.pop("E2I_ALLOW_MOCK_CONNECTOR", None)
            connector = _get_default_data_connector()
            assert isinstance(connector, MockDataConnector)
