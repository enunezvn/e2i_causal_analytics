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
* Codex iter-1 H3: HierarchicalAnalyzerNode silent mock data path also fails closed.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.agents.heterogeneous_optimizer.connectors import MockDataConnector
from src.agents.heterogeneous_optimizer.nodes.cate_estimator import (
    _get_default_data_connector,
    _mock_connector_allowed,
)
from src.agents.heterogeneous_optimizer.nodes.hierarchical_analyzer import (
    HierarchicalAnalyzerNode,
)
from src.agents.heterogeneous_optimizer.nodes.uplift_analyzer import (
    UpliftAnalyzerNode,
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

    @pytest.mark.parametrize("value", ["development", "dev", "test", "testing", "local"])
    def test_full_allowlist_matrix(self, value):
        """Codex iter-2 M2: pin full _KNOWN_DEV_ENVIRONMENTS matrix to
        prevent silent drift in this site relative to agent_import_guard.
        """
        with patch.dict("os.environ", {"ENVIRONMENT": value}, clear=False):
            import os as _os

            _os.environ.pop("E2I_ALLOW_MOCK_CONNECTOR", None)
            assert _mock_connector_allowed() is True

    def test_unset_environment_denies(self):
        """Codex iter-1 H1: unset ENVIRONMENT must NOT enable mock fallback."""
        with patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop("ENVIRONMENT", None)
            _os.environ.pop("E2I_ALLOW_MOCK_CONNECTOR", None)
            assert _mock_connector_allowed() is False

    def test_misspelled_environment_denies(self):
        """Misspelled ENVIRONMENT (not in dev allowlist) → mock forbidden."""
        with patch.dict("os.environ", {"ENVIRONMENT": "dvelopment"}, clear=False):  # typo
            import os as _os

            _os.environ.pop("E2I_ALLOW_MOCK_CONNECTOR", None)
            assert _mock_connector_allowed() is False


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


class TestHierarchicalAnalyzerFailClosed:
    """Codex iter-1 H3: HierarchicalAnalyzerNode also previously fell
    silently through to ``_generate_mock_data`` whenever no
    ``data_connector`` was attached and no ``tier0_data`` was in state.
    These tests pin the new env-gated fail-closed contract.
    """

    def _state(self):
        return {
            "treatment_var": "T",
            "outcome_var": "Y",
            "effect_modifiers": ["x1"],
            "segment_vars": ["region"],
            "data_source": "synthetic",
            "filters": None,
            "tier0_data": None,
        }

    @pytest.mark.asyncio
    async def test_raises_when_no_connector_and_mock_forbidden(self, monkeypatch):
        """No connector + no tier0_data + mock forbidden → fail closed.

        The #30 self-heal first tries to resolve a real connector (like the CATE
        estimator); when none is available it raises rather than fabricating.
        We force the no-real-connector condition so the test is deterministic
        regardless of CI Supabase creds.
        """
        import src.agents.heterogeneous_optimizer.nodes.cate_estimator as ce

        def _raise():
            raise RuntimeError(
                "Failed to initialize Supabase data connector; mock fallback is disabled"
            )

        monkeypatch.setattr(ce, "_get_default_data_connector", _raise)
        node = HierarchicalAnalyzerNode()
        # explicitly forbid mock so the self-heal must resolve a REAL connector
        with patch.dict(
            "os.environ",
            {"E2I_ALLOW_MOCK_CONNECTOR": "0"},
            clear=False,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await node._get_data(self._state())
            assert "fallback is disabled" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_returns_mock_data_when_explicitly_allowed(self):
        """Mock-allowed env returns mock DataFrame."""
        node = HierarchicalAnalyzerNode()
        with patch.dict(
            "os.environ",
            {"E2I_ALLOW_MOCK_CONNECTOR": "1"},
            clear=False,
        ):
            df = await node._get_data(self._state())
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.asyncio
    async def test_raises_when_environment_unset(self, monkeypatch):
        """Codex iter-1 H1/H3: unset ENVIRONMENT must fail closed here too — the
        #30 self-heal resolves a real connector or raises; it never fabricates."""
        import src.agents.heterogeneous_optimizer.nodes.cate_estimator as ce

        def _raise():
            raise RuntimeError(
                "Failed to initialize Supabase data connector; mock fallback is disabled"
            )

        monkeypatch.setattr(ce, "_get_default_data_connector", _raise)
        node = HierarchicalAnalyzerNode()
        with patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop("ENVIRONMENT", None)
            _os.environ.pop("E2I_ALLOW_MOCK_CONNECTOR", None)
            with pytest.raises(RuntimeError):
                await node._get_data(self._state())


class TestUpliftAnalyzerFailClosed:
    """Codex iter-2 H1: UpliftAnalyzerNode previously had the SAME silent
    `_generate_mock_data` fallback as hierarchical_analyzer. The node is
    exported in __init__.py and is reachable if wired into the graph or
    invoked directly. Same env-gate applied.
    """

    def _state(self):
        return {
            "treatment_var": "T",
            "outcome_var": "Y",
            "effect_modifiers": ["x1"],
            "segment_vars": ["region"],
            "data_source": "synthetic",
            "filters": None,
        }

    @pytest.mark.asyncio
    async def test_raises_when_no_connector_and_mock_forbidden(self):
        node = UpliftAnalyzerNode()
        with patch.dict(
            "os.environ",
            {"E2I_ALLOW_MOCK_CONNECTOR": "0"},
            clear=False,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await node._get_data(self._state())
            assert "synthetic-data fallback is disabled" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_returns_mock_data_when_explicitly_allowed(self):
        node = UpliftAnalyzerNode()
        with patch.dict(
            "os.environ",
            {"E2I_ALLOW_MOCK_CONNECTOR": "1"},
            clear=False,
        ):
            df = await node._get_data(self._state())
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.asyncio
    async def test_raises_when_environment_unset(self):
        node = UpliftAnalyzerNode()
        with patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop("ENVIRONMENT", None)
            _os.environ.pop("E2I_ALLOW_MOCK_CONNECTOR", None)
            with pytest.raises(RuntimeError):
                await node._get_data(self._state())
