"""Unit tests for src/api/utils/agent_import_guard.py.

Covers the env-gated fail-closed decision policy that backs all 5 mock-data
route fallbacks (F-010-backend, issue #429).
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.api.utils.agent_import_guard import (
    guard_or_raise,
    raise_503_for_import_error,
    should_fail_closed_on_import_error,
)


class TestShouldFailClosedOnImportError:
    """Decision matrix for ``should_fail_closed_on_import_error``."""

    def test_explicit_require_truthy_forces_fail_closed(self):
        """E2I_REQUIRE_AGENT_IMPORT=1 always fails closed, even in dev env."""
        with patch.dict(
            "os.environ",
            {"E2I_REQUIRE_AGENT_IMPORT": "1", "ENVIRONMENT": "development"},
            clear=False,
        ):
            assert should_fail_closed_on_import_error() is True

    def test_explicit_require_falsy_forces_mock_allowed(self):
        """E2I_REQUIRE_AGENT_IMPORT=0 always allows mock, even in production."""
        with patch.dict(
            "os.environ",
            {"E2I_REQUIRE_AGENT_IMPORT": "0", "ENVIRONMENT": "production"},
            clear=False,
        ):
            assert should_fail_closed_on_import_error() is False

    def test_production_default_fails_closed(self):
        """Unset E2I_REQUIRE_AGENT_IMPORT + ENVIRONMENT=production → fail closed."""
        env = {"ENVIRONMENT": "production"}
        # Ensure unset
        with patch.dict("os.environ", env, clear=False):
            import os as _os

            _os.environ.pop("E2I_REQUIRE_AGENT_IMPORT", None)
            assert should_fail_closed_on_import_error() is True

    def test_development_default_allows_mock(self):
        """Unset E2I_REQUIRE_AGENT_IMPORT + ENVIRONMENT=development → mock allowed."""
        env = {"ENVIRONMENT": "development"}
        with patch.dict("os.environ", env, clear=False):
            import os as _os

            _os.environ.pop("E2I_REQUIRE_AGENT_IMPORT", None)
            assert should_fail_closed_on_import_error() is False

    def test_unset_environment_fails_closed(self):
        """Codex iter-1 H1: Unset ENVIRONMENT must fail closed.

        Missing deployment metadata MUST NOT silently enable fabricated
        data. A misconfigured production deploy missing
        ``ENVIRONMENT=production`` should not get mock responses.
        """
        with patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop("ENVIRONMENT", None)
            _os.environ.pop("E2I_REQUIRE_AGENT_IMPORT", None)
            assert should_fail_closed_on_import_error() is True

    def test_misspelled_environment_fails_closed(self):
        """Codex iter-1 H1: misspelled ENVIRONMENT fails closed."""
        with patch.dict(
            "os.environ", {"ENVIRONMENT": "prodution"}, clear=False  # typo
        ):
            import os as _os

            _os.environ.pop("E2I_REQUIRE_AGENT_IMPORT", None)
            assert should_fail_closed_on_import_error() is True

    @pytest.mark.parametrize(
        "value", ["development", "dev", "test", "testing", "local"]
    )
    def test_known_dev_environments_allow_mock(self, value):
        """Only explicit known dev/test values allow mock-fallback."""
        with patch.dict("os.environ", {"ENVIRONMENT": value}, clear=False):
            import os as _os

            _os.environ.pop("E2I_REQUIRE_AGENT_IMPORT", None)
            assert should_fail_closed_on_import_error() is False

    @pytest.mark.parametrize("value", ["true", "TRUE", "Yes", "on", "ON"])
    def test_truthy_strings_recognized(self, value):
        """Various truthy spellings parse as fail-closed."""
        with patch.dict("os.environ", {"E2I_REQUIRE_AGENT_IMPORT": value}, clear=False):
            assert should_fail_closed_on_import_error() is True

    @pytest.mark.parametrize("value", ["false", "FALSE", "No", "off", "OFF"])
    def test_falsy_strings_recognized(self, value):
        """Various falsy spellings parse as mock-allowed."""
        with patch.dict(
            "os.environ",
            {"E2I_REQUIRE_AGENT_IMPORT": value, "ENVIRONMENT": "production"},
            clear=False,
        ):
            assert should_fail_closed_on_import_error() is False

    def test_unrecognized_value_falls_through_to_environment(self):
        """Unknown E2I_REQUIRE_AGENT_IMPORT value falls back to ENVIRONMENT default."""
        with patch.dict(
            "os.environ",
            {"E2I_REQUIRE_AGENT_IMPORT": "maybe", "ENVIRONMENT": "production"},
            clear=False,
        ):
            # production default => fail closed
            assert should_fail_closed_on_import_error() is True


class TestRaise503ForImportError:
    """The 503 HTTPException builder used in fail-closed paths."""

    def test_returns_http_exception_with_503(self):
        """Builder returns an HTTPException with status_code=503."""
        exc = raise_503_for_import_error(
            ImportError("boom"),
            agent_name="Gap Analyzer",
        )
        assert isinstance(exc, HTTPException)
        assert exc.status_code == 503

    def test_detail_includes_agent_and_import_error(self):
        """Detail payload surfaces structured diagnostic fields."""
        exc = raise_503_for_import_error(
            ImportError("module 'foo' missing"),
            agent_name="Gap Analyzer",
        )
        assert isinstance(exc.detail, dict)
        assert exc.detail["error"] == "agent_unavailable"
        assert exc.detail["agent"] == "Gap Analyzer"
        assert "module 'foo' missing" in exc.detail["import_error"]


class TestGuardOrRaise:
    """``guard_or_raise`` ties the decision policy to the exception builder."""

    def test_raises_503_when_fail_closed(self):
        """In fail-closed mode the helper raises HTTPException(503)."""
        with patch.dict("os.environ", {"E2I_REQUIRE_AGENT_IMPORT": "1"}, clear=False):
            with pytest.raises(HTTPException) as exc_info:
                guard_or_raise(
                    ImportError("missing module"),
                    agent_name="Gap Analyzer",
                )
            assert exc_info.value.status_code == 503

    def test_returns_silently_when_mock_allowed(self):
        """In mock-allowed mode the helper returns None (caller proceeds)."""
        with patch.dict("os.environ", {"E2I_REQUIRE_AGENT_IMPORT": "0"}, clear=False):
            # Must not raise
            guard_or_raise(
                ImportError("missing module"),
                agent_name="Gap Analyzer",
            )
