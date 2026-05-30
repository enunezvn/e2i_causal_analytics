"""Tests for the gunicorn server config (Priority 3: --preload + gc.freeze).

The config ships DARK: ``preload_app`` is False unless ``GUNICORN_PRELOAD`` is
set to a truthy value, so the runtime behaves exactly as today by default.

Covers:
- config/gunicorn.conf.py exists and imports cleanly
- when_ready / post_fork hooks are defined callables
- preload_app is env-gated (dark by default, on when GUNICORN_PRELOAD truthy)
- when_ready / post_fork are inert when preload is off
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GUNICORN_CONF = REPO_ROOT / "config" / "gunicorn.conf.py"


def _load_conf(env: dict[str, str] | None = None) -> ModuleType:
    """Load config/gunicorn.conf.py as a fresh module under the given env.

    The module is loaded fresh each call so that the import-time evaluation of
    ``preload_app`` (which reads GUNICORN_PRELOAD) reflects the patched env.
    """
    env = env or {}
    spec = importlib.util.spec_from_file_location("_gunicorn_conf_under_test", GUNICORN_CONF)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, env, clear=False):
        spec.loader.exec_module(module)
    return module


def test_gunicorn_conf_file_exists() -> None:
    assert GUNICORN_CONF.is_file(), f"missing gunicorn config at {GUNICORN_CONF}"


def test_gunicorn_conf_imports_cleanly() -> None:
    # Default env (GUNICORN_PRELOAD unset) must import without raising.
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GUNICORN_PRELOAD", None)
        module = _load_conf()
    assert module is not None


def test_hooks_are_callables() -> None:
    module = _load_conf()
    assert callable(getattr(module, "when_ready", None)), "when_ready not callable"
    assert callable(getattr(module, "post_fork", None)), "post_fork not callable"


def test_preload_dark_by_default() -> None:
    """GUNICORN_PRELOAD unset => preload_app is False (ships dark)."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GUNICORN_PRELOAD", None)
        module = _load_conf()
    assert module.preload_app is False


@pytest.mark.parametrize("truthy", ["1", "true", "True", "TRUE", "yes", "YES"])
def test_preload_enabled_when_flag_truthy(truthy: str) -> None:
    module = _load_conf({"GUNICORN_PRELOAD": truthy})
    assert module.preload_app is True


@pytest.mark.parametrize("falsy", ["0", "false", "no", "", "off", "anything"])
def test_preload_disabled_when_flag_falsy(falsy: str) -> None:
    module = _load_conf({"GUNICORN_PRELOAD": falsy})
    assert module.preload_app is False


def test_when_ready_inert_when_preload_off() -> None:
    """when_ready must not call gc.freeze when preload is off."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GUNICORN_PRELOAD", None)
        module = _load_conf()
    with mock.patch("gc.freeze") as freeze:
        module.when_ready(mock.MagicMock())
    freeze.assert_not_called()


def test_when_ready_freezes_when_preload_on() -> None:
    module = _load_conf({"GUNICORN_PRELOAD": "true"})
    with mock.patch("gc.freeze") as freeze:
        module.when_ready(mock.MagicMock())
    freeze.assert_called_once()


def test_post_fork_inert_when_preload_off() -> None:
    """post_fork must not re-init singletons when preload is off."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GUNICORN_PRELOAD", None)
        module = _load_conf()
    with mock.patch("src.mlops.shap_explainer_realtime.reset_executor") as reset_exec:
        module.post_fork(mock.MagicMock(), mock.MagicMock())
    reset_exec.assert_not_called()


def test_post_fork_resets_shap_executor_when_preload_on() -> None:
    module = _load_conf({"GUNICORN_PRELOAD": "true"})
    with mock.patch("src.mlops.shap_explainer_realtime.reset_executor") as reset_exec:
        module.post_fork(mock.MagicMock(), mock.MagicMock())
    reset_exec.assert_called_once()


def test_does_not_set_workers() -> None:
    """The config must NOT define `workers` (CLI --workers 2 is source of truth)."""
    module = _load_conf({"GUNICORN_PRELOAD": "true"})
    assert not hasattr(module, "workers"), (
        "gunicorn.conf.py must not set `workers`; the --workers CLI flag owns it"
    )
