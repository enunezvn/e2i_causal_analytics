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


# =============================================================================
# #1560: SIGABRT observability via post_worker_init (BOTH preload modes)
# =============================================================================
#
# Measured constraints that shaped this design (2026-08-13):
# - gunicorn's worker_abort hook NEVER fires under UvicornWorker (its
#   init_signals resets all gunicorn handlers to SIG_DFL), so the mechanism
#   must be armed by us, after that reset.
# - faulthandler.register(SIGABRT) raises RuntimeError (fatal signal);
#   faulthandler.enable() is the supported way and preserves the default
#   die-with-134 disposition after dumping.


def test_post_worker_init_is_defined_and_callable() -> None:
    module = _load_conf()
    assert callable(getattr(module, "post_worker_init", None)), "post_worker_init not callable"


def test_no_worker_abort_hook_defined() -> None:
    """worker_abort is DEAD code under UvicornWorker (init_signals resets
    SIGABRT to SIG_DFL before any murder can land) — defining it would look
    like coverage while providing none."""
    module = _load_conf()
    assert not hasattr(module, "worker_abort")


def test_post_worker_init_enables_faulthandler_even_without_preload() -> None:
    """The arbiter murders workers in both modes; the dump must not be
    preload-gated."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GUNICORN_PRELOAD", None)
        module = _load_conf()
    with mock.patch("faulthandler.enable") as enable:
        module.post_worker_init(mock.MagicMock())
    enable.assert_called_once()
    assert enable.call_args.kwargs.get("all_threads") is True, (
        "must dump ALL threads — the wedged frame can be on any of them"
    )


def test_post_worker_init_logs_armed_marker() -> None:
    """The dispatcher live-verifies arming via docker logs."""
    module = _load_conf()
    worker = mock.MagicMock()
    with mock.patch("faulthandler.enable"):
        module.post_worker_init(worker)
    logged = " ".join(str(c) for c in worker.log.info.call_args_list)
    assert "faulthandler armed" in logged


def test_post_worker_init_survives_enable_failure() -> None:
    """Observability must never break worker boot."""
    module = _load_conf()
    with mock.patch("faulthandler.enable", side_effect=RuntimeError("boom")):
        module.post_worker_init(mock.MagicMock())  # must not raise


# =============================================================================
# #1560: master-side pre-fork warm of heavy lazy leaves
# =============================================================================


def test_when_ready_warms_heavy_leaves_before_freeze_when_preload_on() -> None:
    """econml/causalml (measured absent from the boot import tree) must be
    imported in the master BEFORE gc.freeze so their pages are CoW-shared."""
    module = _load_conf({"GUNICORN_PRELOAD": "true"})
    events: list[tuple[str, str | None]] = []

    # gc.freeze is patched FIRST: mock.patch resolves its target via
    # importlib.import_module, which the second patch replaces.
    with (
        mock.patch("gc.freeze", side_effect=lambda: events.append(("freeze", None))),
        mock.patch(
            "importlib.import_module",
            side_effect=lambda name, *a, **k: events.append(("import", name)),
        ),
    ):
        module.when_ready(mock.MagicMock())

    imported = [name for op, name in events if op == "import"]
    # The SUBMODULES are the real seams: econml/causalml top-level __init__ is
    # near-empty (measured 0.00s) — warming only the package name would look
    # like coverage while warming nothing.
    for expected in (
        "dowhy",
        "econml.dml",
        "econml.dr",
        "econml.inference",
        "econml.metalearners",
        "econml.orf",
        "econml.sklearn_extensions.linear_model",
        "causalml.inference.tree",
        "causalml.inference.meta",
    ):
        assert expected in imported, f"preload warm must import {expected}"
    freeze_idx = events.index(("freeze", None))
    last_import_idx = max(i for i, (op, _) in enumerate(events) if op == "import")
    assert last_import_idx < freeze_idx, "warm imports must complete before gc.freeze"


def test_when_ready_warm_is_fail_open_per_module() -> None:
    """A missing/broken heavy leaf must not block master boot or skip freeze."""
    module = _load_conf({"GUNICORN_PRELOAD": "true"})
    # gc.freeze patched first — see ordering note in the previous test.
    with (
        mock.patch("gc.freeze") as freeze,
        mock.patch("importlib.import_module", side_effect=ImportError("not installed")),
    ):
        module.when_ready(mock.MagicMock())  # must not raise
    freeze.assert_called_once()


def test_when_ready_does_not_warm_when_preload_off() -> None:
    """Dark mode stays EXACTLY as before: no master-side imports at all."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GUNICORN_PRELOAD", None)
        module = _load_conf()
    with mock.patch("importlib.import_module") as imp:
        module.when_ready(mock.MagicMock())
    imp.assert_not_called()


def test_post_fork_logs_forked_warm_marker_when_preload_on() -> None:
    """Dispatcher live-verification marker: every forked worker announces it."""
    module = _load_conf({"GUNICORN_PRELOAD": "true"})
    worker = mock.MagicMock()
    with (
        mock.patch("src.mlops.shap_explainer_realtime.reset_executor"),
        mock.patch("src.api.dependencies.opentelemetry_config.reinitialize_opentelemetry"),
    ):
        module.post_fork(mock.MagicMock(), worker)
    logged = " ".join(str(c) for c in worker.log.info.call_args_list)
    assert "forked warm" in logged


def test_post_fork_logs_no_marker_when_preload_off() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GUNICORN_PRELOAD", None)
        module = _load_conf()
    worker = mock.MagicMock()
    module.post_fork(mock.MagicMock(), worker)
    worker.log.info.assert_not_called()
