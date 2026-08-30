"""#1848 guard: a worker that dies before any test runs must not exit green.

The failure this guards against
-------------------------------
Observed once on the droplet under memory pressure (PR #1847's lane): an xdist
run with the Unit Tests lane's exact shape (``-n 2 --dist=loadscope
--timeout=30``) lost ``gw0`` before collection printed anything and finished
with **rc=0 and ``6 warnings in 13.48s``** -- no test executed, no failure.

Reproduced deterministically here (``xdist_probe_1848/``) and explained by
xdist 3.8.0's controller, ``xdist/dsession.py``:

* ``worker_errordown`` asks the scheduler for the item the dead worker was
  running. A worker that dies **during collection** has none, so no failed
  ``TestReport`` is synthesised and ``session.testsfailed`` stays 0. That
  synthesised report is the *only* thing that makes a run-phase crash red.
* With ``--max-worker-restart=0`` (#1648) the crash triggers shutdown, and the
  surviving worker's ``collectionfinish`` is dropped on the floor when it
  arrives afterwards (``if self.shuttingdown: return``). But if the peer's
  collection arrived *first*, ``session.testscollected`` is already non-zero.
* pytest's ``_main`` then sees ``testsfailed == 0`` and ``testscollected > 0``
  and returns ``None`` -> exit 0. Crash-first ordering gives ``testscollected
  == 0`` -> exit 5 ("no tests collected"), which is red but wrong.

Measured on HEAD with the probe in this directory (controller-side counters
printed from ``pytest_sessionfinish``):

==============================  ==  ==========  =========  ===========
crash                           rc  collected   finished   summary
==============================  ==  ==========  =========  ===========
none (control)                   0  2           2          2 passed
collection, peer already in      0  2           0          4 warnings
collection, before the peer      5  0           0          4 warnings
run phase (inside a test)        1  2           1          1 failed, 1 passed
==============================  ==  ==========  =========  ===========

Why #1648's guard does not cover it
-----------------------------------
``test_xdist_worker_crash_fast_fail.py`` proves that a worker dying **while
running a test** fails the session promptly and names the test. Its mechanism
is ``--max-worker-restart=0`` plus xdist's synthesised failure for the running
item -- the fourth row above. Rows two and three have no running item, so that
guard has nothing to act on; ``--max-worker-restart=0`` still ends the session
at once, but ending it is exactly what produces the green.

What the guard does
-------------------
``tests/xdist_crash_guard.py`` is registered on the xdist controller by
``tests/conftest.py``. It records every ``pytest_testnodedown`` that carries an
error and every ``pytest_runtest_logfinish``. At ``pytest_sessionfinish``, if a
worker crashed, the session would otherwise exit 0 or 5, and fewer items
reported than were collected, it prints a :data:`BANNER` block and forces the
exit status to :data:`GUARD_EXIT_CODE`. Conditioning on the *crash* is what
keeps every healthy run untouched: ``--collect-only`` (nothing reported, no
crash), ``-k`` deselecting everything (rc=5, no crash), ``-n 0`` / ``-p
no:xdist`` (no workers; the guard is not even installed). Counting finished
items against ``testscollected`` keeps a crash *after* everything reported
silent, so the guard is never redder than HEAD unless results were lost.
"""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit.test_tests_meta.test_session_stall_watchdog_1655 import (
    _as_text,
    _run_guarded,
)
from tests.unit.test_tests_meta.xdist_probe_1848 import (
    ENV_CRASH,
    ENV_SENTINEL,
    MODE_COLLECTION,
)
from tests.xdist_crash_guard import (
    BANNER,
    GUARD_EXIT_CODE,
    PLUGIN_NAME,
    XdistCrashGuard,
    install,
)

# tests/unit/test_tests_meta/<this file> -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = "tests/unit/test_tests_meta/xdist_probe_1848"

#: Wall clock for one nested session. Measured end-to-end on the droplet:
#: 1.4s crashing / ~3s healthy with plugin autoload disabled (18s each with
#: the full plugin set, whose import weight in three extra interpreters is
#: also why autoload is off -- see ``_nested_env``).
NESTED_BUDGET_SECONDS = 150

_ANY_PASSED = re.compile(r"\b\d+ passed\b")


# =============================================================================
# In-process: the decision, without spawning anything
# =============================================================================


def _node(gateway_id: str) -> SimpleNamespace:
    return SimpleNamespace(gateway=SimpleNamespace(id=gateway_id))


def _guard_with(
    *,
    crashed: list[str] = (),
    finished: list[str] = (),  # type: ignore[assignment]
) -> XdistCrashGuard:
    guard = XdistCrashGuard()
    for gateway_id in crashed:
        guard.pytest_testnodedown(node=_node(gateway_id), error="Not properly terminated")
    for nodeid in finished:
        guard.pytest_runtest_logfinish(nodeid=nodeid, location=("f.py", 1, "f"))
    return guard


def test_fires_when_a_worker_crashed_and_collected_tests_never_reported() -> None:
    """The observed shape: peer collected 2, gw0 died, nothing ran, exit 0."""
    guard = _guard_with(crashed=["gw0"])
    report = guard.assess(collected=2, exitstatus=pytest.ExitCode.OK)
    assert report is not None
    assert BANNER in report
    assert "gw0" in report
    assert "0 of 2" in report, report


def test_fires_when_the_crash_preceded_every_collection() -> None:
    """Crash-first ordering: rc=5 on HEAD, which misreports 'no tests collected'."""
    guard = _guard_with(crashed=["gw0"])
    report = guard.assess(collected=0, exitstatus=pytest.ExitCode.NO_TESTS_COLLECTED)
    assert report is not None
    assert BANNER in report


def test_silent_when_the_crash_came_after_every_test_reported() -> None:
    """A worker dying after all results are in loses nothing; stay as green as HEAD."""
    guard = _guard_with(crashed=["gw0"], finished=["t.py::a", "t.py::b"])
    assert guard.assess(collected=2, exitstatus=pytest.ExitCode.OK) is None


def test_silent_when_the_session_already_failed() -> None:
    """A run-phase crash is already red via xdist's synthesised failure (#1648)."""
    guard = _guard_with(crashed=["gw0"], finished=["t.py::a"])
    assert guard.assess(collected=2, exitstatus=pytest.ExitCode.TESTS_FAILED) is None


def test_silent_without_a_crash_even_when_nothing_reported() -> None:
    """``--collect-only`` under ``-n 2``: collected N, reported 0, no crash."""
    guard = _guard_with()
    assert guard.assess(collected=2, exitstatus=pytest.ExitCode.OK) is None
    # ``-k`` deselecting everything: pytest's own rc=5 stands, untouched.
    assert guard.assess(collected=0, exitstatus=pytest.ExitCode.NO_TESTS_COLLECTED) is None


def test_a_clean_worker_exit_is_not_a_crash() -> None:
    """xdist fires ``pytest_testnodedown(error=None)`` for every normal worker exit."""
    guard = XdistCrashGuard()
    guard.pytest_testnodedown(node=_node("gw0"), error=None)
    guard.pytest_testnodedown(node=_node("gw1"), error=None)
    assert guard.assess(collected=2, exitstatus=pytest.ExitCode.OK) is None


def _fake_session(*, collected: int) -> tuple[SimpleNamespace, io.StringIO]:
    """A session whose terminal writer is captured, so the print path is checked too."""
    terminal = io.StringIO()
    config = SimpleNamespace(get_terminal_writer=lambda: terminal)
    return (
        SimpleNamespace(testscollected=collected, exitstatus=pytest.ExitCode.OK, config=config),
        terminal,
    )


def test_sessionfinish_forces_the_exit_status_and_prints_the_report() -> None:
    guard = _guard_with(crashed=["gw0"])
    session, terminal = _fake_session(collected=2)
    guard.pytest_sessionfinish(session=session, exitstatus=pytest.ExitCode.OK)
    assert session.exitstatus == GUARD_EXIT_CODE
    assert guard.report is not None and BANNER in guard.report
    assert terminal.getvalue() == guard.report


def test_sessionfinish_leaves_a_healthy_session_alone() -> None:
    guard = _guard_with(finished=["t.py::a", "t.py::b"])
    session, terminal = _fake_session(collected=2)
    guard.pytest_sessionfinish(session=session, exitstatus=pytest.ExitCode.OK)
    assert session.exitstatus == pytest.ExitCode.OK
    assert guard.report is None
    assert terminal.getvalue() == ""


class _FakePluginManager:
    """Mimics the two surfaces ``install`` touches: the hook relay (which only
    has a ``pytest_testnodedown`` attribute once xdist's hookspecs are added --
    however the plugin was loaded, entry point or ``PYTEST_PLUGINS``) and
    ``register``."""

    def __init__(self, *, xdist: bool) -> None:
        self.hook = SimpleNamespace(pytest_testnodedown=object()) if xdist else SimpleNamespace()
        self.registered: dict[str, object] = {}

    def register(self, plugin: object, name: str) -> None:
        self.registered[name] = plugin


def test_install_arms_the_controller_only() -> None:
    controller = SimpleNamespace(pluginmanager=_FakePluginManager(xdist=True))
    guard = install(controller)  # type: ignore[arg-type]
    assert isinstance(guard, XdistCrashGuard)
    assert controller.pluginmanager.registered == {PLUGIN_NAME: guard}

    worker = SimpleNamespace(pluginmanager=_FakePluginManager(xdist=True), workerinput={})
    assert install(worker) is None  # type: ignore[arg-type]
    assert worker.pluginmanager.registered == {}


def test_install_is_inert_without_xdist() -> None:
    """``-p no:xdist``: no workers can crash, and registering a plugin that
    implements ``pytest_testnodedown`` would trip pluggy's unknown-hook check."""
    config = SimpleNamespace(pluginmanager=_FakePluginManager(xdist=False))
    assert install(config) is None  # type: ignore[arg-type]
    assert config.pluginmanager.registered == {}


# =============================================================================
# Nested, faithful: the real tests/conftest.py under the Unit lane's shape
# =============================================================================


def _nested_env(tmp_path: Path, *, crash: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # This test itself runs inside an xdist WORKER in the lane, and the worker's
    # identity must not leak into the nested session (its workers get fresh
    # values from xdist; its controller must have none at all).
    for leaked in ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID"):
        env.pop(leaked, None)
    # Load only the plugins that participate in the mechanism (the watchdog
    # module's measured precedent): xdist for the workers, pytest_timeout
    # because the lane's --timeout is on the command line. The repo's full
    # plugin set costs a heavy import chain in ALL THREE nested interpreters,
    # and that memory pressure is what killed an OUTER lane worker mid-test on
    # the shared droplet. tests/conftest.py is a conftest, not an autoloaded
    # plugin, so the wiring under test still loads exactly as in CI.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTEST_PLUGINS"] = "xdist.plugin,pytest_timeout"
    env.pop(ENV_CRASH, None)
    env[ENV_SENTINEL] = str(tmp_path / "gw1-collected")
    if crash:
        env[ENV_CRASH] = MODE_COLLECTION
    return env


def _run_probe(tmp_path: Path, *, crash: bool) -> tuple[int, str]:
    # From REPO_ROOT so rootdir is the repo: pyproject's addopts
    # (``--max-worker-restart=0``) and tests/conftest.py both apply, exactly as
    # in the lane. The three explicit flags are the lane's own.
    result = _run_guarded(
        [
            sys.executable,
            "-m",
            "pytest",
            PROBE_DIR,
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist=loadscope",
            "--timeout=30",
        ],
        cwd=REPO_ROOT,
        env=_nested_env(tmp_path, crash=crash),
        timeout=NESTED_BUDGET_SECONDS,
    )
    return result.returncode, _as_text(result.stdout) + _as_text(result.stderr)


# The lane runs with --timeout=30 and timeout_method="thread", which would
# os._exit the worker running these ~18s nested sessions. Explicit budget, as
# for the sibling nested-pytest tests.
@pytest.mark.slow
@pytest.mark.timeout(180)
def test_a_collection_phase_worker_crash_is_red_and_named(tmp_path: Path) -> None:
    """RED on HEAD: rc=0, ``4 warnings in 18s``, nothing ran, no explanation."""
    rc, output = _run_probe(tmp_path, crash=True)
    tail = output[-4000:]

    # Positive control for the arm itself: the crash must actually have been
    # induced, or a red result here proves nothing about the guard.
    assert "node down" in output, f"the probe did not crash a worker.\n{tail}"
    assert _ANY_PASSED.search(output) is None, f"a test ran; the crash was too late.\n{tail}"

    assert rc == GUARD_EXIT_CODE, (
        f"a worker crashed before any test ran and the session exited {rc} "
        f"(HEAD: 0 -- the vacuous green of #1848).\n{tail}"
    )
    assert BANNER in output, f"the guard did not explain the failure.\n{tail}"
    assert "gw0" in output.split(BANNER, 1)[1], tail


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_a_healthy_run_is_untouched(tmp_path: Path) -> None:
    """Positive control: same directory, same shape, no crash -> 2 passed, rc 0."""
    rc, output = _run_probe(tmp_path, crash=False)
    tail = output[-4000:]

    assert rc == 0, f"the guard turned a healthy session red.\n{tail}"
    assert "2 passed" in output, tail
    assert "node down" not in output, tail
    assert BANNER not in output, f"the guard fired on a healthy session.\n{tail}"
