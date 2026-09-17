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

#: xdist's controller's own line when a worker dies: ``[gw0] node down: <reason>`` at
#: the start of a line. The guard never kills a worker -- it only observes
#: ``pytest_testnodedown`` and forces an exit status -- so on a probe that induced NO
#: crash, this can only mean the environment killed one.
#:
#: ANCHORED DELIBERATELY, and a plain ``"node down" in output`` is WRONG here: the
#: guard's own report says ``Look for '[gwN] node down:' above`` (xdist_crash_guard.py
#: :182), so the bare substring is satisfied by the guard's output alone. A guard that
#: wrongly fired on a healthy session would then have been classified as an
#: ENVIRONMENT failure and skipped -- the precise trap this split exists to avoid.
#: Measured: with both of ``assess``'s early returns defeated, the bare-substring
#: version skipped instead of failing. ``[gwN]`` in that prose is a literal N, and the
#: line is BANNER-prefixed, so the anchored pattern cannot match it either way.
_WORKER_DIED_RE = re.compile(r"^\[gw\d+\] node down", re.MULTILINE)


def _worker_died(output: str) -> bool:
    return _WORKER_DIED_RE.search(output) is not None


def _mem_available() -> str:
    """``MemAvailable`` right now, for a skip reason. Best-effort and never fatal."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return f"{int(line.split()[1]) // 1024} MiB available"
    except (OSError, ValueError, IndexError):  # pragma: no cover - platform-dependent
        pass
    return "MemAvailable unknown"


def _precondition_failure(rc: int, output: str) -> str | None:
    """Return a reason when the healthy-probe PRECONDITION failed, else ``None``.

    Split out from the nested test so both directions can be exercised in-process,
    on synthetic output, at no memory cost -- which matters here more than usual,
    because the thing being classified is a failure caused by memory pressure.

    The discrimination, and why it is sound rather than convenient:

    * A worker death on a ``crash=False`` probe is the ENVIRONMENT. The probe induced
      no crash, and the guard under test cannot cause one: it implements
      ``pytest_testnodedown``/``pytest_runtest_logfinish``/``pytest_sessionfinish`` and
      does nothing but record, print and set ``session.exitstatus``. So this case says
      nothing about the guard, and reporting it as a guard verdict is a false statement
      about the cause.
    * EVERYTHING ELSE stays a hard failure, deliberately. A blanket
      skip-on-any-nonzero-rc would swallow the real regression this control exists to
      catch -- the guard reddening or annotating a session that was perfectly healthy.
    """
    if not _worker_died(output):
        return None
    return (
        "PRECONDITION FAILED, not a guard result: a worker of the NESTED session died on "
        f"a probe that induced no crash (rc={rc}, {_mem_available()}). This control can "
        "only speak about the guard when the nested session actually stays healthy, and "
        "the guard cannot kill a worker -- it only observes and sets an exit status. "
        "#1848 itself was observed when memory pressure killed a worker on this box, so "
        "this is the same failure mode, still live, not a new one. Re-run when the box "
        "is quieter; the guard's behaviour on a healthy session is covered hermetically "
        "by test_sessionfinish_leaves_a_healthy_session_alone and "
        "test_silent_without_a_crash_even_when_nothing_reported."
    )


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


def test_a_worker_death_on_a_no_crash_probe_reports_as_environment() -> None:
    """The misattribution this replaces: a dead nested worker is NOT the guard.

    Signature of the real event, taken from the crash arm of this same module -- the
    controller prints ``node down``, the guard then fires legitimately (a worker died
    before any test reported), so BANNER is present and rc is GUARD_EXIT_CODE. Read
    through the HEALTHY path, the old first assertion called that "the guard turned a
    healthy session red", which is false in every particular.
    """
    output = (
        f"[gw0] node down: Not properly terminated\n{BANNER}\nreported 0 of 2 collected items\n"
    )
    reason = _precondition_failure(GUARD_EXIT_CODE, output)
    assert reason is not None
    assert reason.startswith("PRECONDITION FAILED")
    # It must say what happened and where to look, not merely decline to judge.
    assert "died" in reason and "no crash" in reason
    assert "test_sessionfinish_leaves_a_healthy_session_alone" in reason


def test_the_guards_own_report_does_not_read_as_a_worker_death() -> None:
    """The bug this discriminator had, pinned so it cannot return.

    ``xdist_crash_guard._render_report`` ends with ``Look for '[gwN] node down:' above``
    (:182), so a plain ``"node down" in output`` is satisfied by the GUARD'S OWN OUTPUT.
    A guard that wrongly fired on a healthy session would then have been classified as
    an ENVIRONMENT failure and skipped -- the exact trap this split exists to avoid,
    reintroduced by the thing meant to prevent it.

    Found by a teeth proof, not by reading: with both of ``assess``'s early returns
    defeated, the nested healthy session came back with the BANNER and a forced exit
    status, and the bare-substring version SKIPPED instead of failing.
    """
    from tests.xdist_crash_guard import _render_report

    report = _render_report(crashed=[], collected=2, finished=2, exitstatus=0)
    assert "node down" in report, (
        "the guard's prose no longer mentions 'node down'; this test's premise is stale"
    )
    # ... and the anchored matcher is not fooled by it.
    assert not _worker_died(report)
    assert _precondition_failure(int(GUARD_EXIT_CODE), report) is None
    # The real controller line still matches, so this is not narrowness for its own sake.
    assert _worker_died("[gw0] node down: Not properly terminated")
    assert _worker_died("prefix\n[gw11] node down: killed\nsuffix")


def test_a_guard_defect_is_never_reported_as_environment() -> None:
    """The trap: a blanket skip-on-nonzero-rc would swallow the regression this
    control exists to catch. Only a worker death is environmental; every other way a
    healthy session can come back wrong stays a hard failure.
    """
    # The guard reddens a session in which nothing died.
    assert _precondition_failure(GUARD_EXIT_CODE, f"{BANNER}\n2 passed\n") is None
    # The guard annotates a green session.
    assert _precondition_failure(0, f"{BANNER}\n2 passed\n") is None
    # A clean run that did not report what it should have.
    assert _precondition_failure(0, "1 passed\n") is None
    # Any other nonzero rc with no worker death.
    assert _precondition_failure(1, "1 failed, 1 passed\n") is None
    # And the healthy case itself is not mistaken for a precondition failure.
    assert _precondition_failure(0, "2 passed in 3.01s\n") is None


def test_the_memory_note_is_present_and_never_raises() -> None:
    """The reason carries the free memory at that instant, because "re-run when the
    box is quieter" is only actionable with a number attached. It is diagnostic, not
    a gate: no threshold is invented, and an unreadable /proc must not turn a skip
    into an error."""
    note = _mem_available()
    assert note.endswith("MiB available") or note == "MemAvailable unknown"
    # The note must reach the reason. Compared by SHAPE, not by value: two calls a
    # moment apart legitimately differ, so an equality check here would be flaky for
    # a reason unrelated to what it is testing.
    reason = _precondition_failure(1, "[gw0] node down: killed")
    assert reason is not None
    assert "MiB available" in reason or "MemAvailable unknown" in reason


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
    assert _worker_died(output), f"the probe did not crash a worker.\n{tail}"
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
    """Positive control: same directory, same shape, no crash -> 2 passed, rc 0.

    The precondition is checked BEFORE any verdict about the guard. Previously the
    first assertion was ``rc == 0`` with the message "the guard turned a healthy
    session red" -- which is false when the box killed the nested ``gw0``, and it
    misdirects the next reader at the exact moment they are least able to check.
    The evidence needed to tell the two apart was already two lines further down.
    """
    rc, output = _run_probe(tmp_path, crash=False)
    tail = output[-4000:]

    reason = _precondition_failure(rc, output)
    if reason is not None:
        pytest.skip(f"{reason}\n{tail}")

    assert rc == 0, f"the guard turned a healthy session red.\n{tail}"
    assert "2 passed" in output, tail
    assert not _worker_died(output), tail
    assert BANNER not in output, f"the guard fired on a healthy session.\n{tail}"
