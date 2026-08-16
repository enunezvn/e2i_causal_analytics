"""Session inactivity watchdog for the pytest CONTROLLER process (issue #1655).

Why this exists
---------------
``pyproject.toml`` sets a per-test ``timeout`` and every CI lane passes its own
``--timeout`` (30s to 900s depending on the lane). That is a per-*test*
guarantee, never a session-level one, and the gap is structural rather than a
pytest-timeout bug:

* ``pytest_timeout`` arms its timer inside :func:`pytest_timeout.pytest_runtest_protocol`,
  a hookwrapper around **one item's** setup/call/teardown.
* Under ``-n N`` that hook runs in the **xdist worker**. The controller never
  arms a timer for anything.

So the moment the session stops making progress *outside* a worker's runtest
protocol -- in the controller's event loop, in a controller-side hook, in
session teardown, in an ``atexit`` flush -- there is no timer anywhere that can
fire. The session simply waits, silently, until the job's ``timeout-minutes``
cancels it, which produces no failure message and names no culprit.

Measured instance (run 31915535379, job 95087170617, ``Unit Tests``):
of 10,561 collected tests 10,556 reported; the controller printed the
``logstart`` line for ``tests/unit/api/test_errors.py::TestRateLimitErrors::test_rate_limit_error``
at 00:09:50 and never printed again; the job was cancelled at its 30-minute cap
at 00:29:13 with three live processes (controller + both workers). The lane's
``--timeout=30`` never fired and there is exactly one ``node down`` in the whole
log, belonging to an earlier unrelated crash. The wedged item is six lines of
in-memory assertions that pass in 4 ms.

Two mechanisms fit that evidence, and this watchdog covers both, which is why
it does not depend on choosing between them:

1. **Controller-side.** The stall is downstream of ``logstart``, in the
   controller's own event loop or in a hook running there. No timer exists in
   that process at all. Reproduced locally with no prior worker crash and
   ``--max-worker-restart=0`` in effect -- so #1648's fix does not cover it --
   by blocking inside a controller-side ``pytest_runtest_logstart`` hook: the
   log shape is identical, a bare nodeid line then silence to the wall clock.
2. **Worker-side, holding the GIL.** ``timeout_method = "thread"`` fires from a
   Python ``threading.Timer``, which needs the GIL to run at all. A worker
   inside a native call that never releases it (the shape of #1548's 1240.5s
   ``dense_tree_shap``) stays alive with no dump and no ``node down`` -- so the
   absence of a second ``node down`` does *not* by itself prove the worker was
   healthy. Nothing in the worker can self-diagnose that state either.

Both look the same from the controller: events stop arriving. That is exactly
what this watchdog measures.

What this does
--------------
The controller records the timestamp of every test-lifecycle event it observes
(``logstart`` / ``logreport`` / ``logfinish``). A daemon thread fails the
session when that timestamp goes stale, printing:

* how long the session has been silent,
* every nodeid that was dispatched but never reported (the culprit),
* the last item that did complete,
* the stack of every thread in the controller.

Then it calls ``os._exit`` so the job **fails** instead of being cancelled.

Choosing the window
-------------------
An inactivity watchdog is only safe if its window clears every phase in which a
*healthy* lane legitimately emits no test report. There are three such phases,
all measured over the six pytest jobs of run 31915535379 (startup = pytest step
start to first report; max gap = longest interval between two consecutive
reports; teardown = last report to end of the pytest step):

===============  =========  =========  ==========
lane             startup    max gap    teardown
===============  =========  =========  ==========
Unit Tests           103.7s      53.1s       10.9s
Heavy Unit Tests      32.4s      10.1s        3.1s
Agents Unit (2)       57.6s      11.7s        4.7s
Integration (3)       61.6s      18.5s        6.4s
Integration (1)       48.1s       5.0s        5.1s
===============  =========  =========  ==========

But observed gaps are a floor, not the bound. The real bound is **how long a
single test is allowed to run**, because the controller emits nothing at all
while one does. That budget is the larger of the lane's ``--timeout`` and any
``@pytest.mark.timeout`` on a test it collects, and it varies enormously here:

* ``tests/unit`` + ``tests/api``: ``--timeout`` 30-180s, largest marker 300s.
* ``tests/integration``: largest marker **2700s**
  (``test_synthetic_cohort_growth.py``, ``slow``-marked).
* ``slow-tests.yml``: ``--timeout`` 300-900s, and its serial lanes run the
  ``slow``-marked tests the backend lanes exclude.

So there is **no single safe number**, and in particular no safe
``CI``-triggered default -- one would have red-X'd the nightly. Each lane opts
in with :data:`ENV_TIMEOUT` sized against its own ceiling, and
``tests/unit/test_tests_meta/test_session_stall_watchdog_1655.py`` enforces
that sizing directly against the workflow files, so enabling it on a new lane
without raising the window fails a test rather than a nightly.
:func:`install` additionally refuses to arm a window at or below the session's
resolved ``--timeout``.

The watchdog deliberately stays armed through session teardown: a hang in the
OPIK trace flush at teardown is the same "~20-minute job cancel" shape (issue
#952) and costs nothing extra to cover. Measured teardown is 3-11s.

Scope
-----
Enabled only when :data:`ENV_TIMEOUT` is set. A local ``pytest`` run is
unaffected, so a developer sitting on a breakpoint is never killed, and no lane
gains a watchdog by accident.

Known limitations:

* The dump covers the controller's threads only. Workers are separate processes
  and signalling them is not safe here -- ``SIGUSR1`` reaching a worker that has
  not yet installed a handler would *terminate* it. The in-flight nodeid names
  the item.
* ``faulthandler.dump_traceback_later`` is process-global and single-slot, so
  arming the floor replaces any other ``dump_traceback_later`` alarm in the
  controller. The repo's only other faulthandler user is
  ``config/gunicorn.conf.py``, which calls ``faulthandler.enable`` in a server
  process -- a different API and a different process -- so there is nothing to
  clobber. The floor is only re-armed, never cancelled, while the session runs.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import traceback
from typing import Callable, Iterable, Mapping, TextIO

import pytest

__all__ = [
    "BANNER",
    "ENV_TIMEOUT",
    "FAULTHANDLER_GRACE_SECONDS",
    "PLUGIN_NAME",
    "STALL_EXIT_CODE",
    "SessionStallWatchdog",
    "install",
    "resolve_per_test_timeout",
    "resolve_timeout",
]

#: Seconds of pytest silence tolerated before the session is failed. This is
#: the ONLY switch: unset means the watchdog does not exist. There is
#: deliberately no ``CI``-derived default -- see "Choosing the window".
ENV_TIMEOUT = "E2I_PYTEST_STALL_TIMEOUT"

#: Extra slack given to the ``faulthandler`` floor so the Python watchdog --
#: which produces the readable diagnosis -- normally wins the race. The floor
#: only fires when the watchdog thread itself cannot be scheduled.
FAULTHANDLER_GRACE_SECONDS = 60.0

#: ``sysexits.h`` EX_SOFTWARE. Deliberately outside pytest's own 0-5 range so
#: the cause is unambiguous in a CI log.
STALL_EXIT_CODE = 70

#: Greppable marker. Present in the arming line and in the stall report.
BANNER = "E2I-STALL-WATCHDOG"

#: Name the watchdog is registered under on the controller's plugin manager.
PLUGIN_NAME = "e2i-stall-watchdog"


def resolve_timeout(environ: Mapping[str, str]) -> float:
    """Return the configured inactivity window in seconds, 0 meaning disabled.

    Opt-in only, and deliberately NOT derived from ``CI``. A window is only
    safe relative to the longest a *single test* in that lane may run, and
    those budgets differ by an order of magnitude across this repo's lanes
    (``--timeout`` from 30s to 900s; ``@pytest.mark.timeout`` up to 2700s in
    ``tests/integration/test_synthetic_cohort_growth.py``). A blanket
    CI-triggered default would have killed the nightly ``slow-tests`` lanes,
    where a single serial test legitimately runs for 45 minutes with the
    controller emitting nothing.

    So each lane opts in with a value sized against its own ceiling, and
    ``tests/unit/test_tests_meta/test_session_stall_watchdog_1655.py`` enforces
    that sizing against the workflow files.
    """
    raw = environ.get(ENV_TIMEOUT)
    if raw is None or not raw.strip():
        return 0.0
    try:
        value = float(raw.strip())
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


def resolve_per_test_timeout(config: pytest.Config) -> float:
    """Return the session's per-test ``--timeout``, mirroring pytest-timeout.

    Same precedence as :func:`pytest_timeout.get_env_settings`: command line,
    then ``PYTEST_TIMEOUT``, then the ini value. Used only to refuse an
    obviously-unsafe window; ``@pytest.mark.timeout`` overrides are per-item
    and invisible here, which is why the workflow-sizing guard test exists.
    """

    def _ini() -> object:
        # ``getini`` raises when pytest-timeout is not installed to register it.
        try:
            return config.getini("timeout")
        except (ValueError, KeyError):
            return None

    for candidate in (
        config.getoption("timeout", None),
        os.environ.get("PYTEST_TIMEOUT"),
        _ini(),
    ):
        if candidate is None or candidate == "":
            continue
        try:
            return float(candidate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return 0.0


class SessionStallWatchdog:
    """Fail the session when the controller stops observing test activity.

    The object is a pytest plugin: register it on the controller's plugin
    manager and its hooks record activity. It is inert until :meth:`start`.
    """

    def __init__(
        self,
        timeout: float,
        *,
        write: Callable[[str], None] | None = None,
        final_write: Callable[[str], None] | None = None,
        poll_interval: float | None = None,
    ) -> None:
        self.timeout = float(timeout)
        # ``write`` runs mid-session and must NOT disturb pytest's capture.
        # ``final_write`` runs immediately before ``os._exit`` and therefore
        # has to break out of capture, since nothing will replay the buffer.
        self._write = write if write is not None else _default_write
        self._final_write = final_write if final_write is not None else self._write
        # Frequent enough to keep the reported overshoot small, cheap enough to
        # be irrelevant next to a multi-minute window.
        self._poll = (
            poll_interval if poll_interval is not None else min(1.0, self.timeout / 20 or 1.0)
        )
        self._lock = threading.Lock()
        self._last_activity = time.monotonic()
        self._last_event = "session start"
        self._last_completed: str | None = None
        self._in_flight: set[str] = set()
        self._started_at = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- activity recording -------------------------------------------------

    def _touch(self, event: str) -> None:
        with self._lock:
            self._last_activity = time.monotonic()
            self._last_event = event

    def note_start(self, nodeid: str) -> None:
        with self._lock:
            self._last_activity = time.monotonic()
            self._last_event = f"logstart {nodeid}"
            self._in_flight.add(nodeid)

    def note_report(self, nodeid: str, when: str | None) -> None:
        self._touch(f"logreport {when or '?'} {nodeid}")

    def note_finish(self, nodeid: str) -> None:
        with self._lock:
            self._last_activity = time.monotonic()
            self._last_event = f"logfinish {nodeid}"
            self._in_flight.discard(nodeid)
            self._last_completed = nodeid

    # -- pytest hooks -------------------------------------------------------

    # tryfirst: record the dispatch before any other plugin's logstart hook
    # gets a chance to block, so a stall *inside* one of those hooks is still
    # attributed to the item it was dispatching (that is the observed shape).
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logstart(self, nodeid: str, location: object) -> None:
        self.note_start(nodeid)

    def pytest_runtest_logreport(self, report: object) -> None:
        self.note_report(str(getattr(report, "nodeid", "?")), getattr(report, "when", None))

    # trylast: only clear the in-flight entry once every other plugin has
    # finished with the item, so a stall in someone else's teardown hook still
    # names it.
    @pytest.hookimpl(trylast=True)
    def pytest_runtest_logfinish(self, nodeid: str, location: object) -> None:
        self.note_finish(nodeid)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Arm the watchdog. No-op when the configured window is 0."""
        if self.timeout <= 0 or self._thread is not None:
            return
        self._started_at = time.monotonic()
        self._last_activity = self._started_at
        self._arm_faulthandler_floor()
        self._thread = threading.Thread(target=self._loop, name="e2i-stall-watchdog", daemon=True)
        self._thread.start()
        self._write(
            f"\n{BANNER}: armed — the session fails if the controller (pid "
            f"{os.getpid()}) observes no test activity for {self.timeout:g}s. "
            f"Override or disable with {ENV_TIMEOUT}.\n"
        )

    def stop(self) -> None:
        """Disarm. Only used by tests; CI leaves it armed through teardown."""
        self._stop.set()
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:  # pragma: no cover - defensive
            pass

    def _arm_faulthandler_floor(self) -> None:
        """Re-arm the GIL-proof floor.

        ``dump_traceback_later`` runs in a C watchdog thread that does not need
        the GIL, so it still fires when a native call in the controller has
        locked every Python thread out (the shape of #1548's 1240s
        ``dense_tree_shap`` call). It is the *floor*, not the primary path: its
        dump may land in pytest's capture buffer rather than the job log, but
        ``exit=True`` still turns a silent hang into a failed job.
        """
        try:
            faulthandler.dump_traceback_later(self.timeout + FAULTHANDLER_GRACE_SECONDS, exit=True)
        except Exception:  # pragma: no cover - defensive
            pass

    # -- the watchdog thread ------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self._poll):
            with self._lock:
                idle = time.monotonic() - self._last_activity
            if idle < self.timeout:
                self._arm_faulthandler_floor()
                continue
            self._fire(idle)
            return

    def _fire(self, idle: float) -> None:
        with self._lock:
            in_flight = sorted(self._in_flight)
            last_event = self._last_event
            last_completed = self._last_completed
            elapsed = time.monotonic() - self._started_at
        self._final_write(
            _render_report(self.timeout, idle, elapsed, last_event, last_completed, in_flight)
        )
        os._exit(STALL_EXIT_CODE)


def install(config: pytest.Config) -> SessionStallWatchdog | None:
    """Arm the watchdog for this session and return it, or ``None`` if inert.

    Called from ``tests/conftest.py``'s ``pytest_configure``, so a lane opts in
    purely by setting :data:`ENV_TIMEOUT` -- no per-lane ``-p`` plumbing.

    Only the **controller** arms it. An xdist worker (``config.workerinput``
    exists) is already covered by ``--timeout`` inside its runtest protocol,
    and a worker legitimately idles for minutes at the tail of a session once
    the scheduler has no more work for it -- watching a worker for inactivity
    would fire on that.

    Refuses to arm a window at or below the session's own per-test
    ``--timeout``: the controller sees nothing at all while one test runs, so
    such a window is guaranteed to fire on a healthy long test. Better to run
    unguarded and say so than to red-X a good session. When no per-test timeout
    is configured at all the check has nothing to compare against and is
    skipped; ``pyproject.toml`` sets ``timeout = 30`` for every session in this
    repo, so that path is not reachable here today.
    """
    if hasattr(config, "workerinput"):
        return None
    timeout = resolve_timeout(os.environ)
    if timeout <= 0:
        return None
    per_test = resolve_per_test_timeout(config)
    if per_test and timeout <= per_test:
        _default_write(
            f"\n{BANNER}: NOT armed — {ENV_TIMEOUT}={timeout:g}s is not above "
            f"this session's per-test --timeout of {per_test:g}s. The controller "
            f"is silent for the whole of a single test, so that window would "
            f"fire on a healthy one. Raise it well clear of the lane's longest "
            f"per-test budget (including any @pytest.mark.timeout).\n"
        )
        return None
    watchdog = SessionStallWatchdog(
        timeout,
        write=_default_write,
        final_write=_make_final_write(config),
    )
    config.pluginmanager.register(watchdog, name=PLUGIN_NAME)
    watchdog.start()
    return watchdog


def _make_final_write(config: pytest.Config) -> Callable[[str], None]:
    """Build a writer that escapes pytest's capture before printing.

    Mirrors what :func:`pytest_timeout.timeout_timer` does before its own
    ``os._exit``: suspend the capture manager so fd 1/2 point back at the real
    job log, then write through the terminal writer. Nothing resumes capture
    afterwards, which is fine -- the process is about to exit.
    """

    def final_write(text: str) -> None:
        try:
            capman = config.pluginmanager.getplugin("capturemanager")
        except Exception:  # pragma: no cover - defensive
            capman = None
        if capman is not None:
            try:
                capman.suspend_global_capture()
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            terminal = config.get_terminal_writer()
            terminal.write(text)
            terminal.flush()
        except Exception:  # pragma: no cover - defensive
            _default_write(text)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # pragma: no cover - defensive
                pass

    return final_write


def _render_report(
    window: float,
    idle: float,
    elapsed: float,
    last_event: str,
    last_completed: str | None,
    in_flight: Iterable[str],
) -> str:
    in_flight = list(in_flight)
    lines = [
        "",
        "=" * 78,
        f"{BANNER}: SESSION STALLED — failing the job instead of waiting for its cap",
        "=" * 78,
        f"No pytest activity for {idle:.1f}s — over the {window:g}s window "
        f"({ENV_TIMEOUT}) — {elapsed:.1f}s into the session.",
        f"Last event seen by the controller: {last_event}",
        f"Last item that completed:          {last_completed or '(none)'}",
    ]
    if in_flight:
        lines.append(f"Dispatched but never reported ({len(in_flight)}) — this is the culprit:")
        lines.extend(f"    {nodeid}" for nodeid in in_flight)
    else:
        lines.append(
            "Nothing was in flight: the session stalled between items or in "
            "teardown (collection, coverage combine, or an atexit flush)."
        )
    lines += [
        "",
        "pytest-timeout cannot cover this: --timeout arms a timer inside an xdist",
        "WORKER around one item's runtest protocol, so a stall in the controller —",
        "or in any phase outside that protocol — is invisible to it (#1655).",
        "",
        "-" * 78,
        "controller thread stacks",
        "-" * 78,
    ]
    lines.append(_format_thread_stacks())
    lines += ["=" * 78, ""]
    return "\n".join(lines)


def _format_thread_stacks() -> str:
    """Render every thread's stack.

    Built in pure Python rather than via ``faulthandler.dump_traceback`` so the
    text can go through the same writer as the rest of the report; the
    faulthandler path needs a real file descriptor, which pytest's fd-level
    capture has redirected out from under us.
    """
    names = {t.ident: t.name for t in threading.enumerate()}
    chunks = []
    for ident, frame in sys._current_frames().items():
        chunks.append(f"--- thread {names.get(ident, '<unknown>')} ({ident}) ---")
        chunks.append("".join(traceback.format_stack(frame)).rstrip())
    return "\n".join(chunks)


def _default_write(text: str) -> None:
    """Write the report where CI will actually see it.

    pytest's fd-level global capture has fd 1/2 pointing at temp files, and
    ``os._exit`` skips the flush that would replay them, so the message has to
    go through the capture manager the way :mod:`pytest_timeout` does. The
    plugin installs a capture-aware writer at configure time; this fallback is
    for the un-captured case.
    """
    stream: TextIO = sys.stderr
    try:
        stream.write(text)
        stream.flush()
    except Exception:  # pragma: no cover - defensive
        pass
