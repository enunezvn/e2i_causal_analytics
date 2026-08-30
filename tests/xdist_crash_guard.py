"""Refuse a green exit when an xdist worker crashed and tests never reported (#1848).

Why this exists
---------------
Under ``-n N`` a worker that dies is reported to the controller through
``DSession.worker_errordown``, which asks the scheduler which item the worker
was *running* and synthesises a failed ``TestReport`` for it. That report is
the only thing that makes a crash red -- ``session.testsfailed`` is bumped by
the terminal reporter when it sees it -- and ``--max-worker-restart=0`` (#1648)
then ends the session promptly with the culprit named.

A worker that dies **during collection** was running nothing. No report is
synthesised, ``testsfailed`` stays 0, the session shuts down, and pytest's
``_main`` returns whatever ``testscollected`` implies:

* the peer's ``collectionfinish`` already arrived -> ``testscollected > 0`` ->
  **exit 0** with nothing run (observed: ``6 warnings in 13.48s``, rc=0);
* the crash arrived first -> the peer's collection is dropped on shutdown ->
  ``testscollected == 0`` -> exit 5, "no tests collected" -- red, but the
  wrong diagnosis.

Both were reproduced deterministically in
``tests/unit/test_tests_meta/test_xdist_vacuous_green_1848.py``; the table in
its docstring has the measured counters.

What this does
--------------
Registered on the xdist **controller** by ``tests/conftest.py``. It records
every ``pytest_testnodedown`` that carries an error and every
``pytest_runtest_logfinish``. At ``pytest_sessionfinish`` it fires -- prints a
:data:`BANNER` block and forces ``session.exitstatus`` to
:data:`GUARD_EXIT_CODE` -- only when ALL of:

1. at least one worker went down with an error;
2. the session would otherwise exit 0 or 5 (anything else is already red and
   owned by the path that made it so);
3. fewer items finished than ``session.testscollected``, or nothing was
   collected at all.

Condition 1 is what keeps every healthy run untouched: ``--collect-only``
reports nothing and crashes nothing; ``-k`` deselecting everything is pytest's
own rc=5, no crash; a worker's normal exit is a ``pytest_testnodedown`` with
``error=None``. Condition 3 keeps a crash *after* every result is in silent, so
the guard is never redder than before unless results were actually lost.

Exit code
---------
:data:`GUARD_EXIT_CODE` is ``ExitCode.TESTS_FAILED`` (1), the code the very
same crash produces when it lands one phase later, inside a test. The
:data:`BANNER` block carries the diagnosis; the code says "failed", which is
what a lost run is.

Scope
-----
Not installed on workers (``config.workerinput``), where none of the hooks
fire, nor when xdist's hookspecs are absent (``-p no:xdist``, or xdist not
loaded at all): a plugin object implementing ``pytest_testnodedown`` cannot be
registered without them, and with no workers there is nothing to guard.
"""

from __future__ import annotations

import sys

import pytest

__all__ = [
    "BANNER",
    "GUARD_EXIT_CODE",
    "PLUGIN_NAME",
    "XdistCrashGuard",
    "install",
]

#: Greppable marker; present in every line of the block the guard prints.
BANNER = "E2I-XDIST-CRASH-GUARD"

#: Name the guard is registered under on the controller's plugin manager.
PLUGIN_NAME = "e2i-xdist-crash-guard"

#: What the session exits with when the guard fires. See "Exit code" above.
GUARD_EXIT_CODE = pytest.ExitCode.TESTS_FAILED

#: Exit statuses the guard may override: the two a lost run can hide behind.
_GREEN_STATUSES = (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED)


class XdistCrashGuard:
    """Controller-side plugin. Inert until a worker goes down with an error."""

    def __init__(self) -> None:
        self.crashed: list[str] = []
        self.finished: set[str] = set()
        self.report: str | None = None

    # -- recording ----------------------------------------------------------

    def pytest_testnodedown(self, node: object, error: object | None) -> None:
        """xdist fires this for every worker exit; ``error`` is ``None`` on a clean one."""
        if error is None:
            return
        gateway = getattr(node, "gateway", None)
        gateway_id = getattr(gateway, "id", None) or repr(node)
        self.crashed.append(f"{gateway_id}: {error}")

    def pytest_runtest_logfinish(self, nodeid: str, location: object) -> None:
        self.finished.add(nodeid)

    # -- the decision, kept pure so it can be tested without a session -------

    def assess(self, *, collected: int, exitstatus: int | pytest.ExitCode) -> str | None:
        """Return the report to print if the session must be failed, else ``None``."""
        if not self.crashed:
            return None
        if int(exitstatus) not in {int(status) for status in _GREEN_STATUSES}:
            return None
        finished = len(self.finished)
        if collected > 0 and finished >= collected:
            return None
        return _render_report(
            crashed=self.crashed,
            collected=collected,
            finished=finished,
            exitstatus=int(exitstatus),
        )

    # -- acting ---------------------------------------------------------------

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(
        self, session: pytest.Session, exitstatus: int | pytest.ExitCode
    ) -> None:
        """Runs on the controller after ``_main`` has decided the exit status.

        ``wrap_session`` returns ``session.exitstatus`` *after* this hook, so
        overwriting it here is what changes the process exit code. The
        terminal reporter's own ``pytest_sessionfinish`` is a wrapper around
        this call and prints its summary afterwards, so the block below lands
        just above the final stats line.
        """
        self.report = self.assess(collected=session.testscollected, exitstatus=exitstatus)
        if self.report is None:
            return
        session.exitstatus = GUARD_EXIT_CODE
        _write(session.config, self.report)


def install(config: pytest.Config) -> XdistCrashGuard | None:
    """Register the guard on the controller; return it, or ``None`` when inert.

    Called from ``tests/conftest.py``'s ``pytest_configure``. A worker returns
    ``None`` (``workerinput`` is set there and none of the hooks fire); so does
    a session without xdist's hookspecs, which the guard needs to register at
    all. The hookspec is probed rather than ``hasplugin("xdist")`` because the
    plugin's registered name depends on how it was loaded: ``"xdist"`` from the
    entry point, but ``"xdist.plugin"`` under ``PYTEST_PLUGINS`` with autoload
    disabled -- the nested sessions in this guard's own test load it that way.
    """
    if hasattr(config, "workerinput"):
        return None
    if not hasattr(config.pluginmanager.hook, "pytest_testnodedown"):
        return None
    guard = XdistCrashGuard()
    config.pluginmanager.register(guard, name=PLUGIN_NAME)
    return guard


def _render_report(*, crashed: list[str], collected: int, finished: int, exitstatus: int) -> str:
    lines = [
        "",
        "=" * 78,
        f"{BANNER}: refusing a green exit -- an xdist worker crashed and not every "
        f"collected test reported (#1848)",
        "=" * 78,
        f"{BANNER}: {finished} of {collected} collected test(s) reported; "
        f"pytest was about to exit {exitstatus}. Crashed worker(s):",
    ]
    lines.extend(f"{BANNER}:     {entry}" for entry in crashed)
    lines += [
        f"{BANNER}: A worker that dies during collection was running no item, so xdist",
        f"{BANNER}: synthesises no failure for it and --max-worker-restart=0 simply ends the",
        f"{BANNER}: session. Look for '[gwN] node down:' above for the crash itself.",
        f"{BANNER}: Exit status forced to {int(GUARD_EXIT_CODE)}.",
        "=" * 78,
        "",
    ]
    return "\n".join(lines)


def _write(config: pytest.Config, text: str) -> None:
    """Print through pytest's terminal writer, falling back to stderr.

    ``get_terminal_writer`` is the terminal reporter's own writer, so the block
    appears in the job log in sequence with the summary rather than being
    swallowed by capture.
    """
    try:
        terminal = config.get_terminal_writer()
        terminal.write(text)
        terminal.flush()
    except Exception:  # pragma: no cover - no terminal plugin, or a closed writer
        sys.stderr.write(text)
        sys.stderr.flush()
