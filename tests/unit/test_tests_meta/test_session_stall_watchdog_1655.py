"""#1655 guard: a stalled session must FAIL fast, naming what was in flight.

The failure this guards against
-------------------------------
``--timeout=30`` is not a session-level guarantee. ``pytest_timeout`` arms its
timer in a hookwrapper around ``pytest_runtest_protocol``, which under ``-n N``
runs **inside the xdist worker**. Nothing in the controller process is ever
covered. So a stall anywhere outside a worker's runtest protocol -- the
controller's event loop, a controller-side hook, session teardown, an
``atexit`` flush -- has no timer at all, and the session waits in silence until
the job's ``timeout-minutes`` *cancels* it: no failure message, no culprit.

Measured instance, run 31915535379 / job 95087170617 (``Unit Tests``):

* 10,556 of 10,561 collected tests reported;
* the controller printed the ``logstart`` line for
  ``tests/unit/api/test_errors.py::TestRateLimitErrors::test_rate_limit_error``
  at 00:09:50 and never printed another line;
* the job was cancelled at its 30-minute cap at 00:29:13, leaving three live
  processes -- controller plus **both** workers;
* the whole log contains exactly one ``node down``, from an earlier unrelated
  crash, and no pytest-timeout stack dump anywhere.

The wedged item is six lines of in-memory assertions that pass in 4 ms (it
passed at 99% in the next run of the same lane). Two mechanisms fit: a
controller-side stall downstream of ``logstart``, where no timer exists at all;
or a worker inside a native call holding the GIL, which starves the
``threading.Timer`` that ``timeout_method = "thread"`` depends on (the shape of
#1548's 1240.5s ``dense_tree_shap``) and therefore also produces no ``node
down``. Both look identical from the controller -- events stop arriving -- and
the watchdog measures exactly that, so it covers both without the diagnosis
having to be settled.

Why #1654 does not cover it
---------------------------
``--max-worker-restart=0`` ends the session at the first *worker death*. The
stall this file guards needs no worker death at all --
:func:`test_a_controller_side_stall_hangs_with_the_watchdog_disabled` reproduces
it with that flag set and zero crashes.

Why a watchdog, and why it is safe
----------------------------------
#1648 rejected a no-output watchdog because it "names no culprit, and the jobs'
``timeout-minutes`` already is one". Both objections are addressed here: the
watchdog prints every dispatched-but-unreported nodeid plus the stack of every
controller thread, and it fires ~14 minutes before the cap did, turning a
cancellation into a failure.

False-positive risk is bounded by measurement, not by hope. Across the six
pytest jobs of run 31915535379 the longest phase in which a *healthy* lane
emits no test report is 103.7s (Unit Tests startup); the longest gap between
two consecutive reports is 53.1s; the longest teardown -- the coverage combine
the watchdog must not kill -- is 10.9s.

Those are floors, though, not the bound: the controller is silent for the whole
of a *single* test, so the window must clear the longest per-test budget the
lane can reach. That is why the watchdog is opt-in per lane rather than
CI-derived (``slow-tests.yml`` runs ``--timeout=900`` serially and collects a
``@pytest.mark.timeout(2700)``), and why
:func:`test_every_lane_that_opts_in_is_sized_above_its_own_longest_test` checks
each opted-in lane's window against its own ceiling, read out of the workflow.

Nothing spawned here may outlive the test (#1842)
------------------------------------------------
Every arm spawns a nested pytest, and the RED arm spawns one that is *designed*
never to end. ``subprocess.run(timeout=...)`` kills that child only on the
Python-level ``TimeoutExpired`` path. If the outer test process dies hard while
blocked in ``communicate()`` -- SIGKILL, SIGTERM (Python installs no handler),
pytest-timeout's ``os._exit`` -- no ``except`` or ``finally`` runs, and the
nested controller reparents to init and spins in ``pbkdf2_hmac`` at 100% of a
host core until someone notices. On the droplet that was pid 3420168, 22 hours
in, on the shared PROD==DEV box. Reproduced by SIGKILLing the outer pytest
mid-wedge: the controller survived at 92% CPU with both execnet workers.

So every nested session goes through :func:`_spawn_nested`, which

* starts it in its own session and process group, so
  :func:`_end_nested_session` can ``killpg`` the controller *and* its workers
  in every Python-level exit path and then prove that nothing is left;
* execs the controller through a tiny wrapper that arms ``PR_SET_PDEATHSIG``
  first, so the kernel SIGKILLs the controller the instant its parent dies
  for any reason -- the path no ``finally`` can reach. The workers then exit
  on stdin EOF (measured: 0.47s).
  :func:`test_the_nested_session_dies_when_its_parent_is_hard_killed` forces
  exactly that path.
"""

from __future__ import annotations

import ast
import os
import re
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from tests.stall_watchdog import (
    BANNER,
    ENV_TIMEOUT,
    STALL_EXIT_CODE,
    resolve_timeout,
)

# tests/unit/test_tests_meta/<this file> -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The nested session's inactivity window. Small enough to keep this file cheap
#: in the Unit Tests lane, large enough that the nested session's own startup
#: cannot trip it (measured boot with autoload disabled: ~1.2s).
NESTED_STALL_SECONDS = 8

#: Wall clock allowed for the nested session. Measured end-to-end: 9.2s.
#: Anything near this bound means the watchdog did not fire.
NESTED_BUDGET_SECONDS = 90

#: How long the disabled-watchdog arm is given to prove it hangs. Four times
#: the nested ``--timeout`` (5s), so "pytest-timeout had every chance to fire
#: and did not" is what the arm actually demonstrates.
HANG_PROOF_SECONDS = 20

_WEDGED_TEST = "test_suite.py::TestWedge::test_never_reports"

# The nested project blocks the CONTROLLER inside ``pytest_runtest_logstart``.
# That is the observed shape: the controller announces the item, then stops.
# ``hashlib.pbkdf2_hmac`` is the faithful stand-in established by #1648 --
# ``time.sleep``, ``re`` backtracking and numpy matmul are all interrupted by
# SIGALRM and would not prove anything about a native stall.
_NESTED_CONFTEST = textwrap.dedent(
    """
    import hashlib
    import sys

    sys.path.insert(0, {repo_root!r})

    from tests.stall_watchdog import install

    _IS_WORKER = False


    def pytest_configure(config):
        global _IS_WORKER
        _IS_WORKER = hasattr(config, "workerinput")
        install(config)


    def pytest_runtest_logstart(nodeid, location):
        if _IS_WORKER or "test_never_reports" not in nodeid:
            return
        # Controller-side, uninterruptible, forever.
        while True:
            hashlib.pbkdf2_hmac("sha256", b"x", b"y", 40_000_000)
    """
)

_NESTED_SUITE = textwrap.dedent(
    """
    class TestEarly:
        def test_reports_normally(self):
            assert True


    class TestAlsoEarly:
        def test_also_reports_normally(self):
            assert True


    class TestWedge:
        def test_never_reports(self):
            assert True
    """
)

# Mirrors the real lanes: thread-method per-test timeout, loadscope, and #1654's
# restart ban -- so the arm below proves the stall survives BOTH existing
# defences rather than only one.
_NESTED_INI = textwrap.dedent(
    """
    [pytest]
    timeout = 5
    timeout_method = thread
    addopts = -v --tb=short -n 2 --dist=loadscope --max-worker-restart=0
    """
)


def _write_nested_project(root: Path) -> None:
    (root / "pytest.ini").write_text(_NESTED_INI)
    (root / "conftest.py").write_text(_NESTED_CONFTEST.format(repo_root=str(REPO_ROOT)))
    (root / "test_suite.py").write_text(_NESTED_SUITE)


def _nested_env(stall_seconds: int | str) -> dict[str, str]:
    env = dict(os.environ)
    env[ENV_TIMEOUT] = str(stall_seconds)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Load only the two plugins whose interaction is under test. This repo's
    # installed plugin set costs ~19.6s of boot per nested session versus ~1.2s
    # without it, and none of those plugins participate in the mechanism: the
    # stall is a controller hook blocking between xdist's ``worker_logstart``
    # and the watchdog thread.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTEST_PLUGINS"] = "xdist.plugin,pytest_timeout"
    return env


# =============================================================================
# Spawning: nothing this file starts may outlive it (#1842)
# =============================================================================

#: How long a SIGKILLed process group is given to drain. Reaping is init's job
#: and takes milliseconds; the execnet workers of a killed controller exit on
#: stdin EOF in ~0.5s (measured).
REAP_GRACE_SECONDS = 10.0

#: ``PR_SET_PDEATHSIG`` is a Linux ``prctl``. Elsewhere the nested session is
#: spawned bare and only the process-group teardown applies.
_PDEATHSIG_SUPPORTED = sys.platform.startswith("linux")

#: Exit status of the wrapper when it cannot arm PDEATHSIG. It refuses to
#: start an unguarded nested session, loudly, rather than run one that can
#: leak -- ``sysexits.h`` EX_OSERR, outside pytest's own 0-5 range.
WRAPPER_UNGUARDED_EXIT = 71

# The nested controller is started through this single-purpose wrapper, not
# a ``preexec_fn``. ``preexec_fn`` runs Python in the forked child *before*
# ``exec`` while the parent is multithreaded -- ``timeout_method = "thread"``
# has a pytest-timeout Timer running for every test here -- which the
# subprocess docs call unsafe: a child deadlocked before exec leaves the parent
# stuck inside ``Popen()`` with PDEATHSIG never installed, which is the very
# orphan this file guards against. The wrapper does its work in a fresh,
# single-threaded process *after* exec, then replaces itself with the real
# command: same pid, same session and group, same command line. It also closes
# the fork->prctl race, which ``preexec_fn`` could not: PDEATHSIG cannot fire
# for a parent that died before it was armed, so the wrapper checks that its
# parent is still the process that spawned it. ``PR_SET_PDEATHSIG`` survives a
# non-setuid ``exec``, so the controller inherits it.
_PDEATHSIG_WRAPPER = textwrap.dedent(
    """
    import ctypes
    import os
    import signal
    import sys

    expected_parent = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # 1 == PR_SET_PDEATHSIG
        print(
            "PR_SET_PDEATHSIG failed (errno %d); refusing to start an unguarded "
            "nested pytest" % ctypes.get_errno(),
            file=sys.stderr,
        )
        sys.exit({unguarded_exit})
    if os.getppid() != expected_parent:
        # The parent died between fork and prctl. Nothing will ever kill us
        # for that death, so end here instead of running orphaned.
        sys.exit({unguarded_exit})
    os.execvp(sys.argv[2], sys.argv[2:])
    """
).format(unguarded_exit=WRAPPER_UNGUARDED_EXIT)


def _spawn_nested(cmd: list[str], *, cwd: str | Path, env: dict[str, str]) -> subprocess.Popen[str]:
    """Start a nested pytest so that it can always be torn down.

    ``start_new_session`` makes the child the leader of a fresh session and
    process group, which its xdist workers inherit -- so one ``killpg`` on
    ``proc.pid`` reaches the controller and every worker. On Linux the child
    is :data:`_PDEATHSIG_WRAPPER`, which arms PDEATHSIG and then ``exec``s
    ``cmd`` in place, so the kernel ends the controller the instant its
    parent dies; see the module docstring.
    """
    if _PDEATHSIG_SUPPORTED:
        cmd = [sys.executable, "-c", _PDEATHSIG_WRAPPER, str(os.getpid()), *cmd]
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _group_members(pgid: int) -> list[str]:
    """Live (non-zombie) processes in process group ``pgid``, read from ``/proc``.

    One line per process -- pid, ppid, state, command -- so a failed guard
    prints what it computed rather than just "something survived".
    """
    members: list[str] = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            stat = (entry / "stat").read_text()
            # ``comm`` may itself contain spaces or ')': split after its close.
            state, ppid, pgrp = stat.rsplit(")", 1)[1].split()[:3]
            if int(pgrp) != pgid or state in "ZX":
                continue
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError):
            continue  # raced with the process exiting
        members.append(f"pid {entry.name} ppid {ppid} state {state}: {cmd.strip()[:200]}")
    return members


def _drain_group(pgid: int, grace: float = REAP_GRACE_SECONDS) -> list[str]:
    """Wait up to ``grace`` seconds for the group to empty; return what is left."""
    deadline = time.monotonic() + grace
    survivors = _group_members(pgid)
    while survivors and time.monotonic() < deadline:
        time.sleep(0.1)
        survivors = _group_members(pgid)
    return survivors


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # nothing left in the group


def _reap(proc: subprocess.Popen[str]) -> None:
    """Bounded wait for the group leader.

    Never unbounded: this file exists because of hangs, and a teardown that
    can hang on the very process it failed to kill would hide that failure
    behind the per-test timeout's ``os._exit``. The survivor scan reports
    anything still alive after the grace period.
    """
    try:
        proc.wait(timeout=REAP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _end_nested_session(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the whole nested session and prove that none of it survived."""
    _kill_group(proc.pid)
    _reap(proc)
    survivors = _drain_group(proc.pid)
    assert not survivors, (
        f"{len(survivors)} process(es) of the nested pytest session (pgid {proc.pid}) "
        f"outlived the test after SIGKILL to the group -- the leak of #1842:\n  "
        + "\n  ".join(survivors)
    )


def _run_guarded(
    cmd: list[str], *, cwd: str | Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run`` semantics with the nested session torn down on every exit path.

    Raises :class:`subprocess.TimeoutExpired` exactly as ``subprocess.run``
    does (partial output attached, as bytes) -- but only after the whole
    process group is dead and proven gone, so the RED arm can still observe
    its hang without leaving it behind.

    Not a ``with Popen(...)`` block on purpose: ``Popen.__exit__`` waits on
    the leader with no bound, which is the hang :func:`_reap` refuses.
    """
    proc = _spawn_nested(cmd, cwd=cwd, env=env)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _end_nested_session(proc)
            raise
        except BaseException:
            _kill_group(proc.pid)  # do not mask the original error; just do not leak
            _reap(proc)
            raise
        _end_nested_session(proc)
    finally:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _run_nested(
    root: Path, *, stall_seconds: int | str, timeout: float
) -> subprocess.CompletedProcess[str]:
    return _run_guarded(
        [sys.executable, "-m", "pytest", str(root), "-p", "no:cacheprovider"],
        cwd=root,
        env=_nested_env(stall_seconds),
        timeout=timeout,
    )


def _as_text(chunk: bytes | str | None) -> str:
    """Normalise captured output.

    ``subprocess.run(text=True)`` hands back ``str`` on completion but
    ``TimeoutExpired`` carries the partial read as ``bytes``.
    """
    if chunk is None:
        return ""
    return chunk.decode(errors="replace") if isinstance(chunk, bytes) else chunk


# =============================================================================
# The bug: a controller-side stall defeats every existing defence
# =============================================================================


# The lane runs with --timeout=30 and timeout_method="thread", which would
# os._exit the worker running these deliberately-long subprocess tests. Both
# arms carry an explicit marker budget well above their measured cost.
@pytest.mark.slow
@pytest.mark.timeout(180)
def test_a_controller_side_stall_hangs_with_the_watchdog_disabled(tmp_path: Path) -> None:
    """RED arm: with the watchdog off, the session hangs and NOTHING kills it.

    ``--timeout=5`` is armed, ``--max-worker-restart=0`` is set, and there is no
    worker crash anywhere. The session still runs until something external ends
    it -- in CI that "something external" is the job cap, 30 minutes later.

    Here that "something external" is :func:`_run_guarded`: the hang is
    observed for ``HANG_PROOF_SECONDS``, then the whole nested process group
    is SIGKILLed and proven gone before ``TimeoutExpired`` reaches this test
    (#1842 -- the wedged controller used to outlive the test).
    """
    _write_nested_project(tmp_path)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        _run_nested(tmp_path, stall_seconds=0, timeout=HANG_PROOF_SECONDS)

    # Prove the run actually reached the wedge rather than merely being slow to
    # start: the controller must have announced the item it then hung on, which
    # is the exact last line of the real job's log.
    partial = _as_text(excinfo.value.stdout) + _as_text(excinfo.value.stderr)
    assert _WEDGED_TEST in partial, (
        "the nested session timed out before it dispatched the wedged item, so "
        f"this arm proves nothing about pytest-timeout.\n{partial[-3000:]}"
    )
    # ...and pytest-timeout, whose 5s cap expired four times over, left no
    # trace: no worker died, no stack was dumped.
    assert "node down" not in partial, partial[-3000:]
    assert "Timeout (>" not in partial, partial[-3000:]


# =============================================================================
# The fix: the watchdog fails the session and names the culprit
# =============================================================================


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_the_watchdog_fails_a_stalled_session_and_names_what_was_in_flight(
    tmp_path: Path,
) -> None:
    """GREEN arm: same project, watchdog armed -> fast failure with a diagnosis."""
    _write_nested_project(tmp_path)

    started = time.monotonic()
    result = _run_nested(
        tmp_path, stall_seconds=NESTED_STALL_SECONDS, timeout=NESTED_BUDGET_SECONDS
    )
    elapsed = time.monotonic() - started
    output = _as_text(result.stdout) + _as_text(result.stderr)

    assert result.returncode == STALL_EXIT_CODE, (
        f"expected the watchdog's exit code {STALL_EXIT_CODE}, got "
        f"{result.returncode} after {elapsed:.1f}s.\n{output[-4000:]}"
    )
    assert elapsed < NESTED_BUDGET_SECONDS, "watchdog did not end the session in budget"

    assert BANNER in output, f"stall report missing its marker.\n{output[-4000:]}"
    assert "SESSION STALLED" in output, output[-4000:]

    # The whole point: the report must name the item that was dispatched and
    # never reported. A watchdog that only caps the wall clock was rejected in
    # #1648 precisely because it cannot do this.
    assert _WEDGED_TEST in output, (
        f"stall report does not name the in-flight item {_WEDGED_TEST!r}.\n{output[-4000:]}"
    )
    assert "never reported" in output, output[-4000:]

    # ...and the stack of the thread that is actually stuck, so the next reader
    # does not have to re-derive the mechanism from a log with no error in it.
    assert "controller thread stacks" in output, output[-4000:]
    assert "pytest_runtest_logstart" in output, (
        f"thread dump does not show the blocked frame.\n{output[-4000:]}"
    )


# =============================================================================
# Wiring: the real tests/conftest.py must arm it, on the CONTROLLER only
# =============================================================================


# 180 not 300: this file must not be what sets the Unit Tests lane's per-test
# ceiling, since that ceiling is the input to the sizing guard below. 180 is
# still ~15x the measured 12s cost.
@pytest.mark.slow
@pytest.mark.timeout(180)
def test_the_repo_conftest_arms_the_watchdog_on_the_controller_only() -> None:
    """Without this, deleting the install call from ``tests/conftest.py`` would
    leave every other test in this file passing while CI loses the backstop.

    ``--collect-only`` is enough and costs ~12s: the watchdog is armed in
    ``pytest_configure``, long before any test runs. Exactly one banner proves
    both halves of the contract -- the controller arms it, and the two xdist
    workers (which have their own ``--timeout`` and legitimately idle for
    minutes at the tail of a session) do not.
    """
    env = dict(os.environ)
    env[ENV_TIMEOUT] = "600"
    env.pop("PYTEST_ADDOPTS", None)

    result = _run_guarded(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_tests_meta/test_no_hardcoded_home_paths.py",
            "-n",
            "2",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=env,
        timeout=240,
    )
    output = _as_text(result.stdout) + _as_text(result.stderr)

    assert output.count(BANNER) == 1, (
        f"expected exactly one {BANNER} arming line (controller only), got "
        f"{output.count(BANNER)}.\n{output[-3000:]}"
    )
    assert "armed" in output, output[-3000:]


# =============================================================================
# Enablement contract: opt-in per lane, never inferred
# =============================================================================


def test_watchdog_is_off_unless_a_lane_opts_in() -> None:
    assert resolve_timeout({}) == 0.0


def test_a_ci_environment_alone_does_not_arm_it() -> None:
    """Deliberate: there is no safe repo-wide default.

    A ``CI``-derived default would have armed the same window on every lane
    that loads ``tests/conftest.py``, including ``slow-tests.yml``, whose
    serial lanes run ``--timeout=900`` tests and one ``@pytest.mark.timeout``
    of 2700s. The controller emits nothing for the whole of such a test, so any
    window short enough to be useful on the Unit Tests lane red-Xs the nightly.
    """
    assert resolve_timeout({"CI": "true"}) == 0.0
    assert resolve_timeout({"GITHUB_ACTIONS": "true"}) == 0.0


def test_the_env_var_sets_the_window() -> None:
    assert resolve_timeout({ENV_TIMEOUT: "45"}) == 45.0


def test_explicit_zero_disables_the_watchdog() -> None:
    """An escape hatch is required: a wedge in the watchdog itself must be
    switchable off from the workflow without a code change."""
    assert resolve_timeout({ENV_TIMEOUT: "0"}) == 0.0
    assert resolve_timeout({ENV_TIMEOUT: "-1"}) == 0.0


def test_unparseable_value_disables_rather_than_crashing_the_session() -> None:
    assert resolve_timeout({ENV_TIMEOUT: "later"}) == 0.0
    assert resolve_timeout({ENV_TIMEOUT: "   "}) == 0.0


# =============================================================================
# Sizing: every lane that opts in must clear its OWN longest per-test budget
# =============================================================================
#
# This is the guard that matters. An inactivity window is not a global
# constant: the controller is silent for the entire duration of a single test,
# so the window has to sit above the largest budget any test in that lane may
# consume -- the larger of the lane's ``--timeout`` and any
# ``@pytest.mark.timeout`` on a test it collects. Those differ by an order of
# magnitude across this repo (30s to 2700s), so enabling the watchdog on a new
# lane without re-sizing must fail HERE rather than in a nightly.

WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

#: Full extra budget of slack. A test cannot exceed its own budget silently --
#: pytest-timeout kills the worker at the cap and xdist reports the crashed
#: test, which is itself an activity event (measured: 11s from ``node down`` to
#: the FAILED line in job 95087170617).
REQUIRED_HEADROOM = 2.0

#: Floor covering the non-test quiet phases: the worst measured startup across
#: the six pytest jobs of run 31915535379 is 103.7s (Unit Tests), the worst
#: teardown 10.9s.
MIN_WINDOW_SECONDS = 300.0

#: Lanes this change enables. Pinned so a future edit that silently drops the
#: variable is caught, not just one that mis-sizes it.
EXPECTED_ENABLED_JOBS = {"unit-tests", "heavy-unit-tests", "agents-tests"}

_LANE_TIMEOUT = re.compile(r"--timeout[= ](\d+)")


def _timeout_marker_value(call: ast.expr) -> float | None:
    """Seconds from a ``pytest.mark.timeout(...)`` call node, if literal.

    Covers both spellings pytest-timeout accepts: the positional
    ``timeout(300)`` and the keyword ``timeout(timeout=300)``.
    """
    if not isinstance(call, ast.Call):
        return None
    try:
        if ast.unparse(call.func) != "pytest.mark.timeout":
            return None
    except Exception:  # pragma: no cover - defensive
        return None
    candidates = list(call.args) + [kw.value for kw in call.keywords if kw.arg == "timeout"]
    for arg in candidates:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
            return float(arg.value)
    return None


def _marker_timeouts(source: str) -> list[float]:
    """Every literal ``pytest.mark.timeout`` budget declared in a module.

    Parsed rather than grepped: this very file quotes
    ``@pytest.mark.timeout(2700)`` in prose, and a substring scan reads that as
    a real 45-minute budget on the Unit Tests lane.

    Covers decorators and module-level ``pytestmark``. Markers whose argument
    is not a literal, and markers applied from a conftest via
    ``pytest_collection_modifyitems``, stay invisible here -- ``install()``'s
    runtime refusal against the session ``--timeout`` is the backstop for those.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[float] = []

    def collect(node: ast.expr) -> None:
        value = _timeout_marker_value(node)
        if value is not None:
            found.append(value)

    for node in ast.walk(tree):
        for decorator in getattr(node, "decorator_list", []):
            collect(decorator)
        # ``pytestmark = pytest.mark.timeout(900)`` / ``= [..., ...]`` applies
        # to every test in the module and carries no decorator.
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets):
                continue
            value = node.value
            if value is None:
                continue
            if isinstance(value, (ast.List, ast.Tuple)):
                for element in value.elts:
                    collect(element)
            else:
                collect(value)
    return found


def _jobs_with_pytest_steps() -> list[tuple[str, str, dict, str]]:
    """Yield ``(workflow, job_name, job, joined-run-scripts)`` for pytest jobs."""
    found = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        try:
            workflow = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue  # test_workflows_yaml_parse.py owns parse failures
        if not isinstance(workflow, dict):
            continue
        for job_name, job in (workflow.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            scripts = [
                step["run"]
                for step in (job.get("steps") or [])
                if isinstance(step, dict) and isinstance(step.get("run"), str)
            ]
            joined = "\n".join(scripts)
            if re.search(r"(^|\s)pytest\s", joined):
                found.append((path.name, job_name, job, joined))
    return found


def _collection_roots_and_ignores(script: str) -> tuple[list[Path], set[Path]]:
    """Split a lane's run script into collection roots and ``--ignore`` paths.

    Tokenised rather than substring-matched. A substring test conflates the
    two: the Unit Tests lane collects ``tests/unit/test_api/`` *and* ignores
    ``tests/unit/test_api/test_episodic_memory_bridge.py``, and
    ``"--ignore=tests/unit/test_api/" in script`` is true because the ignored
    file's path starts with the directory's -- which silently dropped the whole
    directory from the ceiling scan.
    """
    try:
        tokens = shlex.split(script, comments=True)
    except ValueError:  # pragma: no cover - unbalanced quotes in some other step
        tokens = script.split()

    ignores: set[Path] = set()
    roots: list[Path] = []
    for index, token in enumerate(tokens):
        if token.startswith("--ignore="):
            value = token.split("=", 1)[1]
        elif token == "--ignore" and index + 1 < len(tokens):
            value = tokens[index + 1]
        else:
            continue
        ignores.add((REPO_ROOT / value.rstrip("/")).resolve())

    for index, token in enumerate(tokens):
        if not token.startswith("tests/"):
            continue
        if index and tokens[index - 1] == "--ignore":
            continue
        candidate = (REPO_ROOT / token.rstrip("/")).resolve()
        if candidate.exists() and candidate not in ignores:
            roots.append(candidate)
    return roots, ignores


def _largest_per_test_budget(script: str) -> tuple[float, str]:
    """Longest a single test in this lane may run, with what sets it."""
    best = 0.0
    why = "no --timeout and no marker found"
    lane_timeout = _LANE_TIMEOUT.search(script)
    if lane_timeout:
        best = float(lane_timeout.group(1))
        why = f"lane --timeout={lane_timeout.group(1)}"
    roots, ignores = _collection_roots_and_ignores(script)
    for base in roots:
        files = [base] if base.is_file() else base.rglob("*.py")
        for file in files:
            if file.resolve() in ignores:
                continue
            for value in _marker_timeouts(file.read_text(errors="replace")):
                if value > best:
                    best = value
                    why = f"pytest.mark.timeout({int(value)}) in {file.relative_to(REPO_ROOT)}"
    return best, why


def test_every_lane_that_opts_in_is_sized_above_its_own_longest_test() -> None:
    """The whole safety argument, checked against the workflow files."""
    offenders = []
    enabled = set()
    for workflow, job_name, job, script in _jobs_with_pytest_steps():
        raw = (job.get("env") or {}).get(ENV_TIMEOUT)
        if raw is None:
            continue
        window = resolve_timeout({ENV_TIMEOUT: str(raw)})
        if window <= 0:
            continue  # explicitly disabled; nothing to size
        enabled.add(job_name)

        budget, why = _largest_per_test_budget(script)
        if window < REQUIRED_HEADROOM * budget:
            offenders.append(
                f"{workflow}::{job_name} sets {ENV_TIMEOUT}={window:g}s but its "
                f"longest per-test budget is {budget:g}s ({why}); need "
                f">= {REQUIRED_HEADROOM:g}x that"
            )
        if window < MIN_WINDOW_SECONDS:
            offenders.append(
                f"{workflow}::{job_name} sets {ENV_TIMEOUT}={window:g}s, below "
                f"the {MIN_WINDOW_SECONDS:g}s floor that covers a lane's "
                "startup and teardown quiet phases"
            )

    assert not offenders, (
        "the session stall watchdog is mis-sized on some lane, which would "
        "fail a HEALTHY run:\n  " + "\n  ".join(offenders)
    )
    assert enabled == EXPECTED_ENABLED_JOBS, (
        f"lanes with {ENV_TIMEOUT} changed: expected {sorted(EXPECTED_ENABLED_JOBS)}, "
        f"found {sorted(enabled)}. Adding a lane is fine -- size it against that "
        "lane's longest per-test budget and update this set."
    )


def test_install_refuses_a_window_that_cannot_clear_one_test(tmp_path: Path) -> None:
    """Runtime backstop for the same property, for lanes this file cannot see.

    A window at or below the session's own ``--timeout`` is guaranteed to fire
    on a healthy long test, so the watchdog declines to arm and says why rather
    than red-Xing the run.
    """
    (tmp_path / "pytest.ini").write_text(
        textwrap.dedent(
            """
            [pytest]
            timeout = 120
            timeout_method = thread
            """
        )
    )
    (tmp_path / "test_trivial.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            f"""
            import sys

            sys.path.insert(0, {str(REPO_ROOT)!r})

            from tests.stall_watchdog import install


            def pytest_configure(config):
                install(config)
            """
        )
    )

    # 60 <= the 120s per-test timeout
    result = _run_nested(tmp_path, stall_seconds=60, timeout=120)
    output = _as_text(result.stdout) + _as_text(result.stderr)

    assert result.returncode == 0, output[-3000:]
    assert f"{BANNER}: NOT armed" in output, output[-3000:]
    assert "the session fails if the controller" not in output, output[-3000:]


# =============================================================================
# #1842: the RED arm's wedged session must not outlive the test -- ever
# =============================================================================


def test_the_group_scanner_sees_a_live_process() -> None:
    """Positive control for the survivor guard.

    ``_end_nested_session`` passes when :func:`_group_members` returns nothing.
    That has to mean "nothing is alive", never "the scan is broken" (``/proc``
    layout drift, a parse slip) -- so the scanner must at least see us.
    """
    if not Path("/proc/self/stat").exists():
        pytest.skip("the /proc-based scan is Linux-only")
    members = _group_members(os.getpgid(0))
    assert any(line.startswith(f"pid {os.getpid()} ") for line in members), members


# A stand-in for the outer test process: spawn the wedged session through the
# real helper, report its pid, then block in ``communicate()`` exactly where
# the RED arm blocks. The test SIGKILLs this process while it sits there.
_STAND_IN_PARENT = textwrap.dedent(
    """
    import sys

    sys.path.insert(0, {repo_root!r})

    from tests.unit.test_tests_meta.test_session_stall_watchdog_1655 import (
        _nested_env,
        _spawn_nested,
    )

    nested = _spawn_nested(
        [sys.executable, "-m", "pytest", {root!r}, "-p", "no:cacheprovider"],
        cwd={root!r},
        env=_nested_env(0),
    )
    print(nested.pid, flush=True)
    nested.communicate(timeout={budget})
    """
)


@pytest.mark.timeout(60)
def test_the_nested_session_dies_when_its_parent_is_hard_killed(tmp_path: Path) -> None:
    """#1842: the leak path that no ``finally`` can cover.

    The orphan on the droplet was the RED arm's nested *controller*: the outer
    test process was killed while blocked in ``communicate()``, so
    ``subprocess.run``'s ``except TimeoutExpired: process.kill()`` never ran,
    and nothing kernel-side tied the child to its parent. It reparented to
    init and kept spinning in ``pbkdf2_hmac`` -- 22 hours at 100% of a core.

    :func:`_spawn_nested` closes that with ``PR_SET_PDEATHSIG``. This test is
    the scenario itself: spawn the wedged session through the same helper from
    a stand-in parent, SIGKILL that parent (the pid only, not its group), and
    require the whole nested session -- controller and both workers -- to be
    gone. Remove the wrapper's ``prctl`` call and this fails, listing the
    survivors.
    """
    if not _PDEATHSIG_SUPPORTED:
        pytest.skip("PR_SET_PDEATHSIG is Linux-only")
    _write_nested_project(tmp_path)
    script = _STAND_IN_PARENT.format(
        repo_root=str(REPO_ROOT), root=str(tmp_path), budget=NESTED_BUDGET_SECONDS
    )

    nested_pgid = 0
    with _spawn_nested([sys.executable, "-c", script], cwd=tmp_path, env=_nested_env(0)) as parent:
        try:
            assert parent.stdout is not None
            first_line = parent.stdout.readline().strip()
            if not first_line.isdigit():
                _, stderr = parent.communicate(timeout=REAP_GRACE_SECONDS)
                pytest.fail(
                    f"stand-in parent did not report the nested pid: {first_line!r}\n{stderr}"
                )
            nested_pgid = int(first_line)

            # Positive control before killing anything: the scanner must see the
            # nested controller AND its two workers, i.e. the session came up.
            deadline = time.monotonic() + NESTED_BUDGET_SECONDS
            while len(_group_members(nested_pgid)) < 3 and time.monotonic() < deadline:
                time.sleep(0.1)
            before = _group_members(nested_pgid)
            assert len(before) >= 3, "nested session never came up:\n  " + "\n  ".join(before)
            # Let the controller reach the wedge, so the kill lands on the
            # observed shape (a controller spinning in native code).
            time.sleep(2)

            # The hard death. ``kill()`` signals this one pid, not its group --
            # exactly what took the outer test process down on 2026-08-29.
            parent.kill()
            parent.wait()

            survivors = _drain_group(nested_pgid)
            assert not survivors, (
                f"{len(survivors)} process(es) of the nested session (pgid {nested_pgid}) "
                "outlived their hard-killed parent -- PR_SET_PDEATHSIG is not in effect "
                "(#1842):\n  " + "\n  ".join(survivors)
            )
        finally:
            # Whatever happened above, leave nothing behind.
            if nested_pgid:
                _kill_group(nested_pgid)
            _kill_group(parent.pid)
