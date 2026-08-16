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
"""

from __future__ import annotations

import ast
import os
import re
import shlex
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


def _run_nested(
    root: Path, *, stall_seconds: int | str, timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(root), "-p", "no:cacheprovider"],
        cwd=str(root),
        env=_nested_env(stall_seconds),
        capture_output=True,
        text=True,
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

    result = subprocess.run(
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
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
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

    env = _nested_env(60)  # 60 <= the 120s per-test timeout
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-p", "no:cacheprovider"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = _as_text(result.stdout) + _as_text(result.stderr)

    assert result.returncode == 0, output[-3000:]
    assert f"{BANNER}: NOT armed" in output, output[-3000:]
    assert "the session fails if the controller" not in output, output[-3000:]
