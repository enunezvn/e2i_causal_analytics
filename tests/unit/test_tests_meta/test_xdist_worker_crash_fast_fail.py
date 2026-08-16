"""#1648 guard: a crashed xdist worker must fail the lane FAST, not burn the cap.

The failure this guards against
-------------------------------
``pyproject.toml`` sets ``timeout_method = "thread"``. That method never tries
to interrupt the running call -- :func:`pytest_timeout.timeout_timer` dumps the
stacks and then calls ``os._exit(1)`` on the xdist worker, unconditionally. So
*any* test that outlives its lane's ``--timeout`` kills its worker outright.

xdist does not stop there. ``xdist.dsession.get_default_max_worker_restart``
returns ``numprocesses * 4`` when ``--max-worker-restart`` is not given -- **8**
on the ``-n 2`` lanes -- so the session carries on past the kill, and that has
cost a full job cap in two different ways:

* The replacement can be handed the very test that killed its predecessor.
  Because a timeout kill is deterministic it dies the same way, looping to the
  full budget: **9 executions** of one test, each costing a lane timeout plus a
  worker respawn. Measured against this repo's own config with a real 5s cap:
  9 kills / 130.7s versus 1 kill / 25.8s, and the retries recovered *zero*
  extra tests ("9 failed, 340 passed" either way).
* Or the session limps on and stalls somewhere else. That is what PR #1643's
  Unit Tests actually did -- ONE crash, ONE replacement, then 19.5 min of
  silence waiting on a single test that never reported, cancelled at the
  30-min cap.

Ending the session at the first crash covers both.

The fix is ``--max-worker-restart=0`` in ``pyproject.toml``'s ``addopts``. It
is set there rather than on each CI lane's command line because ``addopts``
already owns ``-n 4 --dist=loadscope`` and sits beside the
``timeout_method = "thread"`` setting that makes worker kills possible at all
-- one place to read, one place to change, and every current and future lane
(plus local runs) inherits it.

Rejected alternatives, all measured against a local repro:

* ``--timeout-method=signal`` -- does not enforce the cap on native calls. A
  ``hashlib.pbkdf2_hmac`` burn ran 9.74s against a 2.0s alarm because the
  handler is deferred until the C call returns, and a longer burn hung the
  whole session to the wall cap emitting *no* diagnostics at all -- strictly
  worse than today. This repo's real offenders are exactly that shape (#1548's
  ``dense_tree_shap`` held the GIL for a single 1240.5s C call).
* An xdist version bump -- there is no upstream bug. The ``numprocesses * 4``
  default is deliberate (xdist #226, "avoid workers from restarting endlessly
  due to crashing collections") and is in the shipped 3.8.0 source.
* A no-output watchdog -- caps the wall time but names no culprit, and the
  jobs' ``timeout-minutes`` already is one.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

# tests/unit/test_tests_meta/<this file> -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# The offending test is re-run once per restart, so the session cost is
# (restarts + 1) x (lane timeout + worker respawn). Only 0 collapses that to a
# single cycle.
REQUIRED_FLAG = "--max-worker-restart=0"

# Generous: the nested session measured 16.0s (35.4s including interpreter and
# plugin boot) with restarts disabled, against 99.8s with them enabled. Any
# value between comfortably separates the two.
MAX_NESTED_SESSION_SECONDS = 75.0

# Mimics pytest_timeout.timeout_timer's final act, which is literally
# ``os._exit(1)`` on the worker -- without having to burn a real timeout.
_CRASHING_SUITE = textwrap.dedent(
    """
    import os


    def test_runs_before_the_crash():
        assert True


    def test_kills_its_worker():
        # exactly what pytest_timeout.timeout_timer does when the thread-method
        # cap fires
        os._exit(1)


    def test_queued_behind_the_crash():
        assert True
    """
)


def _read_addopts() -> str:
    """Return the raw ``addopts`` string from pyproject's pytest config."""
    text = PYPROJECT.read_text()
    match = re.search(r"^addopts\s*=\s*\"([^\"]*)\"", text, re.MULTILINE)
    assert match, "pyproject.toml has no [tool.pytest.ini_options] addopts string"
    return match.group(1)


def test_pyproject_addopts_disables_worker_restart() -> None:
    """The SSOT: addopts must turn xdist's restart budget off (#1648)."""
    addopts = _read_addopts()
    assert REQUIRED_FLAG in addopts, (
        f"pyproject.toml addopts is {addopts!r} and does not contain "
        f"{REQUIRED_FLAG!r}. Without it xdist restarts a crashed worker "
        "numprocesses * 4 times (8 on the -n 2 CI lanes), re-running the test "
        "that killed it on every replacement. A single test over its lane's "
        "--timeout then costs 9 x (timeout + respawn) and cancels the job at "
        "its cap instead of failing it (#1648)."
    )


def _strip_shell_comments(script: str) -> str:
    """Drop ``#`` comments so a flag *described* in prose is not read as set.

    The lanes are heavily commented and several comments quote the very flags
    matched below -- feast-apply.yml's step explains that it clears ``addopts``
    to "drop pyproject's forced ``-n 4 --dist=loadscope``" while itself running
    no xdist at all.
    """
    lines = []
    for line in script.splitlines():
        stripped = re.sub(r"(^|\s)#.*$", "", line)
        if stripped.strip():
            lines.append(stripped)
    return "\n".join(lines)


def _pytest_run_blocks() -> list[tuple[str, str, str]]:
    """Yield ``(workflow, job, run-script)`` for every step that runs pytest.

    Scripts come back with shell comments removed.
    """
    blocks: list[tuple[str, str, str]] = []
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
            for step in job.get("steps") or []:
                script = step.get("run") if isinstance(step, dict) else None
                if not isinstance(script, str):
                    continue
                script = _strip_shell_comments(script)
                if re.search(r"(^|\s)pytest\s", script):
                    blocks.append((path.name, job_name, script))
    return blocks


def test_no_workflow_lane_reenables_worker_restart() -> None:
    """No CI lane may override the SSOT back to a restarting configuration.

    A command-line ``--max-worker-restart=N`` beats ``addopts``, and clearing
    ``addopts`` with ``-o addopts=`` drops the flag entirely. Either would
    reintroduce #1648 on that lane alone, silently.
    """
    offenders: list[str] = []
    for workflow, job, script in _pytest_run_blocks():
        # ``-n 0`` and ``-p no:xdist`` mean no workers, so nothing can crash.
        uses_xdist = bool(re.search(r"-n\s+(?:auto|[1-9]\d*)\b", script)) and (
            "-p no:xdist" not in script
        )
        if not uses_xdist:
            continue

        explicit = re.search(r"--max-worker-restart[= ](\d+)", script)
        if explicit and explicit.group(1) != "0":
            offenders.append(f"{workflow}::{job} passes --max-worker-restart={explicit.group(1)}")
        elif not explicit and re.search(r'-o\s+"?addopts=', script):
            offenders.append(f"{workflow}::{job} clears addopts while running xdist")

    assert not offenders, (
        "CI lanes run xdist without the #1648 fast-fail guarantee: "
        + "; ".join(offenders)
        + f". Either drop the override so the lane inherits {REQUIRED_FLAG} from "
        "pyproject's addopts, or pass --max-worker-restart=0 explicitly. A "
        "nonzero budget makes a timeout-killed worker re-run the offending test "
        "on every replacement until the job's cap cancels it."
    )


@pytest.mark.timeout(180)
def test_crashed_worker_fails_session_without_restart_storm(tmp_path: Path) -> None:
    """A worker that dies mid-test must end the session on the FIRST death.

    Runs a nested pytest against this repo's real ``pyproject.toml`` -- so the
    shipped ``addopts`` is what is under test, not a string this test builds --
    with a suite whose middle test calls ``os._exit(1)``.
    """
    pytest.importorskip("xdist", reason="pytest-xdist is what this guards")

    suite = tmp_path / "test_worker_crash_probe.py"
    suite.write_text(_CRASHING_SUITE)

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(suite),
        # Use the repo's real config so addopts is the thing being verified.
        "-c",
        str(PYPROJECT),
        "-p",
        "no:cacheprovider",
        # Mirror the four backend-tests lanes.
        "-n",
        "2",
        "--dist=loadscope",
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=MAX_NESTED_SESSION_SECONDS * 3,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    context = f"\n--- nested pytest ({elapsed:.1f}s, rc={completed.returncode}) ---\n{output}"

    assert completed.returncode != 0, "a crashed worker must fail the session" + context

    restarts = output.count("replacing crashed worker")
    assert restarts == 0, (
        f"xdist restarted the crashed worker {restarts} time(s); each restart "
        "re-runs the test that killed it, so a deterministic timeout kill burns "
        "(restarts + 1) lane timeouts and cancels the job at its cap (#1648)." + context
    )

    assert "worker restarting disabled" in output, (
        "expected xdist's 'worker restarting disabled' notice, which is what "
        "makes the crash a fast, legible failure" + context
    )

    assert "test_kills_its_worker" in output, (
        "the offending test must be named in the output -- identifying the "
        "culprit is the half of #1648 that cost the most time" + context
    )

    assert elapsed < MAX_NESTED_SESSION_SECONDS, (
        f"nested session took {elapsed:.1f}s (limit {MAX_NESTED_SESSION_SECONDS}s); "
        "a crashed worker should end the run promptly rather than working "
        "through a restart budget" + context
    )
