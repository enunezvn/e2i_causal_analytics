"""#2065 — the learning-loop real-DB suite gates every deploy, before migrations and the flip.

The suite in ``tests/unit/test_database/learning_loop/`` needs the droplet's database, and CI's
GitHub-hosted runners cannot reach it, so CI skips it. The deploy is the one automated place
that runs on the box. ``scripts/deploy/realdb_suite_gate.sh`` runs the suite there:

* **after** ``NEW_SHA`` is fixed and the #1785 image assertion, so it tests the tree being
  deployed;
* **before** ``run_migrations.sh``, so the migrations it rehearses (the derived pending list)
  have not been applied yet, and a red suite leaves the database untouched;
* **before** any pull or ``up``, so a red suite leaves every container on the previous sha.

A red suite, a timeout, a suite that ran nothing, or a box without the memory to run it all
FAIL the deploy with the existing "nothing was flipped or migrated" shape. There is no
fail-open path: skipping the gate on a busy box would ship a deploy the gate never saw.

What these tests can and cannot prove: they execute the SHIPPED text (the rollout script and the
gate script) against stubs, so they pin the control flow. They do not run the real suite, the
real docker or the real memory state; the first faithful exercise is the next real deploy.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tests.unit.test_docker.conftest import (
    REPO_ROOT,
    ROLLOUT_ID,
    bash_run,
    index_map,
    redirect_project_dir,
    ssh_script,
    step_with,
    write_stub_bin,
)
from tests.unit.test_docker.test_deploy_up_failure_rollback import _STUBS, NEW_SHA, PREV_SHA

GATE = REPO_ROOT / "scripts" / "deploy" / "realdb_suite_gate.sh"
GATE_CALL = "bash scripts/deploy/realdb_suite_gate.sh"
SUITE = "tests/unit/test_database/learning_loop/"
NOTHING_CHANGED = "nothing was flipped or migrated"


# --------------------------------------------------------------------------- #
# 1. Position in the rollout script
# --------------------------------------------------------------------------- #
def test_gate_runs_after_the_target_is_fixed_and_before_migrations_and_the_flip() -> None:
    script = ssh_script(ROLLOUT_ID)
    order = index_map(
        script,
        {
            "NEW_SHA resolved": "NEW_SHA=$(git rev-parse HEAD)",
            "#1785 manifest assertion": 'images_verdict "$NEW_SHA"',
            "#2065 real-DB suite gate": GATE_CALL,
            "DB migrations": "bash scripts/run_migrations.sh",
            "GHCR pull": "$COMPOSE_CMD pull api",
            # No textual "up -d" landmark: the first one in the text is inside the
            # rollback_to_prev() DEFINITION near the top, which is not where it executes. The
            # executed harness below proves no pull/up runs after a red gate.
        },
    )
    assert all(v >= 0 for v in order.values()), f"a landmark vanished: {order}"
    positions = list(order.values())
    assert positions == sorted(positions), (
        "#2065: the real-DB suite must run on the fixed target, BEFORE the migrations it "
        f"rehearses and before any container changes. Derived index map: {order}"
    )
    assert script.count(GATE_CALL) == 1, "the gate must run exactly once per rollout"


def test_gate_timeout_fits_inside_the_rollout_command_timeout() -> None:
    gate_minutes = int(re.search(r'^SUITE_TIMEOUT="(\d+)m"$', GATE.read_text(), re.M).group(1))
    rollout = step_with(ROLLOUT_ID, "command_timeout", what="the rollout's own budget")
    rollout_minutes = int(re.fullmatch(r"(\d+)m", rollout).group(1))
    # Measured 2026-09-17: the 14 latest successful rollouts took 6.0-10.5 min without the gate.
    slowest_rollout_without_gate = 11
    assert gate_minutes + slowest_rollout_without_gate < rollout_minutes, (
        f"gate timeout {gate_minutes}m + slowest measured rollout "
        f"{slowest_rollout_without_gate}m must stay under the {rollout}m command_timeout, or "
        "a slow suite turns into an SSH kill instead of a clean refusal"
    )


# --------------------------------------------------------------------------- #
# 2. Blocking: the SHIPPED rollout script, with the gate stubbed pass/fail
# --------------------------------------------------------------------------- #
def _rollout(tmp_path: Path, gate_rc: int) -> tuple[int, str, list[str]]:
    project_dir = tmp_path / "repo"
    (project_dir / "docker" / "frontend").mkdir(parents=True)
    (project_dir / "docker" / "frontend" / "Dockerfile").write_text(
        "FROM python:3.12-slim AS production\n"
    )
    (project_dir / "scripts" / "deploy").mkdir(parents=True)
    # Logging stubs: the order of these lines in CALL_LOG is the observable being asserted.
    (project_dir / "scripts" / "run_migrations.sh").write_text(
        '#!/usr/bin/env bash\necho "run_migrations" >> "$CALL_LOG"\nexit 0\n'
    )
    (project_dir / "scripts" / "deploy" / "realdb_suite_gate.sh").write_text(
        f'#!/usr/bin/env bash\necho "realdb_suite_gate" >> "$CALL_LOG"\nexit {gate_rc}\n'
    )
    (project_dir / "scripts" / "deploy" / "check_image_drift.py").write_text(
        "import sys\nsys.exit(0)\n"
    )
    script = redirect_project_dir(ssh_script(ROLLOUT_ID), project_dir)
    stub_bin = write_stub_bin(tmp_path / "stubbin", _STUBS)
    state = tmp_path / "state"
    state.mkdir()
    call_log = tmp_path / "calls.log"
    call_log.write_text("")
    env = {
        "PATH": f"{stub_bin}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "PREV_SHA": PREV_SHA,
        "NEW_SHA": NEW_SHA,
        "CALL_LOG": str(call_log),
        "STUB_STATE": str(state),
    }
    proc = bash_run(project_dir, script, set_e=False, env=env, name="_rollout.sh", timeout=60)
    return proc.returncode, proc.stdout + proc.stderr, call_log.read_text().splitlines()


def test_a_red_suite_fails_the_deploy_before_migrations_and_before_any_container_change(
    tmp_path: Path,
) -> None:
    rc, out, calls = _rollout(tmp_path, gate_rc=1)
    assert "realdb_suite_gate" in calls, "the gate never ran:\n" + "\n".join(calls)
    assert rc != 0, f"a red suite must fail the deploy; rc={rc}\n{out}"
    assert NOTHING_CHANGED in out, out
    assert "run_migrations" not in calls, "migrations ran after a red suite:\n" + "\n".join(calls)
    touched = [c for c in calls if " pull " in f" {c} " or "up -d" in c]
    assert touched == [], f"containers were touched after a red suite: {touched}"


def test_a_green_suite_lets_the_deploy_reach_migrations_then_the_flip(tmp_path: Path) -> None:
    """Positive control: the refusal above is the gate's doing, not a harness that never gets
    that far."""
    rc, out, calls = _rollout(tmp_path, gate_rc=0)
    assert rc == 0, f"rc={rc}\n{out}"
    gate, migrations = calls.index("realdb_suite_gate"), calls.index("run_migrations")
    first_up = next(i for i, c in enumerate(calls) if "up -d" in c)
    assert gate < migrations < first_up, calls


# --------------------------------------------------------------------------- #
# 3. The gate script's own decisions, executed against a fake project
# --------------------------------------------------------------------------- #
FAKE_PYTHON = r"""#!/usr/bin/env bash
# The venv interpreter, faked: `-m pytest` writes a JUnit file and exits FAKE_PYTEST_RC;
# `-m <_pg> --reap` is logged; anything else (the JUnit parse) runs the real python3.
if [ "$1" = "-m" ] && [ "$2" = "pytest" ]; then
  printf '%s\n' "$*" > "$GATE_LOG.argv"
  env > "$GATE_LOG.env"
  for a in "$@"; do case "$a" in --junitxml=*) junit="${a#--junitxml=}" ;; esac; done
  printf '<testsuites><testsuite name="pytest" errors="%s" failures="%s" skipped="%s" tests="%s"/></testsuites>' \
    "${FAKE_ERRORS:-0}" "${FAKE_FAILURES:-0}" "${FAKE_SKIPPED:-2}" "${FAKE_TESTS:-10}" > "$junit"
  exit "${FAKE_PYTEST_RC:-0}"
fi
if [ "$1" = "-m" ]; then echo "python $*" >> "$GATE_LOG.calls"; exit 0; fi
exec python3 "$@"
"""

TIMEOUT_STUB = """#!/usr/bin/env bash
echo "timeout $1" >> "$GATE_LOG.calls"
shift
"$@"
rc=$?
[ -n "${FAKE_TIMEOUT_RC:-}" ] && exit "$FAKE_TIMEOUT_RC"
exit $rc
"""


def _gate(tmp_path: Path, *, avail_mib: int = 6000, python: bool = True, **env: str):
    project = tmp_path / "project"
    (project / "scripts" / "deploy").mkdir(parents=True)
    (project / "scripts" / "deploy" / "realdb_suite_gate.sh").write_text(GATE.read_text())
    if python:
        (project / ".venv" / "bin").mkdir(parents=True)
        fake = project / ".venv" / "bin" / "python"
        fake.write_text(FAKE_PYTHON)
        fake.chmod(0o755)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal: 16375000 kB\nMemAvailable: {avail_mib * 1024} kB\n")
    stub_bin = write_stub_bin(tmp_path / "bin", {"timeout": TIMEOUT_STUB})
    log = tmp_path / "gate"
    full_env = {
        "PATH": f"{stub_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "GATE_LOG": str(log),
        "E2I_REALDB_GATE_MEMINFO": str(meminfo),
        # Set by the caller on purpose: the gate must not pass these through to the suite.
        "E2I_DB_SIMULATE_PENDING": "ml/044_composer_episodes_feedback_id.sql",
        "E2I_LIVE_LLM": "1",
        **env,
    }
    proc = bash_run(
        tmp_path,
        f'bash "{project}/scripts/deploy/realdb_suite_gate.sh"\n',
        set_e=False,
        env=full_env,
        name="run_gate.sh",
        timeout=60,
    )

    def read(suffix: str) -> str:
        path = Path(f"{log}.{suffix}")
        return path.read_text() if path.exists() else ""

    return proc.returncode, proc.stdout + proc.stderr, read


def test_a_green_suite_passes_and_runs_the_suite_opted_in_serially_and_unsimulated(tmp_path):
    rc, out, read = _gate(tmp_path)
    assert rc == 0, out
    argv = read("argv")
    assert "-m pytest -n 0 -p no:cacheprovider" in argv and SUITE in argv, argv
    env = dict(line.split("=", 1) for line in read("env").splitlines() if "=" in line)
    assert env.get("E2I_DB_INTEGRATION") == "1"
    assert "E2I_DB_SIMULATE_PENDING" not in env, "a rehearsal knob leaked into the deploy run"
    assert "E2I_LIVE_LLM" not in env, "the deploy must never spend on live LLM calls"
    assert "timeout 12m" in read("calls")
    assert "8 passed, 2 skipped" in out, out


@pytest.mark.parametrize(
    "fake, expected",
    [
        ({"FAKE_PYTEST_RC": "1", "FAKE_FAILURES": "3"}, "suite failed"),
        ({"FAKE_TIMEOUT_RC": "124"}, "timed out"),
        ({"FAKE_TESTS": "5", "FAKE_SKIPPED": "5"}, "ran nothing"),
    ],
)
def test_a_red_timed_out_or_empty_suite_fails_and_reaps_the_throwaway_containers(
    tmp_path, fake, expected
):
    rc, out, read = _gate(tmp_path, **fake)
    assert rc != 0, out
    assert expected in out, out
    if expected != "ran nothing":
        assert "-m tests.unit.test_database.learning_loop._pg --reap" in read("calls"), read(
            "calls"
        )


def test_too_little_memory_fails_closed_without_starting_the_suite(tmp_path):
    rc, out, read = _gate(tmp_path, avail_mib=1500)
    assert rc != 0, out
    assert "MemAvailable 1500 MiB" in out and "2048 MiB" in out, out
    assert read("argv") == "", "the suite started on a box without the memory for it"


def test_unreadable_memory_or_a_missing_interpreter_fails_closed(tmp_path):
    rc, out, _ = _gate(tmp_path / "a", E2I_REALDB_GATE_MEMINFO=str(tmp_path / "nope"))
    assert rc != 0 and "MemAvailable" in out, out
    rc, out, _ = _gate(tmp_path / "b", python=False)
    assert rc != 0 and ".venv/bin/python" in out, out


def test_the_gate_is_executable_bash_with_no_fail_open_switch() -> None:
    text = GATE.read_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    # The only environment it reads besides PATH/HOME is the meminfo test seam.
    read_vars = set(re.findall(r"\$\{?(E2I_[A-Z_]+)", text))
    assert read_vars == {"E2I_REALDB_GATE_MEMINFO"}, json.dumps(sorted(read_vars))
