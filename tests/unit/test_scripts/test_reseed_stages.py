"""#1577: reseed wrapper stage resilience — real-bash tests, no mocks.

Every weekly Mon-3AM cron run 2026-07-06..2026-08-10 died at stage 1: the
loader exits 1 on partial feature_values failures (28-459 rows/week rejected
by the DB ``valid_event_timestamp`` CHECK) and ``scripts/reseed_synthetic.sh``
ran all five stages under bare ``set -euo pipefail`` — so the kpi_history
backfill, weekly capture, goldstd retrain, and A/B refresh stages NEVER
executed via cron (zero stage markers in ~/logs/e2i-reseed.log, no run ever
printed the done line).

The fix factors a stage runner into ``scripts/lib/reseed_stages.sh``: a failed
stage prints a FAILED marker, later stages still run, the final done line is
always reached with a status summary, and the script's exit code is nonzero
iff any stage failed (honest aggregate — the loader's own exit-1-on-partial-
failure semantics are deliberately unchanged).

These tests drive REAL bash subprocesses: the lib is sourced into throwaway
harness scripts with fake stage commands (true/false/exit N), and the REAL
wrapper runs end-to-end inside a fake project tree whose ``.venv/bin/dotenv``
/ ``.venv/bin/python`` / ``retrain_goldstd.sh`` are argument-echoing stubs
with scriptable exit codes — the exact cron failure scenario, replayed.
"""

import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LIB = REPO_ROOT / "scripts" / "lib" / "reseed_stages.sh"
WRAPPER = REPO_ROOT / "scripts" / "reseed_synthetic.sh"


def _run_lib_harness(body: str) -> subprocess.CompletedProcess:
    """Source the real lib under the wrapper's exact shell options and run
    ``body`` — real bash, fake stage commands."""
    script = f"set -euo pipefail\nsource {shlex.quote(str(LIB))}\n{body}\n"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)


# ---------------------------------------------------------------------------
# Stage-runner lib contract
# ---------------------------------------------------------------------------


class TestStageRunnerLib:
    def test_lib_exists_and_parses(self):
        assert LIB.is_file(), "scripts/lib/reseed_stages.sh missing"
        subprocess.run(["bash", "-n", str(LIB)], check=True)

    def test_all_stages_attempted_after_early_failure(self):
        """The #1577 truncation scenario: stage 1 fails, stages 2 and 3 must
        still run, and the aggregate exit code must stay nonzero."""
        proc = _run_lib_harness(
            """
            # external process exiting 3 — the same shape as the real loader
            reseed_run_stage "loader" bash -c 'echo LOADER_RAN; exit 3'
            reseed_run_stage "kpi_history backfill" echo BACKFILL_RAN
            reseed_run_stage "goldstd retrain" echo RETRAIN_RAN
            reseed_finish "reseed_synthetic"
            """
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        out = proc.stdout
        assert "LOADER_RAN" in out
        assert "=== loader FAILED (exit 3)" in out
        assert "BACKFILL_RAN" in out
        assert "=== kpi_history backfill done" in out
        assert "RETRAIN_RAN" in out
        assert "=== goldstd retrain done" in out
        assert "=== reseed_synthetic done" in out
        assert "FAILED stages: loader" in out

    def test_all_green_exits_zero_with_ok_summary(self):
        proc = _run_lib_harness(
            """
            reseed_run_stage "loader" true
            reseed_run_stage "kpi_history backfill" true
            reseed_finish "reseed_synthetic"
            """
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "=== loader start" in proc.stdout
        assert "=== loader done" in proc.stdout
        assert "=== reseed_synthetic done" in proc.stdout
        assert "(all stages OK)" in proc.stdout
        assert "FAILED" not in proc.stdout

    def test_multiple_failures_all_listed(self):
        proc = _run_lib_harness(
            """
            reseed_run_stage "loader" false
            reseed_run_stage "weekly capture" false
            reseed_run_stage "goldstd retrain" true
            reseed_finish "reseed_synthetic"
            """
        )
        assert proc.returncode == 1
        assert "FAILED stages: loader weekly capture" in proc.stdout

    def test_done_line_is_last_line(self):
        """The done line must be the final observable event — nothing may run
        after the aggregate verdict."""
        proc = _run_lib_harness(
            """
            reseed_run_stage "loader" false
            reseed_finish "reseed_synthetic"
            """
        )
        last = proc.stdout.strip().splitlines()[-1]
        assert last.startswith("=== reseed_synthetic done")

    def test_stage_command_receives_arguments(self):
        """Stages run as plain argv — the wrapper's "$@" forwarding rides
        this."""
        proc = _run_lib_harness(
            """
            reseed_run_stage "argsy" echo hello world
            reseed_finish "reseed_synthetic"
            """
        )
        assert proc.returncode == 0
        assert "hello world" in proc.stdout


# ---------------------------------------------------------------------------
# Real-wrapper end-to-end in a fake project tree
# ---------------------------------------------------------------------------

FAKE_DOTENV = """#!/bin/bash
# fake .venv/bin/dotenv: swallow "-f <file> run --", exec the wrapped command
while [[ $# -gt 0 && "$1" != "--" ]]; do shift; done
shift
exec "$@"
"""

FAKE_PYTHON = """#!/bin/bash
# fake .venv/bin/python: echo argv (marker for assertions), exit per env knob
echo "FAKE_PYTHON: $*"
case "$*" in
  *"--append-frontier"*|*"--anchor-to-now"*) exit "${FAKE_LOADER_EXIT:-0}" ;;
  *"--refresh-ab"*) exit "${FAKE_AB_EXIT:-0}" ;;
  *"src.kpi.history_backfill"*) exit "${FAKE_BACKFILL_EXIT:-0}" ;;
  *"src.kpi.history_capture"*) exit "${FAKE_CAPTURE_EXIT:-0}" ;;
esac
exit 0
"""

FAKE_RETRAIN = """#!/bin/bash
echo "FAKE_RETRAIN ran"
exit "${FAKE_RETRAIN_EXIT:-0}"
"""


@pytest.fixture()
def fake_tree(tmp_path: Path) -> Path:
    """Copy the REAL wrapper + lib into a throwaway project root whose venv
    binaries and retrain script are stubs (fake stage commands — the system
    under test, the wrapper's control flow, runs for real)."""
    scripts = tmp_path / "scripts"
    (scripts / "lib").mkdir(parents=True)
    shutil.copy(WRAPPER, scripts / "reseed_synthetic.sh")
    shutil.copy(LIB, scripts / "lib" / "reseed_stages.sh")

    def _write_exec(path: Path, content: str) -> None:
        path.write_text(content)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _write_exec(scripts / "retrain_goldstd.sh", FAKE_RETRAIN)
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    _write_exec(venv_bin / "dotenv", FAKE_DOTENV)
    _write_exec(venv_bin / "python", FAKE_PYTHON)
    (scripts / "reseed_synthetic.sh").chmod(0o755)
    return tmp_path


def _run_wrapper(tree: Path, *args: str, **env_overrides: str) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_overrides}
    return subprocess.run(
        [str(tree / "scripts" / "reseed_synthetic.sh"), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


class TestWrapperEndToEnd:
    def test_wrapper_syntax_ok(self):
        subprocess.run(["bash", "-n", str(WRAPPER)], check=True)

    def test_loader_failure_no_longer_truncates_the_run(self, fake_tree):
        """THE #1577 acceptance scenario: loader exits 1 (partial failure) —
        every downstream stage must still run, every marker must print, the
        done line must be reached, and the exit code must stay nonzero."""
        proc = _run_wrapper(fake_tree, FAKE_LOADER_EXIT="1")
        out = proc.stdout
        assert "FAILED (exit 1)" in out, out + proc.stderr
        assert "=== kpi_history backfill start" in out
        assert "=== kpi_history backfill done" in out
        assert "=== kpi_history weekly capture start" in out
        assert "=== kpi_history weekly capture done" in out
        assert "FAKE_RETRAIN ran" in out
        assert "=== A/B substrate refresh start" in out
        assert "=== A/B substrate refresh done" in out
        assert "=== reseed_synthetic done" in out
        assert "FAILED stages:" in out
        assert proc.returncode != 0, "partial failure must keep a nonzero aggregate exit"

    def test_all_green_run_exits_zero_and_reaches_done(self, fake_tree):
        proc = _run_wrapper(fake_tree)
        out = proc.stdout
        assert proc.returncode == 0, out + proc.stderr
        assert "=== reseed_synthetic start" in out
        assert "--append-frontier" in out  # loader invoked in append mode
        assert "=== kpi_history backfill done" in out
        assert "=== kpi_history weekly capture done" in out
        assert "FAKE_RETRAIN ran" in out
        assert "=== A/B substrate refresh done" in out
        assert "(all stages OK)" in out

    def test_mid_stage_failure_still_runs_later_stages(self, fake_tree):
        """Not only the loader: any stage failing must not truncate the rest
        (e.g. a backfill defect must not cost the week's capture point)."""
        proc = _run_wrapper(fake_tree, FAKE_BACKFILL_EXIT="1")
        out = proc.stdout
        assert "=== kpi_history weekly capture done" in out
        assert "FAKE_RETRAIN ran" in out
        assert "=== A/B substrate refresh done" in out
        assert "=== reseed_synthetic done" in out
        assert proc.returncode != 0

    def test_skip_retrain_consumed_not_forwarded(self, fake_tree):
        proc = _run_wrapper(fake_tree, "--skip-retrain")
        out = proc.stdout
        assert proc.returncode == 0, out + proc.stderr
        assert "goldstd retrain SKIPPED (--skip-retrain)" in out
        assert "FAKE_RETRAIN ran" not in out
        # never forwarded to load_synthetic_data.py (unknown arg would die)
        assert "--skip-retrain" not in [
            tok
            for line in out.splitlines()
            if line.startswith("FAKE_PYTHON")
            for tok in line.split()
        ]

    def test_extra_args_forwarded_to_loader_and_ab_stages(self, fake_tree):
        proc = _run_wrapper(fake_tree, "--dry-run")
        out = proc.stdout
        assert proc.returncode == 0, out + proc.stderr
        fake_lines = [line for line in out.splitlines() if line.startswith("FAKE_PYTHON")]
        loader_lines = [ln for ln in fake_lines if "--append-frontier" in ln]
        ab_lines = [ln for ln in fake_lines if "--refresh-ab" in ln]
        assert loader_lines and "--dry-run" in loader_lines[0]
        assert ab_lines and "--dry-run" in ab_lines[0]

    def test_full_mode_purges_capture_and_skips_ab_refresh(self, fake_tree):
        proc = _run_wrapper(fake_tree, "--full")
        out = proc.stdout
        assert proc.returncode == 0, out + proc.stderr
        assert "--anchor-to-now" in out  # legacy destructive reseed path
        assert "--append-frontier" not in out
        assert "history_capture --purge" in out  # stale captures purged first
        assert "--refresh-ab" not in out  # full generate path rebuilds AB itself
        assert "FAKE_RETRAIN ran" in out


class TestWrapperTextPins:
    """Environment gotchas the wrapper header documents must survive the
    stage-runner refactor (dotenv path, PYTHONPATH, LOKY, venv preflight)."""

    def test_wrapper_sources_the_stage_lib(self):
        text = WRAPPER.read_text()
        assert "lib/reseed_stages.sh" in text
        assert "reseed_finish" in text

    def test_env_gotchas_preserved(self):
        text = WRAPPER.read_text()
        assert "set -euo pipefail" in text
        assert ".venv/bin/dotenv -f .env run --" in text
        assert "LOKY_MAX_CPU_COUNT=1" in text
        assert 'PYTHONPATH="$PROJECT_ROOT"' in text
        # venv preflight still fails loud before any stage
        assert ".venv/bin/dotenv or .venv/bin/python missing" in text
