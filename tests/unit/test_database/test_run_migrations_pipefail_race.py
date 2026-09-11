"""scripts/run_migrations.sh must not decide "applied?" or "un-wrap?" through ``… | grep -q``.

The runner runs under ``set -o pipefail``. ``grep -q`` exits at its first match; when the
command feeding it has not finished writing, that command dies of SIGPIPE, the pipeline's
status is 141, and the ``if`` takes the "no match" branch:

* ``echo "$APPLIED" | grep -qxF "$key"`` then treats an applied migration as pending and
  re-applies it. Measured 2026-09-11 on the droplet with the real 212-row ledger: 21 misses in
  3,000 lookups of the first key, and the learning-loop runner test re-applied random early
  migrations (043, 056, 064) on a fully recorded copy of prod.
* ``sed … "$migration_file" | grep -qiE <unwrap detector>`` then wraps a file that must run
  un-wrapped (an enum value addition or CONCURRENTLY) in ``--single-transaction``.

Both are checked here without a database: ``psql`` is a stub that reports a ledger holding
every file and logs how each file would have been applied. Each test first shows the old
pipeline losing the race on the same input (negative control), with inputs far larger than any
pipe buffer plus grep's read buffer, so the writer is always still writing when grep -q exits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Each stubbed runner invocation spawns a grep per migration file.
pytestmark = pytest.mark.timeout(300)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run_migrations.sh"

PSQL_STUB = """#!/bin/bash
# Stub psql: the tracking SELECT returns $LEDGER; an apply logs its flags; everything else is silent.
args="$*"
if [[ "$args" == *"SELECT filename FROM public.schema_migrations"* ]]; then
  cat "$LEDGER"
elif [[ "$args" == *"-v ON_ERROR_STOP=1"* ]]; then
  cat > /dev/null
  echo "$args" >> "$APPLY_LOG"
else
  cat > /dev/null 2>&1 || true
fi
"""


OLD_LOOKUP = (
    'set -o pipefail; APPLIED=$(cat "$LEDGER_FILE"); '
    'if echo "$APPLIED" | grep -qxF "$KEY"; then echo found; else echo missed; fi'
)
OLD_DETECTOR = (
    "set -o pipefail; if sed 's/--.*$//' \"$FILE\" | grep -qiE "
    '"ALTER[[:space:]]+TYPE[[:space:]].*ADD[[:space:]]+VALUE|CONCURRENTLY|^[[:space:]]*COMMIT[[:space:]]*;"; '
    "then echo unwrap; else echo wrap; fi"
)
MIB = 1024 * 1024


def _fake_repo(tmp_path: Path, names) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(RUNNER, repo / "scripts" / "run_migrations.sh")
    migrations = repo / "database" / "migrations"
    migrations.mkdir(parents=True)
    for name in names:
        (migrations / name).write_text("SELECT 1;\n")
    return repo


def _env(tmp_path: Path, ledger: Path, apply_log: Path) -> dict:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "psql"
    stub.write_text(PSQL_STUB)
    stub.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "SUPABASE_DB_URL": "postgresql://stub@127.0.0.1:1/stub",
        "LEDGER": str(ledger),
        "APPLY_LOG": str(apply_log),
    }


def _bash(script: str, **env: str) -> str:
    return (
        subprocess.run(
            ["bash", "-c", script],
            env={**os.environ, **env},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
        )
        .stdout.decode()
        .strip()
    )


def test_applied_lookup_never_misses_a_recorded_migration(tmp_path):
    names = [f"00{i}_real.sql" for i in range(5)]
    repo = _fake_repo(tmp_path, names)
    ledger = tmp_path / "ledger.txt"
    # The recorded files come first (grep -q matches at once), then 4 MiB of other keys.
    padding = "".join(f"padding_{i:07d}_{'x' * 40}.sql\n" for i in range(4 * MIB // 60))
    ledger.write_text("".join(f"{n}\n" for n in names) + padding)
    assert ledger.stat().st_size > 4 * MIB - 1024

    control = _bash(OLD_LOOKUP, LEDGER_FILE=str(ledger), KEY=names[0])
    assert control == "missed", "negative control: the old pipeline should lose the race here"

    env = _env(tmp_path, ledger, tmp_path / "apply.log")
    for run in range(3):
        proc = subprocess.run(
            ["bash", str(repo / "scripts" / "run_migrations.sh"), "--dry-run"],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
        )
        out = proc.stdout.decode()
        assert proc.returncode == 0, proc.stderr.decode()
        assert "PENDING" not in out and "No pending migrations." in out, (run, out)


def test_unwrap_detector_never_misses_on_a_large_file(tmp_path):
    repo = _fake_repo(tmp_path, [])
    big = repo / "database" / "migrations" / "001_enum_then_bulk.sql"
    # The match is on the first line; 4 MiB of comment-free statements follow.
    line = "SELECT 'padding padding padding padding padding padding padding';\n"
    big.write_text(
        "ALTER TYPE some_enum ADD VALUE IF NOT EXISTS 'X';\n" + line * (4 * MIB // len(line))
    )
    control = _bash(OLD_DETECTOR, FILE=str(big))
    assert control == "wrap", "negative control: the old pipeline should lose the race here"

    ledger = tmp_path / "ledger.txt"
    ledger.write_text("")
    apply_log = tmp_path / "apply.log"
    env = _env(tmp_path, ledger, apply_log)
    for _ in range(3):
        proc = subprocess.run(
            ["bash", str(repo / "scripts" / "run_migrations.sh")],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr.decode()
    applies = apply_log.read_text().splitlines()
    assert len(applies) == 3
    assert not [a for a in applies if "--single-transaction" in a], applies
