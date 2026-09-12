"""Static checks on the lane's migration files (no database; runs in CI).

``scripts/run_migrations.sh`` wraps a file and its ledger row in ``--single-transaction`` unless
the file matches its un-wrap detector (``ALTER TYPE … ADD VALUE``, ``CONCURRENTLY`` or a bare
``COMMIT;``). ml/039 must take the un-wrapped branch (a new enum value cannot be used in the
transaction that adds it) and ml/040 / ml/041 the wrapped one, so neither can silently lose
atomicity. The real-runner replay is in ``test_migration_runner.py`` (opt-in).
"""

from __future__ import annotations

import re

import pytest

from tests.integration.test_migrations_no_inner_txn import _scan_for_bare_txn
from tests.unit.test_database.learning_loop import _pg

ML = _pg.REPO_ROOT / "database"
RUNNER = _pg.REPO_ROOT / "scripts" / "run_migrations.sh"


def test_runner_detector_is_the_one_the_fixture_mirrors():
    script = RUNNER.read_text()
    assert re.search(
        r"grep -qiE \\\s*\n\s*"
        + re.escape(
            '"ALTER[[:space:]]+TYPE[[:space:]].*ADD[[:space:]]+VALUE|CONCURRENTLY|'
            '^[[:space:]]*COMMIT[[:space:]]*;"'
        )
        + r" \\\s*\n\s*"
        + re.escape('<<< "$(sed \'s/--.*$//\' "$migration_file")"; then'),
        script,
    )


@pytest.mark.parametrize(
    "key, unwrapped",
    [
        ("ml/039_tool_category_cohort.sql", True),
        ("ml/040_tool_registry_startup_sync.sql", False),
        ("ml/041_composer_learning_loop_recording.sql", False),
    ],
)
def test_runner_branch(key, unwrapped):
    path = ML / key
    if not path.exists():
        pytest.fail(f"{key} is missing")
    assert _pg.runner_unwraps(path.read_text()) is unwrapped


@pytest.mark.parametrize("key", _pg.LANE_MIGRATIONS)
def test_no_script_level_transaction_control(key):
    path = ML / key
    if not path.exists():
        pytest.fail(f"{key} is missing")
    assert _scan_for_bare_txn(path) == []


@pytest.mark.parametrize(
    "key",
    ["ml/040_tool_registry_startup_sync.sql", "ml/041_composer_learning_loop_recording.sql"],
)
def test_wrapped_files_never_mention_the_unwrap_triggers(key):
    # Comments are stripped by the runner, but a mention in a comment invites a later edit that
    # moves it into code; keep the words out of the wrapped files entirely.
    text = (ML / key).read_text()
    assert not re.search(r"ADD\s+VALUE|CONCURRENTLY", text, re.IGNORECASE)


def test_039_is_one_idempotent_statement():
    text = (ML / "ml/039_tool_category_cohort.sql").read_text()
    code = [
        line for line in (re.sub(r"--.*$", "", raw).strip() for raw in text.splitlines()) if line
    ]
    assert code == ["ALTER TYPE tool_category ADD VALUE IF NOT EXISTS 'COHORT';"]


@pytest.mark.parametrize("name", ["rollback_040.sql", "rollback_041.sql"])
def test_rollbacks_are_never_auto_applied_and_hold_no_transaction_control(name):
    path = ML / "ml" / name
    assert path.exists()
    assert not [k for k in _pg.runner_migration_keys() if k.endswith(name)]
    # Applied by hand with psql --single-transaction (runbook); its own BEGIN/COMMIT would end it.
    assert _scan_for_bare_txn(path) == []
