"""Static checks on ml/044, which persists WHICH rating labelled a composition (#2035).

No database: CI has none for this table, and the deploy applies the file with
``scripts/run_migrations.sh`` inside ``--single-transaction`` with its ledger row. These pin what
a real apply depends on, and — more importantly — they pin the two design decisions the column
exists to encode: it is nullable with no default (NULL means "claim not recorded", never a claim
by rating 0), and it carries NO foreign key.

The missing foreign key is the load-bearing part. Ratings cascade away with their message or
conversation, and every referential action is worse than a dangling id: CASCADE deletes the
episode, SET NULL re-creates the #2035 starvation the column fixes, RESTRICT lets a learning
table veto deleting a chat message. A later "tidy-up" that adds the FK back would silently
restore the bug, so the absence is asserted, not merely commented.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.integration.test_migrations_no_inner_txn import _scan_for_bare_txn
from tests.unit.test_database.learning_loop import _pg

DATABASE = _pg.REPO_ROOT / "database"
KEY = "ml/044_composer_episodes_feedback_id.sql"
MIGRATION = DATABASE / KEY
ROLLBACK = DATABASE / "ml" / "rollback_044.sql"

COLUMN = "feedback_id"


def _code(path: Path) -> str:
    """SQL with ``--`` and ``/* */`` comments removed."""
    if not path.exists():
        pytest.fail(f"{path.relative_to(_pg.REPO_ROOT)} is missing")
    text = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.DOTALL)
    return "\n".join(re.sub(r"--.*$", "", line) for line in text.splitlines())


def _ddl(path: Path) -> str:
    """:func:`_code`, with single-quoted string literals blanked as well.

    The COMMENT ON COLUMN body documents the very decisions these tests assert, so it contains
    the words "not a foreign key" verbatim. A structural check that scanned it would be reading
    the documentation as the thing being documented.
    """
    return re.sub(r"'(?:[^']|'')*'", "''", _code(path))


@pytest.mark.unit
def test_adds_the_claim_column_idempotently_as_a_bigint():
    """BIGINT because chatbot_message_feedback.id is a bigint identity column."""
    code = _code(MIGRATION)
    m = re.search(
        rf"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+{COLUMN}\s+([^,;]+)", code, flags=re.IGNORECASE
    )
    assert m, f"{COLUMN} is not added idempotently"
    assert " ".join(m.group(1).split()).lower() == "bigint"


@pytest.mark.unit
def test_the_column_is_nullable_with_no_default():
    """NULL is the honest value for "no claim recorded": every pre-044 row, and every episode no
    rating has labelled. A default would make those read as a claim by some rating."""
    code = _ddl(MIGRATION)
    assert not re.search(r"\bDEFAULT\b|\bNOT\s+NULL\b", code, flags=re.IGNORECASE)


@pytest.mark.unit
def test_the_claim_carries_no_foreign_key():
    """The whole point of the column: a dangling id must survive its rating's deletion.

    See the module docstring — every referential action available here is worse than a dangling
    id, and SET NULL in particular re-creates the very starvation #2035 describes.
    """
    code = _ddl(MIGRATION)
    assert not re.search(r"\bREFERENCES\b|\bFOREIGN\s+KEY\b", code, flags=re.IGNORECASE)
    assert not re.search(r"\bON\s+DELETE\b", code, flags=re.IGNORECASE)


@pytest.mark.unit
def test_the_column_is_documented_in_the_database():
    """A bare BIGINT that points at another table without a constraint is unreadable without its
    comment; psql's \\d+ is where the next reader meets it."""
    code = _code(MIGRATION)
    m = re.search(
        rf"COMMENT\s+ON\s+COLUMN\s+(?:public\.)?composer_episodes\.{COLUMN}\s+IS\s+(.*?);",
        code,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert m, "the claim column has no COMMENT ON COLUMN"
    body = m.group(1).lower()
    assert "chatbot_message_feedback" in body, "the comment must name the table it points at"
    assert "044" in body and "not a foreign key" in body


@pytest.mark.unit
def test_no_backfill_invents_a_claim_for_an_existing_row():
    """Nothing can say which rating labelled a row written before the column existed."""
    assert not re.findall(r"\bUPDATE\b", _ddl(MIGRATION), flags=re.IGNORECASE)


@pytest.mark.unit
def test_runner_applies_it_wrapped_with_no_transaction_control_of_its_own():
    assert KEY in _pg.runner_migration_keys()
    assert _pg.runner_unwraps(MIGRATION.read_text()) is False
    assert _scan_for_bare_txn(MIGRATION) == []


@pytest.mark.unit
def test_the_wrapped_file_never_mentions_the_unwrap_triggers():
    # Comments are stripped by the runner, but a mention in a comment invites a later edit that
    # moves it into code and silently costs the file its atomicity.
    assert not re.search(r"ADD\s+VALUE|CONCURRENTLY", MIGRATION.read_text(), re.IGNORECASE)


@pytest.mark.unit
def test_rollback_is_never_auto_applied_and_undoes_the_column_and_ledger_row():
    code = _code(ROLLBACK)
    assert not [k for k in _pg.runner_migration_keys() if k.endswith(ROLLBACK.name)]
    assert _scan_for_bare_txn(ROLLBACK) == []
    assert re.search(rf"DROP\s+COLUMN\s+IF\s+EXISTS\s+{COLUMN}\b", code, re.IGNORECASE)
    assert f"filename = '{KEY}'" in code
    # success / feedback_at are a separate decision from the claim: rolling back the attribution
    # must not silently unlabel episodes.
    assert not re.search(r"\bsuccess\b|\bfeedback_at\b", code, re.IGNORECASE)


def _schema_columns() -> set[str]:
    """Columns of composer_episodes as ml/013's CREATE TABLE plus every later ADD COLUMN."""
    create = _code(DATABASE / "ml" / "013_tool_composer_tables.sql")
    body = re.search(
        r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+composer_episodes\s*\((.*?)\n\);", create, re.DOTALL
    )
    assert body, "composer_episodes CREATE TABLE not found in ml/013"
    columns = {
        m.group(1)
        for m in re.finditer(r"^\s*([a-z_][a-z0-9_]*)\s+\S", body.group(1), re.MULTILINE)
        if m.group(1).upper() != "CONSTRAINT"
    }
    for path in sorted(DATABASE.rglob("*.sql")):
        if path.name.startswith("rollback_"):
            continue
        for stmt in re.findall(
            r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:public\.)?composer_episodes\b(.*?);",
            _code(path),
            flags=re.IGNORECASE | re.DOTALL,
        ):
            columns.update(
                re.findall(r"ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", stmt, re.IGNORECASE)
            )
    return columns


@pytest.mark.unit
def test_every_column_the_linker_writes_exists_in_the_migrated_schema():
    """A written column no migration adds fails every label write on the deployed table."""
    from src.tasks import composition_feedback_tasks as tasks

    schema = _schema_columns()
    assert COLUMN in schema
    written = {"success", "feedback_at", COLUMN}
    assert written <= schema
    # And the reads: a column the matcher gates on but the SELECT never asks for is how the T14
    # ordering defect happened.
    read = set(tasks._EPISODE_COLUMNS.replace("\n", " ").split(", "))
    assert {c.strip() for c in read if c.strip()} <= schema
