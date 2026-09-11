"""Migration 138 adds ``negative_control_outcome`` to the ``refutation_test_type``
Postgres enum (Lane G, #2007).

``src.repositories.causal_validation.save_suite`` (line ~144) inserts every row
of a suite in ONE call; a persisted ``causal_validations.test_type`` value that
is not in the enum raises 22P02 (invalid input value) and drops the WHOLE
suite's persistence, not just the negative-control row. The value must exist
before the new refutation node ships.

``ALTER TYPE ... ADD VALUE`` cannot run inside a transaction block, so
``scripts/run_migrations.sh`` detects it (after stripping ``--`` comments) and
applies this file UN-wrapped -- no ``--single-transaction`` -- then records
the tracking row in a separate statement only on clean exit. That is why this
migration is exactly ONE statement: a second statement that consumed the new
value in the same file would fail (a new enum value is unusable until the
``ALTER TYPE`` commits), and a second statement of any kind reintroduces a
multi-statement file the un-wrapped path was designed to avoid.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.causal_engine.refutation_runner import RefutationTestType

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "database" / "migrations" / "138_refutation_test_type_negative_control.sql"
RUN_MIGRATIONS = REPO / "scripts" / "run_migrations.sh"

ADD_VALUE = "ALTER TYPE refutation_test_type ADD VALUE IF NOT EXISTS 'negative_control_outcome';"


def _stripped_body() -> str:
    sql = MIGRATION.read_text(encoding="utf-8")
    lines = [re.sub(r"--.*$", "", line) for line in sql.split("\n")]
    return "\n".join(lines)


@pytest.mark.unit
def test_adds_the_value_idempotently():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert ADD_VALUE in sql


@pytest.mark.unit
def test_value_matches_the_enum_definition_in_code():
    """``RefutationTestType.NEGATIVE_CONTROL_OUTCOME`` is the single place the
    token is defined in code; the migration must spell it identically."""
    body = _stripped_body()
    assert f"ADD VALUE IF NOT EXISTS '{RefutationTestType.NEGATIVE_CONTROL_OUTCOME.value}'" in body


@pytest.mark.unit
def test_if_not_exists_present():
    body = _stripped_body()
    assert re.search(r"ADD\s+VALUE\s+IF\s+NOT\s+EXISTS", body, re.I)


@pytest.mark.unit
def test_no_own_transaction_markers():
    """This file must not manage its own transaction: a BEGIN/COMMIT here would
    either nest inside or terminate the runner's handling of the un-wrapped
    apply, since ALTER TYPE ... ADD VALUE cannot run inside a transaction
    block at all."""
    body = _stripped_body()
    assert not re.search(r"^\s*(BEGIN|COMMIT)\s*;", body, re.I | re.M)


@pytest.mark.unit
def test_runner_detects_it_as_non_transactional():
    """Replicate scripts/run_migrations.sh's own detection regex (read from the
    script itself, not retyped) against this file's comment-stripped body, so
    a change to either side that breaks the match is caught here."""
    runner_src = RUN_MIGRATIONS.read_text(encoding="utf-8")
    detect = re.search(r'grep -qiE.*?"([^"]+)"', runner_src, re.S)
    assert detect is not None, "could not locate the runner's detection grep -qiE pattern"
    posix_pattern = detect.group(1)
    # Translate the POSIX bracket class used in bash's grep -E to a Python \s
    # equivalent; the alternation and anchors are already ERE-compatible.
    py_pattern = posix_pattern.replace("[[:space:]]", r"\s")
    body = _stripped_body()
    assert re.search(py_pattern, body, re.I) is not None


@pytest.mark.unit
def test_exactly_one_statement_and_nothing_consumes_the_value():
    """A file with more than one statement, or one that reads/writes using the
    new value, risks the runner's un-wrapped-apply contract: the value is not
    usable until the ALTER TYPE itself commits."""
    body = _stripped_body()
    statements = [s.strip() for s in body.split(";") if s.strip()]
    assert len(statements) == 1
    assert not re.search(r"\b(INSERT|UPDATE|SELECT|CREATE)\b", body, re.I)


@pytest.mark.unit
def test_caveat_comment_names_non_transactional_unwrapped():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "non-transactional" in sql
    assert "UN-wrapped" in sql or "un-wrapped" in sql.lower()


@pytest.mark.unit
def test_header_states_intent_and_blast_radius():
    """The header must explain WHY (negative-control refutation test, #2007)
    and the persistence blast radius (save_suite inserts a whole suite's rows
    in one call, so a missing value drops the entire suite)."""
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "#2007" in sql
    assert "negative" in sql.lower() and "control" in sql.lower()
    assert "save_suite" in sql or "suite" in sql.lower()
