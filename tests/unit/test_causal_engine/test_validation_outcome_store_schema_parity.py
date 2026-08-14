"""Schema-shape regression guard for validation_outcomes (#1423).

The Supabase insert path serialises a ValidationOutcome via
``SupabaseValidationOutcomeStore._outcome_to_row`` and inserts the flat dict
directly into ``validation_outcomes``. If that dict carries a key with no
backing column, PostgREST rejects the whole insert with PGRST204 and the store
degrades to the ephemeral in-memory fallback — the Feedback-Learner signal is
then silently dropped on every refutation run (the #1423 incident).

This test locks the row dict to the columns DECLARED in the migration files so
the two cannot drift apart again. It is DB-free: it parses the DDL and calls
the pure serializer, so it runs in the plain unit lane.

Root cause it guards against: migration 007 created the lean table (metadata +
confidence_interval jsonb) while the code writes eight richer top-level fields
(gate_decision, confidence_score, tests_passed/failed/total, raw_suite,
agent_context, dag_hash). migration 021 declared only agent_context — and even
that never reached the live table because the ledger recorded it as applied, so
the runner skipped it. Migration 121 reconciles all eight.

Key parity is necessary but not sufficient: #1611 showed a declared column can
still reject the payload on TYPE. ``outcome_id`` is ``UUID PRIMARY KEY`` while
the generator emitted ``vo_<12 hex>``, so every insert failed 22P02 and hit the
same memory_fallback drop. The type guard below closes that second dimension.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Callable

from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)
from src.causal_engine.validation_outcome import (
    ValidationOutcome,
    ValidationOutcomeType,
    create_validation_outcome,
)
from src.causal_engine.validation_outcome_store import SupabaseValidationOutcomeStore

# tests/unit/test_causal_engine/<this file>  ->  repo root is parents[3]
_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "database" / "migrations"

_TABLE = "validation_outcomes"

# Column defs inside a CREATE TABLE body that are actually table constraints, not columns.
_CONSTRAINT_LEAD = re.compile(r"^\s*(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT)\b", re.IGNORECASE)
_IDENT = re.compile(r'^\s*"?([a-z_][a-z0-9_]*)"?', re.IGNORECASE)


def _declared_columns() -> set[str]:
    """All columns declared for validation_outcomes across every migration file.

    Collects CREATE TABLE column names plus any ``ALTER TABLE ... ADD COLUMN``.
    Over-approximation is safe here (the assertion is a subset check), so this
    deliberately errs toward including anything that parses as a column.
    """
    cols: set[str] = set()
    for sql_file in sorted(_MIGRATIONS_DIR.rglob("*.sql")):
        text = sql_file.read_text(encoding="utf-8")

        # CREATE TABLE [IF NOT EXISTS] [public.]validation_outcomes ( <body> );
        for m in re.finditer(
            r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:public\.)?"
            + _TABLE
            + r"\s*\((?P<body>.*?)\n\)\s*;",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            for raw in m.group("body").splitlines():
                line = raw.strip()
                if not line or line.startswith("--") or line.startswith(")"):
                    continue
                if _CONSTRAINT_LEAD.match(line):
                    continue
                ident = _IDENT.match(line)
                if ident:
                    cols.add(ident.group(1).lower())

        # ALTER TABLE [public.]validation_outcomes ... ADD COLUMN [IF NOT EXISTS] <name>
        for m in re.finditer(
            r"ALTER\s+TABLE\s+(?:public\.)?" + _TABLE + r"\b(?P<stmt>.*?);",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            for a in re.finditer(
                r"ADD\s+COLUMN(?:\s+IF\s+NOT\s+EXISTS)?\s+\"?([a-z_][a-z0-9_]*)\"?",
                m.group("stmt"),
                re.IGNORECASE,
            ):
                cols.add(a.group(1).lower())

    return cols


# A column's SQL type: the leading type token plus any (n) / (p,s) args.
# "double precision" is the one multi-word type this table uses.
_TYPE = r"(?P<type>double\s+precision|[a-z][a-z0-9_]*(?:\s*\([^)]*\))?)"
_COLUMN_TYPE_DECL = re.compile(r'^\s*"?(?P<name>[a-z_][a-z0-9_]*)"?\s+' + _TYPE, re.IGNORECASE)
_ALTER_COLUMN_TYPE = re.compile(
    r'ALTER\s+COLUMN\s+"?(?P<name>[a-z_][a-z0-9_]*)"?\s+(?:SET\s+DATA\s+)?TYPE\s+' + _TYPE,
    re.IGNORECASE,
)


def _declared_column_type(column: str) -> str | None:
    """The SQL type validation_outcomes.<column> ends up with after every migration.

    Later files win, so an ``ALTER COLUMN ... TYPE`` in a newer migration
    supersedes the original ``CREATE TABLE`` declaration.
    """
    latest: str | None = None
    for sql_file in sorted(_MIGRATIONS_DIR.rglob("*.sql")):
        text = sql_file.read_text(encoding="utf-8")
        if _TABLE not in text:
            continue

        for m in re.finditer(
            r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:public\.)?"
            + _TABLE
            + r"\s*\((?P<body>.*?)\n\)\s*;",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            for raw in m.group("body").splitlines():
                line = raw.strip()
                if not line or line.startswith("--") or _CONSTRAINT_LEAD.match(line):
                    continue
                decl = _COLUMN_TYPE_DECL.match(line)
                if decl and decl.group("name").lower() == column:
                    latest = decl.group("type")

        for m in re.finditer(
            r"ALTER\s+TABLE\s+(?:public\.)?" + _TABLE + r"\b(?P<stmt>.*?);",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            for a in _ALTER_COLUMN_TYPE.finditer(m.group("stmt")):
                if a.group("name").lower() == column:
                    latest = a.group("type")

    return " ".join(latest.lower().split()) if latest else None


# How to prove a value would coerce into a declared column type. A type absent
# from this map fails the guard loudly rather than passing vacuously.
_TYPE_COERCERS: dict[str, Callable[[str], object]] = {
    "uuid": uuid.UUID,
    "text": str,
}


def _minimal_suite() -> RefutationSuite:
    """The smallest suite create_validation_outcome accepts, so the id under test
    comes from the REAL generator rather than a hand-written literal."""
    return RefutationSuite(
        passed=True,
        confidence_score=0.9,
        gate_decision=GateDecision.PROCEED,
        tests=[
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.5,
                refuted_effect=0.01,
                delta_percent=2.0,
            )
        ],
        treatment_variable="rep_visits",
        outcome_variable="trx_total",
        brand="Kisqali",
        estimate_id="est-1611",
    )


def _sample_outcome() -> ValidationOutcome:
    """A fully-populated outcome so every serialized key is present."""
    return ValidationOutcome(
        outcome_id="00000000-0000-0000-0000-000000000000",
        outcome_type=ValidationOutcomeType.BLOCKED,
        timestamp="2026-01-01T00:00:00Z",
        estimate_id="est-1",
        treatment_variable="rep_visits",
        outcome_variable="trx",
        brand="Kisqali",
        gate_decision="block",
        confidence_score=0.42,
        tests_passed=3,
        tests_failed=1,
        tests_total=4,
        # failure_patterns left empty: this is a KEY-parity test, and the
        # "failure_patterns" row key is emitted regardless of list contents.
        raw_suite={"suite": "ran"},
        agent_context={"agent": "causal_impact", "query": "impact of rep visits"},
        dag_hash="abc123",
        sample_size=5000,
        effect_size=0.0352,
    )


def test_migrations_directory_is_discoverable() -> None:
    assert _MIGRATIONS_DIR.is_dir(), f"migrations dir not found at {_MIGRATIONS_DIR}"
    assert (_MIGRATIONS_DIR / "007_validation_outcomes.sql").exists()


def test_declared_columns_include_the_007_baseline() -> None:
    # Sanity that the parser actually found the table's columns.
    declared = _declared_columns()
    for baseline in ("outcome_id", "outcome_type", "brand", "effect_size", "metadata"):
        assert baseline in declared, f"parser missed baseline column {baseline!r}"


def _latest_outcome_type_check_values() -> set[str]:
    """Values accepted by the NEWEST outcome_type CHECK constraint on
    validation_outcomes across the migration files.

    Later migrations that DROP + ADD the constraint supersede earlier ones, so
    the effective set is the last one encountered in sorted-file / in-file order.
    """
    latest: set[str] | None = None
    for sql_file in sorted(_MIGRATIONS_DIR.rglob("*.sql")):
        text = sql_file.read_text(encoding="utf-8")
        if _TABLE not in text:
            continue
        for m in re.finditer(
            r"CHECK\s*\(\s*outcome_type\s+IN\s*\((?P<vals>[^)]*)\)",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            vals = set(re.findall(r"'([^']+)'", m.group("vals")))
            if vals:
                latest = vals
    return latest or set()


def test_enum_outcome_types_are_accepted_by_the_check_constraint() -> None:
    """Every ValidationOutcomeType the code can emit must satisfy the newest
    declared outcome_type CHECK.

    Otherwise the insert fails with a 23514 check violation and degrades to the
    memory_fallback — the same silent-drop failure mode as the missing columns.
    RED before migration 121: the 007 CHECK accepts only 'passed' of the enum's
    five values (the E-value BLOCKED verdict from the first live suite would be
    rejected). GREEN once 121 replaces the CHECK with the enum set.
    """
    enum_values = {t.value for t in ValidationOutcomeType}
    accepted = _latest_outcome_type_check_values()
    assert accepted, "no outcome_type CHECK constraint found in the migrations"
    unaccepted = enum_values - accepted
    assert not unaccepted, (
        "ValidationOutcomeType values rejected by the newest outcome_type CHECK "
        f"(insert would fail with a 23514 check violation): {sorted(unaccepted)}"
    )


def test_outcome_row_keys_are_all_declared_in_migrations() -> None:
    """Every key the insert payload writes must have a declared backing column.

    RED before migration 121: gate_decision / confidence_score / tests_* /
    raw_suite / dag_hash are written by _outcome_to_row but declared nowhere
    (and agent_context only in the never-effective 021). GREEN once 121 adds
    all eight.
    """
    store = SupabaseValidationOutcomeStore()
    row = store._outcome_to_row(_sample_outcome())
    declared = _declared_columns()

    missing = {key for key in row if key.lower() not in declared}
    assert not missing, (
        "validation_outcomes insert payload writes columns not declared in any "
        f"migration (would trigger PGRST204 → memory_fallback): {sorted(missing)}"
    )


def test_outcome_id_column_type_is_parsed_from_the_migrations() -> None:
    # Sanity that the type parser sees the real DDL, so the guard below can
    # never pass just because it found nothing.
    assert _declared_column_type("outcome_id") == "uuid"


def test_generated_outcome_id_coerces_to_the_declared_column_type() -> None:
    """The id the generator mints must fit the column it is inserted into.

    RED before the fix: create_validation_outcome minted ``vo_<12 hex>`` while
    outcome_id is declared UUID, so Postgres rejected every insert with 22P02
    ("invalid input syntax for type uuid") and the store degraded to the
    ephemeral memory_fallback — the Feedback-Learner signal dropped on every
    refutation run (#1611). GREEN once the generator mints a bare uuid4.
    """
    declared = _declared_column_type("outcome_id")
    assert declared, "no outcome_id column type found in the migrations"

    coerce = _TYPE_COERCERS.get(declared)
    assert coerce is not None, (
        f"outcome_id is declared {declared!r}, which this guard cannot verify — "
        "add a coercer so the payload/column type contract stays proven"
    )

    store = SupabaseValidationOutcomeStore()
    row = store._outcome_to_row(create_validation_outcome(_minimal_suite()))

    try:
        coerce(row["outcome_id"])
    except (AttributeError, TypeError, ValueError) as exc:
        raise AssertionError(
            f"generated outcome_id {row['outcome_id']!r} does not coerce to the declared "
            f"{declared} column — the insert fails and degrades to memory_fallback: {exc}"
        ) from exc
