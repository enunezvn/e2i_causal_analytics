"""Migration ml/036 content lock — discovery tables move ``ml`` -> ``public`` (#1974).

WHY THIS MIGRATION EXISTS (measured 2026-09-08 on the live, self-contained prod
Supabase): ``ml/026`` created the five causal-discovery tables in a dedicated
``ml`` schema — the ONLY file in ``database/`` that schema-qualifies ``ml.``.
PostgREST exposes only ``public,storage,graphql_public`` (``PGRST106`` on the
``ml`` profile), so no Supabase-client repository could ever write to them; the
tables hold 0 rows after three months. Moving them into ``public`` (free while
empty) makes ``BaseRepository`` work directly and lets the tables inherit
migration 058's anon/authenticated revocation posture, which never reached
``ml`` (``authenticated`` still holds INSERT/SELECT/UPDATE there).

These are text-level pins on the migration file; the faithful proof is the
BEGIN/ROLLBACK rehearsal recorded in the PR (objects land in ``public``, the
renamed enum, grants as asserted, the RPC round-trip, idempotent second apply).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = REPO_ROOT / "database" / "ml" / "036_move_discovery_tables_to_public.sql"

TABLES = (
    "discovered_dags",
    "discovery_algorithm_runs",
    "discovered_edges",
    "driver_rankings",
    "feature_rankings",
)
VIEWS = ("v_recent_discoveries", "v_high_confidence_edges", "v_discordant_features")


def _content() -> str:
    return MIGRATION.read_text()


def _stripped(text: str | None = None) -> str:
    """Mirror run_migrations.sh detection: strip ``--`` line comments first."""
    return re.sub(r"--.*$", "", text if text is not None else _content(), flags=re.MULTILINE)


def _find(pattern: str, text: str) -> int:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    assert match is not None, f"pattern not found: {pattern!r}"
    return match.start()


# ---------------------------------------------------------------------------
# File + runner-wrappability
# ---------------------------------------------------------------------------


def test_migration_file_exists():
    assert MIGRATION.exists(), f"missing migration: {MIGRATION}"


def test_stays_single_transaction_wrappable():
    """run_migrations.sh wraps the file in ``--single-transaction`` unless it
    finds an un-wrappable pattern; the move + grants + assert block must land
    atomically or not at all."""
    stripped = _stripped()
    assert not re.search(r"ALTER\s+TYPE\s.*ADD\s+VALUE", stripped, re.IGNORECASE | re.DOTALL)
    assert not re.search(r"\bCONCURRENT" + r"LY\b", stripped, re.IGNORECASE)
    assert not re.search(r"^\s*(COMMIT|BEGIN)\s*;", stripped, re.IGNORECASE | re.MULTILINE)


def test_never_drops_the_ml_schema():
    """The ``ml`` schema is left in place (other environments may hold objects
    there; dropping it is out of this migration's scope) — and the file says so."""
    assert not re.search(r"DROP\s+SCHEMA", _stripped(), re.IGNORECASE)
    assert re.search(r"DROP\s+SCHEMA", _content(), re.IGNORECASE), (
        "the migration must EXPLAIN (in a comment) why the ml schema is not dropped"
    )


def test_negative_detectors_are_live():
    """Positive control for the two absence assertions above: the detectors
    must fire on a synthetic offender, else the negatives pass vacuously."""
    assert re.search(r"DROP\s+SCHEMA", _stripped("DROP SCHEMA ml CASCADE;"), re.IGNORECASE)
    assert re.search(r"^\s*COMMIT\s*;", _stripped("x;\nCOMMIT;"), re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# The enum clash: ml.gate_decision must be RENAMED before it moves
# ---------------------------------------------------------------------------


def test_gate_decision_renamed_before_set_schema():
    """``public.gate_decision`` already exists (migration 010, labels
    proceed/review/block) and is a DIFFERENT enum from ``ml.gate_decision``
    (accept/review/reject/augment). The ml one must be renamed to
    ``discovery_gate_decision`` BEFORE ``SET SCHEMA public``."""
    stripped = _stripped()
    rename_at = _find(
        r"ALTER\s+TYPE\s+ml\.gate_decision\s+RENAME\s+TO\s+discovery_gate_decision", stripped
    )
    move_at = _find(r"ALTER\s+TYPE\s+ml\.discovery_gate_decision\s+SET\s+SCHEMA\s+public", stripped)
    assert rename_at < move_at
    # The other two enums have no clash and move under their own names.
    _find(r"ALTER\s+TYPE\s+ml\.discovery_algorithm\s+SET\s+SCHEMA\s+public", stripped)
    _find(r"ALTER\s+TYPE\s+ml\.edge_type\s+SET\s+SCHEMA\s+public", stripped)
    # Never the clashing bare name.
    assert not re.search(
        r"ALTER\s+TYPE\s+ml\.gate_decision\s+SET\s+SCHEMA", stripped, re.IGNORECASE
    )


# ---------------------------------------------------------------------------
# Tables / views / functions move, guarded for idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", TABLES)
def test_each_table_moves_guarded(table: str):
    stripped = _stripped()
    _find(rf"ALTER\s+TABLE\s+ml\.{table}\s+SET\s+SCHEMA\s+public", stripped)
    # Guard: the move only runs while the object still lives in ml (second
    # apply / manual out-of-band apply then deploy re-run).
    _find(rf"to_regclass\(\s*'ml\.{table}'\s*\)\s+IS\s+NOT\s+NULL", stripped)


@pytest.mark.parametrize("view", VIEWS)
def test_each_view_moves_guarded(view: str):
    stripped = _stripped()
    _find(rf"ALTER\s+VIEW\s+ml\.{view}\s+SET\s+SCHEMA\s+public", stripped)
    _find(rf"to_regclass\(\s*'ml\.{view}'\s*\)\s+IS\s+NOT\s+NULL", stripped)


def test_trigger_function_moves_by_alter():
    """The updated_at trigger function has no schema references in its body,
    so it can move; the trigger references it by OID and follows."""
    _find(
        r"ALTER\s+FUNCTION\s+ml\.update_discovered_dags_timestamp\(\)\s+SET\s+SCHEMA\s+public",
        _stripped(),
    )


@pytest.mark.parametrize("fn", ("get_dag_edges", "get_feature_ranking_comparison"))
def test_reader_functions_recreated_in_public_without_ml_references(fn: str):
    """Their plpgsql bodies textually reference ``ml.`` tables (resolved at
    execution), so the ml versions must be dropped and recreated in public
    pointing at ``public.`` relations."""
    stripped = _stripped()
    _find(rf"DROP\s+FUNCTION\s+IF\s+EXISTS\s+ml\.{fn}\s*\(", stripped)
    start = _find(rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+public\.{fn}\s*\(", stripped)
    body_end = stripped.find("LANGUAGE plpgsql", start)
    assert body_end > start, f"{fn}: plpgsql body end not found"
    body = stripped[start:body_end]
    assert not re.search(r"\bml\.", body), f"{fn}: recreated body still references ml."


def test_ml_reference_detector_is_live():
    assert re.search(r"\bml\.", "FROM ml.discovered_edges e")
    assert not re.search(r"\bml\.", "FROM public.discovered_edges e")


# ---------------------------------------------------------------------------
# New columns: provenance + the keys the writer needs
# ---------------------------------------------------------------------------


def test_is_synthetic_added_with_platform_convention():
    """Mirror 063/069: BOOLEAN NOT NULL DEFAULT false, a COMMENT, and the
    partial index over the synthetic minority."""
    stripped = _stripped()
    _find(
        r"ALTER\s+TABLE\s+public\.discovered_dags\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+"
        r"is_synthetic\s+BOOLEAN\s+NOT\s+NULL\s+DEFAULT\s+false",
        stripped,
    )
    _find(r"COMMENT\s+ON\s+COLUMN\s+public\.discovered_dags\.is_synthetic\s+IS", stripped)
    _find(
        r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+idx_discovered_dags_is_synthetic\s+"
        r"ON\s+public\.discovered_dags\s*\(\s*is_synthetic\s*\)\s+WHERE\s+is_synthetic",
        stripped,
    )


@pytest.mark.parametrize(
    "column",
    ("dag_version_hash", "query_id", "treatment_variable", "outcome_variable"),
)
def test_writer_key_columns_added(column: str):
    """The 026 shape has no place for the expert-review key (dag_version_hash),
    the run id (query_id) or the estimand; the writer needs all four as real,
    queryable columns rather than jsonb needles."""
    stripped = _stripped()
    _find(
        rf"ALTER\s+TABLE\s+public\.discovered_dags\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+{column}\b",
        stripped,
    )


def test_dag_version_hash_indexed():
    _find(
        r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+idx_discovered_dags_dag_version_hash\s+"
        r"ON\s+public\.discovered_dags\s*\(\s*dag_version_hash\s*\)",
        _stripped(),
    )


# ---------------------------------------------------------------------------
# Grants: 058's posture, made explicit and asserted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", TABLES)
def test_service_role_granted_on_each_table(table: str):
    stripped = _stripped()
    grant_at = _find(rf"GRANT\s+[A-Z,\s]+\s+ON\s+public\.{table}\s+TO\s+service_role", stripped)
    grant_stmt = stripped[grant_at : stripped.find(";", grant_at)]
    for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
        assert re.search(rf"\b{privilege}\b", grant_stmt), f"{table}: missing {privilege}"


@pytest.mark.parametrize("relation", TABLES + VIEWS)
def test_anon_and_authenticated_revoked_on_each_relation(relation: str):
    _find(rf"REVOKE\s+ALL\s+ON\s+public\.{relation}\s+FROM\s+anon\s*,\s*authenticated", _stripped())


def test_grants_and_locations_are_asserted_not_trusted():
    """A DO block must RAISE EXCEPTION if the resulting grants/locations are
    not what the migration intends (default privileges differ per applier
    role — see 058's supabase_admin caveat)."""
    stripped = _stripped()
    assert "RAISE EXCEPTION" in stripped
    for role in ("anon", "authenticated"):
        _find(rf"has_table_privilege\(\s*'{role}'", stripped)
        _find(rf"has_function_privilege\(\s*'{role}'", stripped)
    _find(r"has_table_privilege\(\s*'service_role'", stripped)
    _find(r"has_function_privilege\(\s*'service_role'", stripped)
    # Locations: nothing may remain in ml after the move.
    _find(r"nspname\s*=\s*'ml'", stripped)


# ---------------------------------------------------------------------------
# The atomic writer RPC
# ---------------------------------------------------------------------------


def test_record_rpc_shape():
    stripped = _stripped()
    start = _find(
        r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+public\.record_discovered_dag\s*\(\s*p_payload\s+jsonb\s*\)",
        stripped,
    )
    header = stripped[start : stripped.find("AS $$", start)]
    assert re.search(r"RETURNS\s+jsonb", header, re.IGNORECASE)
    assert re.search(r"SECURITY\s+INVOKER", header, re.IGNORECASE)
    assert re.search(r"SET\s+search_path\s*=\s*public", header, re.IGNORECASE)
    body = stripped[start : stripped.find("$$;", stripped.find("AS $$", start))]
    for table in ("discovered_dags", "discovery_algorithm_runs", "discovered_edges"):
        assert re.search(rf"INSERT\s+INTO\s+public\.{table}", body, re.IGNORECASE), table
    # Provenance is never defaulted silently (ADR-017): a payload that does
    # not STATE is_synthetic is rejected.
    assert re.search(r"RAISE\s+EXCEPTION[^;]*is_synthetic", body, re.IGNORECASE)


def test_record_rpc_locked_to_service_role():
    stripped = _stripped()
    _find(
        r"REVOKE\s+ALL\s+ON\s+FUNCTION\s+public\.record_discovered_dag\s*\(\s*jsonb\s*\)\s+"
        r"FROM\s+PUBLIC\s*,\s*anon\s*,\s*authenticated",
        stripped,
    )
    _find(
        r"GRANT\s+EXECUTE\s+ON\s+FUNCTION\s+public\.record_discovered_dag\s*\(\s*jsonb\s*\)\s+"
        r"TO\s+service_role",
        stripped,
    )


def test_postgrest_schema_reload_noted_and_notified():
    """The event trigger ``pgrst_ddl_watch`` already reloads PostgREST's schema
    cache on DDL here (verified live); the conventional NOTIFY is kept for
    environments without it, and the file says which is which."""
    assert re.search(r"NOTIFY\s+pgrst\s*,\s*'reload schema'", _stripped(), re.IGNORECASE)
    assert "pgrst_ddl_watch" in _content()


# ---------------------------------------------------------------------------
# Cross-language pins: SQL names/labels == Python names/labels
# ---------------------------------------------------------------------------


def test_python_writer_names_match_sql():
    from src.repositories.discovered_dag import DiscoveredDagRepository

    stripped = _stripped()
    assert DiscoveredDagRepository.RPC_NAME == "record_discovered_dag"
    assert DiscoveredDagRepository.table_name == "discovered_dags"
    assert f"public.{DiscoveredDagRepository.RPC_NAME}(" in stripped


def test_enum_labels_asserted_in_sql_match_python_enums():
    """The migration asserts the moved enums' labels; those literal lists must
    be the Python enums' values (the writer casts ``.value`` into them)."""
    from src.causal_engine.discovery.base import (
        DiscoveryAlgorithmType,
        EdgeType,
        GateDecision,
    )

    stripped = _stripped()
    for enum_cls, sql_type in (
        (GateDecision, "discovery_gate_decision"),
        (EdgeType, "edge_type"),
        (DiscoveryAlgorithmType, "discovery_algorithm"),
    ):
        # The type name appears several times (guards, casts); the label
        # assertion is the occurrence followed by an ARRAY[...] of labels.
        windows = [
            stripped[m.start() : m.start() + 600]
            for m in re.finditer(rf"typname\s*=\s*'{sql_type}'", stripped)
        ]
        assert windows, f"{sql_type}: no typname = '{sql_type}' check in the migration"
        labels = [f"'{member.value}'" for member in enum_cls]
        assert any(all(label in w for label in labels) for w in windows), (
            f"{sql_type}: no assertion window carries every Python label {labels}"
        )
