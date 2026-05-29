"""Hermetic guard: every column a Feast PostgreSQLSource query references must
exist in the committed canonical schema DDL.

This runs WITHOUT the feast SDK and WITHOUT a live database — it AST-parses
``feature_repo/data_sources.py`` for the query strings and text-parses every
committed ``*.sql`` under ``database/`` (CREATE TABLE + ADD COLUMN), since the
canonical columns are spread across the base schema and the migrations
(e.g. territory_metrics in 031, business_metrics' Feast columns in 033). So
unlike the feast-gated ``test_data_sources_canonical_tables.py`` (which skips where the app
image has no feast), this guard actually executes in CI and catches source-query
column drift at PR time — the failure mode behind #556 (``business_metrics_source``
selected ``territory_id``/``brand_id``, which migration 033 never put on the
canonical table; ``patient_journey_source`` selected ``therapy_start_date`` /
``days_on_therapy`` / ``churn_risk_score``, which do not exist).

The live offline ``EXPLAIN`` (pre-deploy) and the ``FEAST_INTEGRATION`` parity
test remain the environment-specific backstops; this is the hermetic, always-on
PR-time guard.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_DATA_SOURCES = _ROOT / "feature_repo" / "data_sources.py"
# Canonical schema DDL is spread across the base schema AND the migrations: the
# core tables live in database/core, but e.g. territory_metrics is created in
# migrations/031 and business_metrics' Feast columns are added in migrations/033.
# Scan every committed *.sql under database/ (CREATE TABLE + ADD COLUMN only) —
# over-capturing columns can only relax this guard, never produce a false drift.
_DATABASE_DIR = _ROOT / "database"

# SQL keywords / functions / cast-types that are never column references.
_NON_COLUMN_TOKENS = {
    "select",
    "from",
    "where",
    "and",
    "or",
    "is",
    "not",
    "null",
    "as",
    "coalesce",
    "now",
    "interval",
    "extract",
    "epoch",
    "case",
    "when",
    "then",
    "else",
    "end",
    "on",
    "distinct",
    "in",
    "between",
    "like",
    "cast",
    "true",
    "false",
    "varchar",
    "integer",
    "numeric",
    "text",
    "boolean",
    "timestamp",
    "timestamptz",
    "bigint",
    "date",
    "smallint",
    "real",
    "double",
    "precision",
}
_CONSTRAINT_KW = {"constraint", "primary", "foreign", "unique", "check", "exclude"}


def _extract_queries() -> dict[str, tuple[str, str]]:
    """{source_name: (from_table, query_sql)} for every PostgreSQLSource(...).

    Pure AST — no feast import, so this works where the app image lacks feast.
    """
    tree = ast.parse(_DATA_SOURCES.read_text())
    out: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        fname = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if fname != "PostgreSQLSource":
            continue
        kw = {k.arg: k.value for k in node.keywords}
        qnode = kw.get("query")
        nnode = kw.get("name")
        if not isinstance(qnode, ast.Constant) or not isinstance(nnode, ast.Constant):
            continue
        query = str(qnode.value)
        m = re.search(r"\bFROM\s+([a-z_][a-z0-9_]*)", query, re.IGNORECASE)
        if not m:
            continue
        out[str(nnode.value)] = (m.group(1).lower(), query)
    return out


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def _idents(expr: str) -> set[str]:
    expr = re.sub(r"::\s*[a-z0-9_]+", " ", expr, flags=re.IGNORECASE)  # strip casts
    expr = re.sub(r"'[^']*'", " ", expr)  # strip string literals
    found = re.findall(r"[a-z_][a-z0-9_]*", expr, flags=re.IGNORECASE)
    return {t.lower() for t in found if t.lower() not in _NON_COLUMN_TOKENS}


def _referenced_columns(query: str) -> set[str]:
    up = query.upper()
    si = up.index("SELECT") + len("SELECT")
    fi = up.index(" FROM ", si)
    cols: set[str] = set()
    for item in _split_top_level(query[si:fi]):
        # the alias after AS is a label, not a column reference
        item = re.sub(r"\bAS\s+[a-z_][a-z0-9_]*\s*$", "", item.strip(), flags=re.IGNORECASE)
        cols |= _idents(item)
    wm = re.search(r"\bWHERE\b", query, re.IGNORECASE)
    if wm:
        cols |= _idents(query[wm.end() :])
    return cols


def _ddl_columns() -> dict[str, set[str]]:
    """{table: columns} from base CREATE TABLE + ALTER TABLE ADD/DROP COLUMN.

    Files are processed in sorted (≈ migration) order and ALTERs applied in
    statement order, so an ADD-then-DROP (e.g. a transient column) ends up
    correctly absent and a DROP-then-readd ends up present. Remaining over-capture
    can only relax the guard; under-capture would surface immediately as a false
    'missing column' in this test's own assertions. Actually-dropped/renamed
    columns in the live DB are covered by the EXPLAIN + FEAST_INTEGRATION backstops.
    """
    cols: dict[str, set[str]] = {}

    for sql_path in sorted(_DATABASE_DIR.rglob("*.sql")):
        text = sql_path.read_text(errors="ignore")

        # CREATE TABLE <t> ( ... );  (CREATE VIEW / MATERIALIZED VIEW won't match)
        for m in re.finditer(
            r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+(?:public\.)?([a-z_][a-z0-9_]*)\s*\((.*?)\n\)\s*;",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            table = m.group(1).lower()
            for line in m.group(2).splitlines():
                mm = re.match(r"\s+([a-z_][a-z0-9_]*)\s+\S", line)
                if mm and mm.group(1).lower() not in _CONSTRAINT_KW:
                    cols.setdefault(table, set()).add(mm.group(1).lower())

        # ALTER TABLE <t> ... ADD/DROP COLUMN, applied in statement order.
        for stmt in text.split(";"):
            tm = re.search(r"ALTER TABLE\s+(?:public\.)?([a-z_][a-z0-9_]*)", stmt, re.IGNORECASE)
            if not tm:
                continue
            table = tm.group(1).lower()
            for cm in re.finditer(
                r"ADD COLUMN\s+(?:IF NOT EXISTS\s+)?([a-z_][a-z0-9_]*)", stmt, re.IGNORECASE
            ):
                cols.setdefault(table, set()).add(cm.group(1).lower())
            for dm in re.finditer(
                r"DROP COLUMN\s+(?:IF EXISTS\s+)?([a-z_][a-z0-9_]*)", stmt, re.IGNORECASE
            ):
                cols.get(table, set()).discard(dm.group(1).lower())
    return cols


_QUERIES = _extract_queries()
_DDL = _ddl_columns()


def test_parsers_are_non_vacuous():
    """Guard against a parse failure silently passing every column check."""
    assert len(_QUERIES) >= 5, f"expected >=5 PostgreSQLSources, parsed {sorted(_QUERIES)}"
    for table in (
        "business_metrics",
        "patient_journeys",
        "triggers",
        "hcp_profiles",
        "territory_metrics",
    ):
        assert _DDL.get(table), f"DDL parser found no columns for {table}"


@pytest.mark.parametrize("source_name", sorted(_QUERIES))
def test_source_query_columns_exist_in_canonical_schema(source_name):
    table, query = _QUERIES[source_name]
    referenced = _referenced_columns(query)
    available = _DDL.get(table, set())
    missing = sorted(referenced - available)
    assert not missing, (
        f"{source_name} (FROM {table}) references columns absent from the committed "
        f"canonical schema DDL: {missing}. Either the column was renamed/dropped "
        f"(fix the query in feature_repo/data_sources.py) or the DDL parser missed it. "
        f"Available ({len(available)}): {sorted(available)}"
    )
