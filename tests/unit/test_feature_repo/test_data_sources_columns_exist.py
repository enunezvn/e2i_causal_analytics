"""Hermetic guard: every column a Feast PostgreSQLSource query references must
exist in the committed canonical schema DDL.

This runs WITHOUT the feast SDK and WITHOUT a live database — it AST-parses
``feature_repo/data_sources.py`` for the query strings and text-parses every
committed FORWARD ``*.sql`` under ``database/`` (CREATE TABLE + ADD/DROP/RENAME
COLUMN), since the canonical columns are spread across the base schema and the
migrations (e.g. territory_metrics in 031, business_metrics' Feast columns in
033). So
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
# Scan every committed FORWARD *.sql under database/ (CREATE TABLE + ADD/DROP/
# RENAME COLUMN). Over-capturing columns can only relax this guard — but a RENAME
# RETIRES a name, so once renames are modelled the scan is no longer purely
# additive and a stray reverse rename WOULD produce a false drift. That is why
# only files the migration runner actually applies are scanned; see
# _is_forward_migration.
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


def _is_forward_migration(sql_path: Path) -> bool:
    """Is this a file the migration runner would actually APPLY?

    Mirrors ``scripts/run_migrations.sh``'s ``apply_dir()`` SAFETY rule verbatim:
    "rollback_*/*_rollback and *_validation_queries files are NOT forward
    migrations and are excluded". The DDL model must be built from exactly the
    files that reach the database, so this is the runner's rule rather than an
    ad-hoc skip list — if the runner's rule changes, this must follow it.

    MEASURED EFFECT (2026-09-17). Stubbing this to ``True`` changes the modelled
    columns of FIVE tables — 28 columns lost, 2 gained — via the 30 static
    ADD/DROP COLUMN statements in ``database/ml/rollback_040-044``, and it moves
    the model in BOTH harmful directions:

      * FALSE PASS — ``tool_registry.success_rate``. Forward migration
        ``ml/040_tool_registry_startup_sync.sql:44`` DROPs it; ``ml/rollback_040.sql:57``
        ADDs it back, and ``rollback_040`` sorts AFTER ``040``, so unfiltered the
        rollback wins and the model carries a column the forward path removed.
        A source query naming it would be waved through — the #556 class, with
        this guard blind to it.
      * FALSE FAIL — ``tool_performance.attempts`` and ``twin_simulations.cohort_ate``,
        added by forward migrations ``ml/041``/``ml/042`` and dropped by their
        rollbacks: unfiltered, real columns vanish from the model.

    Note what this filter does NOT do: ``business_metrics`` is byte-identical with
    or without it, because rollback_144's reverse rename is DYNAMIC and no static
    reader can see it. The rename case motivated this filter but is not what it
    currently protects; see TestCommittedRenamesAreModelled for why the rename
    guard is still kept.
    """
    name = sql_path.name.lower()
    return not (
        name.startswith("rollback_") or "_rollback" in name or "_validation_queries" in name
    )


def _ddl_columns() -> dict[str, set[str]]:
    """{table: columns} from base CREATE TABLE + ALTER TABLE ADD/DROP/RENAME COLUMN.

    Files are processed in sorted (≈ migration) order and ALTERs applied in
    statement order, so an ADD-then-DROP (e.g. a transient column) ends up
    correctly absent and a DROP-then-readd ends up present. Remaining over-capture
    can only relax the guard; under-capture would surface immediately as a false
    'missing column' in this test's own assertions.

    RENAME COLUMN is modelled too (migration 144, the repo's first). Unlike ADD,
    a rename is SUBTRACTIVE — it retires the old name — so it is the one statement
    that can turn over-capture into a false drift. Two things keep it sound: only
    forward-migration files are scanned (see :func:`_is_forward_migration`), and a
    rename applies only when the old name is currently modelled, so a rename whose
    source column this parser never saw cannot invent one. Renames performed by
    DYNAMIC SQL (``EXECUTE format(...)`` inside a ``DO`` block) remain invisible to
    any static reader — keep committed renames as plain statements, or the live
    EXPLAIN + FEAST_INTEGRATION backstops are the only thing left to catch them.
    """
    cols: dict[str, set[str]] = {}

    for sql_path in sorted(_DATABASE_DIR.rglob("*.sql")):
        if not _is_forward_migration(sql_path):
            continue
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
            # Apply ADD/DROP COLUMN in the exact order they appear in the
            # statement, so `DROP COLUMN x, ADD COLUMN x` nets to add and
            # `ADD COLUMN x, DROP COLUMN x` nets to drop (single ordered scan,
            # not all-adds-then-all-drops).
            for am in re.finditer(
                r"\b(ADD|DROP) COLUMN\s+(?:IF (?:NOT )?EXISTS\s+)?([a-z_][a-z0-9_]*)",
                stmt,
                re.IGNORECASE,
            ):
                if am.group(1).upper() == "ADD":
                    cols.setdefault(table, set()).add(am.group(2).lower())
                else:
                    cols.get(table, set()).discard(am.group(2).lower())

            # RENAME COLUMN <old> TO <new>, in the same ordered scan. Applied only
            # when <old> is currently modelled, so a rename this parser never saw
            # the source of cannot conjure a column out of nothing.
            for rm in re.finditer(
                r"\bRENAME COLUMN\s+(?:IF EXISTS\s+)?([a-z_][a-z0-9_]*)\s+TO\s+([a-z_][a-z0-9_]*)",
                stmt,
                re.IGNORECASE,
            ):
                old, new = rm.group(1).lower(), rm.group(2).lower()
                if old in cols.get(table, set()):
                    cols[table].discard(old)
                    cols[table].add(new)
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


class TestCommittedRenamesAreModelled:
    """Migration 144 is the repo's FIRST column rename, so this is the first thing
    ever to exercise renames in the DDL model (measured 2026-09-17: no other
    ``RENAME COLUMN`` exists under ``database/``).

    A rename is the one DDL statement that is SUBTRACTIVE — it retires a name.
    That is why it cannot simply be added to the ADD/DROP scan: the model's
    safety argument was "over-capture can only relax the guard", and a rename
    applied from a NON-FORWARD file (a rollback) would silently reverse a real
    rename and produce a FALSE 'missing column'.

    That reversal is REAL BUT CURRENTLY UNREACHABLE, and the reason is an
    asymmetry this lane created: forward 144 now spells its three table renames
    statically (so the model can see them) while rollback_144 still performs the
    reverse renames through ``EXECUTE format(...)``. Measured 2026-09-17 at this
    commit: 3 statically-readable renames in 144, 0 in rollback_144. So
    ``business_metrics`` is modelled identically with or without the forward-only
    filter — the filter's live effect is on five OTHER tables (see
    :func:`_is_forward_migration`).

    These tests are kept anyway, because the masking is one edit from gone:
    144 now carries a comment arguing that committed renames should be static,
    which makes "make the rollback match for consistency" a natural change — and
    that change arms the reversal immediately. The filter is what holds then.
    """

    def test_the_ddl_model_follows_a_committed_column_rename(self):
        available = _DDL.get("business_metrics", set())
        for new in (
            "triggers_delivered_count",
            "triggers_accepted_count",
            "triggers_total_count",
        ):
            assert new in available, f"{new} missing from the modelled schema"

    def test_the_retired_names_are_genuinely_absent(self):
        """Teeth in the FAIL direction: after 144 the old names do not exist, so a
        source query still naming one MUST be caught (the #556 class, post-rename)."""
        available = _DDL.get("business_metrics", set())
        for old in ("trx_count", "nrx_count", "total_rx_count"):
            assert old not in available, f"{old} was retired by migration 144"

    def test_a_non_forward_file_does_not_flip_the_model_back(self):
        """rollback_144 holds the REVERSE rename and sorts AFTER 144 ('1' < 'r').
        Without the forward-only filter it would undo the rename in the model."""
        rollback = _DATABASE_DIR / "migrations" / "rollback_144_per_hcp_trigger_count_columns.sql"
        assert rollback.exists(), "the trap this guards is gone; re-check the filter"
        assert not _is_forward_migration(rollback)
        assert "triggers_delivered_count" in _DDL.get("business_metrics", set())

    def test_excluding_non_forward_files_changes_the_model_in_both_directions(self):
        """MODEL-level teeth for the filter, on the tables it actually affects.

        The sibling rename assertions cannot provide these: business_metrics is
        modelled identically with or without the filter. These two flip when the
        filter is removed, one per harmful direction.

        Both columns were checked against the LIVE database on 2026-09-17
        (read-only, by the lane dispatcher): tool_performance.attempts EXISTS,
        tool_registry.success_rate DOES NOT. That is a snapshot of the live
        schema on that date, not an invariant — the assertions below are about
        the MODEL built from committed forward migrations, which is what this
        guard compares source queries against; the live check is corroboration
        that the model's answer is the true one.
        """
        # FALSE-FAIL direction: added by forward ml/041, dropped by its rollback.
        assert "attempts" in _DDL.get("tool_performance", set())
        # FALSE-PASS direction: dropped by forward ml/040:44, re-added by
        # ml/rollback_040.sql:57, which sorts AFTER it.
        assert "success_rate" not in _DDL.get("tool_registry", set())

    def test_the_forward_only_filter_is_not_vacuous(self):
        """A filter that excludes nothing would pass every test above by accident."""
        excluded = [p for p in _DATABASE_DIR.rglob("*.sql") if not _is_forward_migration(p)]
        assert excluded, "filter excluded no file at all"
        names = {p.name for p in excluded}
        assert "rollback_144_per_hcp_trigger_count_columns.sql" in names
        assert "011_validation_queries.sql" in names
        # ...and it must not swallow real forward migrations.
        assert _is_forward_migration(
            _DATABASE_DIR / "migrations" / "033_feast_canonical_schema.sql"
        )


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
