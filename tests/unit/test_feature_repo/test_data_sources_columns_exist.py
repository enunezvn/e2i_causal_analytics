"""Hermetic guard: every column a Feast PostgreSQLSource query references must
exist in the committed canonical schema DDL.

This runs WITHOUT the feast SDK and WITHOUT a live database — it AST-parses
``feature_repo/data_sources.py`` for the query strings and text-parses every
committed FORWARD ``*.sql`` under ``database/`` (CREATE TABLE + ADD/DROP COLUMN),
since the canonical columns are spread across the base schema and the migrations
(e.g. territory_metrics in 031, business_metrics' Feast columns in 033, and the
per_hcp_rollup count columns across the 144/145 expand/contract pair). So
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
from collections.abc import Iterable
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_DATA_SOURCES = _ROOT / "feature_repo" / "data_sources.py"
# Canonical schema DDL is spread across the base schema AND the migrations: the
# core tables live in database/core, but e.g. territory_metrics is created in
# migrations/031 and business_metrics' Feast columns are added in migrations/033.
# Scan every committed FORWARD *.sql under database/ (CREATE TABLE + ADD/DROP
# COLUMN). Over-capturing columns can only relax this guard — but a DROP RETIRES a
# name, so the scan is not purely additive and a stray reverse statement WOULD
# produce a false drift. That is why only files that actually reach the database
# on the forward path are scanned; see _is_forward_migration.
_DATABASE_DIR = _ROOT / "database"

# Forward DDL that a HUMAN applies, after the runner's pass — the contract half of
# an expand/contract pair (database/deferred/145, which retires the legacy
# per_hcp_rollup count columns that migration 144 expanded away from). These files
# are real forward DDL; they simply land in a LATER deploy than everything the
# runner applies, so the model must apply them LAST rather than in the
# alphabetical position their directory name happens to occupy.
#
# This is not cosmetic. Plain sorted order puts "database/deferred/145" ahead of
# "database/migrations/033", and 033 re-ADDs trx_count — so 145's DROP would be
# silently undone and the model would carry three columns the canonical schema
# retires. TestSchemaModelFollowsTheExpandContractPair::
# test_the_deferred_contract_is_applied_after_everything_the_runner_applies
# measures exactly that flip.
_DEFERRED_DIR = _DATABASE_DIR / "deferred"

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

    MEASURED EFFECT (2026-09-17, re-measured 2026-09-18). Stubbing this to ``True`` changes the modelled
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

    THIRD DIRECTION, added 2026-09-18 — and this one is on ``business_metrics``
    itself. The filter used to have no effect on that table, because rollback_144
    reversed a rename through ``EXECUTE format(...)``, which no static reader can
    see. Migration 144 is now an EXPAND, so its rollback is three plain
    ``ALTER TABLE public.business_metrics DROP COLUMN IF EXISTS
    triggers_{delivered,accepted,total}_count;`` statements — fully visible to this
    parser, and sorting AFTER ``144_*`` inside ``database/migrations/``. Unfiltered,
    the model would lose the three canonical columns every Feast source now selects
    and fail every source query. See
    TestSchemaModelFollowsTheExpandContractPair::test_the_rollback_does_not_unbuild_the_expand.
    """
    name = sql_path.name.lower()
    return not (
        name.startswith("rollback_") or "_rollback" in name or "_validation_queries" in name
    )


def _scan_order(paths: Iterable[Path]) -> list[Path]:
    """Sorted, except that by-hand ``database/deferred/`` files come last.

    The model applies statements in file order, so file order has to match the
    order in which the statements reach the database: everything the runner
    applies, then the contract migrations a human applies afterwards. See
    ``_DEFERRED_DIR``.
    """
    return sorted(paths, key=lambda p: (_DEFERRED_DIR in p.parents, str(p)))


def _ddl_columns(paths: Iterable[Path] | None = None) -> dict[str, set[str]]:
    """{table: columns} from base CREATE TABLE + ALTER TABLE ADD/DROP COLUMN.

    Files are processed in :func:`_scan_order` (≈ the order they reach the
    database) and ALTERs applied in statement order, so an ADD-then-DROP (e.g. a
    transient column) ends up correctly absent and a DROP-then-readd ends up
    present. Remaining over-capture can only relax the guard; under-capture would
    surface immediately as a false 'missing column' in this test's own assertions.

    DROP COLUMN is the one statement that is SUBTRACTIVE, which is what makes the
    scan order and the forward-only filter load-bearing rather than cosmetic: a
    DROP read from a file that never runs (a rollback), or read before the ADD it
    is meant to follow, produces a FALSE drift rather than a harmless extra
    column. See :func:`_is_forward_migration` and :func:`_scan_order`.

    ``paths``, when given, is used EXACTLY as passed — neither re-sorted nor
    re-filtered — so a test can build the model over a different file set or a
    different order and compare. That is the only way to show that the scan order
    and the forward-only filter change the answer rather than merely sounding like
    they should.

    Statements written in DYNAMIC SQL (``EXECUTE format(...)`` inside a ``DO``
    block) are invisible to any static reader; keep committed schema changes as
    plain statements, or the live EXPLAIN + FEAST_INTEGRATION backstops are the
    only thing left to catch them.
    """
    cols: dict[str, set[str]] = {}

    if paths is None:
        paths = _scan_order(p for p in _DATABASE_DIR.rglob("*.sql") if _is_forward_migration(p))

    for sql_path in paths:
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


class TestSchemaModelFollowsTheExpandContractPair:
    """Migration 144 is the repo's first EXPAND/CONTRACT pair, and the first thing
    ever to make this model's two ordering rules matter on a table a Feast source
    actually reads.

    144 (``database/migrations/``, applied by the runner) ADDs
    ``business_metrics.triggers_{delivered,accepted,total}_count`` beside the
    legacy ``{trx,nrx,total_rx}_count``. 145 (``database/deferred/``, applied by
    hand in a LATER deploy) retires the legacy three. The canonical schema a Feast
    source must target is the end state: canonical present, legacy gone.

    Both halves are plain, statically-readable statements, which is the whole
    reason this parser can see them — and it is also what arms two failure modes
    that were unreachable while 144 renamed through dynamic SQL. Each test below
    builds the model a second way and asserts the answer CHANGES, so none of them
    can pass for a reason unrelated to the rule it names.
    """

    def test_the_canonical_columns_are_modelled(self):
        available = _DDL.get("business_metrics", set())
        for canonical in (
            "triggers_delivered_count",
            "triggers_accepted_count",
            "triggers_total_count",
        ):
            assert canonical in available, f"{canonical} missing from the modelled schema"

    def test_the_legacy_names_are_retired_by_the_deferred_contract(self):
        """Teeth in the FAIL direction: the canonical schema these source queries
        are checked against is the POST-contract one, so a source still naming a
        legacy column must be caught (the #556 class)."""
        available = _DDL.get("business_metrics", set())
        for legacy in ("trx_count", "nrx_count", "total_rx_count"):
            assert legacy not in available, f"{legacy} is retired by database/deferred/145"

    def test_the_deferred_contract_is_applied_after_everything_the_runner_applies(self):
        """THE ORDERING TEETH. ``database/deferred/145`` sorts BEFORE
        ``database/migrations/033``, and 033 re-ADDs ``trx_count``. Under plain
        sorted order the contract's DROP is therefore undone by a migration that
        predates it by a hundred files, and the model silently carries three
        retired columns. Measured here rather than asserted in prose.
        """
        every = [p for p in _DATABASE_DIR.rglob("*.sql") if _is_forward_migration(p)]
        naive = _ddl_columns(sorted(every))["business_metrics"]
        ordered = _ddl_columns(_scan_order(every))["business_metrics"]
        assert "trx_count" in naive, (
            "plain sorted order no longer re-adds trx_count — the trap this rule "
            "guards has moved; re-derive it before trusting _scan_order"
        )
        assert "trx_count" not in ordered
        assert naive - ordered == {"trx_count", "nrx_count", "total_rx_count"}, naive - ordered

    def test_the_rollback_does_not_unbuild_the_expand(self):
        """rollback_144 now holds three STATIC ``DROP COLUMN`` statements against
        the columns 144 adds, and sorts AFTER ``144_*``. Without the forward-only
        filter the model would lose every canonical column the Feast sources
        select — a false drift on the live table, which is new: while 144 renamed
        dynamically, the filter had no effect on business_metrics at all."""
        rollback = _DATABASE_DIR / "migrations" / "rollback_144_per_hcp_trigger_count_columns.sql"
        assert rollback.exists(), "the trap this guards is gone; re-check the filter"
        assert not _is_forward_migration(rollback)
        assert "DROP COLUMN IF EXISTS triggers_delivered_count" in rollback.read_text(), (
            "the rollback no longer removes the expanded columns statically — this "
            "test's premise, and the filter's third measured direction, have moved"
        )
        unfiltered = _ddl_columns(_scan_order(_DATABASE_DIR.rglob("*.sql")))["business_metrics"]
        assert "triggers_delivered_count" not in unfiltered, (
            "dropping the forward-only filter no longer costs the canonical columns"
        )
        assert "triggers_delivered_count" in _DDL["business_metrics"]

    def test_excluding_non_forward_files_changes_the_model_in_both_directions(self):
        """MODEL-level teeth for the filter on two OTHER tables, one per harmful
        direction — kept because they are independent of anything this lane did.

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
        # The deferred contract is FORWARD DDL — it is deferred in ORDER, not
        # excluded. Confusing the two rules would retire nothing.
        assert _is_forward_migration(_DEFERRED_DIR / "145_drop_legacy_per_hcp_count_columns.sql")


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
