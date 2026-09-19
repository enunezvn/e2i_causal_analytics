"""Migration 143: canonical Rx-volume statements over business_metrics (canonical TRx lane)."""

import re
from pathlib import Path

import pytest

from scripts import gen_canonical_volume_registry as gen
from src.kpi.measure_basis import tables_in_sql

MIGRATION = (
    Path(__file__).resolve().parents[3]
    / "database"
    / "migrations"
    / "143_canonical_volume_kpis.sql"
)
ROWS = gen.registry_rows()


def test_committed_migration_is_the_generator_output():
    # BYTES, not text: a CRLF rewrite changes the file while read_text() still matches.
    assert MIGRATION.read_bytes() == gen.render().encode("utf-8")


def test_seventeen_bases_each_with_one_synthetic_twin():
    ids = [qid for qid, *_ in ROWS]
    assert len(ids) == len(set(ids)) == 34
    bases = {i for i in ids if not i.endswith("_include_synthetic")}
    assert len(bases) == 17
    assert {b + "_include_synthetic" for b in bases} == set(ids) - bases
    for metric in ("trx", "nrx", "nbrx", "trx_share"):
        for variant in ("", "_region", "_windowed", "_windowed_region"):
            assert f"canonical_volume_{metric}{variant}" in bases
    assert "canonical_volume_monthly_series" in bases


@pytest.mark.parametrize("qid,sql,max_params,note", ROWS, ids=[r[0] for r in ROWS])
def test_statement_contract(qid, sql, max_params, note):
    # 044's CHECK: read-only statements only.
    assert re.match(r"^\s*(with|select)\s", sql, re.IGNORECASE)
    # The derived substrate must be exactly business_metrics (measure_basis reads this).
    assert tables_in_sql(sql) == ["business_metrics"]
    placeholders = {int(p) for p in re.findall(r"\$(\d+)", sql)}
    assert placeholders == set(range(1, max_params + 1)) and max_params <= 4
    # Never serves the in-progress calendar month.
    assert gen.IN_PROGRESS_MONTH in sql
    if qid.endswith("_include_synthetic"):
        assert "is_synthetic = false" not in sql
    else:
        assert "is_synthetic = false" in sql
    assert note


def test_region_statements_filter_case_insensitively():
    for qid, sql, _, _ in ROWS:
        if "_region" in qid:
            assert "LOWER(bm.region::text) = LOWER($2)" in sql, qid


def test_windowed_statements_count_only_months_fully_inside_the_window():
    for qid, sql, _, _ in ROWS:
        if "_windowed" in qid:
            assert "bm.metric_date >= $" in sql, qid
            assert "+ INTERVAL '1 month' - INTERVAL '1 day')::date <= $" in sql, qid
            assert "months_in_window" in sql, qid


def test_the_rekey_is_collision_guarded_and_records_what_it_moved():
    """codex r1 HIGH (collisions) + codex r2 MEDIUM (payload = all four columns;
    provenance recorded so rollback restores only moved rows)."""
    text = MIGRATION.read_text()
    audit, block = gen.audit_table_sql(), gen.rekey_sql()
    assert audit in text and block in text and text.index(audit) < text.index(block)
    assert gen.REKEY_AUDIT_TABLE == "public.kpi_history_rekey_143"
    assert "RAISE EXCEPTION" in block and "re-key refused" in block
    assert gen.PAYLOAD_COLUMNS == ("value", "status", "source", "is_synthetic")
    for column in gen.PAYLOAD_COLUMNS:
        assert f"d.{column} IS NOT DISTINCT FROM s.{column}" in block, column
    assert "'absorbed'" in block and "'moved'" in block
    # only recorded source rows are deleted
    assert f"USING {gen.REKEY_AUDIT_TABLE} a" in block
    assert not re.search(r"UPDATE\s+public\.kpi_history", text)
    for old, new in gen.KPI_HISTORY_REKEY:
        assert f"['{old}', '{new}']" in block


def test_notes_describe_the_aggregation_each_statement_performs():
    """Nothing reads kpi_query_registry.note (measure_basis selects query_id,sql only),
    but a note that misdescribes its own SQL is a trap for the next reader: the windowed
    statements sum the months inside a window, they do not serve one month."""
    for qid, _sql, _n, note in ROWS:
        if "_windowed" in qid:
            assert "latest complete month" not in note, qid
            assert "summed over complete months fully inside [start, end]" in note, qid
        elif "monthly_series" not in qid:
            assert "for the latest complete month at the global TRx frontier" in note, qid
        if "_region" in qid:
            assert "for the specified region" in note, qid
        if "trx_share" in qid:
            assert "brand TRx / portfolio TRx" in note, qid


def test_the_rollback_restores_only_what_the_migration_moved():
    path = MIGRATION.parent / "rollback_143_canonical_volume_kpis.sql"
    rollback = path.read_text()
    assert path.read_bytes() == gen.rollback_render().encode("utf-8")
    restore = gen.restore_sql()
    assert restore in rollback
    assert "a.disposition = 'moved' AND t.id = a.dest_history_id" in restore
    for column in gen.PAYLOAD_COLUMNS:
        assert f"t.{column} IS NOT DISTINCT FROM a.{column}" in restore, column
    assert f"DROP TABLE {gen.REKEY_AUDIT_TABLE};" in restore
    assert "DELETE FROM public.kpi_query_registry WHERE query_id IN (" in rollback
    # canonical history is removed BEFORE the event rows are restored, so a restored
    # row cannot collide with it
    assert rollback.index("source = 'business_metrics.value'") < rollback.index("DO $restore$")

    # The ledger row is retired by the rollback ITSELF, as its last statement before
    # NOTIFY, so that under `psql --single-transaction` the schema change and the
    # ledger move commit together (codex iter3 HIGH-2). Doing it as a second psql
    # invocation left a window in which 143 was undone while the runner still
    # believed it applied, so the next deploy would skip re-applying it.
    #
    # codex iter4 MED-1: this has to be asserted SEMANTICALLY. The byte-equality
    # check on line 106 compares the committed file against `rollback_render()`, so
    # it says the file matches the generator -- it says nothing about what either
    # one CONTAINS. Deleting the DELETE from `rollback_render()` and regenerating
    # keeps both sides equal and left every other assertion here green; the property
    # would have regressed without a red test. Assert the content, not the agreement.
    ledger = re.findall(
        r"DELETE FROM public\.schema_migrations WHERE filename = '([^']+)';", rollback
    )
    assert ledger == ["143_canonical_volume_kpis.sql"], (
        f"the rollback does not retire exactly its own ledger row: {ledger}"
    )
    tail = rollback[rollback.index("DELETE FROM public.schema_migrations") :]
    assert "DROP TABLE" not in tail and "DO $restore$" not in tail, (
        "the ledger DELETE runs before some schema statement; a failure in between "
        "would leave the ledger claiming a rollback that did not finish"
    )


def test_every_read_of_the_table_applies_the_null_dimension_rule():
    """codex r2 HIGH: headline, share, frontier, windowed and series statements all
    exclude rows with a NULL brand or region, exactly like the history handler."""
    from src.kpi.volume_family import DIMENSIONED_ROW_SQL

    for qid, sql, _, _ in ROWS:
        reads = sql.count("FROM business_metrics")
        assert (
            reads >= 1 and sql.count(f"FROM business_metrics WHERE {DIMENSIONED_ROW_SQL}") == reads
        ), qid


def test_the_twinned_bases_are_in_the_resolver_set():
    from src.kpi.synthetic_mode import SYNTHETIC_TWINNED_QUERY_IDS

    bases = {qid for qid, *_ in ROWS if not qid.endswith("_include_synthetic")}
    assert bases <= SYNTHETIC_TWINNED_QUERY_IDS
