#!/usr/bin/env python3
"""Generate migration 066 (M4): rewrite every taggable kpi_query statement to
default-exclude synthetic rows, + a parallel *_include_synthetic opt-in family.

Deterministic subquery-wrap transform: each `FROM/JOIN <taggable> [alias]` becomes
`FROM/JOIN (SELECT * FROM <taggable> WHERE is_synthetic = false) <alias-or-table>`.
Using the table name as the subquery alias when none is given preserves every
qualified (`te.col` / `treatment_events.col`) and unqualified column reference.
"""
import json
import re
import subprocess
import sys

TAGGABLE = [
    "triggers", "business_metrics", "ml_predictions", "agent_activities",
    "causal_paths", "patient_journeys", "treatment_events", "hcp_profiles",
    "user_sessions", "hcp_intent_surveys", "episodic_memories",
    "ab_experiment_assignments",
]

KEYWORDS = {
    "where", "group", "order", "inner", "left", "right", "full", "cross",
    "join", "on", "using", "limit", "union", "except", "intersect", "having",
    "window", "offset", "fetch", "for", "returning", "as", "natural",
}


def dump_registry():
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres",
         "-tAc",
         "SELECT coalesce(json_agg(json_build_object("
         "'query_id', query_id, 'sql', sql, 'max_params', max_params)), '[]') "
         "FROM kpi_query_registry;"],
        capture_output=True, text=True, check=True,
    )
    return {r["query_id"]: r for r in json.loads(out.stdout)}


def wrap_table(sql: str, table: str) -> tuple[str, bool]:
    """Wrap each FROM/JOIN <table> [alias] with a synthetic-excluding subquery."""
    pattern = re.compile(
        r"\b(FROM|JOIN)\s+(?:public\.)?" + re.escape(table)
        + r"\b(\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
        re.IGNORECASE,
    )
    changed = False

    def repl(m):
        nonlocal changed
        kw = m.group(1)
        grp2 = m.group(2) or ""
        rawalias = m.group(3)
        if rawalias and rawalias.lower() not in KEYWORDS:
            alias, trailing = rawalias, ""
        else:
            alias, trailing = table, grp2  # put back a wrongly-consumed keyword
        changed = True
        return f"{kw} (SELECT * FROM {table} WHERE is_synthetic = false) {alias}{trailing}"

    return pattern.sub(repl, sql), changed


_UNWRAP = re.compile(
    r"\(SELECT \* FROM (?:public\.)?(\w+) WHERE is_synthetic = false\)(\s+)(\w+)"
)


def unwrap(sql: str) -> str:
    """Reverse a prior subquery-wrap so the generator is idempotent (self-heals a
    registry that was already wrapped, even multiply/nested). Iterates to a
    fixpoint: a double-wrap collapses to single on pass 1, to pristine on pass 2."""
    def r(m):
        table, ws, alias = m.group(1), m.group(2), m.group(3)
        return table if alias == table else f"{table}{ws}{alias}"
    prev = None
    while prev != sql:
        prev, sql = sql, _UNWRAP.sub(r, sql)
    return sql


def rewrite(sql: str) -> tuple[str, bool]:
    any_changed = False
    for t in TAGGABLE:
        sql, ch = wrap_table(sql, t)
        any_changed = any_changed or ch
    return sql, any_changed


def main():
    reg = dump_registry()
    base_rows, optin_rows, rewritten_ids = [], [], []
    for qid, row in sorted(reg.items()):
        orig = row["sql"]
        assert "$kpi$" not in orig and "$note$" not in orig, f"{qid} contains dollar tag"
        if qid.endswith("_include_synthetic"):
            continue  # never re-derive opt-in rows from opt-in rows
        orig = unwrap(orig)  # idempotent: clean any prior wrap before re-wrapping
        new_sql, changed = rewrite(orig)
        if not changed:
            continue  # not taggable (e.g. view-backed) -> leave untouched
        rewritten_ids.append(qid)
        mp = row["max_params"]
        base_rows.append((qid, new_sql, mp))
        optin_rows.append((qid + "_include_synthetic", orig, mp))

    def fmt(rows):
        parts = []
        for qid, sql, mp in rows:
            note = "M4: default-exclude synthetic" if not qid.endswith("_include_synthetic") \
                else "M4 opt-in: INCLUDES synthetic (validation runs only)"
            parts.append(f"    ('{qid}', $kpi${sql}$kpi$, {mp}, $note${note}$note$)")
        return ",\n".join(parts)

    header = (
        "-- ============================================================================\n"
        "-- Migration 066 (M4): default-exclude synthetic rows from the kpi_query RPC\n"
        "-- (synthetic-causal-validation). Clients pass only query_id -> the predicate\n"
        "-- MUST live in the server-side statement. Each taggable FROM/JOIN <t> is\n"
        "-- wrapped as (SELECT * FROM <t> WHERE is_synthetic = false) <alias> (alias-\n"
        "-- preserving, so qualified/unqualified column refs keep resolving). A parallel\n"
        "-- *_include_synthetic family (verbatim originals) is added for validation runs.\n"
        f"-- Auto-generated from the live registry; {len(rewritten_ids)} taggable statements.\n"
        "-- Idempotent (ON CONFLICT DO UPDATE). Depends on: 063 (M1 is_synthetic cols).\n"
        "-- ----------------------------------------------------------------------------\n"
    )
    sql_out = (
        header
        + "INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES\n"
        + fmt(base_rows)
        + "\nON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, "
          "max_params = EXCLUDED.max_params, note = EXCLUDED.note;\n\n"
        + "INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES\n"
        + fmt(optin_rows)
        + "\nON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, "
          "max_params = EXCLUDED.max_params, note = EXCLUDED.note;\n\n"
        + "NOTIFY pgrst, 'reload schema';\n"
    )
    dest = sys.argv[1]
    with open(dest, "w") as f:
        f.write(sql_out)
    print(f"wrote {dest}: {len(rewritten_ids)} taggable statements rewritten")
    print("rewritten_ids:", " ".join(sorted(rewritten_ids)))


if __name__ == "__main__":
    main()
