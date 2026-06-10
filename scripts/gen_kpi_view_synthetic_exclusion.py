#!/usr/bin/env python3
"""Generate migration 067: close the view-backed kpi_query leak codex flagged.
(1) add is_synthetic to the 3 view-backed tables M1 missed; (2) CREATE OR REPLACE
the 8 KPI views that read a taggable table, wrapping each taggable FROM/JOIN as
(SELECT * FROM <t> WHERE is_synthetic = false) <alias> (alias-preserving)."""

import re
import subprocess
import sys

NEW_TAG_TABLES = ["data_source_tracking", "etl_pipeline_metrics", "ml_annotations"]

# All is_synthetic-bearing tables (M1's 12 + the 3 added here).
TAGGABLE = [
    "triggers",
    "business_metrics",
    "ml_predictions",
    "agent_activities",
    "causal_paths",
    "patient_journeys",
    "treatment_events",
    "hcp_profiles",
    "user_sessions",
    "hcp_intent_surveys",
    "episodic_memories",
    "ab_experiment_assignments",
] + NEW_TAG_TABLES

# KPI views that read a taggable table (confirmed leaky / KPI-view infra).
VIEWS = [
    "v_patient_eligibility",
    "v_kpi_active_users",
    "v_kpi_intent_to_prescribe",
    "v_kpi_data_lag",
    "v_kpi_cross_source_match",
    "v_kpi_stacking_lift",
    "v_kpi_time_to_release",
    "v_kpi_change_fail_rate",
]

KEYWORDS = {
    "where",
    "group",
    "order",
    "inner",
    "left",
    "right",
    "full",
    "cross",
    "join",
    "on",
    "using",
    "limit",
    "union",
    "except",
    "intersect",
    "having",
    "window",
    "offset",
    "fetch",
    "for",
    "returning",
    "as",
    "natural",
}


def psql(sql):
    return subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tAc", sql],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def wrap_table(sql, table):
    pat = re.compile(
        r"\b(FROM|JOIN)\s+(?:public\.)?"
        + re.escape(table)
        + r"\b(\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
        re.IGNORECASE,
    )

    def repl(m):
        kw, grp2, rawalias = m.group(1), m.group(2) or "", m.group(3)
        if rawalias and rawalias.lower() not in KEYWORDS:
            alias, trailing = rawalias, ""
        else:
            alias, trailing = table, grp2
        return f"{kw} (SELECT * FROM {table} WHERE is_synthetic = false) {alias}{trailing}"

    return pat.sub(repl, sql)


def main():
    parts = [
        "-- ============================================================================",
        "-- Migration 067: close the view-backed kpi_query synthetic leak (codex HIGH).",
        "-- (1) is_synthetic on the 3 view-backed tables M1 missed; (2) CREATE OR REPLACE",
        "-- the KPI views that read a taggable table to default-exclude synthetic rows.",
        "-- Wrap is alias-preserving so view output columns are unchanged (CREATE OR",
        "-- REPLACE safe). Idempotent. Depends on: 063 (M1 is_synthetic columns).",
        "-- ----------------------------------------------------------------------------",
        "",
    ]
    for t in NEW_TAG_TABLES:
        parts.append(
            f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;"
        )
    parts.append("")
    for v in VIEWS:
        body = (
            psql(f"SELECT pg_get_viewdef('public.{v}'::regclass, true);")
            .strip()
            .rstrip(";")
            .strip()
        )
        for t in TAGGABLE:
            body = wrap_table(body, t)
        parts.append(f"CREATE OR REPLACE VIEW public.{v} AS\n{body};")
        parts.append("")
    parts.append("NOTIFY pgrst, 'reload schema';")
    parts.append("-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)")
    with open(sys.argv[1], "w") as f:
        f.write("\n".join(parts) + "\n")
    print(f"wrote {sys.argv[1]}: {len(NEW_TAG_TABLES)} tagged tables, {len(VIEWS)} views patched")


if __name__ == "__main__":
    main()
