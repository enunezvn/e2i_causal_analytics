#!/usr/bin/env python3
"""Codegen for data-frontier anchoring of the rolling-window KPI registry rows.

WHY (2026-07-03): the synthetic gold-standard substrate is calendar-fixed BY
DESIGN -- ``SplitBoundaries`` (src/ml/synthetic/config.py) pins the simulation
to 2022-01-01..2024-12-31 (prescriptions trail to 2025-04-23) so the goldstd
models, walk-forward backtests, and causal ground-truth certifications stay
reproducible. The registry's rolling windows (``<col> >= NOW() - INTERVAL '30
days'``, migrations 044/046/066/077/078) were written when the substrate looked
fresh; against a static substrate they silently decay to EMPTY as wall-clock
time passes the seed date. COUNT(*) over an empty window returns 0 (not NULL),
so the engine's fail-loud "no data" path never fires -- the chatbot and the
Home tiles presented a fabricated-looking ``NBRx = 0.0``. Reseeding cannot fix
this (the generator reproduces the same fixed window) and shifting the window
forward would invalidate the gold standard.

THE FIX: re-register each affected row with its window anchored at the DATA
FRONTIER -- ``MAX(<domain timestamp>)`` over the query's own domain -- so the
figure means "the most recent 30 days of available data". The anchor domain
keeps the statement's synthetic scoping and content-type filters (e.g.
``event_type = 'prescription'``: anchoring Rx volumes on the consultation
frontier would re-create the empty-window bug) but drops brand/region params so
all brands/regions share one comparable as-of date.

TRANSFORM (exact-or-refuse, everything else byte-for-byte):

    SELECT base.*, (<anchor>)::date AS data_through FROM (<original
    statement with every NOW() replaced by (<anchor>)>) base

- every ``NOW()`` occurrence is replaced, with a per-id expected-count
  tripwire (conversion_rate/trx_share have 2; everything else 1);
- ``data_through`` rides along as an output column so consumers (chatbot
  ``kpi_calculate_tool``, Home ``get_kpi_summary``) can cite the real as-of
  date instead of implying wall-clock recency;
- arity (max_params) is untouched -- the RPC enforces param_count equality;
- the RPC's ``SELECT row_to_json(_sub) FROM (%s) AS _sub`` wrapper composes
  fine with the outer SELECT.

SKIPPED: ``business_impact_{mau,wau}_fallback[_include_synthetic]`` -- their
``user_sessions`` domain is REAL accruing app usage, where NOW() is correct.
NULL-frontier note: on a deployment whose domain is empty (e.g. the
synthetic-EXCLUDING variants on a synthetic-only instance) the window predicate
is NULL -> counts return 0 and ratios NULL, exactly as before this migration;
``data_through`` is NULL there, which is the honest signal.

Ground truth: the statements are recovered by REPLAYING the checked-in
migration files (last-writer-wins per query_id) -- verified byte-identical to
the live registry for all 74 NOW()-anchored rows on 2026-07-03. The replay
deliberately excludes this script's own output file so regeneration stays
idempotent.

Generate the migration with::

    python scripts/gen_kpi_frontier_anchoring.py > database/migrations/089_kpi_data_frontier_anchoring.sql
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "database" / "migrations"
MIGRATION_FILENAME = "089_kpi_data_frontier_anchoring.sql"
MIGRATION_PATH = MIGRATIONS_DIR / MIGRATION_FILENAME

# ---------------------------------------------------------------------------
# Frontier anchors: family -> {include_synthetic: anchor SELECT}
# The anchor mirrors the statement's own domain scoping (synthetic wrapper,
# content-type filter) minus brand/region params -- one shared frontier per
# domain so cross-brand/region figures stay comparable.
# ---------------------------------------------------------------------------

ANCHORS: dict[str, dict[bool, str]] = {
    # Prescription volumes (TRx/NRx/NBRx/share) + recall's positive-outcome
    # cohort: the frontier is the latest PRESCRIPTION, not any event -- the
    # 2026-06-20 consultation batch reaches 2026-06-09 while prescriptions end
    # 2025-04-23; anchoring on all events would re-create the empty window.
    "rx": {
        True: (
            "SELECT MAX(event_date) FROM treatment_events "
            "WHERE event_type::text = 'prescription'"
        ),
        False: (
            "SELECT MAX(event_date) FROM (SELECT * FROM treatment_events "
            "WHERE is_synthetic = false) treatment_events "
            "WHERE event_type::text = 'prescription'"
        ),
    },
    # HCP reach counts DISTINCT hcp_id across all event types -- its true
    # domain is events ATTRIBUTABLE TO AN HCP. Anchoring on the bare
    # all-events frontier lands on the 2026-06-20 consultation batch, whose
    # rows carry NO hcp_id (live-verified: reach degenerates to 0 for every
    # brand); the hcp_id IS NOT NULL filter is intrinsic to the metric.
    "treatment_events_hcp": {
        True: (
            "SELECT MAX(event_date) FROM treatment_events WHERE hcp_id IS NOT NULL"
        ),
        False: (
            "SELECT MAX(event_date) FROM (SELECT * FROM treatment_events "
            "WHERE is_synthetic = false) treatment_events WHERE hcp_id IS NOT NULL"
        ),
    },
    "triggers": {
        True: "SELECT MAX(trigger_timestamp) FROM triggers",
        False: (
            "SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers "
            "WHERE is_synthetic = false) triggers"
        ),
    },
    "ml_predictions_ts": {
        True: "SELECT MAX(prediction_timestamp) FROM ml_predictions",
        False: (
            "SELECT MAX(prediction_timestamp) FROM (SELECT * FROM ml_predictions "
            "WHERE is_synthetic = false) ml_predictions"
        ),
    },
    "ml_predictions_created": {
        True: "SELECT MAX(created_at) FROM ml_predictions",
        False: (
            "SELECT MAX(created_at) FROM (SELECT * FROM ml_predictions "
            "WHERE is_synthetic = false) ml_predictions"
        ),
    },
    "patient_journeys_created": {
        True: "SELECT MAX(created_at) FROM patient_journeys",
        False: (
            "SELECT MAX(created_at) FROM (SELECT * FROM patient_journeys "
            "WHERE is_synthetic = false) patient_journeys"
        ),
    },
    "business_metrics": {
        True: "SELECT MAX(metric_date) FROM business_metrics",
        False: (
            "SELECT MAX(metric_date) FROM (SELECT * FROM business_metrics "
            "WHERE is_synthetic = false) business_metrics"
        ),
    },
    "agent_activities": {
        True: "SELECT MAX(activity_timestamp) FROM agent_activities",
        False: (
            "SELECT MAX(activity_timestamp) FROM (SELECT * FROM agent_activities "
            "WHERE is_synthetic = false) agent_activities"
        ),
    },
    "hcp_intent_surveys": {
        True: "SELECT MAX(survey_date) FROM hcp_intent_surveys",
        False: (
            "SELECT MAX(survey_date) FROM (SELECT * FROM hcp_intent_surveys "
            "WHERE is_synthetic = false) hcp_intent_surveys"
        ),
    },
}

# ---------------------------------------------------------------------------
# Targets: query_id -> (anchor family, expected NOW() occurrences).
# Explicit enumeration (not suffix math): vetted SQL transforms are listed one
# by one so review sees exactly what is being re-registered.
# ---------------------------------------------------------------------------


def _expand(base: str, family: str, n_now: int, *, region: bool) -> dict[str, tuple[str, int]]:
    """base + _include_synthetic (+ _region twins when the base has them)."""
    ids = {base: (family, n_now), f"{base}_include_synthetic": (family, n_now)}
    if region:
        ids[f"{base}_region"] = (family, n_now)
        ids[f"{base}_region_include_synthetic"] = (family, n_now)
    return ids


TARGETS: dict[str, tuple[str, int]] = {
    # WS3 business-impact volumes & ratios (chatbot kpi_calculate_tool + Home tiles)
    **_expand("business_impact_trx", "rx", 1, region=True),
    **_expand("business_impact_nrx", "rx", 1, region=True),
    **_expand("business_impact_nbrx", "rx", 1, region=True),
    **_expand("business_impact_trx_share", "rx", 2, region=True),
    **_expand("business_impact_conversion_rate", "triggers", 2, region=True),
    **_expand("business_impact_hcp_reach", "treatment_events_hcp", 1, region=True),
    **_expand("business_impact_roi_business_metrics", "business_metrics", 1, region=False),
    **_expand("business_impact_roi_agent_activities", "agent_activities", 1, region=False),
    # WS2 trigger performance
    **_expand("trigger_performance_acceptance_rate", "triggers", 1, region=True),
    **_expand("trigger_performance_cfr", "triggers", 1, region=True),
    **_expand("trigger_performance_false_alert_rate", "triggers", 1, region=True),
    **_expand("trigger_performance_lead_time", "triggers", 1, region=True),
    **_expand("trigger_performance_override_rate", "triggers", 1, region=True),
    **_expand("trigger_performance_precision", "triggers", 1, region=True),
    # recall's window is on the PRESCRIPTION cohort (positive_outcomes CTE)
    **_expand("trigger_performance_recall", "rx", 1, region=True),
    # causal / model-performance / data-quality
    **_expand("causal_metrics_ate", "ml_predictions_ts", 1, region=False),
    **_expand("causal_metrics_cate", "ml_predictions_ts", 1, region=False),
    **_expand("model_performance_shap_coverage", "ml_predictions_created", 1, region=False),
    **_expand(
        "data_quality_completeness_pass_rate", "patient_journeys_created", 1, region=True
    ),
    # brand-specific probes
    **_expand("brand_specific_kisqali_oncologist_reach", "triggers", 1, region=False),
    **_expand("brand_specific_remi_intent_delta_fallback", "hcp_intent_surveys", 1, region=False),
}

# user_sessions is REAL accruing app usage -- NOW() is the correct anchor.
SKIPPED_LIVE_DOMAIN_IDS: frozenset[str] = frozenset(
    {
        "business_impact_mau_fallback",
        "business_impact_mau_fallback_include_synthetic",
        "business_impact_wau_fallback",
        "business_impact_wau_fallback_include_synthetic",
    }
)


# ---------------------------------------------------------------------------
# Migration replay: recover the current vetted text from the checked-in files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One kpi_query_registry row."""

    query_id: str
    sql: str
    max_params: int
    note: str | None


# ('id', $tag$sql$tag$, N, $tag2$note$tag2$ | NULL) -- dollar-quote tags vary
_ROW_RE = re.compile(
    r"\(\s*'(?P<id>[a-z0-9_]+)'\s*,\s*"
    r"\$(?P<tag>[A-Za-z0-9_]*)\$(?P<sql>.*?)\$(?P=tag)\$\s*,\s*"
    r"(?P<n>\d+)\s*,\s*"
    r"(?:\$(?P<ntag>[A-Za-z0-9_]*)\$(?P<note>.*?)\$(?P=ntag)\$|NULL)\s*\)",
    re.DOTALL,
)

# UPDATE ... SET sql = $tag$...$tag$ WHERE query_id = 'id' (e.g. migration 080)
_UPDATE_RE = re.compile(
    r"UPDATE\s+(?:public\.)?kpi_query_registry\s+"
    r"SET\s+sql\s*=\s*\$(?P<tag>[A-Za-z0-9_]*)\$(?P<sql>.*?)\$(?P=tag)\$\s*"
    r"WHERE\s+query_id\s*=\s*'(?P<id>[a-z0-9_]+)'",
    re.DOTALL | re.IGNORECASE,
)


def replay_registry(migrations_dir: Path = MIGRATIONS_DIR) -> dict[str, Row]:
    """Rebuild the registry state by replaying migrations in filename order.

    Excludes this script's own output (089) so regeneration never re-transforms
    already-transformed SQL -- the transform would fail loud anyway (no NOW()
    left to replace), but excluding it keeps replay a faithful "pre-089" view.
    """
    rows: dict[str, Row] = {}
    for path in sorted(migrations_dir.glob("*.sql")):
        if path.name == MIGRATION_FILENAME:
            continue
        text = path.read_text()
        if "kpi_query_registry" not in text:
            continue
        for m in _ROW_RE.finditer(text):
            rows[m.group("id")] = Row(
                query_id=m.group("id"),
                sql=m.group("sql"),
                max_params=int(m.group("n")),
                note=m.group("note"),
            )
        for m in _UPDATE_RE.finditer(text):
            prior = rows.get(m.group("id"))
            rows[m.group("id")] = Row(
                query_id=m.group("id"),
                sql=m.group("sql"),
                max_params=prior.max_params if prior else 0,
                note=prior.note if prior else None,
            )
    return rows


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def transform_statement(query_id: str, sql: str) -> str:
    """Frontier-anchor one vetted statement (exact-or-refuse)."""
    family, expected_now = TARGETS[query_id]
    include_synthetic = query_id.endswith("_include_synthetic")
    anchor = ANCHORS[family][include_synthetic]

    found = sql.count("NOW()")
    if found != expected_now:
        raise ValueError(
            f"{query_id}: found {found} NOW() occurrence(s), expected {expected_now} -- "
            f"the vetted base drifted; refusing to emit a mis-anchored statement"
        )
    body = sql.replace("NOW()", f"({anchor})")
    if "NOW()" in body:
        raise ValueError(f"{query_id}: NOW() survived the transform")
    return f"SELECT base.*, ({anchor})::date AS data_through FROM ({body}) base"


def generate_rows() -> list[Row]:
    """All 70 frontier-anchored rows, in stable (sorted) order."""
    replayed = replay_registry()
    missing = sorted(set(TARGETS) - set(replayed))
    if missing:
        raise KeyError(f"target ids missing from migration replay: {missing}")
    out: list[Row] = []
    for query_id in sorted(TARGETS):
        base = replayed[query_id]
        note_prefix = f"{base.note}; " if base.note else ""
        out.append(
            Row(
                query_id=query_id,
                sql=transform_statement(query_id, base.sql),
                max_params=base.max_params,
                note=f"{note_prefix}089 frontier-anchored (window ends at domain MAX, not NOW())",
            )
        )
    return out


def render_migration() -> str:
    """Render the idempotent INSERT ... ON CONFLICT migration (style of 084)."""
    header = """\
-- ============================================================================
-- 089_kpi_data_frontier_anchoring.sql
-- Re-register the rolling-window KPI statements to anchor at the DATA FRONTIER
-- (MAX(<domain timestamp>) over each query's own domain) instead of NOW().
--
-- WHY: the synthetic gold-standard substrate is calendar-fixed BY DESIGN
-- (SplitBoundaries 2022-01-01..2024-12-31; prescriptions trail to 2025-04-23)
-- so goldstd models / backtests / causal certifications stay reproducible.
-- `NOW() - INTERVAL` windows against that static substrate silently decayed to
-- empty as wall-clock time passed the seed date: COUNT(*) returns 0 (not
-- NULL), the engine's fail-loud path never fires, and the chatbot / Home tiles
-- presented a fabricated-looking 0.0 (observed 2026-07-03: Kisqali NBRx).
-- Reseeding cannot fix this (the generator reproduces the same fixed window);
-- shifting the seed window forward would invalidate the gold standard.
--
-- WHY IN-PLACE (unlike additive 077/084): the NOW() semantics of these rows
-- ARE the defect -- every consumer (chatbot kpi_calculate_tool, Home
-- get_kpi_summary tiles, KPI grid) needs the healed meaning. The explicit
-- `*_windowed*` variants (084) are untouched: user-supplied absolute windows
-- keep their meaning.
--
-- TRANSFORM (codegen, exact-or-refuse):
--   SELECT base.*, (<anchor>)::date AS data_through
--   FROM (<original with every NOW() -> (<anchor>)>) base
-- The anchor keeps each statement's synthetic scoping and content-type filters
-- (Rx volumes anchor on the latest PRESCRIPTION, not any event) but drops
-- brand/region params so all brands share one comparable as-of date. The new
-- `data_through` output column lets consumers cite the real as-of date.
-- Arity (max_params) is unchanged. On an empty domain the frontier is NULL:
-- counts return 0 and ratios NULL exactly as before; data_through NULL.
--
-- SKIPPED: business_impact_{mau,wau}_fallback* -- user_sessions is real
-- accruing app usage; NOW() is correct there.
--
-- AUTO-GENERATED by scripts/gen_kpi_frontier_anchoring.py (do not hand-edit;
-- regenerate). Idempotent (ON CONFLICT DO UPDATE). Depends on: 044 (registry
-- + RPC), 046 (brand-specific), 066 (synthetic twins), 077/078 (region
-- variants), 080 (dq denominator).
-- ============================================================================

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
"""
    rows = []
    for v in generate_rows():
        rows.append(
            f"    ('{v.query_id}', $kpi${v.sql}$kpi$, {v.max_params}, $note${v.note}$note$)"
        )
    body = ",\n".join(rows)
    footer = (
        "\nON CONFLICT (query_id) DO UPDATE SET "
        "sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;\n"
        "\n-- PostgREST caches the schema; reload so the updated rows are visible.\n"
        "NOTIFY pgrst, 'reload schema';\n"
    )
    return header + body + footer


def main(argv: list[str] | None = None) -> int:
    """Write the migration SQL to stdout (redirect into the migration file)."""
    sys.stdout.write(render_migration())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
