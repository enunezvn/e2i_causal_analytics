"""Golden tests for the data-frontier anchoring codegen (migration 089).

These pin the SECURITY-SENSITIVE transform that re-registers the rolling-window
KPI registry rows to anchor at their domain's data frontier instead of NOW().

WHY (2026-07-03 incident): the synthetic gold-standard substrate is
calendar-fixed BY DESIGN (SplitBoundaries 2022-01-01..2024-12-31; prescriptions
trail to 2025-04-23) so the goldstd models / walk-forward backtests / causal
ground-truth certifications stay reproducible. ``NOW() - INTERVAL`` windows
against that static substrate silently decay to empty as wall-clock time passes
the seed date: COUNT(*) returns 0 (not NULL), so the engine's fail-loud path
never fires and the chatbot/Home tiles present a fabricated-looking 0.0
(observed: Kisqali NBRx). The fix anchors each rolling window at
``MAX(<domain ts>)`` over the query's own domain -- "the most recent 30 days of
data" -- and surfaces the anchor as a ``data_through`` output column so every
consumer can cite the real as-of date.

The transform must be exact-or-refuse: every NOW() occurrence replaced (with a
per-id expected-count tripwire), nothing else rewritten, and the emitted
migration byte-identical to the checked-in file (regeneration drift guard).
"""

from pathlib import Path

import pytest

from scripts.gen_kpi_frontier_anchoring import (
    ANCHORS,
    MIGRATION_PATH,
    SKIPPED_LIVE_DOMAIN_IDS,
    TARGETS,
    generate_rows,
    render_migration,
    replay_registry,
)

ROWS = {row.query_id: row for row in generate_rows()}


# ---------------------------------------------------------------------------
# Coverage: exactly the 70 stale-tail-prone ids; live-domain ids untouched
# ---------------------------------------------------------------------------


def test_emits_exactly_the_target_ids():
    assert set(ROWS) == set(TARGETS)
    assert len(ROWS) == 70


def test_live_session_domains_are_skipped():
    # user_sessions is REAL app usage -- an accruing live feed. NOW() is the
    # correct anchor there; frontier-anchoring MAU/WAU would freeze them.
    assert SKIPPED_LIVE_DOMAIN_IDS == frozenset(
        {
            "business_impact_mau_fallback",
            "business_impact_mau_fallback_include_synthetic",
            "business_impact_wau_fallback",
            "business_impact_wau_fallback_include_synthetic",
        }
    )
    assert not SKIPPED_LIVE_DOMAIN_IDS & set(ROWS)


def test_drift_tripwire_no_unclassified_now_rows():
    """Every NOW()-anchored registry row must be a TARGET or an explicit SKIP.

    A future migration adding a rolling-window row forces a decision here
    instead of silently re-introducing wall-clock decay.
    """
    replayed = replay_registry()
    now_ids = {qid for qid, row in replayed.items() if "NOW()" in row.sql}
    assert now_ids == set(TARGETS) | SKIPPED_LIVE_DOMAIN_IDS


# ---------------------------------------------------------------------------
# Transform shape: wrapper + full NOW() replacement + provenance column
# ---------------------------------------------------------------------------


def test_every_row_is_fully_transformed():
    replayed = replay_registry()
    for qid, row in ROWS.items():
        assert "NOW()" not in row.sql, qid
        assert row.sql.startswith("SELECT base.*, ("), qid
        assert ")::date AS data_through FROM (" in row.sql, qid
        assert row.sql.endswith(") base"), qid
        # arity is untouched -- the RPC enforces param_count == max_params
        assert row.max_params == replayed[qid].max_params, qid


def test_nbrx_include_synthetic_matches_design():
    sql = ROWS["business_impact_nbrx_include_synthetic"].sql
    anchor = ANCHORS["rx"][True]
    assert anchor == (
        "SELECT MAX(event_date) FROM treatment_events WHERE event_type::text = 'prescription'"
    )
    # window now ends at the prescription frontier, not wall-clock now
    assert f"first_date >= ({anchor}) - INTERVAL '30 days'" in sql
    # everything else preserved byte-for-byte from the vetted base
    assert "WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date" in sql
    assert "($1::text IS NULL OR brand::text = $1)" in sql
    assert ROWS["business_impact_nbrx_include_synthetic"].max_params == 1


def test_exclude_variants_anchor_on_synthetic_excluded_domain():
    sql = ROWS["business_impact_nbrx"].sql
    assert ANCHORS["rx"][False] in sql
    assert "(SELECT * FROM treatment_events WHERE is_synthetic = false)" in ANCHORS["rx"][False]


def test_conversion_rate_replaces_both_windows():
    # conversion_rate has TWO NOW() windows (triggered + converted CTEs); both
    # must anchor to the SAME triggers frontier, plus one data_through output.
    sql = ROWS["business_impact_conversion_rate_include_synthetic"].sql
    anchor = f"({ANCHORS['triggers'][True]})"
    assert sql.count(anchor) == 3  # 2 window predicates + 1 data_through
    assert "NOW()" not in sql


def test_trx_share_replaces_both_windows():
    sql = ROWS["business_impact_trx_share_include_synthetic"].sql
    assert sql.count(f"({ANCHORS['rx'][True]})") == 3


def test_recall_anchors_on_prescriptions_not_triggers():
    # trigger_performance_recall windows on treatment_events prescriptions
    # (the positive-outcome cohort), so its frontier is the Rx frontier.
    sql = ROWS["trigger_performance_recall_include_synthetic"].sql
    assert ANCHORS["rx"][True] in sql
    assert "MAX(trigger_timestamp)" not in sql


def test_per_family_anchor_columns():
    assert "MAX(trigger_timestamp)" in ROWS["trigger_performance_precision"].sql
    assert "MAX(prediction_timestamp)" in ROWS["causal_metrics_ate_include_synthetic"].sql
    assert "MAX(created_at)" in ROWS["model_performance_shap_coverage_include_synthetic"].sql
    assert "MAX(created_at)" in ROWS["data_quality_completeness_pass_rate_include_synthetic"].sql
    assert "MAX(metric_date)" in ROWS["business_impact_roi_business_metrics_include_synthetic"].sql
    assert "MAX(activity_timestamp)" in ROWS["business_impact_roi_agent_activities"].sql
    assert "MAX(survey_date)" in ROWS["brand_specific_remi_intent_delta_fallback"].sql
    # hcp_reach counts DISTINCT hcp_id -> the frontier of HCP-ATTRIBUTABLE
    # events (the 2026-06-20 consultation batch has no hcp_id; anchoring on
    # the bare all-events frontier degenerated reach to 0, live-verified)
    anchor = ANCHORS["treatment_events_hcp"][True]
    assert "hcp_id IS NOT NULL" in anchor
    assert anchor in ROWS["business_impact_hcp_reach_include_synthetic"].sql


def test_multirow_grouped_query_keeps_shape():
    # CATE is a GROUP BY multi-row result; the wrapper must preserve grouping
    # and ordering while adding the constant data_through column.
    sql = ROWS["causal_metrics_cate_include_synthetic"].sql
    assert "GROUP BY segment_assignment" in sql
    assert "ORDER BY AVG(heterogeneous_effect) DESC" in sql
    assert ROWS["causal_metrics_cate_include_synthetic"].max_params == 1


def test_transform_refuses_unexpected_now_count():
    from scripts.gen_kpi_frontier_anchoring import transform_statement

    with pytest.raises(ValueError, match="expected 2"):
        transform_statement(
            "business_impact_conversion_rate",
            "SELECT 1 WHERE x >= NOW() - INTERVAL '30 days'",
        )


# ---------------------------------------------------------------------------
# Migration file: idempotent, regenerable, checked-in byte-for-byte
# ---------------------------------------------------------------------------


def test_migration_is_idempotent_and_reloads_postgrest():
    out = render_migration()
    assert "ON CONFLICT (query_id) DO UPDATE" in out
    assert "NOTIFY pgrst, 'reload schema';" in out


def test_checked_in_migration_matches_regeneration():
    """The checked-in 089 file must be exactly what the script emits."""
    assert Path(MIGRATION_PATH).read_text() == render_migration()


def test_replay_excludes_own_output():
    """Replay must ignore 089 itself, or regeneration after the migration
    lands would try to transform already-transformed SQL."""
    replayed = replay_registry()
    sql = replayed["business_impact_nbrx_include_synthetic"].sql
    assert "NOW()" in sql  # still the pre-089 text
    assert "data_through" not in sql
