-- Migration 126: region-aware kpi_history coverage view (#1536)
-- =============================================================
-- The backfill now writes region-scoped series into kpi_history (region-only
-- and brand×region rows, mirroring the vetted live region variants of
-- migrations 077/078/113/125). The coverage view (migration 098) grouped by
-- (kpi_id, brand) over ALL rows — region rows would silently inflate a brand
-- scope's point count and duplicate brand entries in the coverage endpoint's
-- aggregation. Group by the full scope lattice instead; the endpoint keeps
-- the brand axis computed from region='' rows only (semantics unchanged) and
-- surfaces the lattice through a new `scopes` field.
--
-- Safe to apply BEFORE the code deploys: while every kpi_history row still
-- has region='' the view emits exactly one row per (kpi_id, brand), which the
-- pre-#1536 endpoint aggregates identically (the extra `region` column is
-- simply ignored by its row reads).

-- `region` is appended as the LAST column: CREATE OR REPLACE VIEW only allows
-- adding columns at the end (inserting mid-list errors), and consumers read
-- rows as dicts so column order is immaterial.
CREATE OR REPLACE VIEW v_kpi_history_coverage AS
SELECT
    kpi_id,
    brand,
    COUNT(*)         AS points,
    MIN(metric_date) AS first_date,
    MAX(metric_date) AS last_date,
    region
FROM kpi_history
GROUP BY kpi_id, brand, region;

COMMENT ON VIEW v_kpi_history_coverage IS
    'Coverage summary of kpi_history per (kpi_id, brand, region): point count + date span. Backs GET /api/kpis/history/coverage.';

-- PostgREST caches the schema; reload so the changed view shape is visible.
NOTIFY pgrst, 'reload schema';
