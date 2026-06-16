-- Migration 079: kpi_history — materialized per-month KPI values for the
-- Time-Series "KPI history" view.
--
-- WHY: the KPI engine computes a single point-in-time value per KPI; there is no
-- KPI time series. This table holds REAL monthly KPI points produced by the
-- walk-forward backfill (src/kpi/history_backfill.py) — either read directly from
-- an already-monthly source (e.g. WS3-BI-010 ROI <- business_metrics.roi) or
-- recomputed "as of" each month from a dated source table. Only genuinely
-- time-varying KPIs are populated; point-in-time KPIs are intentionally absent
-- (the UI shows an honest empty-state rather than a fabricated flat series).
--
-- The /api/kpis/{kpi_id}/history endpoint reads this table.

CREATE TABLE IF NOT EXISTS kpi_history (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kpi_id       TEXT NOT NULL,
    -- '' (empty) = global / all-brands or all-regions; keeps the UNIQUE key clean
    -- (NULLs would be treated as distinct and allow duplicate points).
    brand        TEXT NOT NULL DEFAULT '',
    region       TEXT NOT NULL DEFAULT '',
    metric_date  DATE NOT NULL,
    value        DOUBLE PRECISION NOT NULL,
    status       TEXT,                                  -- on_target | warning | critical | unknown
    -- Provenance of the point: how it was produced, for audit (no silent mocks).
    source       TEXT NOT NULL,                          -- e.g. 'business_metrics', 'asof:treatment_events'
    is_synthetic BOOLEAN NOT NULL DEFAULT TRUE,          -- backfill runs on synthetic-gold data
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_kpi_history_point UNIQUE (kpi_id, brand, region, metric_date)
);

CREATE INDEX IF NOT EXISTS idx_kpi_history_lookup
    ON kpi_history (kpi_id, brand, region, metric_date);

COMMENT ON TABLE kpi_history IS
    'Materialized monthly KPI values (real backfill / as-of recompute) for the time-series KPI-history view. Populated by src/kpi/history_backfill.py.';

-- PostgREST caches the schema; a freshly-created table is invisible to the REST
-- API (and thus to the backfill's upsert and the /api/kpis/{id}/history read)
-- until the cache is reloaded. Mirrors migration 074. NOTIFY is transactional —
-- it delivers on COMMIT, so it is safe inside run_migrations.sh's wrapped txn.
NOTIFY pgrst, 'reload schema';
