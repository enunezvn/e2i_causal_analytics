-- Migration 147: the Digital Twin cohort OUTCOME gets its own column on business_metrics
--
-- WHY: per_hcp_rollup rows have two writers. The per-HCP ETL
-- (src/etl/business_metrics_per_hcp_etl.py) recomputes conversion_rate on every upsert as
-- accepted / delivered triggers, a ratio <= 1. The synthetic-gold DGP backfill
-- (scripts/backfill_segment_engagement.py) planted its OUTCOME into that same column
-- (baseline + SUM tau_k * Tbin_k + noise; mean ~1.19, max ~4). Two writers, one column:
--   * every row the ETL touched lost its planted outcome while keeping its planted treatments,
--     so the twin would estimate an effect of ~0 from rows that still looked usable;
--   * re-planting made the ETL's own recompute preview report rows_changed on rows it had just
--     written, which the #2114 live certification asserts is zero;
--   * Feast's hcp_conversion_features served a "conversion rate" of up to 4.
-- On 2026-09-19 a full-window per-HCP backfill replaced the rows wholesale and the twin went
-- dark for every brand (usable cohort rows per brand ~4,000 -> 1-6, 500 needed).
--
-- The eight treatment channels (migrations 033 + 099) were never shared: the ETL inserts them as
-- NULL and its ON CONFLICT arm does not name them. This gives the outcome the same standing.
--
-- Additive + nullable, the pattern of migrations 033 and 099. No existing reader breaks: the column
-- stays NULL until the backfill --execute populates per_hcp_rollup rows, and until then the twin
-- reports the honest "no effect data" state it reports today. conversion_rate is left exactly as the
-- ETL writes it. The four split views are written `SELECT *`, which PostgreSQL froze into a column
-- list when 146 rebuilt them: they keep their 36 columns and simply do not expose this one. Nothing
-- reads the cohort through a view (the twin's loader selects from the table).
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS cohort_conversion_outcome NUMERIC;

COMMENT ON COLUMN business_metrics.cohort_conversion_outcome IS
  'per_hcp_rollup OUTCOME of the synthetic-gold multi-channel intervention DGP (scripts/backfill_segment_engagement.py): baseline(market_share, triggers_total_count, region) + SUM tau_k[region] * above-median(T_k) + noise. Read only by the Digital Twin cohort loader/estimator. NOT a rate (unbounded above) and NOT conversion_rate, which the per-HCP ETL owns.';
