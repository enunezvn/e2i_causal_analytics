-- ============================================================================
-- Migration 139: causal_validations.delta_percent DECIMAL(8,4) -> NUMERIC(12,4)
-- (#2029, lane 1 of the #1991 debts 3/4 wave; follows migration 138 / Lane G,
-- which clamped only the negative-control test's delta_percent).
-- ============================================================================
-- WHAT: widen public.causal_validations.delta_percent from DECIMAL(8,4)
--   (max 9999.9999, defined in database/ml/010_causal_validation_tables.sql)
--   to NUMERIC(12,4) (max 99999999.9999).
--
-- WHY: every refuter's delta_percent is |delta| / |original| * 100 and the
--   four perturbation tests compute it UNCLAMPED with a 1e-10 floor
--   (src/causal_engine/refutation_runner.py _run_placebo_test /
--   _run_random_common_cause_test / _run_data_subset_test /
--   _run_bootstrap_test). A near-zero original claim therefore overflows the
--   column (numeric field overflow, SQLSTATE 22003).
--
-- Blast radius: src/repositories/causal_validation.py save_suite() inserts
--   every row of a RefutationSuite in ONE Supabase call
--   (`.insert(rows).execute()`), so ONE overflowing row drops persistence for
--   the WHOLE suite (logged, returns []). Lane G clamped only the
--   negative-control test. The repository now clamps EVERY row at the write
--   boundary to this column's bound (src/repositories/causal_validation.py
--   DELTA_PERCENT_COLUMN_MAX) and keeps the exact value in
--   details_json.delta_percent_exact when clamping occurred.
--
-- SAFETY: on the live Postgres 15.8 (measured 2026-09-12), increasing the
--   precision of a numeric column while keeping the same scale is a
--   metadata-only ALTER (no table rewrite; rehearsed in BEGIN..ROLLBACK, see
--   docs/demos/results/2026-09-12_lane2029/rehearsal_139.txt). Idempotent:
--   the ALTER is guarded on information_schema so a re-run is a no-op.
--   No BEGIN/COMMIT of its own: scripts/run_migrations.sh wraps this file in
--   --single-transaction and records the ledger row in that same transaction.
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'causal_validations'
          AND column_name = 'delta_percent'
          -- Only ever WIDEN: a column someone later widened beyond 12 must not be narrowed back (a rewrite that could overflow).
          AND (numeric_precision IS NULL OR numeric_precision < 12 OR numeric_scale IS DISTINCT FROM 4)
    ) THEN
        ALTER TABLE public.causal_validations ALTER COLUMN delta_percent TYPE NUMERIC(12, 4);
    END IF;
END $$;

COMMENT ON COLUMN public.causal_validations.delta_percent IS
    'Percentage change from the original effect, |delta|/|original|*100, clamped by the '
    'repository to 99999999.9999 (migration 139, #2029); the exact value rides in '
    'details_json.delta_percent_exact when clamping occurred.';
