-- ============================================================================
-- Migration 114: goldstd holdout enlargement (backlog #44, plan B1) —
-- ml_split_registry e2i_pilot_v3 → v3.1.0: test_ratio 0.15→0.10,
-- holdout_ratio 0.05→0.10.
--
-- WHY: the WS1-MP-006 "Calibration Slope Deviation" KPI is red for
-- Remibrutinib because the gold-standard holdout is a small chronological
-- tail (n=415, ~5% row quota; observed slope 1.4455 = 99.6th percentile of
-- random same-size slices vs walk-forward truth ≈1.072). Doubling the quota
-- yields remi holdout n≈844 (slope SE ≈0.08).
--
-- MECHANISM (measured 2026-07-21): the live substrate's data_split is cut on
-- cumulative ROW share by src/ml/synthetic/generators/base.py::_assign_splits
-- (defaults now 60/20/10/10 — changed in lockstep with this migration), NOT
-- on this registry's date columns. The reseed (`reseed_synthetic.sh --full` →
-- load_synthetic_data.py --anchor-to-now) does NOT refresh this registry row;
-- it is metadata consumed by src/ml/data_loader.py's legacy load path and
-- scripts/validate_kpi_coverage.py, and must be updated here to stay honest.
--
-- UPDATE-IN-PLACE (not insert-new-row), deliberately:
--   * config_name is UNIQUE and consumers resolve the row by
--     config_name='e2i_pilot_v3'; a new row would need a new name and every
--     resolver would keep reading the stale row.
--   * split_config_id (56d62bc0-…) is the FK target of
--     ml_patient_split_assignments / ml_preprocessing_metadata /
--     ml_leakage_audit and is stamped on seeded rows; preserving the UUID
--     preserves referential continuity.
--   * The registry's own design versions in place (config_version +
--     updated_at; the row was already updated in place on 2025-12-02).
-- Date columns are intentionally untouched: they document the legacy fixed
-- band; the operative lever is the ratio quota (see notes stamp below).
--
-- Idempotent: unconditional-by-name UPDATE converges to the same terminal
-- state on re-run. Applied in the batched reseed window, BEFORE the --full
-- reseed, so registry metadata and substrate flip together.
-- ----------------------------------------------------------------------------

UPDATE ml_split_registry
SET config_version = '3.1.0',
    test_ratio     = 0.10,
    holdout_ratio  = 0.10,
    notes          = COALESCE(notes || E'\n', '')
                     || 'v3.1.0 (migration 114, backlog #44, 2026-07-21): test 0.15->0.10, '
                     || 'holdout 0.05->0.10 (goldstd holdout enlargement; remi holdout '
                     || 'n~415->~844). Ratios are the operative reseed lever '
                     || '(row-share quota in generators/base.py::_assign_splits); the date '
                     || 'columns document the legacy fixed band and are not enforced.',
    updated_at     = NOW()
WHERE config_name = 'e2i_pilot_v3'
  AND (config_version IS DISTINCT FROM '3.1.0');

-- Fail loudly if the expected row is absent (schema drift / wrong database).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM ml_split_registry
        WHERE config_name = 'e2i_pilot_v3' AND config_version = '3.1.0'
    ) THEN
        RAISE EXCEPTION
            'migration 114: ml_split_registry row config_name=e2i_pilot_v3 not found '
            'or not updated to 3.1.0 — expected the seeded pilot registry row';
    END IF;
END $$;

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
