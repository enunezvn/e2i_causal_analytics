-- ROLLBACK for migration 144 (canonical TRx lane). NOT a forward migration:
-- scripts/run_migrations.sh skips rollback_*.sql in apply_dir(), which is what
-- makes it safe to ship this file beside 144 -- were it applied in sequence, it
-- would rename the columns straight back and 144 would silently do nothing.
-- Apply by hand only when the deploy's container replacement failed (plan Task 29
-- Step 6): it restores the legacy per_hcp_rollup column names the pre-lane
-- containers read.
--
-- Symmetric with 144 and guarded the same way, so it is equally safe to re-run:
-- three table columns plus the same column on each of the four split views.
--
-- DELIBERATELY STILL DYNAMIC. Forward 144 spells its three TABLE renames as plain
-- ALTER TABLE ... RENAME COLUMN statements so that static readers of database/ can
-- see them (the hermetic Feast column guard,
-- tests/unit/test_feature_repo/test_data_sources_columns_exist.py, models the
-- schema by text-parsing these files). Making THIS file match "for consistency"
-- is the natural next edit -- and it would make the reverse renames statically
-- readable too. That is safe ONLY because that guard skips non-forward files via
-- _is_forward_migration, mirroring run_migrations.sh apply_dir(). If you make this
-- file static, read that filter first: without it the reverse renames would be
-- applied after 144's (rollback_* sorts after 144_*) and the guard would report
-- the renamed Feast source columns as absent.
DO $rollback$
DECLARE
    pair text[];
    split_view text;
BEGIN
    FOREACH pair SLICE 1 IN ARRAY ARRAY[
        ['triggers_delivered_count', 'trx_count'],
        ['triggers_accepted_count', 'nrx_count'],
        ['triggers_total_count', 'total_rx_count']
    ]::text[] LOOP
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = 'business_metrics'
               AND column_name = pair[1]
        ) THEN
            EXECUTE format('ALTER TABLE public.business_metrics RENAME COLUMN %I TO %I', pair[1], pair[2]);
        END IF;
        FOREACH split_view IN ARRAY ARRAY[
            'v_train_business_metrics', 'v_test_business_metrics',
            'v_validation_business_metrics', 'v_holdout_business_metrics'
        ] LOOP
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                 WHERE table_schema = 'public' AND table_name = split_view
                   AND column_name = pair[1]
            ) THEN
                EXECUTE format('ALTER VIEW public.%I RENAME COLUMN %I TO %I', split_view, pair[1], pair[2]);
            END IF;
        END LOOP;
    END LOOP;
END
$rollback$;

NOTIFY pgrst, 'reload schema';
