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
