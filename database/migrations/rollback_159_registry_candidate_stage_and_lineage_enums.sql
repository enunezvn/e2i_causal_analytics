-- ROLLBACK for migration 159 (#2310, #2308). NOT a forward migration: scripts/run_migrations.sh
-- skips rollback_*.sql.
--
-- Postgres cannot drop an enum value. Removing 'candidate' / 'registered' would mean
-- rebuilding model_stage_enum / deployment_status_enum and every column, view and function
-- that uses them (ml_model_registry.stage, ml_deployments.status, the model-health views of
-- migrations 096/103, ml_bentoml_services.status). An unused extra value is harmless to the
-- pre-159 code (no reader lists the enum's values), so this rollback does not rebuild the
-- types. It REFUSES while any row still uses a 159 value -- roll back 160 first (its
-- rollback moves the backfilled rows off them) and move any later rows by hand -- and
-- otherwise leaves the values and 159's ledger row in place (re-applying 159 is a no-op).
--
-- Apply by hand:
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/migrations/rollback_159_registry_candidate_stage_and_lineage_enums.sql

DO $$
DECLARE
    v_candidates INTEGER;
    v_registered INTEGER;
BEGIN
    SELECT count(*) INTO v_candidates FROM ml_model_registry WHERE stage::text = 'candidate';
    SELECT count(*) INTO v_registered FROM ml_deployments WHERE status::text = 'registered';
    IF v_candidates > 0 OR v_registered > 0 THEN
        RAISE EXCEPTION
            'rollback 159 refused: % registry rows at stage candidate, % deployments at status registered',
            v_candidates, v_registered;
    END IF;
    RAISE NOTICE 'rollback 159: no row uses candidate/registered; the enum values stay (Postgres cannot drop them)';
END
$$;
