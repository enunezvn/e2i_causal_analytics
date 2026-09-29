-- ROLLBACK for migration 162 (hcp_adoption_goldstd_v, #2287). NOT a forward migration:
-- scripts/run_migrations.sh skips rollback_*.sql in apply_dir().
--
-- Drops the view and clears 162's ledger row. Refuses (before touching anything) while
-- any ml_model_registry row still names the view in its cohort contract: dropping it
-- under a live contract would turn the sweep's honest "no_cohort_contract" refusal into
-- a retrain that fails in the loader. Roll back migration 163 first. Both encodings of a
-- contract are checked: a bare table name and a table-cohort dict's JSON.
--
-- Apply by hand:
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/migrations/rollback_162_hcp_adoption_goldstd_view.sql

DO $$
DECLARE
    n_contracts bigint;
BEGIN
    SELECT count(*) INTO n_contracts
      FROM ml_model_registry
     WHERE cohort_data_source = 'hcp_adoption_goldstd_v'
        OR cohort_data_source LIKE '%"hcp_adoption_goldstd_v"%';
    IF n_contracts > 0 THEN
        RAISE EXCEPTION
            'rollback_162: % ml_model_registry row(s) still name hcp_adoption_goldstd_v in cohort_data_source; apply rollback_163 first',
            n_contracts;
    END IF;
END $$;

DROP VIEW IF EXISTS public.hcp_adoption_goldstd_v;

DELETE FROM public.schema_migrations WHERE filename = '162_hcp_adoption_goldstd_view.sql';

NOTIFY pgrst, 'reload schema';
