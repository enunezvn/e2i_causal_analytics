-- ROLLBACK for migration 157 (adaptive_validity_verdicts run key, #2260). NOT a forward
-- migration: scripts/run_migrations.sh skips rollback_*.sql in apply_dir().
--
-- Restores 040's unique index and drops the run-id column. Pair it with a code rollback:
-- post-157 mirror code names the four-expression conflict target and fails on 040's
-- index. Refuses (before touching anything) when two runs share an old key, since 040's
-- index could not be rebuilt over them; resolve those rows first.
--
-- Apply by hand:
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/migrations/rollback_157_adaptive_validity_verdicts_run_key.sql

DO $$
DECLARE
    n_collisions bigint;
BEGIN
    SELECT count(*) INTO n_collisions FROM (
        SELECT 1
        FROM adaptive_validity_verdicts
        GROUP BY COALESCE(experiment_id, '__unknown__'), COALESCE(feature, '__unknown__'), written_at
        HAVING count(*) > 1
    ) dup;
    IF n_collisions > 0 THEN
        RAISE EXCEPTION
            'rollback_157: % (experiment_id, feature, written_at) key(s) hold more than one run; 040''s unique index cannot be restored over them',
            n_collisions;
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS uix_adaptive_validity_verdicts_natural_key
    ON adaptive_validity_verdicts (
        COALESCE(experiment_id, '__unknown__'),
        COALESCE(feature, '__unknown__'),
        written_at
    );

DROP INDEX IF EXISTS uix_adaptive_validity_verdicts_run_key;

ALTER TABLE adaptive_validity_verdicts DROP COLUMN IF EXISTS audit_workflow_id;

-- run_migrations.sh skips any file already in the ledger; without this the next deploy
-- would silently skip re-applying 157.
DELETE FROM public.schema_migrations WHERE filename = '157_adaptive_validity_verdicts_run_key.sql';
