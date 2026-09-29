-- Migration 160: retrain lineage + exact MLflow version on ml_model_registry, and the backfill
-- of the rows retrains already wrote (#2310, #2311, #2308). Needs 159's enum values, which
-- the runner commits before this file runs (159 is applied un-wrapped; this file is wrapped).
--
-- WHY (#2310): a retrain's registry row carries no link to the model it retrains, and it
-- shares stage 'staging' with the gold-standard reference rows, so readers cannot tell the
-- two apart. Lineage and role are recorded separately:
--   * lineage: retrain_of_id -> the parent row. Immutable once set (trigger below); it is a
--     record of where the row came from and is never used to exclude a row;
--   * role:    stage 'candidate' (159) for an unreviewed retrain.
-- WHY (#2311): the deployer registers an MLflow model version but the numeric version is
-- stored nowhere (the retrain task reports None; job 836578bf created v4). A retry can
-- register a second version for the same run, so (name, run id) does not identify one
-- version. mlflow_model_version records the exact number the deployer registered.
-- ml_retraining_history.deployment_id already exists (017) and is never written; the retrain
-- task now writes it, and this file backfills it for the completed jobs.
--
-- BACKFILL (measured read-only 2026-09-28): ml_retraining_history links a parent
-- (model_id) to the candidate's version label (new_model_version) under the parent's
-- model_name. Three registry rows match: cff4f2b5 / faf4ed1d (completed jobs 836578bf /
-- 36cd579b, stage staging) and c524db0f (FAILED job 073c38eb, stage development), all under
-- parent 4ec55d13 (initiation_kisqali_goldstd_lr_v1 v1.0). Completed -> 'candidate', failed
-- -> 'archived'. The backfill is pinned to those three (row, job, parent) triples AND still
-- requires the history join to hold for each. Only rows at 'staging'/'development' and
-- not champion move; any other row,
-- and any row an operator has already moved, is left alone. A candidate matched by more
-- than one history row is skipped (not determinable). Their 'active' ml_deployments rows
-- without an endpoint become 'registered' (#2308), and each completed job records its one
-- deployment. MLflow versions 3/4/5 are pinned by (row id, mlflow_run_id): the MLflow
-- registry holds exactly one version per run for these three (read-only search 2026-09-28).
--
-- SAFETY: compare-and-set on every UPDATE; a second application matches zero rows.
-- NOTICEs print before/after counts. No MLflow write happens here (MLflow still lists
-- v3/v4/v5 at Staging; that is an operator step, not a database migration).
--
-- NOTE: no transaction control here -- the migration runner wraps each file.

-- ---------------------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------------------

ALTER TABLE ml_model_registry
    ADD COLUMN IF NOT EXISTS retrain_of_id UUID NULL REFERENCES ml_model_registry(id);

ALTER TABLE ml_model_registry
    ADD COLUMN IF NOT EXISTS mlflow_model_version INTEGER NULL;

CREATE INDEX IF NOT EXISTS idx_ml_model_registry_retrain_of
    ON ml_model_registry (retrain_of_id)
    WHERE retrain_of_id IS NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'ml_model_registry_retrain_of_not_self'
                      AND conrelid = 'public.ml_model_registry'::regclass) THEN
        ALTER TABLE ml_model_registry
            ADD CONSTRAINT ml_model_registry_retrain_of_not_self
            CHECK (retrain_of_id IS NULL OR retrain_of_id <> id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'ml_model_registry_mlflow_model_version_positive'
                      AND conrelid = 'public.ml_model_registry'::regclass) THEN
        ALTER TABLE ml_model_registry
            ADD CONSTRAINT ml_model_registry_mlflow_model_version_positive
            CHECK (mlflow_model_version IS NULL OR mlflow_model_version > 0);
    END IF;
END
$$;

COMMENT ON COLUMN ml_model_registry.retrain_of_id IS
    'Parent registry row this row was retrained from (#2310). Immutable once set; lineage only, never a filter.';
COMMENT ON COLUMN ml_model_registry.mlflow_model_version IS
    'Exact MLflow model-registry version number the deployer registered for this row (#2311).';

-- Lineage is immutable once written: NULL -> parent is allowed (the backfill below, and the
-- writer's single insert), any change of a non-NULL value is refused.
CREATE OR REPLACE FUNCTION ml_model_registry_retrain_of_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $fn$
BEGIN
    IF OLD.retrain_of_id IS NOT NULL
       AND NEW.retrain_of_id IS DISTINCT FROM OLD.retrain_of_id THEN
        RAISE EXCEPTION
            'ml_model_registry.retrain_of_id is immutable (row %, % -> %)',
            OLD.id, OLD.retrain_of_id, NEW.retrain_of_id
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$fn$;

DROP TRIGGER IF EXISTS tr_ml_model_registry_retrain_of_immutable ON ml_model_registry;
CREATE TRIGGER tr_ml_model_registry_retrain_of_immutable
    BEFORE UPDATE OF retrain_of_id ON ml_model_registry
    FOR EACH ROW EXECUTE FUNCTION ml_model_registry_retrain_of_immutable();

-- ---------------------------------------------------------------------------------------
-- Backfill
-- ---------------------------------------------------------------------------------------

DO $$
DECLARE
    v_rec RECORD;
    v_n   INTEGER;
BEGIN
    CREATE TEMP TABLE _m160_matched ON COMMIT DROP AS
    SELECT c.id AS candidate_id, p.id AS parent_id, h.id AS history_id, h.status AS job_status
      FROM ml_retraining_history h
      JOIN ml_model_registry p ON p.id = h.model_id
      JOIN ml_model_registry c
        ON c.model_name = p.model_name
       AND c.model_version = h.new_model_version
       AND c.id <> p.id
      -- Pinned to the three audited rows (codex r1): a retrain written between the audit
      -- and this migration's apply is NOT restaged here (the new writer registers its own
      -- candidates; an old-code row is left for review rather than guessed at).
     WHERE (c.id, h.id, p.id) IN (
            ('c524db0f-1df1-4f21-806f-3b857c5245b9'::uuid,
             '073c38eb-586b-4f83-adb3-4a7ec0d1d20b'::uuid,
             '4ec55d13-46c8-4df4-9ec8-7723fad67fb3'::uuid),
            ('cff4f2b5-a87e-4947-aa2a-243a7fb0ee45'::uuid,
             '836578bf-0456-433d-9593-dbd373581579'::uuid,
             '4ec55d13-46c8-4df4-9ec8-7723fad67fb3'::uuid),
            ('faf4ed1d-15cf-4ae2-88c7-e927e64ec53f'::uuid,
             '36cd579b-102f-41ec-9003-9261d315f785'::uuid,
             '4ec55d13-46c8-4df4-9ec8-7723fad67fb3'::uuid));

    -- A candidate matched by more than one history row is not determinable: skip it.
    DELETE FROM _m160_matched m
     WHERE m.candidate_id IN (SELECT candidate_id FROM _m160_matched
                               GROUP BY candidate_id HAVING count(*) > 1);

    FOR v_rec IN
        SELECT m.job_status, r.stage::text AS stage, (r.retrain_of_id IS NOT NULL) AS linked,
               count(*) AS n
          FROM _m160_matched m JOIN ml_model_registry r ON r.id = m.candidate_id
         GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    LOOP
        RAISE NOTICE '160 before: job_status=% stage=% linked=% rows=%',
            v_rec.job_status, v_rec.stage, v_rec.linked, v_rec.n;
    END LOOP;

    UPDATE ml_model_registry r
       SET retrain_of_id = m.parent_id
      FROM _m160_matched m
     WHERE r.id = m.candidate_id
       AND r.retrain_of_id IS NULL;
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE '160 retrain_of_id set: % rows', v_n;

    UPDATE ml_model_registry r
       SET stage = 'candidate'
      FROM _m160_matched m
     WHERE r.id = m.candidate_id
       AND m.job_status = 'completed'
       AND r.retrain_of_id = m.parent_id
       AND r.stage IN ('staging', 'development')
       AND r.is_champion IS NOT TRUE;
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE '160 stage -> candidate: % rows', v_n;

    UPDATE ml_model_registry r
       SET stage = 'archived'
      FROM _m160_matched m
     WHERE r.id = m.candidate_id
       AND m.job_status = 'failed'
       AND r.retrain_of_id = m.parent_id
       AND r.stage IN ('staging', 'development')
       AND r.is_champion IS NOT TRUE;
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE '160 stage -> archived (failed job): % rows', v_n;

    -- #2308: a register-only deployment of a backfilled row is not active.
    UPDATE ml_deployments d
       SET status = 'registered'
      FROM _m160_matched m
     WHERE d.model_registry_id = m.candidate_id
       AND d.status = 'active'
       AND d.endpoint_url IS NULL
       AND d.endpoint_name IS NULL;
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE '160 ml_deployments active -> registered: % rows', v_n;

    -- #2311: each completed job records its one deployment (skipped when not exactly one).
    UPDATE ml_retraining_history h
       SET deployment_id = d.id
      FROM _m160_matched m
      JOIN ml_deployments d ON d.model_registry_id = m.candidate_id
     WHERE h.id = m.history_id
       AND m.job_status = 'completed'
       AND h.deployment_id IS NULL
       AND (SELECT count(*) FROM ml_deployments d2
             WHERE d2.model_registry_id = m.candidate_id) = 1;
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE '160 ml_retraining_history.deployment_id set: % rows', v_n;

    FOR v_rec IN
        SELECT m.job_status, r.stage::text AS stage, (r.retrain_of_id IS NOT NULL) AS linked,
               count(*) AS n
          FROM _m160_matched m JOIN ml_model_registry r ON r.id = m.candidate_id
         GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    LOOP
        RAISE NOTICE '160 after: job_status=% stage=% linked=% rows=%',
            v_rec.job_status, v_rec.stage, v_rec.linked, v_rec.n;
    END LOOP;
END
$$;

-- #2311: the exact MLflow versions of the three backfilled rows (one version per run in the
-- MLflow registry, read-only search 2026-09-28). Pinned to (id, name, mlflow_run_id).
UPDATE ml_model_registry
   SET mlflow_model_version = 3
 WHERE id = 'c524db0f-1df1-4f21-806f-3b857c5245b9'
   AND model_name = 'initiation_kisqali_goldstd_lr_v1'
   AND mlflow_run_id = 'fef4480afb82435d894629decb4d73d5'
   AND mlflow_model_version IS NULL;

UPDATE ml_model_registry
   SET mlflow_model_version = 4
 WHERE id = 'cff4f2b5-a87e-4947-aa2a-243a7fb0ee45'
   AND model_name = 'initiation_kisqali_goldstd_lr_v1'
   AND mlflow_run_id = '830aef651bba4bf5bb7202a96e614e7a'
   AND mlflow_model_version IS NULL;

UPDATE ml_model_registry
   SET mlflow_model_version = 5
 WHERE id = 'faf4ed1d-15cf-4ae2-88c7-e927e64ec53f'
   AND model_name = 'initiation_kisqali_goldstd_lr_v1'
   AND mlflow_run_id = 'df85634d4315423d97bdf115ae02d4af'
   AND mlflow_model_version IS NULL;
