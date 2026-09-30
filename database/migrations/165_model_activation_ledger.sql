-- Migration 165: model activation ledger + transactional, self-checking activate/rollback RPCs (#2318).
--
-- WHY: an owner-approved activation makes a retrained 'candidate' row the served row for its
-- model_name and archives the row it retrains (the predecessor). The DB side must be ONE
-- transaction (registry roles + the candidate's ml_deployments row + the ledger), and a
-- rollback needs the predecessor's prior stage / champion flag / artifact path and the prior
-- MLflow alias recorded somewhere: the predecessor has no MLflow version and no ml_deployments
-- row (plan F11, F13). scripts/model_activation.py (Lane 6) drives the phases; this file is
-- the durable state machine plus the last line of defence.
--
-- WHAT:
--   * ml_model_activations -- one row per activation: identities, prior state, the gate
--     report, the approval, a phase and write-once progress markers. At most one live row per
--     name; a rolled-back candidate is never activated again (OD-7).
--   * ml_activation_production_allowlist -- EMPTY. Names allowed to activate at 'production'
--     (the hcp_adoption synthetic_gold exemption, OD-6); rows only by an owner-approved migration.
--   * activation_gate_config() / activation_gate_passes() -- the frozen acceptance rule (OD-2).
--     The constants MUST equal src/mlops/activation/holdout_gate.GateConfig (a unit test
--     compares them). The report must be COMPLETE and INTERNALLY CONSISTENT as Lane 5's
--     evaluate_gate writes it: every field it always emits, typed and in range, and every
--     derived field (auc_delta, the DeLong lower bound, the bootstrap counts) re-derived from
--     the numbers it follows from, so a verdict cannot be edited without its evidence. Every
--     field is type-checked as jsonb: a NUMERIC cast of the string "NaN" compares greater than
--     every number in Postgres, so a string never counts as a number.
--   * The ledger INSERT takes the same per-name advisory lock and SHARE ROW EXCLUSIVE registry
--     lock as the RPCs before it counts canonical rows, so a concurrent registry writer is
--     either counted or waits and then meets the role guard (codex r1).
--   * abort_serving_restored_at -- an abort after the candidate bundle went live reaches
--     'aborted' only once the predecessor bundle is live again and verified (codex r1).
--   * activate_model_candidate() / rollback_model_activation() -- SECURITY DEFINER RPCs that
--     re-check the gate, every identity and the single-canonical-row invariant under a lock
--     that blocks every other registry writer for the (milliseconds) switch.
--   * tr_ml_model_registry_activation_role_guard -- the weekly re-registration
--     (register_model_row's upsert) keeps refitting an activation's rows but can no longer
--     change their stage, champion flag or registered_at (OD-5); and no OTHER row of that name
--     may become canonical or champion while the activation is live. It runs in the writer's
--     own transaction, so there is no check-then-write race.
--   * ml_activation_rpc_authority -- how the two RPCs (and only they) get past that guard: each
--     writes its own transaction id there before touching the registry and deletes it before
--     returning. No API role can read or write the table. (A custom GUC flag would not do:
--     any role can set_config() one -- codex r1.)
--
-- TRUST NOTE: service_role can already UPDATE ml_model_registry directly, so none of this
-- defends against a malicious service-key holder: one can still forge a complete, consistent
-- report (the predicate checks completeness and consistency, not authenticity). It makes a
-- CLI bug, a stale or an incomplete/inconsistent report fail closed, and keeps the audit trail
-- append-only.
--
-- ROLLBACK: rollback_165_model_activation_ledger.sql (refuses while an activation is live).
--
-- NOTE: no BEGIN/COMMIT here -- scripts/run_migrations.sh wraps each file.

CREATE TABLE IF NOT EXISTS public.ml_model_activations (
    id                              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name                      varchar(255) NOT NULL,
    candidate_registry_id           uuid NOT NULL REFERENCES public.ml_model_registry(id),
    predecessor_registry_id         uuid NOT NULL REFERENCES public.ml_model_registry(id),
    served_stage                    model_stage_enum NOT NULL,
    predecessor_prior_stage         model_stage_enum NOT NULL,
    predecessor_prior_is_champion   boolean NOT NULL,
    predecessor_prior_artifact_path text,
    candidate_deployment_id         uuid NOT NULL REFERENCES public.ml_deployments(id),
    candidate_mlflow_model_version  integer NOT NULL CHECK (candidate_mlflow_model_version > 0),
    prior_mlflow_served_version     integer,          -- alias 'served' before activation (NULL = none)
    candidate_bundle_sha256         char(64) NOT NULL CHECK (candidate_bundle_sha256 ~ '^[0-9a-f]{64}$'),
    candidate_bundle_path           text NOT NULL,
    predecessor_bundle_sha256       char(64) NOT NULL CHECK (predecessor_bundle_sha256 ~ '^[0-9a-f]{64}$'),
    predecessor_bundle_path         text NOT NULL,
    gate_report                     jsonb NOT NULL,
    phase                           text NOT NULL DEFAULT 'prepared' CHECK (phase IN
        ('prepared', 'serving_switched', 'active', 'aborting', 'aborted', 'rolling_back', 'rolled_back')),
    approved_by                     text NOT NULL CHECK (length(btrim(approved_by)) > 0),
    rolled_back_by                  text,
    rollback_reason                 text,
    created_at                      timestamptz NOT NULL DEFAULT now(),
    serving_switched_at             timestamptz,       -- candidate bundle live, sidecar verified
    activated_at                    timestamptz,       -- activation DB switch committed (RPC only)
    mlflow_synced_at                timestamptz,
    shap_refreshed_at               timestamptz,
    rollback_serving_at             timestamptz,       -- predecessor bundle live again, verified
    rollback_db_at                  timestamptz,       -- rollback DB switch committed (RPC only)
    rollback_mlflow_synced_at       timestamptz,
    rollback_shap_refreshed_at      timestamptz,
    rolled_back_at                  timestamptz,       -- every rollback surface done
    abort_serving_restored_at       timestamptz,       -- after an abort: predecessor bundle live again, verified
    CONSTRAINT ml_model_activations_distinct CHECK (candidate_registry_id <> predecessor_registry_id),
    CONSTRAINT ml_model_activations_served_stage CHECK (served_stage IN ('staging', 'production')),
    -- The candidate takes over the predecessor's served stage: an hcp_adoption name (served at
    -- production, where chat propensity reads it) can never be activated at staging.
    CONSTRAINT ml_model_activations_same_stage CHECK (served_stage = predecessor_prior_stage),
    -- The versioned bundles live OUTSIDE the sidecar's discovery root: two files with one name
    -- under shap_serving/ are served nondeterministically (plan F1).
    CONSTRAINT ml_model_activations_bundle_paths CHECK (
        length(btrim(candidate_bundle_path)) > 0 AND length(btrim(predecessor_bundle_path)) > 0
        AND candidate_bundle_path NOT LIKE '%/shap_serving/%'
        AND predecessor_bundle_path NOT LIKE '%/shap_serving/%'),
    CONSTRAINT ml_model_activations_rollback_audit CHECK (
        phase NOT IN ('rolling_back', 'rolled_back')
        OR (length(btrim(coalesce(rolled_back_by, ''))) > 0 AND length(btrim(coalesce(rollback_reason, ''))) > 0))
);

-- At most one live activation per served name. Replacing an activated model means rolling it
-- back first (no supersession in v1: every live state has exactly one undo path).
CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_model_activations_one_live
    ON public.ml_model_activations (model_name)
    WHERE phase IN ('prepared', 'serving_switched', 'active', 'aborting', 'rolling_back');

-- OD-7: a candidate gets one activation that was not aborted. Once it has been active (and
-- possibly rolled back) it can never be activated again; a new retrain is required. An abort
-- happens before any DB switch, so the same candidate may be retried after one.
CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_model_activations_candidate_once
    ON public.ml_model_activations (candidate_registry_id)
    WHERE phase <> 'aborted';

COMMENT ON TABLE public.ml_model_activations IS
    'Candidate activation ledger (migration 165, #2318): identities, prior state, gate report and '
    'approval of each activation, driven through its phases by scripts/model_activation.py; the '
    'registry switch is activate_model_candidate / rollback_model_activation. service_role only.';

-- OD-6: names allowed to activate at 'production' (synthetic_gold exemption). EMPTY by
-- default; rows are added only by an owner-approved migration citing the ruling.
CREATE TABLE IF NOT EXISTS public.ml_activation_production_allowlist (
    model_name varchar(255) PRIMARY KEY,
    ruling     text NOT NULL CHECK (length(btrim(ruling)) > 0),
    added_at   timestamptz NOT NULL DEFAULT now()
);

-- The RPCs' authority over the registry role guard: the transaction id of an RPC that is
-- switching roles right now. Written and deleted only by activate_model_candidate /
-- rollback_model_activation (SECURITY DEFINER, owner postgres) inside their own transaction;
-- no API role can read or write it, so unlike a GUC it cannot be set by a caller.
CREATE TABLE IF NOT EXISTS public.ml_activation_rpc_authority (
    xact xid8 PRIMARY KEY
);

-- ---------------------------------------------------------------------------------------------
-- The frozen acceptance rule (OD-2). Changing a constant is a reviewed migration AND a
-- GateConfig change; the unit test fails if they differ.
-- ---------------------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.activation_gate_config() RETURNS jsonb
LANGUAGE sql IMMUTABLE AS $$
    SELECT '{"auc_margin": 0.010, "alpha": 0.05, "brier_margin": 0.005, "slope_band": [0.8, 1.25], "bootstrap_b": 2000, "seed": 0, "min_class_n": 100, "min_usable_bootstrap_frac": 0.90}'::jsonb
$$;

-- A report field as a number, or NULL. Only a jsonb number counts: jsonb numbers are never
-- NaN or infinite, whereas the STRING "NaN" casts to a numeric that beats every bound.
CREATE OR REPLACE FUNCTION public._activation_num(r jsonb, VARIADIC path text[]) RETURNS numeric
LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE WHEN jsonb_typeof(r #> path) = 'number' THEN (r #>> path)::numeric END
$$;

-- NULL-safe: any missing, mistyped or out-of-range field makes it FALSE, never NULL. The report
-- is Lane 5's evaluate_gate output; every check is re-derived here rather than trusting
-- "passed", and the report must describe exactly this model, this target stage, these two
-- bundles and one snapshot of the holdout rows. It must also be complete and consistent:
-- every field Lane 5 always emits (the nullable ones present, number or null), AUCs, PR-AUCs
-- and Briers in [0, 1], se_delta >= 0, integral counts, and the derived fields within 1e-9 of
-- what they are computed from -- auc_delta = auc_candidate - auc_served, auc_lower_bound =
-- auc_delta - z * se_delta, bootstrap_usable + bootstrap_skipped_single_class = bootstrap_b.
-- z is Lane 5's norm.ppf(1 - alpha) for alpha = 0.05, hard-coded here so that r->'config'
-- still equals GateConfig exactly (a unit test compares it with scipy). (STABLE, not
-- IMMUTABLE: it reads the allowlist, so it cannot sit in a CHECK constraint; the insert
-- trigger and the activation RPC both call it.)
CREATE OR REPLACE FUNCTION public.activation_gate_passes(
    r jsonb, cand_sha text, served_sha text, p_name text, p_stage text)
RETURNS boolean LANGUAGE sql STABLE SET search_path = pg_catalog, public AS $$
    SELECT coalesce((
        SELECT
            jsonb_typeof(r) = 'object'
            AND r->'kind' = '"deterministic_acceptance_rule"'::jsonb
            AND r->'config' = v.cfg                           -- every constant, no extra key
            AND r->'passed' = 'true'::jsonb
            AND r->'failed_checks' = '[]'::jsonb
            AND r->'model_name' = to_jsonb(p_name)
            AND r->'served_stage' = to_jsonb(p_stage)                  -- the target stage
            AND cand_sha ~ '^[0-9a-f]{64}$' AND served_sha ~ '^[0-9a-f]{64}$'
            AND cand_sha <> served_sha
            AND r->'candidate_bundle_sha256' = to_jsonb(cand_sha)
            AND r->'served_bundle_sha256' = to_jsonb(served_sha)
            -- one snapshot of the OOS rows, and the counts are its counts
            AND r->'snapshot'->'splits' = '["test", "holdout"]'::jsonb
            AND jsonb_typeof(r->'snapshot'->'rows_sha256') = 'string'
            AND r->'snapshot'->>'rows_sha256' ~ '^[0-9a-f]{64}$'
            AND v.n = trunc(v.n) AND v.n_pos = trunc(v.n_pos)
            AND public._activation_num(r, 'snapshot', 'n') = v.n
            AND public._activation_num(r, 'snapshot', 'n_pos') = v.n_pos
            AND v.n_pos >= (v.cfg->>'min_class_n')::numeric
            AND v.n - v.n_pos >= (v.cfg->>'min_class_n')::numeric
            AND abs(v.prevalence - v.n_pos / v.n) <= 1e-9
            -- the rest of Lane 5's schema: typed, in range
            AND v.auc_served BETWEEN 0 AND 1 AND v.auc_candidate BETWEEN 0 AND 1
            AND v.pr_auc_served BETWEEN 0 AND 1 AND v.pr_auc_candidate BETWEEN 0 AND 1
            AND v.brier_served BETWEEN 0 AND 1 AND v.brier_candidate BETWEEN 0 AND 1
            AND v.se_delta >= 0
            AND v.usable = trunc(v.usable) AND v.usable >= 0
            AND v.skipped = trunc(v.skipped) AND v.skipped >= 0
            AND jsonb_typeof(r->'delong_corr') IN ('number', 'null')
            AND jsonb_typeof(r->'calibration_intercept') IN ('number', 'null')
            -- Lane 5 reports the bootstrap AUC p05 exactly when some resample was usable
            AND jsonb_typeof(r->'bootstrap_auc_delta_p05') = CASE WHEN v.usable > 0 THEN 'number' ELSE 'null' END
            -- the derived fields follow from the numbers they are computed from
            AND abs(v.auc_delta - (v.auc_candidate - v.auc_served)) <= 1e-9
            AND abs(v.auc_lower_bound - (v.auc_delta - v.z * v.se_delta)) <= 1e-9
            AND v.usable + v.skipped = (v.cfg->>'bootstrap_b')::numeric
            -- the four checks, re-derived from the numbers
            AND v.auc_lower_bound > -(v.cfg->>'auc_margin')::numeric
            AND v.brier_delta_upper < (v.cfg->>'brier_margin')::numeric
            AND v.slope BETWEEN (v.cfg->'slope_band'->>0)::numeric AND (v.cfg->'slope_band'->>1)::numeric
            AND v.usable >= (v.cfg->>'min_usable_bootstrap_frac')::numeric * (v.cfg->>'bootstrap_b')::numeric
            -- #1354 calibration pathology: required for every hcp_adoption name and for any
            -- production activation, which also needs the owner's allowlist entry (OD-6).
            -- Lane 5 runs it for exactly these (evaluate_gate's served_stage).
            AND (
                (p_name NOT LIKE 'hcp\_adoption\_%' AND p_stage <> 'production')
                OR (
                    r->'hcp_pathology_passed' = 'true'::jsonb
                    AND r->'hcp_pathology_slope_ok' = 'true'::jsonb
                    AND r->'hcp_pathology_brier_ok' = 'true'::jsonb
                    AND r->'hcp_pathology_reasons' = '[]'::jsonb
                    AND v.brier_candidate < v.prevalence * (1 - v.prevalence)
                    AND v.slope BETWEEN 0.5 AND 2.0))
            AND (p_stage <> 'production' OR EXISTS (
                SELECT 1 FROM public.ml_activation_production_allowlist l WHERE l.model_name = p_name))
        FROM (SELECT
                public.activation_gate_config() AS cfg,
                1.6448536269514722::numeric AS z,           -- scipy norm.ppf(1 - 0.05)
                public._activation_num(r, 'n') AS n,
                public._activation_num(r, 'n_pos') AS n_pos,
                public._activation_num(r, 'prevalence') AS prevalence,
                public._activation_num(r, 'auc_served') AS auc_served,
                public._activation_num(r, 'auc_candidate') AS auc_candidate,
                public._activation_num(r, 'auc_delta') AS auc_delta,
                public._activation_num(r, 'se_delta') AS se_delta,
                public._activation_num(r, 'auc_lower_bound') AS auc_lower_bound,
                public._activation_num(r, 'brier_served') AS brier_served,
                public._activation_num(r, 'brier_candidate') AS brier_candidate,
                public._activation_num(r, 'brier_delta_upper') AS brier_delta_upper,
                public._activation_num(r, 'pr_auc_served') AS pr_auc_served,
                public._activation_num(r, 'pr_auc_candidate') AS pr_auc_candidate,
                public._activation_num(r, 'calibration_slope') AS slope,
                public._activation_num(r, 'bootstrap_usable') AS usable,
                public._activation_num(r, 'bootstrap_skipped_single_class') AS skipped) v
    ), false)
$$;

-- ---------------------------------------------------------------------------------------------
-- Ledger guards
-- ---------------------------------------------------------------------------------------------

-- Identity, prior state, evidence and approval never change after insert; phase moves only
-- along the state machine; a progress marker is set once, only in the phase it belongs to,
-- and never cleared or rewritten.
CREATE OR REPLACE FUNCTION public.ml_model_activations_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, public AS $$
BEGIN
    IF (NEW.id, NEW.model_name, NEW.candidate_registry_id, NEW.predecessor_registry_id, NEW.served_stage,
        NEW.predecessor_prior_stage, NEW.predecessor_prior_is_champion, NEW.predecessor_prior_artifact_path,
        NEW.candidate_deployment_id, NEW.candidate_mlflow_model_version, NEW.prior_mlflow_served_version,
        NEW.candidate_bundle_sha256, NEW.candidate_bundle_path, NEW.predecessor_bundle_sha256,
        NEW.predecessor_bundle_path, NEW.gate_report, NEW.approved_by, NEW.created_at)
       IS DISTINCT FROM
       (OLD.id, OLD.model_name, OLD.candidate_registry_id, OLD.predecessor_registry_id, OLD.served_stage,
        OLD.predecessor_prior_stage, OLD.predecessor_prior_is_champion, OLD.predecessor_prior_artifact_path,
        OLD.candidate_deployment_id, OLD.candidate_mlflow_model_version, OLD.prior_mlflow_served_version,
        OLD.candidate_bundle_sha256, OLD.candidate_bundle_path, OLD.predecessor_bundle_sha256,
        OLD.predecessor_bundle_path, OLD.gate_report, OLD.approved_by, OLD.created_at) THEN
        RAISE EXCEPTION 'ml_model_activations %: identity/evidence/approval columns are immutable', OLD.id;
    END IF;
    IF (OLD.rolled_back_by IS NOT NULL AND NEW.rolled_back_by IS DISTINCT FROM OLD.rolled_back_by)
       OR (OLD.rollback_reason IS NOT NULL AND NEW.rollback_reason IS DISTINCT FROM OLD.rollback_reason) THEN
        RAISE EXCEPTION 'ml_model_activations %: rollback audit is immutable once set', OLD.id;
    END IF;
    IF (OLD.serving_switched_at IS NOT NULL AND NEW.serving_switched_at IS DISTINCT FROM OLD.serving_switched_at)
       OR (OLD.activated_at IS NOT NULL AND NEW.activated_at IS DISTINCT FROM OLD.activated_at)
       OR (OLD.mlflow_synced_at IS NOT NULL AND NEW.mlflow_synced_at IS DISTINCT FROM OLD.mlflow_synced_at)
       OR (OLD.shap_refreshed_at IS NOT NULL AND NEW.shap_refreshed_at IS DISTINCT FROM OLD.shap_refreshed_at)
       OR (OLD.rollback_serving_at IS NOT NULL AND NEW.rollback_serving_at IS DISTINCT FROM OLD.rollback_serving_at)
       OR (OLD.rollback_db_at IS NOT NULL AND NEW.rollback_db_at IS DISTINCT FROM OLD.rollback_db_at)
       OR (OLD.rollback_mlflow_synced_at IS NOT NULL AND NEW.rollback_mlflow_synced_at IS DISTINCT FROM OLD.rollback_mlflow_synced_at)
       OR (OLD.rollback_shap_refreshed_at IS NOT NULL AND NEW.rollback_shap_refreshed_at IS DISTINCT FROM OLD.rollback_shap_refreshed_at)
       OR (OLD.rolled_back_at IS NOT NULL AND NEW.rolled_back_at IS DISTINCT FROM OLD.rolled_back_at)
       OR (OLD.abort_serving_restored_at IS NOT NULL
           AND NEW.abort_serving_restored_at IS DISTINCT FROM OLD.abort_serving_restored_at) THEN
        RAISE EXCEPTION 'ml_model_activations %: progress markers are write-once', OLD.id;
    END IF;
    IF NEW.phase IS DISTINCT FROM OLD.phase AND (OLD.phase, NEW.phase) NOT IN (
        ('prepared', 'serving_switched'), ('prepared', 'aborting'),
        ('serving_switched', 'active'), ('serving_switched', 'aborting'),
        ('aborting', 'aborted'),
        ('active', 'rolling_back'), ('rolling_back', 'rolled_back')) THEN
        RAISE EXCEPTION 'ml_model_activations %: illegal phase % -> %', OLD.id, OLD.phase, NEW.phase;
    END IF;
    IF (NEW.phase = 'serving_switched' AND NEW.serving_switched_at IS NULL)
       OR (NEW.phase = 'active' AND NEW.activated_at IS NULL)
       -- an abort after the candidate bundle went live ends only with the predecessor's back
       OR (NEW.phase = 'aborted' AND NEW.serving_switched_at IS NOT NULL
           AND NEW.abort_serving_restored_at IS NULL)
       OR (NEW.phase = 'rolled_back' AND (NEW.rollback_serving_at IS NULL OR NEW.rollback_db_at IS NULL
            OR NEW.rollback_mlflow_synced_at IS NULL OR NEW.rollback_shap_refreshed_at IS NULL
            OR NEW.rolled_back_at IS NULL)) THEN
        RAISE EXCEPTION 'ml_model_activations %: phase % requires its progress markers', OLD.id, NEW.phase;
    END IF;
    -- A marker (or the rollback audit) may only be FIRST set in the phase it belongs to, and
    -- after the step it depends on. So a marker can never be pre-set to skip a surface.
    IF (OLD.serving_switched_at IS NULL AND NEW.serving_switched_at IS NOT NULL AND NEW.phase <> 'serving_switched')
       OR (OLD.activated_at IS NULL AND NEW.activated_at IS NOT NULL AND NEW.phase <> 'active')
       OR (OLD.mlflow_synced_at IS NULL AND NEW.mlflow_synced_at IS NOT NULL
           AND (NEW.phase <> 'active' OR NEW.activated_at IS NULL))
       OR (OLD.shap_refreshed_at IS NULL AND NEW.shap_refreshed_at IS NOT NULL
           AND (NEW.phase <> 'active' OR NEW.activated_at IS NULL))
       OR (OLD.rolled_back_by IS NULL AND NEW.rolled_back_by IS NOT NULL AND NEW.phase <> 'rolling_back')
       OR (OLD.rollback_reason IS NULL AND NEW.rollback_reason IS NOT NULL AND NEW.phase <> 'rolling_back')
       OR (OLD.rollback_serving_at IS NULL AND NEW.rollback_serving_at IS NOT NULL AND NEW.phase <> 'rolling_back')
       OR (OLD.rollback_db_at IS NULL AND NEW.rollback_db_at IS NOT NULL
           AND (NEW.phase <> 'rolling_back' OR NEW.rollback_serving_at IS NULL))
       OR (OLD.rollback_mlflow_synced_at IS NULL AND NEW.rollback_mlflow_synced_at IS NOT NULL
           AND (NEW.phase <> 'rolling_back' OR NEW.rollback_db_at IS NULL))
       OR (OLD.rollback_shap_refreshed_at IS NULL AND NEW.rollback_shap_refreshed_at IS NOT NULL
           AND (NEW.phase <> 'rolling_back' OR NEW.rollback_db_at IS NULL))
       OR (OLD.rolled_back_at IS NULL AND NEW.rolled_back_at IS NOT NULL AND NEW.phase <> 'rolled_back')
       OR (OLD.abort_serving_restored_at IS NULL AND NEW.abort_serving_restored_at IS NOT NULL
           AND (NEW.phase <> 'aborting' OR NEW.serving_switched_at IS NULL)) THEN
        RAISE EXCEPTION 'ml_model_activations %: marker set outside its phase', OLD.id;
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS tr_ml_model_activations_guard ON public.ml_model_activations;
CREATE TRIGGER tr_ml_model_activations_guard BEFORE UPDATE ON public.ml_model_activations
    FOR EACH ROW EXECUTE FUNCTION public.ml_model_activations_guard();

-- A new row starts 'prepared' with no markers, carries a report that satisfies the rule, and
-- names rows that are what it says they are. The RPC re-checks all of it under its lock; this
-- makes a wrong row fail at insert rather than after the bundle swap. It reads the registry
-- under the RPCs' locks (codex r1): without them, a concurrent writer's uncommitted canonical
-- row is invisible here while this uncommitted ledger row is invisible to that writer's role
-- guard, and both commit -- a live activation over two canonical rows. With them, a writer
-- that got in first is counted once it commits; one that comes later waits for this
-- transaction and then meets the role guard. The trigger runs as the inserting role:
-- service_role's UPDATE on ml_model_registry permits SHARE ROW EXCLUSIVE (PG: any lock mode
-- with UPDATE, DELETE or TRUNCATE). lock_timeout stays set for the rest of the transaction.
CREATE OR REPLACE FUNCTION public.ml_model_activations_insert_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, public AS $$
DECLARE k int;
BEGIN
    IF NEW.phase <> 'prepared' OR NEW.serving_switched_at IS NOT NULL OR NEW.activated_at IS NOT NULL
       OR NEW.mlflow_synced_at IS NOT NULL OR NEW.shap_refreshed_at IS NOT NULL
       OR NEW.rollback_serving_at IS NOT NULL OR NEW.rollback_db_at IS NOT NULL
       OR NEW.rollback_mlflow_synced_at IS NOT NULL OR NEW.rollback_shap_refreshed_at IS NOT NULL
       OR NEW.rolled_back_at IS NOT NULL OR NEW.rolled_back_by IS NOT NULL OR NEW.rollback_reason IS NOT NULL
       OR NEW.abort_serving_restored_at IS NOT NULL THEN
        RAISE EXCEPTION 'ml_model_activations: a new row starts in phase prepared with no markers';
    END IF;
    IF public.activation_gate_passes(NEW.gate_report, NEW.candidate_bundle_sha256,
            NEW.predecessor_bundle_sha256, NEW.model_name, NEW.served_stage::text) IS NOT TRUE THEN
        RAISE EXCEPTION 'ml_model_activations: gate report does not satisfy the acceptance rule';
    END IF;
    PERFORM set_config('lock_timeout', '10s', true);
    PERFORM pg_advisory_xact_lock(hashtextextended('ml_model_activation:' || NEW.model_name, 0));
    LOCK TABLE public.ml_model_registry IN SHARE ROW EXCLUSIVE MODE;
    PERFORM 1 FROM public.ml_model_registry
     WHERE id = NEW.candidate_registry_id AND stage = 'candidate' AND model_name = NEW.model_name
       AND retrain_of_id = NEW.predecessor_registry_id
       AND mlflow_model_version = NEW.candidate_mlflow_model_version;
    IF NOT FOUND THEN RAISE EXCEPTION 'row % is not a candidate retrain of % for %',
        NEW.candidate_registry_id, NEW.predecessor_registry_id, NEW.model_name; END IF;
    PERFORM 1 FROM public.ml_model_registry
     WHERE id = NEW.predecessor_registry_id AND model_name = NEW.model_name
       AND stage = NEW.predecessor_prior_stage
       AND coalesce(is_champion, false) = NEW.predecessor_prior_is_champion
       AND artifact_path IS NOT DISTINCT FROM NEW.predecessor_prior_artifact_path;
    IF NOT FOUND THEN RAISE EXCEPTION 'predecessor % changed since the gate ran', NEW.predecessor_registry_id; END IF;
    SELECT count(*) INTO k FROM public.ml_model_registry
     WHERE model_name = NEW.model_name AND (stage IS NULL OR stage NOT IN ('candidate', 'archived', 'deprecated'));
    IF k <> 1 THEN
        RAISE EXCEPTION 'model % must have exactly one canonical row (%), found %', NEW.model_name,
            NEW.predecessor_registry_id, k;
    END IF;
    PERFORM 1 FROM public.ml_deployments
     WHERE id = NEW.candidate_deployment_id AND model_registry_id = NEW.candidate_registry_id;
    IF NOT FOUND THEN RAISE EXCEPTION 'deployment % does not belong to the candidate', NEW.candidate_deployment_id; END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS tr_ml_model_activations_insert_guard ON public.ml_model_activations;
CREATE TRIGGER tr_ml_model_activations_insert_guard BEFORE INSERT ON public.ml_model_activations
    FOR EACH ROW EXECUTE FUNCTION public.ml_model_activations_insert_guard();

-- ---------------------------------------------------------------------------------------------
-- The single-canonical-row invariant, under a lock. The predicate is exactly the readers'
-- (model_registry_roles.CANONICAL_STAGE_FILTER: stage NULL or not candidate/archived/deprecated,
-- whatever is_synthetic says -- resolve_canonical_model_id does not filter it either).
-- SHARE ROW EXCLUSIVE blocks every other writer of ml_model_registry until the RPC commits, so
-- no concurrent INSERT of a new canonical row can slip between the count and the commit; the
-- advisory lock serialises the RPCs per name. lock_timeout makes a stuck writer an error, not
-- a hang.
-- ---------------------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public._activation_lock_and_count_served(p_name text, p_expect uuid)
RETURNS void LANGUAGE plpgsql SET search_path = pg_catalog, public AS $$
DECLARE k int; ids text;
BEGIN
    PERFORM set_config('lock_timeout', '10s', true);
    PERFORM pg_advisory_xact_lock(hashtextextended('ml_model_activation:' || p_name, 0));
    LOCK TABLE public.ml_model_registry IN SHARE ROW EXCLUSIVE MODE;
    SELECT count(*), string_agg(id::text, ',' ORDER BY id) INTO k, ids FROM public.ml_model_registry
     WHERE model_name = p_name AND (stage IS NULL OR stage NOT IN ('candidate', 'archived', 'deprecated'));
    IF k <> 1 OR ids <> p_expect::text THEN
        RAISE EXCEPTION 'model % must have exactly one canonical row (%), found % (%)', p_name, p_expect, k, ids;
    END IF;
END $$;

-- ---------------------------------------------------------------------------------------------
-- The two RPCs. SECURITY DEFINER (owner postgres, search_path pinned): the only writers of
-- activated_at / rollback_db_at, and the only callers allowed past the registry role guard:
-- each writes its transaction id into ml_activation_rpc_authority before touching the
-- registry and deletes it before returning (an error rolls the row back with everything else).
-- ---------------------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.activate_model_candidate(p_activation_id uuid)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE a public.ml_model_activations%ROWTYPE; n int;
BEGIN
    SELECT * INTO a FROM public.ml_model_activations WHERE id = p_activation_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'activation % not found', p_activation_id; END IF;
    IF a.phase = 'active' THEN   -- idempotent re-run (a lost response): verify, never trust the phase
        PERFORM public._activation_lock_and_count_served(a.model_name, a.candidate_registry_id);
        PERFORM 1 FROM public.ml_model_registry c, public.ml_model_registry p, public.ml_deployments d
         WHERE c.id = a.candidate_registry_id AND c.stage = a.served_stage
           AND coalesce(c.is_champion, false) = a.predecessor_prior_is_champion
           AND c.artifact_path = a.candidate_bundle_path
           AND p.id = a.predecessor_registry_id AND p.stage = 'archived' AND NOT coalesce(p.is_champion, false)
           AND d.id = a.candidate_deployment_id AND d.status = 'active';
        IF NOT FOUND THEN RAISE EXCEPTION 'activation % is active but the registry has drifted', a.id; END IF;
        RETURN;
    END IF;
    IF a.phase <> 'serving_switched' THEN
        RAISE EXCEPTION 'activation % is in phase %, expected serving_switched', a.id, a.phase;
    END IF;
    IF public.activation_gate_passes(a.gate_report, a.candidate_bundle_sha256, a.predecessor_bundle_sha256,
                                     a.model_name, a.served_stage::text) IS NOT TRUE THEN
        RAISE EXCEPTION 'activation %: gate report does not satisfy the acceptance rule now', a.id;
    END IF;
    INSERT INTO public.ml_activation_rpc_authority (xact) VALUES (pg_current_xact_id());  -- role guard: let through
    PERFORM public._activation_lock_and_count_served(a.model_name, a.predecessor_registry_id);
    PERFORM 1 FROM public.ml_model_registry
     WHERE id = a.candidate_registry_id AND stage = 'candidate' AND model_name = a.model_name
       AND retrain_of_id = a.predecessor_registry_id
       AND mlflow_model_version = a.candidate_mlflow_model_version;
    IF NOT FOUND THEN RAISE EXCEPTION 'row % is not a candidate retrain of % for %',
        a.candidate_registry_id, a.predecessor_registry_id, a.model_name; END IF;
    PERFORM 1 FROM public.ml_model_registry
     WHERE id = a.predecessor_registry_id AND model_name = a.model_name
       AND stage = a.predecessor_prior_stage
       AND coalesce(is_champion, false) = a.predecessor_prior_is_champion
       AND artifact_path IS NOT DISTINCT FROM a.predecessor_prior_artifact_path;
    IF NOT FOUND THEN RAISE EXCEPTION 'predecessor % changed since the gate ran', a.predecessor_registry_id; END IF;
    PERFORM 1 FROM public.ml_deployments
     WHERE id = a.candidate_deployment_id AND model_registry_id = a.candidate_registry_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'deployment % does not belong to the candidate', a.candidate_deployment_id; END IF;

    -- Predecessor first: tr_single_champion (same experiment_id) then finds nothing to demote.
    UPDATE public.ml_model_registry SET stage = 'archived', is_champion = false
     WHERE id = a.predecessor_registry_id;
    GET DIAGNOSTICS n = ROW_COUNT; IF n <> 1 THEN RAISE EXCEPTION 'predecessor update touched % rows', n; END IF;
    UPDATE public.ml_model_registry
       SET stage = a.served_stage, is_champion = a.predecessor_prior_is_champion,
           artifact_path = a.candidate_bundle_path, preprocessing_pipeline_path = a.candidate_bundle_path,
           promoted_at = now()
     WHERE id = a.candidate_registry_id;
    GET DIAGNOSTICS n = ROW_COUNT; IF n <> 1 THEN RAISE EXCEPTION 'candidate update touched % rows', n; END IF;
    UPDATE public.ml_deployments
       SET status = 'active', environment = a.served_stage::text, deployed_at = now(),
           endpoint_name = a.model_name, endpoint_url = 'bentoml://e2i_bentoml/' || a.model_name
     WHERE id = a.candidate_deployment_id;
    GET DIAGNOSTICS n = ROW_COUNT; IF n <> 1 THEN RAISE EXCEPTION 'deployment update touched % rows', n; END IF;
    PERFORM public._activation_lock_and_count_served(a.model_name, a.candidate_registry_id);  -- postcondition
    UPDATE public.ml_model_activations SET phase = 'active', activated_at = now() WHERE id = a.id;
    DELETE FROM public.ml_activation_rpc_authority WHERE xact = pg_current_xact_id();
END $$;

-- Called in phase 'rolling_back' after the CLI restored and verified the predecessor bundle
-- (rollback_serving_at set). Sets rollback_db_at; the phase stays 'rolling_back' until the CLI
-- has also restored MLflow and the SHAP cache, then the CLI moves it to 'rolled_back'.
CREATE OR REPLACE FUNCTION public.rollback_model_activation(p_activation_id uuid)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE a public.ml_model_activations%ROWTYPE; n int;
BEGIN
    SELECT * INTO a FROM public.ml_model_activations WHERE id = p_activation_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'activation % not found', p_activation_id; END IF;
    IF a.phase NOT IN ('rolling_back', 'rolled_back') THEN
        RAISE EXCEPTION 'activation % is in phase %, expected rolling_back', a.id, a.phase;
    END IF;
    IF a.rollback_db_at IS NOT NULL THEN   -- idempotent re-run: verify the postcondition
        PERFORM public._activation_lock_and_count_served(a.model_name, a.predecessor_registry_id);
        PERFORM 1 FROM public.ml_model_registry c, public.ml_model_registry p, public.ml_deployments d
         WHERE c.id = a.candidate_registry_id AND c.stage = 'archived' AND NOT coalesce(c.is_champion, false)
           AND p.id = a.predecessor_registry_id AND p.stage = a.predecessor_prior_stage
           AND coalesce(p.is_champion, false) = a.predecessor_prior_is_champion
           AND p.artifact_path IS NOT DISTINCT FROM a.predecessor_prior_artifact_path
           AND d.id = a.candidate_deployment_id AND d.status = 'rolled_back';
        IF NOT FOUND THEN RAISE EXCEPTION 'activation % rolled back but the registry has drifted', a.id; END IF;
        RETURN;
    END IF;
    IF a.rollback_serving_at IS NULL THEN
        RAISE EXCEPTION 'activation %: restore and verify the predecessor bundle before the DB rollback', a.id;
    END IF;
    INSERT INTO public.ml_activation_rpc_authority (xact) VALUES (pg_current_xact_id());  -- role guard: let through
    PERFORM public._activation_lock_and_count_served(a.model_name, a.candidate_registry_id);

    UPDATE public.ml_model_registry SET stage = 'archived', is_champion = false
     WHERE id = a.candidate_registry_id;                                  -- OD-7
    GET DIAGNOSTICS n = ROW_COUNT; IF n <> 1 THEN RAISE EXCEPTION 'candidate update touched % rows', n; END IF;
    UPDATE public.ml_model_registry
       SET stage = a.predecessor_prior_stage, is_champion = a.predecessor_prior_is_champion,
           artifact_path = a.predecessor_prior_artifact_path
     WHERE id = a.predecessor_registry_id;
    GET DIAGNOSTICS n = ROW_COUNT; IF n <> 1 THEN RAISE EXCEPTION 'predecessor update touched % rows', n; END IF;
    UPDATE public.ml_deployments
       SET status = 'rolled_back', rolled_back_at = now(), rollback_reason = a.rollback_reason,
           deactivated_at = now()
     WHERE id = a.candidate_deployment_id;
    GET DIAGNOSTICS n = ROW_COUNT; IF n <> 1 THEN RAISE EXCEPTION 'deployment update touched % rows', n; END IF;
    PERFORM public._activation_lock_and_count_served(a.model_name, a.predecessor_registry_id);  -- postcondition
    UPDATE public.ml_model_activations SET rollback_db_at = now() WHERE id = a.id;
    DELETE FROM public.ml_activation_rpc_authority WHERE xact = pg_current_xact_id();
END $$;

-- ---------------------------------------------------------------------------------------------
-- OD-5: the weekly re-registration must never change the ROLE of a live activation's rows.
--   * Predecessor / candidate of a live activation: stage, is_champion and registered_at are
--     kept (the candidate also keeps its bundle paths); everything else -- the refit's auc,
--     trained_at, artifact -- still lands, so the weekly refit keeps producing a standing
--     challenger (commit 717dd7158's intent). A NOTICE says so; register_model_row's read-back
--     learns to accept it in Lane 7.
--   * Any OTHER row of that name: refused if the write would make it canonical or champion
--     (it would split the served name in two).
-- Enforced inside the writer's own transaction; the RPCs' table lock orders it after a
-- concurrent switch. Named to sort BEFORE tr_single_champion (BEFORE row triggers fire in name
-- order), so a restored champion flag is what that trigger sees.
-- SECURITY DEFINER so that any registry writer can read the ledger and the RPC authority it
-- checks; the only bypass is a row for the writer's own transaction in
-- ml_activation_rpc_authority, which only the two RPCs (and the owner) can write.
-- ---------------------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.ml_model_registry_activation_role_guard() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE
    live_candidate uuid;
    becomes boolean;
BEGIN
    IF EXISTS (SELECT 1 FROM public.ml_activation_rpc_authority WHERE xact = pg_current_xact_id()) THEN
        RETURN NEW;   -- one of the two RPCs, inside its own transaction
    END IF;
    IF TG_OP = 'UPDATE' THEN
        SELECT a.candidate_registry_id INTO live_candidate FROM public.ml_model_activations a
         WHERE a.phase IN ('prepared', 'serving_switched', 'active', 'aborting', 'rolling_back')
           AND OLD.id IN (a.predecessor_registry_id, a.candidate_registry_id);
        IF FOUND THEN
            IF NEW.model_name IS DISTINCT FROM OLD.model_name THEN
                RAISE EXCEPTION 'ml_model_registry %: cannot rename a row of a live activation (#2318)', OLD.id;
            END IF;
            IF (NEW.stage, NEW.is_champion, NEW.registered_at) IS DISTINCT FROM (OLD.stage, OLD.is_champion, OLD.registered_at) THEN
                RAISE NOTICE 'ml_model_registry %: role preserved, it belongs to a live activation (#2318)', OLD.id;
            END IF;
            NEW.stage := OLD.stage;
            NEW.is_champion := OLD.is_champion;
            NEW.registered_at := OLD.registered_at;
            IF OLD.id = live_candidate THEN
                NEW.artifact_path := OLD.artifact_path;
                NEW.preprocessing_pipeline_path := OLD.preprocessing_pipeline_path;
            END IF;
            RETURN NEW;
        END IF;
    END IF;
    becomes := (NEW.stage IS NULL OR NEW.stage NOT IN ('candidate', 'archived', 'deprecated'))
               OR coalesce(NEW.is_champion, false);
    IF NOT becomes THEN
        RETURN NEW;
    END IF;
    IF TG_OP = 'INSERT' THEN
        -- An upsert of an existing (model_name, model_version) fires BEFORE INSERT first and
        -- then, on the conflict, BEFORE UPDATE: that path is decided above.
        PERFORM 1 FROM public.ml_model_registry r
         WHERE r.model_name = NEW.model_name AND r.model_version = NEW.model_version;
        IF FOUND THEN
            RETURN NEW;
        END IF;
    ELSE
        becomes := NEW.model_name IS DISTINCT FROM OLD.model_name
            OR ((NEW.stage IS NULL OR NEW.stage NOT IN ('candidate', 'archived', 'deprecated'))
                AND NOT (OLD.stage IS NULL OR OLD.stage NOT IN ('candidate', 'archived', 'deprecated')))
            OR (coalesce(NEW.is_champion, false) AND NOT coalesce(OLD.is_champion, false));
    END IF;
    IF becomes AND EXISTS (
        SELECT 1 FROM public.ml_model_activations a
         WHERE a.model_name = NEW.model_name
           AND a.phase IN ('prepared', 'serving_switched', 'active', 'aborting', 'rolling_back')) THEN
        RAISE EXCEPTION 'ml_model_registry: % v% would become canonical or champion while a live activation holds the name (#2318)',
            NEW.model_name, NEW.model_version;
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS tr_ml_model_registry_activation_role_guard ON public.ml_model_registry;
CREATE TRIGGER tr_ml_model_registry_activation_role_guard BEFORE INSERT OR UPDATE ON public.ml_model_registry
    FOR EACH ROW EXECUTE FUNCTION public.ml_model_registry_activation_role_guard();

-- ---------------------------------------------------------------------------------------------
-- Grants. The default ACL on public (pg_default_acl, measured read-only on prod 2026-09-30)
-- gives service_role ALL on every new table and anon/authenticated/service_role EXECUTE on
-- every new function, so everything is revoked from service_role too and granted back exactly.
-- A table-level UPDATE would make the column list below moot.
-- ---------------------------------------------------------------------------------------------
REVOKE ALL ON TABLE public.ml_model_activations FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT, INSERT ON TABLE public.ml_model_activations TO service_role;
GRANT UPDATE (phase, rolled_back_by, rollback_reason, serving_switched_at, mlflow_synced_at,
              shap_refreshed_at, rollback_serving_at, rollback_mlflow_synced_at,
              rollback_shap_refreshed_at, rolled_back_at, abort_serving_restored_at)
    ON public.ml_model_activations TO service_role;
ALTER TABLE public.ml_model_activations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS ml_model_activations_service ON public.ml_model_activations;
CREATE POLICY ml_model_activations_service ON public.ml_model_activations
    FOR ALL TO service_role USING (true) WITH CHECK (true);

REVOKE ALL ON TABLE public.ml_activation_production_allowlist FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE public.ml_activation_production_allowlist TO service_role;

-- Nobody but the owner (and so the SECURITY DEFINER RPCs and role guard) touches the authority.
REVOKE ALL ON TABLE public.ml_activation_rpc_authority FROM PUBLIC, anon, authenticated, service_role;
ALTER TABLE public.ml_activation_rpc_authority ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON FUNCTION public.activation_gate_config() FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public._activation_num(jsonb, text[]) FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.activation_gate_passes(jsonb, text, text, text, text) FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.ml_model_activations_guard() FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.ml_model_activations_insert_guard() FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public._activation_lock_and_count_served(text, uuid) FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.activate_model_candidate(uuid) FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.rollback_model_activation(uuid) FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.ml_model_registry_activation_role_guard() FROM PUBLIC, anon, authenticated, service_role;
-- The insert trigger runs as the inserting role and calls the predicate.
GRANT EXECUTE ON FUNCTION public.activation_gate_config() TO service_role;
GRANT EXECUTE ON FUNCTION public._activation_num(jsonb, text[]) TO service_role;
GRANT EXECUTE ON FUNCTION public.activation_gate_passes(jsonb, text, text, text, text) TO service_role;
GRANT EXECUTE ON FUNCTION public.activate_model_candidate(uuid) TO service_role;
GRANT EXECUTE ON FUNCTION public.rollback_model_activation(uuid) TO service_role;

NOTIFY pgrst, 'reload schema';
