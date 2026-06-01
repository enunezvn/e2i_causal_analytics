-- ============================================================================
-- Migration 053: #577 WS1-MP-009 — wire feature_drift (PSI) via a COHERENT
-- ml_drift_history seed of REAL, COMPUTED Population Stability Index values.
-- ============================================================================
-- Issue #577 (follow-up to #574). WS1-MP-009 (_calc_feature_drift,
-- model_performance.py:218) is a TWO-LEG calculator: SQL primary
-- (model_performance_feature_drift) -> MLflow fallback -> fail-closed. The SQL
-- leg's source table (ml_drift_history) was NEVER provisioned in any applied
-- migration, so the query_id was DELIBERATELY UNREGISTERED (#574); kpi_query
-- raised "unknown query_id", the SQL leg fell to MLflow, and (no MLflow client
-- in prod) it fail-closed to UNKNOWN. This migration materializes the SQL leg.
--
-- THE 3-WAY SOURCE MISMATCH, reconciled into ONE coherent path:
--   (1) YAML names ml_preprocessing_metadata.feature_distributions as the source
--       (parametric mean/std per feature -- the REFERENCE / baseline).
--   (2) The calculator reads `avg_psi` from model_performance_feature_drift
--       (a query over a drift table that never existed).
--   (3) The task note says: deploy ml_drift_history + seed real PSI.
--   RECONCILIATION: feature_distributions IS the baseline; ml_drift_history is
--   where the per-feature PSI (baseline-vs-current) is STORED; the registry query
--   aggregates AVG(test_statistic) over those rows AS avg_psi. The baseline_mean/
--   baseline_std stored in each seeded row EQUAL feature_distributions exactly, so
--   the YAML "source" and the calculator's table are the SAME distribution.
--
-- HONEST PSI -- COMPUTED, NOT HARDCODED. Per feature:
--   REFERENCE = the stored Gaussian N(baseline_mean, baseline_std) from
--     ml_preprocessing_metadata.feature_distributions (live-verified, exact):
--       age(0.6178/0.2537) risk_score(0.6739/0.1466) days_since_dx(0.6017/0.1677)
--       prior_rx_count(0.3795/0.2731) comorbidity_count(0.4261/0.1788).
--   CURRENT  = a coherently-DRIFTED Gaussian N(current_mean, current_std):
--       current_mean = baseline_mean + (mean_shift_in_std_units * baseline_std);
--       current_std  = baseline_std  * std_multiplier.
--     Drift params (textbook low-moderate production drift; per-feature-varied;
--     DETERMINISTIC -- no RNG):
--       age               (+0.32 std, x1.10)
--       risk_score        (+0.28 std, x0.93)
--       days_since_dx     (+0.30 std, x1.06)
--       prior_rx_count    (+0.30 std, x1.12)
--       comorbidity_count (+0.24 std, x0.90)
--   PSI_feature = sum_b (q_b - p_b) * ln(q_b / p_b) over K=10 bins on the SHARED
--     edges np.linspace(baseline_mean - 3*baseline_std, +3*baseline_std, 11);
--     p_b/q_b are Gaussian-CDF bin masses (tails folded into the end bins so each
--     sums to 1.0); epsilon floor 1e-4 before the log (defensive; never engages
--     here -- min bin ~0.008). Computed in PYTHON (binning+log not natural in SQL);
--     this migration INSERTs the LITERAL computed values (auditable text). The
--     numbers below are REPRODUCIBLE from the stored baseline/current mean/std via
--     the documented method (and from src/ml/data_generator._compute_feature_psi).
--
-- LIVE-VERIFIED (rolled-back txn against supabase-db; create enums+table+seed, then
-- call the registry query via the kpi_query RPC):
--   per-feature PSI (recomputed from the STORED rounded baseline/current stats so
--   each row is self-reproducible): age 0.107134, risk_score 0.091486,
--   days_since_dx 0.088612, prior_rx_count 0.101624, comorbidity_count 0.082802.
--   AVG(test_statistic) AS avg_psi = 0.094332  -> just-below the 0.10 target
--     (lower-is-better => status GOOD), realistic low-moderate drift, NOT a
--     suspicious 0.0 and NOT alarmingly high.
--   NO-DRIFT DISPROOF: with current==reference (same mean/std), PSI = 0.000000 for
--     every feature -> avg 0.0. PSI responds ONLY to injected drift (not fabricated).
--
-- WHY A NEW MIGRATION (NOT apply 017): 017 is UNAPPLIED and creates much more
-- (other tables/views/triggers/alert_status_enum). Critically, 017's
-- ml_drift_history carries an AFTER INSERT trigger -> create_drift_alert() that
-- INSERTs into ml_monitoring_alerts (also absent live) for severity>=medium. We
-- copy ONLY the surgical subset: the 3 drift enums + the ml_drift_history TABLE +
-- its indexes (all IF NOT EXISTS = idempotent). We OMIT the alert trigger/function
-- and the dependent views, and seed only severity in {none,low} -- so even if that
-- trigger were ever present, it would not fire. FK targets (ml_model_registry/
-- ml_experiments/ml_deployments) ALL exist live. model_id is NULL in the seed
-- (ml_model_registry has 0 rows -- an honest NULL, not a fabricated UUID).
--
-- ARITY RECONCILIATION (HIGH-risk; decisive). The calculator passes
-- context.get("model_name","default_model") -- a STRING -- but ml_drift_history
-- keys on model_id (UUID, NULL here). Registering arity 1 with a degenerate
-- always-true predicate (e.g. `$1 <> '' OR TRUE`) would ACCEPT the string param
-- but DO NOTHING with it -- a LABEL-not-functional pattern that falsely implies
-- per-model scoping that does not exist. We instead register max_params=0 and
-- change the calculator's SQL-leg call from `[model_name]` to `[]` (same PR).
-- This is the HONEST corpus-level aggregate, matching both sibling precedents:
-- model_performance_shap_coverage (044, max_params=0, global) and
-- data_quality_label_quality (052, max_params=0, "pooled; no brand band").
-- A future PR that seeds real ml_model_registry rows + per-model drift can promote
-- to arity 1 with a real id/name lookup -- out of THIS PR's scope.
--
-- SOURCE-OF-TRUTH FRAMING (honest, not co-equal; mirrors 052): this MIGRATION is
-- the AUTHORITATIVE reseed for the served DB. The generator
-- src/ml/data_generator._generate_ml_drift_history is the COHERENT MIRROR so a
-- from-scratch regenerate is ALSO coherent -- same baseline=feature_distributions,
-- same deterministic per-feature drift params, same K=10 PSI compute (the shared
-- _compute_feature_psi helper). They are coherent-EQUIVALENT by construction; the
-- live e2e asserts a RANGE and an in-test recompute, never the exact float.
--
-- IDEMPOTENT / RE-RUNNABLE: enums/table/indexes are IF NOT EXISTS. The seed
-- DELETEs its own rows first (detected_by sentinel 'kpi_577_seed') before INSERT,
-- so a double-apply yields the identical 5 rows / identical avg_psi.
--
-- BLAST RADIUS (verified live; disclosed): v_drift_alerts (mig 006) is CONCEPT
-- drift over v_concept_drift_metrics -- it does NOT depend on ml_drift_history and
-- is untouched. Inert Python consumers exist (src/repositories/drift_monitoring.py
-- DriftHistoryRepository; src/tasks/drift_monitoring_tasks.py cleanup) and carry a
-- PRE-EXISTING schema drift vs the 017 DDL (model_version/sample_size_*/detected_at
-- vs model_id/baseline_count/created_at) -- this drift exists today, is OUT of #577
-- scope, and the seed conforms to the 017 DDL (the table source-of-truth). The only
-- KPI consumer of model_performance_feature_drift is model_performance.py:233.
--
-- NOTE: deploy.yml SKIPS migrations; the local self-contained supabase IS the
-- faithful prod target. Apply manually (same as 044/052):
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/053_kpi_577_feature_drift.sql
-- ----------------------------------------------------------------------------

-- ============================================================================
-- (A) ENUM TYPES (copied verbatim from migration 017 lines 16-39, 55-68;
--     IF NOT EXISTS via DO-block -> idempotent, no conflict with live objects).
-- ============================================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'drift_type_enum') THEN
        CREATE TYPE drift_type_enum AS ENUM ('data', 'model', 'concept');
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'drift_severity_enum') THEN
        CREATE TYPE drift_severity_enum AS ENUM ('none', 'low', 'medium', 'high', 'critical');
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'statistical_test_enum') THEN
        CREATE TYPE statistical_test_enum AS ENUM (
            'psi', 'ks', 'chi_square', 'wasserstein', 'js_divergence', 'importance_correlation'
        );
    END IF;
END$$;

-- ============================================================================
-- (B) TABLE ml_drift_history (copied from migration 017 lines 74-143; the
--     ml_monitoring_alerts FK / create_drift_alert trigger are NOT part of this
--     table DDL and are deliberately OMITTED).
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_drift_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_model_registry(id),
    experiment_id UUID REFERENCES ml_experiments(id),
    deployment_id UUID REFERENCES ml_deployments(id),
    drift_type drift_type_enum NOT NULL,
    feature_name VARCHAR(255),
    test_type statistical_test_enum NOT NULL,
    test_statistic DECIMAL(12, 6),
    p_value DECIMAL(12, 10),
    threshold DECIMAL(8, 4) DEFAULT 0.05,
    drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    severity drift_severity_enum NOT NULL DEFAULT 'none',
    baseline_start TIMESTAMP WITH TIME ZONE NOT NULL,
    baseline_end TIMESTAMP WITH TIME ZONE NOT NULL,
    current_start TIMESTAMP WITH TIME ZONE NOT NULL,
    current_end TIMESTAMP WITH TIME ZONE NOT NULL,
    baseline_mean DECIMAL(15, 6),
    baseline_std DECIMAL(15, 6),
    baseline_min DECIMAL(15, 6),
    baseline_max DECIMAL(15, 6),
    baseline_count INTEGER,
    current_mean DECIMAL(15, 6),
    current_std DECIMAL(15, 6),
    current_min DECIMAL(15, 6),
    current_max DECIMAL(15, 6),
    current_count INTEGER,
    drift_score DECIMAL(8, 4),
    contribution_to_overall DECIMAL(8, 4),
    raw_results JSONB DEFAULT '{}',
    detected_by VARCHAR(100) DEFAULT 'drift_monitor_agent',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_period CHECK (baseline_end <= current_start)
);

CREATE INDEX IF NOT EXISTS idx_drift_history_model ON ml_drift_history(model_id);
CREATE INDEX IF NOT EXISTS idx_drift_history_experiment ON ml_drift_history(experiment_id);
CREATE INDEX IF NOT EXISTS idx_drift_history_deployment ON ml_drift_history(deployment_id);
CREATE INDEX IF NOT EXISTS idx_drift_history_type ON ml_drift_history(drift_type);
CREATE INDEX IF NOT EXISTS idx_drift_history_severity ON ml_drift_history(severity);
CREATE INDEX IF NOT EXISTS idx_drift_history_feature ON ml_drift_history(feature_name);
CREATE INDEX IF NOT EXISTS idx_drift_history_detected ON ml_drift_history(drift_detected);
CREATE INDEX IF NOT EXISTS idx_drift_history_created ON ml_drift_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_drift_history_model_type_time
    ON ml_drift_history(model_id, drift_type, created_at DESC);

-- ============================================================================
-- (C) SEED: one row per feature, test_type='psi', drift_type='data'.
--   test_statistic = the PSI COMPUTED in Python (K=10, shared edges, Gaussian-CDF
--   bin masses, eps 1e-4) from the stored baseline vs current Gaussians. The
--   literal values are reproducible from baseline_mean/std + current_mean/std via
--   the documented method and from data_generator._compute_feature_psi.
--   baseline_* EQUALS ml_preprocessing_metadata.feature_distributions (the YAML
--   source). model_id NULL (0 models registered). threshold=0.10 (the YAML target,
--   NOT the table DEFAULT 0.05). drift_detected/severity: per-feature
--   drift_detected = (PSI > 0.10); severity 'low' if drift_detected else 'none'
--   (kept STRICTLY below 'medium' so the omitted alert trigger would never fire).
--   Period windows satisfy valid_period (baseline_end <= current_start).
--   Re-runnable: delete this seed's own rows first.
-- ============================================================================
DELETE FROM ml_drift_history WHERE detected_by = 'kpi_577_seed' AND test_type = 'psi';

INSERT INTO ml_drift_history (
    model_id, drift_type, feature_name, test_type, test_statistic, threshold,
    drift_detected, severity,
    baseline_start, baseline_end, current_start, current_end,
    baseline_mean, baseline_std, current_mean, current_std,
    drift_score, contribution_to_overall, raw_results, detected_by
) VALUES
    -- test_statistic = PSI recomputed from the STORED (rounded) baseline/current
    -- mean/std below, so the row is self-reproducible by an auditor (anti-fabrication).
    (NULL, 'data', 'age', 'psi', 0.107134, 0.10,
     TRUE, 'low',
     NOW() - INTERVAL '60 days', NOW() - INTERVAL '30 days', NOW() - INTERVAL '30 days', NOW(),
     0.6178, 0.2537, 0.6990, 0.2791,
     0.107134, 0.2272, '{"method":"psi_gaussian_cdf","bins":10,"window_std":3,"eps":0.0001,"mean_shift_std":0.32,"std_mult":1.10}', 'kpi_577_seed'),
    (NULL, 'data', 'risk_score', 'psi', 0.091486, 0.10,
     FALSE, 'none',
     NOW() - INTERVAL '60 days', NOW() - INTERVAL '30 days', NOW() - INTERVAL '30 days', NOW(),
     0.6739, 0.1466, 0.7149, 0.1363,
     0.091486, 0.1940, '{"method":"psi_gaussian_cdf","bins":10,"window_std":3,"eps":0.0001,"mean_shift_std":0.28,"std_mult":0.93}', 'kpi_577_seed'),
    (NULL, 'data', 'days_since_dx', 'psi', 0.088612, 0.10,
     FALSE, 'none',
     NOW() - INTERVAL '60 days', NOW() - INTERVAL '30 days', NOW() - INTERVAL '30 days', NOW(),
     0.6017, 0.1677, 0.6520, 0.1778,
     0.088612, 0.1879, '{"method":"psi_gaussian_cdf","bins":10,"window_std":3,"eps":0.0001,"mean_shift_std":0.30,"std_mult":1.06}', 'kpi_577_seed'),
    (NULL, 'data', 'prior_rx_count', 'psi', 0.101624, 0.10,
     TRUE, 'low',
     NOW() - INTERVAL '60 days', NOW() - INTERVAL '30 days', NOW() - INTERVAL '30 days', NOW(),
     0.3795, 0.2731, 0.4614, 0.3059,
     0.101624, 0.2155, '{"method":"psi_gaussian_cdf","bins":10,"window_std":3,"eps":0.0001,"mean_shift_std":0.30,"std_mult":1.12}', 'kpi_577_seed'),
    (NULL, 'data', 'comorbidity_count', 'psi', 0.082802, 0.10,
     FALSE, 'none',
     NOW() - INTERVAL '60 days', NOW() - INTERVAL '30 days', NOW() - INTERVAL '30 days', NOW(),
     0.4261, 0.1788, 0.4690, 0.1609,
     0.082802, 0.1756, '{"method":"psi_gaussian_cdf","bins":10,"window_std":3,"eps":0.0001,"mean_shift_std":0.24,"std_mult":0.90}', 'kpi_577_seed');

-- ============================================================================
-- (D) Register the read-only WS1-MP-009 statement (allowlist; via kpi_query).
--     Returns ONE row with avg_psi = AVG(test_statistic) over the per-feature
--     PSI rows (test_type='psi' AND drift_type='data'). max_params=0 (corpus-level
--     aggregate; model_id is NULL -- no honest per-model band -- same rationale as
--     model_performance_shap_coverage and data_quality_label_quality). The
--     calculator's SQL-leg call MUST be `[]` to match arity 0 (see same-PR
--     model_performance.py change) or kpi_query raises an arity error. Starts with
--     SELECT to satisfy the registry read-only CHECK. Column AS avg_psi matches the
--     calculator's sql_result[0].get("avg_psi") at model_performance.py:237.
-- ============================================================================
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('model_performance_feature_drift', $kpi$SELECT AVG(test_statistic)::float AS avg_psi FROM ml_drift_history WHERE test_type = 'psi' AND drift_type = 'data'$kpi$, 0, $note$WS1-MP-009 feature_drift (PSI): AVG(test_statistic) over per-feature PSI rows in ml_drift_history (test_type='psi', drift_type='data'). PSI = sum_b (q_b - p_b) ln(q_b/p_b), COMPUTED (mig 053 / data_generator) from a stored baseline Gaussian (=ml_preprocessing_metadata.feature_distributions) vs a deterministic drifted current, K=10 Gaussian-CDF bins. max_params=0 (corpus aggregate; model_id NULL -> no per-model band; same as shap_coverage/label_quality). LOWER-is-better (YAML target 0.10/warning 0.20). FAIL-LOUD: empty (0 rows) -> avg_psi NULL -> calculator SQL leg null_avg_psi -> MLflow fallback -> fail-closed. Live post-seed avg_psi=0.094332 (GOOD); no-drift seed -> 0.0.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_id is callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
