-- Migration 148: optum_biologic_persistence_causal — the REAL-claims causal cohort
--
-- WHY: no causal DAG was ever built and no causal effect ever estimated on real
-- claims data (docs/demos/results/2026-09-22_discovery_real_claims_disproof/).
-- The Optum mart converter drops index_biologic_brand on purpose — the manifest
-- declares it a post-index mart_treatment column that would leak a PREDICTION
-- target. For causal estimation that column IS the treatment. This table holds
-- the separate causal export (scripts/convert_optum_mart.py --cohort
-- persistence_causal): Dupixent vs Xolair initiators (the only contrast the drop
-- observes), the 64 pre-index baseline features (MART_SAFE_FEATURES, measured
-- at the diagnosis index which precedes treatment start), the binary treatment
-- and four outcomes. The prediction tables are untouched.
--
-- Outcome choice (owner decision, spec 2026-09-22 §7): persistent_at_180d_g28 is
-- PRIMARY — covered through day 152 (a 28-day grace) AND no internal gap > 60 d.
-- The shipped persistent_at_180d is days-supply sensitive per brand (14-day
-- Dupixent fills vs 28-45-day Xolair fills; the -17.6 pp raw gap collapses with a
-- 14-day grace and inverts from 28 d) and is stored only so it can be reported
-- ALONGSIDE that sweep, never as an effect.
--
-- One column per exported field; tests/unit/test_scripts/
-- test_optum_causal_cohort_contract.py pins that equality against the export.
-- Grain = patient (patient_id PK; the loader upserts on it — idempotent).
-- Additive + idempotent (IF NOT EXISTS). is_synthetic defaults false: every
-- row here is real claims data and the real-mode provenance filter keeps it.
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

CREATE TABLE IF NOT EXISTS public.optum_biologic_persistence_causal (
    -- identity + journey metadata (the converter's record shape)
    patient_id                      VARCHAR(40) PRIMARY KEY,
    patient_journey_id              VARCHAR(40) NOT NULL,
    patient_hash                    VARCHAR(20) NOT NULL,
    index_date                      DATE NOT NULL,
    journey_start_date              DATE NOT NULL,
    journey_status                  TEXT NOT NULL DEFAULT 'active',
    discontinuation_flag            SMALLINT NOT NULL DEFAULT 0,
    data_quality_score              NUMERIC,
    data_split                      TEXT,
    -- treatment (Lane A): the contrast the drop observes
    index_biologic_brand            TEXT NOT NULL,
    treatment_dupixent              SMALLINT NOT NULL CHECK (treatment_dupixent IN (0, 1)),
    treatment_start_date            DATE NOT NULL,
    -- outcomes
    persistent_at_180d_g28          SMALLINT NOT NULL CHECK (persistent_at_180d_g28 IN (0, 1)),
    discontinued_180d               SMALLINT NOT NULL CHECK (discontinued_180d IN (0, 1)),
    biologic_switch_180d_flag       SMALLINT NOT NULL CHECK (biologic_switch_180d_flag IN (0, 1)),
    persistent_at_180d              SMALLINT NOT NULL CHECK (persistent_at_180d IN (0, 1)),
    -- 64 pre-index baseline features (MART_SAFE_FEATURES): demographics + payer
    age_at_index                    NUMERIC,
    gdr_cd                          TEXT,
    payer_category                  TEXT,
    payer_product                   TEXT,
    payer_bus                       TEXT,
    health_exchange_flag            INTEGER,
    lis_dual_flag                   INTEGER,
    enrollment_duration_days        INTEGER,
    geographic_region               TEXT,
    -- comorbidity summaries
    charlson_score                  INTEGER,
    charlson_risk_band              TEXT,
    elixhauser_van_walraven_score   INTEGER,
    elixhauser_risk_band            TEXT,
    comorbidity_diag_distinct_count INTEGER,
    comorbidity_diag_claim_count    INTEGER,
    high_comorbidity_burden_flag    INTEGER,
    -- Charlson components
    cci_mi                          INTEGER,
    cci_chf                         INTEGER,
    cci_pvd                         INTEGER,
    cci_cerebrovascular             INTEGER,
    cci_dementia                    INTEGER,
    cci_chronic_pulmonary           INTEGER,
    cci_rheumatic                   INTEGER,
    cci_peptic_ulcer                INTEGER,
    cci_mild_liver                  INTEGER,
    cci_diabetes_no_complication    INTEGER,
    cci_diabetes_complication       INTEGER,
    cci_paraplegia                  INTEGER,
    cci_renal                       INTEGER,
    cci_malignancy                  INTEGER,
    cci_severe_liver                INTEGER,
    cci_metastatic_cancer           INTEGER,
    cci_hiv                         INTEGER,
    -- Elixhauser components
    elx_chf                         INTEGER,
    elx_cardiac_arrhythmia          INTEGER,
    elx_valvular_disease            INTEGER,
    elx_pulmonary_circulation       INTEGER,
    elx_pvd                         INTEGER,
    elx_hypertension_uncomplicated  INTEGER,
    elx_hypertension_complicated    INTEGER,
    elx_paralysis                   INTEGER,
    elx_other_neurological          INTEGER,
    elx_chronic_pulmonary           INTEGER,
    elx_diabetes_uncomplicated      INTEGER,
    elx_diabetes_complicated        INTEGER,
    elx_hypothyroidism              INTEGER,
    elx_renal_failure               INTEGER,
    elx_liver_disease               INTEGER,
    elx_peptic_ulcer                INTEGER,
    elx_aids_hiv                    INTEGER,
    elx_lymphoma                    INTEGER,
    elx_metastatic_cancer           INTEGER,
    elx_solid_tumor_no_metastasis   INTEGER,
    elx_rheumatoid_collagen         INTEGER,
    elx_coagulopathy                INTEGER,
    elx_obesity                     INTEGER,
    elx_weight_loss                 INTEGER,
    elx_fluid_electrolyte           INTEGER,
    elx_blood_loss_anemia           INTEGER,
    elx_deficiency_anemia           INTEGER,
    elx_alcohol_abuse               INTEGER,
    elx_drug_abuse                  INTEGER,
    elx_psychoses                   INTEGER,
    elx_depression                  INTEGER,
    -- provenance + bookkeeping
    is_synthetic                    BOOLEAN NOT NULL DEFAULT false,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The causal loader filters is_synthetic and reads the treatment + one outcome
-- per run; the cert's arm-split probe counts by treatment.
CREATE INDEX IF NOT EXISTS idx_optum_biologic_persistence_causal_treatment
    ON public.optum_biologic_persistence_causal (treatment_dupixent, is_synthetic);
CREATE INDEX IF NOT EXISTS idx_optum_biologic_persistence_causal_outcomes
    ON public.optum_biologic_persistence_causal (persistent_at_180d_g28, discontinued_180d, biologic_switch_180d_flag);

COMMENT ON TABLE public.optum_biologic_persistence_causal IS
    'Lane A (2026-09-22): REAL Optum claims causal cohort — Dupixent vs Xolair CSU '
    'escalation-therapy initiators with the 64 pre-index baseline features, the binary '
    'treatment and four outcomes. Written by scripts/load_optum_causal_cohort.py from '
    'the persistence_causal export; read by the causal_impact agent as dataset '
    'optum_biologic_persistence. is_synthetic is false on every row.';
COMMENT ON COLUMN public.optum_biologic_persistence_causal.treatment_dupixent IS
    '1 = DUPIXENT, 0 = XOLAIR (index_biologic_brand). The only contrast the drop observes.';
COMMENT ON COLUMN public.optum_biologic_persistence_causal.persistent_at_180d_g28 IS
    'PRIMARY outcome: covered through day 152 (28-day grace) AND no internal gap > 60 d. '
    'Brand-invariant across a 28-60 d grace (persistence definition disproof 2026-09-22).';
COMMENT ON COLUMN public.optum_biologic_persistence_causal.persistent_at_180d IS
    'The shipped prediction definition (covered through day 180 AND gap <= 60). Days-supply '
    'sensitive per brand (14-d Dupixent vs 28-45-d Xolair fills): report ONLY alongside the '
    'grace sweep, never as an effect.';
COMMENT ON COLUMN public.optum_biologic_persistence_causal.discontinued_180d IS
    'Secondary outcome, as shipped: not covered through day 180 AND a >= 90 d internal or '
    'terminal gap. Brand-robust across the gap sweep.';
