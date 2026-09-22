-- Migration 149: csu_escalation_causal — the remibrutinib-vs-competitor causal
-- cohort table (Lane C pre-wiring, spec 2026-09-22 §3C.2)
--
-- WHY: the program's target causal question is remibrutinib (Rhapsido) vs a
-- competitor CSU biologic (Xolair / Dupixent) -> persistence. Remibrutinib is
-- absent from every real claims drop to date (FDA CSU approval 2025-09-30, the
-- drops' last day), so the registry entry ``csu_escalation_causal``
-- (src/api/routes/causal/datasets.py) is registered NOW against this table and
-- becomes live with NO registry change at the post-launch refresh: the real
-- export (scripts/convert_optum_mart.py select_csu_escalation_contrast) then
-- replaces the rows. Until then the table is backed by the SYNTHETIC CSU cohort
-- (scripts/build_csu_escalation_synthetic_cohort.py), every row
-- is_synthetic = true, so the real-mode provenance filter returns no rows —
-- a synthetic estimate is never served as real.
--
-- Shape: one column per field of the Lane A causal export (migration 148,
-- optum_biologic_persistence_causal), with the treatment column renamed:
-- treatment_remibrutinib (1 = RHAPSIDO, 0 = XOLAIR / DUPIXENT). The 64
-- pre-index baseline features (MART_SAFE_FEATURES) and the four outcomes are
-- identical, so the two datasets share one covariate + outcome contract.
-- tests/unit/test_scripts/test_csu_escalation_cohort_contract.py pins that
-- equality against the synthetic builder's columns.
-- Grain = patient (patient_id PK; a loader upserts on it — idempotent).
-- Additive + idempotent (IF NOT EXISTS).
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

CREATE TABLE IF NOT EXISTS public.csu_escalation_causal (
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
    -- treatment (Lane C): remibrutinib vs the competitor pool
    index_biologic_brand            TEXT NOT NULL,
    treatment_remibrutinib          SMALLINT NOT NULL CHECK (treatment_remibrutinib IN (0, 1)),
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
-- per run; a cert's arm-split probe counts by treatment.
CREATE INDEX IF NOT EXISTS idx_csu_escalation_causal_treatment
    ON public.csu_escalation_causal (treatment_remibrutinib, is_synthetic);
CREATE INDEX IF NOT EXISTS idx_csu_escalation_causal_outcomes
    ON public.csu_escalation_causal (persistent_at_180d_g28, discontinued_180d, biologic_switch_180d_flag);

COMMENT ON TABLE public.csu_escalation_causal IS
    'Lane C (2026-09-22): the remibrutinib (Rhapsido) vs competitor CSU biologic causal '
    'cohort with the 64 pre-index baseline features, the binary treatment and four outcomes. '
    'Backed by the SYNTHETIC CSU cohort (is_synthetic = true, planted truth in its sidecar) '
    'until the post-launch claims refresh replaces the rows with the real export; read by '
    'the causal_impact agent as dataset csu_escalation_causal.';
COMMENT ON COLUMN public.csu_escalation_causal.treatment_remibrutinib IS
    '1 = RHAPSIDO (remibrutinib), 0 = XOLAIR / DUPIXENT (index_biologic_brand, canonical arm label).';
COMMENT ON COLUMN public.csu_escalation_causal.persistent_at_180d_g28 IS
    'PRIMARY outcome: covered through day 152 (28-day grace) AND no internal gap > 60 d '
    '(the brand-invariant definition, persistence definition disproof 2026-09-22).';
COMMENT ON COLUMN public.csu_escalation_causal.is_synthetic IS
    'true on every synthetic-backing row; the real-mode provenance filter excludes them, so '
    'real mode returns no rows until the post-launch load writes is_synthetic = false rows.';
