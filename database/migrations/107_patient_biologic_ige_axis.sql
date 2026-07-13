-- ============================================================================
-- Migration 107 (Phase 2, 2026-07-13): anti-IgE clinical axis for CSU/Remibrutinib.
--
-- Adds the biologic-experience + baseline serum-IgE columns the copilot chatbot
-- used to FABRICATE ("biologic-naive vs biologic-experienced CSU patients … IgE").
-- They are now REAL data — but populated ONLY for Remibrutinib (CSU) rows; NULL for
-- the oncology (Kisqali) / PNH (Fabhalta) brands, consistent with the Phase 2
-- brand-gating of every indication-specific eligibility column
-- (src.ml.synthetic.clinical_codes.BRAND_ELIGIBILITY_FIELDS). Correlational Phase 2
-- columns — no differential causal effect is wired (that is Phase 3).
--
-- Additive + idempotent. Mirrors the 087/088 contract: ADD COLUMN IF NOT EXISTS,
-- nullable, NO BEGIN/COMMIT (the migration runner wraps its own txn), NOTIFY pgrst.
-- ----------------------------------------------------------------------------

ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS biologic_experienced SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS ige_level            NUMERIC;

COMMENT ON COLUMN patient_journeys.biologic_experienced IS
    'CSU/Remibrutinib only: prior anti-IgE (e.g. omalizumab) exposure (0/1). NULL for oncology/PNH brands (Phase 2 brand-gating, migration 107). Correlational — differential causal effect is deferred to Phase 3.';
COMMENT ON COLUMN patient_journeys.ige_level IS
    'CSU/Remibrutinib only: baseline total serum IgE (IU/mL, lognormal ~150 median). NULL for oncology/PNH brands (Phase 2 brand-gating, migration 107).';

-- ----------------------------------------------------------------------------
-- Phase 2 brand-gating of the EXISTING rows: NULL every indication-specific
-- eligibility column on rows whose brand does NOT own it, so off-brand clinical
-- attributes are ABSENT rather than fabricated (the DGP previously stamped a CSU
-- UAS7 on Kisqali oncology rows, a renal eGFR on CSU rows, etc.). Mirrors
-- src.ml.synthetic.clinical_codes.BRAND_ELIGIBILITY_FIELDS exactly.
--
-- Deterministic, set-based, IDEMPOTENT (NULLing an already-NULL value is a no-op),
-- and fresh-DB-safe: on a fresh build this migration runs BEFORE the data load, so
-- these UPDATEs touch 0 rows and the generator then loads already-gated rows. On the
-- existing droplet it gates the ~25k live rows in place — NO reseed, so the causal
-- substrate + Phase 1 severity/line counts + the (boosted) persistence labels are
-- untouched. primary_diagnosis_code is the row's own diagnosis and is NOT gated.
--
-- ORDER: the brand-aware causal/segment code must be DEPLOYED before this is applied
-- on a populated DB (an off-brand NULL clinical column would otherwise feed NaN to
-- EconML on the old read path). biologic_experienced / ige_level start NULL and are
-- populated for Remibrutinib by scripts/backfill_biologic_ige.py (needs RNG).
UPDATE patient_journeys
   SET urticaria_severity_uas7 = NULL,
       prior_antihistamine_therapy = NULL
 WHERE brand <> 'Remibrutinib';

UPDATE patient_journeys
   SET hr_status = NULL,
       her2_status = NULL,
       disease_stage = NULL,
       ecog_performance_status = NULL
 WHERE brand <> 'Kisqali';

UPDATE patient_journeys
   SET ldh_ratio = NULL,
       complement_inhibitor_status = NULL,
       proteinuria_g_day = NULL,
       egfr = NULL
 WHERE brand <> 'Fabhalta';

-- PostgREST caches the schema; reload so the new columns are visible to the API.
NOTIFY pgrst, 'reload schema';
