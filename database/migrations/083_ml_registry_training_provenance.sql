-- Migration 083: training_provenance on the model catalog (#968)
--
-- The gold-standard cohort models are REAL fitted estimators (is_synthetic stays
-- FALSE so serving/explain/predictions/health keep surfacing them via their
-- unconditional `.eq("is_synthetic", False)` filters), but they are trained ONLY
-- on the synthetic-gold cohort. `is_synthetic` alone therefore cannot tell a
-- real-mode consumer "this model was trained on synthetic data" — it conflates
-- "is the row part of the synthetic substrate" with "what data was it trained on".
--
-- This adds an explicit, self-describing training-data origin and backs the
-- staging->production promotion gate (`transition_stage` refuses a
-- synthetic_gold -> production transition; see src/repositories/ml_experiment.py).
--
-- Idempotent: ADD COLUMN IF NOT EXISTS + guarded constraint/index creation +
-- a NULL-only backfill (safe to re-run; the deploy may apply this more than once).

ALTER TABLE ml_model_registry ADD COLUMN IF NOT EXISTS training_provenance TEXT;
ALTER TABLE ml_experiments    ADD COLUMN IF NOT EXISTS training_provenance TEXT;

-- Allowed values: synthetic_gold | real | mixed | NULL (legacy/unknown).
-- (NULL satisfies the IN-check, so pre-existing rows are not rejected.)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'valid_training_provenance_registry'
    ) THEN
        ALTER TABLE ml_model_registry
            ADD CONSTRAINT valid_training_provenance_registry
            CHECK (training_provenance IN ('synthetic_gold', 'real', 'mixed'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'valid_training_provenance_experiments'
    ) THEN
        ALTER TABLE ml_experiments
            ADD CONSTRAINT valid_training_provenance_experiments
            CHECK (training_provenance IN ('synthetic_gold', 'real', 'mixed'));
    END IF;
END $$;

COMMENT ON COLUMN ml_model_registry.training_provenance IS
    'Training-data origin (#968): synthetic_gold = trained on the Shard-09 '
    'synthetic-gold cohort; real = trained on production data; mixed = both; '
    'NULL = legacy/unknown. DISTINCT from is_synthetic (which stays FALSE for the '
    'gold-standard models so they remain servable/explainable). Gates promotion.';
COMMENT ON COLUMN ml_experiments.training_provenance IS
    'Training-data origin (#968): synthetic_gold | real | mixed | NULL. '
    'See ml_model_registry.training_provenance.';

-- Partial index for the promotion gate / catalog filter (only labelled rows).
CREATE INDEX IF NOT EXISTS idx_ml_model_registry_training_provenance
    ON ml_model_registry (training_provenance)
    WHERE training_provenance IS NOT NULL;

-- Backfill the already-registered gold-standard models + their experiments
-- (the synthetic-gold-trained rows this migration exists to label). NULL-only so
-- re-runs and any future real-data retrains (which set 'real') are not clobbered.
UPDATE ml_model_registry
   SET training_provenance = 'synthetic_gold'
 WHERE training_provenance IS NULL
   AND model_name LIKE '%goldstd%';

UPDATE ml_experiments
   SET training_provenance = 'synthetic_gold'
 WHERE training_provenance IS NULL
   AND created_by = 'gold_standard_eval';
