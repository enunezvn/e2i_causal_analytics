-- ============================================================================
-- Migration 052: #577 WS1-DQ-008 — wire label_quality (IAA) via a COHERENT
-- LATENT-TRUTH annotation rework + generalized Fleiss kappa.
-- ============================================================================
-- Issue #577 (follow-up to #574). WS1-DQ-008 (_calc_label_quality, data_quality.py
-- :221) was fail-loud (RuntimeError "no iaa_score source column available (#574)")
-- because ml_annotations carried NO usable agreement signal:
--   (1) annotation_value->>'label' was INDEPENDENT NOISE — random.choice over
--       {positive,negative,uncertain} drawn per-annotation (data_generator.py:1209).
--       Live mean pairwise agreement = 0.3467 ~ 1/3 chance; generalized Fleiss
--       kappa over the real layout = 0.0174 ~ 0 (a generation artifact, not a
--       label-quality signal — the same CM-004/005 / action_taken incoherence).
--   (2) STRUCTURAL DEFECT (verified live): the generator drew annotation_type PER
--       ANNOTATOR, so within a group the 2-4 raters labeled DIFFERENT subjects
--       (48 of 50 groups span >1 annotation_type). IAA is UNDEFINED unless all
--       raters in a group CO-RATE THE SAME subject.
--
-- THE HONEST FIX (user-approved coherent rework; mirrors CM-004/005 + 051): make
-- the synthetic data internally coherent, THEN compute. Two surgical in-place
-- relabels (no row insert/delete — preserves all 153 annotation_ids / 50 groups):
--   (A) GROUP-LEVEL annotation_type: assign ONE type per iaa_group so all raters
--       co-rate the same subject (the IAA precondition the original hypothesis missed).
--   (B) LATENT-TRUTH label: each group gets a deterministic true label; each
--       annotator agrees with prob 92, else emits one of the OTHER two categories.
-- The calculator then COMPUTES the realized generalized Fleiss kappa over the
-- per-group label distribution — a MEASURED statistic, NEVER a hardcoded constant.
--
-- METHODOLOGY (resolves the open question in favor of KEEP-VARYING-RATERS):
-- raters/group VARY {2:14, 3:19, 4:17}. We do NOT normalize to a fixed n (that
-- would fabricate or delete raters). The calculator computes the GENERALIZED Fleiss
-- kappa (Fleiss 1971, per-subject n_i) in PURE NUMPY; statsmodels.fleiss_kappa
-- (fixed-n only; it asserts n_total == n_sub*n_rat) is the TEST-ONLY oracle.
-- VERIFIED: hand-rolled == statsmodels to 1e-9 on every fixed-n subset
-- (n=2: 0.6485, n=3: 0.7236, n=4: 0.8494).
--
-- LIVE-VERIFIED (rolled-back txn against supabase-db, 153 rows / 50 groups):
--   p_agree=92 -> realized corpus generalized kappa = 0.7565 (Landis-Koch
--     SUBSTANTIAL; clears warning 0.70, BELOW target 0.85 — so DQ-008 reads
--     WARNING). category marginals balanced (Pe=0.3428 ~ 1/3, so kappa is NOT
--     inflated by a skewed prior).
--   NOISE baseline (pre-reseed) kappa = 0.0174 ~ 0.  SHUFFLE (n=200) mean = -0.0072
--     ~ 0 -> kappa responds ONLY to injected agreement (non-fabricated).
--   CALIBRATION IS MEASURED, NOT CHERRY-PICKED: over 40 independent label draws at
--     p_agree=0.92 the generalized kappa has mean 0.7519, sd 0.0682; the shipped
--     hashtext realization 0.7565 sits at z=+0.07 (essentially the mean). We do NOT
--     crank p_agree toward 1.0 to clear 0.85 (kappa=1.0 is suspiciously perfect);
--     p_agree=95 realizes only ~0.78 here. 0.7565 is an honest expert-panel value
--     (literature norms for 3-cat clinical IAA: 0.62-0.90). The honest lever for a
--     robust GOOD is MORE raters/group (variance shrinks), not higher fidelity —
--     the team may wish to revisit the 0.85 target (out of #577 scope).
--   IDEMPOTENT / RE-RUNNABLE: hashtext recomputes identically -> double-apply yields
--     identical 0.7565. Idempotency invariant: iaa_group_id is IMMUTABLE, so both
--     UPDATEs recompute the same per-group type + truth on re-apply (do NOT re-key
--     either UPDATE on a mutable column). No ADD COLUMN (annotation_value/type exist).
--
-- SOURCE-OF-TRUTH FRAMING (honest, not co-equal): this MIGRATION is the AUTHORITATIVE
-- reseed for the served DB (its two UPDATEs overwrite every label/type). The generator
-- src/ml/data_generator.py._generate_ml_annotations is the COHERENT MIRROR so a
-- from-scratch regenerate (without 052) is ALSO coherent — same group-level type +
-- latent-truth + 0.92 concordance design. They are coherent-EQUIVALENT, NOT
-- byte-identical (the generator uses unseeded module-level random; this migration uses
-- hashtext determinism). The live e2e asserts a RANGE (kappa > 0.6) against the
-- migration-reseeded DB, never the exact 0.7565.
--
-- WHY SURGICAL (not truncate+reload): ml_annotations is consumed by data_loader.py
-- (structure-agnostic JSONB decode) + v_kpi_label_quality + the new calculator only.
-- In-place UPDATE preserves every annotation_id (no orphan risk) and the row count.
--
-- BLAST RADIUS (verified live; disclosed honestly): the LABEL relabel is invisible to
-- v_kpi_label_quality (it reads the annotation_confidence COLUMN + is_adjudicated, never
-- the label). The annotation_confidence numeric column AND the JSONB 'confidence'/'notes'
-- sub-keys are ALL left untouched (only annotation_value->>'label' + annotation_type
-- change). The REQUIRED group-level annotation_type fix DOES reshuffle the view's
-- PER-TYPE rows (groups per type go from mixed 24-34 to a clean partition ~11/14/14/11;
-- per-type total_annotations/avg_confidence shift) — but CORPUS TOTALS are preserved
-- (153 rows, 50 groups, overall avg_confidence 0.8469), and NO calculator/view/frontend
-- reads the per-type breakdown of v_kpi_label_quality (grep-verified). 0 kpi_query_registry
-- rows reference ml_annotations today. This per-type reshuffle is the correct, disclosed
-- cost of IAA coherence.
--
-- SNAPSHOT pre-apply: /tmp/577_safety/ml_annotations_pre052.csv
--   (annotation_id, iaa_group_id, annotation_type, annotation_value->>'label').
--
-- NOTE: deploy.yml SKIPS migrations; the local self-contained supabase is the faithful
-- target. Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/052_kpi_577_label_quality_iaa.sql
-- ----------------------------------------------------------------------------

-- (A) STRUCTURAL FIX: one annotation_type per iaa_group (raters co-rate the SAME
--     subject — the IAA precondition). Deterministic via the 'atype' hashtext salt,
--     keyed on the IMMUTABLE iaa_group_id (idempotency invariant).
UPDATE public.ml_annotations a
   SET annotation_type = (ARRAY['diagnosis_validation', 'outcome_label', 'treatment_response', 'adverse_event'])[
       1 + (abs(hashtext(a.iaa_group_id || 'atype')::bigint) % 4)];

-- (B) LATENT-TRUTH relabel of annotation_value->>'label' (jsonb_set; in-place, no row
--     churn). Per-group true label (salt 'truth', keyed on iaa_group_id); each annotator
--     agrees with prob 92 (salt 'agree', keyed on annotation_id), else picks one of the
--     OTHER two categories (salt 'flip'). DISTINCT salts keep the draws independent so
--     kappa is a real concordance signal, not a deterministic artifact of correlated
--     hashes. ::bigint cast guards abs(hashtext()) against INT_MIN overflow.
UPDATE public.ml_annotations a
   SET annotation_value = jsonb_set(
       a.annotation_value, '{label}',
       to_jsonb(
         CASE
           WHEN (abs(hashtext(a.annotation_id || 'agree')::bigint) % 100) < 92
           THEN g.true_label
           ELSE (
             SELECT c FROM (SELECT unnest(ARRAY['positive', 'negative', 'uncertain']) AS c) cats
              WHERE c <> g.true_label
              OFFSET (abs(hashtext(a.annotation_id || 'flip')::bigint) % 2) LIMIT 1
           )
         END
       ), false)
  FROM (
    SELECT iaa_group_id,
           (ARRAY['positive', 'negative', 'uncertain'])[
             1 + (abs(hashtext(iaa_group_id || 'truth')::bigint) % 3)] AS true_label
    FROM public.ml_annotations GROUP BY iaa_group_id
  ) g
 WHERE a.iaa_group_id = g.iaa_group_id;

-- (C) Register the read-only WS1-DQ-008 statement (allowlist; executed only via kpi_query).
--     Returns ONE fixed-width row PER iaa_group: per-category counts + rater count, so the
--     Python calculator pivots into a subjects×categories matrix and computes the
--     generalized Fleiss kappa. max_params=0 (corpus-level pooled kappa; ml_annotations has
--     no brand column — same rationale as DQ-002). HAVING n_raters>=2 drops singletons (no
--     pairwise agreement term). Starts with WITH to satisfy the registry read-only CHECK.
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('data_quality_label_quality', $kpi$WITH per_group AS (SELECT iaa_group_id, COUNT(*) FILTER (WHERE annotation_value->>'label' = 'positive') AS n_positive, COUNT(*) FILTER (WHERE annotation_value->>'label' = 'negative') AS n_negative, COUNT(*) FILTER (WHERE annotation_value->>'label' = 'uncertain') AS n_uncertain, COUNT(*) AS n_raters FROM public.ml_annotations WHERE iaa_group_id IS NOT NULL AND annotation_value->>'label' IS NOT NULL GROUP BY iaa_group_id HAVING COUNT(*) >= 2) SELECT iaa_group_id, n_positive, n_negative, n_uncertain, n_raters FROM per_group$kpi$, 0, $note$WS1-DQ-008 label_quality (IAA): returns per iaa_group the fixed-width category counts (n_positive/n_negative/n_uncertain) + n_raters over annotation_value->>'label', for the calculator to compute the corpus-level GENERALIZED Fleiss kappa (Fleiss 1971, per-subject n_i; varying 2-4 raters/group). max_params=0 (pooled; no brand band — ml_annotations has no brand). HAVING n_raters>=2 drops singletons (kappa undefined for 1 rating). FAIL-LOUD: empty result (0 groups) -> calculator raises 'unavailable'; a real kappa (incl. low/negative) is RETURNED. The YAML formula 'avg(agreement_score) for iaa_groups' is loose shorthand — the honest standard IAA is chance-corrected Fleiss kappa. Higher-is-better (NOT in lower_is_better set {DQ-006,DQ-007}). Live post-reseed: corpus kappa=0.7565 (noise baseline 0.0174; shuffle ~0).$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_id is callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
