-- ============================================================================
-- Migration 048: #577 PR2 (causal trio) — wire CM-004 (counterfactual) by making
-- the synthetic counterfactual_outcome COHERENT, then registering the read-only
-- allowlist statement.
-- ============================================================================
-- Issue #577 (follow-up to #574). CM-004 previously raised a fail-loud RuntimeError
-- ("ml_predictions lacks treatment_received/counterfactual_treatment"). An adversarial
-- design-review established the REAL problem: counterfactual_outcome was an INDEPENDENT
-- uniform(0.2,0.8) draw (data_generator.py) with NO relationship to the factual
-- prediction_value (uniform(0,1)) or treatment_effect_estimate (uniform(0.05,0.3)) —
-- three independent draws. So mean(counterfactual_outcome) ~ E[uniform] ~ 0.5 was just
-- averaging noise under a causal label, and the factual−counterfactual contrast ~ 0
-- (two unrelated uniforms). Wiring it as-is would have been fabrication.
--
-- THE HONEST FIX (user-approved coherent rework): make counterfactual_outcome a REAL
-- do-contrast of the factual — the predicted outcome under the alternative arm =
-- factual minus the (additive) treatment effect, floored at 0 (an outcome cannot be
-- negative). Then the per-row contrast (prediction_value − counterfactual_outcome)
-- equals treatment_effect_estimate (exactly, for the ~84% of rows where the factual
-- exceeds the effect; the ~16% floored rows have counterfactual = 0, the smallest
-- physically-valid outcome). The generator (src/ml/data_generator.py) makes the same
-- edit so a fresh full regenerate stays coherent.
--
-- CM-004's VALUE is the counterfactual LEVEL E[Y(a')] = mean(counterfactual_outcome) —
-- deliberately distinct from CM-001 (ATE), which is the contrast E[Y(1)−Y(0)].
--
-- WHY A SURGICAL RESEED (not truncate+reload): ml_predictions carries out-of-band rows;
-- this migration only UPDATEs counterfactual_outcome in place (preserving every other
-- column and every row), scoped to rows that have a treatment_effect_estimate (the
-- coherent causal subset). Re-runnable (the UPDATE is idempotent — same inputs, same
-- output). Snapshot taken before first apply (/tmp/577_safety/ml_predictions_pre048.sql).
--
-- NOTE: CM-005 (mediation) is NOT wired here — causal_chain edges still don't reconcile
-- with causal_effect_size, so an honest decomposition needs a further generator rework
-- (PR3). It remains fail-loud.
--
-- deploy.yml SKIPS migrations; the local self-contained supabase is the faithful target.
-- Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/048_kpi_577_counterfactual.sql
-- ----------------------------------------------------------------------------

-- (A) Reseed counterfactual_outcome as the floored factual − treatment-effect contrast,
--     in place, for the coherent causal subset (rows with a treatment_effect_estimate).
-- ROUND to scale 3 (counterfactual_outcome is numeric(4,3)) so the stored value is the
-- exact floored contrast — no silent re-rounding on store.
UPDATE public.ml_predictions
   SET counterfactual_outcome = GREATEST(0, ROUND((prediction_value - treatment_effect_estimate)::numeric, 3))
 WHERE treatment_effect_estimate IS NOT NULL
   AND prediction_value IS NOT NULL;

-- (B) Register the read-only CM-004 statement (allowlist; executed only via kpi_query).
--     $1 = optional prediction_type filter ('' => all types). Reports the counterfactual
--     LEVEL plus the factual mean and the contrast (= mean treatment effect) for context.
-- The WHERE clause matches the migration's coherent subset EXACTLY (counterfactual_outcome
-- + prediction_value + treatment_effect_estimate all NOT NULL) so no row with a stale,
-- non-reseeded counterfactual_outcome can leak into the aggregate. mean_realized_contrast =
-- AVG(prediction_value − counterfactual_outcome) is the TRUE floor-attenuated contrast (it
-- equals the treatment effect on unclamped rows and is <= mean_effect when flooring bites);
-- mean_effect is the NOMINAL mean treatment_effect_estimate, returned for comparison.
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('causal_metrics_counterfactual', $kpi$SELECT AVG(counterfactual_outcome)::float AS mean_counterfactual, AVG(prediction_value)::float AS mean_factual, AVG(prediction_value - counterfactual_outcome)::float AS mean_realized_contrast, AVG(treatment_effect_estimate)::float AS mean_effect, COUNT(*)::int AS n FROM ml_predictions WHERE counterfactual_outcome IS NOT NULL AND prediction_value IS NOT NULL AND treatment_effect_estimate IS NOT NULL AND ($1 = '' OR prediction_type::text = $1)$kpi$, 1, $note$CM-004 counterfactual: counterfactual outcome LEVEL E[Y(a')] = mean(counterfactual_outcome), where counterfactual_outcome = GREATEST(0, prediction_value − treatment_effect_estimate) (a real do-contrast). $1 = prediction_type filter ('' = all). mean_realized_contrast = mean(factual − counterfactual) is the TRUE contrast (equals the treatment effect on unclamped rows; floor-attenuated, <= mean_effect, where the effect exceeds the factual); mean_effect = nominal mean treatment_effect_estimate. Distinct from CM-001 ATE. NULL mean_counterfactual (fail-loud) when no coherent rows match.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_id is callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
