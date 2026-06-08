-- ============================================================================
-- Migration 059: WS1-MP-001 (Model Accuracy / ROC-AUC) — wire the REAL
-- ml_predictions.model_auc column via the kpi_query allowlist.
-- ============================================================================
-- THE GAP: config/kpi_definitions.yaml declares WS1-MP-001 ("roc_auc") with
--   tables:[ml_predictions] columns:[ml_predictions.model_auc]
-- but src/kpi/calculators/model_performance.py::_calc_roc_auc IGNORED that and
-- read MLflow only (get_latest_versions("default_model")/run metric "roc_auc").
-- In prod no "default_model" is registered and the latest runs log no roc_auc,
-- so _get_metric_from_mlflow fail-closes to (None, "model_not_found"/"metric_
-- not_found") and GET /api/kpis/WS1-MP-001 returns no value — even though the
-- declared source HAS real data. The Home "Model Accuracy" tile therefore had
-- no real value to show and was hardcoded to a fabricated 94.2%.
--
-- THE FIX (this migration + same-PR calculator change): register a read-only
-- statement that AVERAGEs the real ml_predictions.model_auc, and change
-- _calc_roc_auc to try this SQL leg FIRST (kpi_query allowlist, exactly like
-- the sibling model_performance_shap_coverage / model_performance_feature_drift
-- calculators) and fall back to MLflow only when SQL is genuinely unavailable.
-- The MLflow leg is KEPT as the fail-closed fallback (not deleted).
--
-- LIVE-VERIFIED (against supabase-db): ml_predictions has 4364 rows, 626 with a
-- non-null model_auc; AVG(model_auc) = 0.7997923... (~0.80). So after this
-- migration + calculator change, GET /api/kpis/WS1-MP-001 returns ~0.80 — the
-- real corpus mean ROC-AUC, not a fabricated constant.
--
-- max_params=0: corpus-level aggregate (no per-model band; ml_model_registry
-- has 0 rows so an honest per-model lookup is not available). Same arity-0
-- rationale as model_performance_shap_coverage (mig 044) and
-- model_performance_feature_drift (mig 053). The calculator's SQL-leg call is
-- `_execute_query("model_performance_roc_auc", [])` — arity 0 matches.
--
-- FAIL-LOUD: empty table or all-NULL model_auc -> AVG over 0 rows is NULL ->
-- the calculator's `roc_auc is None` branch falls through to the MLflow leg ->
-- fail-closed UNKNOWN. No fabricated default is ever returned.
--
-- Column alias `roc_auc` matches model_performance.py result[0].get("roc_auc").
-- Starts with SELECT to satisfy the registry read-only CHECK
-- (kpi_query_registry_readonly_chk: sql ~* '^\s*(with|select)\s').
--
-- IDEMPOTENT / RE-RUNNABLE: ON CONFLICT (query_id) DO UPDATE.
--
-- NOTE: deploy.yml SKIPS migrations; the local self-contained supabase IS the
-- faithful prod target. Apply manually (same as 044/052/053):
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/059_kpi_ws1_mp_001_roc_auc_from_ml_predictions.sql
-- ----------------------------------------------------------------------------

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('model_performance_roc_auc',
     $kpi$SELECT AVG(model_auc)::float AS roc_auc FROM public.ml_predictions WHERE model_auc IS NOT NULL$kpi$,
     0,
     $note$WS1-MP-001 ROC-AUC: AVG(model_auc) over ml_predictions rows with a non-null model_auc (the source declared by kpi_definitions.yaml). max_params=0 (corpus aggregate; no per-model band — ml_model_registry empty). LOWER-is-NOT-better. FAIL-LOUD: 0 rows / all-NULL -> AVG NULL -> calculator falls back to the MLflow leg -> fail-closed UNKNOWN. Live AVG ~= 0.7998 (n=626). Calculator: src/kpi/calculators/model_performance.py::_calc_roc_auc (SQL primary, MLflow fallback).$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_id is callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
