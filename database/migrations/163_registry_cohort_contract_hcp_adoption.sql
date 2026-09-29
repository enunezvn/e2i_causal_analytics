-- Migration 163: the cohort contract for the 3 HCP-adoption goldstd registry rows
-- (Part of #2286 / #2287; closes the "STAYS NULL" items migration 151 left for them).
--
-- WHY: migration 151 set cohort_target_outcome = 'adopted' on
-- hcp_adoption_<brand>_goldstd_lr_v1 but left cohort_data_source NULL for three
-- reasons, now all resolved:
--   1. the scope_definer rewrote 'adopted' -> 'will_adopt' -- fixed by #2284: the
--      sweep's retrain input pins target_outcome as target_variable_hint
--      (src/tasks/drift_monitoring_tasks.py _cohort_input_from_training_config), so
--      scope_spec.prediction_target is the physical column 'adopted';
--   2. hcp_brand_adoption was not in MLDataLoader.ML_TABLES (#2286) -- the contract
--      names the view instead, which IS allowlisted and provenance-tagged;
--   3. the frame is a two-table join (#2287) -- migration 162's view
--      hcp_adoption_goldstd_v materialises exactly that join.
-- Until a row carries BOTH data_source and target_outcome, has_cohort_contract() is
-- False and the drift sweep refuses to enqueue (retraining_blocked_reason =
-- "no_cohort_contract").
--
-- cohort_data_source (per brand), from the goldstd training code
-- (src/mlops/gold_standard_eval/cohort_spec.py make_hcp_spec, feature_builder.py
-- _load_hcp_frame): relation hcp_adoption_goldstd_v; filters brand=<brand> AND
-- is_synthetic=true (the builder's own predicates; explicit so the load does not depend
-- on E2I_INCLUDE_SYNTHETIC); columns = _HCP_COVARIATES + the label 'adopted'. The view
-- carries data_split, so the loader honours the goldstd 60/20/10/10 split verbatim
-- (#2207 split contract) -- the champions were fit on train+validation and scored on
-- holdout of that same split.
--
-- cohort_feature_manifest_source = 'synthetic_csu', for the reason 151 gave the 9
-- patient rows: it is the Layer-5 manifest of the DGP that seeded these rows. Its
-- _HCP_FEATURES (src/data/manifests/synthetic_csu_feature_manifest.py) declare
-- influence_network_size / peer_influence_score / years_experience / specialty
-- pre-index (geographic_region via the patient list), because the adoption DGP draws
-- adopted FROM them (hcp_brand_adoption_generator: centrality_z = standardised
-- peer_influence_score = log1p(influence_network_size), then _compute_adoption), and
-- the manifest docstring records this exact false positive on hcp_adoption (the
-- drivers dropped, AUC 0.78 -> 0.51). Measured on the data_preparer walk
-- (docs/demos/results/2026-09-29_2286_2287_hcp_adoption_contract/): WITHOUT it Layer 3
-- flags peer_influence_score and influence_network_size HIGH and routes every brand's
-- retrain to LLM leakage remediation; WITH it declared-safe immunity keeps them and
-- the QC gate passes (gate_passed=true, blocking_issues=[]) for all 3 brands.
--
-- SAFETY: data_source and manifest are written as ONE compare-and-set unit -- only a
-- row whose cohort_data_source AND cohort_feature_manifest_source are still NULL AND
-- whose cohort_target_outcome is exactly 'adopted' (151's value) is written, so a
-- healed or hand-edited contract is never overwritten and a mixed contract is never
-- composed; scoped to the production rows (stage = 'production', is_synthetic = false)
-- -- ml_model_registry is unique on (model_name, model_version), so an archived version,
-- a staged one or a retrain candidate can share the name and must not be touched. The JSON literals are encode_data_source()
-- output (json.dumps(sort_keys=True)), so the sweep's decode/encode round trip is the
-- identity. Idempotent: a second application matches zero rows.
--
-- ORDER: needs 162 (the view) applied and the #2286 allowlist deployed before the sweep
-- reads a row. The deploy applies migrations BEFORE it recreates the app containers,
-- and a failed recreate rolls the containers back but not the migrations. In either
-- state an OLD worker sees this contract without the view in ML_TABLES, and a retrain
-- the daily sweep (01:45, auto_approve=False) enqueues then fails LOUD in the loader
-- ("not supported") -- never wrong values. Deploy outside 01:45; if the deploy rolls
-- back, apply rollback_163 until the code is live.
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

UPDATE ml_model_registry
   SET cohort_data_source = '{"columns": ["peer_influence_score", "influence_network_size", "years_experience", "specialty", "geographic_region", "adopted"], "filters": {"brand": "Remibrutinib", "is_synthetic": true}, "table": "hcp_adoption_goldstd_v", "type": "table"}',
       cohort_feature_manifest_source = 'synthetic_csu'
 WHERE model_name = 'hcp_adoption_remibrutinib_goldstd_lr_v1'
   AND stage = 'production'
   AND is_synthetic = false
   AND cohort_data_source IS NULL
   AND cohort_feature_manifest_source IS NULL
   AND cohort_target_outcome = 'adopted';

UPDATE ml_model_registry
   SET cohort_data_source = '{"columns": ["peer_influence_score", "influence_network_size", "years_experience", "specialty", "geographic_region", "adopted"], "filters": {"brand": "Fabhalta", "is_synthetic": true}, "table": "hcp_adoption_goldstd_v", "type": "table"}',
       cohort_feature_manifest_source = 'synthetic_csu'
 WHERE model_name = 'hcp_adoption_fabhalta_goldstd_lr_v1'
   AND stage = 'production'
   AND is_synthetic = false
   AND cohort_data_source IS NULL
   AND cohort_feature_manifest_source IS NULL
   AND cohort_target_outcome = 'adopted';

UPDATE ml_model_registry
   SET cohort_data_source = '{"columns": ["peer_influence_score", "influence_network_size", "years_experience", "specialty", "geographic_region", "adopted"], "filters": {"brand": "Kisqali", "is_synthetic": true}, "table": "hcp_adoption_goldstd_v", "type": "table"}',
       cohort_feature_manifest_source = 'synthetic_csu'
 WHERE model_name = 'hcp_adoption_kisqali_goldstd_lr_v1'
   AND stage = 'production'
   AND is_synthetic = false
   AND cohort_data_source IS NULL
   AND cohort_feature_manifest_source IS NULL
   AND cohort_target_outcome = 'adopted';
