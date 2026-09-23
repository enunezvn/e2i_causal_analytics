-- Migration 158: backfill ml_model_registry.training_provenance on the ONLY rows whose
-- training data is provable (#2259). Every other NULL row stays NULL.
--
-- WHY: #2259 makes the #968 promotion gate fail closed on NULL provenance. The owner approved
-- a provable-only backfill. The investigation (2026-09-23) classified all 1442 NULL rows:
--   * 1440 are the synthetic MLOps generator's fabricated rows (is_synthetic = true, no
--     artifact, no MLflow run, metrics from an RNG). No training happened, so there is no
--     training provenance to record; they stay NULL (the gate documents their exemption).
--   * 0 are provable from a cohort contract, a linked experiment, or an MLflow run.
--   * 2 are provable from the CODE of their writer (rule D): csu_treatment_initiation_lr_full_v1
--     and _balanced_v1, registered 2026-06-10 by src/mlops/prediction_synthesizer_deploy.py
--     under experiment csu_treatment_initiation_live_v1 (DEPLOY_EXPERIMENT_NAME). Every version
--     of that file up to the registration (721c41f72, b57583b1e, 054af77c7, a7ba54909,
--     cb0d6b73d) trains through generate_scenario(C_TREATMENT_CSU_RESPONSE), and no version
--     loads real data.
--
-- VALUE: 'synthetic_gold', the only synthetic value valid_training_provenance_registry allows
-- (migration 083). CAVEAT, stated so nobody reads more into it: the training data was the
-- synthetic_v2 scenario C cohort, not strictly the gold-standard cohort. What the value
-- asserts, and what the gate needs, is "trained on synthetic data only".
--
-- Both rows are archived today, so this changes no serving and no production decision; it
-- records what is known. The code now passes the same value for new rows.
--
-- SAFETY: compare-and-set. It writes only the two ids, only while training_provenance IS NULL
-- (a row healed by other means is never overwritten), and only while each row still matches
-- the proof (writer's experiment, names, version, artifact location, is_synthetic = false). A
-- second application matches zero rows.
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

UPDATE ml_model_registry r
   SET training_provenance = 'synthetic_gold'
  FROM ml_experiments e
 WHERE e.id = r.experiment_id
   AND r.id IN ('d765b451-12df-46df-955f-63359b506b52', '5fd7826b-28d7-491b-b9b1-8b5494dbe1ff')
   AND r.training_provenance IS NULL
   AND r.is_synthetic = false
   AND e.experiment_name = 'csu_treatment_initiation_live_v1'
   AND r.model_name IN ('csu_treatment_initiation_lr_full_v1',
                        'csu_treatment_initiation_lr_balanced_v1')
   AND r.model_version = '1.0'
   AND r.artifact_path LIKE '%/ml_artifacts/csu_treatment_initiation/%';
