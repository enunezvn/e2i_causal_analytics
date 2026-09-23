-- ROLLBACK for migration 155 (#2248). NOT a forward migration: scripts/run_migrations.sh
-- skips rollback_*.sql in apply_dir().
--
-- Removes ONLY what 155 could have written: the calibration_method key, where it still
-- holds 'sigmoid', on the 12 exact (id, trained_at) registrations 155 pins. A row
-- re-registered since (trained_at restamped) carries a method its own artifact recorded
-- and is left alone. hyperparameters on those rows was '{}' before 155 (audited
-- 2026-09-23), so removing the key restores it. A retrain of them then uses the auto policy.
--
-- Apply by hand:
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/migrations/rollback_155_registry_goldstd_calibration_method.sql

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = 'b933a5d0-cf0d-44cd-ac75-d34fab2259ea'
   AND model_name = 'discontinuation_fabhalta_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:04:58.395706+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = 'f2da0bd5-4d6b-47c4-9d30-6c8b4b59de46'
   AND model_name = 'discontinuation_kisqali_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:05:22.83531+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = '5d0486b4-fd74-463e-aab0-d91ac9fe0fca'
   AND model_name = 'discontinuation_remibrutinib_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:04:44.114318+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = 'dab9a91b-708d-4b0a-a332-803cef3d3245'
   AND model_name = 'hcp_adoption_fabhalta_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:06:23.393962+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = '8d1244df-7c38-435e-820f-1f201e51af24'
   AND model_name = 'hcp_adoption_kisqali_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:06:38.973136+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = '8b7a1e4e-be3c-469f-822b-46d9ab3dad81'
   AND model_name = 'hcp_adoption_remibrutinib_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:05:57.641562+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = 'a11a50e2-42ed-4233-8711-bbdf233d3a0d'
   AND model_name = 'initiation_fabhalta_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:03:12.266514+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = '4ec55d13-46c8-4df4-9ec8-7723fad67fb3'
   AND model_name = 'initiation_kisqali_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:03:31.021107+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = '2db8b0e0-1d9e-4e18-bf7a-576a4796610d'
   AND model_name = 'initiation_remibrutinib_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:02:51.799003+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = '3f5011b5-b16a-4ee7-afd8-4be9862e15d4'
   AND model_name = 'persistence_fabhalta_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:04:08.079086+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = 'd3a926ac-e24a-4847-ac15-152d2983e466'
   AND model_name = 'persistence_kisqali_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:04:23.038637+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';

UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE id = '64728d92-378a-407b-b670-4eb3337a63c1'
   AND model_name = 'persistence_remibrutinib_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:03:49.02015+00'
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';
