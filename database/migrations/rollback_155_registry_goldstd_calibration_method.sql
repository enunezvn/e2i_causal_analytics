-- ROLLBACK for migration 155 (#2248). NOT a forward migration: scripts/run_migrations.sh
-- skips rollback_*.sql in apply_dir().
--
-- Removes ONLY the key 155 added, and only where it still holds the value 155 wrote, on
-- the 12 goldstd rows. A retrain of those rows then falls back to the auto policy.
--
-- Apply by hand:
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/migrations/rollback_155_registry_goldstd_calibration_method.sql
UPDATE ml_model_registry
   SET hyperparameters = hyperparameters - 'calibration_method'
 WHERE model_name IN (
         'initiation_fabhalta_goldstd_lr_v1',
         'initiation_kisqali_goldstd_lr_v1',
         'initiation_remibrutinib_goldstd_lr_v1',
         'persistence_fabhalta_goldstd_lr_v1',
         'persistence_kisqali_goldstd_lr_v1',
         'persistence_remibrutinib_goldstd_lr_v1',
         'discontinuation_fabhalta_goldstd_lr_v1',
         'discontinuation_kisqali_goldstd_lr_v1',
         'discontinuation_remibrutinib_goldstd_lr_v1',
         'hcp_adoption_fabhalta_goldstd_lr_v1',
         'hcp_adoption_kisqali_goldstd_lr_v1',
         'hcp_adoption_remibrutinib_goldstd_lr_v1'
       )
   AND hyperparameters ->> 'calibration_method' = 'sigmoid';
