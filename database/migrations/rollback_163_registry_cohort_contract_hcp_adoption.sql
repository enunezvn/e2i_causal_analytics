-- ROLLBACK for migration 163 (HCP-adoption cohort contract, #2286/#2287). NOT a forward
-- migration: scripts/run_migrations.sh skips rollback_*.sql in apply_dir().
--
-- Restores cohort_data_source and cohort_feature_manifest_source to NULL on the 3 rows --
-- but only where they still hold EXACTLY the pair 163 wrote, so a contract healed or
-- edited since is left alone. cohort_target_outcome ('adopted', migration 151) is
-- untouched. Clears 163's ledger row. Apply BEFORE rollback_162 (which refuses otherwise).
--
-- Apply by hand:
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/migrations/rollback_163_registry_cohort_contract_hcp_adoption.sql

UPDATE ml_model_registry
   SET cohort_data_source = NULL,
       cohort_feature_manifest_source = NULL
 WHERE model_name = 'hcp_adoption_remibrutinib_goldstd_lr_v1'
   AND is_synthetic = false
   AND cohort_data_source = '{"columns": ["peer_influence_score", "influence_network_size", "years_experience", "specialty", "geographic_region", "adopted"], "filters": {"brand": "Remibrutinib", "is_synthetic": true}, "table": "hcp_adoption_goldstd_v", "type": "table"}'
   AND cohort_feature_manifest_source = 'synthetic_csu';

UPDATE ml_model_registry
   SET cohort_data_source = NULL,
       cohort_feature_manifest_source = NULL
 WHERE model_name = 'hcp_adoption_fabhalta_goldstd_lr_v1'
   AND is_synthetic = false
   AND cohort_data_source = '{"columns": ["peer_influence_score", "influence_network_size", "years_experience", "specialty", "geographic_region", "adopted"], "filters": {"brand": "Fabhalta", "is_synthetic": true}, "table": "hcp_adoption_goldstd_v", "type": "table"}'
   AND cohort_feature_manifest_source = 'synthetic_csu';

UPDATE ml_model_registry
   SET cohort_data_source = NULL,
       cohort_feature_manifest_source = NULL
 WHERE model_name = 'hcp_adoption_kisqali_goldstd_lr_v1'
   AND is_synthetic = false
   AND cohort_data_source = '{"columns": ["peer_influence_score", "influence_network_size", "years_experience", "specialty", "geographic_region", "adopted"], "filters": {"brand": "Kisqali", "is_synthetic": true}, "table": "hcp_adoption_goldstd_v", "type": "table"}'
   AND cohort_feature_manifest_source = 'synthetic_csu';

DELETE FROM public.schema_migrations WHERE filename = '163_registry_cohort_contract_hcp_adoption.sql';
