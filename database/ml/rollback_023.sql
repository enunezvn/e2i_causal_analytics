-- Rollback migration 023_energy_score_tables.sql
-- This script removes the 023 schema to make way for the correct 011 schema
-- Created: 2025-12-27

-- Drop function first (depends on table and types)
DROP FUNCTION IF EXISTS log_estimator_evaluation;

-- Drop views (depend on table)
DROP VIEW IF EXISTS v_selection_comparison;
DROP VIEW IF EXISTS v_energy_score_trends;
DROP VIEW IF EXISTS v_estimator_performance;

-- Drop table (depends on types)
DROP TABLE IF EXISTS estimator_evaluations;

-- Drop ENUM types
DROP TYPE IF EXISTS quality_tier;
DROP TYPE IF EXISTS selection_strategy;
DROP TYPE IF EXISTS estimator_type;

-- Verify cleanup
DO $$
BEGIN
    RAISE NOTICE 'Rollback 023 complete. Verifying cleanup...';

    -- Check if types still exist
    IF EXISTS (SELECT 1 FROM pg_type WHERE typname IN ('estimator_type', 'selection_strategy', 'quality_tier')) THEN
        RAISE WARNING 'Some ENUM types still exist!';
    ELSE
        RAISE NOTICE 'All ENUM types removed successfully.';
    END IF;

    -- Check if table still exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'estimator_evaluations') THEN
        RAISE WARNING 'Table estimator_evaluations still exists!';
    ELSE
        RAISE NOTICE 'Table estimator_evaluations removed successfully.';
    END IF;
END $$;
