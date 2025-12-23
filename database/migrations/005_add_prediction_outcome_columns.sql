-- Migration: 005_add_prediction_outcome_columns.sql
-- Description: Add ground truth columns to ml_predictions for concept drift detection
-- Date: 2025-12-23
-- Author: Claude Code
--
-- These columns enable tracking actual outcomes after predictions are made,
-- which is required for concept drift detection. The label lag (time between
-- prediction and outcome observation) varies by use case:
--   - HCP conversion: days to weeks
--   - Patient journey: weeks to months
--   - Market share: monthly cycles

-- Add ground truth columns to ml_predictions
ALTER TABLE ml_predictions
ADD COLUMN IF NOT EXISTS actual_outcome DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS outcome_recorded_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS outcome_source VARCHAR(50),
ADD COLUMN IF NOT EXISTS outcome_observation_window_days INTEGER;

-- Add comments for documentation
COMMENT ON COLUMN ml_predictions.actual_outcome IS 'The actual observed outcome (nullable until known)';
COMMENT ON COLUMN ml_predictions.outcome_recorded_at IS 'When the outcome was recorded';
COMMENT ON COLUMN ml_predictions.outcome_source IS 'Source of outcome: manual, automated, feedback_loop';
COMMENT ON COLUMN ml_predictions.outcome_observation_window_days IS 'Days between prediction and outcome observation';

-- Create index for querying predictions with outcomes (for concept drift analysis)
CREATE INDEX IF NOT EXISTS idx_ml_predictions_outcome_recorded
ON ml_predictions(outcome_recorded_at)
WHERE actual_outcome IS NOT NULL;

-- Create index for querying by outcome source
CREATE INDEX IF NOT EXISTS idx_ml_predictions_outcome_source
ON ml_predictions(outcome_source)
WHERE outcome_source IS NOT NULL;
