-- Migration 099: per-channel intervention treatment columns on business_metrics
--
-- WHY: the Digital Twin intervention dropdown exposes only interventions whose
-- effect is causally IDENTIFIED in the synthetic-gold cohort (PR #1050
-- identification gate). Today only digital_engagement (engagement_score) has a
-- planted effect, so the dropdown collapses to one option. These columns give
-- every canonical intervention -- plus the two new program-level levers
-- (patient_support_program, rep_training_quality) -- its own treatment channel
-- in business_metrics per_hcp_rollup rows, so the extended DGP backfill
-- (scripts/backfill_segment_engagement.py) can plant a documented, confounded,
-- per-region causal effect per channel and the direct DML estimator can
-- recover it honestly.
--
-- Additive + nullable: identical pattern to migration 033 (which added
-- engagement_score / call_frequency / conversion_rate). No existing reader
-- breaks; columns stay NULL until the backfill --execute populates
-- per_hcp_rollup rows.
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS email_campaign_count       NUMERIC;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS speaker_program_count      NUMERIC;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS sample_volume              NUMERIC;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS peer_influence_score       NUMERIC;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS patient_support_enrollment NUMERIC;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS rep_training_score         NUMERIC;

COMMENT ON COLUMN business_metrics.email_campaign_count IS
  'per_hcp_rollup treatment channel: emails delivered to the HCP in period (synthetic-gold DGP, intervention email_campaign)';
COMMENT ON COLUMN business_metrics.speaker_program_count IS
  'per_hcp_rollup treatment channel: speaker-program invitations attended (synthetic-gold DGP, intervention speaker_program_invitation)';
COMMENT ON COLUMN business_metrics.sample_volume IS
  'per_hcp_rollup treatment channel: sample units received (synthetic-gold DGP, intervention sample_distribution)';
COMMENT ON COLUMN business_metrics.peer_influence_score IS
  'per_hcp_rollup treatment channel: peer-network exposure intensity 0-10 (synthetic-gold DGP, intervention peer_influence_activation)';
COMMENT ON COLUMN business_metrics.patient_support_enrollment IS
  'per_hcp_rollup treatment channel: share (0-1) of the HCP''s patients enrolled in patient-support programs (synthetic-gold DGP, intervention patient_support_program)';
COMMENT ON COLUMN business_metrics.rep_training_score IS
  'per_hcp_rollup treatment channel: territory rep-training quality 0-10 (synthetic-gold DGP, intervention rep_training_quality)';
