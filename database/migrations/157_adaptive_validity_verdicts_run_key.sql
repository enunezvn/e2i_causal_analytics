-- ============================================================================
-- Migration 157: adaptive_validity_verdicts keys on the RUN (issue #2260)
-- ============================================================================
-- 040 keyed the mirror on (experiment_id, feature, written_at), and written_at
-- has second resolution. Since #2257, runs of one scope share the scope's
-- experiment id, so two runs of one scope in one second produced the same key
-- and scripts/mirror_audit_sidecar_to_supabase.py kept one run's verdicts and
-- overwrote the other's.
--
-- Sidecar schema 1.10 carries the pipeline run's ``audit_workflow_id`` (minted
-- once per MLFoundationPipeline.run and threaded into the data-preparer state).
-- This migration adds it as a column and replaces 040's unique index with one
-- that includes it.
--
-- NULL run id (a pre-1.10 sidecar): the column stays NULL (no fake id is
-- written) and the index folds NULL to the nil UUID, so legacy sidecars dedup
-- exactly as they did under 040. Same COALESCE-sentinel device 040 uses for
-- experiment_id / feature (Postgres UNIQUE treats every NULL as distinct).
--
-- DEPLOY LOCKSTEP: the mirror's ON CONFLICT target must match the unique
-- index expression for expression. Pre-157 code names 040's three expressions
-- and fails ("no unique or exclusion constraint matching") once 040's index is
-- gone; post-157 code names four and fails without this file. Apply with the
-- code deploy. The mirror has no scheduler today and prod held 0 rows when
-- this was written (2026-09-23).
--
-- Adding a column can only make keys MORE distinct, so the new index cannot
-- fail on rows 040's index already admitted.
--
-- No script-level BEGIN/COMMIT: scripts/run_migrations.sh wraps the file in
-- --single-transaction, so the drop and the create land together.
-- ============================================================================

ALTER TABLE adaptive_validity_verdicts
    ADD COLUMN IF NOT EXISTS audit_workflow_id UUID;

COMMENT ON COLUMN adaptive_validity_verdicts.audit_workflow_id IS
'The ML-foundation pipeline run that wrote the sidecar (sidecar payload.audit_workflow_id, schema 1.10+). NULL for pre-1.10 sidecars. Part of the natural key (migration 157, #2260): runs of one scope share experiment_id and written_at has second resolution.';

CREATE UNIQUE INDEX IF NOT EXISTS uix_adaptive_validity_verdicts_run_key
    ON adaptive_validity_verdicts (
        COALESCE(experiment_id, '__unknown__'),
        COALESCE(feature, '__unknown__'),
        written_at,
        COALESCE(audit_workflow_id, '00000000-0000-0000-0000-000000000000'::uuid)
    );

COMMENT ON INDEX uix_adaptive_validity_verdicts_run_key IS
'Mirror upsert key (#2260): 040''s (experiment_id, feature, written_at) plus the run id. NULL run id folds to the nil UUID so legacy sidecars keep 040''s dedup. The ON CONFLICT target in scripts/mirror_audit_sidecar_to_supabase.py names these four expressions.';

-- 040's key is dropped, not kept beside the new one: it forbids two runs of one
-- scope in one second from coexisting, which is the bug.
DROP INDEX IF EXISTS uix_adaptive_validity_verdicts_natural_key;
