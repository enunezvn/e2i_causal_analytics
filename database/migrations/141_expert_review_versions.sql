-- ============================================================================
-- Migration 141: expert_review_versions (#1991 debt 3, lane 4)
-- ============================================================================
-- WHAT: one row per DAG structure an estimand's review has covered. A REVIEW-band
--   run on an estimand with a pending review APPENDS here when its hash differs
--   from the latest version (expert_review_gate.check_approval), and the API
--   renders the diff between consecutive versions (dag_hash.get_dag_changes).
--   Backfill: one version per existing review from its current hash and snapshot,
--   so history starts full (41 reviews, all with a hash, on 2026-09-13).
-- WHY reviewer_id feeds query_id: expert_reviews.reviewer_id holds the REQUESTER
--   query id (migration 136), which is the run that produced the structure.
--   created_at (not updated_at, which is trigger-maintained) is the version time.
-- WHY explicit grants (a deviation from local precedent, not a copy of it): no
--   migration in this repo grants a per-table REVOKE/GRANT block -- grepping
--   database/migrations/ for "REVOKE ALL ON public.expert_reviews" / "GRANT ...
--   ON public.expert_reviews" finds nothing, including in expert_reviews' own
--   140/136/097. Every table created after migration 058 (which neutralised the
--   Supabase default-privilege re-grant to anon/authenticated and left service_role
--   the ambient default ACL) relies on that implicit default: postgres owns it,
--   service_role gets ALL (measured live 2026-09-13: service_role holds
--   SELECT/INSERT/UPDATE/DELETE/TRUNCATE/REFERENCES/TRIGGER on expert_reviews,
--   not just SELECT/INSERT), anon/authenticated/PUBLIC get nothing. There is no
--   literal per-table pattern to mirror. This table is deliberately narrower:
--   it is an append-only version log with no UPDATE/DELETE consumer in Tasks 4
--   (append) or 5 (read) -- so an explicit REVOKE ALL + GRANT SELECT, INSERT
--   restricts even service_role below the ambient default (no UPDATE/DELETE/
--   TRUNCATE on history), least-privilege for an audit trail rather than a
--   convention this repo already follows. FROM PUBLIC is a no-op today (PG15
--   dropped default table grants to PUBLIC, confirmed live: PUBLIC holds no
--   grant on expert_reviews) but kept for the same explicit-defense-in-depth
--   style as migration 044's kpi_query_registry REVOKE.
-- SAFETY: purely additive; the backfill is idempotent (ON CONFLICT DO NOTHING on
--   the (review_id, dag_version_hash) key); no own transaction (run_migrations.sh
--   wraps the file in --single-transaction and appends the ledger row).
-- REVERSE: DROP TABLE public.expert_review_versions;
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.expert_review_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID NOT NULL REFERENCES public.expert_reviews(review_id) ON DELETE CASCADE,
    dag_version_hash VARCHAR(64) NOT NULL,
    adjustment_set_hash VARCHAR(64),
    dag_structure_json JSONB,
    query_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (review_id, dag_version_hash)
);

CREATE INDEX IF NOT EXISTS idx_erv_review_created
    ON public.expert_review_versions (review_id, created_at);

INSERT INTO public.expert_review_versions (review_id, dag_version_hash, dag_structure_json, query_id, created_at)
SELECT review_id, dag_version_hash, dag_structure_json, reviewer_id, created_at
  FROM public.expert_reviews
 WHERE dag_version_hash IS NOT NULL
ON CONFLICT (review_id, dag_version_hash) DO NOTHING;

REVOKE ALL ON public.expert_review_versions FROM PUBLIC, anon, authenticated;
-- service_role also needs an explicit REVOKE: the default ACL (migration 058)
-- already granted it ALL at CREATE TABLE time, above, before this line runs --
-- GRANT SELECT, INSERT alone would only ADD those two on top of the untouched
-- ALL grant, not narrow it (caught live in the 140+141 rehearsal: the grants
-- readout showed service_role still holding DELETE/UPDATE/TRUNCATE after a
-- REVOKE/GRANT pair that omitted this line).
REVOKE ALL ON public.expert_review_versions FROM service_role;
GRANT SELECT, INSERT ON public.expert_review_versions TO service_role;

COMMENT ON TABLE public.expert_review_versions IS
    'Structure versions an expert review has covered (migration 141, #1991 debt 3): the diff '
    'between consecutive rows is what a reviewer sees when a re-run changes the DAG.';
