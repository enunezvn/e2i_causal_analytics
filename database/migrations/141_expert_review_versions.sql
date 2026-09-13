-- ============================================================================
-- Migration 141: expert_review_versions (#1991 debt 3, lane 4)
-- ============================================================================
-- WHAT: one row per DAG structure an estimand's review has covered. A REVIEW-band
--   run appends a row when its hash differs from the review's CURRENT (latest)
--   version (expert_review_gate.check_approval); a revert (A -> B -> A -- discovery
--   is not fully deterministic across restarts, migration 140's sibling PR) appends
--   AGAIN rather than being suppressed, so the log is a timeline, not a set, and
--   consecutive rows always diff honestly for the API (dag_hash.get_dag_changes).
--   Backfill: one version per existing review from its current hash and snapshot,
--   so history starts full (41 reviews, all with a hash, on 2026-09-13).
--   adjustment_set_hash is NULL on all 41 backfilled rows: expert_reviews has no such
--   column to backfill FROM, so the review's structure hash is preserved but its
--   adjustment-set hash is unknown until Task 4 begins appending -- every row Task 4
--   appends going forward populates adjustment_set_hash (it is computed at append time,
--   not derived from expert_reviews).
--   Writers MUST use PostgREST/supabase-py insert(), never upsert(): upsert() defaults to
--   ON CONFLICT ... DO UPDATE, which requires the UPDATE privilege on this table --
--   service_role does not hold it (see the grants WHY below) and would get 42501.
-- WHY no UNIQUE (review_id, dag_version_hash): a revert's third row would collide
--   with the first row's hash under that constraint -- Task 4's PostgREST insert
--   has no ON CONFLICT, so the insert would fail 23505 outright, and even a
--   DO NOTHING would leave the LATEST stored row at the prior (B) hash while the
--   review's actual current hash is back to A, a plausible-wrong display. The
--   backfill's own idempotence (re-running this file must not duplicate the 41
--   seed rows) is instead a WHERE NOT EXISTS guard on this migration's one-time
--   INSERT; ongoing idempotence for a REPEATED hash (not a revert) is a Python
--   concern (expert_review_gate compares against the latest row), not a DB
--   constraint's job.
-- WHY ck_erv_snapshot_object: migration 137 found 40 expert_reviews rows where
--   dag_structure_json had been stored as a JSON *string* scalar, not an object
--   (src/repositories/expert_review.py once json.dumps'ed before the write). This
--   table's diff path (dag_hash.get_dag_changes) must never see that shape again --
--   a CHECK enforces object-or-NULL on every row from the start, including this
--   migration's own backfill INSERT.
-- WHY reviewer_id feeds query_id: expert_reviews.reviewer_id holds the REQUESTER
--   query id (migration 136), which is the run that produced the structure.
--   created_at (not updated_at, which is trigger-maintained) is the version time.
-- WHY explicit grants (a deviation from local precedent, not a copy of it): no
--   migration touching expert_reviews itself (097/136/140) grants a per-table
--   REVOKE/GRANT block -- grepping database/migrations/ for "REVOKE ALL ON
--   public.expert_reviews" / "GRANT ... ON public.expert_reviews" finds nothing
--   there (migration 044's kpi_query_registry DOES use this per-table style, so
--   the pattern exists in this repo -- expert_reviews itself just never adopted
--   it). expert_reviews instead relies on the implicit default ACL that migration
--   058 left in place for postgres/service_role after neutralising the
--   anon/authenticated re-grant: postgres owns it, service_role gets ALL
--   (measured live 2026-09-13: service_role holds SELECT/INSERT/UPDATE/DELETE/
--   TRUNCATE/REFERENCES/TRIGGER on expert_reviews, not just SELECT/INSERT),
--   anon/authenticated/PUBLIC get nothing. This table is deliberately narrower:
--   it is an append-only timeline with no UPDATE/DELETE consumer in Tasks 4
--   (append) or 5 (read) -- so an explicit REVOKE ALL + GRANT SELECT, INSERT
--   restricts even service_role below the ambient default (no UPDATE/DELETE/
--   TRUNCATE on history), least-privilege for an audit trail rather than the
--   implicit-default convention expert_reviews itself follows. FROM PUBLIC is a
--   no-op today (PG15 dropped default table grants to PUBLIC, confirmed live:
--   PUBLIC holds no grant on expert_reviews) but kept for the same explicit
--   defense-in-depth style as 044's kpi_query_registry REVOKE. service_role is
--   SELECT+INSERT ONLY by design; a future UPDATE/DELETE path on this table
--   (e.g. a correction workflow) needs its own migration to loosen this.
-- SAFETY: purely additive; the backfill is idempotent (a WHERE NOT EXISTS guard
--   on (review_id, dag_version_hash), since there is no unique constraint to
--   target with ON CONFLICT -- see the timeline WHY above); no own transaction
--   (run_migrations.sh wraps the file in --single-transaction and appends the
--   ledger row).
-- REVERSE (manual, not run by this file): DROP TABLE public.expert_review_versions;
--   also DELETE FROM public.schema_migrations WHERE filename =
--   '141_expert_review_versions.sql' -- run_migrations.sh's ledger check (line ~117)
--   skips any file already recorded there, so leaving that row in place would make a
--   later re-apply of this file silently no-op instead of recreating the table.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.expert_review_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID NOT NULL REFERENCES public.expert_reviews(review_id) ON DELETE CASCADE,
    dag_version_hash VARCHAR(64) NOT NULL,
    adjustment_set_hash VARCHAR(64),
    dag_structure_json JSONB,
    query_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_erv_snapshot_object CHECK (
        dag_structure_json IS NULL OR jsonb_typeof(dag_structure_json) = 'object'
    )
);

-- The ordering index for "versions of a review in time" -- this table is a timeline, not a
-- set keyed on (review_id, dag_version_hash); see the WHY above for why there is no UNIQUE.
CREATE INDEX IF NOT EXISTS idx_erv_review_created
    ON public.expert_review_versions (review_id, created_at);

INSERT INTO public.expert_review_versions (review_id, dag_version_hash, dag_structure_json, query_id, created_at)
SELECT r.review_id, r.dag_version_hash, r.dag_structure_json, r.reviewer_id, r.created_at
  FROM public.expert_reviews r
 WHERE NOT EXISTS (
           SELECT 1 FROM public.expert_review_versions v
            WHERE v.review_id = r.review_id AND v.dag_version_hash = r.dag_version_hash
       )
   AND r.dag_version_hash IS NOT NULL;

REVOKE ALL ON public.expert_review_versions FROM PUBLIC, anon, authenticated;
-- service_role also needs an explicit REVOKE: the Supabase default ACL, which migration 058
-- left in place for service_role, already granted it ALL at CREATE TABLE time, above, before
-- this line runs -- GRANT SELECT, INSERT alone would only ADD those two on top of the
-- untouched ALL grant, not narrow it (caught live in the 140+141 rehearsal: the grants
-- readout showed service_role still holding DELETE/UPDATE/TRUNCATE after a REVOKE/GRANT pair
-- that omitted this line).
REVOKE ALL ON public.expert_review_versions FROM service_role;
GRANT SELECT, INSERT ON public.expert_review_versions TO service_role;

COMMENT ON TABLE public.expert_review_versions IS
    'Structure versions an expert review has covered (migration 141, #1991 debt 3): the diff '
    'between consecutive rows is what a reviewer sees when a re-run changes the DAG.';
