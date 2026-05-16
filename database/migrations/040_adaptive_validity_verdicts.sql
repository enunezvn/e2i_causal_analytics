-- ============================================================================
-- Migration 040: adaptive_validity_verdicts — Supabase mirror of per-run
--                audit sidecars for cross-experiment queryability (issue #238)
-- ============================================================================
-- Plan reference: cross-experiment audit-trail queryability.
--
-- Background: ``write_adaptive_verdicts_sidecar`` (src/agents/ml_foundation/
-- data_preparer/graph.py) emits one JSON sidecar per data-preparer run under
-- ``$ADAPTIVE_VALIDITY_ARTIFACTS_DIR``. Sidecars are canonical; this table
-- is a QUERYABLE MIRROR. The nightly batch worker
-- ``scripts/mirror_audit_sidecar_to_supabase.py`` upserts rows here.
--
-- Why a mirror table at all: directory scans across many sidecars don't
-- scale for cross-experiment questions like "disagreement rate by feature
-- family in Q1" or "which features has the evaluator critiqued most this
-- month". A small mirror table with the right indexes makes those direct
-- SQL queries.
--
-- ----------------------------------------------------------------------------
-- NULL-handling decision (Postgres-UNIQUE-permits-multiple-NULLs gotcha;
-- see memory `[[pr250-completion-20260516]]`).
-- ----------------------------------------------------------------------------
-- The natural key is ``(experiment_id, feature, written_at)``. Postgres
-- UNIQUE permits multiple NULLs (each NULL is distinct), so a NULL on any
-- key column would defeat the constraint and let duplicates accrue
-- silently.
--
-- Producer-side (``SidecarReader._build_record``) ALREADY coerces missing
-- ``experiment_id`` / ``feature`` to the literal string ``"<unknown>"``
-- (see ``str(payload.get("experiment_id", "<unknown>"))`` and the same
-- pattern for ``feature``). That means real upsert traffic never carries
-- NULL on these two columns.
--
-- HOWEVER, defense-in-depth: a future caller (e.g. an admin INSERT, a
-- backfill script, a different reader) could insert NULLs and silently
-- break the dedup contract. So this migration enforces BOTH:
--
--   (a) Column-level ``NOT NULL`` on ``experiment_id`` and ``feature``,
--       with a column DEFAULT of ``'__unknown__'`` — any insert that
--       omits the column gets the sentinel automatically.
--
--   (b) A partial-unique-index keyed on ``(experiment_id, feature,
--       written_at)`` with ``COALESCE(experiment_id, '__unknown__')`` /
--       ``COALESCE(feature, '__unknown__')`` so even a hypothetical NULL
--       (which (a) should already prevent) folds to the sentinel.
--
-- ``written_at`` is required and never imputed (the reader skips
-- sidecars with unparseable ``written_at``), so it stays NOT NULL with
-- no sentinel.
--
-- Sentinel string ``'__unknown__'`` is chosen distinct from the reader's
-- in-memory ``"<unknown>"`` so a row that landed because the producer
-- omitted the key is distinguishable from one where the value is the
-- literal text ``<unknown>``. The two-sentinel split is intentional.
--
-- ----------------------------------------------------------------------------
-- Index design.
-- ----------------------------------------------------------------------------
-- Two read-path indexes match the ``scripts/query_audit_trail.py`` queries:
--
--   - ``(feature, written_at DESC)`` for "trail per feature" and
--     "top-K features by recent disagreement rate".
--   - ``(experiment_id, written_at DESC)`` for "trail per experiment"
--     and the worker's ``max(imported_at)`` cursor query.
--
-- ----------------------------------------------------------------------------
-- Grants.
-- ----------------------------------------------------------------------------
-- Matches the Supabase standard pattern:
--   - ``service_role``: full DML (the nightly worker authenticates as this).
--   - ``authenticated``: SELECT only (dashboards / read-only consumers).
--   - ``anon`` is intentionally NOT granted — audit signals are internal.
--
-- ----------------------------------------------------------------------------
-- Transaction control.
-- ----------------------------------------------------------------------------
-- No script-level ``BEGIN;`` / ``COMMIT;``. ``scripts/run_migrations.sh``
-- invokes psql with ``--single-transaction`` and appends an INSERT INTO
-- schema_migrations after the file; an inner COMMIT would prematurely
-- commit before the bookkeeping insert. Matches migrations 038 and 039.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Table
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS adaptive_validity_verdicts (
    -- Natural-key columns. NOT NULL with sentinel DEFAULT prevents
    -- silent-NULL inserts that would defeat the unique index.
    experiment_id   TEXT        NOT NULL DEFAULT '__unknown__',
    feature         TEXT        NOT NULL DEFAULT '__unknown__',
    written_at      TIMESTAMPTZ NOT NULL,
    -- Provenance: filesystem path of the sidecar this row came from.
    -- Not part of the natural key — multiple runs that produced the
    -- same (experiment_id, feature, written_at) overwrite each other.
    source_path     TEXT        NOT NULL,
    -- The complete worker verdict dict, verbatim from the sidecar.
    -- Mirrors ``VerdictRecord.raw_verdict``.
    verdict         JSONB       NOT NULL,
    -- Optional evaluator audit subset (5 evaluator_* keys), surfaced
    -- separately so the consumer can SELECT WHERE evaluator_audit IS
    -- NOT NULL without parsing ``verdict``. NULL when the evaluator
    -- was disabled or pre-existed the audit-signal feature.
    evaluator_audit JSONB,
    -- Mirror-side bookkeeping. The nightly worker reads
    -- ``SELECT max(imported_at) FROM adaptive_validity_verdicts``
    -- to set its "process sidecars since" cursor.
    imported_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE adaptive_validity_verdicts IS
'Queryable mirror of adaptive_verdicts_*.json sidecars from the data-preparer audit trail. The on-disk JSON files (under $ADAPTIVE_VALIDITY_ARTIFACTS_DIR) remain canonical; this table is the cross-experiment query surface. Mirrored nightly by scripts/mirror_audit_sidecar_to_supabase.py. Issue #238.';

COMMENT ON COLUMN adaptive_validity_verdicts.experiment_id IS
'Sidecar payload.experiment_id (TEXT, sentinel ''__unknown__'' if missing). Part of the (experiment_id, feature, written_at) natural key.';

COMMENT ON COLUMN adaptive_validity_verdicts.feature IS
'Verdict-level ``feature`` field (TEXT, sentinel ''__unknown__'' if missing). Part of the natural key.';

COMMENT ON COLUMN adaptive_validity_verdicts.written_at IS
'Sidecar payload.written_at parsed as TIMESTAMPTZ. NEVER NULL — the reader skips sidecars with unparseable written_at, so any row that lands here has a real timestamp.';

COMMENT ON COLUMN adaptive_validity_verdicts.source_path IS
'Filesystem path of the sidecar JSON this row was imported from. Provenance only; not part of the natural key.';

COMMENT ON COLUMN adaptive_validity_verdicts.verdict IS
'Complete verdict dict (raw_verdict). Includes leakage detection fields (z_score, p_value, delta_auc) plus producer-side context. Worker upsert replaces this column on conflict.';

COMMENT ON COLUMN adaptive_validity_verdicts.evaluator_audit IS
'Optional subset of evaluator_* fields (satisfied, rationale_complete, missed_considerations, notes, model, plus telemetry from issue #241). NULL when the evaluator was disabled. SELECT WHERE evaluator_audit IS NOT NULL is the natural disagreement-feed query.';

COMMENT ON COLUMN adaptive_validity_verdicts.imported_at IS
'When this row was written by the mirror worker. The worker queries max(imported_at) as its "since" cursor on each invocation.';

-- ----------------------------------------------------------------------------
-- 2. Natural-key partial-unique-index (idempotent upsert key).
-- ----------------------------------------------------------------------------
-- COALESCE wraps experiment_id / feature in case a future caller inserts
-- a literal NULL (column-level NOT NULL + DEFAULT should already prevent
-- this; the COALESCE is belt-and-suspenders matching the PR #250 precedent
-- at `database/memory/021_insight_lifecycle.sql`).
CREATE UNIQUE INDEX IF NOT EXISTS uix_adaptive_validity_verdicts_natural_key
    ON adaptive_validity_verdicts (
        COALESCE(experiment_id, '__unknown__'),
        COALESCE(feature, '__unknown__'),
        written_at
    );

-- ----------------------------------------------------------------------------
-- 3. Read-path indexes.
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_adaptive_validity_verdicts_feature
    ON adaptive_validity_verdicts (feature, written_at DESC);

CREATE INDEX IF NOT EXISTS idx_adaptive_validity_verdicts_experiment
    ON adaptive_validity_verdicts (experiment_id, written_at DESC);

-- Read-path: the worker's "since cursor" query is bare
-- ``SELECT max(imported_at) FROM adaptive_validity_verdicts`` with no
-- additional predicates, which Postgres satisfies via the table-level
-- index-only scan whenever a btree on imported_at exists. The natural
-- key index doesn't cover ``imported_at``, so add a dedicated one.
CREATE INDEX IF NOT EXISTS idx_adaptive_validity_verdicts_imported_at
    ON adaptive_validity_verdicts (imported_at DESC);

-- ----------------------------------------------------------------------------
-- 4. Permissions.
-- ----------------------------------------------------------------------------
-- service_role: full DML (the nightly mirror worker writes here).
-- authenticated: SELECT only (dashboard / read-only API consumers).
-- anon: intentionally not granted; audit signals are internal-only.
GRANT SELECT, INSERT, UPDATE, DELETE ON adaptive_validity_verdicts TO service_role;
GRANT SELECT ON adaptive_validity_verdicts TO authenticated;

-- ----------------------------------------------------------------------------
-- 5. Row-level security.
-- ----------------------------------------------------------------------------
-- The table holds no per-tenant data (no brand / region scoping in
-- sidecars), so RLS is enabled with permissive policies that
-- approximate the GRANT shape. service_role has BYPASS RLS by default
-- in Supabase; ``authenticated`` reads everything. This matches the
-- pattern used by other "internal mirror" tables in this codebase.
ALTER TABLE adaptive_validity_verdicts ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    CREATE POLICY adaptive_validity_verdicts_read_authenticated
        ON adaptive_validity_verdicts
        FOR SELECT
        TO authenticated
        USING (true);
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- (No COMMIT; psql --single-transaction owns the outer txn. See header.)
