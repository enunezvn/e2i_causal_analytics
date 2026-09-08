-- ============================================================================
-- E2I Causal Analytics - Migration 036: move the causal-discovery tables
--                                        from schema ml into public (#1974)
-- ============================================================================
-- Date: 2026-09-08
-- Description: ml/026 created discovered_dags, discovery_algorithm_runs,
-- discovered_edges, driver_rankings and feature_rankings (+ 3 views, 3 enum
-- types, 3 functions and the updated_at trigger) in a dedicated ``ml`` schema.
-- It is the ONLY file under database/ that schema-qualifies ``ml.`` — every
-- other database/ml/*.sql creates its tables in public.
--
-- WHY MOVE (all measured 2026-09-08 on the live, self-contained prod Supabase):
--   * PostgREST exposes only ``public,storage,graphql_public``: a request on
--     the ``ml`` profile answers PGRST106 ("Invalid schema: ml"). No
--     Supabase-client repository could ever write these tables, so they have
--     held 0 rows since 026 was applied on 2026-06-04. Issue #1974's premise
--     ("migration 026 never applied") was measured FALSE — the blocker is the
--     schema exposure, not the migration.
--   * Migration 058 REVOKEd the anon/authenticated over-grant on public but
--     never reached ``ml``: ``authenticated`` still holds INSERT/SELECT/UPDATE
--     on all five tables while ``service_role`` (the backend's role) holds
--     NO table grant there at all. Moving into public and granting explicitly
--     fixes both in one place.
--   * All five tables hold 0 rows, so the move is free and lossless.
--   * In public, BaseRepository / HAS_PROVENANCE work directly; no
--     PGRST_DB_SCHEMAS change, no RLS policy and no SECURITY DEFINER is needed.
--
-- NAME CLASH (measured): ``public.gate_decision`` ALREADY EXISTS — migration
-- 010's refutation-gate enum (proceed / review / block). ``ml.gate_decision``
-- is a DIFFERENT enum (the discovery gate: accept / review / reject / augment),
-- so it is RENAMED to ``discovery_gate_decision`` before it moves. No other
-- name clashes exist in public for the tables, views, functions, indexes or
-- the two remaining types (verified against pg_class / pg_proc / pg_type /
-- pg_indexes on 2026-09-08).
--
-- WHAT MOVES BY OID (verified in the BEGIN/ROLLBACK rehearsal, not assumed):
--   * views reference their tables by OID -> pg_get_viewdef prints the new
--     schema after the move; the ALTER VIEW ... SET SCHEMA only relocates
--     the view object itself;
--   * the trigger is a table property and follows the table; its function
--     is referenced by OID, so ``ALTER FUNCTION ... SET SCHEMA`` is enough
--     (its body has no schema references);
--   * FK constraints, indexes and the enum-typed columns follow by OID (the
--     renamed enum is still the SAME type OID, so ``discovered_dags.gate_decision``
--     needs no ALTER COLUMN);
--   * the two READER functions do NOT: their plpgsql bodies name
--     ``ml.discovered_edges`` / ``ml.feature_rankings`` textually and plpgsql
--     resolves names at execution, so they are dropped in ml and recreated in
--     public pointing at public relations.
--
-- NEW COLUMNS on discovered_dags (the 026 shape had no place for them):
--   * is_synthetic  — the platform provenance convention (063/067/069;
--                     ADR-017 honest provenance). The writer must STATE it;
--                     record_discovered_dag() rejects a payload that omits it.
--   * dag_version_hash — the expert-review key (expert_reviews.dag_version_hash,
--                     VARCHAR(64) sha256), so a persisted DAG joins to its review.
--   * query_id       — the agent run id (agent-analyze's analysis_id / the
--                     orchestrator's query_id), so a DAG is findable from a run.
--   * treatment_variable / outcome_variable — the estimand the DAG served
--                     (mirrors expert_reviews' column names).
--
-- WRITER: public.record_discovered_dag(jsonb) inserts the DAG row + one
-- discovery_algorithm_runs row per algorithm + one discovered_edges row per
-- ensemble edge ATOMICALLY (one call, one transaction — no partial rows on
-- failure) and returns a receipt {dag_id, n_algorithm_runs, n_edges} the
-- caller verifies against what it sent. SECURITY INVOKER: it relies on the
-- explicit service_role grants below, never on definer privileges.
--
-- GRANTS: explicit ``service_role`` table grants (precedent:
-- public.expert_reviews) + ``REVOKE ALL ... FROM anon, authenticated`` (058's
-- posture). The DO block at the end ASSERTS the resulting grants and object
-- locations instead of trusting default privileges — 058 documents that the
-- supabase_admin default-privilege revoke is skipped when the applier is
-- postgres, so defaults differ by applier role.
--
-- NO ``DROP SCHEMA ml`` HERE: after this file the schema is empty on prod, but
-- (a) 026 still says ``CREATE SCHEMA IF NOT EXISTS ml`` and runs first on a
-- fresh database, (b) other environments may hold objects there, and (c)
-- dropping a schema is destructive and out of this migration's scope. Only
-- the ``authenticated`` USAGE over-grant on it is revoked.
--
-- POSTGREST SCHEMA CACHE: this database has the ``pgrst_ddl_watch`` event
-- trigger (ddl_command_end -> NOTIFY pgrst, 'reload schema'; verified
-- ``select evtname from pg_event_trigger`` 2026-09-08), so every ALTER/CREATE
-- below already triggers a reload and no NOTIFY is strictly needed here. The
-- conventional NOTIFY at the end is kept for an environment without that
-- trigger (55 sibling migrations carry it) — belt and braces, not a
-- dependency.
--
-- IDEMPOTENT: every move is guarded on the object still living in ml; column
-- adds are IF NOT EXISTS; CREATE OR REPLACE / GRANT / REVOKE / COMMENT are
-- re-runnable. A manual out-of-band apply bypasses run_migrations.sh's ledger
-- (public.schema_migrations), so the next deploy re-runs this file — the
-- rehearsal applies it TWICE in one transaction to prove that is a no-op.
--
-- NO SCRIPT-LEVEL BEGIN/COMMIT: scripts/run_migrations.sh wraps the file in
-- ``psql --single-transaction``. For a manual apply:
--   docker exec -i supabase-db psql -U postgres -d postgres \
--       -v ON_ERROR_STOP=1 --single-transaction < database/ml/036_move_discovery_tables_to_public.sql
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1) ENUM TYPES: rename the clashing one, then move all three
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    -- Refuse to proceed if the target name is already taken by something that
    -- is not our renamed type (a partial earlier run leaves ml.discovery_gate_decision,
    -- which the second guard handles; anything else is a real conflict).
    IF EXISTS (
        SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE n.nspname = 'ml' AND t.typname = 'gate_decision'
    ) AND EXISTS (
        SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE n.nspname = 'public' AND t.typname = 'discovery_gate_decision'
    ) THEN
        RAISE EXCEPTION 'migration 036: public.discovery_gate_decision already exists while ml.gate_decision is still present — resolve manually';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE n.nspname = 'ml' AND t.typname = 'gate_decision'
    ) THEN
        ALTER TYPE ml.gate_decision RENAME TO discovery_gate_decision;
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE n.nspname = 'ml' AND t.typname = 'discovery_gate_decision'
    ) THEN
        ALTER TYPE ml.discovery_gate_decision SET SCHEMA public;
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE n.nspname = 'ml' AND t.typname = 'discovery_algorithm'
    ) THEN
        ALTER TYPE ml.discovery_algorithm SET SCHEMA public;
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE n.nspname = 'ml' AND t.typname = 'edge_type'
    ) THEN
        ALTER TYPE ml.edge_type SET SCHEMA public;
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 2) TABLES (FKs, indexes, the trigger and enum-typed columns follow by OID)
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF to_regclass('ml.discovered_dags') IS NOT NULL THEN
        ALTER TABLE ml.discovered_dags SET SCHEMA public;
    END IF;
    IF to_regclass('ml.discovery_algorithm_runs') IS NOT NULL THEN
        ALTER TABLE ml.discovery_algorithm_runs SET SCHEMA public;
    END IF;
    IF to_regclass('ml.discovered_edges') IS NOT NULL THEN
        ALTER TABLE ml.discovered_edges SET SCHEMA public;
    END IF;
    IF to_regclass('ml.driver_rankings') IS NOT NULL THEN
        ALTER TABLE ml.driver_rankings SET SCHEMA public;
    END IF;
    IF to_regclass('ml.feature_rankings') IS NOT NULL THEN
        ALTER TABLE ml.feature_rankings SET SCHEMA public;
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 3) VIEWS (definitions reference the tables by OID and survive the move)
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF to_regclass('ml.v_recent_discoveries') IS NOT NULL THEN
        ALTER VIEW ml.v_recent_discoveries SET SCHEMA public;
    END IF;
    IF to_regclass('ml.v_high_confidence_edges') IS NOT NULL THEN
        ALTER VIEW ml.v_high_confidence_edges SET SCHEMA public;
    END IF;
    IF to_regclass('ml.v_discordant_features') IS NOT NULL THEN
        ALTER VIEW ml.v_discordant_features SET SCHEMA public;
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 4) FUNCTIONS
-- ----------------------------------------------------------------------------
-- 4a) The trigger function has no schema references: move it. The trigger on
--     discovered_dags references it by OID and keeps firing.
DO $$
BEGIN
    IF to_regprocedure('ml.update_discovered_dags_timestamp()') IS NOT NULL THEN
        ALTER FUNCTION ml.update_discovered_dags_timestamp() SET SCHEMA public;
    END IF;
END $$;

-- 4b) The readers name ml.* relations in their plpgsql bodies (resolved at
--     execution): drop the ml versions and recreate them in public, definitions
--     otherwise unchanged from 026.
DROP FUNCTION IF EXISTS ml.get_dag_edges(uuid, double precision);
CREATE OR REPLACE FUNCTION public.get_dag_edges(
    p_dag_id UUID,
    p_min_confidence FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    source_node VARCHAR(255),
    target_node VARCHAR(255),
    edge_type public.edge_type,
    confidence FLOAT,
    algorithm_votes INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.source_node,
        e.target_node,
        e.edge_type,
        e.confidence,
        e.algorithm_votes
    FROM public.discovered_edges e
    WHERE e.dag_id = p_dag_id
      AND e.confidence >= p_min_confidence
    ORDER BY e.confidence DESC;
END;
$$ LANGUAGE plpgsql;

DROP FUNCTION IF EXISTS ml.get_feature_ranking_comparison(uuid);
CREATE OR REPLACE FUNCTION public.get_feature_ranking_comparison(
    p_ranking_id UUID
)
RETURNS TABLE (
    feature_name VARCHAR(255),
    causal_rank INTEGER,
    predictive_rank INTEGER,
    rank_difference INTEGER,
    causal_score FLOAT,
    predictive_score FLOAT,
    classification TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fr.feature_name,
        fr.causal_rank,
        fr.predictive_rank,
        fr.rank_difference,
        fr.causal_score,
        fr.predictive_score,
        CASE
            WHEN ABS(fr.rank_difference) <= 2 THEN 'concordant'
            WHEN fr.rank_difference > 2 THEN 'predictive_overweighted'
            ELSE 'causal_overweighted'
        END as classification
    FROM public.feature_rankings fr
    WHERE fr.ranking_id = p_ranking_id
    ORDER BY fr.causal_rank;
END;
$$ LANGUAGE plpgsql;

-- ----------------------------------------------------------------------------
-- 5) NEW COLUMNS on discovered_dags
-- ----------------------------------------------------------------------------
-- Provenance (platform convention: 063/067/069, ml/031). Synthetic rows set
-- TRUE; BaseRepository readers with HAS_PROVENANCE default-exclude them.
ALTER TABLE public.discovered_dags ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
COMMENT ON COLUMN public.discovered_dags.is_synthetic IS
    'TRUE = the DAG was discovered from data the read path admitted as synthetic '
    '(the frame carried is_synthetic rows, the caller declared data_source=synthetic, '
    'or the deployment runs with E2I_INCLUDE_SYNTHETIC so provenance-unfiltered reads '
    'include the synthetic showcase substrate). Excluded by default from real-mode '
    'reads (BaseRepository.HAS_PROVENANCE). The writer must STATE it: '
    'record_discovered_dag() rejects a payload without it. Added by ml/036 (#1974).';
-- Partial index over the synthetic minority (mirrors 063's pattern).
CREATE INDEX IF NOT EXISTS idx_discovered_dags_is_synthetic
    ON public.discovered_dags (is_synthetic) WHERE is_synthetic;

-- The expert-review key: expert_reviews.dag_version_hash is VARCHAR(64)
-- (compute_dag_hash -> sha256 hex). Nullable: a DAG persisted by a future
-- non-agent writer may have no hash; the agent writer always sets it.
ALTER TABLE public.discovered_dags ADD COLUMN IF NOT EXISTS dag_version_hash VARCHAR(64);
COMMENT ON COLUMN public.discovered_dags.dag_version_hash IS
    'SHA256 of the SHIPPED DAG (compute_dag_hash) — the same key expert_reviews '
    'uses, so a discovered DAG joins to its review row. Note the shipped DAG can '
    'differ from the discovered edge_list (gate REVIEW/REJECT ship the manual DAG; '
    'AUGMENT adds edges) — metadata.shipped_dag records exactly what was hashed. '
    'Added by ml/036 (#1974).';
CREATE INDEX IF NOT EXISTS idx_discovered_dags_dag_version_hash
    ON public.discovered_dags (dag_version_hash);

-- The agent run this DAG belongs to (agent-analyze analysis_id / orchestrator
-- query_id). session_id stays what 026 intended (a UUID session); the API
-- path does not set one, so query_id is the reliable run key.
ALTER TABLE public.discovered_dags ADD COLUMN IF NOT EXISTS query_id TEXT;
COMMENT ON COLUMN public.discovered_dags.query_id IS
    'Run identifier of the causal_impact run that produced this DAG '
    '(POST /causal/agent-analyze analysis_id, or the orchestrator query_id). '
    'Added by ml/036 (#1974).';
CREATE INDEX IF NOT EXISTS idx_discovered_dags_query_id
    ON public.discovered_dags (query_id);

-- The estimand the DAG served (column names mirror expert_reviews).
ALTER TABLE public.discovered_dags ADD COLUMN IF NOT EXISTS treatment_variable VARCHAR(255);
ALTER TABLE public.discovered_dags ADD COLUMN IF NOT EXISTS outcome_variable VARCHAR(255);
COMMENT ON COLUMN public.discovered_dags.treatment_variable IS
    'Treatment variable of the causal question this discovery served. Added by ml/036 (#1974).';
COMMENT ON COLUMN public.discovered_dags.outcome_variable IS
    'Outcome variable of the causal question this discovery served. Added by ml/036 (#1974).';

-- 5b) Expose provenance on the two DAG-derived views (after the column adds) (ml/031 precedent:
--     "expose is_synthetic on ml_model_health_dashboard"). BaseRepository's
--     HAS_PROVENANCE governs Python readers only; an operator querying the
--     views directly needs the column to filter on. CREATE OR REPLACE VIEW
--     may only APPEND columns, so the 026 column list is kept verbatim and the
--     new columns (all functionally dependent on d.id, the GROUP BY key) come
--     last. v_discordant_features is ranking-derived and untouched
--     (driver_rankings carries no provenance column).
CREATE OR REPLACE VIEW public.v_recent_discoveries AS
SELECT
    d.id,
    d.session_id,
    d.discovery_timestamp,
    d.n_samples,
    d.n_features,
    d.n_edges,
    d.gate_decision,
    d.gate_confidence,
    d.total_runtime_seconds,
    d.algorithms_used,
    d.ensemble_threshold,
    COUNT(DISTINCT ar.id) as n_algorithm_runs,
    AVG(ar.runtime_seconds) as avg_algorithm_runtime,
    SUM(CASE WHEN ar.converged THEN 1 ELSE 0 END) as n_converged,
    d.is_synthetic,
    d.dag_version_hash,
    d.query_id,
    d.treatment_variable,
    d.outcome_variable
FROM public.discovered_dags d
LEFT JOIN public.discovery_algorithm_runs ar ON ar.dag_id = d.id
GROUP BY d.id
ORDER BY d.created_at DESC;

CREATE OR REPLACE VIEW public.v_high_confidence_edges AS
SELECT
    e.id,
    e.dag_id,
    e.source_node,
    e.target_node,
    e.edge_type,
    e.confidence,
    e.algorithm_votes,
    e.algorithms,
    d.gate_decision,
    d.session_id,
    d.is_synthetic
FROM public.discovered_edges e
JOIN public.discovered_dags d ON d.id = e.dag_id
WHERE e.confidence >= 0.8
ORDER BY e.confidence DESC;

-- ----------------------------------------------------------------------------
-- 6) THE ATOMIC WRITER RPC
-- ----------------------------------------------------------------------------
-- Payload contract (built by src/repositories/discovered_dag.py
-- build_discovered_dag_payload; every key maps to a column):
--   {
--     session_id, query_id, dag_version_hash, treatment_variable, outcome_variable,
--     discovery_timestamp, n_samples, n_features, feature_names,
--     config, algorithms_used, ensemble_threshold, alpha,
--     n_edges, n_nodes, edge_list, confidence_scores, adjacency_matrix,
--     gate_decision, gate_confidence, gate_reasons,
--     total_runtime_seconds, metadata, is_synthetic,
--     algorithm_runs: [{algorithm, runtime_seconds, converged, n_edges, edge_list,
--                       adjacency_matrix, score, parameters, metadata}, ...],
--     edges: [{source_node, target_node, edge_type, confidence, algorithm_votes,
--              algorithms, metadata}, ...]
--   }
-- NOT NULL columns are inserted WITHOUT COALESCE on purpose: a payload that
-- omits n_samples / n_features / a run's runtime_seconds fails the insert
-- visibly rather than being filled with a plausible default.
CREATE OR REPLACE FUNCTION public.record_discovered_dag(p_payload jsonb)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $$
DECLARE
    v_dag_id uuid;
    v_n_runs integer := 0;
    v_n_edges integer := 0;
BEGIN
    IF p_payload IS NULL OR jsonb_typeof(p_payload) <> 'object' THEN
        RAISE EXCEPTION 'record_discovered_dag: payload must be a JSON object';
    END IF;
    -- ADR-017 honest provenance: never default is_synthetic silently.
    IF p_payload->>'is_synthetic' IS NULL THEN
        RAISE EXCEPTION 'record_discovered_dag: is_synthetic must be stated explicitly in the payload';
    END IF;

    INSERT INTO public.discovered_dags (
        session_id, query_id, dag_version_hash, treatment_variable, outcome_variable,
        discovery_timestamp, n_samples, n_features, feature_names,
        config, algorithms_used, ensemble_threshold, alpha,
        n_edges, n_nodes, edge_list, confidence_scores, adjacency_matrix,
        gate_decision, gate_confidence, gate_reasons,
        total_runtime_seconds, metadata, is_synthetic
    ) VALUES (
        (p_payload->>'session_id')::uuid,
        p_payload->>'query_id',
        p_payload->>'dag_version_hash',
        p_payload->>'treatment_variable',
        p_payload->>'outcome_variable',
        COALESCE((p_payload->>'discovery_timestamp')::timestamptz, NOW()),
        (p_payload->>'n_samples')::integer,
        (p_payload->>'n_features')::integer,
        COALESCE(NULLIF(p_payload->'feature_names', 'null'::jsonb), '[]'::jsonb),
        COALESCE(NULLIF(p_payload->'config', 'null'::jsonb), '{}'::jsonb),
        COALESCE(
            ARRAY(SELECT jsonb_array_elements_text(NULLIF(p_payload->'algorithms_used', 'null'::jsonb))),
            '{}'::text[]
        ),
        (p_payload->>'ensemble_threshold')::float,
        (p_payload->>'alpha')::float,
        (p_payload->>'n_edges')::integer,
        (p_payload->>'n_nodes')::integer,
        COALESCE(NULLIF(p_payload->'edge_list', 'null'::jsonb), '[]'::jsonb),
        COALESCE(NULLIF(p_payload->'confidence_scores', 'null'::jsonb), '{}'::jsonb),
        NULLIF(p_payload->'adjacency_matrix', 'null'::jsonb),
        (p_payload->>'gate_decision')::public.discovery_gate_decision,
        (p_payload->>'gate_confidence')::float,
        COALESCE(NULLIF(p_payload->'gate_reasons', 'null'::jsonb), '[]'::jsonb),
        (p_payload->>'total_runtime_seconds')::float,
        COALESCE(NULLIF(p_payload->'metadata', 'null'::jsonb), '{}'::jsonb),
        (p_payload->>'is_synthetic')::boolean
    )
    RETURNING id INTO v_dag_id;

    INSERT INTO public.discovery_algorithm_runs (
        dag_id, algorithm, runtime_seconds, converged, n_edges, edge_list,
        adjacency_matrix, score, parameters, metadata
    )
    SELECT
        v_dag_id,
        (r->>'algorithm')::public.discovery_algorithm,
        (r->>'runtime_seconds')::float,
        (r->>'converged')::boolean,
        (r->>'n_edges')::integer,
        COALESCE(NULLIF(r->'edge_list', 'null'::jsonb), '[]'::jsonb),
        NULLIF(r->'adjacency_matrix', 'null'::jsonb),
        (r->>'score')::float,
        COALESCE(NULLIF(r->'parameters', 'null'::jsonb), '{}'::jsonb),
        COALESCE(NULLIF(r->'metadata', 'null'::jsonb), '{}'::jsonb)
    FROM jsonb_array_elements(COALESCE(NULLIF(p_payload->'algorithm_runs', 'null'::jsonb), '[]'::jsonb)) AS r;
    GET DIAGNOSTICS v_n_runs = ROW_COUNT;

    INSERT INTO public.discovered_edges (
        dag_id, source_node, target_node, edge_type, confidence, algorithm_votes,
        algorithms, metadata
    )
    SELECT
        v_dag_id,
        e->>'source_node',
        e->>'target_node',
        (e->>'edge_type')::public.edge_type,
        (e->>'confidence')::float,
        (e->>'algorithm_votes')::integer,
        COALESCE(
            ARRAY(SELECT jsonb_array_elements_text(NULLIF(e->'algorithms', 'null'::jsonb))),
            '{}'::text[]
        ),
        COALESCE(NULLIF(e->'metadata', 'null'::jsonb), '{}'::jsonb)
    FROM jsonb_array_elements(COALESCE(NULLIF(p_payload->'edges', 'null'::jsonb), '[]'::jsonb)) AS e;
    GET DIAGNOSTICS v_n_edges = ROW_COUNT;

    RETURN jsonb_build_object(
        'dag_id', v_dag_id,
        'n_algorithm_runs', v_n_runs,
        'n_edges', v_n_edges
    );
END;
$$;

COMMENT ON FUNCTION public.record_discovered_dag(jsonb) IS
    'Atomic writer for a causal-discovery run: inserts the discovered_dags row, one '
    'discovery_algorithm_runs row per algorithm and one discovered_edges row per '
    'ensemble edge in a single call, returning {dag_id, n_algorithm_runs, n_edges}. '
    'SECURITY INVOKER, service_role only. Called by DiscoveredDagRepository '
    '(src/repositories/discovered_dag.py) from the causal_impact graph_builder node. '
    'Added by ml/036 (#1974).';

-- ----------------------------------------------------------------------------
-- 7) GRANTS: explicit service_role access, 058's anon/authenticated posture
-- ----------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON public.discovered_dags TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON public.discovery_algorithm_runs TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON public.discovered_edges TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON public.driver_rankings TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON public.feature_rankings TO service_role;
GRANT SELECT ON public.v_recent_discoveries TO service_role;
GRANT SELECT ON public.v_high_confidence_edges TO service_role;
GRANT SELECT ON public.v_discordant_features TO service_role;

REVOKE ALL ON public.discovered_dags FROM anon, authenticated;
REVOKE ALL ON public.discovery_algorithm_runs FROM anon, authenticated;
REVOKE ALL ON public.discovered_edges FROM anon, authenticated;
REVOKE ALL ON public.driver_rankings FROM anon, authenticated;
REVOKE ALL ON public.feature_rankings FROM anon, authenticated;
REVOKE ALL ON public.v_recent_discoveries FROM anon, authenticated;
REVOKE ALL ON public.v_high_confidence_edges FROM anon, authenticated;
REVOKE ALL ON public.v_discordant_features FROM anon, authenticated;

-- Functions: the postgres default ACL in public grants EXECUTE to
-- anon/authenticated on every new function (pg_default_acl, measured), so
-- lock the writer and the two readers to service_role explicitly. The trigger
-- function keeps its default EXECUTE — it runs as a side effect of UPDATE and
-- is not a callable surface.
REVOKE ALL ON FUNCTION public.record_discovered_dag(jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.record_discovered_dag(jsonb) TO service_role;
REVOKE ALL ON FUNCTION public.get_dag_edges(uuid, double precision) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_dag_edges(uuid, double precision) TO service_role;
REVOKE ALL ON FUNCTION public.get_feature_ranking_comparison(uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_feature_ranking_comparison(uuid) TO service_role;

-- 026 granted USAGE on the (now empty) ml schema to authenticated — the
-- over-grant 058 never reached. Remove it; service_role USAGE is harmless
-- and left for 026's own idempotent re-apply on a fresh database.
REVOKE ALL ON SCHEMA ml FROM authenticated;

-- ----------------------------------------------------------------------------
-- 8) COMMENTS (026's, re-pointed at public)
-- ----------------------------------------------------------------------------
COMMENT ON TABLE public.discovered_dags IS
    'Discovered causal DAG structures from structure learning (moved from ml by ml/036, #1974). '
    'Written by DiscoveredDagRepository from the causal_impact graph_builder node on every run '
    'where auto-discovery actually ran; edge_list/discovered_edges are the ENSEMBLE edges, '
    'metadata.shipped_dag is the DAG that shipped after the gate (hashed as dag_version_hash).';
COMMENT ON TABLE public.discovery_algorithm_runs IS
    'Individual algorithm run results within a discovery (moved from ml by ml/036).';
COMMENT ON TABLE public.discovered_edges IS
    'Edges in discovered DAGs with confidence metadata (moved from ml by ml/036).';
COMMENT ON TABLE public.driver_rankings IS
    'Causal vs predictive feature importance rankings (moved from ml by ml/036). '
    'Still writer-less: DriverRanker runs in the feature_analyzer agent (causal_ranker '
    'node) and in the tool-registry rank_drivers tool, neither of which persists here '
    'yet; it does not run on the causal_impact path that writes discovered_dags.';
COMMENT ON TABLE public.feature_rankings IS
    'Detailed per-feature ranking information (moved from ml by ml/036). See driver_rankings.';
COMMENT ON VIEW public.v_recent_discoveries IS 'Summary view of recent causal discovery runs';
COMMENT ON VIEW public.v_high_confidence_edges IS 'Edges with confidence >= 0.8';
COMMENT ON VIEW public.v_discordant_features IS
    'Features with large rank differences between causal and predictive';

-- ----------------------------------------------------------------------------
-- 9) ASSERT the outcome — locations, types, trigger, grants, validity
-- ----------------------------------------------------------------------------
DO $$
DECLARE
    v_rel text;
    v_fn text;
    v_labels text[];
    v_count integer;
BEGIN
    -- 9a) Every relation lives in public and none remains in ml.
    FOREACH v_rel IN ARRAY ARRAY[
        'discovered_dags', 'discovery_algorithm_runs', 'discovered_edges',
        'driver_rankings', 'feature_rankings',
        'v_recent_discoveries', 'v_high_confidence_edges', 'v_discordant_features'
    ] LOOP
        IF to_regclass('public.' || v_rel) IS NULL THEN
            RAISE EXCEPTION 'migration 036: public.% is missing after the move', v_rel;
        END IF;
        IF to_regclass('ml.' || v_rel) IS NOT NULL THEN
            RAISE EXCEPTION 'migration 036: ml.% still exists after the move', v_rel;
        END IF;
    END LOOP;

    -- 9b) Types: the renamed discovery enum with 026's labels; the pre-existing
    --     refutation enum untouched; nothing left in ml.
    SELECT array_agg(e.enumlabel::text ORDER BY e.enumsortorder) INTO v_labels
      FROM pg_type t
      JOIN pg_namespace n ON n.oid = t.typnamespace
      JOIN pg_enum e ON e.enumtypid = t.oid
     WHERE n.nspname = 'public' AND t.typname = 'discovery_gate_decision';
    IF v_labels IS DISTINCT FROM ARRAY['accept', 'review', 'reject', 'augment'] THEN
        RAISE EXCEPTION 'migration 036: public.discovery_gate_decision labels are % (expected accept/review/reject/augment)', v_labels;
    END IF;
    SELECT array_agg(e.enumlabel::text ORDER BY e.enumsortorder) INTO v_labels
      FROM pg_type t
      JOIN pg_namespace n ON n.oid = t.typnamespace
      JOIN pg_enum e ON e.enumtypid = t.oid
     WHERE n.nspname = 'public' AND t.typname = 'gate_decision';
    IF v_labels IS DISTINCT FROM ARRAY['proceed', 'review', 'block'] THEN
        RAISE EXCEPTION 'migration 036: public.gate_decision (migration 010) was disturbed: %', v_labels;
    END IF;
    SELECT array_agg(e.enumlabel::text ORDER BY e.enumsortorder) INTO v_labels
      FROM pg_type t
      JOIN pg_namespace n ON n.oid = t.typnamespace
      JOIN pg_enum e ON e.enumtypid = t.oid
     WHERE n.nspname = 'public' AND t.typname = 'edge_type';
    IF v_labels IS DISTINCT FROM ARRAY['directed', 'undirected', 'bidirected'] THEN
        RAISE EXCEPTION 'migration 036: public.edge_type labels are %', v_labels;
    END IF;
    SELECT array_agg(e.enumlabel::text ORDER BY e.enumsortorder) INTO v_labels
      FROM pg_type t
      JOIN pg_namespace n ON n.oid = t.typnamespace
      JOIN pg_enum e ON e.enumtypid = t.oid
     WHERE n.nspname = 'public' AND t.typname = 'discovery_algorithm';
    IF v_labels IS DISTINCT FROM ARRAY['ges', 'pc', 'fci', 'lingam', 'direct_lingam', 'ica_lingam'] THEN
        RAISE EXCEPTION 'migration 036: public.discovery_algorithm labels are %', v_labels;
    END IF;
    SELECT count(*) INTO v_count
      FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
     WHERE n.nspname = 'ml'
       AND t.typname IN ('gate_decision', 'discovery_gate_decision', 'discovery_algorithm', 'edge_type');
    IF v_count <> 0 THEN
        RAISE EXCEPTION 'migration 036: % discovery enum type(s) still in schema ml', v_count;
    END IF;
    -- The enum-typed column followed its type by OID.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'discovered_dags'
           AND column_name = 'gate_decision' AND udt_name = 'discovery_gate_decision'
    ) THEN
        RAISE EXCEPTION 'migration 036: discovered_dags.gate_decision is not typed public.discovery_gate_decision';
    END IF;

    -- 9c) Functions in public, none left in ml; trigger still wired.
    FOREACH v_fn IN ARRAY ARRAY[
        'public.record_discovered_dag(jsonb)',
        'public.get_dag_edges(uuid, double precision)',
        'public.get_feature_ranking_comparison(uuid)',
        'public.update_discovered_dags_timestamp()'
    ] LOOP
        IF to_regprocedure(v_fn) IS NULL THEN
            RAISE EXCEPTION 'migration 036: function % is missing', v_fn;
        END IF;
    END LOOP;
    SELECT count(*) INTO v_count
      FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
     WHERE n.nspname = 'ml'
       AND p.proname IN ('get_dag_edges', 'get_feature_ranking_comparison', 'update_discovered_dags_timestamp');
    IF v_count <> 0 THEN
        RAISE EXCEPTION 'migration 036: % discovery function(s) still in schema ml', v_count;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
         WHERE tgname = 'trigger_update_discovered_dags_timestamp'
           AND tgrelid = 'public.discovered_dags'::regclass
           AND tgfoid = 'public.update_discovered_dags_timestamp()'::regprocedure
    ) THEN
        RAISE EXCEPTION 'migration 036: updated_at trigger is not wired to public.discovered_dags / public.update_discovered_dags_timestamp()';
    END IF;

    -- 9d) New columns present with the intended nullability/default.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'discovered_dags'
           AND column_name = 'is_synthetic' AND is_nullable = 'NO'
           AND column_default = 'false'
    ) THEN
        RAISE EXCEPTION 'migration 036: discovered_dags.is_synthetic is not BOOLEAN NOT NULL DEFAULT false';
    END IF;
    FOREACH v_rel IN ARRAY ARRAY['dag_version_hash', 'query_id', 'treatment_variable', 'outcome_variable'] LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = 'discovered_dags' AND column_name = v_rel
        ) THEN
            RAISE EXCEPTION 'migration 036: discovered_dags.% is missing', v_rel;
        END IF;
    END LOOP;

    -- 9e) Grants: service_role can write the tables and read the views;
    --     anon/authenticated can do NOTHING on any of the eight relations.
    FOREACH v_rel IN ARRAY ARRAY[
        'discovered_dags', 'discovery_algorithm_runs', 'discovered_edges',
        'driver_rankings', 'feature_rankings'
    ] LOOP
        IF NOT (
            has_table_privilege('service_role', 'public.' || v_rel, 'SELECT')
            AND has_table_privilege('service_role', 'public.' || v_rel, 'INSERT')
            AND has_table_privilege('service_role', 'public.' || v_rel, 'UPDATE')
            AND has_table_privilege('service_role', 'public.' || v_rel, 'DELETE')
        ) THEN
            RAISE EXCEPTION 'migration 036: service_role lacks SELECT/INSERT/UPDATE/DELETE on public.%', v_rel;
        END IF;
    END LOOP;
    FOREACH v_rel IN ARRAY ARRAY[
        'v_recent_discoveries', 'v_high_confidence_edges', 'v_discordant_features'
    ] LOOP
        IF NOT has_table_privilege('service_role', 'public.' || v_rel, 'SELECT') THEN
            RAISE EXCEPTION 'migration 036: service_role lacks SELECT on public.%', v_rel;
        END IF;
    END LOOP;
    FOREACH v_rel IN ARRAY ARRAY[
        'discovered_dags', 'discovery_algorithm_runs', 'discovered_edges',
        'driver_rankings', 'feature_rankings',
        'v_recent_discoveries', 'v_high_confidence_edges', 'v_discordant_features'
    ] LOOP
        IF has_table_privilege('anon', 'public.' || v_rel, 'SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER')
        THEN
            RAISE EXCEPTION 'migration 036: anon still holds a privilege on public.%', v_rel;
        END IF;
        IF has_table_privilege('authenticated', 'public.' || v_rel, 'SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER')
        THEN
            RAISE EXCEPTION 'migration 036: authenticated still holds a privilege on public.%', v_rel;
        END IF;
    END LOOP;
    FOREACH v_fn IN ARRAY ARRAY[
        'public.record_discovered_dag(jsonb)',
        'public.get_dag_edges(uuid, double precision)',
        'public.get_feature_ranking_comparison(uuid)'
    ] LOOP
        IF NOT has_function_privilege('service_role', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 036: service_role cannot EXECUTE %', v_fn;
        END IF;
        IF has_function_privilege('anon', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 036: anon can still EXECUTE %', v_fn;
        END IF;
        IF has_function_privilege('authenticated', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 036: authenticated can still EXECUTE %', v_fn;
        END IF;
    END LOOP;
    IF has_schema_privilege('authenticated', 'ml', 'USAGE') THEN
        RAISE EXCEPTION 'migration 036: authenticated still has USAGE on schema ml';
    END IF;

    -- 9e') Provenance exposed on the DAG-derived views (5b).
    FOREACH v_rel IN ARRAY ARRAY['v_recent_discoveries', 'v_high_confidence_edges'] LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = v_rel AND column_name = 'is_synthetic'
        ) THEN
            RAISE EXCEPTION 'migration 036: public.% does not expose is_synthetic', v_rel;
        END IF;
    END LOOP;

    -- 9f) Views and readers are VALID after the move (executing them is the
    --     proof; an OID-dangling definition would error here, not at 3 a.m.).
    PERFORM 1 FROM public.v_recent_discoveries LIMIT 0;
    PERFORM 1 FROM public.v_high_confidence_edges LIMIT 0;
    PERFORM 1 FROM public.v_discordant_features LIMIT 0;
    PERFORM 1 FROM public.get_dag_edges(gen_random_uuid(), 0.0);
    PERFORM 1 FROM public.get_feature_ranking_comparison(gen_random_uuid());

    RAISE NOTICE 'migration 036: discovery tables verified in public with service_role-only grants';
END $$;

-- See the header: pgrst_ddl_watch already reloads on the DDL above; kept for
-- environments without that event trigger.
NOTIFY pgrst, 'reload schema';

-- ============================================================================
-- MIGRATION COMPLETE
-- Rollback (only meaningful while the tables are still empty): ALTER TABLE /
-- VIEW / FUNCTION / TYPE ... SET SCHEMA ml in reverse, rename
-- discovery_gate_decision back to gate_decision inside ml, DROP the five
-- ml/036 columns and record_discovered_dag(jsonb), then re-run 026's grants.
-- ============================================================================
