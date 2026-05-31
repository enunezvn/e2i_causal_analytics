-- ============================================================================
-- Migration 047: #577 PR1 (causal trio) — wire CM-003 (causal_impact) to REAL
-- data via the kpi_query allowlist. Registry INSERT ONLY — no schema or data
-- change.
-- ============================================================================
-- Issue #577 (follow-up to #574). CM-003 previously raised a fail-loud
-- RuntimeError ("causal_paths has no intervention_name column"). An adversarial
-- design-review established that no intervention_name column is needed: the
-- HONEST metric is a DESCRIPTIVE aggregate — the average strength of the
-- discovered causal effects already stored in causal_paths.causal_effect_size
-- (populated for all 50 paths, well-varied 0.051-0.395).
--
-- WHY THIS IS HONEST (and not the #574 relabel trap): the metric value is
-- AVG(causal_effect_size), which the metric name ("causal impact") truthfully
-- describes. start_node is the discovered path SOURCE (where a chain begins) and
-- is surfaced ONLY as a descriptive breakdown — it is deliberately NOT presented
-- as a do()-style intervention target (that WOULD be the relabel #574 forbids,
-- and is the reason the prior RuntimeError mentioned "intervention_name"). The
-- calculator carries this code-anchor in its metadata note.
--
-- Sibling parity: CM-001 (ATE) / CM-002 (CATE) — the already-accepted causal
-- metrics — aggregate the equally-synthetic treatment_effect_estimate /
-- heterogeneous_effect columns. CM-003 has the same standing.
--
-- NOTE: CM-004 (counterfactual) and CM-005 (mediation) are NOT wired here. Their
-- source columns are independent uniform noise (counterfactual_outcome has no
-- relationship to any factual outcome; causal_chain edges do not reconcile with
-- causal_effect_size), so wiring them honestly requires a generator-coherence
-- rework first — tracked as PR2/PR3 of the causal trio. They remain fail-loud.
--
-- deploy.yml SKIPS migrations; the local self-contained supabase is the faithful
-- target. Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/047_kpi_577_causal_impact.sql
-- ----------------------------------------------------------------------------

-- Register the read-only CM-003 statement (allowlist; executed only via kpi_query).
-- $1 = optional validation_status filter ('' => all discovered paths; e.g.
-- 'validated' => audited only). Returns one row per start_node so the calculator
-- can report both the path-level mean and a descriptive breakdown.
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('causal_metrics_causal_impact', $kpi$SELECT start_node, AVG(causal_effect_size)::float AS effect, COUNT(*)::int AS n_paths, AVG(confidence_level)::float AS avg_confidence FROM causal_paths WHERE causal_effect_size IS NOT NULL AND ($1 = '' OR validation_status::text = $1) GROUP BY start_node ORDER BY effect DESC$kpi$, 1, $note$CM-003 causal_impact: path-level mean causal_effect_size across DISCOVERED causal paths (a descriptive aggregate, NOT the effect of intervening). $1 = validation_status filter ('' = all paths; 'validated' = audited only). One row per start_node (the discovered path SOURCE, surfaced as a breakdown only — NOT an intervention target, #574). The calculator computes value = SUM(effect*n_paths)/SUM(n_paths) and returns NULL (fail-loud) when no paths exist.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_id is callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
