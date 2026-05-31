-- ============================================================================
-- Migration 049: #577 PR3 (causal trio) — wire CM-005 (mediation_effect) by adding
-- a COHERENT direct/indirect decomposition to causal_paths, then registering the
-- read-only allowlist statement.
-- ============================================================================
-- Issue #577 (follow-up to #574). CM-005 previously raised a fail-loud RuntimeError
-- ("causal_paths lacks direct/indirect/total_effect decomposition"). An adversarial
-- design-review rejected the original idea (seeding indirect = causal_effect_size *
-- f(mediator_count)) as FABRICATION: mediation is not determined by mediator COUNT, and
-- that formula is a constant-in-path-shape, not a measured fraction. It also found the
-- existing causal_chain.edges (uniform 0.1-0.5) do NOT reconcile with causal_effect_size
-- (uniform 0.05-0.4) — they are independent draws.
--
-- THE HONEST FIX (user-approved coherent rework): derive the decomposition from the REAL
-- causal_chain edge MAGNITUDES (the textbook serial-mediation determinant), not the count.
--   * total            = causal_effect_size (UNCHANGED — preserves CM-003).
--   * indirect channel = the serial path coefficient through the mediators = the PRODUCT
--                        of the causal_chain edge effects (exp(sum(ln(effect)))). For paths
--                        with no mediators (k=0) the indirect channel is 0.
--   * direct channel   = a synthesized X->Y magnitude (the direct path bypassing the
--                        mediators does not exist in the source, so it is generated;
--                        deterministic via hashtext here, random in the generator).
--   * proportion p     = indirect_channel / (indirect_channel + direct_channel) — a
--                        continuous value in [0,1) GROUNDED in the edge magnitudes (NOT a
--                        count formula, NOT a constant).
--   * indirect_effect  = round(total * p, 4);  direct_effect = round(total - indirect, 4)
--                        => direct + indirect = total exactly (the CHECK).
-- proportion mediated = indirect/total then varies by path (live: k=0 -> 0, k=1 ~ 0.32,
-- k=2 ~ 0.10, k=3 ~ 0.03; overall ~0.13) — a real distribution, decreasing with chain
-- length (serial attenuation), with k=0 contributing 0 (no mediation channel).
--
-- WHY SURGICAL (not truncate+reload): this ADDs two nullable columns and UPDATEs them in
-- place (every existing column/row preserved). Re-runnable: ADD COLUMN IF NOT EXISTS, the
-- UPDATE recomputes deterministically from causal_chain + hashtext, and the CHECK is
-- dropped-then-added. Snapshot taken pre-apply (/tmp/577_safety/causal_paths_pre049.sql).
-- The generator (src/ml/data_generator.py) makes the same edit so a fresh regenerate stays
-- coherent. CM-005 reports proportion mediated; direct_effect is the X->Y effect NOT
-- flowing through the identified mediators (the direct path / residual).
--
-- deploy.yml SKIPS migrations; the local self-contained supabase is the faithful target.
-- Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/049_kpi_577_mediation.sql
-- ----------------------------------------------------------------------------

-- (A) Add the decomposition columns (nullable; numeric(6,4) = exact scale-4 store).
ALTER TABLE public.causal_paths ADD COLUMN IF NOT EXISTS direct_effect numeric(6, 4);
ALTER TABLE public.causal_paths ADD COLUMN IF NOT EXISTS indirect_effect numeric(6, 4);

-- (B) Reseed the coherent decomposition. indirect channel = product of the causal_chain
--     edge effects (serial path coefficient); direct channel = a deterministic synthesized
--     magnitude; proportion = indirect/(indirect+direct); then indirect_effect = total*p and
--     direct_effect = total - indirect (so they sum to total exactly).
WITH edge_prod AS (
    SELECT cp.path_id,
           cp.causal_effect_size AS total,
           COALESCE(array_length(cp.mediators_identified, 1), 0) AS k,
           exp(sum(ln((e->>'effect')::numeric))) AS edge_product
    FROM public.causal_paths cp,
         LATERAL jsonb_array_elements(cp.causal_chain->'edges') AS e
    WHERE cp.causal_effect_size IS NOT NULL
    GROUP BY cp.path_id, cp.causal_effect_size, cp.mediators_identified
),
decomp AS (
    SELECT path_id,
           total,
           CASE WHEN k = 0 THEN 0::numeric ELSE edge_product END AS med_mag,
           0.10 + (abs(hashtext(path_id || 'direct')) % 21) / 100.0 AS direct_mag
    FROM edge_prod
),
final AS (
    SELECT path_id,
           total,
           CASE WHEN (med_mag + direct_mag) > 0
                THEN round((total * med_mag / (med_mag + direct_mag))::numeric, 4)
                ELSE 0::numeric END AS indirect
    FROM decomp
)
UPDATE public.causal_paths cp
   SET indirect_effect = f.indirect,
       direct_effect = round((f.total - f.indirect)::numeric, 4)
  FROM final f
 WHERE cp.path_id = f.path_id;

-- (C) Enforce the decomposition invariant (NULL-tolerant so unrelated/partial inserts are
--     not blocked). Dropped-then-added for idempotency.
ALTER TABLE public.causal_paths DROP CONSTRAINT IF EXISTS causal_paths_effect_decomp_chk;
ALTER TABLE public.causal_paths ADD CONSTRAINT causal_paths_effect_decomp_chk
    CHECK (
        direct_effect IS NULL OR indirect_effect IS NULL OR causal_effect_size IS NULL
        OR abs(direct_effect + indirect_effect - causal_effect_size) < 0.001
    );

-- (D) Register the read-only CM-005 statement (allowlist; executed only via kpi_query).
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('causal_metrics_mediation', $kpi$SELECT AVG(indirect_effect / NULLIF(causal_effect_size, 0))::float AS proportion_mediated, COUNT(*)::int AS n_paths, AVG(indirect_effect)::float AS mean_indirect, AVG(direct_effect)::float AS mean_direct FROM causal_paths WHERE causal_effect_size > 0 AND indirect_effect IS NOT NULL AND direct_effect IS NOT NULL$kpi$, 0, $note$CM-005 mediation_effect: proportion mediated = mean(indirect_effect / causal_effect_size) over discovered paths. total = causal_effect_size; indirect_effect is the serial-mediation effect through the identified mediators (grounded in the PRODUCT of the causal_chain edge magnitudes), direct_effect = total - indirect (the X->Y effect not through the mediators). direct + indirect = total (enforced by causal_paths_effect_decomp_chk). Paths with no mediators contribute 0. NULL proportion_mediated (fail-loud) when no paths exist.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_id is callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
