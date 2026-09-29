-- Migration 162: hcp_adoption_goldstd_v -- the HCP-adoption goldstd frame as ONE relation
-- (Part of #2287, blocker 3 of 3 against auto-retraining the three
-- hcp_adoption_<brand>_goldstd_lr_v1 champions; the loader allowlist is #2286).
--
-- WHY: a table cohort contract ({"type":"table","table":...,"filters":{...},
-- "columns":[...]}, #2207 / migration 151) names exactly one relation, but the goldstd
-- HCP frame is hcp_brand_adoption JOIN hcp_profiles: the label and the row provenance
-- live on hcp_brand_adoption, the 5 covariates on hcp_profiles. This view materialises
-- that join so the contract stays single-relation. The alternative -- a join spec in
-- the contract -- was rejected: every consumer that resolves data_source["table"]
-- (data_loader, the #2320 pandera projection, the GE contract suite, the sweep's
-- brand derivation, training_provenance_from_contract) would have had to learn it.
--
-- FRAME FIDELITY (src/mlops/gold_standard_eval/feature_builder.py _load_hcp_frame):
--   * the goldstd builder selects hcp_brand_adoption.{hcp_id, consideration_date,
--     data_split, adopted} + the FK-embedded hcp_profiles(<the 5 covariates>) with
--     brand=<brand> AND is_synthetic=true on hcp_brand_adoption. A PostgREST FK embed
--     is a LEFT join, so this view is a LEFT JOIN too (hcp_id is NOT NULL and an FK,
--     so no row is ever lost or duplicated: hcp_profiles.hcp_id is the PK).
--   * is_synthetic and data_split are the ADOPTION row's -- the builder never filters
--     on hcp_profiles.is_synthetic, so neither does a contract load of this view.
--   * covariates are exactly cohort_spec._HCP_COVARIATES. treatment_arm is NOT
--     exposed: it is not a goldstd feature and a column-scoped contract never selects
--     it; a wider view would only widen what an unscoped load could read.
--   * id and created_at (the adoption row's) are row metadata, never contract columns:
--     MLDataLoader.count_records selects "id" and its date helpers default to
--     "created_at", so an allowlisted relation must answer both (codex r1 on PR #2326).
--
-- SECURITY: security_invoker = true, so a caller reads through its OWN grants on the
-- base tables (both are service_role-only since 058, RLS off); a view without it
-- would run as its owner (postgres) and could re-expose the tables to anyone granted
-- the view. The explicit REVOKE neutralises any default ACL that grants new public
-- relations to anon/authenticated (058's M9 finding); service_role gets SELECT only.
--
-- IDEMPOTENT: CREATE OR REPLACE VIEW + ALTER VIEW SET + REVOKE/GRANT all re-run
-- cleanly. NOTIFY makes PostgREST reload its schema cache so the view is servable.
--
-- NOTE: no BEGIN/COMMIT here -- scripts/run_migrations.sh wraps each file.

CREATE OR REPLACE VIEW public.hcp_adoption_goldstd_v
WITH (security_invoker = true) AS
SELECT
    a.id,
    a.hcp_id,
    a.brand,
    a.consideration_date,
    a.adopted,
    a.data_split,
    a.is_synthetic,
    a.created_at,
    p.peer_influence_score,
    p.influence_network_size,
    p.years_experience,
    p.specialty,
    p.geographic_region
FROM public.hcp_brand_adoption a
LEFT JOIN public.hcp_profiles p ON p.hcp_id = a.hcp_id;

REVOKE ALL ON public.hcp_adoption_goldstd_v FROM PUBLIC, anon, authenticated;
GRANT SELECT ON public.hcp_adoption_goldstd_v TO service_role;

COMMENT ON VIEW public.hcp_adoption_goldstd_v IS
    'HCP-adoption goldstd frame (migration 162, #2287): hcp_brand_adoption LEFT JOIN '
    'hcp_profiles, exactly the frame FeatureBuilder._load_hcp_frame builds. One row per '
    '(hcp_id, brand); label adopted; is_synthetic/data_split are the adoption row''s. '
    'Named by the table cohort contract of the hcp_adoption_<brand>_goldstd_lr_v1 rows '
    '(migration 163). security_invoker; service_role only.';

NOTIFY pgrst, 'reload schema';
