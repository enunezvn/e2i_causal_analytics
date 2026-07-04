-- 094: Brand x territory activity view for resource-optimizer synthetic seeding
--
-- The /resource-optimization page seeds a clearly-labelled SYNTHETIC allocation
-- problem when no targets are supplied. The all-brands path reads
-- territory_metrics (which has NO brand column); this view provides the
-- brand-scoped equivalent: per (brand, territory) HCP counts and treatment-event
-- activity via treatment_events joined to hcp_profiles.
--
-- NOTE: deliberately NO is_synthetic filter — on this box every row is
-- synthetic and an is_synthetic=false filter renders a view permanently empty
-- (BR-002 lesson, see v_kpi_intent_to_prescribe). Provenance labelling is the
-- API layer's job.
--
-- Idempotent: CREATE OR REPLACE. No BEGIN/COMMIT (applied via autocommit psql).

CREATE OR REPLACE VIEW v_brand_territory_activity AS
SELECT
    te.brand,
    hp.territory_id,
    COUNT(DISTINCT te.hcp_id) AS active_hcp_count,
    COUNT(*) AS treatment_event_count
FROM treatment_events te
JOIN hcp_profiles hp USING (hcp_id)
WHERE hp.territory_id IS NOT NULL
GROUP BY te.brand, hp.territory_id;

COMMENT ON VIEW v_brand_territory_activity IS
    'Per (brand, territory) HCP counts + treatment-event activity. Feeds the '
    'resource-optimizer synthetic seeding (brand-scoped path). No is_synthetic '
    'filter by design: this instance is 100% synthetic data.';
