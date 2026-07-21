-- ============================================================================
-- 116_kpi_nowcast_registry.sql
-- Claims-arrival LAG-TRIANGLE queries for the completion-factor nowcast
-- estimator (backlog #45, PR-B). Rx-volume family only: TRx WS3-BI-005 /
-- NRx WS3-BI-006 / NBRx WS3-BI-007.
--
-- WHY: the synthetic claims arrival plane (migration 115 adds
-- treatment_events.claim_available_date + adjudication_lag_days; the DGP stamp
-- pass populates them at reseed) lets recent service months be shown as
-- PROVISIONAL (claims still maturing) with a chain-ladder nowcast recovering
-- the mature value. The estimator (src/kpi/nowcast/completion_factor.py) is
-- QUERY-TIME live compute, mirroring the migration-110 segmented-history
-- pattern: no new storage table, no reseed-runbook step; "provisional" is
-- inherently as-of the current frontier.
--
-- SHAPE: one call returns, per calendar service month, the histogram of
-- arrival offsets — rows of (service_month, arrival_offset_days, n) — plus the
-- global prescription date range as scalar columns on every row (data_min /
-- frontier, exactly like migration 110's data_min/data_max, for edge trimming
-- via src/kpi/history_backfill._complete_months). Definitions:
--   * service_month       = date_trunc('month', event_date) — identical month
--                           bucketing to the kpi_history backfill / mig 110.
--   * arrival_offset_days = claim_available_date - month start (integer days).
--                           NULL when the row is UNSTAMPED (pre-#45 rows /
--                           substrate not yet reseeded): the estimator uses the
--                           NULL bucket to measure arrival-plane coverage and
--                           reports "arrival plane not populated" explicitly
--                           instead of fabricating a completion factor.
--   * frontier            = MAX(prescription event_date) — the existing KPI
--                           frontier, unchanged. A claim has "arrived as of the
--                           frontier" iff claim_available_date <= frontier,
--                           i.e. arrival_offset_days <= frontier - month start.
--   * n                   = COUNT of events in that (month, offset) cell. The
--                           per-month total over ALL cells (incl. NULL) is the
--                           MATURE truth — the base KPI value, available
--                           because the substrate is omniscient.
--
-- MIGRATION-113 GUARD (additive-only proof, design item 2): these are the ONLY
-- registry statements that read claim_available_date, and they never apply an
-- as-of cutoff to event_date (the pattern migration 113 falsified: recall
-- 0.1166 false-critical under the #853 anchor-cap pile-up). The certified base
-- TRx/NRx/NBRx statements stay byte-for-byte unchanged; this migration only
-- ADDS parallel query ids. The estimator additionally EXCLUDES the frontier
-- (anchor-cap pile-up) month from completion-curve estimation.
--
-- NBRx JOIN KEY = patient_id (NOT patient_journey_id) — same rationale as
-- migrations 105/110: patient_journey_id is NULL on ~17% of prescriptions.
-- The NBRx arrival event is "the patient's FIRST brand-Rx date became visible":
-- MIN(claim_available_date) over the claims on the patient's true first date
-- (MIN ignores NULLs; an all-NULL group lands in the unstamped bucket). The
-- estimator applies the SAME definition to mature months when estimating the
-- completion curve, so the nowcast is self-consistent.
--
-- ADDITIVE (not in-place), mirroring 105/110/111: new parallel query ids only.
-- The `_include_synthetic` twins (used when the synthetic-gold showcase flag
-- is on, via src/kpi/synthetic_mode.py nowcast_triangle_query_id) drop the
-- `(SELECT * ... WHERE is_synthetic=false)` wrappers.
--
-- Params: [brand] (max_params = 1); NULL brand = all brands. The frontier /
-- data_min scalars are brand-agnostic (global prescription range), matching
-- migration 110.
--
-- Exactly 6 new query ids:
--   {trx,nrx,nbrx}_nowcast_triangle x {plain, _include_synthetic}
--
-- Idempotent (ON CONFLICT DO UPDATE). NO DDL here: the arrival-plane columns
-- are migration 115 (PR-A). Depends on: 044 (registry+RPC — SETOF json
-- supports these multi-row results), 110 (frontier/data_min statement shapes),
-- 115 (claim_available_date column must exist before these statements are
-- EXECUTED; registration itself is plain text and applies cleanly either way).
-- ============================================================================

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    -- ---- TRx arrival lag triangle ----
    ('business_impact_trx_nowcast_triangle', $kpi$SELECT date_trunc('month', te.event_date)::date AS service_month, (te.claim_available_date - date_trunc('month', te.event_date)::date)::int AS arrival_offset_days, COUNT(*) AS n, (SELECT MIN(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS frontier FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te WHERE te.event_type::text = 'prescription' AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$TRx claims-arrival lag triangle (nowcast, backlog #45)$note$),
    ('business_impact_trx_nowcast_triangle_include_synthetic', $kpi$SELECT date_trunc('month', te.event_date)::date AS service_month, (te.claim_available_date - date_trunc('month', te.event_date)::date)::int AS arrival_offset_days, COUNT(*) AS n, (SELECT MIN(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS frontier FROM treatment_events te WHERE te.event_type::text = 'prescription' AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$TRx claims-arrival lag triangle (nowcast, includes synthetic)$note$),
    -- ---- NRx arrival lag triangle (first fill per patient-brand sequence) ----
    ('business_impact_nrx_nowcast_triangle', $kpi$SELECT date_trunc('month', te.event_date)::date AS service_month, (te.claim_available_date - date_trunc('month', te.event_date)::date)::int AS arrival_offset_days, COUNT(*) AS n, (SELECT MIN(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS frontier FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$NRx claims-arrival lag triangle (nowcast, backlog #45)$note$),
    ('business_impact_nrx_nowcast_triangle_include_synthetic', $kpi$SELECT date_trunc('month', te.event_date)::date AS service_month, (te.claim_available_date - date_trunc('month', te.event_date)::date)::int AS arrival_offset_days, COUNT(*) AS n, (SELECT MIN(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS frontier FROM treatment_events te WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$NRx claims-arrival lag triangle (nowcast, includes synthetic)$note$),
    -- ---- NBRx arrival lag triangle (visibility of each patient's FIRST brand Rx) ----
    ('business_impact_nbrx_nowcast_triangle', $kpi$WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) GROUP BY patient_id), first_arrival AS (SELECT fb.patient_id, fb.first_date, MIN(te.claim_available_date) AS first_available FROM first_brand fb JOIN (SELECT * FROM treatment_events WHERE is_synthetic = false) te ON te.patient_id = fb.patient_id AND te.event_date = fb.first_date WHERE te.event_type::text = 'prescription' AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY fb.patient_id, fb.first_date) SELECT date_trunc('month', fa.first_date)::date AS service_month, (fa.first_available - date_trunc('month', fa.first_date)::date)::int AS arrival_offset_days, COUNT(*) AS n, (SELECT MIN(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS frontier FROM first_arrival fa GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$NBRx claims-arrival lag triangle (nowcast, backlog #45)$note$),
    ('business_impact_nbrx_nowcast_triangle_include_synthetic', $kpi$WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM treatment_events WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) GROUP BY patient_id), first_arrival AS (SELECT fb.patient_id, fb.first_date, MIN(te.claim_available_date) AS first_available FROM first_brand fb JOIN treatment_events te ON te.patient_id = fb.patient_id AND te.event_date = fb.first_date WHERE te.event_type::text = 'prescription' AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY fb.patient_id, fb.first_date) SELECT date_trunc('month', fa.first_date)::date AS service_month, (fa.first_available - date_trunc('month', fa.first_date)::date)::int AS arrival_offset_days, COUNT(*) AS n, (SELECT MIN(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS frontier FROM first_arrival fa GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$NBRx claims-arrival lag triangle (nowcast, includes synthetic)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the new ids are visible.
NOTIFY pgrst, 'reload schema';
