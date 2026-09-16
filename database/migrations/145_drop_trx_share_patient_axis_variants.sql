-- ============================================================================
-- Migration 145: delete the patient-axis TRx-share statements.
-- ============================================================================
-- WHY: chat session_1789548670222_fcscf3u (2026-09-16) answered "Remibrutinib
-- TRx Share by severity tier and biologic status" with low 38.7% / medium 33.5% /
-- high 29.8% (high flagged `warning`) and biologic-naive = biologic-experienced
-- = 1.000. Both were wrong by construction, not by data.
--
-- These statements compute  brand Rx / ALL portfolio Rx  among the patients in
-- one axis bucket. Every patient is on exactly ONE tracked brand (measured live
-- 2026-09-16: every prescription's brand equals its patient's brand; 614 / 597 /
-- 603 rows, zero cross rows), so the denominator adds the Rx of breast-cancer
-- (Kisqali) and PNH (Fabhalta) patients who happen to carry the same tier label.
-- High severity read lowest only because Fabhalta's high-tier patients script
-- heavily (222 of 551 high-tier Rx). biologic_experienced / ige_level are
-- populated for Remibrutinib only, so on those axes the denominator IS the brand
-- and the "share" is always 1.0. therapy_line is populated for every brand and
-- has the severity-tier defect.
--
-- Intent: migrations 105 / 108 / 111 mirrored the (correct) volume variants
-- byte-for-byte onto share (commit dbdf2fe80); no consumer asked for a
-- cross-indication share, and the frontend chart router serves Rx volume only.
-- The meaningful per-bucket answer is the brand's own TRx by the axis, whose
-- buckets sum to the brand total (Remibrutinib live: high 27% / medium 54% /
-- low 19% of TRx).
--
-- BusinessImpactCalculator._calc_trx_share now refuses every patient axis before
-- any query, and kpi_calculate_tool refuses it before any calculator, so these
-- 12 rows are unreachable. Brand-level share (plain / windowed / region, with
-- their _include_synthetic twins) is untouched.
--
-- Ids are enumerated, never LIKE-matched: `business_impact_trx_share_%` would
-- also match the brand-level rows that stay live.
--
-- ROLLBACK: re-apply 105_kpi_segment_variants.sql, 108_kpi_biologic_ige_variants.sql
-- and 111_kpi_conversion_share_axis_window.sql (all idempotent ON CONFLICT DO
-- UPDATE) -- but the calculator refuses these axes, so rows alone restore nothing.
-- ============================================================================

DELETE FROM kpi_query_registry
WHERE query_id IN (
    'business_impact_trx_share_segment',
    'business_impact_trx_share_segment_include_synthetic',
    'business_impact_trx_share_segment_windowed',
    'business_impact_trx_share_segment_windowed_include_synthetic',
    'business_impact_trx_share_line',
    'business_impact_trx_share_line_include_synthetic',
    'business_impact_trx_share_line_windowed',
    'business_impact_trx_share_line_windowed_include_synthetic',
    'business_impact_trx_share_biologic',
    'business_impact_trx_share_biologic_include_synthetic',
    'business_impact_trx_share_ige_tier',
    'business_impact_trx_share_ige_tier_include_synthetic'
);
