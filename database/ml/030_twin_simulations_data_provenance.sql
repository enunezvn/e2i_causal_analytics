-- ============================================================================
-- Migration: 030_twin_simulations_data_provenance.sql
-- Purpose: Persist the ATE estimate's provenance on twin simulations (#705 H5b)
-- Dependencies: 012_digital_twin_tables.sql (twin_simulations)
-- ============================================================================
--
-- R1 surfaced data_provenance on the live POST /simulate response (in-memory).
-- This adds the persisted column so a stored simulation — and therefore the
-- GET /simulations, /simulations/history and /simulations/{id} reads — carries
-- the same honest source label ('synthetic_uplift_v1' for the synthetic-DGP
-- uplift model, 'rwd_uplift' for real-world; NULL for legacy/error rows).
--
-- Additive + idempotent: existing rows get NULL (correctly "unknown provenance").
-- Safe to apply before the H5b code lands — current code simply does not write it.
-- ============================================================================

ALTER TABLE twin_simulations
    ADD COLUMN IF NOT EXISTS data_provenance text;

COMMENT ON COLUMN twin_simulations.data_provenance IS
    'Origin of the simulated ATE: synthetic_uplift_v1 (synthetic-DGP-trained '
    'uplift, ~constant per brand/intervention in v1) or rwd_uplift (real-world). '
    'NULL for legacy/error results. Added by migration 030 (#705 H5b).';
