-- =============================================================================
-- ROI Views Security Fix
-- Migration: 014_roi_calculations_security_fix.sql
-- Date: 2025-12-20
-- Purpose: Fix SECURITY DEFINER issue on ROI views
-- Issue: Views were created with SECURITY DEFINER which bypasses RLS
-- Fix: Explicitly set SECURITY INVOKER to respect querying user's RLS policies
-- =============================================================================

-- Fix v_roi_calculations_summary
ALTER VIEW v_roi_calculations_summary SET (security_invoker = on);

-- Fix v_roi_by_brand
ALTER VIEW v_roi_by_brand SET (security_invoker = on);

-- Fix v_roi_by_workstream
ALTER VIEW v_roi_by_workstream SET (security_invoker = on);

-- Fix v_roi_value_driver_contribution
ALTER VIEW v_roi_value_driver_contribution SET (security_invoker = on);
