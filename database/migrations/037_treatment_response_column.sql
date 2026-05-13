-- Migration 037: Add treatment_response column to treatment_events
-- Issue #157 PR C (Sub-PR-A) — CSU claim-pattern response proxies.
--
-- Context: src/kpi/calculators/brand_specific.py:123 queries
--   `te.treatment_response IN ('inadequate', 'uncontrolled', 'refractory')`
-- against a column that did NOT exist in v3 schema. This migration creates
-- the column and constrains it to the agreed vocabulary.
--
-- Vocabulary (per issue #157 pre-implementation gate, supervisor-resolved
-- 2026-05-13):
--   * controlled     — persistence met, no rescue steroid burst, no ED visit
--                      (NEW value added for the CSU biologic-response proxy)
--   * inadequate     — persistence met but rescue events present
--   * uncontrolled   — pre-existing brand-specific.py vocabulary
--   * refractory     — biologic switch or immunosuppressant addition
--   * discontinued   — gap > BIOLOGIC_DISCONT_GAP_DAYS (90d) within 180d
--
-- NULL is allowed for rows that do not represent biologic fills, do not
-- meet the >=60d coverage / >=90d follow-up pre-conditions, or for which
-- the response is otherwise indeterminate.
--
-- Forward-only. No DROP COLUMN rollback path is provided; rolling back
-- would silently break the KPI calculator (BR-001 AH Uncontrolled %).

ALTER TABLE treatment_events
    ADD COLUMN IF NOT EXISTS treatment_response VARCHAR(20);

-- CHECK constraint enforcing the vocabulary. NULL is permitted (column
-- semantics: "response indeterminate" for non-biologic events or for
-- biologic events that fail the pre-conditions).
ALTER TABLE treatment_events
    DROP CONSTRAINT IF EXISTS treatment_events_treatment_response_chk;

ALTER TABLE treatment_events
    ADD CONSTRAINT treatment_events_treatment_response_chk
    CHECK (treatment_response IS NULL OR treatment_response IN (
        'controlled',
        'inadequate',
        'uncontrolled',
        'refractory',
        'discontinued'
    ));

COMMENT ON COLUMN treatment_events.treatment_response IS
    'CSU biologic claim-pattern response proxy. NULL outside the CSU '
    'biologic-fill universe or when pre-conditions (>=60d coverage, '
    '>=90d follow-up) are unmet. See issue #157 PR C and '
    'docs/OPTUM_CONVERSION.md for the derivation rules.';
