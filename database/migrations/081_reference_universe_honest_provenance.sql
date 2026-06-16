-- =============================================================================
-- 081: reference_universe must carry HONEST synthetic provenance.
-- =============================================================================
-- The reference_universe rows are SYNTHETIC (they exist only to provide coverage
-- denominators for WS1-DQ-001/002 over synthetic patient_journeys/hcp_profiles).
-- The live rows were seeded with REAL third-party vendor feed names in
-- `data_source` — dishonest provenance that makes synthetic rows look real:
--
--     universe_type | data_source  | rows
--     --------------+--------------+-----
--     hcp           | IQVIA_APLD   |  72
--     patient       | HealthVerity |  12
--
-- The committed synthetic generator (src/ml/data_generator.py) emits the single
-- honest label SYNTHETIC_DATA_SOURCE = 'synthetic_e2i_v3' for every
-- reference_universe row, and the honesty gate
-- tests/unit/test_ml/test_data_generator_provenance.py asserts no row claims a
-- real vendor (IQVIA_APLD / IQVIA_LAAD / HealthVerity / Komodo / Veeva) and that
-- every label starts with 'synthetic'. That gate guards NEW generated rows; these
-- live rows are orphaned legacy seed that predates the gate, so they slipped past.
--
-- FIX: force every reference_universe row to the committed honest label, matching
-- exactly what a fresh _generate_reference_universe() would produce. Idempotent
-- (IS DISTINCT FROM → re-running affects 0 rows) and a no-op on a fresh/empty DB
-- (nothing seeds this table in migrations). No code reads data_source from
-- reference_universe, so relabeling is consumer-safe.
-- =============================================================================

UPDATE reference_universe
SET data_source = 'synthetic_e2i_v3'
WHERE data_source IS DISTINCT FROM 'synthetic_e2i_v3';
