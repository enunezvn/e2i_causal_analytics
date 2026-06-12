-- ============================================================================
-- Migration 071: add gap_analyzer / orchestrator domain event labels to
-- memory_event_type (issue #883 PR A; follow-up to #876 / het #873 / 070).
-- ============================================================================
-- gap_analyzer's episodic-memory write (store_gap_analysis) and orchestrator's
-- (store_orchestration) use these labels; without them the INSERT raises 22P02
-- (invalid enum value) and the episodic persist silently fails (the hooks'
-- broad ``except`` swallows it). The same missing values also 22P02 the
-- agents' event_type-FILTERED episodic searches (_get_episodic_context /
-- get_historical_roi_data / get_opportunity_benchmarks for gap_analyzer; the
-- orchestrator read filter) — the migration-046 het read-path lesson. Per the
-- settled convention (#788/#785, #873, #876), domain events belong in
-- memory_event_type while memory_outcome_type stays the generic STATE enum;
-- the agents' outcome_type mapping is fixed in code in the same change.
-- Follows the 020/039/040/046/065/070 extension pattern.
--
-- CAVEAT: ALTER TYPE ... ADD VALUE is non-transactional. run_migrations.sh
-- detects "ALTER TYPE ... ADD VALUE" and applies this file UN-wrapped (no
-- --single-transaction), tracking it separately on clean exit. Do NOT add any
-- statement here that consumes the new values (they are unusable until commit).
-- ----------------------------------------------------------------------------

ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'gap_analysis_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'orchestration_completed';
