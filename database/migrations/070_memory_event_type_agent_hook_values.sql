-- ============================================================================
-- Migration 070: add health_score / experiment_monitor domain event labels to
-- memory_event_type (issue #876; follow-up to het #873 / causal_impact #788).
-- ============================================================================
-- health_score's and experiment_monitor's episodic-memory writes use these
-- labels; without them the INSERT raises 22P02 (invalid enum value) and the
-- episodic persist silently fails (the hooks' broad ``except`` swallows it).
-- The same missing values also 22P02 the agents' event_type-FILTERED episodic
-- searches (get_score_history / get_srm_history / _get_alert_history / ...) —
-- the migration-046 het read-path lesson. Per the 3x-settled convention
-- (#788/#785, #873), domain events belong in memory_event_type while
-- memory_outcome_type stays the generic STATE enum; the agents' outcome_type
-- mapping is fixed in code in the same change. (explainer's
-- 'explanation_generated' and tool_composer's 'composition_completed' already
-- exist — only these three are missing.) Follows the 020/039/040/046/065
-- extension pattern.
--
-- CAVEAT: ALTER TYPE ... ADD VALUE is non-transactional. run_migrations.sh
-- detects "ALTER TYPE ... ADD VALUE" and applies this file UN-wrapped (no
-- --single-transaction), tracking it separately on clean exit. Do NOT add any
-- statement here that consumes the new values (they are unusable until commit).
-- ----------------------------------------------------------------------------

ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'health_check_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'experiment_alert_generated';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'experiment_monitoring_completed';
