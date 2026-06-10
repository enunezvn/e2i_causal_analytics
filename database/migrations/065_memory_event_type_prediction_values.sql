-- ============================================================================
-- Migration 065 (M3): add prediction_completed / prediction_delivered to
-- memory_event_type (synthetic-causal-validation; Shard 08 episodic write).
-- ============================================================================
-- prediction_synthesizer's episodic-memory write (gate 7) needs these labels;
-- without them the INSERT raises 22P02 (invalid enum value) and the episodic
-- persist silently fails. Follows the 020/039/040 extension pattern.
--
-- CAVEAT: ALTER TYPE ... ADD VALUE is non-transactional. run_migrations.sh
-- detects "ALTER TYPE ... ADD VALUE" and applies this file UN-wrapped (no
-- --single-transaction), tracking it separately on clean exit. Do NOT add any
-- statement here that consumes the new value (it is unusable until commit).
-- ----------------------------------------------------------------------------

ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'prediction_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'prediction_delivered';
