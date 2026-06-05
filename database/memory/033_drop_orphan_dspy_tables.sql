-- =============================================================================
-- Drop the orphaned MIPROv2-era DSPy tables (audit 2026-06-05, F3 / D3)
-- Migration: 033_drop_orphan_dspy_tables.sql
-- Date: 2026-06-05
-- =============================================================================
--
-- `dspy_optimization_runs`, `dspy_prompt_versions`, `dspy_cognitive_context_history`
-- (migration 014, built for the now-baseline MIPROv2 optimizer) have ZERO src/
-- writers or readers (verified). Their function is superseded three ways, all
-- live or newer:
--   * the newer GEPA tables in `database/ml/023_gepa_optimization_tables.sql`
--     (`prompt_optimization_runs`, `optimized_instructions`) — 023 even tags
--     `miprov2` as "Previous DSPy optimizer (baseline)";
--   * file artifacts — compiled DSPy programs/prompts saved/loaded as
--     `artifacts/dspy/*.json` and GEPA module versions as `optimized_modules/*.json`;
--   * MLflow for optimization-run tracking.
--
-- DO NOT drop in this migration (intentionally preserved):
--   * `dspy_agent_training_signals` — LIVE (writer src/rag/memory_adapters.py:779,
--     reader :814 get_signals_for_optimization);
--   * the `database/ml/023` GEPA tables — the current (unwired-but-roadmapped) stake.
--
-- Idempotent (IF EXISTS). CASCADE clears dependent indexes/FKs.
-- =============================================================================

DROP TABLE IF EXISTS dspy_optimization_runs CASCADE;
DROP TABLE IF EXISTS dspy_prompt_versions CASCADE;
DROP TABLE IF EXISTS dspy_cognitive_context_history CASCADE;
