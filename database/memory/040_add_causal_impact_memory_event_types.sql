-- ============================================================================
-- Migration 040: Add causal_impact memory event types
-- ============================================================================
-- Purpose: The causal_impact episodic write path emits event_types that were
--          never added to the ``memory_event_type`` enum, so every causal_impact
--          episodic write failed with
--          ``invalid input value for enum memory_event_type`` (code 22P02) — the
--          error was swallowed by the agents' bare ``except`` so it was invisible
--          in normal operation. This is the Tier-1 analog of the Tier-0 drift that
--          migration 039 fixed (#787), and the real blocker for #785 (a Tier-1
--          ``tool_composer`` / ``causal_impact`` run grows episodic_memories) and
--          for #788 (a faithful causal_impact run writes a 1536-dim episodic).
--
--          * ``causal_analysis_completed`` — emitted by the canonical
--            ``contribute_to_memory`` → ``store_causal_analysis`` path
--            (memory_hooks.py) AND read back by ``get_prior_analyses`` /
--            ``_get_episodic_context``. This is the live path that tool_composer
--            and causal_impact.run() drive.
--          * ``causal_analysis`` — the default event_type of the
--            ``save_episodic_memory`` contract method (agent.py).
--
-- Reference: #788 (causal_impact episodic write-path repair), #785 (at-scale populate).
-- Safety: additive only (ADD VALUE IF NOT EXISTS) — cannot break existing rows.
-- Mirrors migrations 020 (Tier-1) and 039 (Tier-0).
-- ============================================================================

ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'causal_analysis_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'causal_analysis';
