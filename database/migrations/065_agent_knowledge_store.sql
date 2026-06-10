-- ============================================================================
-- Migration 065: agent_knowledge_store — durable backend for the feedback
-- learner's knowledge updates (issue #837, F15 follow-up)
-- ============================================================================
-- The feedback_learner's KnowledgeUpdaterNode proposes updates of four
-- knowledge_types (baseline/agent_config/prompt/threshold), each carrying a
-- free-form ``proposed_change`` suggestion string (src/agents/feedback_learner/
-- state.py: proposed_change Optional[str]). Until now NO real store backed
-- ``store.update(...)`` for these types, so ``applied_updates`` was always empty
-- and ``update_effectiveness`` was honestly reported as None
-- (update_backend_wired=False, per the F15 fix #838).
--
-- This table is that real backend. It holds the CURRENT recorded value per
-- (knowledge_type, key) — ``key`` is the affected agent — plus the justification
-- and a monotonically bumped version. The store writes here and READS BACK to
-- confirm persistence before counting an update as applied, so
-- ``update_effectiveness`` becomes a real measured ratio of
-- durably-persisted / proposed.
--
-- NOTE (honesty): this measures durable PERSISTENCE of the recorded learning,
-- not downstream behavioural impact. Agent-side CONSUMPTION of these values
-- (reading them back to change runtime behaviour) is a separate, future loop;
-- nothing here claims the stored suggestion has yet altered an agent.
--
-- This is NOT the API-route ledger ``feedback_knowledge_updates`` (migration
-- 059), which is keyed by the DIFFERENT api.routes.feedback.KnowledgeUpdate
-- (update_type/status/target_agent) and records the update audit trail. This
-- table holds the agent learning-cycle's current knowledge VALUES.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.agent_knowledge_store (
    knowledge_type TEXT        NOT NULL,
    key            TEXT        NOT NULL,
    value          JSONB       NOT NULL,
    justification  TEXT,
    version        INTEGER     NOT NULL DEFAULT 1,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (knowledge_type, key)
);

-- List/scan by type (e.g. "all current baseline updates").
CREATE INDEX IF NOT EXISTS idx_agent_knowledge_store_type
    ON public.agent_knowledge_store (knowledge_type);

-- The backend reaches this as the service role (bypasses RLS); server-side-only,
-- no anon/authenticated grants (consistent with migrations 058/059). RLS left
-- disabled.
