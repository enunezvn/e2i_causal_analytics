-- ============================================================================
-- Migration 059: persist gaps + feedback API route stores (M2)
-- ============================================================================
-- src/api/routes/gaps.py and src/api/routes/feedback.py kept their results in
-- process-local dicts (_analyses_store, _learning_store, _patterns_store,
-- _updates_store, _feedback_store), each marked "# IN-MEMORY STORAGE (replace
-- with Supabase in production)". The live e2i_api runs gunicorn --workers 2, so
-- a POST routed to one worker is invisible to a GET routed to the other (and
-- --max-requests recycling wipes the dict mid-life). This migration creates the
-- backing tables. Rows store the full pydantic response JSON in `payload` plus
-- the scalar columns the route list/health queries filter and sort on.
-- ============================================================================

-- Gap analyses (keyed by GapAnalysisResponse.analysis_id, e.g. "gap_<hex12>").
CREATE TABLE IF NOT EXISTS public.gap_analyses (
    analysis_id TEXT PRIMARY KEY,
    brand       TEXT NOT NULL,
    status      TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- list_opportunities() filters status=COMPLETED then brand; index both.
CREATE INDEX IF NOT EXISTS idx_gap_analyses_brand_status
    ON public.gap_analyses (brand, status);
-- get_gap_health() reads the most-recent timestamp.
CREATE INDEX IF NOT EXISTS idx_gap_analyses_updated_at
    ON public.gap_analyses (updated_at DESC);

-- Feedback learning batches (keyed by LearningResponse.batch_id "fb_<hex12>").
CREATE TABLE IF NOT EXISTS public.feedback_learning_batches (
    batch_id   TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feedback_batches_updated_at
    ON public.feedback_learning_batches (updated_at DESC);

-- Detected patterns (keyed by DetectedPattern.pattern_id).
CREATE TABLE IF NOT EXISTS public.feedback_patterns (
    pattern_id   TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    severity     TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feedback_patterns_severity_type
    ON public.feedback_patterns (severity, pattern_type);

-- Knowledge updates (keyed by KnowledgeUpdate.update_id; apply/rollback mutate).
CREATE TABLE IF NOT EXISTS public.feedback_knowledge_updates (
    update_id    TEXT PRIMARY KEY,
    update_type  TEXT NOT NULL,
    status       TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feedback_updates_status
    ON public.feedback_knowledge_updates (status);
CREATE INDEX IF NOT EXISTS idx_feedback_updates_type_agent
    ON public.feedback_knowledge_updates (update_type, target_agent);

-- Raw feedback items appended by POST /feedback/process (FeedbackItem.feedback_id).
CREATE TABLE IF NOT EXISTS public.feedback_items (
    feedback_id  TEXT PRIMARY KEY,
    source_agent TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feedback_items_source_agent
    ON public.feedback_items (source_agent);

-- The backend reaches these as the service role (get_supabase_client), which
-- bypasses RLS; no anon/authenticated grants are added (consistent with the
-- 058 over-grant revoke). RLS left disabled — server-side-only tables.
