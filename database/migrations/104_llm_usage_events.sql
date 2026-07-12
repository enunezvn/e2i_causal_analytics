-- Migration 104: llm_usage_events — per-call LLM usage capture (admin
-- observability, spec 2026-07-12).
--
-- One row per completed LLM call, written fail-open by the backend capture
-- hooks (llm_factory LangChain callback + global litellm logger). user_id /
-- session_id are NULL for platform-level (non-chat) calls — attribution is
-- honest-only, never guessed. No cost column: cost is computed at read time
-- from tokens x the pricing table (src/services/llm_pricing.py) so pricing
-- corrections apply retroactively.
--
-- Grants: mirror 101 — RLS on, admins read, service_role (recorder) bypasses.
-- NOTE: no BEGIN/COMMIT here — the migration runner wraps files itself.

CREATE TABLE IF NOT EXISTS llm_usage_events (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    surface       TEXT NOT NULL DEFAULT 'other',
    component     TEXT,
    user_id       UUID,
    session_id    VARCHAR,
    request_id    TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage_events (created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_user    ON llm_usage_events (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_session ON llm_usage_events (session_id);

ALTER TABLE llm_usage_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS llm_usage_admin_read ON llm_usage_events;
CREATE POLICY llm_usage_admin_read ON llm_usage_events
    FOR SELECT TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM chatbot_user_profiles p
            WHERE p.id = auth.uid() AND p.role = 'admin'
        )
    );
-- service_role bypasses RLS; the recorder writes with the service-role client.
