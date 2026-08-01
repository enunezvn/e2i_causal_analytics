-- ============================================================================
-- 123_chatbot_message_owner_inherit.sql
-- Inherit chatbot_messages / chatbot_message_feedback owner from the parent
-- conversation instead of casting SPLIT_PART(session_id,'~',1) (#1405).
--
-- WHY: `computed_user_id UUID GENERATED ALWAYS AS
--   (CAST(SPLIT_PART(session_id,'~',1) AS UUID)) STORED` (mig 029 line 99-101,
-- and the identical column on chatbot_message_feedback, mig 031 line 44-46)
-- raised Postgres 22P02 and DROPPED every message/feedback row whenever a
-- CopilotKit threadId's first '~'-segment was not a valid UUID. A dropped
-- chatbot_messages row severs the human feedback chain at the root:
-- chatbot_message_feedback.message_id and chatbot_analytics.message_id both FK
-- to chatbot_messages(id), and add_feedback resolves message_id from a persisted
-- row — so the explicit-thumbs signal (the only human ground truth) never
-- reaches the routing labeler (#1341) or the Tier-5 feedback-learner.
--
-- Separately, the real CopilotKit UI mints BARE-uuid threadIds (no user~ prefix),
-- so SPLIT_PART(session_id,'~',1) was the thread's own random uuid — unrelated to
-- the signed-in user — for 836/874 message rows. The true owner already lives in
-- chatbot_conversations.user_id, now written from the JWT-verified identity
-- (copilotkit _ensure_conversation_exists, same PR) instead of the anon constant.
--
-- WHAT: drop the fragile generated expression on both computed_user_id columns and
-- maintain them via a BEFORE INSERT/UPDATE-OF-session_id trigger that inherits the
-- parent conversation's user_id (looked up by session_id — the FK guarantees the
-- conversation exists). No cast => no 22P02; and message-owner == conversation-owner
-- by construction, so both RLS views (messages/feedback key on computed_user_id,
-- conversations on user_id) stay consistent. session_id itself is untouched, so every
-- read (all keyed on the session_id string via the service-role client) round-trips
-- unchanged. The RLS policies + indexes that reference computed_user_id are preserved
-- (the column stays; only its generation is replaced).
-- ----------------------------------------------------------------------------

-- Shared trigger: computed_user_id := parent conversation's user_id.
-- SELECT INTO leaves NULL if no conversation is found (defensive; FK prevents it) —
-- honest NULL, never a fabricated owner.
CREATE OR REPLACE FUNCTION public.chatbot_inherit_conversation_owner()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    SELECT user_id
      INTO NEW.computed_user_id
      FROM public.chatbot_conversations
     WHERE session_id = NEW.session_id;
    RETURN NEW;
END;
$$;

-- chatbot_messages
ALTER TABLE public.chatbot_messages ALTER COLUMN computed_user_id DROP EXPRESSION;
DROP TRIGGER IF EXISTS trg_chatbot_messages_inherit_owner ON public.chatbot_messages;
CREATE TRIGGER trg_chatbot_messages_inherit_owner
    BEFORE INSERT OR UPDATE OF session_id ON public.chatbot_messages
    FOR EACH ROW EXECUTE FUNCTION public.chatbot_inherit_conversation_owner();

-- chatbot_message_feedback (same generated-column failure mode; session_id FKs to conversations)
ALTER TABLE public.chatbot_message_feedback ALTER COLUMN computed_user_id DROP EXPRESSION;
DROP TRIGGER IF EXISTS trg_chatbot_message_feedback_inherit_owner ON public.chatbot_message_feedback;
CREATE TRIGGER trg_chatbot_message_feedback_inherit_owner
    BEFORE INSERT OR UPDATE OF session_id ON public.chatbot_message_feedback
    FOR EACH ROW EXECUTE FUNCTION public.chatbot_inherit_conversation_owner();

-- Backfill existing rows so message-owner == conversation-owner holds universally.
-- computed_user_id has no application readers (dormant RLS only), so this is a
-- consistency normalization: the real historical owner was discarded at write time
-- and is not retroactively recoverable, so rows normalize to their conversation's
-- recorded owner (anon for pre-fix auto-created conversations). IS DISTINCT FROM
-- keeps it idempotent and a no-op on already-correct rows.
UPDATE public.chatbot_messages m
   SET computed_user_id = c.user_id
  FROM public.chatbot_conversations c
 WHERE c.session_id = m.session_id
   AND m.computed_user_id IS DISTINCT FROM c.user_id;

UPDATE public.chatbot_message_feedback f
   SET computed_user_id = c.user_id
  FROM public.chatbot_conversations c
 WHERE c.session_id = f.session_id
   AND f.computed_user_id IS DISTINCT FROM c.user_id;

-- PostgREST caches the schema; reload so the column's new (non-generated) shape is visible.
NOTIFY pgrst, 'reload schema';
