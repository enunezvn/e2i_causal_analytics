-- =============================================================================
-- E2I Causal Analytics - Chatbot Row Level Security Policies
-- =============================================================================
-- Version: 1.0.0
-- Created: 2026-01-08
-- Description: RLS policies for chatbot tables
--
-- Security Model:
--   - Users can only access their own data
--   - Admins can access all data
--   - Soft-delete via is_archived (no hard deletes allowed)
--   - Service role bypasses RLS for backend operations
-- =============================================================================

-- =============================================================================
-- ENABLE ROW LEVEL SECURITY
-- =============================================================================

ALTER TABLE public.chatbot_user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chatbot_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chatbot_messages ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- CHATBOT_USER_PROFILES POLICIES
-- =============================================================================

-- Users can view their own profile
DROP POLICY IF EXISTS "Users can view their own chatbot profile" ON public.chatbot_user_profiles;
CREATE POLICY "Users can view their own chatbot profile"
ON public.chatbot_user_profiles
FOR SELECT
USING (auth.uid() = id);

-- Users can update their own profile (but cannot change is_admin)
DROP POLICY IF EXISTS "Users can update their own chatbot profile" ON public.chatbot_user_profiles;
CREATE POLICY "Users can update their own chatbot profile"
ON public.chatbot_user_profiles
FOR UPDATE
USING (auth.uid() = id)
WITH CHECK (
    auth.uid() = id
    AND is_admin IS NOT DISTINCT FROM (
        SELECT cup.is_admin
        FROM public.chatbot_user_profiles cup
        WHERE cup.id = auth.uid()
    )
);

-- Admins can view all profiles
DROP POLICY IF EXISTS "Admins can view all chatbot profiles" ON public.chatbot_user_profiles;
CREATE POLICY "Admins can view all chatbot profiles"
ON public.chatbot_user_profiles
FOR SELECT
USING (public.chatbot_is_admin());

-- Admins can update all profiles (including is_admin)
DROP POLICY IF EXISTS "Admins can update all chatbot profiles" ON public.chatbot_user_profiles;
CREATE POLICY "Admins can update all chatbot profiles"
ON public.chatbot_user_profiles
FOR UPDATE
USING (public.chatbot_is_admin());

-- No deletes allowed (soft delete via deactivation)
DROP POLICY IF EXISTS "Deny delete for chatbot_user_profiles" ON public.chatbot_user_profiles;
CREATE POLICY "Deny delete for chatbot_user_profiles"
ON public.chatbot_user_profiles
FOR DELETE
USING (false);

-- =============================================================================
-- CHATBOT_CONVERSATIONS POLICIES
-- =============================================================================

-- Users can view their own conversations
DROP POLICY IF EXISTS "Users can view their own chatbot conversations" ON public.chatbot_conversations;
CREATE POLICY "Users can view their own chatbot conversations"
ON public.chatbot_conversations
FOR SELECT
USING (auth.uid() = user_id);

-- Users can insert their own conversations
DROP POLICY IF EXISTS "Users can insert their own chatbot conversations" ON public.chatbot_conversations;
CREATE POLICY "Users can insert their own chatbot conversations"
ON public.chatbot_conversations
FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- Users can update their own conversations (archive, pin, title)
DROP POLICY IF EXISTS "Users can update their own chatbot conversations" ON public.chatbot_conversations;
CREATE POLICY "Users can update their own chatbot conversations"
ON public.chatbot_conversations
FOR UPDATE
USING (auth.uid() = user_id);

-- Admins can view all conversations
DROP POLICY IF EXISTS "Admins can view all chatbot conversations" ON public.chatbot_conversations;
CREATE POLICY "Admins can view all chatbot conversations"
ON public.chatbot_conversations
FOR SELECT
USING (public.chatbot_is_admin());

-- Admins can update all conversations
DROP POLICY IF EXISTS "Admins can update all chatbot conversations" ON public.chatbot_conversations;
CREATE POLICY "Admins can update all chatbot conversations"
ON public.chatbot_conversations
FOR UPDATE
USING (public.chatbot_is_admin());

-- Admins can insert conversations (for support/testing)
DROP POLICY IF EXISTS "Admins can insert chatbot conversations" ON public.chatbot_conversations;
CREATE POLICY "Admins can insert chatbot conversations"
ON public.chatbot_conversations
FOR INSERT
WITH CHECK (public.chatbot_is_admin());

-- No deletes allowed (soft delete via is_archived)
DROP POLICY IF EXISTS "Deny delete for chatbot_conversations" ON public.chatbot_conversations;
CREATE POLICY "Deny delete for chatbot_conversations"
ON public.chatbot_conversations
FOR DELETE
USING (false);

-- =============================================================================
-- CHATBOT_MESSAGES POLICIES
-- =============================================================================

-- Users can view messages from their own conversations
-- Uses computed_user_id which is extracted from session_id
DROP POLICY IF EXISTS "Users can view their own chatbot messages" ON public.chatbot_messages;
CREATE POLICY "Users can view their own chatbot messages"
ON public.chatbot_messages
FOR SELECT
USING (auth.uid() = computed_user_id);

-- Users can insert messages in their own conversations
DROP POLICY IF EXISTS "Users can insert messages in their chatbot conversations" ON public.chatbot_messages;
CREATE POLICY "Users can insert messages in their chatbot conversations"
ON public.chatbot_messages
FOR INSERT
WITH CHECK (auth.uid() = computed_user_id);

-- Admins can view all messages
DROP POLICY IF EXISTS "Admins can view all chatbot messages" ON public.chatbot_messages;
CREATE POLICY "Admins can view all chatbot messages"
ON public.chatbot_messages
FOR SELECT
USING (public.chatbot_is_admin());

-- Admins can insert messages (for support/testing)
DROP POLICY IF EXISTS "Admins can insert chatbot messages" ON public.chatbot_messages;
CREATE POLICY "Admins can insert chatbot messages"
ON public.chatbot_messages
FOR INSERT
WITH CHECK (public.chatbot_is_admin());

-- No updates allowed on messages (immutable)
DROP POLICY IF EXISTS "Deny update for chatbot_messages" ON public.chatbot_messages;
CREATE POLICY "Deny update for chatbot_messages"
ON public.chatbot_messages
FOR UPDATE
USING (false);

-- No deletes allowed on messages (immutable audit trail)
DROP POLICY IF EXISTS "Deny delete for chatbot_messages" ON public.chatbot_messages;
CREATE POLICY "Deny delete for chatbot_messages"
ON public.chatbot_messages
FOR DELETE
USING (false);

-- =============================================================================
-- SERVICE ROLE ACCESS
-- =============================================================================

-- Note: The service_role key automatically bypasses RLS
-- This allows backend services to perform operations without RLS restrictions
-- Ensure service_role key is only used server-side and never exposed to clients

-- =============================================================================
-- VERIFY RLS IS ENABLED
-- =============================================================================

-- Verification query (run manually to confirm RLS is active)
-- SELECT
--     schemaname,
--     tablename,
--     rowsecurity
-- FROM pg_tables
-- WHERE tablename LIKE 'chatbot_%'
--   AND schemaname = 'public';

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON POLICY "Users can view their own chatbot profile" ON public.chatbot_user_profiles IS
    'Allows users to SELECT their own profile record';
COMMENT ON POLICY "Users can view their own chatbot conversations" ON public.chatbot_conversations IS
    'Allows users to SELECT conversations where user_id matches auth.uid()';
COMMENT ON POLICY "Users can view their own chatbot messages" ON public.chatbot_messages IS
    'Allows users to SELECT messages where computed_user_id (from session_id) matches auth.uid()';
COMMENT ON POLICY "Deny delete for chatbot_messages" ON public.chatbot_messages IS
    'Messages are immutable audit records - no deletions allowed';
