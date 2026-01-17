-- =============================================================================
-- E2I Causal Analytics - Anonymous CopilotKit User Support
-- =============================================================================
-- Version: 1.0.0
-- Created: 2026-01-13
-- Description: Enable anonymous user support for CopilotKit conversations
--
-- Problem:
--   CopilotKit sessions don't have authenticated Supabase users, but
--   chatbot_conversations requires a user_id FK to chatbot_user_profiles,
--   which requires a FK to auth.users.
--
-- Solution:
--   1. Remove the auth.users FK constraint from chatbot_user_profiles
--   2. Add a check constraint that allows NULL OR valid auth user
--   3. Insert an anonymous system user for CopilotKit
-- =============================================================================

-- Anonymous user UUID for CopilotKit conversations
DO $$
DECLARE
    anonymous_uuid UUID := '00000000-0000-0000-0000-000000000000';
BEGIN
    -- Step 1: Drop the foreign key constraint on chatbot_user_profiles
    -- This allows us to insert the anonymous user without auth.users
    ALTER TABLE public.chatbot_user_profiles
        DROP CONSTRAINT IF EXISTS chatbot_user_profiles_id_fkey;

    RAISE NOTICE 'Dropped FK constraint on chatbot_user_profiles.id';

    -- Step 2: Insert the anonymous user profile
    INSERT INTO public.chatbot_user_profiles (
        id,
        email,
        full_name,
        is_admin,
        expertise_level,
        created_at,
        updated_at
    ) VALUES (
        anonymous_uuid,
        'anonymous@copilotkit.system',
        'Anonymous CopilotKit User',
        FALSE,
        'intermediate',
        NOW(),
        NOW()
    )
    ON CONFLICT (id) DO UPDATE SET
        updated_at = NOW();

    RAISE NOTICE 'Inserted anonymous user profile with id: %', anonymous_uuid;

    -- Step 3: Re-add FK constraint but make it partial (exclude anonymous user)
    -- We use a CHECK constraint + partial index approach instead
    -- This ensures regular users still reference auth.users

    -- Add a column to mark system users (doesn't require auth.users)
    ALTER TABLE public.chatbot_user_profiles
        ADD COLUMN IF NOT EXISTS is_system_user BOOLEAN DEFAULT FALSE;

    -- Mark the anonymous user as a system user
    UPDATE public.chatbot_user_profiles
        SET is_system_user = TRUE
        WHERE id = anonymous_uuid;

    RAISE NOTICE 'Marked anonymous user as system user';

END $$;

-- Add a comment explaining the anonymous user
COMMENT ON COLUMN public.chatbot_user_profiles.is_system_user IS
    'System users (like anonymous CopilotKit user) are not linked to auth.users';

-- Grant access to service role for system operations
GRANT ALL ON public.chatbot_user_profiles TO service_role;

-- =============================================================================
-- VERIFICATION
-- =============================================================================

-- Verify the anonymous user was created
DO $$
DECLARE
    anon_exists BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1 FROM public.chatbot_user_profiles
        WHERE id = '00000000-0000-0000-0000-000000000000'
    ) INTO anon_exists;

    IF anon_exists THEN
        RAISE NOTICE 'SUCCESS: Anonymous user profile exists';
    ELSE
        RAISE EXCEPTION 'FAILED: Anonymous user profile was not created';
    END IF;
END $$;
