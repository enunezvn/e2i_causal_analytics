-- =============================================================================
-- E2I Causal Analytics - Chatbot User Profiles
-- =============================================================================
-- Version: 1.0.0
-- Created: 2026-01-08
-- Description: User profiles for chatbot with E2I-specific preferences
--
-- Features:
--   - Basic user profile linked to auth.users
--   - E2I-specific preferences (brand, region, expertise level)
--   - Auto-profile creation trigger on user signup
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- ENUMS
-- =============================================================================

-- User expertise levels for contextual responses
DO $$ BEGIN
    CREATE TYPE public.chatbot_expertise_level AS ENUM (
        'basic',        -- New users, simplified explanations
        'intermediate', -- Familiar with analytics, standard detail
        'advanced',     -- Power users, technical depth
        'expert'        -- Data scientists, full technical access
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- TABLES
-- =============================================================================

-- Chatbot user profiles with E2I extensions
CREATE TABLE IF NOT EXISTS public.chatbot_user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    full_name TEXT,
    is_admin BOOLEAN DEFAULT FALSE,

    -- E2I-specific preferences
    brand_preference TEXT,  -- Default brand context (Kisqali, Fabhalta, Remibrutinib)
    region_preference TEXT, -- Default region filter
    expertise_level public.chatbot_expertise_level DEFAULT 'intermediate',

    -- User settings
    default_time_range TEXT DEFAULT 'last_30_days',
    show_technical_details BOOLEAN DEFAULT TRUE,
    enable_recommendations BOOLEAN DEFAULT TRUE,

    -- Usage tracking
    total_conversations INTEGER DEFAULT 0,
    total_messages INTEGER DEFAULT 0,
    last_active_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_chatbot_user_profiles_email
    ON public.chatbot_user_profiles(email);

CREATE INDEX IF NOT EXISTS idx_chatbot_user_profiles_brand_preference
    ON public.chatbot_user_profiles(brand_preference)
    WHERE brand_preference IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chatbot_user_profiles_last_active
    ON public.chatbot_user_profiles(last_active_at DESC);

CREATE INDEX IF NOT EXISTS idx_chatbot_user_profiles_admin
    ON public.chatbot_user_profiles(is_admin)
    WHERE is_admin = TRUE;

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_chatbot_user_profiles_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_chatbot_user_profiles_timestamp
    ON public.chatbot_user_profiles;
CREATE TRIGGER trigger_update_chatbot_user_profiles_timestamp
    BEFORE UPDATE ON public.chatbot_user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION public.update_chatbot_user_profiles_timestamp();

-- Auto-create profile when user signs up
CREATE OR REPLACE FUNCTION public.handle_new_chatbot_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.chatbot_user_profiles (id, email)
    VALUES (NEW.id, NEW.email)
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop existing trigger if it exists, then create
DROP TRIGGER IF EXISTS on_auth_user_created_chatbot ON auth.users;
CREATE TRIGGER on_auth_user_created_chatbot
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_chatbot_user();

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Check if current user is admin
CREATE OR REPLACE FUNCTION public.chatbot_is_admin()
RETURNS BOOLEAN AS $$
DECLARE
    is_admin_user BOOLEAN;
BEGIN
    SELECT COALESCE(cup.is_admin, FALSE) INTO is_admin_user
    FROM public.chatbot_user_profiles cup
    WHERE cup.id = auth.uid();

    RETURN COALESCE(is_admin_user, FALSE);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Get user's brand preference (for default context)
CREATE OR REPLACE FUNCTION public.get_user_brand_preference(p_user_id UUID)
RETURNS TEXT AS $$
DECLARE
    brand TEXT;
BEGIN
    SELECT brand_preference INTO brand
    FROM public.chatbot_user_profiles
    WHERE id = p_user_id;

    RETURN brand;
END;
$$ LANGUAGE plpgsql;

-- Update user activity stats
CREATE OR REPLACE FUNCTION public.update_chatbot_user_activity(
    p_user_id UUID,
    p_new_conversation BOOLEAN DEFAULT FALSE,
    p_new_messages INTEGER DEFAULT 0
)
RETURNS VOID AS $$
BEGIN
    UPDATE public.chatbot_user_profiles
    SET
        total_conversations = total_conversations + CASE WHEN p_new_conversation THEN 1 ELSE 0 END,
        total_messages = total_messages + p_new_messages,
        last_active_at = NOW(),
        updated_at = NOW()
    WHERE id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- GRANTS
-- =============================================================================

GRANT SELECT, INSERT, UPDATE ON public.chatbot_user_profiles TO authenticated;
GRANT EXECUTE ON FUNCTION public.chatbot_is_admin() TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_user_brand_preference(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION public.update_chatbot_user_activity(UUID, BOOLEAN, INTEGER) TO authenticated;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE public.chatbot_user_profiles IS
    'User profiles for E2I chatbot with brand/region preferences and expertise levels';
COMMENT ON COLUMN public.chatbot_user_profiles.brand_preference IS
    'Default brand context for queries (Kisqali, Fabhalta, Remibrutinib)';
COMMENT ON COLUMN public.chatbot_user_profiles.expertise_level IS
    'User expertise level affecting response detail and complexity';
COMMENT ON FUNCTION public.chatbot_is_admin() IS
    'Check if current authenticated user has admin privileges for chatbot';
