-- Migration: Add Role-Based Access Control (RBAC) to user_profiles
-- Date: 2026-01-21
-- Description: Implements a 4-role hierarchical system: ADMIN > OPERATOR > ANALYST > VIEWER

-- Step 1: Create the user_role enum type
CREATE TYPE user_role AS ENUM ('viewer', 'analyst', 'operator', 'admin');

-- Step 2: Add role column to user_profiles with default 'viewer'
ALTER TABLE user_profiles ADD COLUMN role user_role DEFAULT 'viewer' NOT NULL;

-- Step 3: Migrate existing admins - set role to 'admin' where is_admin is TRUE
UPDATE user_profiles SET role = 'admin' WHERE is_admin = TRUE;

-- Step 4: Create helper function to get numeric level for role comparison
-- Higher number = more privileges
CREATE OR REPLACE FUNCTION role_level(r user_role)
RETURNS INTEGER
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
    RETURN CASE r
        WHEN 'viewer' THEN 1
        WHEN 'analyst' THEN 2
        WHEN 'operator' THEN 3
        WHEN 'admin' THEN 4
        ELSE 0
    END;
END;
$$;

-- Step 5: Create helper function to check if current user has at least the required role
-- Uses auth.uid() from Supabase to get current user
CREATE OR REPLACE FUNCTION has_role(required_role user_role)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
STABLE
AS $$
DECLARE
    user_role_level INTEGER;
    required_level INTEGER;
BEGIN
    -- Get the current user's role level
    SELECT role_level(role) INTO user_role_level
    FROM user_profiles
    WHERE id = auth.uid();

    -- If user not found, deny access
    IF user_role_level IS NULL THEN
        RETURN FALSE;
    END IF;

    -- Get required role level
    required_level := role_level(required_role);

    -- User has access if their level >= required level
    RETURN user_role_level >= required_level;
END;
$$;

-- Step 6: Create index on role column for efficient filtering
CREATE INDEX idx_user_profiles_role ON user_profiles(role);

-- Step 7: Add comment documenting the role hierarchy
COMMENT ON COLUMN user_profiles.role IS 'User role for RBAC: viewer (read-only) < analyst (run analyses) < operator (manage experiments) < admin (full access)';
COMMENT ON FUNCTION role_level(user_role) IS 'Returns numeric level for role comparison: viewer=1, analyst=2, operator=3, admin=4';
COMMENT ON FUNCTION has_role(user_role) IS 'Checks if current authenticated user has at least the specified role level';
