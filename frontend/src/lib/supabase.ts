/**
 * Supabase Client Configuration
 * =============================
 *
 * Singleton Supabase client for authentication and database operations.
 * Uses environment variables from config/env.ts for configuration.
 *
 * Usage:
 *   import { supabase } from '@/lib/supabase'
 *   const { data, error } = await supabase.auth.signInWithPassword({...})
 */

import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import { env } from '@/config/env';

/**
 * Database types (extend as needed for your schema)
 * These can be auto-generated using Supabase CLI: supabase gen types typescript
 */
export interface Database {
  public: {
    Tables: Record<string, unknown>;
    Views: Record<string, unknown>;
    Functions: Record<string, unknown>;
    Enums: Record<string, unknown>;
  };
}

/**
 * User metadata type from Supabase Auth
 */
export interface UserMetadata {
  name?: string;
  avatar_url?: string;
  email_verified?: boolean;
}

/**
 * App metadata type (set by admin/service role)
 */
export interface AppMetadata {
  role?: 'admin' | 'user';
  is_admin?: boolean;
  provider?: string;
  providers?: string[];
}

/**
 * Supabase client singleton
 *
 * Configured with:
 * - Auto token refresh
 * - Session persistence in localStorage
 * - URL detection for auth redirects
 */
let supabaseClient: SupabaseClient<Database> | null = null;

/**
 * Get or create the Supabase client singleton
 * @returns Configured Supabase client
 * @throws Error if Supabase is not configured
 */
export function getSupabaseClient(): SupabaseClient<Database> {
  if (supabaseClient) {
    return supabaseClient;
  }

  if (!env.supabaseUrl || !env.supabaseAnonKey) {
    throw new Error(
      'Supabase is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY environment variables.'
    );
  }

  supabaseClient = createClient<Database>(env.supabaseUrl, env.supabaseAnonKey, {
    auth: {
      // Automatically refresh the token before it expires
      autoRefreshToken: true,
      // Persist session in localStorage
      persistSession: true,
      // Detect session from URL (for OAuth/magic link flows)
      detectSessionInUrl: true,
      // Storage key for session
      storageKey: 'e2i-auth-token',
    },
  });

  return supabaseClient;
}

/**
 * Convenience export for the Supabase client
 * Use this in most cases for cleaner imports
 *
 * NOTE: This is a lazy getter - the client is only created when first accessed.
 * This allows the app to load even when Supabase is not configured.
 */
export const supabase = new Proxy({} as SupabaseClient<Database>, {
  get(_target, prop) {
    return Reflect.get(getSupabaseClient(), prop);
  },
});

/**
 * Check if Supabase is configured
 * @returns true if both URL and anon key are set
 */
export function isSupabaseConfigured(): boolean {
  return Boolean(env.supabaseUrl && env.supabaseAnonKey);
}

/**
 * Get the current access token for API requests
 * @returns The JWT access token or null if not authenticated
 */
export async function getAccessToken(): Promise<string | null> {
  const {
    data: { session },
  } = await supabase.auth.getSession();
  return session?.access_token ?? null;
}

/**
 * Helper to get user's display name
 * @param user - Supabase user object
 * @returns Display name from metadata or email prefix
 */
export function getUserDisplayName(
  user: { email?: string; user_metadata?: UserMetadata } | null
): string {
  if (!user) return 'Guest';
  return user.user_metadata?.name ?? user.email?.split('@')[0] ?? 'User';
}

export default supabase;
