/// <reference types="vite/client" />

/**
 * Vite Environment Variable Type Definitions
 * ==========================================
 *
 * TypeScript declarations for environment variables exposed by Vite.
 * All variables must be prefixed with VITE_ to be accessible client-side.
 *
 * Usage:
 *   - Set in .env file: VITE_API_URL=http://localhost:8000
 *   - Access in code: import.meta.env.VITE_API_URL
 */

interface ImportMetaEnv {
  /**
   * Base URL for the backend API
   * @example "http://localhost:8000"
   */
  readonly VITE_API_URL: string;

  /**
   * Supabase project URL
   * @example "https://your-project.supabase.co"
   */
  readonly VITE_SUPABASE_URL: string;

  /**
   * Supabase anonymous key for client-side operations
   * Safe to expose in client-side code (has row-level security)
   */
  readonly VITE_SUPABASE_ANON_KEY: string;

  /**
   * Application version from package.json
   * @example "0.1.0"
   */
  readonly VITE_APP_VERSION?: string;

  /**
   * Enable debug mode for additional logging
   * @example "true" or "false"
   */
  readonly VITE_DEBUG?: string;

  /**
   * Current build mode (set by Vite)
   */
  readonly MODE: 'development' | 'production' | 'test';

  /**
   * Base URL for the application (set by Vite)
   */
  readonly BASE_URL: string;

  /**
   * Whether running in production mode (set by Vite)
   */
  readonly PROD: boolean;

  /**
   * Whether running in development mode (set by Vite)
   */
  readonly DEV: boolean;

  /**
   * Whether running in SSR mode (set by Vite)
   */
  readonly SSR: boolean;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
