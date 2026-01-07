/**
 * Environment Configuration
 * =========================
 *
 * Centralized environment variable access with type safety and validation.
 * All Vite environment variables must use the VITE_ prefix.
 *
 * Usage:
 *   import { env } from '@/config/env'
 *   const apiUrl = env.apiUrl
 */

/**
 * Environment configuration interface
 */
export interface EnvConfig {
  /** Base URL for the backend API (e.g., http://localhost:8000) */
  apiUrl: string;
  /** Supabase project URL */
  supabaseUrl: string;
  /** Supabase anonymous key for client-side operations */
  supabaseAnonKey: string;
  /** Current environment mode */
  mode: 'development' | 'production' | 'test';
  /** Whether the app is running in development mode */
  isDev: boolean;
  /** Whether the app is running in production mode */
  isProd: boolean;
  /** App version from package.json */
  appVersion: string;
  /** Whether CopilotKit AI chat is enabled (requires backend) */
  copilotEnabled: boolean;
}

/**
 * Get a required environment variable with validation
 * @param key - The environment variable key (without VITE_ prefix)
 * @param defaultValue - Optional default value if not set
 * @returns The environment variable value
 * @throws Error if the variable is not set and no default is provided
 */
function getEnvVar(key: string, defaultValue?: string): string {
  const fullKey = `VITE_${key}`;
  const value = import.meta.env[fullKey] as string | undefined;

  if (value === undefined || value === '') {
    if (defaultValue !== undefined) {
      return defaultValue;
    }
    // In development, warn but use a sensible default
    if (import.meta.env.DEV) {
      console.warn(
        `Environment variable ${fullKey} is not set. Using default value.`
      );
      return '';
    }
    throw new Error(`Missing required environment variable: ${fullKey}`);
  }

  return value;
}

/**
 * Determine the current environment mode
 */
function getMode(): 'development' | 'production' | 'test' {
  if (import.meta.env.MODE === 'test') return 'test';
  if (import.meta.env.PROD) return 'production';
  return 'development';
}

/**
 * Environment configuration object
 * Provides type-safe access to all environment variables
 */
export const env: EnvConfig = {
  // API Configuration
  apiUrl: getEnvVar('API_URL', 'http://localhost:8000'),

  // Supabase Configuration
  supabaseUrl: getEnvVar('SUPABASE_URL', ''),
  supabaseAnonKey: getEnvVar('SUPABASE_ANON_KEY', ''),

  // Environment Mode
  mode: getMode(),
  isDev: import.meta.env.DEV,
  isProd: import.meta.env.PROD,

  // App Version
  appVersion: import.meta.env.VITE_APP_VERSION ?? '0.1.0',

  // CopilotKit Configuration
  // Enabled by default in production, can be disabled via VITE_COPILOT_ENABLED=false
  // In development, disabled by default (requires backend running)
  // CI E2E tests build with VITE_COPILOT_ENABLED=false to test without backend
  copilotEnabled:
    import.meta.env.VITE_COPILOT_ENABLED !== undefined
      ? import.meta.env.VITE_COPILOT_ENABLED === 'true'
      : import.meta.env.PROD, // Default: enabled in prod, disabled in dev
};

/**
 * API endpoint paths configuration
 * Centralized location for all API endpoint definitions
 */
export const apiEndpoints = {
  // Health endpoints
  health: '/health',
  healthz: '/healthz',
  ready: '/ready',

  // Graph endpoints
  graph: {
    nodes: '/graph/nodes',
    node: (id: string) => `/graph/nodes/${id}`,
    nodeNetwork: (id: string) => `/graph/nodes/${id}/network`,
    relationships: '/graph/relationships',
    traverse: '/graph/traverse',
    causalChains: '/graph/causal-chains',
    query: '/graph/query',
    episodes: '/graph/episodes',
    search: '/graph/search',
    stats: '/graph/stats',
    stream: '/graph/stream',
  },

  // Memory endpoints
  memory: {
    working: '/memory/working',
    semantic: '/memory/semantic',
    episodic: '/memory/episodic',
  },

  // Cognitive endpoints
  cognitive: {
    process: '/cognitive/process',
    status: '/cognitive/status',
  },

  // Explain endpoints
  explain: {
    model: '/explain/model',
    prediction: '/explain/prediction',
    shap: '/explain/shap',
  },

  // RAG endpoints
  rag: {
    query: '/rag/query',
    documents: '/rag/documents',
  },
} as const;

/**
 * Build full API URL from endpoint path
 * @param endpoint - The API endpoint path
 * @returns The full API URL
 */
export function buildApiUrl(endpoint: string): string {
  const baseUrl = env.apiUrl.replace(/\/$/, ''); // Remove trailing slash
  const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${baseUrl}${path}`;
}

export default env;
