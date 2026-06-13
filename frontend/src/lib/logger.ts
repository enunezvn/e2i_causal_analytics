/**
 * Gated Logger Utility
 * ====================
 *
 * A thin wrapper over `console` that suppresses developer-noise logging
 * (`debug` / `log` / `info`) in production builds while ALWAYS forwarding
 * `warn` / `error` so genuine error reporting survives in production.
 *
 * Motivation (#18 console hygiene):
 *   The app shipped ~30 raw `console.log` / `console.debug` runtime calls
 *   (API request/response payloads, auth-lifecycle events with user emails,
 *   websocket chatter, a per-render chat-sidebar log). Those leak internal
 *   state and PII to the browser console of every production user. We want
 *   them silent in prod but still available while developing.
 *
 * Gating:
 *   - Emits in development (`env.isDev`), OR
 *   - When `VITE_DEBUG === 'true'` (lets us force-enable verbose logging in a
 *     production build for field debugging without a code change).
 *   `warn`/`error` ignore the gate — they always emit.
 *
 * Usage:
 *   import { logger } from '@/lib/logger';
 *   logger.debug('[API] GET /foo', { correlationId });
 *   logger.warn('degraded fallback path');
 *   logger.error('request failed', err);
 */

import { env } from '@/config/env';

/**
 * Whether developer-noise logging (debug/log/info) should be forwarded.
 *
 * Read at module-construction time. `import.meta.env.VITE_DEBUG` is statically
 * replaced by Vite at build time, so a production bundle with VITE_DEBUG unset
 * tree-shakes to a constant `false` and the noise channels become no-ops.
 */
const verboseEnabled: boolean =
  env.isDev || import.meta.env.VITE_DEBUG === 'true';

export interface Logger {
  /** Verbose diagnostic logging. Gated: silent in production unless VITE_DEBUG=true. */
  debug: (...args: unknown[]) => void;
  /** General logging. Gated: silent in production unless VITE_DEBUG=true. */
  log: (...args: unknown[]) => void;
  /** Informational logging. Gated: silent in production unless VITE_DEBUG=true. */
  info: (...args: unknown[]) => void;
  /** Warning. ALWAYS emitted — warnings are not dev-noise. */
  warn: (...args: unknown[]) => void;
  /** Error. ALWAYS emitted — error reporting must survive production. */
  error: (...args: unknown[]) => void;
}

// NOTE: this module is the single sanctioned console boundary. The
// `no-console` lint rule is disabled for this file via eslint.config.js.
export const logger: Logger = {
  debug: (...args: unknown[]): void => {
    if (verboseEnabled) {
      console.debug(...args);
    }
  },
  log: (...args: unknown[]): void => {
    if (verboseEnabled) {
      console.log(...args);
    }
  },
  info: (...args: unknown[]): void => {
    if (verboseEnabled) {
      console.info(...args);
    }
  },
  warn: (...args: unknown[]): void => {
    console.warn(...args);
  },
  error: (...args: unknown[]): void => {
    console.error(...args);
  },
};

export default logger;
