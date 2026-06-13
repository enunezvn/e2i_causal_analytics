/**
 * Logger Utility Tests
 * ====================
 *
 * Verifies the gated logger no-ops in production builds (#18 console hygiene)
 * while preserving error/warn reporting that must survive in production.
 *
 * Intent: stop dev-noise / API-payload logging from shipping to production
 * consoles, WITHOUT silencing genuine error reporting. So:
 *   - logger.debug / logger.log  -> gated (silent unless dev OR VITE_DEBUG)
 *   - logger.warn / logger.error -> ALWAYS emit (error reporting is not noise)
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The env module is the codebase's single source of truth for isDev/isProd.
// We mock it per-test so we can assert behavior in both dev and prod builds
// without depending on the ambient vitest MODE.
vi.mock('@/config/env', () => ({
  env: { isDev: false, isProd: true, mode: 'production', isTest: false },
}));

import { env } from '@/config/env';

// Helper: re-import the logger with a fresh module registry so the mocked
// env value is read at construction time.
async function loadLogger() {
  vi.resetModules();
  const mod = await import('./logger');
  return mod.logger;
}

describe('logger', () => {
  let debugSpy: ReturnType<typeof vi.spyOn>;
  let logSpy: ReturnType<typeof vi.spyOn>;
  let infoSpy: ReturnType<typeof vi.spyOn>;
  let warnSpy: ReturnType<typeof vi.spyOn>;
  let errorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  describe('when DEV is false (production build)', () => {
    beforeEach(() => {
      vi.mocked(env).isDev = false;
      vi.stubEnv('VITE_DEBUG', '');
    });

    it('no-ops logger.debug', async () => {
      const logger = await loadLogger();
      logger.debug('[API] GET /foo', { payload: 'secret' });
      expect(debugSpy).not.toHaveBeenCalled();
    });

    it('no-ops logger.log', async () => {
      const logger = await loadLogger();
      logger.log('per-render noise');
      expect(logSpy).not.toHaveBeenCalled();
    });

    it('no-ops logger.info', async () => {
      const logger = await loadLogger();
      logger.info('informational dev message');
      expect(infoSpy).not.toHaveBeenCalled();
    });

    it('STILL emits logger.warn (warnings are not dev-noise)', async () => {
      const logger = await loadLogger();
      logger.warn('degraded fallback path');
      expect(warnSpy).toHaveBeenCalledWith('degraded fallback path');
    });

    it('STILL emits logger.error (error reporting must survive prod)', async () => {
      const logger = await loadLogger();
      const err = new Error('boom');
      logger.error('request failed', err);
      expect(errorSpy).toHaveBeenCalledWith('request failed', err);
    });
  });

  describe('when DEV is true (development build)', () => {
    beforeEach(() => {
      vi.mocked(env).isDev = true;
      vi.stubEnv('VITE_DEBUG', '');
    });

    it('forwards logger.debug to console.debug', async () => {
      const logger = await loadLogger();
      logger.debug('[API] GET /foo', { ok: true });
      expect(debugSpy).toHaveBeenCalledWith('[API] GET /foo', { ok: true });
    });

    it('forwards logger.log to console.log', async () => {
      const logger = await loadLogger();
      logger.log('dev trace');
      expect(logSpy).toHaveBeenCalledWith('dev trace');
    });

    it('forwards logger.info to console.info', async () => {
      const logger = await loadLogger();
      logger.info('dev info');
      expect(infoSpy).toHaveBeenCalledWith('dev info');
    });
  });

  describe('VITE_DEBUG override in a production build', () => {
    beforeEach(() => {
      vi.mocked(env).isDev = false;
    });

    it('forwards logger.debug when VITE_DEBUG=true even with DEV false', async () => {
      vi.stubEnv('VITE_DEBUG', 'true');
      const logger = await loadLogger();
      logger.debug('forced debug');
      expect(debugSpy).toHaveBeenCalledWith('forced debug');
    });

    it('stays silent when VITE_DEBUG is some other value', async () => {
      vi.stubEnv('VITE_DEBUG', 'false');
      const logger = await loadLogger();
      logger.debug('should not log');
      expect(debugSpy).not.toHaveBeenCalled();
    });
  });
});
