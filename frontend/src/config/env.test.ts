/**
 * Environment Configuration Tests
 * ================================
 *
 * Tests for centralized environment variable access and validation.
 */

import { describe, it, expect } from 'vitest';
import { env, apiEndpoints, buildApiUrl, type EnvConfig } from './env';

// =============================================================================
// ENV CONFIG TESTS
// =============================================================================

describe('env config', () => {
  it('has correct structure', () => {
    expect(env).toHaveProperty('apiUrl');
    expect(env).toHaveProperty('supabaseUrl');
    expect(env).toHaveProperty('supabaseAnonKey');
    expect(env).toHaveProperty('mode');
    expect(env).toHaveProperty('isDev');
    expect(env).toHaveProperty('isProd');
    expect(env).toHaveProperty('appVersion');
  });

  it('has string apiUrl', () => {
    expect(typeof env.apiUrl).toBe('string');
    expect(env.apiUrl.length).toBeGreaterThan(0);
  });

  it('has string supabaseUrl', () => {
    expect(typeof env.supabaseUrl).toBe('string');
  });

  it('has string supabaseAnonKey', () => {
    expect(typeof env.supabaseAnonKey).toBe('string');
  });

  it('has valid mode value', () => {
    expect(['development', 'production', 'test']).toContain(env.mode);
  });

  it('has boolean isDev', () => {
    expect(typeof env.isDev).toBe('boolean');
  });

  it('has boolean isProd', () => {
    expect(typeof env.isProd).toBe('boolean');
  });

  it('has string appVersion', () => {
    expect(typeof env.appVersion).toBe('string');
  });

  it('isDev and isProd are mutually exclusive', () => {
    // In test environment, isDev is typically true
    // The key is they shouldn't both be true
    expect(!(env.isDev && env.isProd)).toBe(true);
  });
});

// =============================================================================
// API ENDPOINTS TESTS
// =============================================================================

describe('apiEndpoints', () => {
  describe('health endpoints', () => {
    it('has health endpoint', () => {
      expect(apiEndpoints.health).toBe('/health');
    });

    it('has healthz endpoint', () => {
      expect(apiEndpoints.healthz).toBe('/healthz');
    });

    it('has ready endpoint', () => {
      expect(apiEndpoints.ready).toBe('/ready');
    });
  });

  describe('graph endpoints', () => {
    it('has static graph endpoints', () => {
      expect(apiEndpoints.graph.nodes).toBe('/graph/nodes');
      expect(apiEndpoints.graph.relationships).toBe('/graph/relationships');
      expect(apiEndpoints.graph.traverse).toBe('/graph/traverse');
      expect(apiEndpoints.graph.causalChains).toBe('/graph/causal-chains');
      expect(apiEndpoints.graph.query).toBe('/graph/query');
      expect(apiEndpoints.graph.episodes).toBe('/graph/episodes');
      expect(apiEndpoints.graph.search).toBe('/graph/search');
      expect(apiEndpoints.graph.stats).toBe('/graph/stats');
      expect(apiEndpoints.graph.stream).toBe('/graph/stream');
    });

    it('generates node endpoint with id', () => {
      expect(apiEndpoints.graph.node('123')).toBe('/graph/nodes/123');
      expect(apiEndpoints.graph.node('abc-def')).toBe('/graph/nodes/abc-def');
    });

    it('generates nodeNetwork endpoint with id', () => {
      expect(apiEndpoints.graph.nodeNetwork('123')).toBe('/graph/nodes/123/network');
      expect(apiEndpoints.graph.nodeNetwork('xyz')).toBe('/graph/nodes/xyz/network');
    });
  });

  describe('memory endpoints', () => {
    it('has working memory endpoint', () => {
      expect(apiEndpoints.memory.working).toBe('/memory/working');
    });

    it('has semantic memory endpoint', () => {
      expect(apiEndpoints.memory.semantic).toBe('/memory/semantic');
    });

    it('has episodic memory endpoint', () => {
      expect(apiEndpoints.memory.episodic).toBe('/memory/episodic');
    });
  });

  describe('cognitive endpoints', () => {
    it('has process endpoint', () => {
      expect(apiEndpoints.cognitive.process).toBe('/cognitive/process');
    });

    it('has status endpoint', () => {
      expect(apiEndpoints.cognitive.status).toBe('/cognitive/status');
    });
  });

  describe('explain endpoints', () => {
    it('has model endpoint', () => {
      expect(apiEndpoints.explain.model).toBe('/explain/model');
    });

    it('has prediction endpoint', () => {
      expect(apiEndpoints.explain.prediction).toBe('/explain/prediction');
    });

    it('has shap endpoint', () => {
      expect(apiEndpoints.explain.shap).toBe('/explain/shap');
    });
  });

  describe('rag endpoints', () => {
    it('has query endpoint', () => {
      expect(apiEndpoints.rag.query).toBe('/rag/query');
    });

    it('has documents endpoint', () => {
      expect(apiEndpoints.rag.documents).toBe('/rag/documents');
    });
  });
});

// =============================================================================
// BUILD API URL TESTS
// =============================================================================

describe('buildApiUrl', () => {
  it('builds URL with leading slash', () => {
    const result = buildApiUrl('/health');
    expect(result).toContain('/health');
    expect(result.startsWith('http')).toBe(true);
  });

  it('builds URL without leading slash', () => {
    const result = buildApiUrl('health');
    expect(result).toContain('/health');
    expect(result.startsWith('http')).toBe(true);
  });

  it('handles endpoint with path segments', () => {
    const result = buildApiUrl('/graph/nodes/123');
    expect(result).toContain('/graph/nodes/123');
  });

  it('handles query parameters', () => {
    const result = buildApiUrl('/search?q=test');
    expect(result).toContain('/search?q=test');
  });

  it('combines with base URL correctly', () => {
    const result = buildApiUrl('/api/endpoint');
    // Should not have double slashes (except in http://)
    expect(result.match(/[^:]\/\//)).toBeNull();
  });

  it('returns URL with env.apiUrl as base', () => {
    const result = buildApiUrl('/test');
    expect(result.startsWith(env.apiUrl.replace(/\/$/, ''))).toBe(true);
  });

  it('handles empty endpoint', () => {
    const result = buildApiUrl('');
    expect(result).toBe(env.apiUrl.replace(/\/$/, '') + '/');
  });

  it('handles complex paths', () => {
    const result = buildApiUrl('/graph/nodes/abc-123/network');
    expect(result).toContain('/graph/nodes/abc-123/network');
  });
});

// =============================================================================
// TYPE TESTS
// =============================================================================

describe('EnvConfig type', () => {
  it('matches env object shape', () => {
    // TypeScript compile-time check - if this compiles, types are correct
    const config: EnvConfig = {
      apiUrl: 'http://localhost:8000',
      supabaseUrl: 'https://example.supabase.co',
      supabaseAnonKey: 'test-key',
      mode: 'development',
      isDev: true,
      isProd: false,
      appVersion: '1.0.0',
      copilotEnabled: true,
    };

    expect(config.apiUrl).toBe('http://localhost:8000');
    expect(config.mode).toBe('development');
  });

  it('validates mode enum values', () => {
    const validModes: EnvConfig['mode'][] = ['development', 'production', 'test'];
    expect(validModes).toContain('development');
    expect(validModes).toContain('production');
    expect(validModes).toContain('test');
  });
});
