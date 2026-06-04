/**
 * Predictions API Client Tests
 * ============================
 *
 * Unit tests for the predictions API client functions.
 * Uses MSW to mock API responses.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import {
  predict,
  predictBatch,
  getModelHealth,
  getModelInfo,
  getModelsStatus,
} from './predictions';
import { server } from '@/mocks/server';
import { env } from '@/config/env';
import { ApiValidationError } from '@/lib/api-client';

describe('Predictions API Client', () => {
  describe('predict', () => {
    it('should make a single prediction', async () => {
      const result = await predict('churn_model', {
        features: { hcp_id: 'HCP001', territory: 'Northeast' },
      });

      expect(result).toBeDefined();
      expect(result.model_name).toBe('churn_model');
      expect(result.prediction).toBeDefined();
      expect(result.latency_ms).toBeDefined();
    });

    it('should return probabilities when requested', async () => {
      const result = await predict('churn_model', {
        features: { hcp_id: 'HCP001' },
        return_probabilities: true,
      });

      expect(result).toBeDefined();
      expect(result.probabilities).toBeDefined();
    });
  });

  describe('predictBatch', () => {
    it('should make batch predictions', async () => {
      const result = await predictBatch('churn_model', {
        instances: [
          { features: { hcp_id: 'HCP001' } },
          { features: { hcp_id: 'HCP002' } },
        ],
      });

      expect(result).toBeDefined();
      expect(result.model_name).toBe('churn_model');
      expect(result.total_count).toBeDefined();
      expect(result.success_count).toBeDefined();
      expect(Array.isArray(result.predictions)).toBe(true);
    });
  });

  describe('getModelHealth', () => {
    it('should fetch model health status', async () => {
      const result = await getModelHealth('churn_model');

      expect(result).toBeDefined();
      expect(result.model_name).toBe('churn_model');
      expect(result.status).toBeDefined();
      expect(result.endpoint).toBeDefined();
    });
  });

  describe('getModelInfo', () => {
    it('should fetch model metadata', async () => {
      const result = await getModelInfo('churn_model');

      expect(result).toBeDefined();
      expect(result.name).toBe('churn_model');
    });

    it('should include model type and version', async () => {
      const result = await getModelInfo('churn_model');

      expect(result.type).toBeDefined();
      expect(result.version).toBeDefined();
    });
  });

  describe('getModelsStatus', () => {
    it('should fetch all models status', async () => {
      const result = await getModelsStatus();

      expect(result).toBeDefined();
      expect(result.total_models).toBeDefined();
      expect(result.healthy_count).toBeDefined();
      expect(result.unhealthy_count).toBeDefined();
      expect(Array.isArray(result.models)).toBe(true);
    });

    it('should filter specific models', async () => {
      const result = await getModelsStatus(['churn_model', 'conversion_model']);

      expect(result).toBeDefined();
      expect(result.models.length).toBeLessThanOrEqual(2);
    });

    it('serializes `models` as REPEATED keys, not a comma-joined string', async () => {
      // Backend reads `models: Optional[List[str]] = Query(...)` (predictions.py
      // models_status). A comma-joined `?models=a,b` would be parsed as ONE
      // model named "a,b"; the array must be sent as `?models=a&models=b`.
      let capturedUrl = '';
      server.use(
        http.get(`${env.apiUrl}/models/status`, ({ request }) => {
          capturedUrl = request.url;
          return HttpResponse.json({
            total_models: 2,
            healthy_count: 2,
            unhealthy_count: 0,
            models: [
              {
                model_name: 'churn_model',
                status: 'healthy',
                endpoint: 'http://x/churn',
                last_check: new Date().toISOString(),
              },
              {
                model_name: 'conversion_model',
                status: 'healthy',
                endpoint: 'http://x/conversion',
                last_check: new Date().toISOString(),
              },
            ],
            timestamp: new Date().toISOString(),
          });
        })
      );

      await getModelsStatus(['churn_model', 'conversion_model']);

      const params = new URL(capturedUrl).searchParams.getAll('models');
      expect(params).toEqual(['churn_model', 'conversion_model']);
      // The old `.join(',')` bug produced a single `models=churn_model,conversion_model`.
      expect(params).not.toEqual(['churn_model,conversion_model']);
      expect(new URL(capturedUrl).search).not.toContain('models%5B%5D');
    });
  });

  // ===========================================================================
  // RESPONSE VALIDATION (C31)
  // ===========================================================================
  describe('response validation (C31)', () => {
    it('getModelsStatus passes a valid response through (schema wired)', async () => {
      const result = await getModelsStatus();
      expect(Array.isArray(result.models)).toBe(true);
      expect(result.total_models).toBeDefined();
    });

    it('getModelsStatus throws ApiValidationError on a malformed response', async () => {
      server.use(
        http.get(`${env.apiUrl}/models/status`, () =>
          HttpResponse.json({
            total_models: 'lots', // wrong type
            models: 'not-an-array',
          })
        )
      );

      await expect(getModelsStatus()).rejects.toBeInstanceOf(ApiValidationError);
    });

    it('getModelHealth throws ApiValidationError on a malformed response', async () => {
      server.use(
        http.get(`${env.apiUrl}/models/:modelName/health`, () =>
          HttpResponse.json({ status: 'healthy' }) // missing model_name/endpoint/last_check
        )
      );

      await expect(getModelHealth('churn_model')).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });

    it('getModelInfo passes through a BentoML /metadata payload that omits "name" (route has no response_model → intentionally NOT validated)', async () => {
      // The backend GET /models/{name}/info route declares NO response_model: it
      // returns the BentoML /metadata JSON verbatim, which does not guarantee a
      // `name` key. Validating it with a `name`-required schema would false-reject
      // a perfectly valid 200 response and break useModelInfo in production.
      server.use(
        http.get(`${env.apiUrl}/models/:modelName/info`, () =>
          HttpResponse.json({ version: '1.2.0', type: 'sklearn' })
        )
      );

      await expect(getModelInfo('churn_model')).resolves.toBeDefined();
    });
  });
});
