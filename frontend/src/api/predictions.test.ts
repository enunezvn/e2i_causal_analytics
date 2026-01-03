/**
 * Predictions API Client Tests
 * ============================
 *
 * Unit tests for the predictions API client functions.
 * Uses MSW to mock API responses.
 */

import { describe, it, expect } from 'vitest';
import {
  predict,
  predictBatch,
  getModelHealth,
  getModelInfo,
  getModelsStatus,
} from './predictions';

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
  });
});
