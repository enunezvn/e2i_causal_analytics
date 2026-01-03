/**
 * KPI API Client Tests
 * ====================
 *
 * Unit tests for the KPI API client functions.
 * Uses MSW to mock API responses.
 */

import { describe, it, expect } from 'vitest';
import {
  listKPIs,
  getWorkstreams,
  getKPIMetadata,
  getKPIValue,
  calculateKPI,
  batchCalculateKPIs,
  invalidateKPICache,
  getKPIHealth,
} from './kpi';
import { Workstream } from '@/types/kpi';

describe('KPI API Client', () => {
  describe('listKPIs', () => {
    it('should fetch list of KPIs', async () => {
      const result = await listKPIs();

      expect(result).toBeDefined();
      expect(Array.isArray(result.kpis)).toBe(true);
      expect(result.total).toBeDefined();
    });

    it('should filter by workstream', async () => {
      const result = await listKPIs({ workstream: Workstream.WS1_DATA_QUALITY });

      expect(result).toBeDefined();
      expect(Array.isArray(result.kpis)).toBe(true);
    });
  });

  describe('getWorkstreams', () => {
    it('should fetch available workstreams', async () => {
      const result = await getWorkstreams();

      expect(result).toBeDefined();
      expect(Array.isArray(result.workstreams)).toBe(true);
      expect(result.total).toBeDefined();
    });
  });

  describe('getKPIMetadata', () => {
    it('should fetch KPI metadata', async () => {
      const result = await getKPIMetadata('WS1-DQ-001');

      expect(result).toBeDefined();
      expect(result.id).toBe('WS1-DQ-001');
      expect(result.name).toBeDefined();
    });
  });

  describe('getKPIValue', () => {
    it('should fetch KPI calculated value', async () => {
      const result = await getKPIValue('WS1-DQ-001');

      expect(result).toBeDefined();
      expect(result.kpi_id).toBe('WS1-DQ-001');
      expect(result.value).toBeDefined();
    });

    it('should filter by brand', async () => {
      const result = await getKPIValue('WS1-DQ-001', 'remibrutinib');

      expect(result).toBeDefined();
      expect(result.kpi_id).toBe('WS1-DQ-001');
    });
  });

  describe('calculateKPI', () => {
    it('should calculate a KPI', async () => {
      const result = await calculateKPI({
        kpi_id: 'WS1-DQ-001',
        context: { brand: 'remibrutinib' },
      });

      expect(result).toBeDefined();
      expect(result.kpi_id).toBe('WS1-DQ-001');
      expect(result.value).toBeDefined();
    });

    it('should force refresh when requested', async () => {
      const result = await calculateKPI({
        kpi_id: 'WS1-DQ-001',
        force_refresh: true,
      });

      expect(result).toBeDefined();
      expect(result.cached).toBe(false);
    });
  });

  describe('batchCalculateKPIs', () => {
    it('should calculate multiple KPIs', async () => {
      const result = await batchCalculateKPIs({
        kpi_ids: ['WS1-DQ-001', 'WS1-DQ-002'],
      });

      expect(result).toBeDefined();
      expect(result.total_kpis).toBeDefined();
      expect(result.successful).toBeDefined();
      expect(Array.isArray(result.results)).toBe(true);
    });

    it('should calculate by workstream', async () => {
      const result = await batchCalculateKPIs({
        workstream: 'ws1_data_quality',
      });

      expect(result).toBeDefined();
      expect(result.total_kpis).toBeGreaterThan(0);
    });
  });

  describe('invalidateKPICache', () => {
    it('should invalidate specific KPI cache', async () => {
      const result = await invalidateKPICache({
        kpi_id: 'WS1-DQ-001',
      });

      expect(result).toBeDefined();
      expect(result.invalidated_count).toBeDefined();
    });

    it('should invalidate all cache', async () => {
      const result = await invalidateKPICache({
        invalidate_all: true,
      });

      expect(result).toBeDefined();
      expect(result.invalidated_count).toBeGreaterThan(0);
    });
  });

  describe('getKPIHealth', () => {
    it('should fetch KPI system health', async () => {
      const result = await getKPIHealth();

      expect(result).toBeDefined();
      expect(result.status).toBeDefined();
      expect(result.total_kpis).toBeDefined();
    });
  });
});
