/**
 * Monitoring API Client Tests
 * ===========================
 *
 * Unit tests for the monitoring API client functions.
 * Uses MSW to mock API responses.
 */

import { describe, it, expect } from 'vitest';
import {
  triggerDriftDetection,
  getDriftDetectionStatus,
  getLatestDriftStatus,
  getDriftHistory,
  listAlerts,
  getAlert,
  updateAlert,
  listMonitoringRuns,
  getModelHealth,
  recordPerformance,
  getPerformanceTrend,
  getPerformanceAlerts,
  compareModelPerformance,
  triggerProductionSweep,
  evaluateRetrainingNeed,
  triggerRetraining,
  getRetrainingStatus,
  completeRetraining,
  rollbackRetraining,
  triggerRetrainingSweep,
} from './monitoring';
import { AlertAction, DriftSeverity, TriggerReason } from '@/types/monitoring';

describe('Monitoring API Client', () => {
  describe('Drift Detection', () => {
    it('triggerDriftDetection - should trigger drift detection', async () => {
      const result = await triggerDriftDetection({
        model_id: 'propensity_v2.1.0',
        time_window: '7d',
        check_data_drift: true,
      });

      expect(result).toBeDefined();
      expect(result.model_id).toBeDefined();
    });

    it('getDriftDetectionStatus - should get task status', async () => {
      const result = await getDriftDetectionStatus('task_abc123');

      expect(result).toBeDefined();
      expect(result.status).toBeDefined();
    });

    it('getLatestDriftStatus - should get latest drift status', async () => {
      const result = await getLatestDriftStatus('propensity_v2.1.0');

      expect(result).toBeDefined();
      expect(result.model_id).toBeDefined();
    });

    it('getDriftHistory - should fetch drift history', async () => {
      const result = await getDriftHistory({ model_id: 'propensity_v2.1.0' });

      expect(result).toBeDefined();
      expect(result.model_id).toBeDefined();
    });

    it('getDriftHistory - should filter by feature name', async () => {
      const result = await getDriftHistory({
        model_id: 'propensity_v2.1.0',
        feature_name: 'days_since_visit',
        days: 30,
      });

      expect(result).toBeDefined();
      expect(result.model_id).toBeDefined();
    });
  });

  describe('Alerts', () => {
    it('listAlerts - should fetch alerts list', async () => {
      const result = await listAlerts();

      expect(result).toBeDefined();
      expect(Array.isArray(result.alerts)).toBe(true);
    });

    it('listAlerts - should filter by severity', async () => {
      const result = await listAlerts({ severity: DriftSeverity.CRITICAL });

      expect(result).toBeDefined();
      expect(Array.isArray(result.alerts)).toBe(true);
    });

    it('getAlert - should fetch specific alert', async () => {
      const result = await getAlert('alert-uuid-123');

      expect(result).toBeDefined();
      expect(result.id).toBeDefined();
    });

    it('updateAlert - should acknowledge alert', async () => {
      const result = await updateAlert('alert-uuid-123', {
        action: AlertAction.ACKNOWLEDGE,
        user_id: 'user_123',
        notes: 'Investigating',
      });

      expect(result).toBeDefined();
      expect(result.id).toBeDefined();
    });
  });

  describe('Monitoring Runs', () => {
    it('listMonitoringRuns - should list runs', async () => {
      const result = await listMonitoringRuns();

      expect(result).toBeDefined();
      expect(Array.isArray(result.runs)).toBe(true);
    });

    it('listMonitoringRuns - should filter by model', async () => {
      const result = await listMonitoringRuns({
        model_id: 'propensity_v2.1.0',
        days: 7,
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.runs)).toBe(true);
    });
  });

  describe('Model Health', () => {
    it('getModelHealth - should fetch model health summary', async () => {
      const result = await getModelHealth('propensity_v2.1.0');

      expect(result).toBeDefined();
      expect(result.model_id).toBeDefined();
      expect(result.overall_health).toBeDefined();
    });
  });

  describe('Performance Tracking', () => {
    it('recordPerformance - should record metrics', async () => {
      const result = await recordPerformance({
        model_id: 'propensity_v2.1.0',
        predictions: [1, 0, 1, 1, 0],
        actuals: [1, 0, 1, 0, 0],
        prediction_scores: [0.85, 0.23, 0.91, 0.67, 0.12],
      });

      expect(result).toBeDefined();
      expect(result.model_id).toBeDefined();
    });

    it('getPerformanceTrend - should fetch performance trend', async () => {
      const result = await getPerformanceTrend({
        model_id: 'propensity_v2.1.0',
        metric_name: 'accuracy',
      });

      expect(result).toBeDefined();
      expect(result.model_id).toBeDefined();
      expect(result.trend).toBeDefined();
    });

    it('getPerformanceAlerts - should fetch performance alerts', async () => {
      const result = await getPerformanceAlerts('propensity_v2.1.0');

      expect(result).toBeDefined();
      expect(Array.isArray(result.alerts)).toBe(true);
    });

    it('compareModelPerformance - should compare models', async () => {
      const result = await compareModelPerformance(
        'propensity_v2.1.0',
        'propensity_v2.0.0',
        'accuracy'
      );

      expect(result).toBeDefined();
      expect(result.better_model).toBeDefined();
    });
  });

  describe('Production Sweep', () => {
    it('triggerProductionSweep - should trigger sweep', async () => {
      const result = await triggerProductionSweep('7d');

      expect(result).toBeDefined();
      expect(result.task_id).toBeDefined();
    });
  });

  describe('Retraining', () => {
    it('evaluateRetrainingNeed - should evaluate retraining need', async () => {
      const result = await evaluateRetrainingNeed('propensity_v2.1.0');

      expect(result).toBeDefined();
      expect(result.should_retrain).toBeDefined();
    });

    it('triggerRetraining - should trigger retraining', async () => {
      const result = await triggerRetraining('propensity_v2.1.0', {
        reason: TriggerReason.DATA_DRIFT,
        notes: 'Significant drift detected',
        auto_approve: false,
      });

      expect(result).toBeDefined();
      expect(result.job_id).toBeDefined();
    });

    it('getRetrainingStatus - should get status', async () => {
      const result = await getRetrainingStatus('job-uuid-123');

      expect(result).toBeDefined();
      expect(result.job_id).toBeDefined();
    });

    it('completeRetraining - should mark complete', async () => {
      const result = await completeRetraining('job-uuid-123', {
        performance_after: 0.92,
        success: true,
        notes: 'Model retrained successfully',
      });

      expect(result).toBeDefined();
      expect(result.job_id).toBeDefined();
    });

    it('rollbackRetraining - should rollback', async () => {
      const result = await rollbackRetraining('job-uuid-123', {
        reason: 'Performance degradation',
      });

      expect(result).toBeDefined();
      expect(result.job_id).toBeDefined();
    });

    it('triggerRetrainingSweep - should trigger sweep', async () => {
      const result = await triggerRetrainingSweep();

      expect(result).toBeDefined();
      expect(result.task_id).toBeDefined();
    });
  });
});
