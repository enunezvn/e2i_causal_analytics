/**
 * Monitoring API Query Hooks Tests
 * =================================
 *
 * Tests for TanStack Query hooks for the E2I Model Monitoring API.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';
import type {
  DriftDetectionResponse,
  DriftHistoryResponse,
  DriftHistoryItem,
  TaskStatusResponse,
  AlertListResponse,
  AlertItem,
  AlertListParams,
  MonitoringRunsResponse,
  MonitoringRunItem,
  ModelHealthSummary,
  PerformanceTrendResponse,
  PerformanceAlertsResponse,
  ModelComparisonResponse,
  PerformanceRecordResponse,
  ProductionSweepResponse,
  RetrainingDecisionResponse,
  RetrainingJobResponse,
  DriftResult,
  PerformanceMetricItem,
} from '@/types/monitoring';
import {
  AlertStatus,
  AlertAction,
  TriggerReason,
  RetrainingStatus,
  DriftType,
  DriftSeverity,
} from '@/types/monitoring';

// Mock the API functions
vi.mock('@/api/monitoring', () => ({
  triggerDriftDetection: vi.fn(),
  getDriftDetectionStatus: vi.fn(),
  getLatestDriftStatus: vi.fn(),
  getDriftHistory: vi.fn(),
  listAlerts: vi.fn(),
  getAlert: vi.fn(),
  updateAlert: vi.fn(),
  listMonitoringRuns: vi.fn(),
  getModelHealth: vi.fn(),
  recordPerformance: vi.fn(),
  getPerformanceTrend: vi.fn(),
  getPerformanceAlerts: vi.fn(),
  compareModelPerformance: vi.fn(),
  triggerProductionSweep: vi.fn(),
  evaluateRetrainingNeed: vi.fn(),
  triggerRetraining: vi.fn(),
  getRetrainingStatus: vi.fn(),
  completeRetraining: vi.fn(),
  rollbackRetraining: vi.fn(),
  triggerRetrainingSweep: vi.fn(),
}));

// Mock query-client
vi.mock('@/lib/query-client', () => ({
  queryKeys: {
    all: ['e2i'] as const,
    monitoring: {
      all: () => ['e2i', 'monitoring'] as const,
      driftLatest: (modelId: string) => ['e2i', 'monitoring', 'drift', 'latest', modelId] as const,
      driftHistory: (modelId: string) => ['e2i', 'monitoring', 'drift', 'history', modelId] as const,
      driftStatus: (taskId: string) => ['e2i', 'monitoring', 'drift', 'status', taskId] as const,
      alerts: () => ['e2i', 'monitoring', 'alerts'] as const,
      alert: (alertId: string) => ['e2i', 'monitoring', 'alert', alertId] as const,
      runs: () => ['e2i', 'monitoring', 'runs'] as const,
      modelHealth: (modelId: string) => ['e2i', 'monitoring', 'health', modelId] as const,
      performanceTrend: (modelId: string) => ['e2i', 'monitoring', 'performance', 'trend', modelId] as const,
      performanceAlerts: (modelId: string) => ['e2i', 'monitoring', 'performance', 'alerts', modelId] as const,
      performanceCompare: (modelId: string, otherModelId: string) =>
        ['e2i', 'monitoring', 'performance', 'compare', modelId, otherModelId] as const,
      retrainingStatus: (jobId: string) => ['e2i', 'monitoring', 'retraining', jobId] as const,
    },
  },
}));

import {
  useLatestDriftStatus,
  useDriftHistory,
  useDriftTaskStatus,
  useTriggerDriftDetection,
  useAlerts,
  useAlert,
  useUpdateAlert,
  useMonitoringRuns,
  useModelHealth,
  usePerformanceTrend,
  usePerformanceAlerts,
  useModelComparison,
  useRecordPerformance,
  useProductionSweep,
  useEvaluateRetraining,
  useTriggerRetraining,
  useRetrainingStatus,
  useCompleteRetraining,
  useRollbackRetraining,
  useRetrainingSweep,
  prefetchModelHealth,
  prefetchAlerts,
  prefetchLatestDriftStatus,
} from './use-monitoring';
import * as monitoringApi from '@/api/monitoring';

// =============================================================================
// TEST UTILITIES
// =============================================================================

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

function createWrapper() {
  const queryClient = createTestQueryClient();
  return {
    queryClient,
    wrapper: ({ children }: { children: React.ReactNode }) =>
      React.createElement(QueryClientProvider, { client: queryClient }, children),
  };
}

// =============================================================================
// MOCK DATA
// =============================================================================

const mockDriftResult: DriftResult = {
  feature: 'engagement_score',
  drift_type: DriftType.DATA,
  test_statistic: 0.12,
  p_value: 0.08,
  drift_detected: false,
  severity: DriftSeverity.LOW,
  baseline_period: '2024-01-01 to 2024-01-07',
  current_period: '2024-01-08 to 2024-01-15',
};

const mockDriftResponse: DriftDetectionResponse = {
  task_id: 'drift_task_001',
  model_id: 'propensity_v2.1.0',
  status: 'completed',
  overall_drift_score: 0.15,
  features_checked: 10,
  features_with_drift: [],
  results: [mockDriftResult],
  drift_summary: 'No significant drift detected',
  recommended_actions: [],
  detection_latency_ms: 150,
  timestamp: '2024-01-15T12:00:00Z',
};

const mockDriftHistoryItem: DriftHistoryItem = {
  id: 'drift_history_001',
  model_version: 'propensity_v2.1.0',
  feature_name: 'engagement_score',
  drift_type: 'data',
  drift_score: 0.12,
  severity: 'low',
  detected_at: '2024-01-15T12:00:00Z',
  baseline_start: '2024-01-01T00:00:00Z',
  baseline_end: '2024-01-07T23:59:59Z',
  current_start: '2024-01-08T00:00:00Z',
  current_end: '2024-01-15T23:59:59Z',
};

const mockDriftHistoryResponse: DriftHistoryResponse = {
  model_id: 'propensity_v2.1.0',
  total_records: 1,
  records: [mockDriftHistoryItem],
};

const mockTaskStatusResponse: TaskStatusResponse = {
  task_id: 'task_abc123',
  status: 'completed',
  ready: true,
  result: mockDriftResponse,
};

const mockAlertItem: AlertItem = {
  id: 'alert_001',
  model_version: 'propensity_v2.1.0',
  alert_type: 'drift',
  severity: 'warning',
  title: 'Drift Detected',
  description: 'Data drift detected in engagement_score feature',
  status: AlertStatus.ACTIVE,
  triggered_at: '2024-01-15T12:00:00Z',
};

const mockAlertListResponse: AlertListResponse = {
  total_count: 1,
  active_count: 1,
  alerts: [mockAlertItem],
};

const mockMonitoringRunItem: MonitoringRunItem = {
  id: 'run_001',
  model_version: 'propensity_v2.1.0',
  run_type: 'drift_detection',
  started_at: '2024-01-15T12:00:00Z',
  completed_at: '2024-01-15T12:02:00Z',
  features_checked: 10,
  drift_detected_count: 0,
  alerts_generated: 0,
  duration_ms: 120000,
};

const mockMonitoringRunsResponse: MonitoringRunsResponse = {
  model_id: 'propensity_v2.1.0',
  total_runs: 1,
  runs: [mockMonitoringRunItem],
};

const mockModelHealthSummary: ModelHealthSummary = {
  model_id: 'propensity_v2.1.0',
  overall_health: 'healthy',
  last_check: '2024-01-15T12:00:00Z',
  drift_score: 0.15,
  active_alerts: 0,
  performance_trend: 'stable',
  recommendations: [],
};

const mockPerformanceMetricItem: PerformanceMetricItem = {
  metric_name: 'accuracy',
  metric_value: 0.92,
  recorded_at: '2024-01-15T12:00:00Z',
};

const mockPerformanceTrendResponse: PerformanceTrendResponse = {
  model_id: 'propensity_v2.1.0',
  metric_name: 'accuracy',
  current_value: 0.91,
  baseline_value: 0.92,
  change_percent: -1.09,
  trend: 'stable',
  is_significant: false,
  alert_threshold_breached: false,
  history: [mockPerformanceMetricItem],
};

const mockPerformanceAlertsResponse: PerformanceAlertsResponse = {
  model_id: 'propensity_v2.1.0',
  alert_count: 0,
  alerts: [],
};

const mockModelComparisonResponse: ModelComparisonResponse = {
  model_id: 'propensity_v2.1.0',
  other_model_id: 'propensity_v2.0.0',
  metric_name: 'accuracy',
  model_value: 0.92,
  other_model_value: 0.89,
  difference: 0.03,
  difference_percent: 3.37,
  better_model: 'propensity_v2.1.0',
};

const mockPerformanceRecordResponse: PerformanceRecordResponse = {
  model_id: 'propensity_v2.1.0',
  recorded_at: '2024-01-15T12:00:00Z',
  sample_size: 100,
  metrics: { accuracy: 0.92, f1_score: 0.88 },
  alerts_generated: 0,
};

const mockProductionSweepResponse: ProductionSweepResponse = {
  task_id: 'sweep_task_001',
  status: 'completed',
  message: 'Production sweep completed successfully',
  time_window: '7d',
};

const mockRetrainingDecisionResponse: RetrainingDecisionResponse = {
  model_id: 'propensity_v2.1.0',
  should_retrain: true,
  confidence: 0.85,
  reasons: ['Data drift detected', 'Performance degradation observed'],
  trigger_factors: { drift_score: 0.35, accuracy_drop: 0.05 },
  cooldown_active: false,
  recommended_action: 'Schedule retraining within 7 days',
};

const mockRetrainingJobResponse: RetrainingJobResponse = {
  job_id: 'retrain_job_001',
  model_version: 'propensity_v2.1.0',
  status: RetrainingStatus.PENDING,
  trigger_reason: 'data_drift',
  triggered_at: '2024-01-15T12:00:00Z',
  triggered_by: 'ui_user',
};

// =============================================================================
// DRIFT DETECTION HOOK TESTS
// =============================================================================

describe('useLatestDriftStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches latest drift status successfully', async () => {
    vi.mocked(monitoringApi.getLatestDriftStatus).mockResolvedValueOnce(mockDriftResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useLatestDriftStatus('propensity_v2.1.0'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockDriftResponse);
    expect(monitoringApi.getLatestDriftStatus).toHaveBeenCalledWith('propensity_v2.1.0', 10);
  });

  it('passes limit parameter', async () => {
    vi.mocked(monitoringApi.getLatestDriftStatus).mockResolvedValueOnce(mockDriftResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useLatestDriftStatus('propensity_v2.1.0', 5), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.getLatestDriftStatus).toHaveBeenCalledWith('propensity_v2.1.0', 5);
  });

  it('is disabled when modelId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useLatestDriftStatus(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(monitoringApi.getLatestDriftStatus).not.toHaveBeenCalled();
  });
});

describe('useDriftHistory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches drift history successfully', async () => {
    vi.mocked(monitoringApi.getDriftHistory).mockResolvedValueOnce(mockDriftHistoryResponse);
    const { wrapper } = createWrapper();
    const params = { model_id: 'propensity_v2.1.0', days: 30 };

    const { result } = renderHook(() => useDriftHistory(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockDriftHistoryResponse);
    expect(monitoringApi.getDriftHistory).toHaveBeenCalledWith(params);
  });

  it('is disabled when model_id is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useDriftHistory({ model_id: '' }), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(monitoringApi.getDriftHistory).not.toHaveBeenCalled();
  });
});

describe('useDriftTaskStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches task status successfully', async () => {
    vi.mocked(monitoringApi.getDriftDetectionStatus).mockResolvedValueOnce(mockTaskStatusResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useDriftTaskStatus('task_abc123'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockTaskStatusResponse);
    expect(monitoringApi.getDriftDetectionStatus).toHaveBeenCalledWith('task_abc123');
  });

  it('is disabled when taskId is null', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useDriftTaskStatus(null), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(monitoringApi.getDriftDetectionStatus).not.toHaveBeenCalled();
  });
});

describe('useTriggerDriftDetection', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('triggers drift detection successfully', async () => {
    vi.mocked(monitoringApi.triggerDriftDetection).mockResolvedValueOnce(mockDriftResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useTriggerDriftDetection(), { wrapper });

    result.current.mutate({
      request: { model_id: 'propensity_v2.1.0', time_window: '7d' },
      asyncMode: true,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.triggerDriftDetection).toHaveBeenCalledWith(
      { model_id: 'propensity_v2.1.0', time_window: '7d' },
      true
    );
    expect(invalidateSpy).toHaveBeenCalled();
  });
});

// =============================================================================
// ALERT HOOK TESTS
// =============================================================================

describe('useAlerts', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches alerts successfully', async () => {
    vi.mocked(monitoringApi.listAlerts).mockResolvedValueOnce(mockAlertListResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useAlerts(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockAlertListResponse);
    expect(monitoringApi.listAlerts).toHaveBeenCalledWith(undefined);
  });

  it('passes params to API', async () => {
    vi.mocked(monitoringApi.listAlerts).mockResolvedValueOnce(mockAlertListResponse);
    const { wrapper } = createWrapper();
    const params: AlertListParams = { status: AlertStatus.ACTIVE, severity: DriftSeverity.CRITICAL };

    const { result } = renderHook(() => useAlerts(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.listAlerts).toHaveBeenCalledWith(params);
  });
});

describe('useAlert', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches single alert successfully', async () => {
    vi.mocked(monitoringApi.getAlert).mockResolvedValueOnce(mockAlertItem);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useAlert('alert_001'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockAlertItem);
    expect(monitoringApi.getAlert).toHaveBeenCalledWith('alert_001');
  });

  it('is disabled when alertId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useAlert(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(monitoringApi.getAlert).not.toHaveBeenCalled();
  });
});

describe('useUpdateAlert', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('updates alert successfully', async () => {
    const updatedAlert: AlertItem = { ...mockAlertItem, status: AlertStatus.ACKNOWLEDGED };
    vi.mocked(monitoringApi.updateAlert).mockResolvedValueOnce(updatedAlert);
    const { wrapper, queryClient } = createWrapper();
    const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useUpdateAlert(), { wrapper });

    result.current.mutate({
      alertId: 'alert_001',
      request: { action: AlertAction.ACKNOWLEDGE, user_id: 'user_123' },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.updateAlert).toHaveBeenCalledWith('alert_001', {
      action: AlertAction.ACKNOWLEDGE,
      user_id: 'user_123',
    });
    expect(setQueryDataSpy).toHaveBeenCalled();
    expect(invalidateSpy).toHaveBeenCalled();
  });
});

// =============================================================================
// MONITORING RUNS HOOK TESTS
// =============================================================================

describe('useMonitoringRuns', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches monitoring runs successfully', async () => {
    vi.mocked(monitoringApi.listMonitoringRuns).mockResolvedValueOnce(mockMonitoringRunsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useMonitoringRuns(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockMonitoringRunsResponse);
  });

  it('passes params to API', async () => {
    vi.mocked(monitoringApi.listMonitoringRuns).mockResolvedValueOnce(mockMonitoringRunsResponse);
    const { wrapper } = createWrapper();
    const params = { model_id: 'propensity_v2.1.0', days: 7 };

    const { result } = renderHook(() => useMonitoringRuns(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.listMonitoringRuns).toHaveBeenCalledWith(params);
  });
});

// =============================================================================
// MODEL HEALTH HOOK TESTS
// =============================================================================

describe('useModelHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches model health successfully', async () => {
    vi.mocked(monitoringApi.getModelHealth).mockResolvedValueOnce(mockModelHealthSummary);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelHealth('propensity_v2.1.0'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockModelHealthSummary);
    expect(monitoringApi.getModelHealth).toHaveBeenCalledWith('propensity_v2.1.0');
  });

  it('is disabled when modelId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelHealth(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(monitoringApi.getModelHealth).not.toHaveBeenCalled();
  });
});

// =============================================================================
// PERFORMANCE TRACKING HOOK TESTS
// =============================================================================

describe('usePerformanceTrend', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches performance trend successfully', async () => {
    vi.mocked(monitoringApi.getPerformanceTrend).mockResolvedValueOnce(mockPerformanceTrendResponse);
    const { wrapper } = createWrapper();
    const params = { model_id: 'propensity_v2.1.0', metric_name: 'accuracy', days: 30 };

    const { result } = renderHook(() => usePerformanceTrend(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockPerformanceTrendResponse);
    expect(monitoringApi.getPerformanceTrend).toHaveBeenCalledWith(params);
  });

  it('is disabled when model_id is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(
      () => usePerformanceTrend({ model_id: '', metric_name: 'accuracy' }),
      { wrapper }
    );

    expect(result.current.fetchStatus).toBe('idle');
    expect(monitoringApi.getPerformanceTrend).not.toHaveBeenCalled();
  });
});

describe('usePerformanceAlerts', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches performance alerts successfully', async () => {
    vi.mocked(monitoringApi.getPerformanceAlerts).mockResolvedValueOnce(mockPerformanceAlertsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => usePerformanceAlerts('propensity_v2.1.0'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockPerformanceAlertsResponse);
    expect(monitoringApi.getPerformanceAlerts).toHaveBeenCalledWith('propensity_v2.1.0');
  });

  it('is disabled when modelId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => usePerformanceAlerts(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(monitoringApi.getPerformanceAlerts).not.toHaveBeenCalled();
  });
});

describe('useModelComparison', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('compares models successfully', async () => {
    vi.mocked(monitoringApi.compareModelPerformance).mockResolvedValueOnce(mockModelComparisonResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(
      () => useModelComparison('propensity_v2.1.0', 'propensity_v2.0.0', 'accuracy'),
      { wrapper }
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockModelComparisonResponse);
    expect(monitoringApi.compareModelPerformance).toHaveBeenCalledWith(
      'propensity_v2.1.0',
      'propensity_v2.0.0',
      'accuracy'
    );
  });

  it('uses default metric name', async () => {
    vi.mocked(monitoringApi.compareModelPerformance).mockResolvedValueOnce(mockModelComparisonResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(
      () => useModelComparison('propensity_v2.1.0', 'propensity_v2.0.0'),
      { wrapper }
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.compareModelPerformance).toHaveBeenCalledWith(
      'propensity_v2.1.0',
      'propensity_v2.0.0',
      'accuracy'
    );
  });

  it('is disabled when modelId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(
      () => useModelComparison('', 'propensity_v2.0.0'),
      { wrapper }
    );

    expect(result.current.fetchStatus).toBe('idle');
    expect(monitoringApi.compareModelPerformance).not.toHaveBeenCalled();
  });

  it('is disabled when otherModelId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(
      () => useModelComparison('propensity_v2.1.0', ''),
      { wrapper }
    );

    expect(result.current.fetchStatus).toBe('idle');
    expect(monitoringApi.compareModelPerformance).not.toHaveBeenCalled();
  });
});

describe('useRecordPerformance', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('records performance successfully', async () => {
    vi.mocked(monitoringApi.recordPerformance).mockResolvedValueOnce(mockPerformanceRecordResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useRecordPerformance(), { wrapper });

    result.current.mutate({
      request: {
        model_id: 'propensity_v2.1.0',
        predictions: [1, 0, 1],
        actuals: [1, 0, 0],
      },
      asyncMode: true,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.recordPerformance).toHaveBeenCalled();
    expect(invalidateSpy).toHaveBeenCalled();
  });
});

// =============================================================================
// PRODUCTION SWEEP HOOK TESTS
// =============================================================================

describe('useProductionSweep', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('triggers production sweep successfully', async () => {
    vi.mocked(monitoringApi.triggerProductionSweep).mockResolvedValueOnce(mockProductionSweepResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useProductionSweep(), { wrapper });

    result.current.mutate({ timeWindow: '7d' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockProductionSweepResponse);
    expect(monitoringApi.triggerProductionSweep).toHaveBeenCalledWith('7d');
  });

  it('uses default time window', async () => {
    vi.mocked(monitoringApi.triggerProductionSweep).mockResolvedValueOnce(mockProductionSweepResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useProductionSweep(), { wrapper });

    result.current.mutate({});

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.triggerProductionSweep).toHaveBeenCalledWith('7d');
  });
});

// =============================================================================
// RETRAINING HOOK TESTS
// =============================================================================

describe('useEvaluateRetraining', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('evaluates retraining need successfully', async () => {
    vi.mocked(monitoringApi.evaluateRetrainingNeed).mockResolvedValueOnce(mockRetrainingDecisionResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useEvaluateRetraining(), { wrapper });

    result.current.mutate({ modelId: 'propensity_v2.1.0' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockRetrainingDecisionResponse);
    expect(monitoringApi.evaluateRetrainingNeed).toHaveBeenCalledWith('propensity_v2.1.0');
  });
});

describe('useTriggerRetraining', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('triggers retraining successfully', async () => {
    vi.mocked(monitoringApi.triggerRetraining).mockResolvedValueOnce(mockRetrainingJobResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useTriggerRetraining(), { wrapper });

    result.current.mutate({
      modelId: 'propensity_v2.1.0',
      request: { reason: TriggerReason.DATA_DRIFT },
      triggeredBy: 'ui_user',
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.triggerRetraining).toHaveBeenCalledWith(
      'propensity_v2.1.0',
      { reason: TriggerReason.DATA_DRIFT },
      'ui_user'
    );
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('uses default triggeredBy', async () => {
    vi.mocked(monitoringApi.triggerRetraining).mockResolvedValueOnce(mockRetrainingJobResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useTriggerRetraining(), { wrapper });

    result.current.mutate({
      modelId: 'propensity_v2.1.0',
      request: { reason: TriggerReason.DATA_DRIFT },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.triggerRetraining).toHaveBeenCalledWith(
      'propensity_v2.1.0',
      { reason: TriggerReason.DATA_DRIFT },
      'ui_user'
    );
  });
});

describe('useRetrainingStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches retraining status successfully', async () => {
    vi.mocked(monitoringApi.getRetrainingStatus).mockResolvedValueOnce(mockRetrainingJobResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRetrainingStatus('retrain_job_001'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockRetrainingJobResponse);
    expect(monitoringApi.getRetrainingStatus).toHaveBeenCalledWith('retrain_job_001');
  });

  it('is disabled when jobId is null', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRetrainingStatus(null), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(monitoringApi.getRetrainingStatus).not.toHaveBeenCalled();
  });
});

describe('useCompleteRetraining', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('completes retraining successfully', async () => {
    const completedJob: RetrainingJobResponse = { ...mockRetrainingJobResponse, status: RetrainingStatus.COMPLETED };
    vi.mocked(monitoringApi.completeRetraining).mockResolvedValueOnce(completedJob);
    const { wrapper, queryClient } = createWrapper();
    const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useCompleteRetraining(), { wrapper });

    result.current.mutate({
      jobId: 'retrain_job_001',
      request: { performance_after: 0.92, success: true },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.completeRetraining).toHaveBeenCalledWith('retrain_job_001', {
      performance_after: 0.92,
      success: true,
    });
    expect(setQueryDataSpy).toHaveBeenCalled();
    expect(invalidateSpy).toHaveBeenCalled();
  });
});

describe('useRollbackRetraining', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('rolls back retraining successfully', async () => {
    const rolledBackJob: RetrainingJobResponse = { ...mockRetrainingJobResponse, status: RetrainingStatus.ROLLED_BACK };
    vi.mocked(monitoringApi.rollbackRetraining).mockResolvedValueOnce(rolledBackJob);
    const { wrapper, queryClient } = createWrapper();
    const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useRollbackRetraining(), { wrapper });

    result.current.mutate({
      jobId: 'retrain_job_001',
      request: { reason: 'Performance degradation' },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(monitoringApi.rollbackRetraining).toHaveBeenCalledWith('retrain_job_001', {
      reason: 'Performance degradation',
    });
    expect(setQueryDataSpy).toHaveBeenCalled();
    expect(invalidateSpy).toHaveBeenCalled();
  });
});

describe('useRetrainingSweep', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('triggers retraining sweep successfully', async () => {
    vi.mocked(monitoringApi.triggerRetrainingSweep).mockResolvedValueOnce(mockProductionSweepResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRetrainingSweep(), { wrapper });

    result.current.mutate();

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockProductionSweepResponse);
    expect(monitoringApi.triggerRetrainingSweep).toHaveBeenCalled();
  });
});

// =============================================================================
// PREFETCH HELPER TESTS
// =============================================================================

describe('prefetchModelHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches model health', async () => {
    vi.mocked(monitoringApi.getModelHealth).mockResolvedValueOnce(mockModelHealthSummary);
    const queryClient = createTestQueryClient();

    await prefetchModelHealth(queryClient, 'propensity_v2.1.0');

    expect(monitoringApi.getModelHealth).toHaveBeenCalledWith('propensity_v2.1.0');
  });
});

describe('prefetchAlerts', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches alerts', async () => {
    vi.mocked(monitoringApi.listAlerts).mockResolvedValueOnce(mockAlertListResponse);
    const queryClient = createTestQueryClient();

    await prefetchAlerts(queryClient);

    expect(monitoringApi.listAlerts).toHaveBeenCalledWith(undefined);
  });

  it('prefetches with params', async () => {
    vi.mocked(monitoringApi.listAlerts).mockResolvedValueOnce(mockAlertListResponse);
    const queryClient = createTestQueryClient();
    const params: AlertListParams = { status: AlertStatus.ACTIVE };

    await prefetchAlerts(queryClient, params);

    expect(monitoringApi.listAlerts).toHaveBeenCalledWith(params);
  });
});

describe('prefetchLatestDriftStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches latest drift status', async () => {
    vi.mocked(monitoringApi.getLatestDriftStatus).mockResolvedValueOnce(mockDriftResponse);
    const queryClient = createTestQueryClient();

    await prefetchLatestDriftStatus(queryClient, 'propensity_v2.1.0');

    expect(monitoringApi.getLatestDriftStatus).toHaveBeenCalledWith('propensity_v2.1.0', 10);
  });

  it('prefetches with limit', async () => {
    vi.mocked(monitoringApi.getLatestDriftStatus).mockResolvedValueOnce(mockDriftResponse);
    const queryClient = createTestQueryClient();

    await prefetchLatestDriftStatus(queryClient, 'propensity_v2.1.0', 5);

    expect(monitoringApi.getLatestDriftStatus).toHaveBeenCalledWith('propensity_v2.1.0', 5);
  });
});
