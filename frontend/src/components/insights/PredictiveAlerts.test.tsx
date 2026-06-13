/**
 * PredictiveAlerts Tests
 * ======================
 *
 * Red-first guard for the fake-alerts finding: the widget formerly fell
 * back to SAMPLE_ALERTS (fabricated critical "Model Drift Detected - SE
 * Region", AUC 0.82 warning, etc.) whenever the alerts response was empty
 * OR the query failed — a silent fail-open showing fake critical alerts.
 *
 * Desired behavior: real alerts, honest empty ("no active alerts"), or a
 * labeled error. Never SAMPLE_ALERTS.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PredictiveAlerts } from './PredictiveAlerts';
import * as useMonitoring from '@/hooks/api/use-monitoring';
import { AlertStatus } from '@/types/monitoring';
import type { AlertListResponse } from '@/types/monitoring';

vi.mock('@/hooks/api/use-monitoring');

type AlertsQuery = ReturnType<typeof useMonitoring.useAlerts>;

function mockAlerts(overrides: Partial<AlertsQuery> = {}) {
  vi.mocked(useMonitoring.useAlerts).mockReturnValue({
    data: undefined,
    isLoading: false,
    isError: false,
    error: null,
    ...overrides,
  } as unknown as AlertsQuery);
}

const EMPTY_RESPONSE: AlertListResponse = {
  total_count: 0,
  active_count: 0,
  alerts: [],
};

const REAL_RESPONSE: AlertListResponse = {
  total_count: 1,
  active_count: 1,
  alerts: [
    {
      id: 'al_1',
      model_version: 'churn_v3.0.0',
      alert_type: 'drift',
      severity: 'critical',
      title: 'PSI drift on payer_mix',
      description: 'Population stability index exceeded threshold.',
      status: AlertStatus.ACTIVE,
      triggered_at: '2026-06-12T01:00:00Z',
    },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
  mockAlerts();
});

describe('PredictiveAlerts — no SAMPLE_ALERTS fallback', () => {
  it('renders an honest empty state when the API returns zero alerts', () => {
    mockAlerts({ data: EMPTY_RESPONSE } as unknown as Partial<AlertsQuery>);
    render(<PredictiveAlerts />);

    // Fabricated alerts must never render.
    expect(
      screen.queryByText(/Model Drift Detected - SE Region/)
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(/New High-Value Segment Identified/)
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/Current AUC: 0\.82/)).not.toBeInTheDocument();

    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    expect(screen.getByText(/no active alerts/i)).toBeInTheDocument();
  });

  it('renders a labeled error state when the alerts query fails (no fake fallback)', () => {
    mockAlerts({
      isError: true,
      error: new Error('monitoring service unavailable'),
    } as unknown as Partial<AlertsQuery>);
    render(<PredictiveAlerts />);

    expect(
      screen.queryByText(/Model Drift Detected - SE Region/)
    ).not.toBeInTheDocument();
    expect(screen.getByText(/unable to load alerts/i)).toBeInTheDocument();
  });

  it('renders real alerts from the API', () => {
    mockAlerts({ data: REAL_RESPONSE } as unknown as Partial<AlertsQuery>);
    render(<PredictiveAlerts />);

    expect(screen.getByText('PSI drift on payer_mix')).toBeInTheDocument();
    expect(
      screen.queryByText(/Model Drift Detected - SE Region/)
    ).not.toBeInTheDocument();
  });

  it('shows skeletons while loading', () => {
    mockAlerts({ isLoading: true } as unknown as Partial<AlertsQuery>);
    const { container } = render(<PredictiveAlerts />);
    expect(container.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0);
  });

  // #26: alert.severity carries the DriftSeverity vocabulary (high/medium/low),
  // not just critical/warning/info. The old local map silently collapsed
  // "high" -> "info" (blue), under-stating a high-severity drift alert. The
  // shared mapper must render it as "Warning".
  it('renders a backend "high" drift severity as a Warning (not Info)', () => {
    const HIGH_DRIFT: AlertListResponse = {
      total_count: 1,
      active_count: 1,
      alerts: [
        {
          id: 'al_high',
          model_version: 'churn_v3.0.0',
          alert_type: 'drift',
          severity: 'high',
          title: 'Feature drift on region',
          description: 'High drift detected.',
          status: AlertStatus.ACTIVE,
          triggered_at: '2026-06-12T01:00:00Z',
        },
      ],
    };
    mockAlerts({ data: HIGH_DRIFT } as unknown as Partial<AlertsQuery>);
    render(<PredictiveAlerts />);

    expect(screen.getByText('Feature drift on region')).toBeInTheDocument();
    expect(screen.getByText('Warning')).toBeInTheDocument();
    expect(screen.queryByText('Info')).not.toBeInTheDocument();
  });
});
