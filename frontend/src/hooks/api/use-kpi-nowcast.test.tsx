/**
 * useKPIHistoryNowcast — Rx-volume family gating (backlog #45, PR-C)
 * ==================================================================
 *
 * The /history/nowcast endpoint serves ONLY the Rx-volume family
 * (WS3-BI-005 TRx, WS3-BI-006 NRx, WS3-BI-007 NBRx) and 422s every other
 * KPI. The hook must therefore HARD-gate the fetch on the family — no
 * options spread may point it at an off-family KPI (spec item e).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as React from 'react';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('@/api/kpi');

import * as kpiApi from '@/api/kpi';
import * as useKpiModule from './use-kpi';

// Runtime lookups so the file stays runnable while the exports are
// unimplemented (red phase) — each test reds on its own assertion.
const useKPIHistoryNowcast = (
  useKpiModule as unknown as Record<string, unknown>
)['useKPIHistoryNowcast'] as
  | ((kpiId: string, brand?: string, options?: { enabled?: boolean }) => unknown)
  | undefined;
const RX_VOLUME_KPI_IDS = (useKpiModule as unknown as Record<string, unknown>)[
  'RX_VOLUME_KPI_IDS'
] as Set<string> | undefined;

const mockGetNowcast = (kpiApi as unknown as Record<string, ReturnType<typeof vi.fn>>)[
  'getKPIHistoryNowcast'
];

const NOWCAST_FIXTURE = {
  kpi_id: 'WS3-BI-005',
  brand: 'Kisqali',
  data_through: '2026-07-21',
  insufficient_maturity: false,
  reason: null,
  mature_months_used: 30,
  anchor_cap_month: '2026-07-01',
  arrival_plane_coverage: 1,
  ci_level: 0.95,
  count: 1,
  points: [
    {
      metric_date: '2026-06-01',
      mature_value: 1322,
      provisional_value: 1057,
      provisional: true,
      completion_factor: 0.8,
      nowcast_value: 1321.25,
      nowcast_ci_lower: 1274,
      nowcast_ci_upper: 1369,
    },
  ],
};

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0, staleTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('RX_VOLUME_KPI_IDS', () => {
  it('pins exactly the three Rx-volume KPI ids the endpoint serves', () => {
    expect(RX_VOLUME_KPI_IDS).toBeInstanceOf(Set);
    expect([...(RX_VOLUME_KPI_IDS ?? [])].sort()).toEqual([
      'WS3-BI-005',
      'WS3-BI-006',
      'WS3-BI-007',
    ]);
  });
});

describe('useKPIHistoryNowcast', () => {
  it('is exported from use-kpi', () => {
    expect(useKPIHistoryNowcast).toBeTypeOf('function');
  });

  it('never fetches for an off-family KPI — even when enabled is forced (e)', async () => {
    renderHook(() => useKPIHistoryNowcast?.('WS3-BI-010', '', { enabled: true }), {
      wrapper: createWrapper(),
    });

    // Give react-query a macrotask to (incorrectly) start any fetch.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(mockGetNowcast).not.toHaveBeenCalled();
  });

  it('fetches for an Rx-volume KPI with the brand param', async () => {
    mockGetNowcast?.mockResolvedValue(NOWCAST_FIXTURE);
    const { result } = renderHook(() => useKPIHistoryNowcast?.('WS3-BI-005', 'Kisqali'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect((result.current as { isSuccess?: boolean } | undefined)?.isSuccess).toBe(true);
    });
    expect(mockGetNowcast).toHaveBeenCalledWith('WS3-BI-005', 'Kisqali');
    expect((result.current as { data?: unknown }).data).toEqual(NOWCAST_FIXTURE);
  });

  it('options.enabled=false suppresses even Rx-family fetches', async () => {
    renderHook(() => useKPIHistoryNowcast?.('WS3-BI-006', '', { enabled: false }), {
      wrapper: createWrapper(),
    });

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(mockGetNowcast).not.toHaveBeenCalled();
  });
});
