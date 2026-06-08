import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createElement, type ReactNode } from 'react';
import { useExecutiveInsights } from './use-executive-insights';
import * as api from '@/api/executive-insights';

vi.mock('@/api/executive-insights');

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return createElement(QueryClientProvider, { client }, children);
}

describe('useExecutiveInsights', () => {
  beforeEach(() => vi.clearAllMocks());

  it('fetches crystallized insights for a brand', async () => {
    vi.mocked(api.listExecutiveInsights).mockResolvedValue([
      {
        insight_id: 'ei_1',
        title: 'TRx uplift',
        narrative: 'Detailing frequency is the strongest driver.',
        brand: 'Remibrutinib',
        crystallized_at: '2026-06-08T00:00:00Z',
        source_count: 3,
        effect_size: 0.42,
        effect_direction: 'positive',
      },
    ]);

    const { result } = renderHook(() => useExecutiveInsights('Remibrutinib'), {
      wrapper,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.[0].title).toBe('TRx uplift');
    expect(api.listExecutiveInsights).toHaveBeenCalledWith({
      brand: 'Remibrutinib',
    });
  });

  it('is disabled when brand is empty', () => {
    const { result } = renderHook(() => useExecutiveInsights(''), { wrapper });
    expect(result.current.fetchStatus).toBe('idle');
    expect(api.listExecutiveInsights).not.toHaveBeenCalled();
  });
});
