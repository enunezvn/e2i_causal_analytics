import type { ReactNode } from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { KpiTrendChart } from './KpiTrendChart';
import type { KPIHistoryResponse } from '@/types/kpi';

// recharts ResponsiveContainer needs a real size; stub it so the chart renders
// in jsdom (same approach as GapAnalysis.test.tsx).
vi.mock('recharts', async () => {
  const actual = await vi.importActual('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
  };
});

describe('KpiTrendChart', () => {
  it('shows a loading state while the action handler is fetching', () => {
    render(<KpiTrendChart kpiId="trx" loading />);
    expect(screen.getByText(/Loading trx trend/i)).toBeInTheDocument();
    expect(screen.queryByTestId('kpi-trend-chart')).not.toBeInTheDocument();
  });

  it('shows an explicit "couldn\'t load" state when there is no result (distinct from empty)', () => {
    // Not loading and no data => the fetch failed / id was invalid.
    render(<KpiTrendChart kpiId="bogus" />);
    expect(screen.getByText(/Couldn’t load the trend/i)).toBeInTheDocument();
    expect(screen.queryByText(/No historical series/i)).not.toBeInTheDocument();
    expect(screen.queryByTestId('kpi-trend-chart')).not.toBeInTheDocument();
  });

  it('renders an honest empty state when no real series exists (never fabricates points)', () => {
    const data: KPIHistoryResponse = {
      kpi_id: 'trx',
      brand: '',
      region: '',
      count: 0,
      points: [],
    };
    render(<KpiTrendChart kpiId="trx" data={data} />);
    expect(screen.getByText(/No historical series available/i)).toBeInTheDocument();
    expect(screen.queryByTestId('kpi-trend-chart')).not.toBeInTheDocument();
  });

  it('renders the chart with the scope and point count when real points are present', () => {
    const data: KPIHistoryResponse = {
      kpi_id: 'trx',
      brand: 'Kisqali',
      region: '',
      count: 2,
      points: [
        { metric_date: '2026-01-01', value: 100 },
        { metric_date: '2026-02-01', value: 120 },
      ],
    };
    render(<KpiTrendChart kpiId="trx" title="TRx trend" data={data} />);
    expect(screen.getByText('TRx trend')).toBeInTheDocument();
    expect(screen.getByTestId('kpi-trend-chart')).toBeInTheDocument();
    expect(screen.getByText(/Kisqali/)).toBeInTheDocument();
    expect(screen.getByText(/2 months/)).toBeInTheDocument();
  });
});
