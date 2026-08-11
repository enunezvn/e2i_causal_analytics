import type { ReactNode } from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { KpiTrendChart } from './KpiTrendChart';
import type { KPIHistoryResponse, KPISegmentedHistoryResponse } from '@/types/kpi';

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

  it('names the region in the empty state for a region-scoped miss (#1536)', () => {
    // The generic "point-in-time KPIs have no monthly history" explanation is
    // WRONG for a region miss (TRx has global history) — say the scope instead.
    const data: KPIHistoryResponse = {
      kpi_id: 'trx',
      brand: '',
      region: 'northeast',
      count: 0,
      points: [],
    };
    render(<KpiTrendChart kpiId="trx" data={data} />);
    expect(screen.getByText(/No northeast series available/i)).toBeInTheDocument();
    expect(screen.queryByText(/Point-in-time KPIs/i)).not.toBeInTheDocument();
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

  it('renders one comparison chart with a legend entry per bucket when segmented', () => {
    const segmented: KPISegmentedHistoryResponse = {
      kpi_id: 'WS3-BI-005',
      brand: 'Remibrutinib',
      axis: 'segment',
      data_through: '2026-07-13',
      count: 3,
      series: [
        {
          key: 'low_severity',
          label: 'Low severity',
          count: 2,
          points: [
            { metric_date: '2026-05-01', value: 57 },
            { metric_date: '2026-06-01', value: 272 },
          ],
        },
        {
          key: 'medium_severity',
          label: 'Medium severity',
          count: 2,
          points: [
            { metric_date: '2026-05-01', value: 140 },
            { metric_date: '2026-06-01', value: 715 },
          ],
        },
        {
          key: 'high_severity',
          label: 'High severity',
          count: 2,
          points: [
            { metric_date: '2026-05-01', value: 116 },
            { metric_date: '2026-06-01', value: 335 },
          ],
        },
      ],
    };
    render(<KpiTrendChart kpiId="trx" segmented={segmented} />);
    expect(screen.getByTestId('kpi-trend-chart-segmented')).toBeInTheDocument();
    expect(screen.queryByTestId('kpi-trend-chart')).not.toBeInTheDocument();
    expect(screen.getByText(/by severity tier/)).toBeInTheDocument();
    expect(screen.getByText(/Remibrutinib/)).toBeInTheDocument();
    expect(screen.getByText(/data through 2026-07-13/)).toBeInTheDocument();
  });

  it('labels the LOT axis and honors the therapy_line scope line', () => {
    const segmented: KPISegmentedHistoryResponse = {
      kpi_id: 'WS3-BI-005',
      brand: '',
      axis: 'therapy_line',
      data_through: null,
      count: 1,
      series: [
        {
          key: '2',
          label: '2 prior lines',
          count: 1,
          points: [{ metric_date: '2026-06-01', value: 313 }],
        },
      ],
    };
    render(<KpiTrendChart kpiId="trx" segmented={segmented} />);
    expect(screen.getByText(/by line of therapy/)).toBeInTheDocument();
    expect(screen.getByText(/All brands/)).toBeInTheDocument();
  });

  it('shows an honest empty state for a segmented response with no series', () => {
    const segmented: KPISegmentedHistoryResponse = {
      kpi_id: 'WS3-BI-007',
      brand: 'Fabhalta',
      axis: 'segment',
      data_through: null,
      count: 0,
      series: [],
    };
    render(<KpiTrendChart kpiId="nbrx" segmented={segmented} />);
    expect(screen.getByText(/No severity-tier series available/i)).toBeInTheDocument();
    expect(screen.queryByTestId('kpi-trend-chart-segmented')).not.toBeInTheDocument();
  });
});
