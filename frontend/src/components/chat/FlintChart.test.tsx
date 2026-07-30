/**
 * FlintChart tests — 2026-07-30
 * =============================
 *
 * The states matter more than the pixels here. A chart that failed to build and
 * a KPI that genuinely has no data look identical if both render as a blank
 * card, and the platform's whole posture is that an empty result must be
 * legible as empty rather than mistaken for a real zero. So: loading, empty and
 * error are asserted as visibly distinct, and the figure path is asserted to
 * hand Plotly exactly the traces it was given.
 *
 * Plotly is stubbed because jsdom has no WebGL; what is under test is the
 * component's wiring, not Plotly's rendering.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { FlintChart } from './FlintChart';
import type { KpiChartData } from '@/lib/kpi-chart-router';

const mockReact = vi.fn();
const mockPurge = vi.fn();

vi.mock('plotly.js-dist-min', () => ({
  default: {
    react: (...args: unknown[]) => mockReact(...args),
    purge: (...args: unknown[]) => mockPurge(...args),
  },
}));

/** Routed data as kpi-chart-router would produce it for a real monthly series. */
const CHART_DATA: KpiChartData = {
  rows: [
    { month: '2026-01-01', value: 1200 },
    { month: '2026-02-01', value: 1310 },
  ],
  semanticTypes: { month: 'Date', value: 'Count' },
  encoding: { axis: 'month', value: 'value' },
  chartType: 'Line Chart',
  title: 'TRx trend',
  subtitle: 'All brands · 2 months',
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe('FlintChart states', () => {
  it('shows a loading state while the handler is fetching', () => {
    render(<FlintChart title="TRx trend" loading />);
    expect(screen.getByText(/Building TRx trend/)).toBeInTheDocument();
    expect(screen.queryByTestId('flint-chart')).not.toBeInTheDocument();
  });

  it('states the reason a KPI has no data', () => {
    render(
      <FlintChart
        title="NBRx trend"
        emptyReason="NBRx is tracked per brand only — pass a brand."
        subtitle="All brands"
      />
    );
    expect(screen.getByText(/tracked per brand only/)).toBeInTheDocument();
    expect(screen.queryByTestId('flint-chart')).not.toBeInTheDocument();
  });

  it('distinguishes a build failure from having no data', () => {
    // "Couldn't build" must never read as "there is no data" — one is a bug to
    // report, the other is a fact about the KPI.
    render(<FlintChart title="ROC-AUC" error="Field “nope” is not present in the data." />);
    expect(screen.getByText(/Couldn’t build this chart/)).toBeInTheDocument();
    expect(screen.getByText(/is not present in the data/)).toBeInTheDocument();
  });

  it('renders the title and scope line from the routed data', () => {
    render(<FlintChart title="fallback" chartData={CHART_DATA} />);
    expect(screen.getByText('TRx trend')).toBeInTheDocument();
    expect(screen.getByText(/2 months/)).toBeInTheDocument();
  });

  it('treats routed-but-empty rows as an empty state, not a chart', () => {
    render(
      <FlintChart
        title="NBRx"
        chartData={{ ...CHART_DATA, rows: [] }}
        emptyReason="No series for NBRx without a brand."
      />
    );
    expect(screen.getByText(/No series for NBRx/)).toBeInTheDocument();
    expect(screen.queryByTestId('flint-chart')).not.toBeInTheDocument();
  });

  it('surfaces a compile failure as an error, not as empty', async () => {
    // A field the rows do not carry: flint would compile a blank plot, the
    // validation shim turns it into a stated reason.
    render(
      <FlintChart
        title="TRx trend"
        chartData={{ ...CHART_DATA, encoding: { axis: 'month', value: 'nope' } }}
      />
    );
    expect(await screen.findByText(/Couldn’t build this chart/)).toBeInTheDocument();
    expect(await screen.findByText(/nope/)).toBeInTheDocument();
  });
});

describe('FlintChart rendering', () => {
  it('hands Plotly traces carrying the real routed values', async () => {
    render(<FlintChart title="TRx trend" chartData={CHART_DATA} />);

    await waitFor(() => expect(mockReact).toHaveBeenCalled());
    const [, data, layout] = mockReact.mock.calls[0] as [unknown, Array<{ y?: unknown[] }>, Record<string, unknown>];
    // The values Plotly receives must be exactly what the router fetched.
    expect(data[0].y).toEqual([1200, 1310]);
    expect(layout).toMatchObject({ autosize: true });
  });

  it('purges Plotly on unmount', async () => {
    // A long chat thread would otherwise accumulate listeners and GL contexts.
    const { unmount } = render(<FlintChart title="TRx trend" chartData={CHART_DATA} />);
    await waitFor(() => expect(mockReact).toHaveBeenCalled());
    unmount();
    await waitFor(() => expect(mockPurge).toHaveBeenCalled());
  });
});
