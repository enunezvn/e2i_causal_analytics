/**
 * ToolComposerSection tests — the verdict word leads every tool row, a tool without enough
 * runs says so rather than showing a number it does not have, and a failed composition says
 * which step classes caused it.
 *
 * The payload is the fixture captured from a REAL database run (Task 14's real-DB test), not a
 * hand-written shape: a component that only ever sees an idealised response is not tested
 * against the one the API actually returns.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, within } from '@testing-library/react';

import type { ToolComposerObservabilityResponse } from '@/types/admin';
import PAYLOAD from './__fixtures__/tool-composer-observability.json';

vi.mock('@/hooks/api/use-admin', () => ({
  useToolComposerObservability: vi.fn(),
}));

import * as adminHooks from '@/hooks/api/use-admin';
import { ToolComposerSection } from './ToolComposerSection';

const payload = PAYLOAD as unknown as ToolComposerObservabilityResponse;

function mockData(data: unknown, extra: Record<string, unknown> = {}) {
  vi.mocked(adminHooks.useToolComposerObservability).mockReturnValue({
    data,
    isLoading: false,
    isError: false,
    ...extra,
  } as never);
}

describe('ToolComposerSection', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockData(payload);
  });

  it('leads each tool row with its verdict word', () => {
    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    const cells = within(row).getAllByRole('cell');
    expect(cells[0]).toHaveTextContent(/caveat/i);
  });

  it('says how few runs a tool has instead of implying a verdict', () => {
    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /cate_analyzer/ });
    expect(within(row).getByText(/too few runs/i)).toBeInTheDocument();
    expect(within(row).getByText(/n=1/)).toBeInTheDocument();
  });

  it('shows an em dash for latency it has not measured, and labels the declared number', () => {
    render(<ToolComposerSection days={30} />);

    const measured = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    expect(within(measured).getByText(/514 ms|513\.5/)).toBeInTheDocument();

    const unmeasured = screen.getByRole('row', { name: /cate_analyzer/ });
    expect(within(unmeasured).getByText('—')).toBeInTheDocument();
    expect(within(unmeasured).getByText(/declared/i)).toBeInTheDocument();
  });

  it('renders every composition count, so the cards account for the total', () => {
    // Values, not just labels: a card bound to the wrong field, or a label with nothing behind
    // it, is exactly how a composition goes missing from the reader's arithmetic.
    // Every count distinct, so a card bound to the wrong field cannot coincide with the right
    // value. Abandoned is a subset of unfinished, not another additive outcome.
    const counts = {
      ...payload.compositions,
      total: 25,
      success: 11,
      partial: 7,
      failed: 4,
      cancelled: 2,
      unfinished: 1,
      abandoned: 3,
    };
    mockData({ ...payload, compositions: counts });

    render(<ToolComposerSection days={30} />);

    const shown = (label: string) =>
      screen.getByText(label).parentElement?.textContent?.replace(label, '').trim();
    expect(shown('Compositions')).toBe('25');
    expect(shown('Success')).toBe('11');
    expect(shown('Partial')).toBe('7');
    expect(shown('Failed')).toBe('4');
    expect(shown('Cancelled')).toBe('2');
    expect(shown('Unfinished')).toBe('1');
    expect(shown('Abandoned')).toBe('3');
    // The four outcomes plus the unfinished ones account for every composition counted.
    expect(
      counts.success + counts.partial + counts.failed + counts.cancelled + counts.unfinished,
    ).toBe(counts.total);
  });

  it('lists each recent failure with its phase, step classes and bounded preview', () => {
    render(<ToolComposerSection days={30} />);

    const failure = screen.getByTestId('recent-failure-comp_failed');
    expect(failure).toHaveTextContent('execute');
    expect(failure).toHaveTextContent('gap_calculator: error');
    expect(failure).toHaveTextContent('cate_analyzer: dependency unmet');
    const preview = within(failure).getByTestId('query-preview').textContent ?? '';
    expect(preview.length).toBeLessThanOrEqual(101);
  });

  it('shows an empty state when nothing failed in the window', () => {
    mockData({ ...payload, recent_failures: [] });

    render(<ToolComposerSection days={30} />);

    expect(screen.getByText(/no failed compositions in this window/i)).toBeInTheDocument();
  });

  it('says so honestly while loading and when the read fails', () => {
    mockData(undefined, { isLoading: true });
    const { unmount } = render(<ToolComposerSection days={30} />);
    expect(screen.getByText(/loading tool composer/i)).toBeInTheDocument();
    unmount();

    mockData(undefined, { isLoading: false, isError: true });
    render(<ToolComposerSection days={30} />);
    expect(screen.getByText(/failed to load tool composer/i)).toBeInTheDocument();
  });

  it('passes the tab-level window straight through to the query', () => {
    render(<ToolComposerSection days={90} />);

    expect(adminHooks.useToolComposerObservability).toHaveBeenCalledWith(90);
  });
});
