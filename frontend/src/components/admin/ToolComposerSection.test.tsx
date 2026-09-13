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

  it('shows the most common refusal reason with the coded count and the uncoded remainder', () => {
    const tools = payload.tools.map((tool, index) =>
      index === 0
        ? {
            ...tool,
            n_refused: 50,
            most_common_refusal_reason: 'coverage_gap',
            most_common_refusal_sentence:
              'the data does not cover everything the question asked about',
            n_refused_coded: 3,
            n_most_common_refusal_reason: 1,
          }
        : tool,
    );
    mockData({ ...payload, tools });

    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    const note = within(row).getByText(
      /most common: the data does not cover everything the question asked about/,
    );
    expect(note).toBeInTheDocument();
    // The note is its own secondary line, not appended inline to the refused count: a long
    // catalogue sentence must not widen the Refused column.
    expect(note.tagName).toBe('DIV');
    // 50 refused, 3 coded, 1 of those 3 is the most common code: the other 47 were never coded
    // at all, and that remainder must be visible or the minority code reads as representative.
    expect(within(row).getByText(/\(1 of 3 coded; 47 uncoded\)/)).toBeInTheDocument();
  });

  it('drops the uncoded suffix, not "; 0 uncoded", when every refusal is coded', () => {
    const tools = payload.tools.map((tool, index) =>
      index === 0
        ? {
            ...tool,
            n_refused: 3,
            most_common_refusal_reason: 'coverage_gap',
            most_common_refusal_sentence:
              'the data does not cover everything the question asked about',
            n_refused_coded: 3,
            n_most_common_refusal_reason: 1,
          }
        : tool,
    );
    mockData({ ...payload, tools });

    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    expect(within(row).getByText(/\(1 of 3 coded\)/)).toBeInTheDocument();
    expect(within(row).queryByText(/uncoded/)).not.toBeInTheDocument();
  });

  it('renders the bare code when the reason sentence is unknown to this build', () => {
    const tools = payload.tools.map((tool, index) =>
      index === 0
        ? {
            ...tool,
            most_common_refusal_reason: 'some_future_code',
            most_common_refusal_sentence: null,
            n_refused_coded: 2,
            n_most_common_refusal_reason: 2,
          }
        : tool,
    );
    mockData({ ...payload, tools });

    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    expect(within(row).getByText(/most common: some_future_code/)).toBeInTheDocument();
  });

  it('omits the coded parenthetical when both counts are unknown, never rendering "0 of n"', () => {
    const tools = payload.tools.map((tool, index) =>
      index === 0
        ? {
            ...tool,
            most_common_refusal_reason: 'non_binary_treatment',
            most_common_refusal_sentence: 'the treatment column is not a binary 0/1 indicator',
            n_refused_coded: null,
            n_most_common_refusal_reason: null,
          }
        : tool,
    );
    mockData({ ...payload, tools });

    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    expect(
      within(row).getByText(/most common: the treatment column is not a binary 0\/1 indicator/),
    ).toBeInTheDocument();
    expect(within(row).queryByText(/coded/)).not.toBeInTheDocument();
  });

  it('omits "coded" when only the most-common count (k) is unknown, n known', () => {
    const tools = payload.tools.map((tool, index) =>
      index === 0
        ? {
            ...tool,
            most_common_refusal_reason: 'non_binary_treatment',
            most_common_refusal_sentence: 'the treatment column is not a binary 0/1 indicator',
            n_refused_coded: 3,
            n_most_common_refusal_reason: null,
          }
        : tool,
    );
    mockData({ ...payload, tools });

    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    expect(
      within(row).getByText(/most common: the treatment column is not a binary 0\/1 indicator/),
    ).toBeInTheDocument();
    expect(within(row).queryByText(/coded/)).not.toBeInTheDocument();
  });

  it('omits "coded" when only the coded-total count (n) is unknown, k known', () => {
    const tools = payload.tools.map((tool, index) =>
      index === 0
        ? {
            ...tool,
            most_common_refusal_reason: 'non_binary_treatment',
            most_common_refusal_sentence: 'the treatment column is not a binary 0/1 indicator',
            n_refused_coded: null,
            n_most_common_refusal_reason: 1,
          }
        : tool,
    );
    mockData({ ...payload, tools });

    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    expect(
      within(row).getByText(/most common: the treatment column is not a binary 0\/1 indicator/),
    ).toBeInTheDocument();
    expect(within(row).queryByText(/coded/)).not.toBeInTheDocument();
  });

  it('renders a real 0 for the most-common count, never hiding it behind a truthy check', () => {
    const tools = payload.tools.map((tool, index) =>
      index === 0
        ? {
            ...tool,
            n_refused: 3,
            most_common_refusal_reason: 'non_binary_treatment',
            most_common_refusal_sentence: 'the treatment column is not a binary 0/1 indicator',
            n_refused_coded: 3,
            n_most_common_refusal_reason: 0,
          }
        : tool,
    );
    mockData({ ...payload, tools });

    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    expect(within(row).getByText(/\(0 of 3 coded\)/)).toBeInTheDocument();
  });

  it('formats the coded counts with fmtInt, matching how n_refused is already formatted', () => {
    const tools = payload.tools.map((tool, index) =>
      index === 0
        ? {
            ...tool,
            n_refused: 12345,
            most_common_refusal_reason: 'coverage_gap',
            most_common_refusal_sentence:
              'the data does not cover everything the question asked about',
            n_refused_coded: 2345,
            n_most_common_refusal_reason: 1234,
          }
        : tool,
    );
    mockData({ ...payload, tools });

    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    expect(within(row).getByText(/^12,345/)).toBeInTheDocument();
    expect(
      within(row).getByText(/\(1,234 of 2,345 coded; 10,000 uncoded\)/),
    ).toBeInTheDocument();
  });

  it('leaves the refused cell unchanged when there is no most-common refusal reason (pre-ml/043, key absent)', () => {
    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    const cells = within(row).getAllByRole('cell');
    expect(cells[4].textContent).toBe('0');
  });

  it('leaves the refused cell unchanged when the reason is an explicit post-ml/043 null, not an absent key', () => {
    const tools = payload.tools.map((tool, index) =>
      index === 0
        ? {
            ...tool,
            // A post-043 backend ALWAYS sends these four keys. A tool with no coded refusal
            // sends explicit nulls (and n_refused_coded: 0), never omits the keys.
            most_common_refusal_reason: null,
            most_common_refusal_sentence: null,
            n_refused_coded: 0,
            n_most_common_refusal_reason: null,
          }
        : tool,
    );
    mockData({ ...payload, tools });

    render(<ToolComposerSection days={30} />);

    const row = screen.getByRole('row', { name: /sensitivity_analyzer/ });
    const cells = within(row).getAllByRole('cell');
    expect(cells[4].textContent).toBe('0');
    expect(within(row).queryByText(/most common/)).not.toBeInTheDocument();
    expect(within(row).queryByText(/null/)).not.toBeInTheDocument();
  });

  it('appends the reason sentence to a step class when present', () => {
    const recentFailures = payload.recent_failures.map((failure, index) =>
      index === 0
        ? {
            ...failure,
            step_classes: failure.step_classes.map((step, stepIndex) =>
              stepIndex === 0
                ? {
                    ...step,
                    reason_code: 'coverage_gap',
                    reason: 'the data does not cover everything the question asked about',
                  }
                : step,
            ),
          }
        : failure,
    );
    mockData({ ...payload, recent_failures: recentFailures });

    render(<ToolComposerSection days={30} />);

    const failure = screen.getByTestId('recent-failure-comp_failed');
    expect(failure).toHaveTextContent(
      'gap_calculator: error — the data does not cover everything the question asked about',
    );
  });

  it('shows the bare reason code on a step class when the sentence is unknown to this build', () => {
    const recentFailures = payload.recent_failures.map((failure, index) =>
      index === 0
        ? {
            ...failure,
            step_classes: failure.step_classes.map((step, stepIndex) =>
              stepIndex === 0 ? { ...step, reason_code: 'some_future_code', reason: null } : step,
            ),
          }
        : failure,
    );
    mockData({ ...payload, recent_failures: recentFailures });

    render(<ToolComposerSection days={30} />);

    const failure = screen.getByTestId('recent-failure-comp_failed');
    expect(failure).toHaveTextContent('gap_calculator: error — some_future_code');
  });

  it('leaves a step class unchanged when it has no reason (pre-ml/043, keys absent)', () => {
    render(<ToolComposerSection days={30} />);

    const failure = screen.getByTestId('recent-failure-comp_failed');
    expect(failure).toHaveTextContent('gap_calculator: error');
    expect(failure).not.toHaveTextContent('gap_calculator: error —');
  });

  it('leaves a step class unchanged when reason_code and reason are explicit post-ml/043 nulls', () => {
    const recentFailures = payload.recent_failures.map((failure, index) =>
      index === 0
        ? {
            ...failure,
            step_classes: failure.step_classes.map((step, stepIndex) =>
              stepIndex === 0 ? { ...step, reason_code: null, reason: null } : step,
            ),
          }
        : failure,
    );
    mockData({ ...payload, recent_failures: recentFailures });

    render(<ToolComposerSection days={30} />);

    const failure = screen.getByTestId('recent-failure-comp_failed');
    expect(failure).toHaveTextContent('gap_calculator: error');
    expect(failure).not.toHaveTextContent('gap_calculator: error —');
    expect(failure).not.toHaveTextContent('null');
  });

  it('joins step classes with "; " so a sentence containing a comma cannot be read as a step boundary', () => {
    const recentFailures = payload.recent_failures.map((failure, index) =>
      index === 0
        ? {
            ...failure,
            step_classes: [
              {
                ...failure.step_classes[0],
                reason_code: 'coverage_gap',
                reason: 'the data does not cover everything, including edge cases',
              },
              {
                ...failure.step_classes[1],
                reason_code: 'non_binary_treatment',
                reason: 'the treatment column is not a binary 0/1 indicator',
              },
            ],
          }
        : failure,
    );
    mockData({ ...payload, recent_failures: recentFailures });

    render(<ToolComposerSection days={30} />);

    const failure = screen.getByTestId('recent-failure-comp_failed');
    expect(failure).toHaveTextContent(
      'gap_calculator: error — the data does not cover everything, including edge cases; cate_analyzer: dependency unmet — the treatment column is not a binary 0/1 indicator',
    );
  });

  // #2050: the numeric details recorded with a refusal. The step-classes line is the failure's
  // second paragraph; each step is one child span, the later ones led by the "; " join.
  const stepTexts = (failure: HTMLElement) =>
    Array.from(failure.querySelectorAll('p')[1].children).map((el) =>
      (el.textContent ?? '').replace(/^; /, ''),
    );

  const withFirstFailureSteps = (steps: Record<string, unknown>[]) =>
    payload.recent_failures.map((failure, index) =>
      index === 0
        ? {
            ...failure,
            step_classes: steps.map((step, stepIndex) => ({
              ...failure.step_classes[stepIndex],
              ...step,
            })),
          }
        : failure,
    );

  it('keeps two refusals under one code distinguishable by their recorded details', () => {
    // cate_analyzer's "no treatment contrast" and "no usable outcome" share one code and one
    // sentence; only the details recorded with each refusal tell an operator which it was.
    const sentence = 'fewer groups were present than the comparison requires';
    const refusal = {
      tool_name: 'cate_analyzer',
      outcome_class: 'refused',
      reason_code: 'insufficient_groups',
      reason: sentence,
    };
    mockData({
      ...payload,
      recent_failures: withFirstFailureSteps([
        {
          ...refusal,
          reason_details: { n_segments_named: 4, n_no_contrast: 3, n_non_finite: 0 },
        },
        {
          ...refusal,
          reason_details: { n_segments_named: 4, n_no_contrast: 0, n_non_finite: 3 },
        },
      ]),
    });

    render(<ToolComposerSection days={30} />);

    const [first, second] = stepTexts(screen.getByTestId('recent-failure-comp_failed'));
    expect(first).toBe(
      `cate_analyzer: refused — ${sentence} (n_no_contrast=3, n_non_finite=0, n_segments_named=4)`,
    );
    expect(second).toBe(
      `cate_analyzer: refused — ${sentence} (n_no_contrast=0, n_non_finite=3, n_segments_named=4)`,
    );
    expect(first).not.toBe(second);
  });

  it('formats details with sorted keys, exact locale-free numbers and true/false', () => {
    // An operator diagnostic: no group separator (de-DE would render 12345 as "12.345", which
    // reads as a decimal, and en-US's "," collides with the pair join) and no rounding.
    mockData({
      ...payload,
      recent_failures: withFirstFailureSteps([
        {
          reason_code: 'coverage_gap',
          reason: 'the data does not cover everything the question asked about',
          reason_details: {
            share_kept: 0.6666666666666666,
            n_rows: 12345,
            is_scoped: true,
            has_outcome: false,
          },
        },
        {},
      ]),
    });

    render(<ToolComposerSection days={30} />);

    const [first] = stepTexts(screen.getByTestId('recent-failure-comp_failed'));
    expect(first).toBe(
      'gap_calculator: error — the data does not cover everything the question asked about' +
        ' (has_outcome=false, is_scoped=true, n_rows=12345, share_kept=0.6666666666666666)',
    );
  });

  it('lets a long unbroken detail key wrap instead of overflowing the failure card', () => {
    mockData({
      ...payload,
      recent_failures: withFirstFailureSteps([
        {
          reason_code: 'coverage_gap',
          reason: 'the data does not cover everything the question asked about',
          reason_details: { n_rows_dropped_for_missing_outcome_or_treatment_or_segment: 7 },
        },
        {},
      ]),
    });

    render(<ToolComposerSection days={30} />);

    const stepLine = screen.getByTestId('recent-failure-comp_failed').querySelectorAll('p')[1];
    expect(stepLine).toHaveClass('break-words');
  });

  it('leaves a step class unchanged when its details are an empty mapping', () => {
    mockData({
      ...payload,
      recent_failures: withFirstFailureSteps([
        {
          reason_code: 'coverage_gap',
          reason: 'the data does not cover everything the question asked about',
          reason_details: {},
        },
        {},
      ]),
    });

    render(<ToolComposerSection days={30} />);

    const [first, second] = stepTexts(screen.getByTestId('recent-failure-comp_failed'));
    expect(first).toBe(
      'gap_calculator: error — the data does not cover everything the question asked about',
    );
    // The fixture's second step carries no details key at all (a pre-9a payload).
    expect(second).toBe('cate_analyzer: dependency unmet');
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
