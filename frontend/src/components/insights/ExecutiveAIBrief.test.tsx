/**
 * ExecutiveAIBrief Tests
 * ======================
 *
 * Guards two generations of findings:
 * - The fake-brief finding: the widget formerly booted `useState(SAMPLE_BRIEF)`
 *   ($2.3M, 847 HCPs, beta=0.42, 12.3% MoM TRx) and spliced fake sections into
 *   every real answer with hardcoded confidence badges. None of that may return.
 * - The PR-5 rewire (review finding 1: the brief read as a description, not a
 *   strategic distillation): the card posts ONLY the brand to
 *   `POST /api/insights/executive-brief`; the grounding figures are derived
 *   server-side from the gap-analysis feed (codex PR-5 round 3 — caller-posted
 *   figures would let anyone mint a grounded-looking brief), and no-signal /
 *   feed-outage states come back as honest labelled fallback text.
 *
 * Desired behavior: real crystallized insights, else the real grounded
 * distillation alone, else an honest empty state or a labeled error.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { ExecutiveAIBrief } from './ExecutiveAIBrief';
import * as useExec from '@/hooks/api/use-executive-insights';
import * as useIns from '@/hooks/api/use-insights';
import type { ExecutiveBriefInsightRequest } from '@/types/insights';

vi.mock('@/hooks/api/use-executive-insights');
vi.mock('@/hooks/api/use-insights');

type BriefMutation = ReturnType<typeof useIns.useExecutiveBriefInsight>;
type ExecQuery = ReturnType<typeof useExec.useExecutiveInsights>;
type MockFn = ReturnType<typeof vi.fn>;

function mockBrief(overrides: Partial<BriefMutation> = {}) {
  vi.mocked(useIns.useExecutiveBriefInsight).mockReturnValue({
    mutate: vi.fn(),
    reset: vi.fn(),
    data: undefined,
    variables: undefined,
    error: null,
    isPending: false,
    ...overrides,
  } as unknown as BriefMutation);
}

function mockExec(overrides: Partial<ExecQuery> = {}) {
  vi.mocked(useExec.useExecutiveInsights).mockReturnValue({
    data: [],
    isLoading: false,
    isError: false,
    isSuccess: true,
    ...overrides,
  } as unknown as ExecQuery);
}

/** A real-shaped insights-endpoint response (LLM path). */
const DISTILLATION = {
  insight:
    'Prioritize the Northeast TRX gap: $2.4M at stake at 4.0x ROI. Sequence: specialty coverage first (medium effort), then rebalance calls.',
  key_takeaways: ['Fund the Northeast expansion first', 'Revisit after one quarter'],
  grounding: [{ label: 'Brand', value: 'Kisqali' }],
  is_fallback: false,
  generated_at: '2026-07-05T00:00:00Z',
  provenance: 'Gap-analyzer ROI opportunities (server-derived)',
};

beforeEach(() => {
  vi.clearAllMocks();
  mockBrief();
  mockExec();
});

describe('ExecutiveAIBrief — real crystallized insights', () => {
  it('renders crystallized insight narratives when the brand has them', () => {
    mockExec({
      data: [
        {
          insight_id: 'ei_1',
          title: 'Detailing drives TRx',
          narrative: 'Detailing frequency is the strongest driver (b=0.42).',
          brand: 'Remibrutinib',
          crystallized_at: '2026-06-08T00:00:00Z',
          source_count: 3,
          effect_size: 0.42,
          effect_direction: 'positive',
        },
      ],
    } as unknown as Partial<ExecQuery>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);
    expect(screen.getByText('Detailing drives TRx')).toBeInTheDocument();
    expect(
      screen.getByText(/Detailing frequency is the strongest driver/)
    ).toBeInTheDocument();
  });

  it('crystallized insights take precedence over the distillation', () => {
    mockExec({
      data: [
        {
          insight_id: 'ei_1',
          title: 'Detailing drives TRx',
          narrative: 'Detailing frequency is the strongest driver (b=0.42).',
          brand: 'Kisqali',
          crystallized_at: '2026-06-08T00:00:00Z',
          source_count: 3,
        },
      ],
    } as unknown as Partial<ExecQuery>);
    mockBrief({
      data: DISTILLATION,
      variables: { brand: 'Kisqali' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Kisqali" />);
    expect(screen.getByText('Detailing drives TRx')).toBeInTheDocument();
    expect(screen.queryByText('Strategic Brief')).not.toBeInTheDocument();
  });
});

describe('ExecutiveAIBrief — server grounding chips (clinical context visibility)', () => {
  it('renders the server grounding chips, including clinical context, under the distillation', () => {
    mockBrief({
      data: {
        ...DISTILLATION,
        grounding: [
          { label: 'Brand', value: 'Remibrutinib' },
          { label: 'Clinical context', value: 'included' },
        ],
      },
      variables: { brand: 'Remibrutinib' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);

    const chipLabel = screen.getByText('Clinical context');
    expect(chipLabel).toBeInTheDocument();
    // the outer chip (parent of the bolded label) carries label AND value
    expect(chipLabel.parentElement?.textContent).toContain('included');
  });

  it('renders no grounding row when the brief carries no grounding chips', () => {
    mockBrief({
      data: { ...DISTILLATION, grounding: [] },
      variables: { brand: 'Remibrutinib' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);
    expect(screen.queryByTestId('brief-grounding')).not.toBeInTheDocument();
  });

  it('does not surface distillation grounding chips when crystallized insights take precedence', () => {
    mockExec({
      data: [
        {
          insight_id: 'ei_1',
          title: 'Detailing drives TRx',
          narrative: 'Detailing frequency is the strongest driver.',
          brand: 'Remibrutinib',
          crystallized_at: '2026-06-08T00:00:00Z',
          source_count: 3,
        },
      ],
    } as unknown as Partial<ExecQuery>);
    mockBrief({
      data: {
        ...DISTILLATION,
        grounding: [{ label: 'Clinical context', value: 'included' }],
      },
      variables: { brand: 'Remibrutinib' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);
    // crystallized path is shown; the distillation's grounding must not leak in
    expect(screen.queryByTestId('brief-grounding')).not.toBeInTheDocument();
  });
});

describe('ExecutiveAIBrief — no SAMPLE_BRIEF fabrication', () => {
  it('renders an honest empty state (not SAMPLE_BRIEF) when nothing has loaded', () => {
    render(<ExecutiveAIBrief brand="Remibrutinib" />);

    // The fabricated sections must never render.
    expect(screen.queryByText('Key Performance Trend')).not.toBeInTheDocument();
    expect(screen.queryByText(/\$2\.3M/)).not.toBeInTheDocument();
    expect(screen.queryByText(/847 high-propensity HCPs/)).not.toBeInTheDocument();
    expect(screen.queryByText(/12\.3% MoM/)).not.toBeInTheDocument();

    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });

  it('renders ONLY the real distillation — no fake sections spliced in', () => {
    mockBrief({
      data: DISTILLATION,
      variables: { brand: 'Kisqali' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Kisqali" />);

    expect(screen.getByText('Strategic Brief')).toBeInTheDocument();
    expect(screen.getByText(/Prioritize the Northeast TRX gap/)).toBeInTheDocument();
    // Real takeaways render as a list.
    expect(screen.getByText('Fund the Northeast expansion first')).toBeInTheDocument();

    // Formerly `...SAMPLE_BRIEF.slice(1)` contaminated every real answer.
    expect(screen.queryByText('Emerging Opportunity')).not.toBeInTheDocument();
    expect(screen.queryByText('Risk Alert')).not.toBeInTheDocument();
    expect(screen.queryByText(/\$2\.3M/)).not.toBeInTheDocument();
    // Hardcoded confidence badges (85%/87%/78%) must not be fabricated.
    expect(screen.queryByText(/87% confidence/)).not.toBeInTheDocument();
    expect(screen.queryByText(/78% confidence/)).not.toBeInTheDocument();
    expect(screen.queryByText(/85% confidence/)).not.toBeInTheDocument();
  });

  it('reports the real generated-section count in the footer', () => {
    mockBrief({
      data: DISTILLATION,
      variables: { brand: 'Kisqali' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Kisqali" />);
    // Formerly hardcoded "3 insights generated" regardless of content.
    expect(screen.getByText(/1 insight generated/)).toBeInTheDocument();
    expect(screen.queryByText(/3 insights generated/)).not.toBeInTheDocument();
  });

  it('shows a labeled error state when the insight call fails and nothing else is available', () => {
    mockBrief({
      error: new Error('insights service unavailable'),
      variables: { brand: 'Remibrutinib' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);
    expect(screen.getByText(/unable to generate brief/i)).toBeInTheDocument();
    expect(screen.getByText(/insights service unavailable/)).toBeInTheDocument();
    expect(screen.queryByText('Key Performance Trend')).not.toBeInTheDocument();
  });
});

describe('ExecutiveAIBrief — honest fallback labeling (PR-5)', () => {
  it('labels the deterministic fallback as a factual summary, distinct from the LLM distillation', () => {
    mockBrief({
      data: {
        ...DISTILLATION,
        insight: 'Scope: Kisqali / $2.4M. Ranked opportunities: 1. Expand specialty coverage…',
        key_takeaways: [],
        is_fallback: true,
      },
      variables: { brand: 'Kisqali' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Kisqali" />);

    expect(screen.getByText('Strategic Brief')).toBeInTheDocument();
    expect(screen.getByText('Factual summary (no LLM distillation)')).toBeInTheDocument();
    expect(
      screen.queryByText('AI distillation of live gap-analysis figures')
    ).not.toBeInTheDocument();
  });

  it('labels the real LLM distillation as such and stamps the footer timestamp', () => {
    mockBrief({
      data: DISTILLATION,
      variables: { brand: 'Kisqali' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Kisqali" />);

    expect(
      screen.getByText('AI distillation of live gap-analysis figures')
    ).toBeInTheDocument();
    expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    expect(screen.queryByText(/Not yet generated/)).not.toBeInTheDocument();
  });

  it('renders the server no-signal fallback text honestly (no client-side signal guessing)', () => {
    // The SERVER decides no-signal (its feed read found nothing) and answers
    // with honest fallback text — the card renders it verbatim as a labelled
    // factual summary, never inventing an empty state that hides the answer.
    mockBrief({
      data: {
        ...DISTILLATION,
        insight:
          'No gap-analysis signal is available for Fabhalta yet — run a gap analysis to generate an executive brief.',
        key_takeaways: [],
        is_fallback: true,
        provenance: 'Gap-analyzer ROI opportunities (server-derived)',
      },
      variables: { brand: 'Fabhalta' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Fabhalta" />);

    expect(screen.getByText(/run a gap analysis/i)).toBeInTheDocument();
    expect(screen.getByText('Factual summary (no LLM distillation)')).toBeInTheDocument();
    expect(screen.queryByTestId('empty-state')).not.toBeInTheDocument();
  });

  it('renders the server feed-outage fallback distinctly from no-signal (codex PR-5 rounds 2-3)', () => {
    mockBrief({
      data: {
        ...DISTILLATION,
        insight:
          'The gap-analysis figures for Kisqali are currently unavailable, so no grounded executive brief can be produced — this is a data-source failure, not an empty portfolio.',
        key_takeaways: [],
        is_fallback: true,
        provenance: 'Gap-analyzer ROI opportunities (unavailable)',
      },
      variables: { brand: 'Kisqali' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Kisqali" />);

    expect(screen.getByText(/data-source failure/i)).toBeInTheDocument();
    expect(screen.queryByText(/run a gap analysis/i)).not.toBeInTheDocument();
  });
});

describe('ExecutiveAIBrief — server-derived request contract', () => {
  function lastRequest(mutate: MockFn): ExecutiveBriefInsightRequest | undefined {
    const calls = mutate.mock.calls;
    return calls[calls.length - 1]?.[0] as ExecutiveBriefInsightRequest | undefined;
  }

  it('posts ONLY the brand — figures are never client-supplied', async () => {
    const mutate = vi.fn();
    mockBrief({ mutate } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Kisqali" />);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(lastRequest(mutate)).toEqual({ brand: 'Kisqali' });
  });

  it('clears the prior brand footer (last-updated + count) on a brand switch', () => {
    // Codex round-2 HIGH(b): the footer leaked brand A's "Last updated" + insight
    // count under brand B. Dynamic mock so reset() actually clears the data,
    // faithful to react-query.
    let briefData: unknown = DISTILLATION;
    const reset = vi.fn(() => { briefData = undefined; });
    vi.mocked(useIns.useExecutiveBriefInsight).mockImplementation(() => ({
      mutate: vi.fn(),
      reset,
      data: briefData,
      variables: { brand: 'Kisqali' },
      error: null,
      isPending: false,
    } as unknown as BriefMutation));

    const { rerender } = render(<ExecutiveAIBrief brand="Kisqali" />);
    expect(screen.getByText(/1 insight generated/)).toBeInTheDocument();
    expect(screen.getByText(/Last updated:/)).toBeInTheDocument();

    rerender(<ExecutiveAIBrief brand="Fabhalta" />);

    // The prior brand's footer state must not linger.
    expect(screen.queryByText(/Last updated:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/1 insight generated/)).not.toBeInTheDocument();
    expect(screen.getByText(/0 insights generated/)).toBeInTheDocument();
  });

  it('never shows the previous brand brief on the switch frame (no stale attribution)', () => {
    mockBrief({
      mutate: vi.fn(),
      data: DISTILLATION,
      variables: { brand: 'Kisqali' },
    } as unknown as Partial<BriefMutation>);

    const { rerender } = render(<ExecutiveAIBrief brand="Kisqali" />);
    expect(screen.getByText(/Prioritize the Northeast TRX gap/)).toBeInTheDocument();

    rerender(<ExecutiveAIBrief brand="Fabhalta" />);

    expect(screen.queryByText(/Prioritize the Northeast TRX gap/)).not.toBeInTheDocument();
  });

  it('drops a LATE-resolving response from the previous brand (codex PR-5 round 1 HIGH)', () => {
    // reset() does not cancel an in-flight mutation. Model the race: brand A's
    // request resolves AFTER the switch to brand B (the hook still surfaces
    // A's data with A's request variables). The attribution guard must refuse
    // to render it under B — even on a fresh mount where brandChanged is
    // false and no reset has run.
    mockBrief({
      mutate: vi.fn(),
      data: DISTILLATION,
      variables: { brand: 'Kisqali' },
    } as unknown as Partial<BriefMutation>);

    render(<ExecutiveAIBrief brand="Fabhalta" />);

    // Brand A's brief must not be attributed to brand B.
    expect(screen.queryByText(/Prioritize the Northeast TRX gap/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Last updated:/)).not.toBeInTheDocument();
    expect(screen.getByText(/0 insights generated/)).toBeInTheDocument();
    // B's honest empty state renders instead.
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });
});
