/**
 * ExecutiveAIBrief Tests
 * ======================
 *
 * Guards two generations of findings:
 * - The fake-brief finding: the widget formerly booted `useState(SAMPLE_BRIEF)`
 *   ($2.3M, 847 HCPs, beta=0.42, 12.3% MoM TRx) and spliced fake sections into
 *   every real answer with hardcoded confidence badges. None of that may return.
 * - The PR-5 rewire (review finding 1: the brief read as a description, not a
 *   strategic distillation): the card now posts the brand's REAL gap-analysis
 *   figures to `POST /api/insights/executive-brief` (DSPy distillation with an
 *   honestly-labelled deterministic fallback). No signal -> NO call: an LLM
 *   riff over zero figures is fabrication, so the honest empty state renders.
 *
 * Desired behavior: real crystallized insights, else the real grounded
 * distillation alone, else an honest empty state or a labeled error.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { ExecutiveAIBrief } from './ExecutiveAIBrief';
import * as useExec from '@/hooks/api/use-executive-insights';
import * as useIns from '@/hooks/api/use-insights';
import { useOpportunities } from '@/hooks/api';
import type { ExecutiveBriefInsightRequest } from '@/types/insights';

vi.mock('@/hooks/api/use-executive-insights');
vi.mock('@/hooks/api/use-insights');
// The brief grounds its request in the brand's real opportunity figures.
// Mock the opportunities feed so these unit tests stay hermetic.
vi.mock('@/hooks/api', () => ({ useOpportunities: vi.fn() }));

type BriefMutation = ReturnType<typeof useIns.useExecutiveBriefInsight>;
type ExecQuery = ReturnType<typeof useExec.useExecutiveInsights>;
type MockFn = ReturnType<typeof vi.fn>;

/** Default the opportunities feed to a settled, empty (no-data) state. */
function mockOpps(overrides: Record<string, unknown> = {}) {
  (useOpportunities as MockFn).mockReturnValue({
    data: undefined,
    isLoading: false,
    isError: false,
    error: null,
    ...overrides,
  });
}

function mockBrief(overrides: Partial<BriefMutation> = {}) {
  vi.mocked(useIns.useExecutiveBriefInsight).mockReturnValue({
    mutate: vi.fn(),
    reset: vi.fn(),
    data: undefined,
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
  provenance: 'Gap-analyzer ROI opportunities (LLM distillation)',
};

const OPP_CONTEXT = {
  total_count: 1,
  quick_wins_count: 1,
  steady_plays_count: 0,
  strategic_bets_count: 0,
  suppressed_count: 0,
  total_addressable_value: 2_400_000,
  opportunities: [
    {
      rank: 1,
      gap: {
        gap_id: 'g1', metric: 'trx', segment: 'region', segment_value: 'Northeast',
        current_value: 85, target_value: 100, gap_size: 15, gap_percentage: 15,
        gap_type: 'vs_target',
      },
      roi_estimate: {
        gap_id: 'g1', estimated_revenue_impact: 2_400_000, estimated_cost_to_close: 300_000,
        expected_roi: 4, risk_adjusted_roi: 3, payback_period_months: 6,
        attribution_level: 'partial', attribution_rate: 0.65, confidence: 0.8,
      },
      recommended_action: 'Expand specialty coverage in the Northeast',
      implementation_difficulty: 'medium',
      time_to_impact: '3-6 months',
      category: 'steady_play',
    },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
  mockBrief();
  mockExec();
  mockOpps();
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
    mockBrief({ data: DISTILLATION } as unknown as Partial<BriefMutation>);
    mockOpps({ data: OPP_CONTEXT });

    render(<ExecutiveAIBrief brand="Kisqali" />);
    expect(screen.getByText('Detailing drives TRx')).toBeInTheDocument();
    expect(screen.queryByText('Strategic Brief')).not.toBeInTheDocument();
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
    mockBrief({ data: DISTILLATION } as unknown as Partial<BriefMutation>);
    mockOpps({ data: OPP_CONTEXT });

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
    mockBrief({ data: DISTILLATION } as unknown as Partial<BriefMutation>);
    mockOpps({ data: OPP_CONTEXT });

    render(<ExecutiveAIBrief brand="Kisqali" />);
    // Formerly hardcoded "3 insights generated" regardless of content.
    expect(screen.getByText(/1 insight generated/)).toBeInTheDocument();
    expect(screen.queryByText(/3 insights generated/)).not.toBeInTheDocument();
  });

  it('shows a labeled error state when the insight call fails and nothing else is available', () => {
    mockBrief({
      error: new Error('insights service unavailable'),
    } as unknown as Partial<BriefMutation>);
    mockOpps({ data: OPP_CONTEXT });

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
    } as unknown as Partial<BriefMutation>);
    mockOpps({ data: OPP_CONTEXT });

    render(<ExecutiveAIBrief brand="Kisqali" />);

    expect(screen.getByText('Strategic Brief')).toBeInTheDocument();
    expect(screen.getByText('Factual summary (LLM unavailable)')).toBeInTheDocument();
    expect(
      screen.queryByText('AI distillation of live gap-analysis figures')
    ).not.toBeInTheDocument();
  });

  it('labels the real LLM distillation as such and stamps the footer timestamp', () => {
    mockBrief({ data: DISTILLATION } as unknown as Partial<BriefMutation>);
    mockOpps({ data: OPP_CONTEXT });

    render(<ExecutiveAIBrief brand="Kisqali" />);

    expect(
      screen.getByText('AI distillation of live gap-analysis figures')
    ).toBeInTheDocument();
    expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    expect(screen.queryByText(/Not yet generated/)).not.toBeInTheDocument();
  });
});

describe('ExecutiveAIBrief — request is grounded in real opportunity figures', () => {
  function lastRequest(mutate: MockFn): ExecutiveBriefInsightRequest | undefined {
    const calls = mutate.mock.calls;
    return calls[calls.length - 1]?.[0] as ExecutiveBriefInsightRequest | undefined;
  }

  it('posts the real opportunity figures once the feed settles', async () => {
    const mutate = vi.fn();
    mockBrief({ mutate } as unknown as Partial<BriefMutation>);
    mockOpps({ data: OPP_CONTEXT });

    render(<ExecutiveAIBrief brand="Kisqali" />);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    const r = lastRequest(mutate)!;
    expect(r.brand).toBe('Kisqali');
    expect(r.total_addressable_value).toBe(2_400_000);
    expect(r.opportunities![0].recommended_action).toBe(
      'Expand specialty coverage in the Northeast'
    );
    expect(r.opportunities![0].expected_roi).toBe(4);
  });

  it('waits for opportunities to settle before generating (no premature ungrounded call)', () => {
    const mutate = vi.fn();
    mockBrief({ mutate } as unknown as Partial<BriefMutation>);
    mockOpps({ data: undefined, isLoading: true });

    render(<ExecutiveAIBrief brand="Kisqali" />);

    expect(mutate).not.toHaveBeenCalled();
  });

  it('does NOT call the endpoint when the feed has no signal — honest empty, never an ungrounded riff', () => {
    // PR-5 contract change: the old RAG path fired a context-free prompt when
    // the feed failed/was empty, producing exactly the generic "description"
    // the review flagged. Now: no real figures -> no call -> honest empty.
    const mutate = vi.fn();
    mockBrief({ mutate } as unknown as Partial<BriefMutation>);
    mockOpps({ data: undefined, isError: true });

    render(<ExecutiveAIBrief brand="Kisqali" />);

    expect(mutate).not.toHaveBeenCalled();
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    expect(screen.getByText(/run a gap analysis/i)).toBeInTheDocument();
  });

  it('calls the endpoint on suppression-only signal (all below break-even is a real brief)', async () => {
    const mutate = vi.fn();
    mockBrief({ mutate } as unknown as Partial<BriefMutation>);
    mockOpps({
      data: {
        ...OPP_CONTEXT,
        opportunities: [],
        total_count: 0,
        quick_wins_count: 0,
        total_addressable_value: 0,
        suppressed_count: 2,
      },
    });

    render(<ExecutiveAIBrief brand="Fabhalta" />);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(lastRequest(mutate)!.suppressed_count).toBe(2);
  });

  it('clears the prior brand footer (last-updated + count) on a CACHED brand switch', () => {
    // Codex round-2 HIGH(b): the footer leaked brand A's "Last updated" + insight
    // count under brand B. Dynamic mock so reset() actually clears the data,
    // faithful to react-query.
    let briefData: unknown = DISTILLATION;
    const reset = vi.fn(() => { briefData = undefined; });
    vi.mocked(useIns.useExecutiveBriefInsight).mockImplementation(() => ({
      mutate: vi.fn(),
      reset,
      data: briefData,
      error: null,
      isPending: false,
    } as unknown as BriefMutation));
    mockOpps({ data: OPP_CONTEXT });

    const { rerender } = render(<ExecutiveAIBrief brand="Kisqali" />);
    expect(screen.getByText(/1 insight generated/)).toBeInTheDocument();
    expect(screen.getByText(/Last updated:/)).toBeInTheDocument();

    // Switch brand; the new brand's opportunities are already cached (settled).
    mockOpps({ data: OPP_CONTEXT });
    rerender(<ExecutiveAIBrief brand="Fabhalta" />);

    // The prior brand's footer state must not linger.
    expect(screen.queryByText(/Last updated:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/1 insight generated/)).not.toBeInTheDocument();
    expect(screen.getByText(/0 insights generated/)).toBeInTheDocument();
  });

  it('never shows the previous brand brief while the new brand opportunities load (no stale attribution)', () => {
    // Codex round-1 HIGH: gating the fire on !oppLoading meant a brand switch
    // held brand A's brief on screen until brand B's /gaps/opportunities
    // resolved. While the new brand's feed is loading, the brief must show the
    // busy state, never the previous brand's content.
    mockBrief({
      mutate: vi.fn(),
      data: DISTILLATION,
    } as unknown as Partial<BriefMutation>);
    mockOpps({ data: OPP_CONTEXT });

    const { rerender } = render(<ExecutiveAIBrief brand="Kisqali" />);
    expect(screen.getByText(/Prioritize the Northeast TRX gap/)).toBeInTheDocument();

    // Brand switches; the new brand's opportunities are still loading.
    mockOpps({ data: undefined, isLoading: true });
    rerender(<ExecutiveAIBrief brand="Fabhalta" />);

    expect(screen.queryByText(/Prioritize the Northeast TRX gap/)).not.toBeInTheDocument();
  });
});
