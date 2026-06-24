/**
 * ExecutiveAIBrief Tests
 * ======================
 *
 * Red-first guards for the fake-brief finding: the widget formerly booted
 * `useState(SAMPLE_BRIEF)` ($2.3M, 847 HCPs, beta=0.42, 12.3% MoM TRx) and
 * spliced `...SAMPLE_BRIEF.slice(1)` into EVERY real RAG response with
 * hardcoded 87%/78% confidence — fake sections contaminated real answers.
 *
 * Desired behavior: real crystallized insights, else the real RAG response
 * alone, else an honest empty state. No fabricated sections or confidence.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { ExecutiveAIBrief } from './ExecutiveAIBrief';
import * as useExec from '@/hooks/api/use-executive-insights';
import * as useCog from '@/hooks/api/use-cognitive';
import { useOpportunities } from '@/hooks/api';

vi.mock('@/hooks/api/use-executive-insights');
vi.mock('@/hooks/api/use-cognitive');
// T7a: the brief now grounds its RAG query in the brand's real opportunity
// figures. Mock the opportunities feed so these unit tests stay hermetic.
vi.mock('@/hooks/api', () => ({ useOpportunities: vi.fn() }));

type RagMutation = ReturnType<typeof useCog.useCognitiveRAG>;
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

function mockRag(overrides: Partial<RagMutation> = {}) {
  vi.mocked(useCog.useCognitiveRAG).mockReturnValue({
    mutate: vi.fn(),
    reset: vi.fn(),
    data: undefined,
    error: null,
    isPending: false,
    ...overrides,
  } as unknown as RagMutation);
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

beforeEach(() => {
  vi.clearAllMocks();
  mockRag();
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

  it('renders ONLY the real RAG response — no fake sections spliced in', () => {
    mockRag({
      data: {
        response: 'Kisqali NBRx grew on improved access in the West region.',
        evidence: [],
        hop_count: 1,
        visualization_config: {},
        routed_agents: [],
      },
    } as unknown as Partial<RagMutation>);

    render(<ExecutiveAIBrief brand="Kisqali" />);

    expect(
      screen.getByText(/Kisqali NBRx grew on improved access/)
    ).toBeInTheDocument();

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
    mockRag({
      data: {
        response: 'Single real insight.',
        evidence: [],
        hop_count: 1,
        visualization_config: {},
        routed_agents: [],
      },
    } as unknown as Partial<RagMutation>);

    render(<ExecutiveAIBrief brand="Kisqali" />);
    // Formerly hardcoded "3 insights generated" regardless of content.
    expect(screen.getByText(/1 insight generated/)).toBeInTheDocument();
    expect(screen.queryByText(/3 insights generated/)).not.toBeInTheDocument();
  });

  it('shows a labeled error state when the RAG call fails and nothing else is available', () => {
    mockRag({
      error: new Error('cognitive engine unavailable'),
    } as unknown as Partial<RagMutation>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);
    expect(screen.getByText(/unable to generate brief/i)).toBeInTheDocument();
    expect(screen.queryByText('Key Performance Trend')).not.toBeInTheDocument();
  });
});

describe('ExecutiveAIBrief — in-band error payload must never render as an insight', () => {
  // The cognitive-RAG endpoint reports failures IN-BAND: HTTP 200 whose
  // payload carries a non-empty `error` and whose `response` field holds the
  // error STRING (hop_count 0, evidence []). This is the exact shape observed
  // live when the LangGraph checkpointer rejected a missing thread_id:
  //   "Unable to complete cognitive search: Checkpointer requires one or more
  //    of the following 'configurable' keys: thread_id, checkpoint_ns, ..."
  // Rendering that string as an "AI-Generated Insight" is the #932/#939
  // error-as-data anti-fabrication defect.
  const ERROR_STRING =
    "Unable to complete cognitive search: Checkpointer requires one or more " +
    "of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id";

  it('does NOT render the backend error string as an insight', () => {
    mockRag({
      data: {
        response: ERROR_STRING,
        evidence: [],
        hop_count: 0,
        visualization_config: {},
        routed_agents: [],
        error: ERROR_STRING,
      },
    } as unknown as Partial<RagMutation>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);

    // The error string must NOT appear as a rendered insight section.
    expect(screen.queryByText('AI-Generated Insight')).not.toBeInTheDocument();
    // The footer must NOT claim an insight was generated.
    expect(screen.queryByText(/1 insight generated/)).not.toBeInTheDocument();
    expect(screen.getByText(/0 insights generated/)).toBeInTheDocument();
  });

  it('shows an honest labeled error state carrying the real backend message', () => {
    mockRag({
      data: {
        response: ERROR_STRING,
        evidence: [],
        hop_count: 0,
        visualization_config: {},
        routed_agents: [],
        error: ERROR_STRING,
      },
    } as unknown as Partial<RagMutation>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);

    expect(screen.getByText(/unable to generate brief/i)).toBeInTheDocument();
    // The honest error state surfaces the real backend message (the error key),
    // labeled as a failure — not dressed up as a generated insight.
    expect(screen.getByText(/Checkpointer requires/)).toBeInTheDocument();
    // And the success footer styling/claim must not fire.
    expect(screen.queryByText(/1 insight generated/)).not.toBeInTheDocument();
  });

  it('treats a zero-hop / zero-evidence response as no-insight, not a real brief', () => {
    // Defense in depth: even without an explicit `error`, a degenerate result
    // (no retrieval hops AND no evidence) is not a grounded answer and must
    // not be presented as one.
    mockRag({
      data: {
        response: 'placeholder-shaped response with no grounding',
        evidence: [],
        hop_count: 0,
        visualization_config: {},
        routed_agents: [],
      },
    } as unknown as Partial<RagMutation>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);

    expect(screen.queryByText('AI-Generated Insight')).not.toBeInTheDocument();
    expect(
      screen.queryByText(/placeholder-shaped response/)
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/1 insight generated/)).not.toBeInTheDocument();
    // No transport error + no real answer => honest empty state.
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });

  it('still renders a genuine grounded answer (hop_count>0, no error)', () => {
    // Guardrail against over-gating: a real answer must still render.
    mockRag({
      data: {
        response: 'Top prescribing gaps are concentrated in the West region.',
        evidence: [{ content: 'West region NBRx lag', source: 'kpi' }],
        hop_count: 2,
        visualization_config: {},
        routed_agents: [],
      },
    } as unknown as Partial<RagMutation>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);

    expect(screen.getByText('AI-Generated Insight')).toBeInTheDocument();
    expect(
      screen.getByText(/Top prescribing gaps are concentrated in the West region/)
    ).toBeInTheDocument();
    expect(screen.getByText(/1 insight generated/)).toBeInTheDocument();
    // A real answer stamps the footer timestamp (not "Not yet generated").
    expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    expect(screen.queryByText(/Not yet generated/)).not.toBeInTheDocument();
  });

  it('does NOT stamp "Last updated" when the response is an in-band error', () => {
    mockRag({
      data: {
        response: ERROR_STRING,
        evidence: [],
        hop_count: 0,
        visualization_config: {},
        routed_agents: [],
        error: ERROR_STRING,
      },
    } as unknown as Partial<RagMutation>);

    render(<ExecutiveAIBrief brand="Remibrutinib" />);
    // An error payload is not a successful update.
    expect(screen.getByText(/Not yet generated/)).toBeInTheDocument();
    expect(screen.queryByText(/Last updated:/)).not.toBeInTheDocument();
  });
});

describe('ExecutiveAIBrief — query is grounded in real opportunity figures (T7a)', () => {
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

  function lastQuery(mutate: MockFn): string {
    const calls = mutate.mock.calls;
    const call = calls[calls.length - 1];
    return (call?.[0] as { query: string } | undefined)?.query ?? '';
  }

  it('grounds the generated brief query in the real opportunity figures once they load', async () => {
    const mutate = vi.fn();
    mockRag({ mutate } as unknown as Partial<RagMutation>);
    mockOpps({ data: OPP_CONTEXT });

    render(<ExecutiveAIBrief brand="Kisqali" />);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    const q = lastQuery(mutate);
    expect(q).toContain('Kisqali');
    expect(q).toContain('Expand specialty coverage in the Northeast');
    expect(q).toMatch(/\$2\.4M/);
  });

  it('waits for opportunities to settle before generating (no premature context-free brief)', () => {
    const mutate = vi.fn();
    mockRag({ mutate } as unknown as Partial<RagMutation>);
    mockOpps({ data: undefined, isLoading: true });

    render(<ExecutiveAIBrief brand="Kisqali" />);

    expect(mutate).not.toHaveBeenCalled();
  });

  it('degrades to the basic prompt (no fabricated numbers) when opportunities fail to load', async () => {
    const mutate = vi.fn();
    mockRag({ mutate } as unknown as Partial<RagMutation>);
    mockOpps({ data: undefined, isError: true });

    render(<ExecutiveAIBrief brand="Kisqali" />);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    const q = lastQuery(mutate);
    expect(q).toContain('Kisqali');
    expect(q).not.toMatch(/\$\d/);
  });

  it('never shows the previous brand brief while the new brand opportunities load (no stale attribution)', () => {
    // Codex round-1 HIGH: gating the fire on !oppLoading meant a brand switch
    // held brand A's brief on screen until brand B's /gaps/opportunities
    // resolved. While the new brand's feed is loading, the brief must show the
    // busy state, never the previous brand's content.
    mockRag({
      mutate: vi.fn(),
      data: {
        response: 'Kisqali real insight from the West region.',
        evidence: [{ content: 'West region NBRx', source: 'kpi' }],
        hop_count: 2,
        visualization_config: {},
        routed_agents: [],
      },
    } as unknown as Partial<RagMutation>);
    mockOpps({ data: OPP_CONTEXT });

    const { rerender } = render(<ExecutiveAIBrief brand="Kisqali" />);
    expect(screen.getByText(/Kisqali real insight/)).toBeInTheDocument();

    // Brand switches; the new brand's opportunities are still loading.
    mockOpps({ data: undefined, isLoading: true });
    rerender(<ExecutiveAIBrief brand="Fabhalta" />);

    expect(screen.queryByText(/Kisqali real insight/)).not.toBeInTheDocument();
  });
});
