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
import { render, screen } from '@testing-library/react';
import { ExecutiveAIBrief } from './ExecutiveAIBrief';
import * as useExec from '@/hooks/api/use-executive-insights';
import * as useCog from '@/hooks/api/use-cognitive';

vi.mock('@/hooks/api/use-executive-insights');
vi.mock('@/hooks/api/use-cognitive');

type RagMutation = ReturnType<typeof useCog.useCognitiveRAG>;
type ExecQuery = ReturnType<typeof useExec.useExecutiveInsights>;

function mockRag(overrides: Partial<RagMutation> = {}) {
  vi.mocked(useCog.useCognitiveRAG).mockReturnValue({
    mutate: vi.fn(),
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
