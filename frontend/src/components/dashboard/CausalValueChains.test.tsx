/**
 * CausalValueChains Tests
 * =======================
 *
 * Verified finding (fix/fe-home-fake-data task 2): SAMPLE_CHAINS fabricated
 * three causal chains ("+12% TRx Accuracy", confidence 0.92, "DoWhy",
 * "2 min ago") and the component returned them on BOTH an empty-but-successful
 * response AND a mutation error (useMutation error leaves `data` undefined),
 * rendered unlabeled under "Primary Causal Value Chains - Live Tracking".
 *
 * The only allowed states are: real data, honest empty, labeled error.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CausalValueChains } from './CausalValueChains';
import type { GraphPath } from '@/types/graph';

vi.mock('@/hooks/api/use-graph', () => ({
  useCausalChains: vi.fn(),
}));

import { useCausalChains } from '@/hooks/api/use-graph';

const mockUseCausalChains = useCausalChains as ReturnType<typeof vi.fn>;

function mockChainsState(state: {
  data?: unknown;
  isPending?: boolean;
  isError?: boolean;
}) {
  mockUseCausalChains.mockReturnValue({
    mutate: vi.fn(),
    data: state.data,
    isPending: state.isPending ?? false,
    isError: state.isError ?? false,
  });
}

/** A minimal real GraphPath as returned by POST /graph/causal-chains. */
function realPath(): GraphPath {
  return {
    nodes: [
      { id: 'n1', type: 'HCP' as never, name: 'Call Frequency', properties: {} },
      { id: 'n2', type: 'KPI' as never, name: 'Rx Propensity', properties: {} },
      { id: 'n3', type: 'KPI' as never, name: 'TRx', properties: { value: 6.4 } },
    ],
    relationships: [
      {
        id: 'r1',
        type: 'INFLUENCES' as never,
        source_id: 'n1',
        target_id: 'n2',
        properties: { method: 'EconML' },
      },
      {
        id: 'r2',
        type: 'INFLUENCES' as never,
        source_id: 'n2',
        target_id: 'n3',
        properties: {},
      },
    ],
    total_confidence: 0.91,
    path_length: 3,
  };
}

/** A real single-edge chain that carries the causal effect on the
 *  RELATIONSHIP as `ate_estimate` — the shape POST /graph/causal-chains
 *  actually returns for discovered chains (verified against the live API). */
function atePath(
  ate: number | null,
  opts: { effectSize?: unknown } = {}
): GraphPath {
  return {
    nodes: [
      { id: 'v1', type: 'Agent' as never, name: 'hcp_engagement_level', properties: {} },
      { id: 'v2', type: 'Agent' as never, name: 'patient_conversion_rate', properties: {} },
    ],
    relationships: [
      {
        id: 'r1',
        type: 'CAUSES' as never,
        source_id: 'v1',
        target_id: 'v2',
        properties: {
          ate_estimate: ate,
          ...(opts.effectSize !== undefined ? { effect_size: opts.effectSize } : {}),
        },
        confidence: 0.9,
      },
    ],
    total_confidence: 0.9,
    path_length: 1,
  };
}

const FABRICATED_MARKERS = [
  '+12% TRx Accuracy',
  '+8.5% Conversion',
  '-4.2pp Gap',
  'Data Quality Impact',
  'HCP Engagement Path',
  'Coverage Equity Chain',
];

describe('CausalValueChains', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders an honest empty state (NOT SAMPLE_CHAINS) when the API succeeds with zero chains', () => {
    mockChainsState({
      data: { chains: [], total_chains: 0, timestamp: new Date().toISOString() },
    });

    render(<CausalValueChains />);

    // The fabricated chains must NOT render on an empty-but-successful response.
    for (const marker of FABRICATED_MARKERS) {
      expect(screen.queryByText(marker)).not.toBeInTheDocument();
    }
    // Honest empty state (repo F-002 convention: EmptyState component).
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    expect(screen.getByText('No causal chains discovered')).toBeInTheDocument();
  });

  it('renders a labeled degraded state (NOT SAMPLE_CHAINS) when the mutation errors', () => {
    // useMutation error: data stays undefined, isError true.
    mockChainsState({ data: undefined, isError: true });

    render(<CausalValueChains />);

    for (const marker of FABRICATED_MARKERS) {
      expect(screen.queryByText(marker)).not.toBeInTheDocument();
    }
    // Clearly labeled degraded state, never unlabeled fakes.
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/causal chains unavailable/i)).toBeInTheDocument();
  });

  it('renders REAL chains from the API response', () => {
    mockChainsState({
      data: {
        chains: [realPath()],
        total_chains: 1,
        aggregate_effect: 0.064,
        timestamp: new Date().toISOString(),
      },
    });

    render(<CausalValueChains />);

    // Real chain card content.
    expect(screen.getByText('Call Frequency → TRx')).toBeInTheDocument();
    expect(screen.getByText('91% confidence')).toBeInTheDocument();
    expect(screen.getByText('EconML')).toBeInTheDocument();
    expect(screen.getByText('+6.4% Impact')).toBeInTheDocument();
    expect(screen.getByText('1 chains discovered')).toBeInTheDocument();
    // No fabricated content alongside real data.
    for (const marker of FABRICATED_MARKERS) {
      expect(screen.queryByText(marker)).not.toBeInTheDocument();
    }
    // No fake recency claim: GraphPath carries no chain timestamp, so none
    // may be invented (codex iter-1 HIGH-2: 'Just now' was a new fabrication).
    expect(screen.queryByText('Just now')).not.toBeInTheDocument();
    expect(screen.queryByText(/min ago/)).not.toBeInTheDocument();
    // No impact-band label: the API provides no impact classification, and
    // deriving 'High Impact' from confidence/path-length is a fabricated
    // magnitude claim (codex iter-6 HIGH). The real magnitude is the result pill.
    expect(screen.queryByText(/High Impact|Medium Impact|Low Impact/)).not.toBeInTheDocument();
  });

  it('renders a quantified ZERO effect as 0.0% Impact, not "Impact not quantified" (codex iter-1 MED-3)', () => {
    const path = realPath();
    path.nodes[2].properties = { value: 0 };

    mockChainsState({
      data: { chains: [path], total_chains: 1, timestamp: new Date().toISOString() },
    });

    render(<CausalValueChains />);

    expect(screen.getByText('0.0% Impact')).toBeInTheDocument();
    expect(screen.queryByText(/impact not quantified/i)).not.toBeInTheDocument();
  });

  it('does NOT fabricate confidence/method when the API omits them (real path honesty)', () => {
    const path = realPath();
    delete (path as Partial<GraphPath>).total_confidence;
    path.relationships[0].properties = {};
    path.nodes[2].properties = {};

    mockChainsState({
      data: { chains: [path], total_chains: 1, timestamp: new Date().toISOString() },
    });

    render(<CausalValueChains />);

    // The old code fabricated `?? 0.8` confidence and `?? 'DoWhy'` method.
    expect(screen.queryByText('80% confidence')).not.toBeInTheDocument();
    expect(screen.queryByText('DoWhy')).not.toBeInTheDocument();
    expect(screen.getByText(/confidence unavailable/i)).toBeInTheDocument();
    // The old code rendered a literal "+X% Impact" placeholder.
    expect(screen.queryByText('+X% Impact')).not.toBeInTheDocument();
    expect(screen.getByText(/impact not quantified/i)).toBeInTheDocument();
  });

  it('derives the chain effect from the relationship ate_estimate (real API shape), as a raw ATE not a fabricated %', () => {
    mockChainsState({
      data: {
        chains: [atePath(0.413004, { effectSize: 'unknown' })],
        total_chains: 1,
        timestamp: new Date().toISOString(),
      },
    });

    render(<CausalValueChains />);

    // The pipeline's real ATE is on the edge — surfaced, not dropped.
    expect(screen.getByText('ATE +0.41')).toBeInTheDocument();
    // The old bug read lastNode.properties.value (never populated) → always
    // "Impact not quantified". That must no longer happen for a real ATE.
    expect(screen.queryByText(/impact not quantified/i)).not.toBeInTheDocument();
    // ate_estimate=0.413 is NOT asserted to be 41.3% — the outcome scale is
    // not claimed, so no fabricated percentage is rendered.
    expect(screen.queryByText(/41\.3?\s*%/)).not.toBeInTheDocument();
    // Full chain title — never silently truncated to 'pati…'.
    expect(
      screen.getByText('hcp_engagement_level → patient_conversion_rate')
    ).toBeInTheDocument();
    expect(screen.queryByText(/…|\.\.\./)).not.toBeInTheDocument();
  });

  it('shows a negative ATE honestly', () => {
    mockChainsState({
      data: { chains: [atePath(-0.2)], total_chains: 1, timestamp: new Date().toISOString() },
    });
    render(<CausalValueChains />);
    expect(screen.getByText('ATE -0.20')).toBeInTheDocument();
  });

  it('prefers the relationship ATE over a legacy terminal-node value', () => {
    const p = atePath(0.413);
    p.nodes[1].properties = { value: 6.4 }; // both present
    mockChainsState({
      data: { chains: [p], total_chains: 1, timestamp: new Date().toISOString() },
    });
    render(<CausalValueChains />);
    expect(screen.getByText('ATE +0.41')).toBeInTheDocument();
    expect(screen.queryByText('+6.4% Impact')).not.toBeInTheDocument();
  });

  it('does NOT treat effect_size as the magnitude — no ATE means "Impact not quantified"', () => {
    // Seed-style edge: effect_size is a bare number, ate_estimate is null.
    // effect_size is a category label in real data, never the magnitude.
    mockChainsState({
      data: {
        chains: [atePath(null, { effectSize: 0.2 })],
        total_chains: 1,
        timestamp: new Date().toISOString(),
      },
    });
    render(<CausalValueChains />);
    expect(screen.getByText(/impact not quantified/i)).toBeInTheDocument();
    expect(screen.queryByText(/ATE/)).not.toBeInTheDocument();
    expect(screen.queryByText('0.2')).not.toBeInTheDocument();
  });

  it('hides the aggregate-effect badge when the API reports null (never a fabricated 0.0%)', () => {
    mockChainsState({
      data: {
        chains: [atePath(0.18)],
        total_chains: 1,
        aggregate_effect: null,
        timestamp: new Date().toISOString(),
      },
    });
    render(<CausalValueChains />);
    expect(screen.queryByText(/aggregate/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/0\.0\s*%/)).not.toBeInTheDocument();
  });

  it('renders the aggregate-effect badge as a raw ATE when the API provides a number', () => {
    mockChainsState({
      data: {
        chains: [atePath(0.18)],
        total_chains: 1,
        aggregate_effect: 0.3,
        timestamp: new Date().toISOString(),
      },
    });
    render(<CausalValueChains />);
    expect(screen.getByText('ATE +0.30 aggregate')).toBeInTheDocument();
  });

  it('shows the loading skeleton while the mutation is pending', () => {
    mockChainsState({ isPending: true });

    const { container } = render(<CausalValueChains />);

    expect(container.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0);
    for (const marker of FABRICATED_MARKERS) {
      expect(screen.queryByText(marker)).not.toBeInTheDocument();
    }
  });

  it('shows the loading skeleton on first mount before the mutation settles (no SAMPLE flash)', () => {
    // Initial mount: mutate() fired in useEffect but state not yet pending.
    mockChainsState({ data: undefined, isPending: false, isError: false });

    const { container } = render(<CausalValueChains />);

    for (const marker of FABRICATED_MARKERS) {
      expect(screen.queryByText(marker)).not.toBeInTheDocument();
    }
    expect(container.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0);
  });
});
