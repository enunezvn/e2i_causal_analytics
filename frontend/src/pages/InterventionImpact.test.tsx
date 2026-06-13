/**
 * InterventionImpact Page Tests
 * =============================
 *
 * F-002 history: SAMPLE_IMPACT_DATA / SAMPLE_TREATMENT_EFFECTS /
 * SAMPLE_BEFORE_AFTER / SAMPLE_SEGMENT_EFFECTS were removed earlier and
 * the analysis tabs render explicit empty states.
 *
 * This round removes the remaining fabrications (adversarially verified):
 * - The INTERVENTIONS selector catalog: four invented pharma programs
 *   ("Q1 2024 HCP Engagement Campaign" etc.) presented as real program
 *   records. No backend interventions-catalog endpoint exists (verified
 *   against the live OpenAPI spec), so the selector is honestly gated.
 * - `results={null}` on ScenarioResults: real simulations ran but their
 *   results were never displayed. Now wired to the real response.
 * - console.log no-op action buttons on RecommendationCards: callbacks
 *   removed so the (callback-gated) buttons no longer fake capability.
 * - Unreachable fabricated narratives ("Positive Impact Detected",
 *   "High-Volume HCPs ... +112.5 units") deleted from the source.
 */

import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import InterventionImpact from './InterventionImpact';
import type { SimulationRequest, SimulationResponse } from '@/types/digital-twin';
import { Recommendation, SimulationStatus } from '@/types/digital-twin';

// ----------------------------------------------------------------------------
// Mocks
// ----------------------------------------------------------------------------

const hoisted = vi.hoisted(() => ({
  mockMutate: vi.fn(),
  simulationState: {
    isPending: false,
    response: null as unknown,
  },
}));

// The digital-twin child components are mocked to thin markers that surface
// the props the page passes down (the wiring under test).
vi.mock('@/components/digital-twin', () => ({
  SimulationPanel: ({
    onSimulate,
    isSimulating,
  }: {
    onSimulate: (req: SimulationRequest) => void;
    isSimulating?: boolean;
  }) => (
    <div data-testid="simulation-panel" data-simulating={String(isSimulating)}>
      <button
        onClick={() =>
          onSimulate({
            intervention_type: 'hcp_engagement',
            brand: 'Kisqali',
            duration_days: 28,
            sample_size: 500,
          } as unknown as SimulationRequest)
        }
      >
        Run Simulation
      </button>
    </div>
  ),
  ScenarioResults: ({
    results,
    isLoading,
  }: {
    results: SimulationResponse | null;
    isLoading?: boolean;
  }) => (
    <div data-testid="scenario-results" data-loading={String(isLoading)}>
      {results ? `ate:${results.simulated_ate}` : 'results:null'}
    </div>
  ),
  RecommendationCards: ({
    recommendation,
    onAccept,
    onRefine,
    onAnalyze,
  }: {
    recommendation: { type: string } | null;
    onAccept?: () => void;
    onRefine?: () => void;
    onAnalyze?: () => void;
  }) => (
    <div
      data-testid="recommendation-cards"
      data-onaccept={typeof onAccept}
      data-onrefine={typeof onRefine}
      data-onanalyze={typeof onAnalyze}
    >
      {recommendation ? `rec:${recommendation.type}` : 'rec:null'}
    </div>
  ),
}));

// useRunSimulation: calling mutate() invokes the page's onSuccess with a
// REAL-shaped SimulationResponse (mirrors the live OpenAPI schema).
vi.mock('@/hooks/api/use-digital-twin', () => ({
  useRunSimulation: (options?: { onSuccess?: (data: unknown) => void }) => ({
    mutate: (req: unknown) => {
      hoisted.mockMutate(req);
      options?.onSuccess?.(hoisted.simulationState.response);
    },
    isPending: hoisted.simulationState.isPending,
  }),
}));

const REAL_SIMULATION: SimulationResponse = {
  simulation_id: 'sim_real_1',
  model_id: 'twin_v1',
  intervention_type: 'hcp_engagement',
  brand: 'Kisqali',
  twin_type: 'hcp',
  twin_count: 500,
  simulated_ate: 0.142,
  simulated_ci_lower: 0.081,
  simulated_ci_upper: 0.203,
  simulated_std_error: 0.031,
  effect_size_cohens_d: 0.46,
  statistical_power: 0.83,
  recommendation: Recommendation.DEPLOY,
  recommendation_rationale: 'Significant positive effect.',
  simulation_confidence: 0.78,
  fidelity_warning: false,
  status: SimulationStatus.COMPLETED,
  execution_time_ms: 2150,
  is_significant: true,
  effect_direction: 'positive',
  created_at: '2026-06-12T03:00:00Z',
};

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

beforeEach(() => {
  vi.clearAllMocks();
  hoisted.simulationState.isPending = false;
  hoisted.simulationState.response = REAL_SIMULATION;
});

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

describe('InterventionImpact', () => {
  it('renders page header with title and description', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByText('Intervention Impact')).toBeInTheDocument();
    expect(
      screen.getByText(/Before\/after comparisons, treatment effects, and counterfactual analysis/i),
    ).toBeInTheDocument();
  });

  it('displays 5 main tabs', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByRole('tab', { name: /Causal Impact/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Before\/After/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Treatment Effects/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Segment Analysis/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Digital Twin/i })).toBeInTheDocument();
  });
});

describe('InterventionImpact — fabricated intervention catalog removed', () => {
  it('does NOT render the invented intervention programs or their selector', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.queryByText('Q1 2024 HCP Engagement Campaign')).not.toBeInTheDocument();
    expect(screen.queryByText('Digital Rep Training Program')).not.toBeInTheDocument();
    expect(screen.queryByText('Kisqali Patient Support Enhancement')).not.toBeInTheDocument();
    expect(screen.queryByText('Remibrutinib Launch Preparation')).not.toBeInTheDocument();
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
  });

  it('explains the missing interventions catalog honestly', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByText(/no intervention catalog/i)).toBeInTheDocument();
  });
});

describe('InterventionImpact - F-002 empty states', () => {
  it('renders empty state on Causal Impact tab when no API data', () => {
    render(<InterventionImpact />, { wrapper: createWrapper() });

    expect(screen.getByText(/No causal impact data available/)).toBeInTheDocument();
    expect(screen.queryByText('Positive Impact Detected')).not.toBeInTheDocument();
  });

  it('renders empty state on Before/After tab when no API data', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Before\/After/i }));

    expect(screen.getByText(/No before\/after data available/)).toBeInTheDocument();
    expect(screen.queryByText('Detailed Comparison')).not.toBeInTheDocument();
  });

  it('renders empty state on Treatment Effects tab when no API data', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Treatment Effects/i }));

    expect(screen.getByText(/No treatment effect estimates available/)).toBeInTheDocument();
    expect(screen.queryByText('large effect')).not.toBeInTheDocument();
  });

  it('renders empty state on Segment Analysis tab when no API data', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Segment Analysis/i }));

    expect(screen.getByText(/No segment heterogeneity data available/)).toBeInTheDocument();
    expect(screen.queryByText('High-Volume HCPs')).not.toBeInTheDocument();
    expect(screen.queryByText('Northeast Region')).not.toBeInTheDocument();
  });
});

describe('InterventionImpact - Digital Twin tab (real wiring)', () => {
  it('navigates to Digital Twin tab', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Digital Twin/i }));
    expect(screen.getByText('About Digital Twin Simulation')).toBeInTheDocument();
  });

  it('passes the REAL simulation response to ScenarioResults after a run', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Digital Twin/i }));

    // Before a run: honest null (empty state inside ScenarioResults).
    expect(screen.getByTestId('scenario-results')).toHaveTextContent('results:null');

    await user.click(screen.getByRole('button', { name: /Run Simulation/i }));

    // The page formerly hardcoded results={null} (TODO) — the run's real
    // response must now reach the results panel.
    expect(hoisted.mockMutate).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId('scenario-results')).toHaveTextContent('ate:0.142');
  });

  it('does NOT wire console.log no-op handlers into RecommendationCards', async () => {
    const user = userEvent.setup();
    render(<InterventionImpact />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Digital Twin/i }));

    const cards = screen.getByTestId('recommendation-cards');
    // No deployment / refinement / analysis flow exists yet; passing
    // console.log stubs faked those capabilities. Callbacks must be absent
    // (RecommendationCards hides its action buttons when they are).
    expect(cards).toHaveAttribute('data-onaccept', 'undefined');
    expect(cards).toHaveAttribute('data-onrefine', 'undefined');
    expect(cards).toHaveAttribute('data-onanalyze', 'undefined');
  });
});

// M4 regression guard (extended): pins the honest state so a future edit
// re-introducing fabricated fixtures fails CI.
describe('InterventionImpact - source regression guards', () => {
  const src = fs.readFileSync(
    path.join(process.cwd(), 'src/pages/InterventionImpact.tsx'),
    'utf8',
  );

  it('source contains no SAMPLE_* analysis fixtures (M4 guard)', () => {
    expect(src).not.toMatch(
      /SAMPLE_IMPACT_DATA|SAMPLE_TREATMENT_EFFECTS|SAMPLE_BEFORE_AFTER|SAMPLE_SEGMENT_EFFECTS/,
    );
  });

  it('source contains no fabricated INTERVENTIONS catalog', () => {
    expect(src).not.toMatch(/const INTERVENTIONS/);
    expect(src).not.toMatch(/Q1 2024 HCP Engagement Campaign/);
  });

  it('source contains no console.log no-op action handlers', () => {
    expect(src).not.toMatch(/console\.log\(/);
  });

  it('source contains no unreachable fabricated narratives', () => {
    expect(src).not.toMatch(/Positive Impact Detected/);
    expect(src).not.toMatch(/\+112\.5 units/);
  });
});
