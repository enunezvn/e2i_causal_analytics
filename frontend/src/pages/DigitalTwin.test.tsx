/**
 * DigitalTwin Page Tests
 * ======================
 *
 * Tests for the Digital Twin simulation page.
 *
 * H1/H2 (#705): the page must render ONLY data the backend actually returns
 * (the flat `SimulationResponse` / `SimulationDetailResponse` shape) and show
 * honest empty/loading states — never the previously hardcoded
 * `SAMPLE_SIMULATION` / `SAMPLE_HISTORY` fabrications or the static
 * `2.4s / 68% / 87%` stat cards. These tests pin that honesty.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DigitalTwin from './DigitalTwin';
import {
  InterventionType,
  RecommendationType,
  Recommendation,
  SimulationStatus,
  type SimulationResponse,
  type SimulationDetailResponse,
} from '@/types/digital-twin';

// Mock the digital twin hooks (including useSimulation for history-detail fetch)
vi.mock('@/hooks/api/use-digital-twin', () => ({
  useDigitalTwinHealth: vi.fn(),
  useSimulationHistory: vi.fn(),
  useRunSimulation: vi.fn(),
  useSimulation: vi.fn(),
  useInterventionTypes: vi.fn(),
}));

import {
  useDigitalTwinHealth,
  useSimulationHistory,
  useRunSimulation,
  useSimulation,
  useInterventionTypes,
} from '@/hooks/api/use-digital-twin';

// Create wrapper with QueryClientProvider
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

// ---------------------------------------------------------------------------
// Fixtures — REAL backend shapes (flat SimulationResponse / history items)
// ---------------------------------------------------------------------------

const mockHealth = {
  status: 'healthy',
  service: 'digital-twin',
  models_available: 3,
  simulations_pending: 0,
  last_simulation_at: '2026-01-01T12:00:00Z',
};

// The real history endpoint returns these flat list items.
const mockHistory = {
  simulations: [
    {
      simulation_id: 'real-sim-001',
      created_at: '2026-06-04T10:00:00Z',
      intervention_type: InterventionType.EMAIL_CAMPAIGN,
      brand: 'Remibrutinib',
      ate_estimate: 0.085,
      recommendation_type: RecommendationType.DEPLOY,
    },
    {
      simulation_id: 'real-sim-002',
      created_at: '2026-06-03T14:30:00Z',
      intervention_type: InterventionType.DIGITAL_ENGAGEMENT,
      brand: 'Fabhalta',
      ate_estimate: 0.012,
      recommendation_type: RecommendationType.REFINE,
    },
  ],
  total: 2,
  offset: 0,
  limit: 10,
};

// The real result of a run (POST /simulate) — FLAT, no trx/nrx/roi/sensitivity/projections.
const mockRunResult: SimulationResponse = {
  simulation_id: 'real-sim-run-123',
  model_id: 'model-abc',
  intervention_type: 'hcp_engagement',
  brand: 'Remibrutinib',
  twin_type: 'hcp',
  twin_count: 5000,
  simulated_ate: 0.085,
  simulated_ci_lower: 0.052,
  simulated_ci_upper: 0.118,
  simulated_std_error: 0.017,
  effect_size_cohens_d: 0.42,
  statistical_power: 0.86,
  recommendation: Recommendation.DEPLOY,
  recommendation_rationale: 'Effect is positive and the 95% CI excludes zero.',
  recommended_sample_size: 4000,
  recommended_duration_weeks: 12,
  simulation_confidence: 0.83,
  fidelity_warning: false,
  model_fidelity_score: 0.79,
  status: SimulationStatus.COMPLETED,
  execution_time_ms: 1840,
  is_significant: true,
  effect_direction: 'positive',
  created_at: '2026-06-05T01:00:00Z',
};

// The real detail fetched when a history item is clicked.
const mockDetail: SimulationDetailResponse = {
  ...mockRunResult,
  simulation_id: 'real-sim-001',
  simulated_ate: 0.067,
  simulated_ci_lower: 0.031,
  simulated_ci_upper: 0.103,
  recommendation_rationale: 'Historical detail rationale for sim-001.',
  population_filters: {},
  effect_heterogeneity: {
    by_specialty: {},
    by_decile: {},
    by_region: {},
    by_adoption_stage: {},
    top_segments: [],
  },
  intervention_config: {},
  completed_at: '2026-06-04T10:05:00Z',
};

// Canonical intervention catalog (mirrors backend INTERVENTION_CATALOG) for
// building brand-aware /intervention-types responses in tests.
const INTERVENTION_CATALOG: ReadonlyArray<[string, string]> = [
  ['email_campaign', 'Email Campaign'],
  ['call_frequency_increase', 'Increased Call Frequency'],
  ['speaker_program_invitation', 'Speaker Program Invitation'],
  ['sample_distribution', 'Sample Distribution'],
  ['peer_influence_activation', 'Peer Influence Activation'],
  ['digital_engagement', 'Digital Engagement'],
];
const ALL_INTERVENTION_VALUES = INTERVENTION_CATALOG.map(([v]) => v);

// Build a useInterventionTypes() hook return where only `availableValues` are
// flagged available (the component exposes only available types) and
// `cohortEstimated` values report effect_basis 'cohort_estimated' (Phase 2).
function interventionTypesResult(
  availableValues: string[],
  cohortEstimated: string[] = []
) {
  return {
    data: {
      interventions: INTERVENTION_CATALOG.map(([value, label]) => ({
        value,
        label,
        effect_basis: cohortEstimated.includes(value) ? 'cohort_estimated' : 'synthetic',
        available: availableValues.includes(value),
      })),
      brand: 'Remibrutinib',
      twin_type: 'hcp',
      timestamp: '2026-06-16T00:00:00Z',
    },
    isLoading: false,
    isError: false,
  };
}

// Markers that ONLY appear in the old fabricated shape — must never render.
const FABRICATED_MARKERS = [
  'TRx Lift',
  'NRx Lift',
  'Data Coverage',
  'Calibration',
  'Temporal Alignment',
  'Feature Completeness',
  'Supporting Evidence',
  'Risk Factors',
  'Simulation indicates strong positive ATE', // SAMPLE_SIMULATION rationale
];

describe('DigitalTwin', () => {
  const mockMutate = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    (useDigitalTwinHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockHealth,
      isLoading: false,
    });
    (useSimulationHistory as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockHistory,
      isLoading: false,
      isFetching: false,
    });
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: undefined,
      isError: false,
      error: null,
    });
    // useSimulation returns a detail only when called with a truthy id.
    (useSimulation as ReturnType<typeof vi.fn>).mockImplementation(
      (id: string) => ({
        data: id ? mockDetail : undefined,
        isLoading: false,
        isError: false,
      })
    );
    // Default: all interventions available for the (Remibrutinib) brand.
    (useInterventionTypes as ReturnType<typeof vi.fn>).mockReturnValue(
      interventionTypesResult(ALL_INTERVENTION_VALUES)
    );
  });

  // -------------------------------------------------------------------------
  // Structural tests (still valid after the honest refactor)
  // -------------------------------------------------------------------------

  it('renders page header with title and description', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.getByText('Digital Twin')).toBeInTheDocument();
    expect(
      screen.getByText('Intervention pre-screening and scenario analysis')
    ).toBeInTheDocument();
  });

  it('displays system health status', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.getByText('Healthy')).toBeInTheDocument();
    expect(screen.getByText('3 models available')).toBeInTheDocument();
  });

  it('renders simulation configuration form', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.getByText('Configure Simulation')).toBeInTheDocument();
    expect(screen.getByText('Intervention Type')).toBeInTheDocument();
    expect(screen.getByText('Brand')).toBeInTheDocument();
    expect(screen.getByText('Sample Size')).toBeInTheDocument();
    expect(screen.getByText('Duration (days)')).toBeInTheDocument();
  });

  it('has run simulation button', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(
      screen.getByRole('button', { name: /Run Simulation/i })
    ).toBeInTheDocument();
  });

  it('displays Results and History tabs', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.getByRole('button', { name: /Results/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /History/i })).toBeInTheDocument();
  });

  it('calls runSimulation when form is submitted', async () => {
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });
    const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
    await act(async () => {
      await user.click(submitButton);
    });
    expect(mockMutate).toHaveBeenCalledWith({
      intervention: {
        intervention_type: InterventionType.EMAIL_CAMPAIGN,
        duration_weeks: 13, // Math.ceil(90 / 7)
      },
      brand: 'Remibrutinib',
      twin_count: 1000,
    });
  });

  // -------------------------------------------------------------------------
  // Phase 1b — the intervention dropdown is driven by /intervention-types
  // (brand-aware availability), not a hardcoded list.
  // -------------------------------------------------------------------------

  it('lists ONLY backend-available interventions in the dropdown', () => {
    // Only two interventions available for this brand.
    (useInterventionTypes as ReturnType<typeof vi.fn>).mockReturnValue(
      interventionTypesResult(['email_campaign', 'digital_engagement'])
    );
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // The two available types are present as <option>s …
    expect(screen.getByRole('option', { name: 'Email Campaign' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Digital Engagement' })).toBeInTheDocument();
    // … and the unavailable ones are NOT rendered (would 503 on /simulate).
    expect(screen.queryByRole('option', { name: 'Sample Distribution' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Speaker Program Invitation' })).not.toBeInTheDocument();
  });

  it('disables Run and explains when no twin model exists for the brand', () => {
    (useInterventionTypes as ReturnType<typeof vi.fn>).mockReturnValue(
      interventionTypesResult([]) // no trained model → nothing available
    );
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(
      screen.getByText(/No trained twin model for Remibrutinib yet/i)
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Run Simulation/i })).toBeDisabled();
  });

  // Phase 2 — the dropdown surfaces HOW the selected intervention's effect is
  // computed: cohort-estimated (brand-specific) vs uniform synthetic.
  it('shows the brand cohort–estimated basis note for a cohort-estimated intervention', async () => {
    // Only the cohort-estimable types are available → selection resets to the
    // first available one (digital_engagement, cohort-estimated).
    (useInterventionTypes as ReturnType<typeof vi.fn>).mockReturnValue(
      interventionTypesResult(
        ['digital_engagement', 'call_frequency_increase'],
        ['digital_engagement', 'call_frequency_increase']
      )
    );
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(await screen.findByText(/brand cohort/i)).toBeInTheDocument();
    expect(screen.queryByText(/uniform synthetic uplift/i)).not.toBeInTheDocument();
  });

  it('shows the uniform synthetic basis note for a non-cohort intervention', () => {
    // Default selection is email_campaign (synthetic, not cohort-estimable).
    (useInterventionTypes as ReturnType<typeof vi.fn>).mockReturnValue(
      interventionTypesResult(ALL_INTERVENTION_VALUES, ['digital_engagement'])
    );
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText(/uniform synthetic uplift/i)).toBeInTheDocument();
  });

  it('shows loading state when simulation is running', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: true,
      data: undefined,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    const button = screen.getByRole('button', { name: /Run Simulation/i });
    expect(button).toBeDisabled();
    expect(button.querySelector('.animate-spin')).toBeInTheDocument();
  });

  it('renders about section with intervention types', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.getByText('About the Digital Twin')).toBeInTheDocument();
    expect(screen.getByText('Intervention Types')).toBeInTheDocument();
    expect(screen.getByText('How It Works')).toBeInTheDocument();
  });

  it('handles unknown health status', () => {
    (useDigitalTwinHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.getByText('Unknown')).toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // H1/H2 — honesty tests (RED against the fabricating implementation)
  // -------------------------------------------------------------------------

  it('shows an honest empty results state on initial load (no fabricated simulation)', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: undefined,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Honest empty prompt is shown...
    expect(
      screen.getByText(/Run a simulation to see results/i)
    ).toBeInTheDocument();
    // ...and NONE of the fabricated sample markers are present.
    for (const marker of FABRICATED_MARKERS) {
      expect(screen.queryByText(marker)).not.toBeInTheDocument();
    }
  });

  it('renders the REAL run result (ATE, CI, recommendation, rationale, exec time)', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: mockRunResult,
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Real ATE point estimate + CI bounds from the flat SimulationResponse.
    expect(screen.getByText(/0\.085/)).toBeInTheDocument();
    expect(screen.getByText(/0\.052/)).toBeInTheDocument();
    expect(screen.getByText(/0\.118/)).toBeInTheDocument();
    // Real recommendation + rationale (not the sample text).
    expect(screen.getByText('Deploy')).toBeInTheDocument();
    expect(
      screen.getByText(/Effect is positive and the 95% CI excludes zero/i)
    ).toBeInTheDocument();
    // Real execution time.
    expect(screen.getByText(/1840\s*ms/i)).toBeInTheDocument();
  });

  it('shows a SYNTHETIC badge when the result data_provenance is synthetic', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: { ...mockRunResult, data_provenance: 'synthetic_uplift_v1' },
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.getByText(/^SYNTHETIC$/)).toBeInTheDocument();
  });

  it('does NOT show a SYNTHETIC badge for a non-synthetic provenance', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: { ...mockRunResult, data_provenance: 'database' },
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.queryByText(/^SYNTHETIC$/)).not.toBeInTheDocument();
  });

  it('does NOT render outcome sections the backend never returns', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: mockRunResult,
      isSuccess: true,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    for (const marker of FABRICATED_MARKERS) {
      expect(screen.queryByText(marker)).not.toBeInTheDocument();
    }
    // The sample's signature expected-value line must be gone.
    expect(screen.queryByText(/Expected Value:/i)).not.toBeInTheDocument();
  });

  it('stat cards do not show hardcoded fabricated metrics (2.4s / 68% / 87%)', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.queryByText('2.4s')).not.toBeInTheDocument();
    expect(screen.queryByText('68%')).not.toBeInTheDocument();
    expect(screen.queryByText('87%')).not.toBeInTheDocument();
  });

  it('shows an honest empty history state when history is unavailable (no SAMPLE_HISTORY)', async () => {
    (useSimulationHistory as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isFetching: false,
    });
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });

    const historyTab = screen.getByRole('button', { name: /History/i });
    await act(async () => {
      await user.click(historyTab);
    });

    // Honest empty state, and NO fabricated history rows (rows render "ATE: x.xx").
    await waitFor(() => {
      expect(screen.getByText(/No simulations/i)).toBeInTheDocument();
    });
    expect(screen.queryByText(/ATE:/i)).not.toBeInTheDocument();
  });

  it('clicking a history item loads its REAL detail via useSimulation (not a sample)', async () => {
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Go to history, click the first real item.
    const historyTab = screen.getByRole('button', { name: /History/i });
    await act(async () => {
      await user.click(historyTab);
    });
    // /Email Campaign/i also matches the form's <option>; pick the history row <p>.
    const matches = await screen.findAllByText(/Email Campaign/i);
    const row = matches.find((el) => el.tagName === 'P') ?? matches[matches.length - 1];
    await act(async () => {
      await user.click(row);
    });

    // The detail's distinctive rationale + ATE render (not the sample 0.18).
    await waitFor(() => {
      expect(
        screen.getByText(/Historical detail rationale for sim-001/i)
      ).toBeInTheDocument();
    });
    expect(screen.getByText(/0\.067/)).toBeInTheDocument();
    expect(useSimulation as ReturnType<typeof vi.fn>).toHaveBeenCalledWith(
      'real-sim-001',
      expect.anything()
    );
  });

  it('shows a running indicator during a re-run even when a previous result is displayed', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: true,
      data: mockRunResult, // a previous run result is still in state
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // A run is in flight → show a running indicator, not the stale result silently.
    expect(screen.getByText(/Running simulation/i)).toBeInTheDocument();
    // The stale rationale must not be presented as the current result.
    expect(
      screen.queryByText(/Effect is positive and the 95% CI excludes zero/i)
    ).not.toBeInTheDocument();
  });

  it('shows an error state when a selected history detail fails to load', async () => {
    (useSimulation as ReturnType<typeof vi.fn>).mockImplementation((id: string) => ({
      data: undefined,
      isLoading: false,
      isError: !!id,
      error: { message: 'Simulation not found' },
    }));
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /History/i }));
    });
    const matches = await screen.findAllByText(/Email Campaign/i);
    const row = matches.find((el) => el.tagName === 'P') ?? matches[matches.length - 1];
    await act(async () => {
      await user.click(row);
    });

    // An honest error state — NOT the generic "run a simulation" empty prompt.
    await waitFor(() => {
      expect(screen.getByText(/could not be loaded/i)).toBeInTheDocument();
    });
    expect(
      screen.queryByText(/Run a simulation to see results/i)
    ).not.toBeInTheDocument();
  });
});
