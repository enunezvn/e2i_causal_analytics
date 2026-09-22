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

import { describe, it, expect, vi, beforeEach, expectTypeOf } from 'vitest';
import { render, screen, waitFor, act, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DigitalTwin from './DigitalTwin';
import {
  EstimateScope,
  InterventionType,
  RecommendationType,
  Recommendation,
  SimulationStatus,
  FidelityStatus,
  type SimulationResponse,
  type SimulationDetailResponse,
  type TwinModelSummary,
} from '@/types/digital-twin';
import type { components } from '@/types/generated/api';

// Mock the digital twin hooks (including useSimulation for history-detail fetch)
vi.mock('@/hooks/api/use-digital-twin', () => ({
  useDigitalTwinHealth: vi.fn(),
  useSimulationHistory: vi.fn(),
  useRunSimulation: vi.fn(),
  useSimulation: vi.fn(),
  useInterventionTypes: vi.fn(),
  useTwinModels: vi.fn(),
}));

// Strategic Interpretation hook (barrel import in the page).
vi.mock('@/hooks/api', () => ({
  useDigitalTwinInsight: vi.fn(),
}));

import {
  useDigitalTwinHealth,
  useSimulationHistory,
  useRunSimulation,
  useSimulation,
  useInterventionTypes,
  useTwinModels,
} from '@/hooks/api/use-digital-twin';
import { useDigitalTwinInsight } from '@/hooks/api';

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
  intervention_type: 'digital_engagement',
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
  fidelity_status: FidelityStatus.VALIDATED,
  status: SimulationStatus.COMPLETED,
  execution_time_ms: 1840,
  is_significant: true,
  effect_direction: 'positive',
  created_at: '2026-06-05T01:00:00Z',
  effect_heterogeneity: {
    by_specialty: {},
    by_decile: {},
    by_region: {},
    by_adoption_stage: {},
    top_segments: [],
    axis_provenance: {},
  },
  subgroups_basis: 'cohort_rows',
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
    axis_provenance: {},
  },
  intervention_config: {},
  subgroups_basis: 'per_twin',
  completed_at: '2026-06-04T10:05:00Z',
};

// The real /digital-twin/models rows as prod holds them (#2206): three brand
// labels over ONE seed-0 synthetic fit — identical fingerprint, brand not a
// feature, R² scored on the synthetic target, fidelity never measured.
function sharedFitModel(brand: string, over: Partial<TwinModelSummary> = {}): TwinModelSummary {
  return {
    model_id: `model-${brand}`,
    model_name: `hcp_twin_${brand}`,
    twin_type: 'hcp',
    brand,
    algorithm: 'random_forest',
    r2_score: 0.8104269688784731,
    training_samples: 2000,
    is_active: true,
    created_at: '2026-06-16T00:00:00Z',
    fidelity_status: FidelityStatus.UNVALIDATED,
    fidelity_score: null,
    fidelity_sample_count: 0,
    data_provenance: 'synthetic',
    r2_score_basis: 'synthetic_target',
    brand_is_feature: false,
    training_fingerprint: 'a1b2c3d4e5f60718',
    shared_fit_model_count: 3,
    shared_fit_with: ['Remibrutinib', 'Fabhalta', 'Kisqali'].filter((b) => b !== brand),
    ...over,
  };
}
const mockModels = {
  total_count: 3,
  models: [sharedFitModel('Remibrutinib'), sharedFitModel('Fabhalta'), sharedFitModel('Kisqali')],
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
  ['patient_support_program', 'Patient Support Program'],
  ['rep_training_quality', 'Rep Training Quality'],
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
        available_for_effect: availableValues.includes(value),
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
// NOTE: 'Supporting Evidence' was removed from this list — it is now a real,
// data-driven section derived from backend fields (T10 port from Intervention
// Impact page). It renders only when a completed simulation is present.
const FABRICATED_MARKERS = [
  'TRx Lift',
  'NRx Lift',
  'Data Coverage',
  'Calibration',
  'Temporal Alignment',
  'Feature Completeness',
  'Risk Factors',
  'Simulation indicates strong positive ATE', // SAMPLE_SIMULATION rationale
];

describe('DigitalTwin', () => {
  const mockMutate = vi.fn();
  const mockInsightMutate = vi.fn();

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
    (useTwinModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockModels,
      isLoading: false,
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
    // Strategic Interpretation card: idle (not yet generated).
    (useDigitalTwinInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockInsightMutate,
      isPending: false,
      data: undefined,
      error: null,
    });
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

  // -------------------------------------------------------------------------
  // Strategic Interpretation card (LLM-grounded, server-derived grounding)
  // -------------------------------------------------------------------------

  it('renders the Strategic Interpretation card and generates for the selected brand', async () => {
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText(/Strategic Interpretation/i)).toBeInTheDocument();
    await act(async () => {
      await user.click(screen.getByRole('button', { name: /Generate strategic insight/i }));
    });
    // Grounding is server-derived; the page only sends the brand under configuration.
    expect(mockInsightMutate).toHaveBeenCalledWith({ brand: 'Remibrutinib' });
  });

  it('renders the returned twin-program insight with grounding chips', () => {
    (useDigitalTwinInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockInsightMutate,
      isPending: false,
      error: null,
      data: {
        insight: 'Digital engagement is the strongest identified lever for Remibrutinib.',
        key_takeaways: ['Pre-screen speaker programs next'],
        grounding: [
          { label: 'Twin models', value: '1' },
          { label: 'Identified interventions', value: '8/8' },
        ],
        is_fallback: false,
        provenance: 'Digital-twin simulation program (server-derived)',
        generated_at: '2026-07-08T00:00:00Z',
      },
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(
      screen.getByText('Digital engagement is the strongest identified lever for Remibrutinib.')
    ).toBeInTheDocument();
    expect(screen.getByText('Pre-screen speaker programs next')).toBeInTheDocument();
    // Chip text is split across nodes (<span>label</span>: value) — match textContent.
    expect(
      screen.getByText((_, el) => el?.textContent === 'Identified interventions: 8/8')
    ).toBeInTheDocument();
  });

  it('renders the full 8-intervention catalog when every effect is identified', () => {
    // Post-mig-099 substrate: every canonical intervention (incl. the two new
    // program-level levers) has a planted, identified effect.
    (useInterventionTypes as ReturnType<typeof vi.fn>).mockReturnValue(
      interventionTypesResult(ALL_INTERVENTION_VALUES, ALL_INTERVENTION_VALUES)
    );
    render(<DigitalTwin />, { wrapper: createWrapper() });

    for (const [, label] of INTERVENTION_CATALOG) {
      expect(screen.getByRole('option', { name: label })).toBeInTheDocument();
    }
    expect(screen.getByRole('option', { name: 'Patient Support Program' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Rep Training Quality' })).toBeInTheDocument();
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

  it('names the missing cohort effect data, not a missing model, when a trained model exists', () => {
    // The live payload of 2026-09-21: a trained model per brand (available) but no
    // treatment channel identified in the cohort (available_for_effect false for all).
    // The helper above couples the two flags, so this state is built explicitly.
    (useInterventionTypes as ReturnType<typeof vi.fn>).mockReturnValue({
      ...interventionTypesResult([]),
      data: {
        ...interventionTypesResult([]).data,
        interventions: INTERVENTION_CATALOG.map(([value, label]) => ({
          value,
          label,
          effect_basis: 'unavailable',
          available: true,
          available_for_effect: false,
        })),
      },
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.queryByText(/No trained twin model/i)).not.toBeInTheDocument();
    expect(
      screen.getByText(/trained twin model exists for Remibrutinib.*no usable treatment data/i)
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Run Simulation/i })).toBeDisabled();
  });

  it('says availability could not be verified, never "no model", when the model lookup failed', () => {
    // codex r3: a repository outage rendered as an ESTABLISHED absence of a model.
    (useInterventionTypes as ReturnType<typeof vi.fn>).mockReturnValue({
      ...interventionTypesResult([]),
      data: {
        ...interventionTypesResult([]).data,
        model_resolution: 'unavailable',
        effect_availability_status: null,
      },
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText(/could not be verified/i)).toBeInTheDocument();
    expect(screen.queryByText(/No trained twin model/i)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Run Simulation/i })).toBeDisabled();
  });

  it('says availability could not be verified, never "restore the cohort", when the probe failed', () => {
    // codex r2: a connection blip used to render as measured-missing data.
    (useInterventionTypes as ReturnType<typeof vi.fn>).mockReturnValue({
      ...interventionTypesResult([]),
      data: {
        ...interventionTypesResult([]).data,
        effect_availability_status: 'unmeasured',
        interventions: INTERVENTION_CATALOG.map(([value, label]) => ({
          value,
          label,
          effect_basis: 'unavailable',
          available: true,
          available_for_effect: false,
        })),
      },
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText(/could not be verified/i)).toBeInTheDocument();
    expect(screen.queryByText(/no usable treatment data/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/No trained twin model/i)).not.toBeInTheDocument();
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
    // Each value now renders EXACTLY twice: once in the metrics grid and once in
    // the Supporting Evidence list. Pinning to 2 catches a regression where the
    // evidence section stops rendering (the grid alone would yield 1).
    expect(screen.getAllByText(/0\.085/)).toHaveLength(2); // ATE (grid value + evidence line)
    expect(screen.getAllByText(/0\.052/)).toHaveLength(2); // CI lower (grid hint + evidence line)
    expect(screen.getAllByText(/0\.118/)).toHaveLength(2); // CI upper (grid hint + evidence line)
    // Real recommendation + rationale (not the sample text).
    expect(screen.getByText('Deploy')).toBeInTheDocument();
    expect(
      screen.getByText(/Effect is positive and the 95% CI excludes zero/i)
    ).toBeInTheDocument();
    // Real execution time.
    expect(screen.getByText(/1840\s*ms/i)).toBeInTheDocument();
  });

  it('labels a region-filtered result with its regions and shows the cohort-wide effect (#2023)', () => {
    // A region filter narrows the ESTIMATE, not just the twins: the headline ATE/CI are
    // the targeted regions'. The page must say which regions, and keep the cohort-wide
    // effect visible so the two numbers are never confused for each other.
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: {
        ...mockRunResult,
        simulated_ate: 0.2585,
        simulated_ci_lower: 0.1995,
        simulated_ci_upper: 0.3174,
        target_regions: ['northeast'],
        cohort_effect: 0.1352,
        cohort_ci_lower: 0.0924,
        cohort_ci_upper: 0.1781,
      },
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // The headline is labelled with the scope it was estimated on.
    expect(screen.getAllByText(/estimated on northeast/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/ATE · northeast/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/0\.259/).length).toBeGreaterThan(0);
    // The cohort-wide effect is reported alongside, not replaced.
    expect(screen.getAllByText(/cohort-wide.*0\.135/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Cohort-wide effect \(all regions\).*0\.092.*0\.178/i)).toBeInTheDocument();
  });

  // #2053 — the backend states what the headline effect was estimated ON. A stored
  // simulation written before that was recorded is 'unknown', and must not read as
  // cohort-wide just because it carries no regions.
  async function openHistoryDetail(detail: SimulationDetailResponse) {
    (useSimulation as ReturnType<typeof vi.fn>).mockImplementation((id: string) => ({
      data: id ? detail : undefined,
      isLoading: false,
      isError: false,
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
    await screen.findByText(/Historical detail rationale for sim-001/i);
  }

  it('labels a stored region-scoped simulation with its regions and the cohort-wide effect (#2053)', async () => {
    await openHistoryDetail({
      ...mockDetail,
      estimate_scope: EstimateScope.REGIONS,
      target_regions: ['midwest'],
      population_filters: { regions: ['midwest'] },
      cohort_effect: 0.1352,
      cohort_ci_lower: 0.0924,
      cohort_ci_upper: 0.1781,
    });

    expect(screen.getAllByText(/estimated on midwest/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/ATE · midwest/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/cohort-wide.*0\.135/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/scope not recorded/i)).not.toBeInTheDocument();
  });

  it('shows a cohort-wide result as a plain ATE with no scope note (#2053)', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: { ...mockRunResult, estimate_scope: EstimateScope.COHORT, target_regions: [] },
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText('ATE')).toBeInTheDocument();
    expect(screen.queryByText(/estimated on/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/scope not recorded/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/cohort-wide effect \(all regions\)/i)).not.toBeInTheDocument();
  });

  it('notes an unrecorded scope on a region-filtered stored simulation, naming its regions (#2053)', async () => {
    // A pre-#2053 row filtered to northeast: whether its ATE is northeast's or the cohort's
    // was never stored, so it may cover only northeast. It is labelled neither way.
    await openHistoryDetail({
      ...mockDetail,
      estimate_scope: EstimateScope.UNKNOWN,
      target_regions: [],
      population_filters: { regions: ['northeast'] },
    });

    expect(
      screen.getByText(/scope not recorded — this effect may cover only northeast/i)
    ).toBeInTheDocument();
    expect(screen.queryByText(/estimated on/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/ATE · /)).not.toBeInTheDocument();
  });

  it('adds no note to an unknown-scope stored simulation that had no regions filter (#2053)', async () => {
    await openHistoryDetail({
      ...mockDetail,
      estimate_scope: EstimateScope.UNKNOWN,
      target_regions: [],
      population_filters: { regions: [] },
    });

    expect(screen.queryByText(/scope not recorded/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/estimated on/i)).not.toBeInTheDocument();
    expect(screen.getByText('ATE')).toBeInTheDocument();
  });

  it('names a region-scoped history row and leaves unfiltered unknown and cohort rows plain (#2053)', async () => {
    // An unknown row with no regions filter (filter_regions absent or empty) stays plain.
    (useSimulationHistory as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        ...mockHistory,
        simulations: [
          {
            ...mockHistory.simulations[0],
            estimate_scope: EstimateScope.REGIONS,
            target_regions: ['northeast'],
          },
          {
            ...mockHistory.simulations[1],
            estimate_scope: EstimateScope.UNKNOWN,
            target_regions: [],
          },
          {
            simulation_id: 'real-sim-003',
            created_at: '2026-06-02T09:00:00Z',
            intervention_type: InterventionType.SAMPLE_DISTRIBUTION,
            brand: 'Kisqali',
            ate_estimate: 0.04,
            recommendation_type: RecommendationType.SKIP,
            estimate_scope: EstimateScope.COHORT,
            target_regions: [],
          },
        ],
        total: 3,
      },
      isLoading: false,
      isFetching: false,
    });
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });
    await act(async () => {
      await user.click(screen.getByRole('button', { name: /History/i }));
    });

    const rowOf = (ate: string) =>
      screen.getByText((_, el) => el?.tagName === 'P' && !!el.textContent?.startsWith(`ATE: ${ate}`));
    expect(rowOf('0.09').textContent).toMatch(/northeast/);
    expect(rowOf('0.01').textContent).toBe('ATE: 0.01');
    expect(rowOf('0.04').textContent).toBe('ATE: 0.04');
  });

  it('notes the unrecorded scope on a region-filtered unknown history row only (#2079)', async () => {
    // The history payload now carries the stored regions filter, so a card applies the
    // detail view's rule: an unknown-scope row filtered to regions may cover only them.
    (useSimulationHistory as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        ...mockHistory,
        simulations: [
          {
            ...mockHistory.simulations[0],
            estimate_scope: EstimateScope.UNKNOWN,
            target_regions: [],
            filter_regions: ['midwest'],
          },
          {
            ...mockHistory.simulations[1],
            estimate_scope: EstimateScope.UNKNOWN,
            target_regions: [],
            filter_regions: [],
          },
          {
            simulation_id: 'real-sim-003',
            created_at: '2026-06-02T09:00:00Z',
            intervention_type: InterventionType.SAMPLE_DISTRIBUTION,
            brand: 'Kisqali',
            ate_estimate: 0.04,
            recommendation_type: RecommendationType.SKIP,
            // A cohort-wide row keeps a plain ATE whatever its filter said.
            estimate_scope: EstimateScope.COHORT,
            target_regions: [],
            filter_regions: ['south'],
          },
        ],
        total: 3,
      },
      isLoading: false,
      isFetching: false,
    });
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });
    await act(async () => {
      await user.click(screen.getByRole('button', { name: /History/i }));
    });

    const rowOf = (ate: string) =>
      screen.getByText((_, el) => el?.tagName === 'P' && !!el.textContent?.startsWith(`ATE: ${ate}`));
    // The qualifier is a separate span spaced by margin, so textContent has no space before it.
    expect(rowOf('0.09').textContent).toBe('ATE: 0.09· scope not recorded — may cover only midwest');
    expect(rowOf('0.01').textContent).toBe('ATE: 0.01');
    expect(rowOf('0.04').textContent).toBe('ATE: 0.04');
  });

  it('renders a Supporting Evidence list derived from the fixture values', async () => {
    // Reuse the exact mockRunResult fixture (is_significant: true,
    // effect_size_cohens_d: 0.42, statistical_power: 0.86,
    // simulated_ate: 0.085, CI [0.052, 0.118]) — no fixture extension needed.
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: mockRunResult,
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Locate the evidence SECTION (the heading's container) and scope all
    // value assertions to it — proving the bullets reflect the fixture, not a
    // hardcoded constant block that merely shares the labels.
    const heading = await screen.findByText('Supporting Evidence');
    const section = heading.closest('div') as HTMLElement;
    expect(section).not.toBeNull();
    const ev = within(section);

    // Label text + every DERIVED value from the fixture must appear here:
    expect(ev.getByText(/statistically significant/i)).toBeInTheDocument();
    expect(ev.getByText(/ATE: 0\.085/)).toBeInTheDocument(); // simulated_ate
    expect(ev.getByText(/Cohen's d\): 0\.42/)).toBeInTheDocument(); // effect_size_cohens_d
    expect(ev.getByText(/Statistical power: 86%/)).toBeInTheDocument(); // 0.86 → 86%
    expect(ev.getByText(/95% CI: \[0\.052, 0\.118\]/)).toBeInTheDocument(); // CI bounds
  });

  it('shows supported specialty effects with their cohort provenance and fallback rule', async () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      isError: false,
      error: null,
      data: {
        ...mockRunResult,
        subgroups_basis: 'cohort_rows',
        effect_heterogeneity: {
          by_specialty: {
            oncology: { ate: 0.21, std: 0, n: 360 },
            hematology: { ate: 0.08, std: 0, n: 300 },
          },
          by_decile: {},
          by_region: {},
          by_adoption_stage: {},
          top_segments: [],
          axis_provenance: {
            specialty: {
              basis: 'cohort_rows',
              source: 'hcp_profiles.specialty',
              min_group_rows: 100,
              min_treated_rows: 20,
              min_control_rows: 20,
              fallback: 'region_then_cohort',
              support_unit: 'cohort_rows',
              estimand: 'observed_region_mix_mean_cate',
              suppressed_groups: { rare: 'group_rows_below_minimum' },
            },
          },
        },
      },
    });

    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(await screen.findByText('Specialty Effects')).toBeInTheDocument();
    expect(screen.getByText('oncology')).toBeInTheDocument();
    expect(screen.getByText('360 cohort rows')).toBeInTheDocument();
    expect(screen.getByText(/hcp_profiles\.specialty/)).toBeInTheDocument();
    expect(screen.getByText(/Observed-region-mix CATE/)).toBeInTheDocument();
    expect(screen.getByText(/not distinct HCPs/)).toBeInTheDocument();
    expect(screen.getByText(/at least 100 rows, 20 treated, and 20 control/)).toBeInTheDocument();
    expect(screen.getByText(/fallback values are not published/)).toBeInTheDocument();
    expect(screen.getByText(/rare.*suppressed/i)).toBeInTheDocument();
  });

  it('still exposes specialty provenance when every specialty is suppressed', async () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      isError: false,
      error: null,
      data: {
        ...mockRunResult,
        subgroups_basis: 'cohort_rows',
        effect_heterogeneity: {
          by_specialty: {},
          by_decile: {},
          by_region: {},
          by_adoption_stage: {},
          top_segments: [],
          axis_provenance: {
            specialty: {
              basis: 'cohort_rows',
              source: 'hcp_profiles.specialty',
              min_group_rows: 100,
              min_treated_rows: 20,
              min_control_rows: 20,
              fallback: 'region_then_cohort',
              support_unit: 'cohort_rows',
              estimand: 'observed_region_mix_mean_cate',
              suppressed_groups: { rare: 'group_rows_below_minimum' },
            },
          },
        },
      },
    });

    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(await screen.findByText('Specialty Effects')).toBeInTheDocument();
    expect(screen.getByText(/No specialty effects met the publication floor/i)).toBeInTheDocument();
    expect(screen.getByText(/rare.*suppressed/i)).toBeInTheDocument();
  });

  it('distinguishes missing specialty source values from insufficient support', async () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      isError: false,
      error: null,
      data: {
        ...mockRunResult,
        subgroups_basis: 'cohort_rows',
        effect_heterogeneity: {
          by_specialty: {},
          by_decile: {},
          by_region: {},
          by_adoption_stage: {},
          top_segments: [],
          axis_provenance: {
            specialty: {
              basis: 'cohort_rows',
              source: 'hcp_profiles.specialty',
              min_group_rows: 100,
              min_treated_rows: 20,
              min_control_rows: 20,
              fallback: 'region_then_cohort',
              support_unit: 'cohort_rows',
              estimand: 'observed_region_mix_mean_cate',
              suppressed_groups: { '<missing>': 'source_value_missing' },
            },
          },
        },
      },
    });

    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(await screen.findByText(/<missing>: source specialty is missing/i)).toBeInTheDocument();
    expect(screen.queryByText(/<missing>.*insufficient support/i)).not.toBeInTheDocument();
  });

  it('does not relabel a provenance-free legacy specialty aggregate as a supported effect', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      isError: false,
      error: null,
      data: {
        ...mockRunResult,
        subgroups_basis: 'twin_weighted_legacy',
        effect_heterogeneity: {
          by_specialty: { oncology: { mean: 0.05, std: 0.01, n: 12 } },
          by_decile: {},
          by_region: {},
          by_adoption_stage: {},
          top_segments: [],
          axis_provenance: {},
        },
      },
    });

    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.queryByText('Specialty Effects')).not.toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // #2206 — honest model surfacing: one shared synthetic fit, unvalidated fidelity
  // ---------------------------------------------------------------------------

  it('states that the brand models are one shared synthetic fit and unvalidated (#2206)', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(
      screen.getByText('3 brand labels over 1 shared synthetic fit · unvalidated')
    ).toBeInTheDocument();
    const card = screen.getByText('Models Available').closest('[title]');
    expect(card?.getAttribute('title')).toMatch(/Brand is routing metadata/);
    expect(card?.getAttribute('title')).toMatch(/self-generated target/);
  });

  it('tells a brand-scoped viewer the shared fit from the census count, not from row counting (codex r1 #3)', () => {
    (useTwinModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { total_count: 1, models: [sharedFitModel('Kisqali', { shared_fit_with: [] })] },
      isLoading: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(
      screen.getByText('3 brand labels over 1 shared synthetic fit · Kisqali unvalidated')
    ).toBeInTheDocument();
    const card = screen.getByText('Models Available').closest('[title]');
    expect(card?.getAttribute('title')).toMatch(/2 other brand models outside your brand grant/);
    expect(card?.getAttribute('title')).not.toMatch(/Remibrutinib|Fabhalta/);
  });

  it('shows Unvalidated on initial load when every trained model is unvalidated (codex r1 #2)', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });
    // No run displayed yet (useRunSimulation data undefined, nothing selected).
    expect(screen.getByText('Unvalidated')).toBeInTheDocument();
    expect(
      screen.getByText('No experiment outcome compared against any twin model yet')
    ).toBeInTheDocument();
    expect(screen.queryByText('—')).not.toBeInTheDocument();
  });

  it('does not call distinct fits shared (#2206)', () => {
    (useTwinModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        total_count: 2,
        models: [
          sharedFitModel('Remibrutinib', { shared_fit_model_count: 1, shared_fit_with: [] }),
          sharedFitModel('Kisqali', {
            training_fingerprint: 'ffff000011112222',
            shared_fit_model_count: 1,
            shared_fit_with: [],
            r2_score_basis: 'rwd_target',
            fidelity_status: FidelityStatus.VALIDATED,
            fidelity_score: 0.9,
            fidelity_sample_count: 3,
          }),
        ],
      },
      isLoading: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(
      screen.getByText('2 brand labels over 2 fits · 1 validated · 1 unvalidated')
    ).toBeInTheDocument();
    expect(screen.queryByText(/shared synthetic fit/)).not.toBeInTheDocument();
  });

  it('reports an unvalidated model fidelity explicitly, never as a blank pass (#2206)', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: {
        ...mockRunResult,
        model_fidelity_score: undefined,
        fidelity_status: FidelityStatus.UNVALIDATED,
        fidelity_warning: true,
        fidelity_warning_reason:
          'Model fidelity is unvalidated: no experiment outcome has been compared against this model yet (fidelity_score is NULL), so its prediction accuracy is unknown. Interpret with caution.',
      },
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    // Stat card value AND the results-panel status line both say the state, not a dash.
    expect(screen.getAllByText('Unvalidated')).toHaveLength(2);
    expect(
      screen.getByText('No experiment outcome compared against this model yet')
    ).toBeInTheDocument();
    // Results panel: explicit status line + the backend's reason in the warning box.
    expect(screen.getByTestId('fidelity-status')).toHaveTextContent('Status: Unvalidated');
    expect(screen.getByText(/Model fidelity is unvalidated/)).toBeInTheDocument();
    expect(screen.queryByText('79%')).not.toBeInTheDocument();
  });

  it('shows the gauge and a Validated status for a measured model (#2206)', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: mockRunResult,
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.getByTestId('fidelity-status')).toHaveTextContent('Status: Validated');
    expect(screen.getAllByText('79%').length).toBeGreaterThan(0);
    expect(screen.queryByText('Unvalidated')).not.toBeInTheDocument();
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

  // The confidence badge's title is selected by the result's data_provenance (#2104):
  // on the cohort path the evidence is the cohort rows, so more twins cannot raise it;
  // on the synthetic path the training frame is drawn from the twins, so it follows
  // the twin sample. Fixture confidence 0.83 -> "Confidence: 83%".
  it('explains on a cohort-path confidence badge that more twins do not raise it (#2104)', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: { ...mockRunResult, data_provenance: 'cohort_estimated_synthetic_gold_v1' },
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    expect(screen.getByText(/Confidence: 83%/)).toHaveAttribute(
      'title',
      expect.stringMatching(/cohort rows.*more twins does not raise/i)
    );
  });

  it('explains on a synthetic-path confidence badge that it follows the twin sample (#2104)', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: { ...mockRunResult, data_provenance: 'synthetic_uplift_v1' },
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    const title = screen.getByText(/Confidence: 83%/).getAttribute('title') ?? '';
    expect(title).toMatch(/twins the estimator fit on.*follows the twin sample/i);
    expect(title).not.toMatch(/more twins does not raise/i);
  });

  it('keeps the confidence badge explanation neutral when the provenance is unknown (#2104)', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      data: { ...mockRunResult, data_provenance: null },
      isSuccess: true,
      isError: false,
    });
    render(<DigitalTwin />, { wrapper: createWrapper() });
    const title = screen.getByText(/Confidence: 83%/).getAttribute('title') ?? '';
    expect(title).toMatch(/Confidence blends the evidence behind this estimate/);
    expect(title).not.toMatch(/more twins does not raise|follows the twin sample/i);
  });

  // A STORED simulation (identified by detail-only `population_filters`) shows the score
  // persisted when it ran, computed by the heuristic in force at that time — the current
  // heuristic wording above would be false for it (#2104, codex r2).
  it('says a stored legacy detail carries the score of its time, scored on twin count (#2104)', async () => {
    await openHistoryDetail({
      ...mockDetail,
      data_provenance: 'cohort_estimated_synthetic_gold_v1',
      subgroups_basis: 'twin_weighted_legacy',
    });
    const title = screen.getByText(/Confidence: 83%/).getAttribute('title') ?? '';
    expect(title).toMatch(/score stored when this simulation ran.*heuristic in force at that time/i);
    expect(title).toMatch(/scored the evidence on the generated twin count/i);
    expect(title).not.toMatch(/more twins does not raise/i);
    // Its evidence term WAS the twin count: no training-row claim may open the sentence.
    expect(title).not.toMatch(/rows the estimator fit on/i);
  });

  it('says a stored cohort_rows detail carries the score of its time, without the invariance claim (#2104)', async () => {
    await openHistoryDetail({
      ...mockDetail,
      data_provenance: 'cohort_estimated_synthetic_gold_v1',
      subgroups_basis: 'cohort_rows',
    });
    const title = screen.getByText(/Confidence: 83%/).getAttribute('title') ?? '';
    expect(title).toMatch(/score stored when this simulation ran.*heuristic in force at that time/i);
    expect(title).not.toMatch(/generated twin count/i);
    expect(title).not.toMatch(/more twins does not raise/i);
    expect(title).not.toMatch(/rows the estimator fit on/i);
  });

  it('pins the handwritten subgroups_basis union to the generated OpenAPI contract (#2104)', () => {
    // The API client imports the handwritten type, so tsc alone never compares the two;
    // this type-level assertion does (checked by tsc, a no-op at runtime).
    expectTypeOf<SimulationDetailResponse['subgroups_basis']>().toEqualTypeOf<
      components['schemas']['SimulationDetailResponse']['subgroups_basis']
    >();
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
    // The detail ATE renders twice: metrics grid value + "ATE: 0.067" evidence line.
    expect(screen.getAllByText(/0\.067/)).toHaveLength(2);
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
