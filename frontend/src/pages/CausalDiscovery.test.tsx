/**
 * CausalDiscovery Page Tests
 * ==========================
 *
 * Tests for the CausalDiscovery page component.
 * Includes tests for:
 * - Page header with technology badges
 * - CausalDiscovery visualization integration
 * - Refutation tests integration (Phase 3.2)
 * - Live API wiring: useRouteQuery + useCausalChains (Issue #303)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as React from 'react';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithAllProviders } from '@/test/utils';
import CausalDiscovery from './CausalDiscovery';

// =============================================================================
// MOCK SETUP
// =============================================================================

// Radix <Select> relies on pointer-capture / portal behaviour that jsdom does
// not implement, so we replace the UI primitive with a native <select> that
// preserves the same value / onValueChange contract AND the trigger's id +
// aria-label (so getByLabelText(/treatment variable/i) keeps working). The
// <Select> mock walks its own children to collect the <SelectItem> values and
// the trigger id/aria-label, then renders ONE native <select>. Tests drive it
// with fireEvent.change like a normal control.
vi.mock('@/components/ui/select', () => {
  type ItemProps = { value: string; children?: React.ReactNode };
  type TriggerProps = {
    id?: string;
    'aria-label'?: string;
    children?: React.ReactNode;
  };

  const Select = ({
    value,
    onValueChange,
    disabled,
    children,
  }: {
    value?: string;
    onValueChange?: (v: string) => void;
    disabled?: boolean;
    children?: React.ReactNode;
  }) => {
    const options: Array<{ value: string; label: React.ReactNode }> = [];
    let triggerId: string | undefined;
    let triggerAriaLabel: string | undefined;

    const walk = (nodes: React.ReactNode) => {
      React.Children.forEach(nodes, (child: unknown) => {
        if (!React.isValidElement(child)) return;
        const el = child as React.ReactElement<
          Partial<ItemProps & TriggerProps> & { children?: React.ReactNode }
        >;
        const name = (el.type as { __mockName?: string })?.__mockName;
        if (name === 'SelectItem' && typeof el.props.value === 'string') {
          options.push({ value: el.props.value, label: el.props.children });
        }
        if (name === 'SelectTrigger') {
          triggerId = el.props.id;
          triggerAriaLabel = el.props['aria-label'];
        }
        if (el.props.children) walk(el.props.children);
      });
    };
    walk(children);

    return (
      <select
        id={triggerId}
        aria-label={triggerAriaLabel}
        value={value ?? ''}
        disabled={disabled}
        onChange={(e) => onValueChange?.(e.target.value)}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    );
  };

  // Tagged passthroughs so the walker can identify them by name.
  const SelectTrigger = (_props: TriggerProps) => null;
  (SelectTrigger as { __mockName?: string }).__mockName = 'SelectTrigger';
  const SelectItem = (_props: ItemProps) => null;
  (SelectItem as { __mockName?: string }).__mockName = 'SelectItem';
  const SelectValue = () => null;
  const SelectContent = ({ children }: { children?: React.ReactNode }) => (
    <>{children}</>
  );

  return { Select, SelectTrigger, SelectContent, SelectItem, SelectValue };
});

// Simplify the covariate multi-select to native checkboxes for deterministic
// jsdom interaction (the real one uses a Radix popover + portal).
vi.mock('@/components/causal/CovariateMultiSelect', () => {
  return {
    CovariateMultiSelect: ({
      options,
      selected,
      onChange,
      disabled,
    }: {
      options: string[];
      selected: string[];
      onChange: (next: string[]) => void;
      disabled?: boolean;
    }) => (
      <div aria-label="Covariates" data-testid="covariate-multiselect">
        {options.map((opt) => (
          <label key={opt}>
            <input
              type="checkbox"
              value={opt}
              checked={selected.includes(opt)}
              disabled={disabled}
              onChange={(e) =>
                onChange(
                  e.target.checked
                    ? [...selected, opt]
                    : selected.filter((v) => v !== opt)
                )
              }
            />
            {opt}
          </label>
        ))}
      </div>
    ),
  };
});

// Mock the CausalDiscovery visualization component to avoid D3 complexities in tests
vi.mock('@/components/visualizations/CausalDiscovery', () => ({
  CausalDiscovery: ({
    showControls,
    showDetails,
    showEffectsTable,
    showRefutationTests,
    nodes,
    edges,
    effects,
    refutationResults,
  }: {
    showControls?: boolean;
    showDetails?: boolean;
    showEffectsTable?: boolean;
    showRefutationTests?: boolean;
    nodes?: Array<{ id: string; label: string }>;
    edges?: Array<{ id: string }>;
    effects?: Array<{
      id: string;
      estimate: number;
      treatment: string;
      ciLower?: number;
      ciUpper?: number;
      confidenceLevel?: number;
    }>;
    refutationResults?: Array<{ id: string }>;
  }) => (
    <div data-testid="causal-discovery-viz">
      <div data-testid="show-controls">{String(showControls)}</div>
      <div data-testid="show-details">{String(showDetails)}</div>
      <div data-testid="show-effects-table">{String(showEffectsTable)}</div>
      <div data-testid="show-refutation-tests">{String(showRefutationTests)}</div>
      <div data-testid="viz-nodes-count">{String(nodes?.length ?? '__undefined__')}</div>
      <div data-testid="viz-edges-count">{String(edges?.length ?? '__undefined__')}</div>
      <div data-testid="viz-effects-count">{String(effects?.length ?? '__undefined__')}</div>
      <div data-testid="viz-refutations-count">
        {String(refutationResults?.length ?? '__undefined__')}
      </div>
      <div data-testid="viz-effect-estimates">
        {(effects ?? [])
          .map(
            (e) =>
              `${e.treatment}:${e.estimate}:ci=${e.ciLower ?? 'none'},${e.ciUpper ?? 'none'}:lvl=${e.confidenceLevel ?? 'none'}`
          )
          .join('|')}
      </div>
      <div data-testid="viz-node-labels">{(nodes ?? []).map((n) => n.label).join('|')}</div>
    </div>
  ),
}));

// Mock the live API hooks so we can assert calls and provide canned responses
const mockRouteMutate = vi.fn();
const mockChainsMutate = vi.fn();
const mockPipelineMutate = vi.fn();

// Mutable state objects (per-test) for the mutation hook returns
type FakeMutationState<TData = unknown> = {
  data: TData | undefined;
  isPending: boolean;
  error: Error | null;
  isSuccess: boolean;
};

const routeState: FakeMutationState = {
  data: undefined,
  isPending: false,
  error: null,
  isSuccess: false,
};

const chainsState: FakeMutationState = {
  data: undefined,
  isPending: false,
  error: null,
  isSuccess: false,
};

const pipelineState: FakeMutationState = {
  data: undefined,
  isPending: false,
  error: null,
  isSuccess: false,
};

// Candidate variables returned by useCausalVariables. The page's selectors and
// the covariate multi-select render from these.
const variablesState: {
  data:
    | {
        dataset: string;
        treatment_candidates: string[];
        outcome_candidates: string[];
        covariate_candidates: string[];
        columns: string[];
      }
    | undefined;
  isLoading: boolean;
} = {
  data: {
    dataset: 'patient_journeys',
    treatment_candidates: ['treatment_arm', 'treatment_initiated'],
    outcome_candidates: [
      'persistent_180d',
      'discontinued_180d',
      'treatment_initiated',
    ],
    covariate_candidates: [
      'disease_severity',
      'engagement_score',
      'age_at_diagnosis',
    ],
    columns: [],
  },
  isLoading: false,
};

vi.mock('@/hooks/api/use-causal', () => ({
  useCausalVariables: () => ({
    data: variablesState.data,
    isLoading: variablesState.isLoading,
  }),
  useRouteQuery: () => ({
    mutate: mockRouteMutate,
    data: routeState.data,
    isPending: routeState.isPending,
    error: routeState.error,
    isSuccess: routeState.isSuccess,
  }),
  useRunParallelPipeline: () => ({
    mutate: mockPipelineMutate,
    data: pipelineState.data,
    isPending: pipelineState.isPending,
    error: pipelineState.error,
    isSuccess: pipelineState.isSuccess,
  }),
}));

// The page fetches real estimation rows BEFORE running the pipeline. Mock it so
// the pipeline test can assert the rows are threaded into the request filters.
// NB: the impl takes an explicit arg so the mock's call signature carries it —
// `tsc -b` (the production build) otherwise infers a zero-arg tuple and rejects
// `mock.calls[0][0]` with TS2493.
const mockGetCausalEstimationData = vi.fn(async (_args?: unknown) => ({
  dataset: 'patient_journeys',
  columns: ['treatment_arm', 'persistent_180d'],
  n_rows: 2,
  estimation_data_records: [
    { treatment_arm: 1, persistent_180d: 1 },
    { treatment_arm: 0, persistent_180d: 0 },
  ],
}));

vi.mock('@/api/causal', () => ({
  getCausalEstimationData: (args: unknown) => mockGetCausalEstimationData(args),
}));

vi.mock('@/hooks/api/use-graph', () => ({
  useCausalChains: () => ({
    mutate: mockChainsMutate,
    data: chainsState.data,
    isPending: chainsState.isPending,
    error: chainsState.error,
    isSuccess: chainsState.isSuccess,
  }),
}));

function resetMutationStates() {
  routeState.data = undefined;
  routeState.isPending = false;
  routeState.error = null;
  routeState.isSuccess = false;
  chainsState.data = undefined;
  chainsState.isPending = false;
  chainsState.error = null;
  chainsState.isSuccess = false;
  pipelineState.data = undefined;
  pipelineState.isPending = false;
  pipelineState.error = null;
  pipelineState.isSuccess = false;
}

describe('CausalDiscovery Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetMutationStates();
    variablesState.isLoading = false;
    variablesState.data = {
      dataset: 'patient_journeys',
      treatment_candidates: ['treatment_arm', 'treatment_initiated'],
      outcome_candidates: [
        'persistent_180d',
        'discontinued_180d',
        'treatment_initiated',
      ],
      covariate_candidates: [
        'disease_severity',
        'engagement_score',
        'age_at_diagnosis',
      ],
      columns: [],
    };
  });

  // =========================================================================
  // PAGE HEADER TESTS
  // =========================================================================

  describe('Page Header', () => {
    it('renders page title', () => {
      renderWithAllProviders(<CausalDiscovery />);

      expect(screen.getByText('Causal Discovery')).toBeInTheDocument();
    });

    it('renders page description', () => {
      renderWithAllProviders(<CausalDiscovery />);

      expect(screen.getByText(/Causal analysis/i)).toBeInTheDocument();
    });
  });

  // =========================================================================
  // TECHNOLOGY BADGES TESTS (Phase 3.2)
  // =========================================================================

  describe('Technology Badges', () => {
    it('displays DoWhy badge', () => {
      renderWithAllProviders(<CausalDiscovery />);

      expect(screen.getByText('DoWhy')).toBeInTheDocument();
    });

    it('displays EconML badge', () => {
      renderWithAllProviders(<CausalDiscovery />);

      expect(screen.getByText('EconML')).toBeInTheDocument();
    });

    it('displays DAG badge', () => {
      renderWithAllProviders(<CausalDiscovery />);

      expect(screen.getByText('DAG')).toBeInTheDocument();
    });

    it('displays Refutation badge', () => {
      renderWithAllProviders(<CausalDiscovery />);

      expect(screen.getByText('Refutation')).toBeInTheDocument();
    });

    it('renders all four technology badges', () => {
      renderWithAllProviders(<CausalDiscovery />);

      // Verify all 4 specific badges are present
      expect(screen.getByText('DoWhy')).toBeInTheDocument();
      expect(screen.getByText('EconML')).toBeInTheDocument();
      expect(screen.getByText('DAG')).toBeInTheDocument();
      expect(screen.getByText('Refutation')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // VISUALIZATION COMPONENT INTEGRATION TESTS
  // =========================================================================

  describe('CausalDiscovery Visualization', () => {
    it('renders the visualization component', () => {
      renderWithAllProviders(<CausalDiscovery />);

      expect(screen.getByTestId('causal-discovery-viz')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // REAL-DATA THREADING TESTS (fix: hardcoded bottom-of-page analysis)
  // =========================================================================
  // The viz formerly received NO data props and fell back to fabricated
  // SAMPLE_ analysis (ATE 0.45, all-passing refutations). The page must
  // thread the real run's outputs down — empty until a run completes.

  describe('Visualization receives real run data (no SAMPLE_ fallback)', () => {
    it('passes EMPTY data arrays (not undefined) before any run, so the viz cannot fall back', () => {
      renderWithAllProviders(<CausalDiscovery />);

      expect(screen.getByTestId('viz-nodes-count')).toHaveTextContent(/^0$/);
      expect(screen.getByTestId('viz-edges-count')).toHaveTextContent(/^0$/);
      expect(screen.getByTestId('viz-effects-count')).toHaveTextContent(/^0$/);
      expect(screen.getByTestId('viz-refutations-count')).toHaveTextContent(/^0$/);
    });

    it('threads parallel-pipeline results into the viz effects table', () => {
      pipelineState.data = {
        pipeline_id: 'pp_1',
        status: 'completed',
        libraries_succeeded: ['dowhy', 'econml'],
        libraries_failed: [],
        library_results: {
          dowhy: { effect_estimate: 0.123, ci_lower: 0.05, ci_upper: 0.2 },
          // econml reports an estimate WITHOUT CI bounds — nothing may be invented.
          econml: { effect_estimate: 0.117 },
        },
        consensus_effect: 0.12,
        consensus_ci_lower: 0.045,
        consensus_ci_upper: 0.195,
        consensus_method: 'variance_weighted',
        total_latency_ms: 900,
        created_at: '2026-06-12T00:00:00Z',
        warnings: [],
      };

      renderWithAllProviders(<CausalDiscovery />);

      const estimates = screen.getByTestId('viz-effect-estimates').textContent ?? '';
      expect(estimates).toContain('0.123:ci=0.05,0.2');
      // Missing CI bounds stay missing — never synthesized from the estimate
      // (the old code did `ci_lower ?? effect_estimate`, faking a zero-width CI).
      expect(estimates).toContain('0.117:ci=none,none');
      expect(estimates).not.toContain('0.117:ci=0.117');
      // No invented confidence level: the pipeline request/response has no
      // confidence_level field, so labeling 0.95 was fabrication.
      expect(estimates).not.toContain('lvl=0.95');
      // The fabricated SAMPLE effect must never appear.
      expect(estimates).not.toContain('0.45');
    });

    it('threads discovered KG chains into the viz DAG nodes', () => {
      chainsState.data = {
        chains: [
          {
            nodes: [
              { id: 'n1', name: 'Rep Visits', type: 'Action' },
              { id: 'n2', name: 'TRx Count', type: 'KPI' },
            ],
            relationships: [{ source_id: 'n1', target_id: 'n2', confidence: 0.8 }],
            path_length: 1,
            total_confidence: 0.8,
          },
        ],
        total_chains: 1,
        query_latency_ms: 40,
      };

      renderWithAllProviders(<CausalDiscovery />);

      expect(screen.getByTestId('viz-nodes-count')).toHaveTextContent(/^2$/);
      expect(screen.getByTestId('viz-edges-count')).toHaveTextContent(/^1$/);
      expect(screen.getByTestId('viz-node-labels')).toHaveTextContent('Rep Visits|TRx Count');
    });
  });

  // =========================================================================
  // LIVE API WIRING TESTS (Issue #303)
  // =========================================================================

  describe('Routing query form (Issue #303)', () => {
    it('renders form inputs for query, treatment_var, outcome_var, and covariates', () => {
      renderWithAllProviders(<CausalDiscovery />);

      // Form inputs should be present
      expect(screen.getByLabelText(/causal question/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/treatment variable/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/outcome variable/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/covariates/i)).toBeInTheDocument();
    });

    it('forwards the user-typed causal question verbatim to useRouteQuery', async () => {
      renderWithAllProviders(<CausalDiscovery />);

      const query = screen.getByLabelText(/causal question/i) as HTMLInputElement;
      // A targeting question — backend router maps this to CausalML.
      fireEvent.change(query, {
        target: { value: 'Who should we target with rep visits?' },
      });

      const submit = screen.getByRole('button', { name: /run routing/i });
      fireEvent.click(submit);

      await waitFor(() => {
        expect(mockRouteMutate).toHaveBeenCalledTimes(1);
      });
      expect(mockRouteMutate.mock.calls[0][0].query).toBe(
        'Who should we target with rep visits?',
      );
    });

    it('submits the form and calls useRouteQuery with treatment, outcome, and covariates', async () => {
      renderWithAllProviders(<CausalDiscovery />);

      const treatment = screen.getByLabelText(/treatment variable/i) as HTMLSelectElement;
      const outcome = screen.getByLabelText(/outcome variable/i) as HTMLSelectElement;

      // Treatment / outcome are now Selects bound to the live candidate lists;
      // pick real columns from the patient_journeys gold-standard frame.
      fireEvent.change(treatment, { target: { value: 'treatment_arm' } });
      fireEvent.change(outcome, { target: { value: 'persistent_180d' } });
      // Covariates default to the three gold-standard controls; deselect
      // engagement_score so we exercise the multi-select wiring too.
      const engagementBox = screen.getByLabelText(
        'engagement_score',
      ) as HTMLInputElement;
      fireEvent.click(engagementBox);

      const submit = screen.getByRole('button', { name: /run routing/i });
      fireEvent.click(submit);

      await waitFor(() => {
        expect(mockRouteMutate).toHaveBeenCalledTimes(1);
      });
      const calledWith = mockRouteMutate.mock.calls[0][0];
      // RouteQueryRequest's schema is { query, treatment_var?, outcome_var?,
      // context?, prefer_library? } — there is no top-level `covariates`
      // field, so a top-level placement would be silently ignored by the
      // backend. Assert the canonical `context.covariates` shape exactly.
      expect(calledWith).toMatchObject({
        treatment_var: 'treatment_arm',
        outcome_var: 'persistent_180d',
        context: { covariates: ['disease_severity', 'age_at_diagnosis'] },
      });
      expect(
        (calledWith as Record<string, unknown>).covariates,
      ).toBeUndefined();
    });

    it('shows recommended library and alternatives from the routing response', () => {
      // Pre-populate routing data
      routeState.data = {
        query: '',
        question_type: 'causal_effect',
        primary_library: 'dowhy',
        secondary_libraries: ['econml', 'causalml'],
        recommended_estimators: ['propensity_score_matching'],
        routing_confidence: 0.87,
        routing_rationale: 'Direct ATE question',
        suggested_pipeline: 'parallel',
      };
      routeState.isSuccess = true;

      renderWithAllProviders(<CausalDiscovery />);

      // Primary library surfaced
      expect(screen.getByTestId('routing-primary-library')).toHaveTextContent(/dowhy/i);
      // Alternatives surfaced
      const alternatives = screen.getByTestId('routing-alternatives');
      expect(alternatives).toHaveTextContent(/econml/i);
      expect(alternatives).toHaveTextContent(/causalml/i);
    });

    it('shows loading state while routing is pending', () => {
      routeState.isPending = true;
      renderWithAllProviders(<CausalDiscovery />);

      expect(screen.getByTestId('routing-loading')).toBeInTheDocument();
    });

    it('shows error state when routing fails', () => {
      routeState.error = new Error('boom from server');
      renderWithAllProviders(<CausalDiscovery />);

      // QueryErrorState renders a title plus the error message; both should
      // be present somewhere in the page.
      expect(screen.getAllByText(/routing failed/i).length).toBeGreaterThan(0);
      // The original error message is surfaced too.
      expect(screen.getByText(/boom from server/i)).toBeInTheDocument();
    });
  });

  describe('Results table (Issue #303)', () => {
    it('renders effect estimate, CI, and library used when routing returns recommendations', () => {
      routeState.data = {
        query: '',
        question_type: 'causal_effect',
        primary_library: 'econml',
        secondary_libraries: ['dowhy'],
        recommended_estimators: ['causal_forest'],
        routing_confidence: 0.91,
        routing_rationale: 'HTE question',
        suggested_pipeline: 'parallel',
      };
      routeState.isSuccess = true;

      renderWithAllProviders(<CausalDiscovery />);

      const table = screen.getByTestId('routing-results-table');
      // Header / columns
      expect(table).toHaveTextContent(/library/i);
      expect(table).toHaveTextContent(/recommended estimator|estimator/i);
      expect(table).toHaveTextContent(/confidence/i);
      // Row content includes primary library
      expect(table).toHaveTextContent(/econml/i);
      // Row content includes the estimator
      expect(table).toHaveTextContent(/causal_forest/i);
      // Confidence rendered as percent (0.91 → 91%)
      expect(table).toHaveTextContent(/91/);
    });

    it('does not mis-label DoWhy estimators on the EconML secondary row', () => {
      // /api/causal/route returns `recommended_estimators` only for the
      // primary library. If we naively zipped them by row index we'd render
      // DoWhy estimators on the EconML row.
      routeState.data = {
        query: '',
        question_type: 'causal_effect',
        primary_library: 'dowhy',
        secondary_libraries: ['econml'],
        recommended_estimators: [
          'propensity_score_matching',
          'inverse_propensity_weighting',
        ],
        routing_confidence: 0.8,
        routing_rationale: '',
        suggested_pipeline: 'parallel',
      };
      routeState.isSuccess = true;

      renderWithAllProviders(<CausalDiscovery />);

      const table = screen.getByTestId('routing-results-table');
      const rows = table.querySelectorAll('tbody > tr');
      expect(rows.length).toBe(2);
      // Primary row (DoWhy) renders its recommended estimators
      expect(rows[0].textContent).toMatch(/dowhy/i);
      expect(rows[0].textContent).toMatch(/propensity_score_matching/);
      // Secondary row (EconML) must NOT show DoWhy estimators
      expect(rows[1].textContent).toMatch(/econml/i);
      expect(rows[1].textContent).not.toMatch(/propensity_score_matching/);
      expect(rows[1].textContent).not.toMatch(/inverse_propensity_weighting/);
    });

    it('renders pipeline effect estimate + CI + library agreement when pipeline returns data', () => {
      routeState.data = {
        query: '',
        question_type: 'causal_effect',
        primary_library: 'dowhy',
        secondary_libraries: ['econml'],
        recommended_estimators: ['propensity_score_matching', 'causal_forest'],
        routing_confidence: 0.78,
        routing_rationale: 'ATE question',
        suggested_pipeline: 'parallel',
      };
      routeState.isSuccess = true;
      pipelineState.data = {
        pipeline_id: 'pl_abc',
        status: 'completed',
        libraries_succeeded: ['dowhy', 'econml'],
        libraries_failed: [],
        library_results: {
          dowhy: {
            effect_estimate: 0.234,
            ci_lower: 0.123,
            ci_upper: 0.345,
          },
          econml: {
            effect_estimate: 0.211,
            ci_lower: 0.101,
            ci_upper: 0.321,
          },
        },
        consensus_effect: 0.225,
        consensus_ci_lower: 0.112,
        consensus_ci_upper: 0.333,
        library_agreement_score: 0.93,
        consensus_method: 'variance_weighted',
        total_latency_ms: 1234,
        created_at: '2026-05-17T00:00:00Z',
        warnings: [],
      };
      pipelineState.isSuccess = true;

      renderWithAllProviders(<CausalDiscovery />);

      const table = screen.getByTestId('routing-results-table');
      // Effect estimate cell shows the numeric value (0.234)
      expect(table).toHaveTextContent(/0\.234/);
      // CI shown as [lower, upper]
      expect(table).toHaveTextContent(/\[0\.123, 0\.345\]/);

      // Consensus block surfaces consensus + library agreement score
      const consensus = screen.getByTestId('pipeline-consensus');
      expect(consensus).toHaveTextContent(/0\.225/);
      expect(consensus).toHaveTextContent(/93/);
    });
  });

  describe('Run parallel pipeline (Issue #303)', () => {
    it('renders a button to run the parallel pipeline', () => {
      renderWithAllProviders(<CausalDiscovery />);

      expect(
        screen.getByRole('button', { name: /run parallel pipeline|run pipeline/i }),
      ).toBeInTheDocument();
    });

    it('fetches real estimation data and threads it + treatment/outcome/covariates into the pipeline request', async () => {
      routeState.data = {
        query: '',
        question_type: 'causal_effect',
        primary_library: 'dowhy',
        secondary_libraries: ['econml'],
        recommended_estimators: [],
        routing_confidence: 0.8,
        routing_rationale: '',
        suggested_pipeline: 'parallel',
      };
      routeState.isSuccess = true;

      renderWithAllProviders(<CausalDiscovery />);

      const treatment = screen.getByLabelText(/treatment variable/i) as HTMLSelectElement;
      const outcome = screen.getByLabelText(/outcome variable/i) as HTMLSelectElement;

      // Real columns from the gold-standard frame.
      fireEvent.change(treatment, { target: { value: 'treatment_arm' } });
      fireEvent.change(outcome, { target: { value: 'persistent_180d' } });

      const button = screen.getByRole('button', {
        name: /run parallel pipeline|run pipeline/i,
      });
      fireEvent.click(button);

      // The page must fetch the real estimation rows BEFORE running.
      await waitFor(() => {
        expect(mockGetCausalEstimationData).toHaveBeenCalledTimes(1);
      });
      expect(mockGetCausalEstimationData.mock.calls[0][0]).toMatchObject({
        treatment_var: 'treatment_arm',
        outcome_var: 'persistent_180d',
        covariates: ['disease_severity', 'engagement_score', 'age_at_diagnosis'],
      });

      await waitFor(() => {
        expect(mockPipelineMutate).toHaveBeenCalledTimes(1);
      });
      const arg = mockPipelineMutate.mock.calls[0][0];
      // Hook expects an object: { request, asyncMode }
      const request = (arg && (arg.request ?? arg)) as Record<string, unknown>;
      expect(request).toMatchObject({
        treatment_var: 'treatment_arm',
        outcome_var: 'persistent_180d',
      });
      expect(request.covariates).toEqual([
        'disease_severity',
        'engagement_score',
        'age_at_diagnosis',
      ]);
      // Real estimation rows are attached via filters so the libraries can fit.
      expect(
        (request.filters as { estimation_data_records?: unknown[] })
          .estimation_data_records,
      ).toHaveLength(2);
      // Libraries should pull from routing (dowhy, econml) when present.
      expect(request.libraries).toEqual(
        expect.arrayContaining(['dowhy', 'econml']),
      );
    });
  });

  describe('KG chain discovery mode (Issue #303)', () => {
    it('renders a toggle/button to switch to KG chain discovery mode', () => {
      renderWithAllProviders(<CausalDiscovery />);

      // Some control labeled like "Discover chains in KG"
      expect(
        screen.getByRole('button', { name: /discover chains|kg chains/i }),
      ).toBeInTheDocument();
    });

    it('invokes useCausalChains when the KG-chains action is triggered', async () => {
      renderWithAllProviders(<CausalDiscovery />);

      const outcomeInput = screen.getByLabelText(/outcome variable/i) as HTMLSelectElement;
      // Pick a real outcome candidate from the gold-standard frame.
      fireEvent.change(outcomeInput, { target: { value: 'discontinued_180d' } });

      const kgButton = screen.getByRole('button', { name: /discover chains|kg chains/i });
      fireEvent.click(kgButton);

      await waitFor(() => {
        expect(mockChainsMutate).toHaveBeenCalledTimes(1);
      });
      const calledWith = mockChainsMutate.mock.calls[0][0];
      // Should pass the outcome as kpi_name (or include it some structured way)
      expect(calledWith.kpi_name).toBe('discontinued_180d');
    });

    it('renders discovered chains when useCausalChains returns data', () => {
      chainsState.data = {
        chains: [
          {
            nodes: [
              { id: 'n1', name: 'Rep Visits', type: 'Treatment' },
              { id: 'n2', name: 'TRx', type: 'KPI' },
            ],
            relationships: [
              {
                id: 'r1',
                source_id: 'n1',
                target_id: 'n2',
                type: 'IMPACTS',
                confidence: 0.85,
              },
            ],
            total_confidence: 0.85,
            path_length: 1,
          },
        ],
        total_chains: 1,
        latency_ms: 12,
        timestamp: '2026-05-17T00:00:00Z',
      };
      chainsState.isSuccess = true;

      renderWithAllProviders(<CausalDiscovery />);

      // Chains panel renders
      expect(screen.getByTestId('kg-chains-panel')).toBeInTheDocument();
      // Chain content is surfaced
      const panel = screen.getByTestId('kg-chains-panel');
      expect(panel).toHaveTextContent(/Rep Visits/);
      expect(panel).toHaveTextContent(/TRx/);
    });
  });
});
