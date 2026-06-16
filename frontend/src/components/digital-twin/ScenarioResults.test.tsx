/**
 * ScenarioResults Component Tests
 * ================================
 *
 * The component formerly rendered LegacySimulationResponse — a UI-shaped
 * type (TRx/NRx lift CIs, ROI, projections time-series, fidelity meters,
 * sensitivity analysis) that the real `POST /api/digital-twin/simulate`
 * endpoint NEVER returns. Because of the shape mismatch the page passed
 * `results={null}` forever ("No Simulation Results" even after a real
 * run), and any future wiring would have required fabricating the legacy
 * fields.
 *
 * These tests pin the rewire: ScenarioResults renders the REAL
 * SimulationResponse (verified against the live backend OpenAPI schema)
 * — simulated ATE + CI, significance, effect size, power, confidence,
 * fidelity warning, provenance. Nothing fabricated.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ScenarioResults } from './ScenarioResults';
import type { SimulationResponse } from '@/types/digital-twin';
import { Recommendation, SimulationStatus } from '@/types/digital-twin';

function createSimulationResponse(
  overrides: Partial<SimulationResponse> = {}
): SimulationResponse {
  return {
    simulation_id: 'sim_001',
    model_id: 'twin_model_v1',
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
    recommendation_rationale: 'Significant positive effect with adequate power.',
    simulation_confidence: 0.78,
    fidelity_warning: false,
    status: SimulationStatus.COMPLETED,
    execution_time_ms: 2150,
    is_significant: true,
    effect_direction: 'positive',
    created_at: '2026-06-12T03:00:00Z',
    data_provenance: 'synthetic_uplift_v1',
    ...overrides,
  };
}

describe('ScenarioResults — real SimulationResponse rendering', () => {
  it('renders the empty state when results is null', () => {
    render(<ScenarioResults results={null} />);
    expect(screen.getByText('No Simulation Results')).toBeInTheDocument();
  });

  it('renders the loading state while a simulation runs', () => {
    render(<ScenarioResults results={null} isLoading />);
    expect(screen.getByText(/Running simulation/)).toBeInTheDocument();
  });

  it('renders an HONEST 503 error card (no trained model) with the backend detail', () => {
    const error = {
      status: 503,
      message: 'Service unavailable',
      data: { detail: 'No trained digital-twin model is available for Fabhalta/hcp.' },
      isNetworkError: false,
      isServerError: false,
    } as unknown as Parameters<typeof ScenarioResults>[0]['error'];

    render(<ScenarioResults results={null} error={error} />);

    // Must NOT look like "never ran".
    expect(screen.queryByText('No Simulation Results')).not.toBeInTheDocument();
    expect(screen.getByText(/No trained twin model is available/)).toBeInTheDocument();
    expect(
      screen.getByText(/No trained digital-twin model is available for Fabhalta\/hcp\./),
    ).toBeInTheDocument();
  });

  it('renders an HONEST timeout (408) error card', () => {
    const error = {
      status: 408,
      message: 'Request timeout',
      data: { detail: 'Twin simulation timed out; retry shortly.' },
      isNetworkError: false,
      isServerError: false,
    } as unknown as Parameters<typeof ScenarioResults>[0]['error'];

    render(<ScenarioResults results={null} error={error} />);
    expect(screen.getByText(/Simulation timed out/)).toBeInTheDocument();
  });

  it('loading takes precedence over an error (retry shows the spinner)', () => {
    const error = {
      status: 500,
      message: 'boom',
      data: null,
      isNetworkError: false,
      isServerError: true,
    } as unknown as Parameters<typeof ScenarioResults>[0]['error'];

    render(<ScenarioResults results={null} error={error} isLoading />);
    expect(screen.getByText(/Running simulation/)).toBeInTheDocument();
  });

  it('renders the real simulated ATE and confidence interval', () => {
    render(<ScenarioResults results={createSimulationResponse()} />);

    expect(screen.getByText('0.142')).toBeInTheDocument();
    expect(screen.getByText(/\[0\.081, 0\.203\]/)).toBeInTheDocument();
  });

  it('renders significance, effect size, and statistical power from the API', () => {
    render(<ScenarioResults results={createSimulationResponse()} />);

    expect(screen.getByText(/significant/i)).toBeInTheDocument();
    expect(screen.getByText('0.46')).toBeInTheDocument();
    expect(screen.getByText('83%')).toBeInTheDocument();
  });

  it('renders simulation metadata (twins, brand, intervention, runtime)', () => {
    render(<ScenarioResults results={createSimulationResponse()} />);

    expect(screen.getByText('500')).toBeInTheDocument();
    expect(screen.getByText(/Kisqali/)).toBeInTheDocument();
    expect(screen.getByText(/hcp_engagement/)).toBeInTheDocument();
    expect(screen.getByText(/2,150\s*ms/)).toBeInTheDocument();
  });

  it('surfaces the fidelity warning when the backend reports one', () => {
    render(
      <ScenarioResults
        results={createSimulationResponse({
          fidelity_warning: true,
          fidelity_warning_reason: 'Twin coverage below threshold for this brand.',
        })}
      />
    );

    expect(
      screen.getByText(/Twin coverage below threshold for this brand/)
    ).toBeInTheDocument();
  });

  it('renders data provenance so users know where the estimate came from', () => {
    render(<ScenarioResults results={createSimulationResponse()} />);
    expect(screen.getByText(/synthetic_uplift_v1/)).toBeInTheDocument();
  });

  it('does NOT render the legacy fabrication-shaped sections (no substrate)', () => {
    render(<ScenarioResults results={createSimulationResponse()} />);

    // The real endpoint returns none of these — they must not render.
    expect(screen.queryByText(/TRx Lift/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/ROI Projection/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Projected Outcomes/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Sensitivity Analysis/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Data Coverage/i)).not.toBeInTheDocument();
  });

  it('marks non-significant results honestly', () => {
    render(
      <ScenarioResults
        results={createSimulationResponse({
          is_significant: false,
          effect_direction: 'neutral',
        })}
      />
    );
    expect(screen.getByText(/not significant/i)).toBeInTheDocument();
  });
});
