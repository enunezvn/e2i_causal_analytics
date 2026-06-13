/**
 * CausalDiscovery Visualization Tests
 * ===================================
 *
 * Red-first guards for the hardcoded-analysis finding: the component
 * formerly fell back to SAMPLE_NODES / SAMPLE_EDGES / SAMPLE_EFFECTS /
 * SAMPLE_REFUTATION_RESULTS (ATE 0.45, CI [0.21, 0.69], p=0.002,
 * all-passing refutation suite) whenever a call site omitted the data
 * props — a permanently-fake causal analysis with no label.
 *
 * Desired behavior: data comes ONLY from props. With no props the
 * component renders honest empty states, so no future call site can
 * resurrect the fabricated analysis.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CausalDiscovery } from './CausalDiscovery';
import type { CausalNode, CausalEdge } from './causal/CausalDAG';

// Mock the D3-based DAG to a marker that surfaces the node/edge props.
vi.mock('./causal/CausalDAG', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./causal/CausalDAG')>();
  return {
    ...actual,
    CausalDAG: ({ nodes, edges }: { nodes: CausalNode[]; edges: CausalEdge[] }) => (
      <div data-testid="causal-dag" data-nodes={nodes.length} data-edges={edges.length} />
    ),
  };
});

describe('CausalDiscovery viz — no SAMPLE_ fallbacks', () => {
  it('renders NO fabricated analysis when data props are omitted', () => {
    render(<CausalDiscovery />);

    // Fabricated effect estimates must not render.
    expect(screen.queryByText('0.450')).not.toBeInTheDocument();
    expect(screen.queryByText('Patient Age')).not.toBeInTheDocument();
    expect(screen.queryByText('Disease Severity')).not.toBeInTheDocument();
    // Fabricated refutation suite must not render.
    expect(screen.queryByText(/random_common_cause/)).not.toBeInTheDocument();
    expect(screen.queryByText(/placebo_treatment/)).not.toBeInTheDocument();

    // The DAG area renders an honest empty state instead of a fake graph.
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });

  it('renders real effects passed via props', () => {
    render(
      <CausalDiscovery
        nodes={[
          { id: 't', label: 'Rep Visits', type: 'treatment' },
          { id: 'o', label: 'TRx Count', type: 'outcome' },
        ]}
        edges={[{ id: 'e', source: 't', target: 'o', type: 'causal', effect: 0.12 }]}
        effects={[
          {
            id: 'fx1',
            treatment: 'rep_visits',
            outcome: 'trx_count',
            estimate: 0.123,
            ciLower: 0.05,
            ciUpper: 0.2,
            isSignificant: true,
          },
        ]}
        refutationResults={[]}
      />
    );

    expect(screen.getByTestId('causal-dag')).toHaveAttribute('data-nodes', '2');
    expect(screen.getByText('rep_visits')).toBeInTheDocument();
    expect(screen.getByText('0.123')).toBeInTheDocument();
    expect(screen.queryByText('0.450')).not.toBeInTheDocument();
  });

  it('keeps honest empty states for effects and refutation tables when given empty arrays', () => {
    render(
      <CausalDiscovery nodes={[]} edges={[]} effects={[]} refutationResults={[]} />
    );

    // EffectsTable / RefutationTests own empty states must show, never samples.
    expect(screen.queryByText('Treatment Adherence')).not.toBeInTheDocument();
    expect(screen.queryByText(/bootstrap/)).not.toBeInTheDocument();
  });
});
