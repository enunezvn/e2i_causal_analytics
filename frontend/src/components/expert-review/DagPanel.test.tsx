/**
 * DagPanel tests — the review's stored DAG snapshot plus, from migration 141,
 * the delta against the PREVIOUS version of the same estimand.
 *
 * The diff only earns its heading when there IS a previous version: a review
 * with one version (or minted before the versions table) must show the graph
 * alone, never an empty "changed" section.
 */
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { DagPanel } from './DagPanel';
import type { DagStructure, ReviewVersion } from '@/types/expert-review';

// The DAG renderer is D3-heavy; this test only asserts it is MOUNTED (its own
// rendering is covered by causal.test.tsx).
vi.mock('@/components/visualizations/causal/CausalDAG', () => {
  const FakeDag = ({ nodes, edges }: { nodes: unknown[]; edges: unknown[] }) => (
    <div data-testid="causal-dag" data-nodes={nodes.length} data-edges={edges.length} />
  );
  return { CausalDAG: FakeDag, default: FakeDag };
});

const STRUCTURE: DagStructure = {
  nodes: ['T', 'Y', 'W'],
  edges: [
    ['T', 'Y'],
    ['W', 'T'],
  ],
  treatment_nodes: ['T'],
  outcome_nodes: ['Y'],
};

const VERSIONS: ReviewVersion[] = [
  { version_id: 'v1', dag_version_hash: 'aaaa1111', created_at: '2026-09-01T00:00:00Z', changes: null },
  {
    version_id: 'v2',
    dag_version_hash: 'bbbb2222',
    created_at: '2026-09-05T00:00:00Z',
    changes: {
      nodes_added: ['W'],
      nodes_removed: [],
      edges_added: [['W', 'T']],
      edges_removed: [],
      adjustment_sets_added: [],
      adjustment_sets_removed: [],
      is_changed: true,
    },
  },
];

describe('DagPanel', () => {
  it('renders the diff against the previous version when there is more than one', () => {
    render(<DagPanel structure={STRUCTURE} versions={VERSIONS} />);
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.getByText('Changed since the previous version')).toBeInTheDocument();
    expect(screen.getByText('+ W')).toBeInTheDocument();
    expect(screen.getByText('+ W → T')).toBeInTheDocument();
  });

  it('renders no diff heading when the newest version changed nothing', () => {
    const unchanged: ReviewVersion[] = [
      VERSIONS[0],
      {
        ...VERSIONS[1],
        changes: {
          nodes_added: [],
          nodes_removed: [],
          edges_added: [],
          edges_removed: [],
          adjustment_sets_added: [],
          adjustment_sets_removed: [],
          is_changed: false,
        },
      },
    ];
    render(<DagPanel structure={STRUCTURE} versions={unchanged} />);
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
    expect(screen.queryByText('No structural change from the previous version.')).toBeNull();
  });

  it('renders no diff heading for a single version', () => {
    render(<DagPanel structure={STRUCTURE} versions={[VERSIONS[0]]} />);
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
  });

  it('renders no diff heading when no versions are supplied', () => {
    render(<DagPanel structure={STRUCTURE} />);
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
  });
});
