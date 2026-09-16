/**
 * DagPanel tests — the review's stored DAG snapshot plus the delta of the
 * version the review is CURRENTLY on (migration 141 + codex round 3).
 *
 * The timeline is a set of facts and may end on a row the review is not on (a
 * run that recorded its version and then lost the compare-and-set advance), so
 * the panel renders the delta the caller NAMES, never the last entry's. With
 * no name it renders no delta at all: silence beats a plausible-wrong change.
 */
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { DagPanel } from './DagPanel';
import type { DagChanges, DagStructure, ReviewVersion } from '@/types/expert-review';

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

const EMPTY_CHANGES: DagChanges = {
  nodes_added: [],
  nodes_removed: [],
  edges_added: [],
  edges_removed: [],
  adjustment_sets_added: [],
  adjustment_sets_removed: [],
  is_changed: false,
};

const VERSIONS: ReviewVersion[] = [
  { version_id: 'v1', dag_version_hash: 'aaaa1111', created_at: '2026-09-01T00:00:00Z', changes: null },
  {
    version_id: 'v2',
    dag_version_hash: 'bbbb2222',
    created_at: '2026-09-05T00:00:00Z',
    changes: { ...EMPTY_CHANGES, nodes_added: ['W'], edges_added: [['W', 'T']], is_changed: true },
  },
];

describe('DagPanel', () => {
  it("renders the named current version's delta", () => {
    render(<DagPanel structure={STRUCTURE} versions={VERSIONS} currentVersionId="v2" />);
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.getByText('Changed since the previous version')).toBeInTheDocument();
    expect(screen.getByText('+ W')).toBeInTheDocument();
    expect(screen.getByText('+ W → T')).toBeInTheDocument();
  });

  it('renders the CURRENT version\'s delta, not the last entry\'s, when the timeline ends on an orphan', () => {
    // The permitted race: both runs read A; C recorded and advanced; B recorded
    // afterwards and lost the advance. The review is on C, the timeline ends
    // ... C, B. Showing B's delta beside C's graph would ask the reviewer to
    // approve a change that is not theirs.
    const orphaned: ReviewVersion[] = [
      { version_id: 'vA', dag_version_hash: 'aaaa1111', changes: null },
      {
        version_id: 'vC',
        dag_version_hash: 'cccc3333',
        changes: { ...EMPTY_CHANGES, nodes_added: ['W'], is_changed: true },
      },
      {
        version_id: 'vB',
        dag_version_hash: 'bbbb2222',
        changes: { ...EMPTY_CHANGES, nodes_added: ['Z'], nodes_removed: ['W'], is_changed: true },
      },
    ];
    render(<DagPanel structure={STRUCTURE} versions={orphaned} currentVersionId="vC" />);
    expect(screen.getByText('Changed since the previous version')).toBeInTheDocument();
    expect(screen.getByText('+ W')).toBeInTheDocument();
    expect(screen.queryByText('+ Z')).toBeNull();
    expect(screen.queryByText('− W')).toBeNull();
  });

  it('renders no diff heading when the current version changed nothing', () => {
    const unchanged: ReviewVersion[] = [VERSIONS[0], { ...VERSIONS[1], changes: EMPTY_CHANGES }];
    render(<DagPanel structure={STRUCTURE} versions={unchanged} currentVersionId="v2" />);
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
    expect(screen.queryByText('No structural change from the previous version.')).toBeNull();
  });

  it('renders NO diff when no current version is named, however long the timeline', () => {
    render(<DagPanel structure={STRUCTURE} versions={VERSIONS} currentVersionId={null} />);
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
    render(<DagPanel structure={STRUCTURE} versions={VERSIONS} />);
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
  });

  it('renders no diff when the named version is not in the timeline', () => {
    render(<DagPanel structure={STRUCTURE} versions={VERSIONS} currentVersionId="v-missing" />);
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
  });

  it('renders no diff heading for the FIRST version (nothing to diff against)', () => {
    render(<DagPanel structure={STRUCTURE} versions={VERSIONS} currentVersionId="v1" />);
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
  });

  it('renders no diff heading when no versions are supplied', () => {
    render(<DagPanel structure={STRUCTURE} />);
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
  });
});
