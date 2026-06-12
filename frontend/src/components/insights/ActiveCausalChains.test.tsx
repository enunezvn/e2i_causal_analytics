/**
 * ActiveCausalChains Tests
 * ========================
 *
 * Red-first guards for the fabricated-graph finding: the widget formerly
 * initialized Cytoscape with SAMPLE_ELEMENTS (a fake "Detailing Frequency
 * -> HCP Awareness -> ... -> TRx Volume" causal graph with invented edge
 * weights). When the API returned no chains, the fake graph silently stayed
 * on screen as if it were live knowledge-graph data.
 *
 * Desired behavior: initialize empty; render real chains; show an honest
 * empty state when no chains exist; labeled error on failure.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ActiveCausalChains } from './ActiveCausalChains';
import * as useCytoscapeModule from '@/hooks/use-cytoscape';
import * as useGraph from '@/hooks/api/use-graph';
import type { CausalChainResponse } from '@/types/graph';

vi.mock('@/hooks/use-cytoscape', async (importOriginal) => {
  const actual = await importOriginal<typeof useCytoscapeModule>();
  return {
    ...actual,
    useCytoscape: vi.fn(),
  };
});
vi.mock('@/hooks/api/use-graph');

type ChainsMutation = ReturnType<typeof useGraph.useCausalChains>;

const cytoscapeApi = {
  containerRef: { current: null },
  cyInstance: null,
  isLoading: false,
  initialize: vi.fn(),
  destroy: vi.fn(),
  setElements: vi.fn(),
  addElements: vi.fn(),
  removeElements: vi.fn(),
  runLayout: vi.fn(),
  fit: vi.fn(),
  center: vi.fn(),
  zoom: vi.fn(),
  getZoom: vi.fn(() => 1),
  selectNodes: vi.fn(),
  clearSelection: vi.fn(),
  getSelectedNodeIds: vi.fn(() => []),
  highlightNode: vi.fn(),
  unhighlightNode: vi.fn(),
  clearHighlights: vi.fn(),
  exportPng: vi.fn(),
};

function mockChains(overrides: Partial<ChainsMutation> = {}) {
  vi.mocked(useGraph.useCausalChains).mockReturnValue({
    mutate: vi.fn(),
    data: undefined,
    error: null,
    isPending: false,
    ...overrides,
  } as unknown as ChainsMutation);
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(useCytoscapeModule.useCytoscape).mockReturnValue(
    cytoscapeApi as unknown as ReturnType<typeof useCytoscapeModule.useCytoscape>
  );
  mockChains();
});

describe('ActiveCausalChains — no SAMPLE_ELEMENTS graph', () => {
  it('initializes Cytoscape with an EMPTY element set, not the fabricated demo graph', () => {
    render(<ActiveCausalChains />);

    const config = vi.mocked(useCytoscapeModule.useCytoscape).mock.calls[0][0];
    expect(config?.elements ?? []).toHaveLength(0);
  });

  it('renders an honest empty state when the API returns zero chains', () => {
    mockChains({
      data: {
        chains: [],
        total_chains: 0,
        query_latency_ms: 12,
      } as unknown as CausalChainResponse,
    } as unknown as Partial<ChainsMutation>);

    render(<ActiveCausalChains />);
    expect(screen.getByText(/no causal chains/i)).toBeInTheDocument();
  });

  it('renders a labeled error state when chain discovery fails', () => {
    mockChains({
      error: new Error('graph service unavailable'),
    } as unknown as Partial<ChainsMutation>);

    render(<ActiveCausalChains />);
    expect(screen.getByText(/unable to load causal chains/i)).toBeInTheDocument();
  });

  it('pushes real chain elements into the graph when the API returns chains', () => {
    mockChains({
      data: {
        chains: [
          {
            nodes: [
              { id: 'n1', name: 'Rep Visits', type: 'Action' },
              { id: 'n2', name: 'TRx', type: 'KPI' },
            ],
            relationships: [
              { source_id: 'n1', target_id: 'n2', confidence: 0.9 },
            ],
            path_length: 1,
            total_confidence: 0.9,
          },
        ],
        total_chains: 1,
        query_latency_ms: 30,
      } as unknown as CausalChainResponse,
    } as unknown as Partial<ChainsMutation>);

    render(<ActiveCausalChains />);

    expect(cytoscapeApi.setElements).toHaveBeenCalled();
    const calls = cytoscapeApi.setElements.mock.calls;
    const elements = calls[calls.length - 1][0] as Array<{
      data: { id?: string; label?: string };
    }>;
    const labels = elements.map((e) => e.data.label).filter(Boolean);
    expect(labels).toContain('Rep Visits');
    // The fabricated demo graph must not be re-injected.
    expect(labels).not.toContain('Detailing Frequency');
  });
});
