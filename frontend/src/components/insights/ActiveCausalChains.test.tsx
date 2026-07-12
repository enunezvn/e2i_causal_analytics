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
import { render, screen, act } from '@testing-library/react';
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

  it('does NOT invent a 0.5 edge weight when the API omits relationship confidence', () => {
    mockChains({
      data: {
        chains: [
          {
            nodes: [
              { id: 'n1', name: 'Rep Visits', type: 'Action' },
              { id: 'n2', name: 'TRx', type: 'KPI' },
            ],
            // confidence intentionally omitted — optional on the wire.
            relationships: [{ source_id: 'n1', target_id: 'n2' }],
            path_length: 1,
          },
        ],
        total_chains: 1,
        query_latency_ms: 30,
      } as unknown as CausalChainResponse,
    } as unknown as Partial<ChainsMutation>);

    render(<ActiveCausalChains />);

    const calls = cytoscapeApi.setElements.mock.calls;
    const elements = calls[calls.length - 1][0] as Array<{
      data: { source?: string; weight?: number };
    }>;
    const edge = elements.find((e) => e.data.source === 'n1');
    expect(edge).toBeDefined();
    // Absent confidence must stay absent — never a fabricated 0.5.
    expect(edge?.data.weight).toBeUndefined();
  });

  it('emits UNIQUE edge ids when several chains share the same start node', () => {
    // Live regression: chains 8-12 of the deployed payload all start at
    // var:treatment_arm; the old id template `edge-${chain.nodes[0].id}-${idx}`
    // collided across chains and Cytoscape throws on duplicate ids just like
    // empty ones ('Can not create second element with ID ...').
    mockChains({
      data: {
        chains: [
          {
            nodes: [
              { id: 'shared', name: 'Shared Start', type: 'Action' },
              { id: 'mid1', name: 'Mid 1', type: 'HCP' },
            ],
            relationships: [
              { type: 'CAUSES', source_id: 'shared', target_id: 'mid1', confidence: 0.8 },
            ],
            path_length: 1,
            total_confidence: 0.8,
          },
          {
            nodes: [
              { id: 'shared', name: 'Shared Start', type: 'Action' },
              { id: 'mid2', name: 'Mid 2', type: 'HCP' },
            ],
            relationships: [
              { type: 'CAUSES', source_id: 'shared', target_id: 'mid2', confidence: 0.7 },
            ],
            path_length: 1,
            total_confidence: 0.7,
          },
        ],
        total_chains: 2,
        query_latency_ms: 25,
      } as unknown as CausalChainResponse,
    } as unknown as Partial<ChainsMutation>);

    render(<ActiveCausalChains />);

    const calls = cytoscapeApi.setElements.mock.calls;
    const elements = calls[calls.length - 1][0] as Array<{ data: { id?: string } }>;
    const ids = elements.map((e) => e.data.id);
    expect(new Set(ids).size).toBe(ids.length);
    // Both distinct edges must survive the uniqueness scheme.
    expect(elements.filter((e) => e.data.id?.startsWith('edge-'))).toHaveLength(2);
  });

  it('skips empty-id nodes and their incident edges instead of crashing the card', () => {
    // Live regression: id-less seeded KPI nodes serialized as id:"" and the
    // bridge edge's target_id:"" — Cytoscape threw 'Can not create element
    // with invalid string ID ``' and the ErrorBoundary killed the whole card.
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    mockChains({
      data: {
        chains: [
          {
            nodes: [
              { id: 'var:persistent_180d', name: 'persistent_180d', type: 'Variable' },
              { id: '', name: 'Patient_Retention', type: 'KPI' },
            ],
            relationships: [
              {
                type: 'CAUSES',
                source_id: 'var:persistent_180d',
                target_id: '',
                confidence: 0.7,
              },
            ],
            path_length: 1,
            total_confidence: 0.7,
          },
          {
            nodes: [
              { id: 'ok1', name: 'OK 1', type: 'Action' },
              { id: 'ok2', name: 'OK 2', type: 'KPI' },
            ],
            relationships: [
              { type: 'CAUSES', source_id: 'ok1', target_id: 'ok2', confidence: 0.9 },
            ],
            path_length: 1,
            total_confidence: 0.9,
          },
        ],
        total_chains: 2,
        query_latency_ms: 25,
      } as unknown as CausalChainResponse,
    } as unknown as Partial<ChainsMutation>);

    render(<ActiveCausalChains />);

    const calls = cytoscapeApi.setElements.mock.calls;
    const elements = calls[calls.length - 1][0] as Array<{
      data: { id?: string; source?: string; target?: string };
    }>;
    // No element may carry an empty id, and no edge may reference one.
    expect(elements.every((e) => e.data.id)).toBe(true);
    expect(elements.every((e) => e.data.target !== '')).toBe(true);
    // The healthy chain still renders fully.
    const ids = elements.map((e) => e.data.id);
    expect(ids).toContain('ok1');
    expect(ids).toContain('ok2');
    // Skips are observable, not silent.
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('wires an onEdgeClick handler into useCytoscape (the hint text promises it)', () => {
    render(<ActiveCausalChains />);

    const handlers = vi.mocked(useCytoscapeModule.useCytoscape).mock.calls[0][1];
    expect(handlers?.onEdgeClick).toBeTypeOf('function');
  });

  it('shows a causal-strength panel when an edge with confidence is clicked', () => {
    mockChains({
      data: {
        chains: [
          {
            nodes: [
              { id: 'n1', name: 'Rep Visits', type: 'Variable' },
              { id: 'n2', name: 'TRx', type: 'KPI' },
            ],
            relationships: [
              { type: 'CAUSES', source_id: 'n1', target_id: 'n2', confidence: 0.9 },
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

    const calls = cytoscapeApi.setElements.mock.calls;
    const elements = calls[calls.length - 1][0] as Array<{
      data: Record<string, unknown>;
    }>;
    const edge = elements.find((e) => e.data.source === 'n1');
    expect(edge).toBeDefined();

    const handlers = vi.mocked(useCytoscapeModule.useCytoscape).mock.calls[0][1];
    act(() => {
      handlers?.onEdgeClick?.(edge!.data.id as string, edge!.data);
    });

    // The panel names both endpoints and states the strength from real data.
    expect(screen.getByText(/rep visits/i)).toBeInTheDocument();
    expect(screen.getByText(/trx/i)).toBeInTheDocument();
    expect(screen.getByText(/0\.90/)).toBeInTheDocument();
    expect(screen.getByText(/CAUSES/)).toBeInTheDocument();
  });

  it('shows an honest unknown-strength message for edges without confidence', () => {
    mockChains({
      data: {
        chains: [
          {
            nodes: [
              { id: 'n1', name: 'Rep Visits', type: 'Variable' },
              { id: 'n2', name: 'TRx', type: 'KPI' },
            ],
            relationships: [{ type: 'CAUSES', source_id: 'n1', target_id: 'n2' }],
            path_length: 1,
          },
        ],
        total_chains: 1,
        query_latency_ms: 30,
      } as unknown as CausalChainResponse,
    } as unknown as Partial<ChainsMutation>);

    render(<ActiveCausalChains />);

    const calls = cytoscapeApi.setElements.mock.calls;
    const elements = calls[calls.length - 1][0] as Array<{
      data: Record<string, unknown>;
    }>;
    const edge = elements.find((e) => e.data.source === 'n1');
    const handlers = vi.mocked(useCytoscapeModule.useCytoscape).mock.calls[0][1];
    act(() => {
      handlers?.onEdgeClick?.(edge!.data.id as string, edge!.data);
    });

    // No confidence on the wire -> the panel says so, never invents a figure.
    expect(screen.getByText(/unknown/i)).toBeInTheDocument();
    expect(screen.queryByText(/0\.50/)).not.toBeInTheDocument();
  });

  it('carries relType, endpoint labels, and ate_estimate into edge element data', () => {
    mockChains({
      data: {
        chains: [
          {
            nodes: [
              { id: 'n1', name: 'Rep Visits', type: 'Variable' },
              { id: 'n2', name: 'TRx', type: 'KPI' },
            ],
            relationships: [
              {
                type: 'CAUSES',
                source_id: 'n1',
                target_id: 'n2',
                confidence: 0.9,
                properties: { ate_estimate: 0.041 },
              },
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

    const calls = cytoscapeApi.setElements.mock.calls;
    const elements = calls[calls.length - 1][0] as Array<{
      data: Record<string, unknown>;
    }>;
    const edge = elements.find((e) => e.data.source === 'n1');
    expect(edge?.data.relType).toBe('CAUSES');
    expect(edge?.data.sourceLabel).toBe('Rep Visits');
    expect(edge?.data.targetLabel).toBe('TRx');
    expect(edge?.data.ate).toBe(0.041);
  });

  it('clicking an edge after a node replaces the node panel with the edge panel', () => {
    mockChains({
      data: {
        chains: [
          {
            nodes: [
              { id: 'n1', name: 'Rep Visits', type: 'Variable' },
              { id: 'n2', name: 'TRx', type: 'KPI' },
            ],
            relationships: [
              { type: 'CAUSES', source_id: 'n1', target_id: 'n2', confidence: 0.9 },
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

    const handlers = vi.mocked(useCytoscapeModule.useCytoscape).mock.calls[0][1];
    act(() => {
      handlers?.onNodeClick?.('n1', { label: 'Rep Visits', type: 'Variable' });
    });
    expect(screen.getByText(/click edges to see causal strength/i)).toBeInTheDocument();

    const calls = cytoscapeApi.setElements.mock.calls;
    const elements = calls[calls.length - 1][0] as Array<{
      data: Record<string, unknown>;
    }>;
    const edge = elements.find((e) => e.data.source === 'n1');
    act(() => {
      handlers?.onEdgeClick?.(edge!.data.id as string, edge!.data);
    });
    // Node hint gone, strength panel shown.
    expect(screen.queryByText(/click edges to see causal strength/i)).not.toBeInTheDocument();
    expect(screen.getByText(/0\.90/)).toBeInTheDocument();
  });
});

describe('ActiveCausalChains — SSOT role-driven node categories', () => {
  it('colors nodes by the role property stamped on causal_paths variables', () => {
    mockChains({
      data: {
        chains: [
          {
            nodes: [
              {
                id: 'var:treatment_arm',
                name: 'treatment_arm',
                type: 'Variable',
                properties: { role: 'driver' },
              },
              {
                id: 'var:adherence',
                name: 'adherence',
                type: 'Variable',
                properties: { role: 'mediator' },
              },
              {
                id: 'var:trx_volume',
                name: 'trx_volume',
                type: 'Variable',
                properties: { role: 'outcome' },
              },
            ],
            relationships: [
              {
                type: 'CAUSES',
                source_id: 'var:treatment_arm',
                target_id: 'var:adherence',
                confidence: 0.8,
              },
              {
                type: 'CAUSES',
                source_id: 'var:adherence',
                target_id: 'var:trx_volume',
                confidence: 0.8,
              },
            ],
            path_length: 2,
            total_confidence: 0.8,
          },
        ],
        total_chains: 1,
        query_latency_ms: 30,
      } as unknown as CausalChainResponse,
    } as unknown as Partial<ChainsMutation>);

    render(<ActiveCausalChains />);

    const calls = cytoscapeApi.setElements.mock.calls;
    const elements = calls[calls.length - 1][0] as Array<{
      data: { id?: string; vizType?: string };
    }>;
    const byId = new Map(elements.map((e) => [e.data.id, e.data.vizType]));
    expect(byId.get('var:treatment_arm')).toBe('driver');
    expect(byId.get('var:adherence')).toBe('mediator');
    expect(byId.get('var:trx_volume')).toBe('outcome');
  });

  it('falls back to the entity-type map when no role is stamped', () => {
    mockChains({
      data: {
        chains: [
          {
            nodes: [
              { id: 'var:x', name: 'x', type: 'Variable' },
              { id: 'kpi:TRx', name: 'TRx', type: 'KPI' },
            ],
            relationships: [
              { type: 'CAUSES', source_id: 'var:x', target_id: 'kpi:TRx', confidence: 0.7 },
            ],
            path_length: 1,
            total_confidence: 0.7,
          },
        ],
        total_chains: 1,
        query_latency_ms: 30,
      } as unknown as CausalChainResponse,
    } as unknown as Partial<ChainsMutation>);

    render(<ActiveCausalChains />);

    const calls = cytoscapeApi.setElements.mock.calls;
    const elements = calls[calls.length - 1][0] as Array<{
      data: { id?: string; vizType?: string };
    }>;
    const byId = new Map(elements.map((e) => [e.data.id, e.data.vizType]));
    expect(byId.get('var:x')).toBe('mediator');
    expect(byId.get('kpi:TRx')).toBe('outcome');
  });

  it('legend names the categories the graph can actually show — no dead Moderator entry', () => {
    render(<ActiveCausalChains />);

    expect(screen.getByText('Driver')).toBeInTheDocument();
    expect(screen.getByText('Mediator')).toBeInTheDocument();
    expect(screen.getByText('Outcome')).toBeInTheDocument();
    // Moderation is not representable as a cause->effect edge in this DAG;
    // an amber "Moderator" legend entry promised nodes that can never render.
    expect(screen.queryByText('Moderator')).not.toBeInTheDocument();
    expect(screen.queryByText('Intervention')).not.toBeInTheDocument();
  });

  it('clears previously rendered elements when a later response has zero chains', () => {
    mockChains({
      data: {
        chains: [],
        total_chains: 0,
        query_latency_ms: 10,
      } as unknown as CausalChainResponse,
    } as unknown as Partial<ChainsMutation>);

    render(<ActiveCausalChains />);

    // The graph must be explicitly emptied so stale chains cannot linger
    // beneath the empty-state overlay.
    expect(cytoscapeApi.setElements).toHaveBeenCalledWith([]);
  });
});
