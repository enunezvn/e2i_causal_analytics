/**
 * KnowledgeGraph Page Tests
 * =========================
 *
 * Tests for the KnowledgeGraph page component.
 * Includes tests for:
 * - Page header
 * - Search functionality
 * - Stats cards (computed from the RENDERED graph, not the global stats endpoint)
 * - Per-brand causal graph (brand dropdown + brand-tagged CAUSES subgraph)
 * - Graph visualization
 * - Node/Edge details panel
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import KnowledgeGraphPage from './KnowledgeGraph';

// Mock the graph hooks. NOTE: the page intentionally does NOT call useGraphStats
// — stats are computed from the rendered (scoped + connected) graph so the cards
// match the canvas. Only useNodes/useRelationships are consumed.
vi.mock('@/hooks/api/use-graph', () => ({
  useNodes: vi.fn(),
  useRelationships: vi.fn(),
}));

// Mock the KnowledgeGraph visualization component
vi.mock('@/components/visualizations/KnowledgeGraph', () => ({
  KnowledgeGraph: ({ nodes, relationships, isLoading, error, onNodeSelect, onEdgeSelect, styleEdgesByEffect }: {
    nodes: unknown[];
    relationships: unknown[];
    isLoading: boolean;
    error: Error | null;
    onNodeSelect?: (node: unknown) => void;
    onEdgeSelect?: (edge: unknown) => void;
    styleEdgesByEffect?: boolean;
  }) => (
    <div data-testid="knowledge-graph-viz">
      <div data-testid="nodes-count">{nodes.length}</div>
      <div data-testid="relationships-count">{relationships.length}</div>
      <div data-testid="is-loading">{String(isLoading)}</div>
      <div data-testid="has-error">{String(!!error)}</div>
      <div data-testid="style-by-effect">{String(!!styleEdgesByEffect)}</div>
      <button onClick={() => onNodeSelect?.({ id: 'test-node', name: 'Test Node', type: 'Agent', properties: {}, created_at: '2026-01-04' })}>
        Select Node
      </button>
      <button onClick={() => onEdgeSelect?.({ id: 'test-edge', type: 'RELATES_TO', source_id: 'a', target_id: 'b', properties: { ate_estimate: 0.42 }, confidence: 0.85, created_at: '2026-01-04' })}>
        Select Edge
      </button>
    </div>
  ),
}));

// Mock ONLY the KG insight mutation hook (the rest of the barrel stays real):
// the page's stale-insight guard is exercised by controlling data/reset here.
vi.mock('@/hooks/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/api')>()),
  useKnowledgeGraphInsight: vi.fn(),
}));

import { useNodes, useRelationships } from '@/hooks/api/use-graph';
import { useKnowledgeGraphInsight } from '@/hooks/api';

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

// Per-brand causal gold standard: Variable nodes connected by brand-tagged
// CAUSES edges (sync_causal_paths_to_falkordb stamps `brand` on each edge). The
// page renders ONLY the selected brand's causal subgraph. The default brand is
// Kisqali, whose chain (treatment -> outcome -> trx) yields 3 nodes / 2 edges.
// The Fabhalta chain (adherence -> persistence) must NOT appear under Kisqali.
const mockNodes = [
  // Kisqali main chain (3 nodes -> kept).
  { id: 'var:treatment', name: 'treatment', type: 'Variable', properties: {}, created_at: '2026-01-04' },
  { id: 'var:outcome', name: 'outcome', type: 'Variable', properties: {}, created_at: '2026-01-04' },
  { id: 'var:trx', name: 'trx', type: 'Variable', properties: {}, created_at: '2026-01-04' },
  // Kisqali off-chain pair (size-2 component -> pruned).
  { id: 'var:sideA', name: 'sideA', type: 'Variable', properties: {}, created_at: '2026-01-04' },
  { id: 'var:sideB', name: 'sideB', type: 'Variable', properties: {}, created_at: '2026-01-04' },
  // Fabhalta chain (3 nodes -> kept).
  { id: 'var:adherence', name: 'adherence', type: 'Variable', properties: {}, created_at: '2026-01-04' },
  { id: 'var:persistence', name: 'persistence', type: 'Variable', properties: {}, created_at: '2026-01-04' },
  { id: 'var:discontinuation', name: 'discontinuation', type: 'Variable', properties: {}, created_at: '2026-01-04' },
];

const mockRelationships = [
  // Kisqali main chain (default). Second edge uses lowercase 'kisqali' to prove
  // the brand match is case-insensitive (the graph has Kisqali AND kisqali dupes).
  { id: 'k1', type: 'CAUSES', source_id: 'var:treatment', target_id: 'var:outcome', properties: { brand: 'Kisqali' }, confidence: 0.9, created_at: '2026-01-04' },
  { id: 'k2', type: 'CAUSES', source_id: 'var:outcome', target_id: 'var:trx', properties: { brand: 'kisqali' }, confidence: 0.9, created_at: '2026-01-04' },
  // Kisqali off-chain PAIR — a within-brand size-2 component that must be pruned.
  { id: 'k3', type: 'CAUSES', source_id: 'var:sideA', target_id: 'var:sideB', properties: { brand: 'Kisqali' }, confidence: 0.7, created_at: '2026-01-04' },
  // Fabhalta chain (different brand — excluded when Kisqali is selected).
  { id: 'f1', type: 'CAUSES', source_id: 'var:adherence', target_id: 'var:persistence', properties: { brand: 'Fabhalta' }, confidence: 0.8, created_at: '2026-01-04' },
  { id: 'f2', type: 'CAUSES', source_id: 'var:persistence', target_id: 'var:discontinuation', properties: { brand: 'Fabhalta' }, confidence: 0.8, created_at: '2026-01-04' },
];

/** Assert a number is shown inside the stats card identified by its description. */
function expectStatValue(description: string, value: string) {
  // CardDescription and the CardTitle holding the value share the CardHeader parent.
  const header = screen.getByText(description).parentElement as HTMLElement;
  expect(within(header).getByText(value)).toBeInTheDocument();
}

// Mutable KG-insight mutation state, reassigned per test. A single object per
// test keeps `kgInsight.reset` referentially stable across re-renders, exactly
// like TanStack Query's real useMutation result.
let mockKgInsight: {
  mutate: ReturnType<typeof vi.fn>;
  reset: ReturnType<typeof vi.fn>;
  data: Record<string, unknown> | undefined;
  isPending: boolean;
  error: Error | null;
};

describe('KnowledgeGraphPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    (useNodes as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { nodes: mockNodes },
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });
    (useRelationships as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { relationships: mockRelationships },
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });
    mockKgInsight = {
      mutate: vi.fn(),
      reset: vi.fn(),
      data: undefined,
      isPending: false,
      error: null,
    };
    (useKnowledgeGraphInsight as ReturnType<typeof vi.fn>).mockImplementation(
      () => mockKgInsight
    );
  });

  // =========================================================================
  // PAGE HEADER TESTS
  // =========================================================================

  describe('Page Header', () => {
    it('renders page title', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Knowledge Graph')).toBeInTheDocument();
    });

    it('renders page description', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText(/Explore the knowledge graph visualization/)).toBeInTheDocument();
    });
  });

  // =========================================================================
  // SEARCH FUNCTIONALITY TESTS
  // =========================================================================

  describe('Search Functionality', () => {
    it('renders search input', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByPlaceholderText(/Search nodes by name or type/)).toBeInTheDocument();
    });

    it('filters nodes when typing in search', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const searchInput = screen.getByPlaceholderText(/Search nodes by name or type/);
      fireEvent.change(searchInput, { target: { value: 'Patient' } });

      // Should show search results info
      expect(screen.getByText(/Found \d+ nodes/)).toBeInTheDocument();
    });

    it('shows clear button when search has value', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const searchInput = screen.getByPlaceholderText(/Search nodes by name or type/);
      fireEvent.change(searchInput, { target: { value: 'test' } });

      // Clear button should appear
      const buttons = screen.getAllByRole('button');
      const clearButton = buttons.find((btn) => btn.querySelector('.lucide-x'));
      expect(clearButton).toBeTruthy();
    });

    it('clears search when clear button clicked', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const searchInput = screen.getByPlaceholderText(/Search nodes by name or type/) as HTMLInputElement;
      fireEvent.change(searchInput, { target: { value: 'test' } });

      // Click clear button
      const buttons = screen.getAllByRole('button');
      const clearButton = buttons.find(btn => btn.querySelector('.lucide-x'));
      if (clearButton) {
        fireEvent.click(clearButton);
      }

      expect(searchInput.value).toBe('');
    });
  });

  // =========================================================================
  // STATS CARDS TESTS (reflect the rendered graph)
  // =========================================================================

  describe('Stats Cards', () => {
    it('displays total nodes count from the rendered graph', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Total Nodes')).toBeInTheDocument();
      // Default 'All brands': the whole causal graph (8 connected Variables).
      expectStatValue('Total Nodes', '8');
    });

    it('displays total relationships count from the rendered graph', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Total Relationships')).toBeInTheDocument();
      expectStatValue('Total Relationships', '5');
    });

    it('displays selected info card', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Selected')).toBeInTheDocument();
      expect(screen.getByText('None')).toBeInTheDocument();
    });

    it('shows node type badges computed from the rendered nodes', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      // Default 'All brands' shows all 8 connected Variables across brands.
      expect(screen.getByText('Variable: 8')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // PER-BRAND CAUSAL GRAPH TESTS
  // =========================================================================

  describe('Causal gold-standard graph', () => {
    it('defaults to All brands → the whole causal graph (no per-brand restriction)', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      // All brands: Kisqali chain (treatment/outcome/trx + sideA/sideB) AND the
      // Fabhalta chain (adherence/persistence/discontinuation) = 8 nodes / 5 edges.
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('8');
      expect(screen.getByTestId('relationships-count')).toHaveTextContent('5');
    });

    it('fetches the full causal node + relationship layer, curated (no agent-written pollution)', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const nodesCall = (useNodes as ReturnType<typeof vi.fn>).mock.calls[0][0];
      // Treatment dropped from the fetch: product/regimen nodes carry no causal
      // edge (CAUSES/EXPLAINS/INFLUENCES/AFFECTS), so they only ever rendered as
      // isolated singletons. We no longer fetch them.
      expect(nodesCall.entity_types).toBe('Variable,KPI,CausalPath,Region');
      // curated_only excludes agent-written runtime nodes from the gold-standard view.
      expect(nodesCall.curated_only).toBe(true);
      const relCall = (useRelationships as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(relCall.relationship_types).toBe('CAUSES,EXPLAINS,INFLUENCES,AFFECTS');
      expect(relCall.curated_only).toBe(true);
    });

    it('narrows to one brand when the dropdown changes (Fabhalta)', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const select = screen.getByRole('combobox', { name: /brand/i });
      fireEvent.change(select, { target: { value: 'Fabhalta' } });

      // Only Fabhalta's chain (adherence->persistence->discontinuation): 3 / 2.
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('3');
      expect(screen.getByTestId('relationships-count')).toHaveTextContent('2');
    });

    it('keeps a brand\'s full chains incl. off-chain pairs (no size pruning)', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const select = screen.getByRole('combobox', { name: /brand/i });
      fireEvent.change(select, { target: { value: 'Kisqali' } });

      // Kisqali: main chain (treatment/outcome/trx) + the sideA-sideB pair — all
      // KEPT now (5 nodes / 3 edges); the Fabhalta chain is excluded.
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('5');
      expect(screen.getByTestId('relationships-count')).toHaveTextContent('3');
    });

    it('renders a brand dropdown defaulting to All brands', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const select = screen.getByRole('combobox', { name: /brand/i }) as HTMLSelectElement;
      expect(select).toBeInTheDocument();
      expect(select.value).toBe('All');
    });
  });

  // =========================================================================
  // VARIABLE SELECTOR TESTS
  // =========================================================================
  // The Variable dropdown derives its options from the brand-scoped graph's
  // Variable nodes (client-side, no refetch) and narrows the canvas to the
  // causal chains through the selected variable (ancestors ∪ descendants along
  // CAUSES + attached structural context).

  describe('Variable selector', () => {
    it('renders defaulting to All variables, with options from the rendered graph', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const select = screen.getByRole('combobox', { name: /variable/i }) as HTMLSelectElement;
      expect(select.value).toBe('All');
      // All-brands scope → all 8 connected Variables are offered.
      const options = within(select).getAllByRole('option');
      expect(options).toHaveLength(9); // 'All variables' + 8
      expect(within(select).getByRole('option', { name: 'outcome' })).toBeInTheDocument();
    });

    it('narrows the graph to the chains through the selected variable', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const select = screen.getByRole('combobox', { name: /variable/i });
      fireEvent.change(select, { target: { value: 'var:outcome' } });

      // outcome's chain: treatment (ancestor) + outcome + trx (descendant).
      // The sideA/sideB pair and the Fabhalta chain are excluded.
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('3');
      expect(screen.getByTestId('relationships-count')).toHaveTextContent('2');
    });

    it('keeps the selected variable across a brand switch when still in scope', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const variable = screen.getByRole('combobox', { name: /variable/i }) as HTMLSelectElement;
      fireEvent.change(variable, { target: { value: 'var:outcome' } });
      const brand = screen.getByRole('combobox', { name: /brand/i });
      fireEvent.change(brand, { target: { value: 'Kisqali' } });

      // outcome exists in the Kisqali scope → the narrowed view survives the
      // switch (comparing one variable across brands is the selector's point).
      expect(variable.value).toBe('var:outcome');
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('3');
      expect(screen.getByTestId('relationships-count')).toHaveTextContent('2');
    });

    it('clamps back to All variables when the new brand scope lacks the variable', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const variable = screen.getByRole('combobox', { name: /variable/i }) as HTMLSelectElement;
      fireEvent.change(variable, { target: { value: 'var:adherence' } });
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('3');

      const brand = screen.getByRole('combobox', { name: /brand/i });
      fireEvent.change(brand, { target: { value: 'Kisqali' } });

      // adherence is Fabhalta-only → selection clamps to 'All' and the full
      // Kisqali scope renders (5 nodes / 3 edges).
      expect(variable.value).toBe('All');
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('5');
      expect(screen.getByTestId('relationships-count')).toHaveTextContent('3');
    });
  });

  // =========================================================================
  // EFFECT-STYLING TESTS
  // =========================================================================
  // Brands share the chain topology by design — the brand-specific signal is
  // each edge's ATE/confidence. When ONE brand is selected the viz styles edges
  // by that brand's estimates; the All view stays neutral (the deduped
  // representative edge's ATE belongs to a single arbitrary brand).

  describe('Effect styling', () => {
    it('is off for All brands and on for a single brand, with a legend', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('style-by-effect')).toHaveTextContent('false');
      expect(screen.queryByText(/Edges reflect/)).not.toBeInTheDocument();

      const brand = screen.getByRole('combobox', { name: /brand/i });
      fireEvent.change(brand, { target: { value: 'Fabhalta' } });

      expect(screen.getByTestId('style-by-effect')).toHaveTextContent('true');
      expect(screen.getByText(/Edges reflect Fabhalta/)).toBeInTheDocument();
    });
  });

  // =========================================================================
  // EDGE DE-DUPLICATION TESTS
  // =========================================================================
  // The synthetic gold standard stamps the SAME logical CAUSES edge once per
  // (brand × region) — up to 3 brands × 4 regions = 12 parallel copies — which
  // rendered as an unreadable hairball. The page collapses parallel edges that
  // share (source, type, target) into one logical edge.

  describe('Edge de-duplication', () => {
    it('collapses parallel (brand×region) edges of the same source→target→type into one', () => {
      (useNodes as ReturnType<typeof vi.fn>).mockReturnValue({
        data: {
          nodes: [
            { id: 'var:a', name: 'a', type: 'Variable', properties: {}, created_at: '2026-01-04' },
            { id: 'var:b', name: 'b', type: 'Variable', properties: {}, created_at: '2026-01-04' },
          ],
        },
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      });
      (useRelationships as ReturnType<typeof vi.fn>).mockReturnValue({
        data: {
          relationships: [
            { id: 'e1', type: 'CAUSES', source_id: 'var:a', target_id: 'var:b', properties: { brand: 'Kisqali', region: 'northeast' }, confidence: 0.9, created_at: '2026-01-04' },
            { id: 'e2', type: 'CAUSES', source_id: 'var:a', target_id: 'var:b', properties: { brand: 'Kisqali', region: 'south' }, confidence: 0.9, created_at: '2026-01-04' },
            { id: 'e3', type: 'CAUSES', source_id: 'var:a', target_id: 'var:b', properties: { brand: 'Fabhalta', region: 'west' }, confidence: 0.9, created_at: '2026-01-04' },
          ],
        },
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      });

      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      // 3 physical (brand×region) edges between the same pair collapse to 1.
      expect(screen.getByTestId('relationships-count')).toHaveTextContent('1');
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('2');
    });

    it('does NOT collapse edges that differ in direction or type', () => {
      (useNodes as ReturnType<typeof vi.fn>).mockReturnValue({
        data: {
          nodes: [
            { id: 'var:a', name: 'a', type: 'Variable', properties: {}, created_at: '2026-01-04' },
            { id: 'var:b', name: 'b', type: 'Variable', properties: {}, created_at: '2026-01-04' },
          ],
        },
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      });
      (useRelationships as ReturnType<typeof vi.fn>).mockReturnValue({
        data: {
          relationships: [
            { id: 'e1', type: 'CAUSES', source_id: 'var:a', target_id: 'var:b', properties: { brand: 'Kisqali', region: 'northeast' }, confidence: 0.9, created_at: '2026-01-04' },
            { id: 'e2', type: 'CAUSES', source_id: 'var:a', target_id: 'var:b', properties: { brand: 'Kisqali', region: 'south' }, confidence: 0.9, created_at: '2026-01-04' },
            // reverse direction — a genuinely distinct edge, must survive
            { id: 'e3', type: 'CAUSES', source_id: 'var:b', target_id: 'var:a', properties: { brand: 'Kisqali', region: 'south' }, confidence: 0.9, created_at: '2026-01-04' },
          ],
        },
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      });

      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      // a→b (2 copies → 1) plus the distinct b→a (1) = 2 logical edges.
      expect(screen.getByTestId('relationships-count')).toHaveTextContent('2');
    });
  });

  // =========================================================================
  // LOADING STATE TESTS
  // =========================================================================

  describe('Loading States', () => {
    it('shows loading state when nodes are loading', () => {
      (useNodes as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
        refetch: vi.fn(),
      });

      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('is-loading')).toHaveTextContent('true');
    });

    it('shows loading state when relationships are loading', () => {
      (useRelationships as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
        refetch: vi.fn(),
      });

      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('is-loading')).toHaveTextContent('true');
    });

    it('shows skeleton in stats when loading', () => {
      (useNodes as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
        refetch: vi.fn(),
      });

      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      // Should have animated pulse skeleton elements
      const skeletons = document.querySelectorAll('.animate-pulse');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });

  // =========================================================================
  // ERROR STATE TESTS
  // =========================================================================

  describe('Error States', () => {
    it('shows error state when nodes fail to load', () => {
      (useNodes as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load nodes'),
        refetch: vi.fn(),
      });

      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('has-error')).toHaveTextContent('true');
    });

    it('shows error state when relationships fail to load', () => {
      (useRelationships as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load relationships'),
        refetch: vi.fn(),
      });

      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('has-error')).toHaveTextContent('true');
    });
  });

  // =========================================================================
  // NODE SELECTION TESTS
  // =========================================================================

  describe('Node Selection', () => {
    it('updates selected card when node is selected', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      // Initially shows "None"
      expect(screen.getByText('None')).toBeInTheDocument();

      // Click the test select node button
      fireEvent.click(screen.getByText('Select Node'));

      // Should show Node Details panel (the card title changes)
      expect(screen.getByText('Node Details')).toBeInTheDocument();
    });

    it('shows node details panel when node selected', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByText('Select Node'));

      expect(screen.getByText('Node Details')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // EDGE SELECTION TESTS
  // =========================================================================

  describe('Edge Selection', () => {
    it('updates selected card when edge is selected', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      // Click the test select edge button
      fireEvent.click(screen.getByText('Select Edge'));

      // Should show Relationship Details panel
      expect(screen.getByText('Relationship Details')).toBeInTheDocument();
    });

    it('shows edge details panel when edge selected', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByText('Select Edge'));

      expect(screen.getByText('Relationship Details')).toBeInTheDocument();
    });

    it('shows confidence percentage for selected edge', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByText('Select Edge'));

      expect(screen.getByText('85.0%')).toBeInTheDocument();
    });

    it('shows the effect size (ATE) when the edge carries one', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByText('Select Edge'));

      expect(screen.getByText('Effect size (ATE)')).toBeInTheDocument();
      expect(screen.getByText('0.420')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // GRAPH VISUALIZATION TESTS
  // =========================================================================

  describe('Graph Visualization', () => {
    it('renders the visualization component', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('knowledge-graph-viz')).toBeInTheDocument();
    });

    it('passes the connected nodes to visualization', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      // Default 'All brands' → all 8 connected causal Variables.
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('8');
    });

    it('passes relationships to visualization', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('relationships-count')).toHaveTextContent('5');
    });

    it('renders graph card with title', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Graph Visualization')).toBeInTheDocument();
      expect(screen.getByText(/Interactive knowledge graph/)).toBeInTheDocument();
    });
  });

  // =========================================================================
  // STRATEGIC INTERPRETATION TESTS
  // =========================================================================
  // The shared StrategicInsightCard is wired below the stats cards and above the
  // graph visualization. It ALWAYS renders its "Strategic Interpretation" header
  // (the agentic insight is generated lazily via the card's Generate button).

  describe('Strategic Interpretation', () => {
    it('renders the strategic insight card', async () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(
        await screen.findByText(/strategic interpretation/i)
      ).toBeInTheDocument();
    });

    it('generates for the currently selected brand', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const brand = screen.getByRole('combobox', { name: /brand/i });
      fireEvent.change(brand, { target: { value: 'Fabhalta' } });
      fireEvent.click(screen.getByRole('button', { name: /generate strategic insight/i }));

      expect(mockKgInsight.mutate).toHaveBeenCalledWith({
        brand: 'Fabhalta',
        curated_only: true,
        variable: null,
      });
    });

    it('generates for the selected variable scope (page-parity grounding)', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const variable = screen.getByRole('combobox', { name: /variable/i });
      fireEvent.change(variable, { target: { value: 'var:outcome' } });
      fireEvent.click(screen.getByRole('button', { name: /generate strategic insight/i }));

      expect(mockKgInsight.mutate).toHaveBeenCalledWith({
        brand: 'All',
        curated_only: true,
        variable: 'var:outcome',
      });
    });

    it('resets the interpretation on a brand switch (no stale cross-brand text)', () => {
      // The hook always reports data — simulating a resolved mutation whose
      // result would otherwise linger (or a late resolution repopulating data
      // AFTER reset() cleared it, which TanStack mutations do).
      mockKgInsight.data = {
        insight: 'Kisqali-centric interpretation',
        key_takeaways: [],
        grounding: [],
        is_fallback: false,
      };
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      // Never generated in this scope → data is suppressed, Generate offered.
      expect(screen.queryByText('Kisqali-centric interpretation')).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: /generate strategic insight/i }));
      // Submitted for the active scope ('All') → the result now surfaces.
      expect(screen.getByText('Kisqali-centric interpretation')).toBeInTheDocument();

      mockKgInsight.reset.mockClear();
      const brand = screen.getByRole('combobox', { name: /brand/i });
      fireEvent.change(brand, { target: { value: 'Fabhalta' } });

      // Brand switch: the mutation is reset AND the (still-populated) data is
      // suppressed — the card returns to its Generate state under Fabhalta.
      expect(mockKgInsight.reset).toHaveBeenCalled();
      expect(screen.queryByText('Kisqali-centric interpretation')).not.toBeInTheDocument();
      expect(
        screen.getByRole('button', { name: /generate strategic insight/i })
      ).toBeInTheDocument();
    });

    it('resets the interpretation on a variable switch (no stale cross-scope text)', () => {
      // Same late-resolution hazard as the brand switch: the insight generated
      // for the whole-graph scope must not linger once the analyst narrows to
      // one variable's neighborhood — the narrative would describe a graph the
      // canvas no longer shows.
      mockKgInsight.data = {
        insight: 'Whole-graph interpretation',
        key_takeaways: [],
        grounding: [],
        is_fallback: false,
      };
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByRole('button', { name: /generate strategic insight/i }));
      expect(screen.getByText('Whole-graph interpretation')).toBeInTheDocument();

      mockKgInsight.reset.mockClear();
      const variable = screen.getByRole('combobox', { name: /variable/i });
      fireEvent.change(variable, { target: { value: 'var:outcome' } });

      expect(mockKgInsight.reset).toHaveBeenCalled();
      expect(screen.queryByText('Whole-graph interpretation')).not.toBeInTheDocument();
      expect(
        screen.getByRole('button', { name: /generate strategic insight/i })
      ).toBeInTheDocument();
    });
  });

  // =========================================================================
  // EMPTY STATE TESTS
  // =========================================================================

  describe('Empty States', () => {
    it('handles empty nodes gracefully', () => {
      (useNodes as ReturnType<typeof vi.fn>).mockReturnValue({
        data: { nodes: [] },
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      });

      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('nodes-count')).toHaveTextContent('0');
    });

    it('handles empty relationships gracefully', () => {
      (useRelationships as ReturnType<typeof vi.fn>).mockReturnValue({
        data: { relationships: [] },
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      });

      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      // With no edges, every node is a singleton -> all dropped by the filter.
      expect(screen.getByTestId('relationships-count')).toHaveTextContent('0');
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('0');
    });
  });
});
