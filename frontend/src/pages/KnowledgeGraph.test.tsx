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
  KnowledgeGraph: ({ nodes, relationships, isLoading, error, onNodeSelect, onEdgeSelect }: {
    nodes: unknown[];
    relationships: unknown[];
    isLoading: boolean;
    error: Error | null;
    onNodeSelect?: (node: unknown) => void;
    onEdgeSelect?: (edge: unknown) => void;
  }) => (
    <div data-testid="knowledge-graph-viz">
      <div data-testid="nodes-count">{nodes.length}</div>
      <div data-testid="relationships-count">{relationships.length}</div>
      <div data-testid="is-loading">{String(isLoading)}</div>
      <div data-testid="has-error">{String(!!error)}</div>
      <button onClick={() => onNodeSelect?.({ id: 'test-node', name: 'Test Node', type: 'Agent', properties: {}, created_at: '2026-01-04' })}>
        Select Node
      </button>
      <button onClick={() => onEdgeSelect?.({ id: 'test-edge', type: 'RELATES_TO', source_id: 'a', target_id: 'b', confidence: 0.85, created_at: '2026-01-04' })}>
        Select Edge
      </button>
    </div>
  ),
}));

import { useNodes, useRelationships } from '@/hooks/api/use-graph';

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
      expect(nodesCall.entity_types).toBe('Variable,KPI,CausalPath,Region,Treatment');
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
