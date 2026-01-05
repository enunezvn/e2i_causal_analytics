/**
 * KnowledgeGraph Page Tests
 * =========================
 *
 * Tests for the KnowledgeGraph page component.
 * Includes tests for:
 * - Page header
 * - Search functionality
 * - Node type legend
 * - Stats cards
 * - Graph visualization
 * - Node/Edge details panel
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import KnowledgeGraphPage from './KnowledgeGraph';

// Mock the graph hooks
vi.mock('@/hooks/api/use-graph', () => ({
  useNodes: vi.fn(),
  useRelationships: vi.fn(),
  useGraphStats: vi.fn(),
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

import { useNodes, useRelationships, useGraphStats } from '@/hooks/api/use-graph';

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

// Sample data
const mockNodes = [
  { id: 'node-1', name: 'Patient A', type: 'Patient', properties: {}, created_at: '2026-01-04' },
  { id: 'node-2', name: 'HCP Smith', type: 'HCP', properties: {}, created_at: '2026-01-04' },
  { id: 'node-3', name: 'Remibrutinib', type: 'Brand', properties: {}, created_at: '2026-01-04' },
];

const mockRelationships = [
  { id: 'rel-1', type: 'PRESCRIBES', source_id: 'node-2', target_id: 'node-3', confidence: 0.9, created_at: '2026-01-04' },
  { id: 'rel-2', type: 'RECEIVES', source_id: 'node-1', target_id: 'node-3', confidence: 0.85, created_at: '2026-01-04' },
];

const mockGraphStats = {
  total_nodes: 150,
  total_relationships: 420,
  nodes_by_type: { Patient: 50, HCP: 40, Brand: 10, Agent: 20, KPI: 30 },
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
    (useGraphStats as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockGraphStats,
      isLoading: false,
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
      const clearButton = screen.getByRole('button', { name: '' }); // X button
      expect(clearButton).toBeInTheDocument();
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
  // LEGEND TESTS
  // =========================================================================

  describe('Node Type Legend', () => {
    it('renders legend card', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Node Type Legend')).toBeInTheDocument();
    });

    it('displays entity types in legend', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Patient')).toBeInTheDocument();
      expect(screen.getByText('HCP')).toBeInTheDocument();
      expect(screen.getByText('Brand')).toBeInTheDocument();
      expect(screen.getByText('Agent')).toBeInTheDocument();
      expect(screen.getByText('KPI')).toBeInTheDocument();
    });

    it('clicking legend item filters by type', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      const patientLabel = screen.getByText('Patient');
      fireEvent.click(patientLabel);

      // Should update search to filter by Patient type
      const searchInput = screen.getByPlaceholderText(/Search nodes by name or type/) as HTMLInputElement;
      expect(searchInput.value).toBe('Patient');
    });
  });

  // =========================================================================
  // STATS CARDS TESTS
  // =========================================================================

  describe('Stats Cards', () => {
    it('displays total nodes count', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Total Nodes')).toBeInTheDocument();
      expect(screen.getByText('150')).toBeInTheDocument();
    });

    it('displays total relationships count', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Total Relationships')).toBeInTheDocument();
      expect(screen.getByText('420')).toBeInTheDocument();
    });

    it('displays selected info card', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Selected')).toBeInTheDocument();
      expect(screen.getByText('None')).toBeInTheDocument();
    });

    it('shows node type badges in total nodes card', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Patient: 50')).toBeInTheDocument();
      expect(screen.getByText('HCP: 40')).toBeInTheDocument();
      expect(screen.getByText('Brand: 10')).toBeInTheDocument();
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
      (useGraphStats as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: true,
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

    it('passes nodes to visualization', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('nodes-count')).toHaveTextContent('3');
    });

    it('passes relationships to visualization', () => {
      render(<KnowledgeGraphPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('relationships-count')).toHaveTextContent('2');
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
      (useGraphStats as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: false,
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

      expect(screen.getByTestId('relationships-count')).toHaveTextContent('0');
    });
  });
});
