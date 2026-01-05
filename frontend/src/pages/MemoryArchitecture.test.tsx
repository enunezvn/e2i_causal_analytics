/**
 * MemoryArchitecture Page Tests
 * =============================
 *
 * Tests for the Memory Architecture visualization page.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import MemoryArchitecture from './MemoryArchitecture';

// Mock the memory hooks
vi.mock('@/hooks/api/use-memory', () => ({
  useMemoryStats: vi.fn(),
  useEpisodicMemories: vi.fn(),
}));

import { useMemoryStats, useEpisodicMemories } from '@/hooks/api/use-memory';

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

// Sample memory stats data
const mockMemoryStats = {
  episodic: {
    total_memories: 1250,
    recent_24h: 45,
  },
  procedural: {
    total_procedures: 89,
    average_success_rate: 0.923,
  },
  semantic: {
    total_entities: 3420,
    total_relationships: 8750,
  },
  last_updated: '2026-01-04T10:30:00Z',
};

// Sample episodic memories (matches EpisodicMemoryResponse type)
const mockEpisodicMemories = [
  {
    id: 'mem-001',
    event_type: 'conversation',
    content: 'User asked about Remibrutinib prescription trends in Q4',
    agent_name: 'orchestrator',
    created_at: '2026-01-04T10:00:00Z',
  },
  {
    id: 'mem-002',
    event_type: 'insight',
    content: 'Causal analysis revealed positive correlation between rep visits and NRx',
    agent_name: 'causal_impact',
    created_at: '2026-01-04T09:30:00Z',
  },
  {
    id: 'mem-003',
    event_type: 'action',
    content: 'Gap Analyzer identified underperforming territory: Northeast region',
    agent_name: 'gap_analyzer',
    created_at: '2026-01-04T09:00:00Z',
  },
];

describe('MemoryArchitecture', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    (useMemoryStats as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockMemoryStats,
      isLoading: false,
      error: null,
    });
    (useEpisodicMemories as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockEpisodicMemories,
      isLoading: false,
    });
  });

  it('renders page header with title and description', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('Memory Architecture')).toBeInTheDocument();
    expect(screen.getByText(/E2I Tri-Memory Cognitive System/)).toBeInTheDocument();
  });

  it('displays overall system status badge', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    // Should show healthy status when data loads successfully
    const statusBadges = screen.getAllByText('Healthy');
    expect(statusBadges.length).toBeGreaterThan(0);
  });

  it('renders architecture diagram with all memory types', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('Cognitive Memory Architecture')).toBeInTheDocument();
    // Memory types appear in both diagram and cards, so use getAllByText
    expect(screen.getAllByText('Working Memory').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Episodic Memory').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Semantic Memory').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Procedural Memory').length).toBeGreaterThan(0);
  });

  it('shows memory backend labels in diagram', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('Redis Cache')).toBeInTheDocument();
    expect(screen.getByText('Supabase PostgreSQL')).toBeInTheDocument();
    expect(screen.getByText('FalkorDB Graph')).toBeInTheDocument();
    expect(screen.getByText('Learned Patterns')).toBeInTheDocument();
  });

  it('displays episodic memory statistics', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('Total Memories')).toBeInTheDocument();
    expect(screen.getByText('1,250')).toBeInTheDocument();
    expect(screen.getByText('Recent (24h)')).toBeInTheDocument();
    expect(screen.getByText('45')).toBeInTheDocument();
  });

  it('displays semantic memory statistics', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('Entities')).toBeInTheDocument();
    expect(screen.getByText('3,420')).toBeInTheDocument();
    expect(screen.getByText('Relationships')).toBeInTheDocument();
    expect(screen.getByText('8,750')).toBeInTheDocument();
  });

  it('displays procedural memory statistics', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('Procedures')).toBeInTheDocument();
    expect(screen.getByText('89')).toBeInTheDocument();
    expect(screen.getByText('Success Rate')).toBeInTheDocument();
    expect(screen.getByText('92.3%')).toBeInTheDocument();
  });

  it('displays working memory info', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('Backend')).toBeInTheDocument();
    expect(screen.getByText('Redis')).toBeInTheDocument();
    expect(screen.getByText('TTL')).toBeInTheDocument();
    expect(screen.getByText('24h')).toBeInTheDocument();
  });

  it('renders recent episodic memories section', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('Recent Episodic Memories')).toBeInTheDocument();
    expect(screen.getByText('Last 5 interactions')).toBeInTheDocument();
  });

  it('displays episodic memory list items', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText(/Remibrutinib prescription trends/)).toBeInTheDocument();
    expect(screen.getByText(/Causal analysis revealed positive correlation/)).toBeInTheDocument();
    expect(screen.getByText(/Gap Analyzer identified underperforming territory/)).toBeInTheDocument();
  });

  it('shows memory type badges on memory items', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('conversation')).toBeInTheDocument();
    expect(screen.getByText('insight')).toBeInTheDocument();
    expect(screen.getByText('action')).toBeInTheDocument();
  });

  it('displays agent names for memories', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    // Check for agent labels (the component shows Agent: for each memory)
    const agentLabels = screen.getAllByText('Agent:');
    expect(agentLabels.length).toBe(3);
  });

  it('renders refresh button', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
  });

  it('shows loading state when stats are loading', () => {
    (useMemoryStats as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    });

    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    // Should show loading indicators (...)
    const loadingIndicators = screen.getAllByText('...');
    expect(loadingIndicators.length).toBeGreaterThan(0);
  });

  it('shows loading spinner for episodic memories', () => {
    (useEpisodicMemories as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
    });

    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    const spinner = document.querySelector('.animate-spin');
    expect(spinner).toBeInTheDocument();
  });

  it('shows error state when stats fail to load', () => {
    (useMemoryStats as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('API Error'),
    });

    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('Failed to load memory statistics')).toBeInTheDocument();
    expect(screen.getByText(/Please check the API connection/)).toBeInTheDocument();
  });

  it('shows unknown status when no data is available', () => {
    (useMemoryStats as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: null,
    });

    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    const unknownBadges = screen.getAllByText('Unknown');
    expect(unknownBadges.length).toBeGreaterThan(0);
  });

  it('displays last updated timestamp', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
  });

  it('renders about section with retrieval methods', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('About the Memory System')).toBeInTheDocument();
    expect(screen.getByText('Memory Integration')).toBeInTheDocument();
    expect(screen.getByText('Retrieval Methods')).toBeInTheDocument();
    expect(screen.getByText(/Vector Search/)).toBeInTheDocument();
    expect(screen.getByText(/Keyword Search/)).toBeInTheDocument();
    expect(screen.getByText(/Hybrid Search/)).toBeInTheDocument();
    expect(screen.getByText(/Graph Traversal/)).toBeInTheDocument();
  });

  it('shows empty state when no episodic memories', () => {
    (useEpisodicMemories as ReturnType<typeof vi.fn>).mockReturnValue({
      data: [],
      isLoading: false,
    });

    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText('No recent episodic memories')).toBeInTheDocument();
  });

  it('handles zero stats gracefully', () => {
    (useMemoryStats as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        episodic: { total_memories: 0, recent_24h: 0 },
        procedural: { total_procedures: 0, average_success_rate: 0 },
        semantic: { total_entities: 0, total_relationships: 0 },
        last_updated: '2026-01-04T10:30:00Z',
      },
      isLoading: false,
      error: null,
    });

    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    // Should still render without errors
    expect(screen.getByText('Memory Architecture')).toBeInTheDocument();
    // Check that zeros are displayed
    const zeroValues = screen.getAllByText('0');
    expect(zeroValues.length).toBeGreaterThan(0);
  });

  it('renders all four memory card sections', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    // Count memory cards by their titles
    const workingMemoryCards = screen.getAllByText('Working Memory');
    const episodicMemoryCards = screen.getAllByText('Episodic Memory');
    const semanticMemoryCards = screen.getAllByText('Semantic Memory');
    const proceduralMemoryCards = screen.getAllByText('Procedural Memory');

    // Each memory type should appear at least once (in diagram and/or card)
    expect(workingMemoryCards.length).toBeGreaterThan(0);
    expect(episodicMemoryCards.length).toBeGreaterThan(0);
    expect(semanticMemoryCards.length).toBeGreaterThan(0);
    expect(proceduralMemoryCards.length).toBeGreaterThan(0);
  });

  it('shows descriptions for each memory type in diagram', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText(/Short-term context, active session state/)).toBeInTheDocument();
    expect(screen.getByText(/Historical interactions, past experiences/)).toBeInTheDocument();
    expect(screen.getByText(/Knowledge graph, entity relationships/)).toBeInTheDocument();
    expect(screen.getByText(/Optimized procedures, behavioral patterns/)).toBeInTheDocument();
  });

  it('displays memory system integration description', () => {
    render(<MemoryArchitecture />, { wrapper: createWrapper() });

    expect(screen.getByText(/Memory systems work together/)).toBeInTheDocument();
  });
});
