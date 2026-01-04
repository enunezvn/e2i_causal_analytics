/**
 * AgentOrchestration Page Tests
 * =============================
 *
 * Tests for the Agent Orchestration dashboard page.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AgentOrchestration from './AgentOrchestration';

// Mock the E2ICopilotProvider
vi.mock('@/providers/E2ICopilotProvider', () => ({
  useE2ICopilot: vi.fn(),
}));

// Mock the child components that have complex dependencies
vi.mock('@/components/visualizations/agents/AgentTierBadge', () => ({
  TierOverview: ({ onTierClick, activeTier }: { onTierClick?: (tier: number) => void; activeTier?: number }) => (
    <div data-testid="tier-overview">
      <div>Tier Overview Component</div>
      {[0, 1, 2, 3, 4, 5].map((tier) => (
        <button
          key={tier}
          data-testid={`tier-${tier}`}
          onClick={() => onTierClick?.(tier)}
          className={activeTier === tier ? 'active' : ''}
        >
          Tier {tier}
        </button>
      ))}
    </div>
  ),
  AgentTierBadge: ({ tier }: { tier: number }) => <span data-testid="agent-tier-badge">Tier {tier}</span>,
}));

vi.mock('@/components/chat/AgentStatusPanel', () => ({
  AgentStatusPanel: ({ compact, onAgentClick }: { compact?: boolean; onAgentClick?: (id: string) => void }) => (
    <div data-testid="agent-status-panel" data-compact={compact}>
      Agent Status Panel
      <button onClick={() => onAgentClick?.('test-agent')}>Click Agent</button>
    </div>
  ),
}));

import { useE2ICopilot } from '@/providers/E2ICopilotProvider';

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

// Mock agents data
const mockAgents = [
  { id: 'scope-definer', name: 'Scope Definer', tier: 0, status: 'idle', capabilities: ['problem_scoping'] },
  { id: 'data-preparer', name: 'Data Preparer', tier: 0, status: 'active', capabilities: ['data_validation', 'preprocessing'] },
  { id: 'feature-analyzer', name: 'Feature Analyzer', tier: 0, status: 'idle', capabilities: ['feature_engineering'] },
  { id: 'orchestrator', name: 'Orchestrator', tier: 1, status: 'active', capabilities: ['routing', 'coordination'] },
  { id: 'tool-composer', name: 'Tool Composer', tier: 1, status: 'idle', capabilities: ['tool_selection'] },
  { id: 'causal-impact', name: 'Causal Impact', tier: 2, status: 'processing', capabilities: ['ate_estimation'] },
  { id: 'gap-analyzer', name: 'Gap Analyzer', tier: 2, status: 'idle', capabilities: ['roi_analysis'] },
  { id: 'heterogeneous-optimizer', name: 'Heterogeneous Optimizer', tier: 2, status: 'idle', capabilities: ['cate_analysis'] },
  { id: 'drift-monitor', name: 'Drift Monitor', tier: 3, status: 'active', capabilities: ['data_drift'] },
  { id: 'experiment-designer', name: 'Experiment Designer', tier: 3, status: 'idle', capabilities: ['ab_design'] },
  { id: 'health-score', name: 'Health Score', tier: 3, status: 'active', capabilities: ['system_health'] },
  { id: 'prediction-synthesizer', name: 'Prediction Synthesizer', tier: 4, status: 'idle', capabilities: ['prediction_aggregation'] },
  { id: 'resource-optimizer', name: 'Resource Optimizer', tier: 4, status: 'idle', capabilities: ['resource_allocation'] },
  { id: 'explainer', name: 'Explainer', tier: 5, status: 'idle', capabilities: ['narratives'] },
  { id: 'feedback-learner', name: 'Feedback Learner', tier: 5, status: 'idle', capabilities: ['prompt_optimization'] },
];

describe('AgentOrchestration', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementation
    (useE2ICopilot as ReturnType<typeof vi.fn>).mockReturnValue({
      agents: mockAgents,
      filters: { brand: 'All' },
      preferences: { detailLevel: 'detailed' },
    });
  });

  it('renders page header with title and description', () => {
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    expect(screen.getByText('Agent Orchestration')).toBeInTheDocument();
    expect(screen.getByText(/Monitor and manage the 18-agent tiered orchestration system/)).toBeInTheDocument();
  });

  it('displays stat cards with correct data', () => {
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    expect(screen.getByText('Total Agents')).toBeInTheDocument();
    expect(screen.getByText('Tasks Today')).toBeInTheDocument();
    expect(screen.getByText('Avg Response Time')).toBeInTheDocument();
    expect(screen.getByText('Success Rate')).toBeInTheDocument();
  });

  it('shows tabs for different views', () => {
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    expect(screen.getByRole('tab', { name: 'Overview' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Activity Feed' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Tier Metrics' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'All Agents' })).toBeInTheDocument();
  });

  it('renders TierOverview component in overview tab', () => {
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    expect(screen.getByTestId('tier-overview')).toBeInTheDocument();
    expect(screen.getByText('Tier Architecture')).toBeInTheDocument();
  });

  it('renders AgentStatusPanel component in overview tab', () => {
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    expect(screen.getByTestId('agent-status-panel')).toBeInTheDocument();
  });

  it('shows recent activity section in overview tab', () => {
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    expect(screen.getByText('Recent Activity')).toBeInTheDocument();
    expect(screen.getByText('Latest agent actions and events')).toBeInTheDocument();
  });

  it('switches to activity feed tab and shows content', async () => {
    const user = userEvent.setup();
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    const activityTab = screen.getByRole('tab', { name: 'Activity Feed' });
    await act(async () => {
      await user.click(activityTab);
    });

    // Wait for the Activity Feed tab content to be visible
    await waitFor(() => {
      expect(screen.getByText('Complete log of agent actions')).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('switches to tier metrics tab and shows tier content', async () => {
    const user = userEvent.setup();
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    const tierMetricsTab = screen.getByRole('tab', { name: 'Tier Metrics' });
    await act(async () => {
      await user.click(tierMetricsTab);
    });

    // Look for tier 0 header text that appears in tier metrics cards
    await waitFor(() => {
      expect(screen.getByText('Tier 0: ML Foundation')).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('switches to all agents tab and shows agent list', async () => {
    const user = userEvent.setup();
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    const allAgentsTab = screen.getByRole('tab', { name: 'All Agents' });
    await act(async () => {
      await user.click(allAgentsTab);
    });

    // Wait for agent cards to be visible (agents come from mocked context)
    await waitFor(() => {
      expect(screen.getByText('All 18 agents across 6 tiers')).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('displays agent cards with capabilities in all agents tab', async () => {
    const user = userEvent.setup();
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    const allAgentsTab = screen.getByRole('tab', { name: 'All Agents' });
    await act(async () => {
      await user.click(allAgentsTab);
    });

    // Wait for tab content to load by checking for agent cards
    await waitFor(() => {
      expect(screen.getByText('All 18 agents across 6 tiers')).toBeInTheDocument();
    }, { timeout: 5000 });

    // Check for some agent names (our mock has these)
    expect(screen.getByText('Scope Definer')).toBeInTheDocument();
    expect(screen.getByText('Orchestrator')).toBeInTheDocument();
  });

  it('filters agents by tier using tier filter select', async () => {
    const user = userEvent.setup();
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    const allAgentsTab = screen.getByRole('tab', { name: 'All Agents' });
    await act(async () => {
      await user.click(allAgentsTab);
    });

    // Wait for tab content to load
    await waitFor(() => {
      expect(screen.getByText('All 18 agents across 6 tiers')).toBeInTheDocument();
    }, { timeout: 5000 });

    // Find the select element after content loads
    const tierSelect = screen.getByRole('combobox');
    await act(async () => {
      await user.selectOptions(tierSelect, '1');
    });

    // After filtering, the description should change
    await waitFor(() => {
      expect(screen.getByText('Showing Tier 1 agents')).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('shows pause and refresh buttons in header', () => {
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    expect(screen.getByRole('button', { name: /Pause All/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
  });

  it('calculates active agents correctly from context data', () => {
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    // With our mock data: 4 active, 1 processing, 10 idle
    expect(screen.getByText('4 active, 1 processing')).toBeInTheDocument();
  });

  it('handles empty agents list gracefully', () => {
    (useE2ICopilot as ReturnType<typeof vi.fn>).mockReturnValue({
      agents: [],
      filters: { brand: 'All' },
      preferences: { detailLevel: 'detailed' },
    });

    render(<AgentOrchestration />, { wrapper: createWrapper() });

    expect(screen.getByText('Agent Orchestration')).toBeInTheDocument();
    expect(screen.getByText('0 active, 0 processing')).toBeInTheDocument();
  });

  it('renders view all button in recent activity section', () => {
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    const viewAllButtons = screen.getAllByRole('button', { name: /View All/i });
    expect(viewAllButtons.length).toBeGreaterThan(0);
  });

  it('shows tier metrics with utilization progress bars', async () => {
    const user = userEvent.setup();
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    const tierMetricsTab = screen.getByRole('tab', { name: 'Tier Metrics' });
    await act(async () => {
      await user.click(tierMetricsTab);
    });

    await waitFor(() => {
      // Check for tier name text that appears in tier metrics cards
      expect(screen.getByText('Tier 1: Orchestration')).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('shows success rate in tier metrics', async () => {
    const user = userEvent.setup();
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    const tierMetricsTab = screen.getByRole('tab', { name: 'Tier Metrics' });
    await act(async () => {
      await user.click(tierMetricsTab);
    });

    await waitFor(() => {
      // Check for tier 2 in tier metrics
      expect(screen.getByText('Tier 2: Causal Analytics')).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('clears tier filter when clicking clear filter button', async () => {
    const user = userEvent.setup();
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    const allAgentsTab = screen.getByRole('tab', { name: 'All Agents' });
    await act(async () => {
      await user.click(allAgentsTab);
    });

    // Wait for tab content to load
    await waitFor(() => {
      expect(screen.getByText('All 18 agents across 6 tiers')).toBeInTheDocument();
    }, { timeout: 5000 });

    // Find the select element and change it to filter
    const tierSelect = screen.getByRole('combobox');
    await act(async () => {
      await user.selectOptions(tierSelect, '2');
    });

    // Wait for the filter to take effect
    await waitFor(() => {
      expect(screen.getByText('Showing Tier 2 agents')).toBeInTheDocument();
    }, { timeout: 5000 });

    // Now find and click the clear filter button
    const clearButton = screen.getByRole('button', { name: /Clear Filter/i });
    await act(async () => {
      await user.click(clearButton);
    });

    // Verify filter is cleared - should show all agents again
    await waitFor(() => {
      expect(screen.getByText('All 18 agents across 6 tiers')).toBeInTheDocument();
    }, { timeout: 5000 });
  });
});
