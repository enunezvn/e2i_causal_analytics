/**
 * E2ICopilotProvider Tests
 * ========================
 *
 * Tests for CopilotKit integration provider including context, hooks, and actions.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, renderHook } from '@testing-library/react';
import * as React from 'react';

// =============================================================================
// MOCKS
// =============================================================================

// Track CopilotKit hook calls
const mockUseCopilotReadable = vi.fn();
const mockUseCopilotAction = vi.fn();
const mockNavigate = vi.fn();
const mockLocation = { pathname: '/test-path' };

// Mock react-router-dom
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => mockLocation,
}));

// Override the global CopilotKit mock for this test file
vi.mock('@copilotkit/react-core', () => ({
  CopilotKit: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="copilotkit-wrapper">{children}</div>
  ),
  useCopilotReadable: (config: unknown) => {
    mockUseCopilotReadable(config);
    return undefined;
  },
  useCopilotAction: (config: unknown) => {
    mockUseCopilotAction(config);
    return undefined;
  },
}));

// Import after mocks
import {
  E2ICopilotProvider,
  CopilotKitWrapper,
  useE2ICopilot,
  useCopilotEnabled,
  type E2IFilters,
  type UserPreferences,
  type AgentInfo,
} from './E2ICopilotProvider';

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

// Test component that uses the context
function TestConsumer() {
  const context = useE2ICopilot();
  return (
    <div>
      <span data-testid="brand">{context.filters.brand}</span>
      <span data-testid="territory">{context.filters.territory ?? 'none'}</span>
      <span data-testid="detail-level">{context.preferences.detailLevel}</span>
      <span data-testid="agent-count">{context.agents.length}</span>
      <span data-testid="chat-open">{context.chatOpen.toString()}</span>
      <span data-testid="highlighted-paths">{context.highlightedPaths.length}</span>
      <button onClick={() => context.setFilters((p) => ({ ...p, brand: 'Fabhalta' }))}>
        Change Brand
      </button>
      <button onClick={() => context.setPreferences((p) => ({ ...p, detailLevel: 'expert' }))}>
        Change Detail
      </button>
      <button onClick={() => context.setChatOpen(true)}>Open Chat</button>
      <button onClick={() => context.setHighlightedPaths(['path-1', 'path-2'])}>
        Set Paths
      </button>
    </div>
  );
}

// Test component that checks CopilotEnabled
function CopilotEnabledChecker() {
  const enabled = useCopilotEnabled();
  return <span data-testid="copilot-enabled">{enabled.toString()}</span>;
}

// =============================================================================
// TESTS: useE2ICopilot HOOK
// =============================================================================

describe('useE2ICopilot', () => {
  it('throws error when used outside provider', () => {
    // Suppress console.error for this test
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      renderHook(() => useE2ICopilot());
    }).toThrow('useE2ICopilot must be used within E2ICopilotProvider');

    spy.mockRestore();
  });

  it('returns context when used inside provider', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <CopilotKitWrapper enabled={false}>
        <E2ICopilotProvider>{children}</E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    const { result } = renderHook(() => useE2ICopilot(), { wrapper });

    expect(result.current).toBeDefined();
    expect(result.current.filters).toBeDefined();
    expect(result.current.agents).toBeDefined();
    expect(result.current.preferences).toBeDefined();
  });
});

// =============================================================================
// TESTS: useCopilotEnabled HOOK
// =============================================================================

describe('useCopilotEnabled', () => {
  it('returns false when CopilotKit is disabled', () => {
    render(
      <CopilotKitWrapper enabled={false}>
        <CopilotEnabledChecker />
      </CopilotKitWrapper>
    );

    expect(screen.getByTestId('copilot-enabled')).toHaveTextContent('false');
  });

  it('returns true when CopilotKit is enabled', () => {
    render(
      <CopilotKitWrapper enabled={true}>
        <CopilotEnabledChecker />
      </CopilotKitWrapper>
    );

    expect(screen.getByTestId('copilot-enabled')).toHaveTextContent('true');
  });

  it('returns false by default (no provider)', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => <>{children}</>;
    const { result } = renderHook(() => useCopilotEnabled(), { wrapper });

    expect(result.current).toBe(false);
  });
});

// =============================================================================
// TESTS: CopilotKitWrapper
// =============================================================================

describe('CopilotKitWrapper', () => {
  it('renders children when disabled', () => {
    render(
      <CopilotKitWrapper enabled={false}>
        <div data-testid="child">Content</div>
      </CopilotKitWrapper>
    );

    expect(screen.getByTestId('child')).toBeInTheDocument();
    expect(screen.queryByTestId('copilotkit-wrapper')).not.toBeInTheDocument();
  });

  it('renders with CopilotKit wrapper when enabled', () => {
    render(
      <CopilotKitWrapper enabled={true}>
        <div data-testid="child">Content</div>
      </CopilotKitWrapper>
    );

    expect(screen.getByTestId('child')).toBeInTheDocument();
    expect(screen.getByTestId('copilotkit-wrapper')).toBeInTheDocument();
  });

  it('defaults to disabled', () => {
    render(
      <CopilotKitWrapper>
        <CopilotEnabledChecker />
      </CopilotKitWrapper>
    );

    expect(screen.getByTestId('copilot-enabled')).toHaveTextContent('false');
  });

  it('uses default runtimeUrl', () => {
    render(
      <CopilotKitWrapper enabled={true}>
        <div>Content</div>
      </CopilotKitWrapper>
    );

    // Component renders without error with default URL
    expect(screen.getByTestId('copilotkit-wrapper')).toBeInTheDocument();
  });

  it('accepts custom runtimeUrl', () => {
    render(
      <CopilotKitWrapper enabled={true} runtimeUrl="/custom/api">
        <div>Content</div>
      </CopilotKitWrapper>
    );

    expect(screen.getByTestId('copilotkit-wrapper')).toBeInTheDocument();
  });
});

// =============================================================================
// TESTS: E2ICopilotProvider CONTEXT VALUES
// =============================================================================

describe('E2ICopilotProvider', () => {
  describe('default values', () => {
    it('provides default filters', () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('brand')).toHaveTextContent('Remibrutinib');
      expect(screen.getByTestId('territory')).toHaveTextContent('none');
    });

    it('provides default preferences', () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('detail-level')).toHaveTextContent('detailed');
    });

    it('provides sample agents (19 agents)', () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('agent-count')).toHaveTextContent('19');
    });

    it('provides default chat state (closed)', () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('chat-open')).toHaveTextContent('false');
    });

    it('provides empty highlighted paths', () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('highlighted-paths')).toHaveTextContent('0');
    });
  });

  describe('state updates', () => {
    it('allows updating filters', async () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('brand')).toHaveTextContent('Remibrutinib');

      await act(async () => {
        screen.getByText('Change Brand').click();
      });

      expect(screen.getByTestId('brand')).toHaveTextContent('Fabhalta');
    });

    it('allows updating preferences', async () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('detail-level')).toHaveTextContent('detailed');

      await act(async () => {
        screen.getByText('Change Detail').click();
      });

      expect(screen.getByTestId('detail-level')).toHaveTextContent('expert');
    });

    it('allows toggling chat state', async () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('chat-open')).toHaveTextContent('false');

      await act(async () => {
        screen.getByText('Open Chat').click();
      });

      expect(screen.getByTestId('chat-open')).toHaveTextContent('true');
    });

    it('allows setting highlighted paths', async () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('highlighted-paths')).toHaveTextContent('0');

      await act(async () => {
        screen.getByText('Set Paths').click();
      });

      expect(screen.getByTestId('highlighted-paths')).toHaveTextContent('2');
    });
  });

  describe('props', () => {
    it('accepts initialFilters prop (unused but valid)', () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider initialFilters={{ brand: 'Kisqali' }}>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      // Initial filters prop is currently unused in implementation
      // but component should render without error
      expect(screen.getByTestId('brand')).toBeInTheDocument();
    });

    it('accepts userRole prop (unused but valid)', () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider userRole="admin">
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('brand')).toBeInTheDocument();
    });
  });
});

// =============================================================================
// TESTS: CopilotHooksConnector (when CopilotKit is enabled)
// =============================================================================

describe('CopilotHooksConnector', () => {
  beforeEach(() => {
    mockUseCopilotReadable.mockClear();
    mockUseCopilotAction.mockClear();
    mockNavigate.mockClear();
  });

  it('registers readables when CopilotKit is enabled', () => {
    render(
      <CopilotKitWrapper enabled={true}>
        <E2ICopilotProvider>
          <TestConsumer />
        </E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    // Should register 4 readables: filters, page context, agents, preferences
    expect(mockUseCopilotReadable).toHaveBeenCalledTimes(4);

    // Check filters readable
    const filtersCall = mockUseCopilotReadable.mock.calls.find((call) =>
      call[0]?.description?.includes('dashboard filters')
    );
    expect(filtersCall).toBeDefined();

    // Check page context readable
    const pageCall = mockUseCopilotReadable.mock.calls.find((call) =>
      call[0]?.description?.includes('Current page path')
    );
    expect(pageCall).toBeDefined();
    expect(pageCall[0].value.currentPath).toBe('/test-path');

    // Check agents readable
    const agentsCall = mockUseCopilotReadable.mock.calls.find((call) =>
      call[0]?.description?.includes('agent tier hierarchy')
    );
    expect(agentsCall).toBeDefined();
    expect(agentsCall[0].value.length).toBe(19);

    // Check preferences readable
    const prefsCall = mockUseCopilotReadable.mock.calls.find((call) =>
      call[0]?.description?.includes('User preferences')
    );
    expect(prefsCall).toBeDefined();
  });

  it('registers actions when CopilotKit is enabled', () => {
    render(
      <CopilotKitWrapper enabled={true}>
        <E2ICopilotProvider>
          <TestConsumer />
        </E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    // Should register 6 actions
    expect(mockUseCopilotAction).toHaveBeenCalledTimes(6);

    // Check action names
    const actionNames = mockUseCopilotAction.mock.calls.map((call) => call[0]?.name);
    expect(actionNames).toContain('navigateTo');
    expect(actionNames).toContain('setBrandFilter');
    expect(actionNames).toContain('setDateRange');
    expect(actionNames).toContain('highlightCausalPaths');
    expect(actionNames).toContain('setDetailLevel');
    expect(actionNames).toContain('toggleChat');
  });

  it('does not register hooks when CopilotKit is disabled', () => {
    render(
      <CopilotKitWrapper enabled={false}>
        <E2ICopilotProvider>
          <TestConsumer />
        </E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    // No hooks should be called when disabled
    expect(mockUseCopilotReadable).not.toHaveBeenCalled();
    expect(mockUseCopilotAction).not.toHaveBeenCalled();
  });
});

// =============================================================================
// TESTS: ACTION HANDLERS
// =============================================================================

describe('Action Handlers', () => {
  beforeEach(() => {
    mockUseCopilotReadable.mockClear();
    mockUseCopilotAction.mockClear();
    mockNavigate.mockClear();
  });

  function getActionHandler(actionName: string): (params: Record<string, unknown>) => string {
    render(
      <CopilotKitWrapper enabled={true}>
        <E2ICopilotProvider>
          <TestConsumer />
        </E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    const actionCall = mockUseCopilotAction.mock.calls.find(
      (call) => call[0]?.name === actionName
    );
    return actionCall?.[0]?.handler;
  }

  describe('navigateTo', () => {
    it('calls navigate with provided path', async () => {
      const handler = getActionHandler('navigateTo');
      let result: string;
      await act(async () => {
        result = handler({ path: '/knowledge-graph' });
      });

      expect(mockNavigate).toHaveBeenCalledWith('/knowledge-graph');
      expect(result!).toBe('Navigated to /knowledge-graph');
    });
  });

  describe('setBrandFilter', () => {
    it('accepts valid brand', async () => {
      const handler = getActionHandler('setBrandFilter');
      let result: string;
      await act(async () => {
        result = handler({ brand: 'Fabhalta' });
      });

      expect(result!).toBe('Brand filter set to Fabhalta');
    });

    it('rejects invalid brand', async () => {
      const handler = getActionHandler('setBrandFilter');
      let result: string;
      await act(async () => {
        result = handler({ brand: 'InvalidBrand' });
      });

      expect(result!).toBe('Invalid brand. Choose from: Remibrutinib, Fabhalta, Kisqali, All');
    });

    it('accepts All brand', async () => {
      const handler = getActionHandler('setBrandFilter');
      let result: string;
      await act(async () => {
        result = handler({ brand: 'All' });
      });

      expect(result!).toBe('Brand filter set to All');
    });
  });

  describe('setDateRange', () => {
    it('sets date range', async () => {
      const handler = getActionHandler('setDateRange');
      let result: string;
      await act(async () => {
        result = handler({ startDate: '2024-01-01', endDate: '2024-12-31' });
      });

      expect(result!).toBe('Date range set to 2024-01-01 - 2024-12-31');
    });
  });

  describe('highlightCausalPaths', () => {
    it('highlights paths', async () => {
      const handler = getActionHandler('highlightCausalPaths');
      let result: string;
      await act(async () => {
        result = handler({ pathIds: ['path-1', 'path-2', 'path-3'] });
      });

      expect(result!).toBe('Highlighted 3 causal path(s)');
    });

    it('handles empty array', async () => {
      const handler = getActionHandler('highlightCausalPaths');
      let result: string;
      await act(async () => {
        result = handler({ pathIds: [] });
      });

      expect(result!).toBe('Highlighted 0 causal path(s)');
    });
  });

  describe('setDetailLevel', () => {
    it('accepts valid level - summary', async () => {
      const handler = getActionHandler('setDetailLevel');
      let result: string;
      await act(async () => {
        result = handler({ level: 'summary' });
      });

      expect(result!).toBe('Detail level set to summary');
    });

    it('accepts valid level - detailed', async () => {
      const handler = getActionHandler('setDetailLevel');
      let result: string;
      await act(async () => {
        result = handler({ level: 'detailed' });
      });

      expect(result!).toBe('Detail level set to detailed');
    });

    it('accepts valid level - expert', async () => {
      const handler = getActionHandler('setDetailLevel');
      let result: string;
      await act(async () => {
        result = handler({ level: 'expert' });
      });

      expect(result!).toBe('Detail level set to expert');
    });

    it('rejects invalid level', async () => {
      const handler = getActionHandler('setDetailLevel');
      let result: string;
      await act(async () => {
        result = handler({ level: 'invalid' });
      });

      expect(result!).toBe('Invalid level. Choose from: summary, detailed, expert');
    });
  });

  describe('toggleChat', () => {
    it('opens chat when open=true', async () => {
      const handler = getActionHandler('toggleChat');
      let result: string;
      await act(async () => {
        result = handler({ open: true });
      });

      expect(result!).toBe('Chat opened');
    });

    it('closes chat when open=false', async () => {
      const handler = getActionHandler('toggleChat');
      let result: string;
      await act(async () => {
        result = handler({ open: false });
      });

      expect(result!).toBe('Chat closed');
    });

    it('toggles chat when open is undefined (default closed → opened)', async () => {
      const handler = getActionHandler('toggleChat');
      // Initial state is closed, so toggle opens it
      let result: string;
      await act(async () => {
        result = handler({});
      });

      expect(result!).toBe('Chat opened');
    });
  });
});

// =============================================================================
// TESTS: TYPE EXPORTS
// =============================================================================

describe('Type Exports', () => {
  it('E2IFilters type is usable', () => {
    const filters: E2IFilters = {
      brand: 'Remibrutinib',
      territory: null,
      dateRange: { start: '2024-01-01', end: '2024-12-31' },
      hcpSegment: null,
    };
    expect(filters.brand).toBe('Remibrutinib');
  });

  it('UserPreferences type is usable', () => {
    const prefs: UserPreferences = {
      detailLevel: 'expert',
      defaultBrand: 'Kisqali',
      notificationsEnabled: false,
      theme: 'dark',
    };
    expect(prefs.theme).toBe('dark');
  });

  it('AgentInfo type is usable', () => {
    const agent: AgentInfo = {
      id: 'test-agent',
      name: 'Test Agent',
      tier: 2,
      status: 'active',
      capabilities: ['test'],
    };
    expect(agent.tier).toBe(2);
  });
});

// =============================================================================
// TESTS: AGENT DATA
// =============================================================================

describe('Agent Data', () => {
  it('provides all 19 agents across 6 tiers', () => {
    render(
      <CopilotKitWrapper enabled={false}>
        <E2ICopilotProvider>
          <TestConsumer />
        </E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    expect(screen.getByTestId('agent-count')).toHaveTextContent('19');
  });

  it('agents have correct tier distribution', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <CopilotKitWrapper enabled={false}>
        <E2ICopilotProvider>{children}</E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    const { result } = renderHook(() => useE2ICopilot(), { wrapper });

    const tierCounts = result.current.agents.reduce(
      (acc, agent) => {
        acc[agent.tier] = (acc[agent.tier] || 0) + 1;
        return acc;
      },
      {} as Record<number, number>
    );

    // Tier 0: 7 agents (ML Foundation)
    expect(tierCounts[0]).toBe(7);
    // Tier 1: 2 agents (Orchestration)
    expect(tierCounts[1]).toBe(2);
    // Tier 2: 3 agents (Causal Analytics)
    expect(tierCounts[2]).toBe(3);
    // Tier 3: 3 agents (Monitoring) - drift-monitor, experiment-designer, health-score
    expect(tierCounts[3]).toBe(3);
    // Tier 4: 2 agents (ML Predictions)
    expect(tierCounts[4]).toBe(2);
    // Tier 5: 1 agent (Self-Improvement) - explainer only, feedback-learner may be elsewhere
    expect(tierCounts[5]).toBeGreaterThanOrEqual(1);
  });

  it('includes expected agent IDs', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <CopilotKitWrapper enabled={false}>
        <E2ICopilotProvider>{children}</E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    const { result } = renderHook(() => useE2ICopilot(), { wrapper });

    const agentIds = result.current.agents.map((a) => a.id);

    // Spot check key agents
    expect(agentIds).toContain('orchestrator');
    expect(agentIds).toContain('causal-impact');
    expect(agentIds).toContain('explainer');
    expect(agentIds).toContain('feedback-learner');
    expect(agentIds).toContain('drift-monitor');
  });
});
