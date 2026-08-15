/**
 * E2ICopilotProvider Tests
 * ========================
 *
 * Tests for CopilotKit integration provider including context, hooks, and actions.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
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

// Mock the KPI api so renderKpiTrend handler tests can assert what it fetches.
// getKPIValue/batchCalculateKPIs back the renderChart action's routing to
// point-in-time KPIs and multi-KPI comparisons.
const mockGetKPIHistory = vi.fn();
const mockGetKPIHistorySegmented = vi.fn();
const mockGetKPIValue = vi.fn();
const mockBatchCalculateKPIs = vi.fn();
vi.mock('@/api/kpi', () => ({
  getKPIHistory: (...args: unknown[]) => mockGetKPIHistory(...args),
  getKPIHistorySegmented: (...args: unknown[]) => mockGetKPIHistorySegmented(...args),
  getKPIValue: (...args: unknown[]) => mockGetKPIValue(...args),
  batchCalculateKPIs: (...args: unknown[]) => mockBatchCalculateKPIs(...args),
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

    it('provides sample agents (22 agents)', () => {
      render(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <TestConsumer />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );

      expect(screen.getByTestId('agent-count')).toHaveTextContent('22');
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
          <E2ICopilotProvider userRole="analyst">
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
    expect(pageCall![0].value.currentPath).toBe('/test-path');

    // Check agents readable
    const agentsCall = mockUseCopilotReadable.mock.calls.find((call) =>
      call[0]?.description?.includes('agent tier hierarchy')
    );
    expect(agentsCall).toBeDefined();
    // #1638: 22, not 21 — this readable's value IS the roster the model
    // reads, and its own description one file over already says 22.
    // Missed in the first pass: the count was aligned everywhere it was
    // WRITTEN and not where it was ASSERTED.
    expect(agentsCall![0].value.length).toBe(22);

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

    // Should register 8 actions
    expect(mockUseCopilotAction).toHaveBeenCalledTimes(8);

    // Check action names
    const actionNames = mockUseCopilotAction.mock.calls.map((call) => call[0]?.name);
    expect(actionNames).toContain('navigateTo');
    expect(actionNames).toContain('setBrandFilter');
    expect(actionNames).toContain('setDateRange');
    expect(actionNames).toContain('highlightCausalPaths');
    expect(actionNames).toContain('setDetailLevel');
    expect(actionNames).toContain('toggleChat');
    expect(actionNames).toContain('renderKpiTrend');
    // Flint-compiled charts for the rest of the KPI registry.
    expect(actionNames).toContain('renderChart');
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

  function getActionRender(
    actionName: string
  ): (state: Record<string, unknown>) => React.ReactElement {
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
    return actionCall?.[0]?.render;
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

  describe('renderKpiTrend', () => {
    // Why: the substrate (kpi_history via /api/kpis/{id}/history) keys on
    // registry codes (NRx = WS3-BI-006), but the model passes the friendly
    // ids the action description teaches. Without the alias hop, "plot NRX
    // trends" fetched the nonexistent id "nrx" and charted honest-empty
    // despite 35 stored monthly points.
    beforeEach(() => {
      mockGetKPIHistory.mockClear();
      mockGetKPIHistorySegmented.mockClear();
      mockGetKPIHistory.mockResolvedValue({
        kpi_id: 'WS3-BI-006',
        brand: '',
        region: '',
        count: 0,
        points: [],
      });
    });

    it('resolves friendly kpiId aliases to registry codes', async () => {
      const handler = getActionHandler('renderKpiTrend') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpiId: 'nrx' });
      });

      expect(mockGetKPIHistory).toHaveBeenCalledWith('WS3-BI-006', undefined, undefined);
    });

    it('passes a canonicalized brand for per-brand KPIs', async () => {
      const handler = getActionHandler('renderKpiTrend') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpiId: 'nbrx', brand: 'remibrutinib' });
      });

      expect(mockGetKPIHistory).toHaveBeenCalledWith('WS3-BI-007', 'Remibrutinib', undefined);
    });

    it('declares the brand parameter so the model can pass it', () => {
      getActionHandler('renderKpiTrend');
      const actionCall = mockUseCopilotAction.mock.calls.find(
        (call) => call[0]?.name === 'renderKpiTrend'
      );
      const paramNames = (
        actionCall![0].parameters as Array<{ name: string }>
      ).map((p) => p.name);
      expect(paramNames).toContain('brand');
    });

    it('declares the segment/LOT parameters so the model can pass them', () => {
      getActionHandler('renderKpiTrend');
      const actionCall = mockUseCopilotAction.mock.calls.find(
        (call) => call[0]?.name === 'renderKpiTrend'
      );
      const paramNames = (
        actionCall![0].parameters as Array<{ name: string }>
      ).map((p) => p.name);
      expect(paramNames).toEqual(
        expect.arrayContaining(['compareBy', 'segment', 'therapyLine'])
      );
    });

    it('routes compareBy to ONE segmented fetch (all tiers), not per-tier calls', async () => {
      mockGetKPIHistorySegmented.mockResolvedValue({ series: [] });
      const handler = getActionHandler('renderKpiTrend') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpiId: 'trx', brand: 'remibrutinib', compareBy: 'severity' });
      });

      expect(mockGetKPIHistorySegmented).toHaveBeenCalledTimes(1);
      expect(mockGetKPIHistorySegmented).toHaveBeenCalledWith(
        'WS3-BI-005',
        'segment',
        'Remibrutinib',
        undefined
      );
      expect(mockGetKPIHistory).not.toHaveBeenCalled();
    });

    it('routes a single severity tier through the segmented fetch with a value', async () => {
      mockGetKPIHistorySegmented.mockResolvedValue({ series: [] });
      const handler = getActionHandler('renderKpiTrend') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpiId: 'trx', segment: 'high' });
      });

      expect(mockGetKPIHistorySegmented).toHaveBeenCalledWith(
        'WS3-BI-005',
        'segment',
        undefined,
        'high_severity'
      );
    });

    it('routes a single line of therapy through the segmented fetch', async () => {
      mockGetKPIHistorySegmented.mockResolvedValue({ series: [] });
      const handler = getActionHandler('renderKpiTrend') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpiId: 'nrx', therapyLine: 'LOT 2' });
      });

      expect(mockGetKPIHistorySegmented).toHaveBeenCalledWith(
        'WS3-BI-006',
        'therapy_line',
        undefined,
        '2'
      );
    });

    it('keeps the plain (unsegmented) path when no axis params are passed', async () => {
      mockGetKPIHistorySegmented.mockClear();
      const handler = getActionHandler('renderKpiTrend') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpiId: 'trx', brand: 'kisqali' });
      });

      expect(mockGetKPIHistory).toHaveBeenCalledWith('WS3-BI-005', 'Kisqali', undefined);
      expect(mockGetKPIHistorySegmented).not.toHaveBeenCalled();
    });

    it('threads a region scope through the plain history fetch (#1536)', async () => {
      const handler = getActionHandler('renderKpiTrend') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpiId: 'trx', brand: 'kisqali', region: 'Northeast' });
      });

      expect(mockGetKPIHistory).toHaveBeenCalledWith('WS3-BI-005', 'Kisqali', 'northeast');
    });

    it('resolves a region synonym via the platform vocabulary (#1538)', async () => {
      const handler = getActionHandler('renderKpiTrend') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpiId: 'trx', brand: 'kisqali', region: 'North East' });
      });

      expect(mockGetKPIHistory).toHaveBeenCalledWith('WS3-BI-005', 'Kisqali', 'northeast');
    });

    it('fetches nothing for an unmappable region (#1538)', async () => {
      mockGetKPIHistory.mockClear();
      const handler = getActionHandler('renderKpiTrend') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpiId: 'trx', region: 'EMEA' });
      });

      // The chart component renders the honest empty state; the fetch must
      // not run — a junk region can never match a row.
      expect(mockGetKPIHistory).not.toHaveBeenCalled();
    });

    it('renders a clarify QUESTION for an ambiguous region, not a dead end (#1565)', () => {
      // "East Coast" spans the northeast AND south census regions; the render
      // must ask which one the user means (mirror of the backend tool's
      // clarify hint), naming the four census regions.
      const renderFn = getActionRender('renderKpiTrend');
      const el = renderFn({
        status: 'complete',
        args: { kpiId: 'trx', region: 'East Coast' },
        result: null,
      });
      const { container } = render(el);
      expect(container.textContent).toMatch(/East Coast/);
      expect(container.textContent).toMatch(/northeast[\s\S]*south[\s\S]*midwest[\s\S]*west/i);
      expect(container.textContent).toMatch(/\?/);
    });

    it('fetches nothing when region is combined with a segment axis — segments are global-only (#1536)', async () => {
      mockGetKPIHistory.mockClear();
      mockGetKPIHistorySegmented.mockClear();
      const handler = getActionHandler('renderKpiTrend') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      let out: unknown = 'sentinel';
      await act(async () => {
        out = await handler({ kpiId: 'trx', segment: 'high', region: 'northeast' });
      });

      expect(out).toBeNull();
      expect(mockGetKPIHistory).not.toHaveBeenCalled();
      expect(mockGetKPIHistorySegmented).not.toHaveBeenCalled();
    });
  });

  describe('renderChart', () => {
    // Why: renderKpiTrend only ever drew a line off kpi_history, so the 38
    // registry KPIs without a materialized series were unreachable from the
    // chat — asking to plot ROC-AUC produced an empty frame despite the value
    // being one call away. renderChart routes each KPI to an endpoint that can
    // serve it and compiles the result with flint-chart.
    beforeEach(() => {
      mockGetKPIHistory.mockClear();
      mockGetKPIValue.mockClear();
      mockBatchCalculateKPIs.mockClear();
      mockGetKPIHistory.mockResolvedValue({
        kpi_id: '',
        brand: '',
        region: '',
        count: 0,
        points: [],
      });
      mockGetKPIValue.mockResolvedValue({
        kpi_id: 'WS1-MP-001',
        value: 0.87,
        status: 'good',
        calculated_at: '2026-07-30T00:00:00Z',
        cached: false,
        metadata: {},
      });
    });

    it('falls back to the current value for a KPI with no series', async () => {
      const handler = getActionHandler('renderChart') as unknown as (
        p: Record<string, unknown>
      ) => Promise<{ rows: unknown[]; chartType: string; emptyReason?: string }>;
      let result!: Awaited<ReturnType<typeof handler>>;
      await act(async () => {
        result = await handler({ kpis: ['roc_auc'] });
      });

      expect(mockGetKPIHistory).toHaveBeenCalledWith('WS1-MP-001', undefined, undefined);
      // Third arg = region (#1538), explicitly undefined when none was asked.
      expect(mockGetKPIValue).toHaveBeenCalledWith('WS1-MP-001', undefined, undefined);
      expect(result.emptyReason).toBeUndefined();
      expect(result.chartType).toBe('KPI Card');
    });

    it('resolves a causal-metric code the old alias regex could not', async () => {
      const handler = getActionHandler('renderChart') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpis: ['cm-001'] });
      });

      expect(mockGetKPIHistory).toHaveBeenCalledWith('CM-001', undefined, undefined);
    });

    it('compares several KPIs through the batch endpoint', async () => {
      mockBatchCalculateKPIs.mockResolvedValue({
        results: [
          { kpi_id: 'WS1-MP-001', value: 0.87, status: 'good', calculated_at: '', cached: false, metadata: {} },
          { kpi_id: 'WS1-MP-002', value: 0.64, status: 'good', calculated_at: '', cached: false, metadata: {} },
        ],
        calculated_at: '',
        total_kpis: 2,
      });
      const handler = getActionHandler('renderChart') as unknown as (
        p: Record<string, unknown>
      ) => Promise<{ rows: unknown[]; chartType: string }>;
      let result!: Awaited<ReturnType<typeof handler>>;
      await act(async () => {
        result = await handler({ kpis: ['roc_auc', 'pr_auc'] });
      });

      expect(mockBatchCalculateKPIs).toHaveBeenCalled();
      expect(result.chartType).toBe('Bar Chart');
      expect(result.rows).toHaveLength(2);
    });

    it('routes a single severity tier, the gap that kept renderKpiTrend around', async () => {
      // renderChart could compare all tiers but not chart one, so a
      // single-tier request had no home here until segment/therapyLine landed.
      mockGetKPIHistorySegmented.mockResolvedValue({ series: [] });
      const handler = getActionHandler('renderChart') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpis: ['trx'], segment: 'high' });
      });

      expect(mockGetKPIHistorySegmented).toHaveBeenCalledWith(
        'WS3-BI-005',
        'segment',
        undefined,
        'high_severity'
      );
    });

    it('routes a single line of therapy', async () => {
      mockGetKPIHistorySegmented.mockResolvedValue({ series: [] });
      const handler = getActionHandler('renderChart') as unknown as (
        p: Record<string, unknown>
      ) => Promise<unknown>;
      await act(async () => {
        await handler({ kpis: ['nrx'], therapyLine: 'LOT 2' });
      });

      expect(mockGetKPIHistorySegmented).toHaveBeenCalledWith(
        'WS3-BI-006',
        'therapy_line',
        undefined,
        '2'
      );
    });

    it('ignores a chart type this build does not support', async () => {
      // A model typo must not fail the turn; the routed default applies.
      const handler = getActionHandler('renderChart') as unknown as (
        p: Record<string, unknown>
      ) => Promise<{ chartType: string }>;
      let result!: Awaited<ReturnType<typeof handler>>;
      await act(async () => {
        result = await handler({ kpis: ['roc_auc'], chartType: 'Forest Plot' });
      });

      expect(result.chartType).toBe('KPI Card');
    });

    it('names the other action, in both directions', () => {
      // The defect this pins: the chat system prompt documented ONLY
      // renderKpiTrend, and renderChart's description deferred to it while
      // renderKpiTrend's said nothing back. Two actions with overlapping
      // capability and a one-way reference steer the model to the narrower
      // one. Each description must state where the boundary is.
      getActionHandler('renderChart');
      const byName = (name: string) =>
        mockUseCopilotAction.mock.calls.find((c) => c[0]?.name === name)![0]
          .description as string;

      expect(byName('renderChart')).toContain('renderKpiTrend');
      expect(byName('renderKpiTrend')).toContain('renderChart');
    });

    it('declares kpis and chartType so the model can pass them', () => {
      getActionHandler('renderChart');
      const actionCall = mockUseCopilotAction.mock.calls.find(
        (call) => call[0]?.name === 'renderChart'
      );
      const paramNames = (
        actionCall![0].parameters as Array<{ name: string }>
      ).map((p) => p.name);
      expect(paramNames).toEqual([
        'kpis',
        'chartType',
        'brand',
        'region',
        'compareBy',
        'segment',
        'therapyLine',
        'title',
      ]);
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
  it('provides all 22 agents across 6 tiers', () => {
    render(
      <CopilotKitWrapper enabled={false}>
        <E2ICopilotProvider>
          <TestConsumer />
        </E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    expect(screen.getByTestId('agent-count')).toHaveTextContent('22');
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

    // Tier 0: 9 agents (ML Foundation) — cohort-profiler added in #1638
    expect(tierCounts[0]).toBe(9);
    // Tier 1: 2 agents (Orchestration)
    expect(tierCounts[1]).toBe(2);
    // Tier 2: 3 agents (Causal Analytics)
    expect(tierCounts[2]).toBe(3);
    // Tier 3: 4 agents (Monitoring) - drift-monitor, experiment-designer, experiment-monitor, health-score
    expect(tierCounts[3]).toBe(4);
    // Tier 4: 2 agents (ML Predictions)
    expect(tierCounts[4]).toBe(2);
    // Tier 5: 2 agents (Self-Improvement) - explainer, feedback-learner. Was a
    // >= 1 assertion hedged on "feedback-learner may be elsewhere"; it is right
    // here, and the registry pin (#1638) makes the exact number checkable.
    expect(tierCounts[5]).toBe(2);
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
