// ============================================================================
// E2I Causal Analytics - CopilotKit Provider
// Main integration with bidirectional state sync
// ============================================================================

'use client';

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  ReactNode,
} from 'react';
import { CopilotKit } from '@copilotkit/react-core';
import {
  useCopilotReadable,
  useCopilotAction,
} from '@copilotkit/react-core';

// -----------------------------------------------------------------------------
// Internal Types (not exported - used within provider)
// -----------------------------------------------------------------------------

/** Filter context for dashboard */
interface FilterContext {
  brand?: 'Remibrutinib' | 'Fabhalta' | 'Kisqali';
  region?: 'northeast' | 'south' | 'midwest' | 'west';
  timeRange?: { start: string; end: string };
  segment?: string;
}

/** Dashboard context for state management */
interface DashboardContext {
  filters: FilterContext;
  activeTab: string;
  selectedKPIs: string[];
  visibleCharts: string[];
  highlightedElements: string[];
  userRole: 'executive' | 'analyst' | 'data_scientist';
}

/** Validation summary from causal analysis */
interface ValidationSummary {
  estimateId: string;
  gateDecision: 'proceed' | 'review' | 'block';
  overallConfidence: number;
  testsRun: number;
  testsPassed: number;
  testsFailed: number;
  testsWarning: number;
  results: unknown[];
  timestamp: string;
}

/** Agent state for tracking agent activity */
interface AgentState {
  id: string;
  status: 'idle' | 'thinking' | 'computing' | 'complete' | 'error';
  currentTask?: string;
}

/** Agent registry entry */
interface AgentRegistryEntry {
  id: string;
  name: string;
  tier: number;
  type: string;
  actions: string[];
  icon?: string;
}

/** Registry of available agents */
const AGENT_REGISTRY: Record<string, AgentRegistryEntry> = {
  orchestrator: {
    id: 'orchestrator',
    name: 'Orchestrator',
    tier: 1,
    type: 'coordinator',
    actions: ['route_query', 'coordinate_agents'],
    icon: '🎯',
  },
  causal_impact: {
    id: 'causal_impact',
    name: 'Causal Impact',
    tier: 2,
    type: 'analytical',
    actions: ['estimate_effect', 'run_refutation'],
    icon: '🔗',
  },
  gap_analyzer: {
    id: 'gap_analyzer',
    name: 'Gap Analyzer',
    tier: 2,
    type: 'analytical',
    actions: ['identify_gaps', 'calculate_roi'],
    icon: '📊',
  },
  heterogeneous_optimizer: {
    id: 'heterogeneous_optimizer',
    name: 'Heterogeneous Optimizer',
    tier: 2,
    type: 'analytical',
    actions: ['analyze_cate', 'segment_rank'],
    icon: '🎯',
  },
  drift_monitor: {
    id: 'drift_monitor',
    name: 'Drift Monitor',
    tier: 3,
    type: 'monitoring',
    actions: ['detect_drift', 'compare_distributions'],
    icon: '📈',
  },
  experiment_designer: {
    id: 'experiment_designer',
    name: 'Experiment Designer',
    tier: 3,
    type: 'planning',
    actions: ['design_experiment', 'power_analysis'],
    icon: '🧪',
  },
  explainer: {
    id: 'explainer',
    name: 'Explainer',
    tier: 5,
    type: 'communication',
    actions: ['generate_explanation', 'visualize_insight'],
    icon: '💡',
  },
  feedback_learner: {
    id: 'feedback_learner',
    name: 'Feedback Learner',
    tier: 5,
    type: 'learning',
    actions: ['process_feedback', 'update_model'],
    icon: '🎓',
  },
};

/** Action type for copilot actions */
type E2ICopilotAction =
  | { type: 'highlightCausalPath'; treatment: string; outcome: string; effect: number; mediators?: string[] }
  | { type: 'showValidationResults'; validation: ValidationSummary }
  | { type: 'highlightChartElement'; chartId: string; elementId: string; annotation?: string }
  | { type: 'showGapDetails'; gapId: string; kpi: string; expected: number; actual: number; rootCauses: string[] }
  | { type: 'generateReport'; format: 'pdf' | 'pptx' | 'docx'; sections: string[] };

// -----------------------------------------------------------------------------
// Exported Types
// -----------------------------------------------------------------------------

/** E2I dashboard filter state */
export interface E2IFilters {
  brand: 'Remibrutinib' | 'Fabhalta' | 'Kisqali';
  territory: string | null;
  dateRange: {
    start: string;
    end: string;
  };
  hcpSegment: string | null;
}

/** User preference settings */
export interface UserPreferences {
  detailLevel: 'summary' | 'detailed' | 'expert';
  defaultBrand: E2IFilters['brand'];
  notificationsEnabled: boolean;
  theme: 'light' | 'dark' | 'system';
}

/** Agent information for display */
export interface AgentInfo {
  id: string;
  name: string;
  tier: number;
  status: 'idle' | 'active' | 'processing' | 'complete' | 'error';
  capabilities: string[];
  currentTask?: string;
}

/** Agent state for CoAgent rendering (LangGraph state sync) */
export interface E2IAgentState {
  agent_status: 'idle' | 'processing' | 'waiting' | 'complete' | 'error';
  progress_percent: number;
  progress_steps: string[];
  tools_executing: string[];
  error_message?: string;
  current_node?: string;
}

/** Props for E2ICopilotProvider */
export interface E2ICopilotProviderProps {
  children: ReactNode;
  apiKey?: string;
  runtimeUrl?: string;
  initialFilters?: FilterContext;
  userRole?: DashboardContext['userRole'];
}

/** Props for CopilotKitWrapper */
export interface CopilotKitWrapperProps {
  children: ReactNode;
  runtimeUrl?: string;
  apiKey?: string;
  /** When false, renders children without CopilotKit wrapper */
  enabled?: boolean;
}

/** Value returned by useE2ICopilot hook */
export interface E2ICopilotContextValue {
  // Filter state
  filters: E2IFilters;
  setFilters: React.Dispatch<React.SetStateAction<E2IFilters>>;

  // Preferences state
  preferences: UserPreferences;
  setPreferences: React.Dispatch<React.SetStateAction<UserPreferences>>;

  // Chat UI state
  chatOpen: boolean;
  setChatOpen: React.Dispatch<React.SetStateAction<boolean>>;

  // Agent information
  agents: AgentInfo[];

  // Dashboard state (legacy)
  dashboard: DashboardContext;
  setDashboard: (updates: Partial<DashboardContext>) => void;

  // Highlights
  highlightedPaths: string[];
  setHighlightedPaths: React.Dispatch<React.SetStateAction<string[]>>;
  highlightedCharts: Map<string, string[]>;
  clearHighlights: () => void;
}

// -----------------------------------------------------------------------------
// Internal Context Types
// -----------------------------------------------------------------------------

interface E2IContextType {
  // Dashboard state
  dashboard: DashboardContext;
  setDashboard: (updates: Partial<DashboardContext>) => void;

  // Agent state
  activeAgents: Map<string, AgentState>;
  lastValidation?: ValidationSummary;

  // UI state controlled by agents
  highlightedPaths: string[];
  setHighlightedPaths: React.Dispatch<React.SetStateAction<string[]>>;
  highlightedCharts: Map<string, string[]>;
  pendingActions: E2ICopilotAction[];

  // Action handlers
  clearHighlights: () => void;
  dismissAction: (index: number) => void;

  // Extended state for hooks
  filters: E2IFilters;
  setFilters: React.Dispatch<React.SetStateAction<E2IFilters>>;
  preferences: UserPreferences;
  setPreferences: React.Dispatch<React.SetStateAction<UserPreferences>>;
  chatOpen: boolean;
  setChatOpen: React.Dispatch<React.SetStateAction<boolean>>;
  agents: AgentInfo[];
}

const E2IContext = createContext<E2IContextType | null>(null);

// Track if we're inside a CopilotKit provider
const CopilotEnabledContext = createContext<boolean>(false);

// -----------------------------------------------------------------------------
// Hooks
// -----------------------------------------------------------------------------

/**
 * Hook to check if CopilotKit is enabled in the current context.
 * Returns false when outside of a CopilotKit provider.
 */
export function useCopilotEnabled(): boolean {
  return useContext(CopilotEnabledContext);
}

/**
 * Hook to access E2I context (legacy name).
 * @throws Error if used outside of E2ICopilotProvider
 */
export function useE2I(): E2IContextType {
  const context = useContext(E2IContext);
  if (!context) {
    throw new Error('useE2I must be used within E2ICopilotProvider');
  }
  return context;
}

/**
 * Hook to access E2I Copilot context with filter, preferences, and chat state.
 * Safe to use outside provider - returns context or throws if required.
 */
export function useE2ICopilot(): E2ICopilotContextValue {
  const context = useContext(E2IContext);
  if (!context) {
    throw new Error('useE2ICopilot must be used within E2ICopilotProvider');
  }
  return {
    filters: context.filters,
    setFilters: context.setFilters,
    preferences: context.preferences,
    setPreferences: context.setPreferences,
    chatOpen: context.chatOpen,
    setChatOpen: context.setChatOpen,
    agents: context.agents,
    dashboard: context.dashboard,
    setDashboard: context.setDashboard,
    highlightedPaths: context.highlightedPaths,
    setHighlightedPaths: context.setHighlightedPaths,
    highlightedCharts: context.highlightedCharts,
    clearHighlights: context.clearHighlights,
  };
}

// -----------------------------------------------------------------------------
// CopilotKitWrapper - Minimal wrapper for CopilotKit
// -----------------------------------------------------------------------------

/**
 * Minimal wrapper around CopilotKit for use without full E2I context.
 * Use E2ICopilotProvider for full functionality.
 */
export const CopilotKitWrapper: React.FC<CopilotKitWrapperProps> = ({
  children,
  runtimeUrl = '/api/copilotkit',
  apiKey,
  enabled = true,
}) => {
  // When disabled, render children directly without CopilotKit
  if (!enabled) {
    return (
      <CopilotEnabledContext.Provider value={false}>
        {children}
      </CopilotEnabledContext.Provider>
    );
  }

  return (
    <CopilotEnabledContext.Provider value={true}>
      <CopilotKit
        runtimeUrl={runtimeUrl}
        publicApiKey={apiKey}
        agent="e2i-orchestrator"
        showDevConsole={process.env.NODE_ENV === 'development'}
      >
        {children}
      </CopilotKit>
    </CopilotEnabledContext.Provider>
  );
};

// -----------------------------------------------------------------------------
// Main Provider Component
// -----------------------------------------------------------------------------

export const E2ICopilotProvider: React.FC<E2ICopilotProviderProps> = ({
  children,
  apiKey,
  runtimeUrl = '/api/copilotkit',
  initialFilters = {},
  userRole = 'analyst',
}) => {
  return (
    <CopilotEnabledContext.Provider value={true}>
      <CopilotKit
        runtimeUrl={runtimeUrl}
        publicApiKey={apiKey}
        agent="e2i-orchestrator"
        showDevConsole={process.env.NODE_ENV === 'development'}
      >
        <E2IContextProvider initialFilters={initialFilters} userRole={userRole}>
          {children}
        </E2IContextProvider>
      </CopilotKit>
    </CopilotEnabledContext.Provider>
  );
};

// -----------------------------------------------------------------------------
// Inner Context Provider (with CopilotKit hooks)
// -----------------------------------------------------------------------------

interface E2IContextProviderProps {
  children: ReactNode;
  initialFilters: FilterContext;
  userRole: DashboardContext['userRole'];
}

// Default values for new state
const DEFAULT_FILTERS: E2IFilters = {
  brand: 'Remibrutinib',
  territory: null,
  dateRange: {
    start: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end: new Date().toISOString().split('T')[0],
  },
  hcpSegment: null,
};

const DEFAULT_PREFERENCES: UserPreferences = {
  detailLevel: 'detailed',
  defaultBrand: 'Remibrutinib',
  notificationsEnabled: true,
  theme: 'system',
};

const E2IContextProvider: React.FC<E2IContextProviderProps> = ({
  children,
  initialFilters,
  userRole,
}) => {
  // Dashboard state
  const [dashboard, setDashboardState] = useState<DashboardContext>({
    filters: initialFilters,
    activeTab: 'overview',
    selectedKPIs: [],
    visibleCharts: [],
    highlightedElements: [],
    userRole,
  });

  // New state for hooks
  const [filters, setFilters] = useState<E2IFilters>(DEFAULT_FILTERS);
  const [preferences, setPreferences] = useState<UserPreferences>(DEFAULT_PREFERENCES);
  const [chatOpen, setChatOpen] = useState<boolean>(false);

  // Agent state
  const [activeAgents, setActiveAgents] = useState<Map<string, AgentState>>(
    new Map()
  );
  const [lastValidation, setLastValidation] = useState<ValidationSummary>();

  // UI state controlled by agents
  const [highlightedPaths, setHighlightedPaths] = useState<string[]>([]);
  const [highlightedCharts, setHighlightedCharts] = useState<
    Map<string, string[]>
  >(new Map());
  const [pendingActions, setPendingActions] = useState<E2ICopilotAction[]>([]);

  // Derive agents list from AGENT_REGISTRY and activeAgents state
  const agents: AgentInfo[] = Object.values(AGENT_REGISTRY).map((agent) => {
    const activeState = activeAgents.get(agent.id);
    return {
      id: agent.id,
      name: agent.name,
      tier: agent.tier,
      status: activeState?.status === 'thinking' ? 'processing' :
              activeState?.status === 'computing' ? 'processing' :
              activeState?.status === 'complete' ? 'complete' :
              activeState?.status === 'error' ? 'error' :
              activeState?.status === 'idle' ? 'idle' : 'idle',
      capabilities: agent.actions || [],
      currentTask: activeState?.currentTask,
    };
  });

  // Update dashboard helper
  const setDashboard = useCallback((updates: Partial<DashboardContext>) => {
    setDashboardState((prev) => ({ ...prev, ...updates }));
  }, []);

  // ---------------------------------------------------------------------------
  // useCopilotReadable - Make app state visible to agents
  // ---------------------------------------------------------------------------

  // Current filter context
  useCopilotReadable({
    description:
      'Current dashboard filters including brand, region, time range, and segments',
    value: dashboard.filters,
  });

  // Active dashboard tab
  useCopilotReadable({
    description: 'Currently active dashboard tab/view',
    value: dashboard.activeTab,
  });

  // Selected KPIs
  useCopilotReadable({
    description: 'KPIs currently selected for analysis',
    value: dashboard.selectedKPIs,
  });

  // Visible charts
  useCopilotReadable({
    description: 'Charts currently visible in the dashboard viewport',
    value: dashboard.visibleCharts,
  });

  // User role (affects explanation detail level)
  useCopilotReadable({
    description:
      'User role: executive (high-level), analyst (detailed), or data_scientist (technical)',
    value: dashboard.userRole,
  });

  // Agent registry (so agents know their capabilities)
  useCopilotReadable({
    description: 'Available E2I agents with their tiers and capabilities',
    value: Object.values(AGENT_REGISTRY).map((a) => ({
      id: a.id,
      name: a.name,
      tier: a.tier,
      type: a.type,
      actions: a.actions,
    })),
  });

  // ---------------------------------------------------------------------------
  // useCopilotAction - Agent-triggered UI actions
  // ---------------------------------------------------------------------------

  // Update filters from agent
  useCopilotAction({
    name: 'updateFilters',
    description:
      'Update dashboard filters based on analysis context. Use when the agent identifies a specific segment or time period to focus on.',
    parameters: [
      {
        name: 'brand',
        type: 'string',
        description: 'Brand to filter: Remibrutinib, Fabhalta, or Kisqali',
        required: false,
      },
      {
        name: 'region',
        type: 'string',
        description: 'Region to filter: northeast, south, midwest, or west',
        required: false,
      },
      {
        name: 'startDate',
        type: 'string',
        description: 'Start date for time range (ISO format)',
        required: false,
      },
      {
        name: 'endDate',
        type: 'string',
        description: 'End date for time range (ISO format)',
        required: false,
      },
    ],
    handler: async ({ brand, region, startDate, endDate }) => {
      const newFilters: Partial<FilterContext> = {};
      if (brand) newFilters.brand = brand as FilterContext['brand'];
      if (region) newFilters.region = region as FilterContext['region'];
      if (startDate && endDate) {
        newFilters.timeRange = { start: startDate, end: endDate };
      }
      setDashboard({ filters: { ...dashboard.filters, ...newFilters } });
      return `Filters updated: ${JSON.stringify(newFilters)}`;
    },
  });

  // Highlight causal path in visualization
  useCopilotAction({
    name: 'highlightCausalPath',
    description:
      'Highlight a causal relationship in the DAG visualization. Use when explaining a specific causal effect.',
    parameters: [
      {
        name: 'treatment',
        type: 'string',
        description: 'Treatment/intervention variable',
        required: true,
      },
      {
        name: 'outcome',
        type: 'string',
        description: 'Outcome variable',
        required: true,
      },
      {
        name: 'effect',
        type: 'number',
        description: 'Estimated causal effect size',
        required: true,
      },
      {
        name: 'mediators',
        type: 'string',
        description: 'Comma-separated list of mediator variables',
        required: false,
      },
    ],
    handler: async ({ treatment, outcome, effect, mediators }) => {
      const pathId = `${treatment}->${outcome}`;
      setHighlightedPaths((prev) => [...prev, pathId]);
      
      // Add to pending actions for UI rendering
      setPendingActions((prev) => [
        ...prev,
        {
          type: 'highlightCausalPath',
          treatment,
          outcome,
          effect,
          mediators: mediators?.split(',').map((m) => m.trim()),
        },
      ]);
      
      return `Highlighted causal path: ${treatment} → ${outcome} (effect: ${effect})`;
    },
  });

  // Show validation results
  useCopilotAction({
    name: 'showValidationResults',
    description:
      'Display causal validation results including refutation tests and gate decision. Use after running causal analysis.',
    parameters: [
      {
        name: 'estimateId',
        type: 'string',
        description: 'ID of the causal estimate',
        required: true,
      },
      {
        name: 'gateDecision',
        type: 'string',
        description: 'Gate decision: proceed, review, or block',
        required: true,
      },
      {
        name: 'confidence',
        type: 'number',
        description: 'Overall confidence score (0-1)',
        required: true,
      },
      {
        name: 'testsPassed',
        type: 'number',
        description: 'Number of tests passed',
        required: true,
      },
      {
        name: 'testsFailed',
        type: 'number',
        description: 'Number of tests failed',
        required: true,
      },
    ],
    handler: async ({
      estimateId,
      gateDecision,
      confidence,
      testsPassed,
      testsFailed,
    }) => {
      const validation: ValidationSummary = {
        estimateId,
        gateDecision: gateDecision as ValidationSummary['gateDecision'],
        overallConfidence: confidence,
        testsRun: testsPassed + testsFailed,
        testsPassed,
        testsFailed,
        testsWarning: 0,
        results: [],
        timestamp: new Date().toISOString(),
      };
      setLastValidation(validation);
      
      setPendingActions((prev) => [
        ...prev,
        { type: 'showValidationResults', validation },
      ]);
      
      return `Validation results displayed: ${gateDecision} (${Math.round(confidence * 100)}% confidence)`;
    },
  });

  // Navigate to dashboard tab
  useCopilotAction({
    name: 'navigateToTab',
    description:
      'Switch the dashboard to a specific tab/view. Use when directing user attention to relevant analysis.',
    parameters: [
      {
        name: 'tab',
        type: 'string',
        description:
          'Tab to navigate to: overview, ws1-data, ws1-ml, ws2, ws3, causal, validation, knowledge-graph',
        required: true,
      },
    ],
    handler: async ({ tab }) => {
      setDashboard({ activeTab: tab });
      return `Navigated to ${tab} tab`;
    },
  });

  // Highlight chart element
  useCopilotAction({
    name: 'highlightChartElement',
    description:
      'Highlight a specific element in a chart. Use when pointing out anomalies, trends, or important data points.',
    parameters: [
      {
        name: 'chartId',
        type: 'string',
        description: 'ID of the chart to highlight',
        required: true,
      },
      {
        name: 'elementId',
        type: 'string',
        description: 'ID of the element within the chart',
        required: true,
      },
      {
        name: 'annotation',
        type: 'string',
        description: 'Annotation text to display',
        required: false,
      },
    ],
    handler: async ({ chartId, elementId, annotation }) => {
      setHighlightedCharts((prev) => {
        const newMap = new Map(prev);
        const existing = newMap.get(chartId) || [];
        newMap.set(chartId, [...existing, elementId]);
        return newMap;
      });
      
      setPendingActions((prev) => [
        ...prev,
        { type: 'highlightChartElement', chartId, elementId, annotation },
      ]);
      
      return `Highlighted ${elementId} in chart ${chartId}`;
    },
  });

  // Show gap details
  useCopilotAction({
    name: 'showGapDetails',
    description:
      'Display detailed information about a performance gap. Use when the Gap Analyzer identifies issues.',
    parameters: [
      {
        name: 'gapId',
        type: 'string',
        description: 'Unique identifier for the gap',
        required: true,
      },
      {
        name: 'kpi',
        type: 'string',
        description: 'KPI name',
        required: true,
      },
      {
        name: 'expected',
        type: 'number',
        description: 'Expected value',
        required: true,
      },
      {
        name: 'actual',
        type: 'number',
        description: 'Actual value',
        required: true,
      },
      {
        name: 'rootCauses',
        type: 'string',
        description: 'Comma-separated list of root causes',
        required: true,
      },
    ],
    handler: async ({ gapId, kpi, expected, actual, rootCauses }) => {
      setPendingActions((prev) => [
        ...prev,
        {
          type: 'showGapDetails',
          gapId,
          kpi,
          expected,
          actual,
          rootCauses: rootCauses.split(',').map((c) => c.trim()),
        },
      ]);
      
      return `Showing gap details for ${kpi}: ${actual} vs expected ${expected}`;
    },
  });

  // Generate report
  useCopilotAction({
    name: 'generateReport',
    description:
      'Generate an exportable report. Use when user requests documentation or wants to share findings.',
    parameters: [
      {
        name: 'format',
        type: 'string',
        description: 'Report format: pdf, pptx, or docx',
        required: true,
      },
      {
        name: 'sections',
        type: 'string',
        description: 'Comma-separated list of sections to include',
        required: true,
      },
    ],
    handler: async ({ format, sections }) => {
      setPendingActions((prev) => [
        ...prev,
        {
          type: 'generateReport',
          format: format as 'pdf' | 'pptx' | 'docx',
          sections: sections.split(',').map((s) => s.trim()),
        },
      ]);
      
      return `Generating ${format} report with sections: ${sections}`;
    },
  });

  // Update agent status (called by orchestrator)
  useCopilotAction({
    name: 'updateAgentStatus',
    description:
      'Update the status of an agent. Used by Orchestrator to track agent activity.',
    parameters: [
      {
        name: 'agentId',
        type: 'string',
        description: 'ID of the agent',
        required: true,
      },
      {
        name: 'status',
        type: 'string',
        description: 'Status: idle, thinking, computing, complete, or error',
        required: true,
      },
      {
        name: 'task',
        type: 'string',
        description: 'Current task description',
        required: false,
      },
    ],
    handler: async ({ agentId, status, task }) => {
      setActiveAgents((prev) => {
        const newMap = new Map(prev);
        newMap.set(agentId, {
          id: agentId,
          status: status as AgentState['status'],
          currentTask: task,
        });
        return newMap;
      });
      
      return `Agent ${agentId} status: ${status}`;
    },
  });

  // ---------------------------------------------------------------------------
  // Helper functions
  // ---------------------------------------------------------------------------

  const clearHighlights = useCallback(() => {
    setHighlightedPaths([]);
    setHighlightedCharts(new Map());
  }, []);

  const dismissAction = useCallback((index: number) => {
    setPendingActions((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // ---------------------------------------------------------------------------
  // Context value
  // ---------------------------------------------------------------------------

  const contextValue: E2IContextType = {
    dashboard,
    setDashboard,
    activeAgents,
    lastValidation,
    highlightedPaths,
    setHighlightedPaths,
    highlightedCharts,
    pendingActions,
    clearHighlights,
    dismissAction,
    // Extended state for hooks
    filters,
    setFilters,
    preferences,
    setPreferences,
    chatOpen,
    setChatOpen,
    agents,
  };

  return (
    <E2IContext.Provider value={contextValue}>{children}</E2IContext.Provider>
  );
};

export default E2ICopilotProvider;
