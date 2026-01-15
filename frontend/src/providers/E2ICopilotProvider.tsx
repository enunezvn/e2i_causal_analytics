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
import { useNavigate, useLocation } from 'react-router-dom';

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

/** Registry of available agents - all 19 agents in 6 tiers */
const AGENT_REGISTRY: Record<string, AgentRegistryEntry> = {
  // Tier 0: ML Foundation (7 agents)
  'scope-definer': {
    id: 'scope-definer',
    name: 'Scope Definer',
    tier: 0,
    type: 'foundation',
    actions: ['define_scope', 'validate_requirements'],
    icon: '📋',
  },
  'data-preparer': {
    id: 'data-preparer',
    name: 'Data Preparer',
    tier: 0,
    type: 'foundation',
    actions: ['prepare_data', 'validate_schema'],
    icon: '📊',
  },
  'feature-analyzer': {
    id: 'feature-analyzer',
    name: 'Feature Analyzer',
    tier: 0,
    type: 'foundation',
    actions: ['analyze_features', 'select_features'],
    icon: '🔍',
  },
  'model-selector': {
    id: 'model-selector',
    name: 'Model Selector',
    tier: 0,
    type: 'foundation',
    actions: ['select_model', 'benchmark_models'],
    icon: '🎯',
  },
  'model-trainer': {
    id: 'model-trainer',
    name: 'Model Trainer',
    tier: 0,
    type: 'foundation',
    actions: ['train_model', 'tune_hyperparameters'],
    icon: '🏋️',
  },
  'model-deployer': {
    id: 'model-deployer',
    name: 'Model Deployer',
    tier: 0,
    type: 'foundation',
    actions: ['deploy_model', 'version_model'],
    icon: '🚀',
  },
  'observability-connector': {
    id: 'observability-connector',
    name: 'Observability Connector',
    tier: 0,
    type: 'foundation',
    actions: ['connect_mlflow', 'setup_monitoring'],
    icon: '📡',
  },
  // Tier 1: Orchestration (2 agents)
  orchestrator: {
    id: 'orchestrator',
    name: 'Orchestrator',
    tier: 1,
    type: 'coordinator',
    actions: ['route_query', 'coordinate_agents'],
    icon: '🎯',
  },
  'tool-composer': {
    id: 'tool-composer',
    name: 'Tool Composer',
    tier: 1,
    type: 'coordinator',
    actions: ['compose_tools', 'decompose_query'],
    icon: '🔧',
  },
  // Tier 2: Causal Analytics (3 agents)
  'causal-impact': {
    id: 'causal-impact',
    name: 'Causal Impact',
    tier: 2,
    type: 'analytical',
    actions: ['estimate_effect', 'run_refutation'],
    icon: '🔗',
  },
  'gap-analyzer': {
    id: 'gap-analyzer',
    name: 'Gap Analyzer',
    tier: 2,
    type: 'analytical',
    actions: ['identify_gaps', 'calculate_roi'],
    icon: '📊',
  },
  'heterogeneous-optimizer': {
    id: 'heterogeneous-optimizer',
    name: 'Heterogeneous Optimizer',
    tier: 2,
    type: 'analytical',
    actions: ['analyze_cate', 'segment_rank'],
    icon: '🎯',
  },
  // Tier 3: Monitoring (3 agents)
  'drift-monitor': {
    id: 'drift-monitor',
    name: 'Drift Monitor',
    tier: 3,
    type: 'monitoring',
    actions: ['detect_drift', 'compare_distributions'],
    icon: '📈',
  },
  'experiment-designer': {
    id: 'experiment-designer',
    name: 'Experiment Designer',
    tier: 3,
    type: 'planning',
    actions: ['design_experiment', 'power_analysis'],
    icon: '🧪',
  },
  'health-score': {
    id: 'health-score',
    name: 'Health Score',
    tier: 3,
    type: 'monitoring',
    actions: ['calculate_health', 'monitor_metrics'],
    icon: '💚',
  },
  // Tier 4: ML Predictions (2 agents)
  'prediction-synthesizer': {
    id: 'prediction-synthesizer',
    name: 'Prediction Synthesizer',
    tier: 4,
    type: 'prediction',
    actions: ['synthesize_predictions', 'aggregate_models'],
    icon: '🔮',
  },
  'resource-optimizer': {
    id: 'resource-optimizer',
    name: 'Resource Optimizer',
    tier: 4,
    type: 'optimization',
    actions: ['optimize_resources', 'allocate_budget'],
    icon: '⚡',
  },
  // Tier 5: Self-Improvement (2 agents)
  explainer: {
    id: 'explainer',
    name: 'Explainer',
    tier: 5,
    type: 'communication',
    actions: ['generate_explanation', 'visualize_insight'],
    icon: '💡',
  },
  'feedback-learner': {
    id: 'feedback-learner',
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
  enabled = false,
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
  initialFilters = {},
  userRole = 'analyst',
}) => {
  // E2ICopilotProvider does NOT wrap CopilotKit - use CopilotKitWrapper externally
  return (
    <E2IContextProvider initialFilters={initialFilters} userRole={userRole}>
      <CopilotHooksConnector />
      {children}
    </E2IContextProvider>
  );
};

// -----------------------------------------------------------------------------
// Inner Context Provider (state only, no CopilotKit hooks)
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
  const [activeAgents] = useState<Map<string, AgentState>>(new Map());
  const [lastValidation] = useState<ValidationSummary>();

  // UI state controlled by agents
  const [highlightedPaths, setHighlightedPaths] = useState<string[]>([]);
  const [highlightedCharts, setHighlightedCharts] = useState<Map<string, string[]>>(new Map());
  const [pendingActions] = useState<E2ICopilotAction[]>([]);

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

  // Helper functions
  const clearHighlights = useCallback(() => {
    setHighlightedPaths([]);
    setHighlightedCharts(new Map());
  }, []);

  const dismissAction = useCallback(() => {
    // No-op when hooks are disabled
  }, []);

  // Context value
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

// -----------------------------------------------------------------------------
// CopilotHooksConnector - Registers CopilotKit hooks when enabled
// -----------------------------------------------------------------------------

const VALID_BRANDS = ['Remibrutinib', 'Fabhalta', 'Kisqali', 'All'];
const VALID_DETAIL_LEVELS = ['summary', 'detailed', 'expert'];

/**
 * Component that conditionally renders hook registration.
 * Only renders CopilotHooksInner when CopilotKit is enabled.
 */
const CopilotHooksConnector: React.FC = () => {
  const enabled = useCopilotEnabled();

  // Don't render hook registration component when disabled
  if (!enabled) {
    return null;
  }

  return <CopilotHooksInner />;
};

/**
 * Inner component that actually registers CopilotKit hooks.
 * Only mounted when CopilotKit is enabled (controlled by CopilotHooksConnector).
 */
const CopilotHooksInner: React.FC = () => {
  const context = useContext(E2IContext);
  const navigate = useNavigate();
  const location = useLocation();

  // ---------------------------------------------------------------------------
  // Readables - Make app state visible to agents (4 readables)
  // ---------------------------------------------------------------------------

  // 1. Current filter context (dashboard filters)
  useCopilotReadable({
    description: 'Current dashboard filters including brand, region, time range, and segments',
    value: context?.filters || DEFAULT_FILTERS,
  });

  // 2. Page context
  useCopilotReadable({
    description: 'Current page path and navigation context',
    value: { currentPath: location.pathname },
  });

  // 3. Agent tier hierarchy
  useCopilotReadable({
    description: 'E2I agent tier hierarchy with 19 agents across 6 tiers',
    value: context?.agents || [],
  });

  // 4. User preferences
  useCopilotReadable({
    description: 'User preferences including detail level and theme',
    value: context?.preferences || DEFAULT_PREFERENCES,
  });

  // ---------------------------------------------------------------------------
  // Actions - Agent-triggered UI actions (6 actions)
  // ---------------------------------------------------------------------------

  // 1. Navigate to path
  useCopilotAction({
    name: 'navigateTo',
    description: 'Navigate to a specific page path',
    parameters: [
      {
        name: 'path',
        type: 'string',
        description: 'The path to navigate to (e.g., /knowledge-graph)',
        required: true,
      },
    ],
    handler: ({ path }: { path: string }) => {
      navigate(path);
      return `Navigated to ${path}`;
    },
  });

  // 2. Set brand filter
  useCopilotAction({
    name: 'setBrandFilter',
    description: 'Set the brand filter for the dashboard',
    parameters: [
      {
        name: 'brand',
        type: 'string',
        description: 'Brand to filter: Remibrutinib, Fabhalta, Kisqali, or All',
        required: true,
      },
    ],
    handler: ({ brand }: { brand: string }) => {
      if (!VALID_BRANDS.includes(brand)) {
        return `Invalid brand. Choose from: ${VALID_BRANDS.join(', ')}`;
      }
      if (context) {
        context.setFilters((prev) => ({
          ...prev,
          brand: brand as E2IFilters['brand'],
        }));
      }
      return `Brand filter set to ${brand}`;
    },
  });

  // 3. Set date range
  useCopilotAction({
    name: 'setDateRange',
    description: 'Set the date range filter for the dashboard',
    parameters: [
      {
        name: 'startDate',
        type: 'string',
        description: 'Start date in ISO format (YYYY-MM-DD)',
        required: true,
      },
      {
        name: 'endDate',
        type: 'string',
        description: 'End date in ISO format (YYYY-MM-DD)',
        required: true,
      },
    ],
    handler: ({ startDate, endDate }: { startDate: string; endDate: string }) => {
      if (context) {
        context.setFilters((prev) => ({
          ...prev,
          dateRange: { start: startDate, end: endDate },
        }));
      }
      return `Date range set to ${startDate} - ${endDate}`;
    },
  });

  // 4. Highlight causal paths
  useCopilotAction({
    name: 'highlightCausalPaths',
    description: 'Highlight causal paths in the visualization',
    parameters: [
      {
        name: 'pathIds',
        type: 'string[]',
        description: 'Array of path IDs to highlight',
        required: true,
      },
    ],
    handler: ({ pathIds }: { pathIds: string[] }) => {
      if (context) {
        context.setHighlightedPaths(pathIds);
      }
      return `Highlighted ${pathIds.length} causal path(s)`;
    },
  });

  // 5. Set detail level
  useCopilotAction({
    name: 'setDetailLevel',
    description: 'Set the detail level for explanations',
    parameters: [
      {
        name: 'level',
        type: 'string',
        description: 'Detail level: summary, detailed, or expert',
        required: true,
      },
    ],
    handler: ({ level }: { level: string }) => {
      if (!VALID_DETAIL_LEVELS.includes(level)) {
        return `Invalid level. Choose from: ${VALID_DETAIL_LEVELS.join(', ')}`;
      }
      if (context) {
        context.setPreferences((prev) => ({
          ...prev,
          detailLevel: level as UserPreferences['detailLevel'],
        }));
      }
      return `Detail level set to ${level}`;
    },
  });

  // 6. Toggle chat
  useCopilotAction({
    name: 'toggleChat',
    description: 'Toggle the chat panel open or closed',
    parameters: [
      {
        name: 'open',
        type: 'boolean',
        description: 'Whether to open (true) or close (false) the chat',
        required: false,
      },
    ],
    handler: ({ open }: { open?: boolean }) => {
      if (context) {
        if (open === undefined) {
          // Toggle
          context.setChatOpen((prev) => !prev);
          return context.chatOpen ? 'Chat closed' : 'Chat opened';
        }
        context.setChatOpen(open);
        return open ? 'Chat opened' : 'Chat closed';
      }
      return 'Chat toggled';
    },
  });

  // Doesn't render anything - just registers hooks
  return null;
};

export default E2ICopilotProvider;
