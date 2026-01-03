/**
 * E2I CopilotKit Provider
 * =======================
 *
 * Wraps the application with CopilotKit context and exposes E2I-specific
 * readables and actions for the AI assistant.
 *
 * Features:
 * - Dashboard filter context exposed to AI
 * - Agent tier information
 * - User preferences
 * - UI interaction actions
 *
 * @module providers/E2ICopilotProvider
 */

import * as React from 'react';
import { CopilotKit } from '@copilotkit/react-core';
import {
  useCopilotReadable,
  useCopilotAction,
} from '@copilotkit/react-core';
import { useNavigate, useLocation } from 'react-router-dom';

// =============================================================================
// TYPES
// =============================================================================

export interface E2IFilters {
  brand: 'Remibrutinib' | 'Fabhalta' | 'Kisqali' | 'All';
  territory: string | null;
  dateRange: {
    start: string;
    end: string;
  };
  hcpSegment: string | null;
}

export interface AgentInfo {
  id: string;
  name: string;
  tier: 0 | 1 | 2 | 3 | 4 | 5;
  status: 'active' | 'idle' | 'processing' | 'error';
  lastActivity?: string;
  capabilities: string[];
}

export interface UserPreferences {
  detailLevel: 'summary' | 'detailed' | 'expert';
  defaultBrand: E2IFilters['brand'];
  notificationsEnabled: boolean;
  theme: 'light' | 'dark' | 'system';
}

export interface E2ICopilotContextValue {
  filters: E2IFilters;
  setFilters: React.Dispatch<React.SetStateAction<E2IFilters>>;
  agents: AgentInfo[];
  preferences: UserPreferences;
  setPreferences: React.Dispatch<React.SetStateAction<UserPreferences>>;
  highlightedPaths: string[];
  setHighlightedPaths: React.Dispatch<React.SetStateAction<string[]>>;
  chatOpen: boolean;
  setChatOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

export interface E2ICopilotProviderProps {
  children: React.ReactNode;
  runtimeUrl?: string;
  initialFilters?: Partial<E2IFilters>;
  userRole?: string;
}

// =============================================================================
// CONTEXT
// =============================================================================

const E2ICopilotContext = React.createContext<E2ICopilotContextValue | null>(null);

export function useE2ICopilot() {
  const context = React.useContext(E2ICopilotContext);
  if (!context) {
    throw new Error('useE2ICopilot must be used within E2ICopilotProvider');
  }
  return context;
}

// =============================================================================
// DEFAULT DATA
// =============================================================================

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

const SAMPLE_AGENTS: AgentInfo[] = [
  // Tier 0 - ML Foundation
  { id: 'scope-definer', name: 'Scope Definer', tier: 0, status: 'idle', capabilities: ['problem_scoping', 'requirement_analysis'] },
  { id: 'data-preparer', name: 'Data Preparer', tier: 0, status: 'idle', capabilities: ['data_validation', 'preprocessing'] },
  { id: 'feature-analyzer', name: 'Feature Analyzer', tier: 0, status: 'idle', capabilities: ['feature_engineering', 'selection'] },
  { id: 'model-selector', name: 'Model Selector', tier: 0, status: 'idle', capabilities: ['model_comparison', 'benchmarking'] },
  { id: 'model-trainer', name: 'Model Trainer', tier: 0, status: 'idle', capabilities: ['training', 'hyperparameter_tuning'] },
  { id: 'model-deployer', name: 'Model Deployer', tier: 0, status: 'idle', capabilities: ['deployment', 'versioning'] },
  { id: 'observability-connector', name: 'Observability Connector', tier: 0, status: 'active', capabilities: ['mlflow', 'opik', 'monitoring'] },
  // Tier 1 - Orchestration
  { id: 'orchestrator', name: 'Orchestrator', tier: 1, status: 'active', capabilities: ['routing', 'coordination', 'agent_dispatch'] },
  { id: 'tool-composer', name: 'Tool Composer', tier: 1, status: 'idle', capabilities: ['tool_selection', 'multi_tool'] },
  // Tier 2 - Causal Analytics
  { id: 'causal-impact', name: 'Causal Impact', tier: 2, status: 'idle', capabilities: ['ate_estimation', 'effect_chains'] },
  { id: 'gap-analyzer', name: 'Gap Analyzer', tier: 2, status: 'idle', capabilities: ['roi_analysis', 'opportunity_detection'] },
  { id: 'heterogeneous-optimizer', name: 'Heterogeneous Optimizer', tier: 2, status: 'idle', capabilities: ['cate_analysis', 'segment_optimization'] },
  // Tier 3 - Monitoring
  { id: 'drift-monitor', name: 'Drift Monitor', tier: 3, status: 'active', capabilities: ['data_drift', 'model_drift'] },
  { id: 'experiment-designer', name: 'Experiment Designer', tier: 3, status: 'idle', capabilities: ['ab_design', 'power_analysis'] },
  { id: 'health-score', name: 'Health Score', tier: 3, status: 'active', capabilities: ['system_health', 'model_health'] },
  // Tier 4 - ML Predictions
  { id: 'prediction-synthesizer', name: 'Prediction Synthesizer', tier: 4, status: 'idle', capabilities: ['prediction_aggregation', 'ensemble'] },
  { id: 'resource-optimizer', name: 'Resource Optimizer', tier: 4, status: 'idle', capabilities: ['resource_allocation', 'scheduling'] },
  // Tier 5 - Self-Improvement
  { id: 'explainer', name: 'Explainer', tier: 5, status: 'idle', capabilities: ['narratives', 'shap_explanations'] },
  { id: 'feedback-learner', name: 'Feedback Learner', tier: 5, status: 'idle', capabilities: ['prompt_optimization', 'self_improvement'] },
];

// =============================================================================
// COPILOT ENABLED CONTEXT
// =============================================================================

// Context to track if CopilotKit is enabled
const CopilotEnabledContext = React.createContext<boolean>(false);

/**
 * Hook to check if CopilotKit is enabled
 */
export function useCopilotEnabled(): boolean {
  return React.useContext(CopilotEnabledContext);
}

// =============================================================================
// INTERNAL CONTEXT PROVIDER (without CopilotKit hooks)
// =============================================================================

function E2IBaseContextProvider({ children }: { children: React.ReactNode }) {
  // State
  const [filters, setFilters] = React.useState<E2IFilters>(DEFAULT_FILTERS);
  const [agents] = React.useState<AgentInfo[]>(SAMPLE_AGENTS);
  const [preferences, setPreferences] = React.useState<UserPreferences>(DEFAULT_PREFERENCES);
  const [highlightedPaths, setHighlightedPaths] = React.useState<string[]>([]);
  const [chatOpen, setChatOpen] = React.useState(false);

  const contextValue: E2ICopilotContextValue = {
    filters,
    setFilters,
    agents,
    preferences,
    setPreferences,
    highlightedPaths,
    setHighlightedPaths,
    chatOpen,
    setChatOpen,
  };

  return (
    <E2ICopilotContext.Provider value={contextValue}>
      {children}
    </E2ICopilotContext.Provider>
  );
}

// =============================================================================
// COPILOTKIT HOOKS CONNECTOR (only when CopilotKit is enabled)
// =============================================================================

function CopilotHooksConnector({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { filters, setFilters, agents, preferences, setPreferences, setHighlightedPaths, chatOpen, setChatOpen } = useE2ICopilot();

  // Expose dashboard filters to AI
  useCopilotReadable({
    description: 'Current dashboard filters including brand, territory, date range, and HCP segment',
    value: filters,
  });

  // Expose current page context
  useCopilotReadable({
    description: 'Current page path and available navigation routes',
    value: {
      currentPath: location.pathname,
      availableRoutes: [
        '/',
        '/knowledge-graph',
        '/causal-discovery',
        '/model-performance',
        '/feature-importance',
        '/data-quality',
        '/system-health',
        '/monitoring',
        '/time-series',
        '/intervention-impact',
        '/predictive-analytics',
      ],
    },
  });

  // Expose agent information to AI
  useCopilotReadable({
    description: 'E2I agent tier hierarchy with 18 agents across 6 tiers (0-5). Tier 0 is ML Foundation, Tier 1 is Orchestration, Tier 2 is Causal Analytics, Tier 3 is Monitoring, Tier 4 is ML Predictions, Tier 5 is Self-Improvement.',
    value: agents,
  });

  // Expose user preferences
  useCopilotReadable({
    description: 'User preferences for detail level, default brand, and notifications',
    value: preferences,
  });

  // Action: Navigate to a page
  useCopilotAction({
    name: 'navigateTo',
    description: 'Navigate to a specific page in the E2I dashboard',
    parameters: [
      {
        name: 'path',
        type: 'string',
        description: 'The path to navigate to (e.g., /knowledge-graph, /model-performance)',
        required: true,
      },
    ],
    handler: ({ path }: { path: string }) => {
      navigate(path);
      return `Navigated to ${path}`;
    },
  });

  // Action: Set brand filter
  useCopilotAction({
    name: 'setBrandFilter',
    description: 'Change the brand filter to Remibrutinib, Fabhalta, Kisqali, or All',
    parameters: [
      {
        name: 'brand',
        type: 'string',
        description: 'The brand to filter by',
        required: true,
      },
    ],
    handler: ({ brand }: { brand: string }) => {
      const validBrands = ['Remibrutinib', 'Fabhalta', 'Kisqali', 'All'];
      if (validBrands.includes(brand)) {
        setFilters((prev) => ({ ...prev, brand: brand as E2IFilters['brand'] }));
        return `Brand filter set to ${brand}`;
      }
      return `Invalid brand. Choose from: ${validBrands.join(', ')}`;
    },
  });

  // Action: Set date range
  useCopilotAction({
    name: 'setDateRange',
    description: 'Set the date range for analytics',
    parameters: [
      {
        name: 'startDate',
        type: 'string',
        description: 'Start date in YYYY-MM-DD format',
        required: true,
      },
      {
        name: 'endDate',
        type: 'string',
        description: 'End date in YYYY-MM-DD format',
        required: true,
      },
    ],
    handler: ({ startDate, endDate }: { startDate: string; endDate: string }) => {
      setFilters((prev) => ({
        ...prev,
        dateRange: { start: startDate, end: endDate },
      }));
      return `Date range set to ${startDate} - ${endDate}`;
    },
  });

  // Action: Highlight causal paths
  useCopilotAction({
    name: 'highlightCausalPaths',
    description: 'Highlight specific causal paths on the Knowledge Graph or Causal Discovery pages',
    parameters: [
      {
        name: 'pathIds',
        type: 'string[]',
        description: 'Array of path IDs to highlight',
        required: true,
      },
    ],
    handler: ({ pathIds }: { pathIds: string[] }) => {
      setHighlightedPaths(pathIds);
      return `Highlighted ${pathIds.length} causal path(s)`;
    },
  });

  // Action: Set detail level
  useCopilotAction({
    name: 'setDetailLevel',
    description: 'Set the response detail level for AI explanations',
    parameters: [
      {
        name: 'level',
        type: 'string',
        description: 'Detail level: summary, detailed, or expert',
        required: true,
      },
    ],
    handler: ({ level }: { level: string }) => {
      const validLevels = ['summary', 'detailed', 'expert'];
      if (validLevels.includes(level)) {
        setPreferences((prev) => ({ ...prev, detailLevel: level as UserPreferences['detailLevel'] }));
        return `Detail level set to ${level}`;
      }
      return `Invalid level. Choose from: ${validLevels.join(', ')}`;
    },
  });

  // Action: Toggle chat sidebar
  useCopilotAction({
    name: 'toggleChat',
    description: 'Open or close the chat sidebar',
    parameters: [
      {
        name: 'open',
        type: 'boolean',
        description: 'Whether to open (true) or close (false) the chat',
        required: false,
      },
    ],
    handler: ({ open }: { open?: boolean }) => {
      const newState = open !== undefined ? open : !chatOpen;
      setChatOpen(newState);
      return newState ? 'Chat opened' : 'Chat closed';
    },
  });

  return <>{children}</>;
}

// =============================================================================
// INTERNAL CONTEXT PROVIDER (conditional CopilotKit hooks)
// =============================================================================

function E2ICopilotContextProvider({ children }: { children: React.ReactNode }) {
  const copilotEnabled = React.useContext(CopilotEnabledContext);

  return (
    <E2IBaseContextProvider>
      {copilotEnabled ? (
        <CopilotHooksConnector>{children}</CopilotHooksConnector>
      ) : (
        children
      )}
    </E2IBaseContextProvider>
  );
}

// =============================================================================
// COPILOTKIT WRAPPER (for main.tsx - outside router)
// =============================================================================

export interface CopilotKitWrapperProps {
  children: React.ReactNode;
  runtimeUrl?: string;
  enabled?: boolean;
}

/**
 * CopilotKitWrapper provides the CopilotKit context.
 * Use this in main.tsx to wrap the router.
 *
 * Set enabled=false to disable CopilotKit when backend is not available.
 *
 * @example
 * ```tsx
 * <CopilotKitWrapper runtimeUrl="/api/copilotkit" enabled={false}>
 *   <AppRouter />
 * </CopilotKitWrapper>
 * ```
 */
export function CopilotKitWrapper({
  children,
  runtimeUrl = '/api/copilotkit',
  enabled = false, // Disabled by default until backend is available
}: CopilotKitWrapperProps) {
  // In development without backend, just render children with disabled context
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
        transcribeAudioUrl="/api/transcribe"
        textToSpeechUrl="/api/tts"
      >
        {children}
      </CopilotKit>
    </CopilotEnabledContext.Provider>
  );
}

// =============================================================================
// MAIN PROVIDER (for router/index.tsx - inside router)
// =============================================================================

/**
 * E2ICopilotProvider provides E2I-specific context with readables and actions.
 * Must be used inside the router since it uses useNavigate/useLocation.
 *
 * @example
 * ```tsx
 * <E2ICopilotProvider initialFilters={{ brand: 'Remibrutinib' }}>
 *   <Layout><Outlet /></Layout>
 * </E2ICopilotProvider>
 * ```
 */
export function E2ICopilotProvider({
  children,
  initialFilters,
  userRole = 'analyst',
}: Omit<E2ICopilotProviderProps, 'runtimeUrl'>) {
  return (
    <E2ICopilotContextProvider>
      {children}
    </E2ICopilotContextProvider>
  );
}

export default E2ICopilotProvider;
