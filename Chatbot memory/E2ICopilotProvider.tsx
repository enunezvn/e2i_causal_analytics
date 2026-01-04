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
  useEffect,
  ReactNode,
} from 'react';
import { CopilotKit } from '@copilotkit/react-core';
import {
  useCopilotReadable,
  useCopilotAction,
  useCopilotChat,
} from '@copilotkit/react-core';
import {
  FilterContext,
  DashboardContext,
  ValidationSummary,
  AgentState,
  AGENT_REGISTRY,
  E2ICopilotAction,
} from '../types';

// -----------------------------------------------------------------------------
// E2I Context Types
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
  highlightedCharts: Map<string, string[]>;
  pendingActions: E2ICopilotAction[];
  
  // Action handlers
  clearHighlights: () => void;
  dismissAction: (index: number) => void;
}

const E2IContext = createContext<E2IContextType | null>(null);

// -----------------------------------------------------------------------------
// Hook to access E2I context
// -----------------------------------------------------------------------------

export function useE2I() {
  const context = useContext(E2IContext);
  if (!context) {
    throw new Error('useE2I must be used within E2ICopilotProvider');
  }
  return context;
}

// -----------------------------------------------------------------------------
// Main Provider Component
// -----------------------------------------------------------------------------

interface E2ICopilotProviderProps {
  children: ReactNode;
  apiKey?: string;
  runtimeUrl?: string;
  initialFilters?: FilterContext;
  userRole?: DashboardContext['userRole'];
}

export const E2ICopilotProvider: React.FC<E2ICopilotProviderProps> = ({
  children,
  apiKey,
  runtimeUrl = '/api/copilotkit',
  initialFilters = {},
  userRole = 'analyst',
}) => {
  return (
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
    render: ({ agentId, status }) => {
      // Custom render for agent status updates
      const agent = AGENT_REGISTRY[agentId];
      if (!agent) return null;
      
      return (
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <span>{agent.icon}</span>
          <span>{agent.name}</span>
          <span className="text-xs">({status})</span>
        </div>
      );
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
    highlightedCharts,
    pendingActions,
    clearHighlights,
    dismissAction,
  };

  return (
    <E2IContext.Provider value={contextValue}>{children}</E2IContext.Provider>
  );
};

export default E2ICopilotProvider;
