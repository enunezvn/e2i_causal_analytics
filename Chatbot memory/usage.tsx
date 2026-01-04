// ============================================================================
// E2I Causal Analytics - CopilotKit Integration Examples
// Demonstrates bidirectional state sync and agent actions
// ============================================================================

/**
 * INSTALLATION
 * ============
 * 
 * npm install @copilotkit/react-core @copilotkit/react-ui framer-motion
 * 
 * # If using LangGraph backend:
 * npm install @copilotkit/runtime
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  E2ICopilotProvider,
  useE2I,
} from '../providers/E2ICopilotProvider';
import { E2IChatSidebar, E2IChatPopup } from '../components/E2IChatSidebar';
import {
  useE2IChat,
  useE2IFilters,
  useE2IAgents,
  useE2IHighlights,
  useE2IValidation,
  useE2IDashboard,
} from '../hooks';
import { FilterContext, Brand, Region } from '../types';

// =============================================================================
// EXAMPLE 1: Basic Integration with Sidebar Chat
// =============================================================================

export function BasicExample() {
  return (
    <E2ICopilotProvider
      runtimeUrl="/api/copilotkit"
      initialFilters={{ brand: 'Remibrutinib' }}
      userRole="analyst"
    >
      <div className="min-h-screen bg-gray-100">
        <header className="bg-white border-b p-4">
          <h1 className="text-2xl font-bold">E2I Dashboard</h1>
        </header>
        
        <main className="p-6">
          {/* Your dashboard content */}
          <DashboardContent />
        </main>
        
        {/* Chat sidebar - agents can see dashboard state! */}
        <E2IChatSidebar defaultOpen={false} />
      </div>
    </E2ICopilotProvider>
  );
}

// =============================================================================
// EXAMPLE 2: Dashboard with Bidirectional State Sync
// =============================================================================

/**
 * KEY ADVANTAGE: useCopilotReadable makes app state visible to agents
 * 
 * When user changes filters in the dashboard, agents automatically know.
 * When agents update filters via actions, dashboard automatically reflects it.
 */

function DashboardContent() {
  const { filters, setBrand, setRegion } = useE2IFilters();
  const { highlightedPaths, isPathHighlighted, clearHighlights } = useE2IHighlights();
  const { validation, isValid, requiresReview } = useE2IValidation();
  const { activeTab, navigateToTab } = useE2IDashboard();

  return (
    <div className="space-y-6">
      {/* Filter Controls - synced with chat context */}
      <div className="bg-white rounded-lg p-4 shadow">
        <h2 className="font-semibold mb-3">Filters</h2>
        <div className="flex gap-4">
          <select
            value={filters.brand || ''}
            onChange={(e) => setBrand(e.target.value as Brand || undefined)}
            className="px-3 py-2 border rounded"
          >
            <option value="">All Brands</option>
            <option value="Remibrutinib">Remibrutinib</option>
            <option value="Fabhalta">Fabhalta</option>
            <option value="Kisqali">Kisqali</option>
          </select>
          
          <select
            value={filters.region || ''}
            onChange={(e) => setRegion(e.target.value as Region || undefined)}
            className="px-3 py-2 border rounded"
          >
            <option value="">All Regions</option>
            <option value="northeast">Northeast</option>
            <option value="south">South</option>
            <option value="midwest">Midwest</option>
            <option value="west">West</option>
          </select>
        </div>
        
        <p className="text-sm text-gray-500 mt-2">
          💡 Agents can see these filters and will adjust their analysis accordingly
        </p>
      </div>

      {/* Agent-highlighted causal paths */}
      {highlightedPaths.length > 0 && (
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <h3 className="font-semibold text-purple-800">
              ⚡ Agent-Highlighted Causal Paths
            </h3>
            <button
              onClick={clearHighlights}
              className="text-sm text-purple-600 hover:underline"
            >
              Clear
            </button>
          </div>
          <ul className="space-y-1">
            {highlightedPaths.map((path) => (
              <li key={path} className="text-sm text-purple-700">
                {path.replace('->', ' → ')}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Validation status */}
      {validation && (
        <div
          className={`rounded-lg p-4 ${
            isValid
              ? 'bg-green-50 border border-green-200'
              : requiresReview
              ? 'bg-amber-50 border border-amber-200'
              : 'bg-red-50 border border-red-200'
          }`}
        >
          <h3 className="font-semibold">
            {isValid ? '✓ Validated' : requiresReview ? '⚠ Needs Review' : '✕ Blocked'}
          </h3>
          <p className="text-sm">
            Confidence: {Math.round(validation.overallConfidence * 100)}%
          </p>
        </div>
      )}

      {/* Dashboard tabs - agents can navigate here */}
      <div className="bg-white rounded-lg shadow">
        <div className="border-b flex">
          {['overview', 'causal', 'validation', 'reports'].map((tab) => (
            <button
              key={tab}
              onClick={() => navigateToTab(tab)}
              className={`px-4 py-3 capitalize ${
                activeTab === tab
                  ? 'border-b-2 border-indigo-500 text-indigo-600'
                  : 'text-gray-500'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
        <div className="p-6">
          <TabContent tab={activeTab} />
        </div>
      </div>
    </div>
  );
}

function TabContent({ tab }: { tab: string }) {
  // Placeholder content
  return (
    <div className="text-gray-500">
      Content for <span className="font-semibold capitalize">{tab}</span> tab
      <p className="text-sm mt-2">
        💡 Ask the assistant: "Navigate to the causal analysis tab"
      </p>
    </div>
  );
}

// =============================================================================
// EXAMPLE 3: Agent Status Display
// =============================================================================

function AgentStatusPanel() {
  const { activeAgents, getActiveAgents, agentCounts } = useE2IAgents();

  const activeList = getActiveAgents();

  return (
    <div className="bg-white rounded-lg p-4 shadow">
      <h3 className="font-semibold mb-3">Agent Activity</h3>
      
      {/* Status counts */}
      <div className="flex gap-4 mb-4 text-sm">
        <span className="text-amber-600">
          🧠 {agentCounts.thinking} thinking
        </span>
        <span className="text-blue-600">
          ⚙️ {agentCounts.computing} computing
        </span>
        <span className="text-green-600">
          ✓ {agentCounts.complete} complete
        </span>
      </div>

      {/* Active agents */}
      {activeList.length > 0 && (
        <div className="space-y-2">
          {activeList.map((agent) => (
            <div
              key={agent.id}
              className="flex items-center gap-2 text-sm p-2 bg-gray-50 rounded"
            >
              <span>{agent.icon}</span>
              <span className="font-medium">{agent.name}</span>
              <span className="text-gray-400">•</span>
              <span className="text-gray-500 capitalize">
                {agent.state?.status}
              </span>
              {agent.state?.currentTask && (
                <span className="text-gray-400 text-xs truncate">
                  - {agent.state.currentTask}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {activeList.length === 0 && (
        <p className="text-gray-400 text-sm">No agents currently active</p>
      )}
    </div>
  );
}

// =============================================================================
// EXAMPLE 4: Chart with Agent Highlights
// =============================================================================

interface ChartWithHighlightsProps {
  chartId: string;
  data: Array<{ id: string; value: number; label: string }>;
}

function ChartWithHighlights({ chartId, data }: ChartWithHighlightsProps) {
  const { getChartHighlights, isChartElementHighlighted } = useE2IHighlights();
  
  const highlights = getChartHighlights(chartId);

  return (
    <div className="bg-white rounded-lg p-4 shadow">
      <h3 className="font-semibold mb-3">Performance Chart</h3>
      
      {/* Simple bar chart with highlights */}
      <div className="space-y-2">
        {data.map((item) => {
          const isHighlighted = isChartElementHighlighted(chartId, item.id);
          
          return (
            <div key={item.id} className="flex items-center gap-3">
              <span className="w-24 text-sm">{item.label}</span>
              <div className="flex-1 h-6 bg-gray-100 rounded overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${
                    isHighlighted
                      ? 'bg-amber-400 ring-2 ring-amber-500'
                      : 'bg-indigo-400'
                  }`}
                  style={{ width: `${item.value}%` }}
                />
              </div>
              <span className="w-12 text-right text-sm">{item.value}%</span>
              {isHighlighted && (
                <span className="text-amber-500 text-sm">← Agent highlight</span>
              )}
            </div>
          );
        })}
      </div>

      {highlights.length > 0 && (
        <p className="text-sm text-amber-600 mt-3">
          💡 The agent has highlighted {highlights.length} element(s) in this chart
        </p>
      )}
    </div>
  );
}

// =============================================================================
// EXAMPLE 5: Custom Action Handler (Report Generation)
// =============================================================================

function ReportGeneratorPanel() {
  const { pendingActions, dismissAction } = useE2I();
  
  const reportActions = pendingActions.filter((a) => a.type === 'generateReport');

  if (reportActions.length === 0) return null;

  return (
    <div className="bg-white rounded-lg p-4 shadow">
      <h3 className="font-semibold mb-3">📄 Pending Reports</h3>
      
      {reportActions.map((action, idx) => {
        if (action.type !== 'generateReport') return null;
        
        return (
          <div
            key={idx}
            className="p-3 bg-gray-50 rounded mb-2 flex justify-between items-center"
          >
            <div>
              <span className="font-medium uppercase">{action.format}</span>
              <span className="text-gray-400 mx-2">•</span>
              <span className="text-sm text-gray-500">
                {action.sections.join(', ')}
              </span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  // Generate report logic here
                  console.log('Generating report:', action);
                  dismissAction(idx);
                }}
                className="px-3 py-1 bg-indigo-500 text-white text-sm rounded hover:bg-indigo-600"
              >
                Generate
              </button>
              <button
                onClick={() => dismissAction(idx)}
                className="px-3 py-1 bg-gray-200 text-gray-700 text-sm rounded hover:bg-gray-300"
              >
                Dismiss
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// =============================================================================
// EXAMPLE 6: Popup Chat (Alternative to Sidebar)
// =============================================================================

export function PopupChatExample() {
  return (
    <E2ICopilotProvider runtimeUrl="/api/copilotkit">
      <div className="min-h-screen bg-gray-100 p-8">
        <h1 className="text-3xl font-bold mb-6">E2I Dashboard</h1>
        
        <p className="text-gray-600 mb-4">
          Press <kbd className="px-2 py-1 bg-gray-200 rounded">⌘/</kbd> to open chat
        </p>
        
        {/* Your dashboard content */}
        <DashboardContent />
        
        {/* Popup chat - keyboard shortcut activated */}
        <E2IChatPopup shortcut="mod+/" />
      </div>
    </E2ICopilotProvider>
  );
}

// =============================================================================
// EXAMPLE 7: Full Dashboard with All Features
// =============================================================================

export function FullDashboardExample() {
  return (
    <E2ICopilotProvider
      runtimeUrl="/api/copilotkit"
      initialFilters={{
        brand: 'Remibrutinib',
        region: 'south',
      }}
      userRole="analyst"
    >
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white border-b shadow-sm">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-2xl">🔬</span>
                <h1 className="text-xl font-bold">E2I Causal Analytics</h1>
              </div>
              <AgentStatusPanel />
            </div>
          </div>
        </header>

        {/* Main content */}
        <main className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-3 gap-6">
            {/* Left column - filters and charts */}
            <div className="col-span-2 space-y-6">
              <DashboardContent />
              <ChartWithHighlights
                chartId="performance-chart"
                data={[
                  { id: 'q1', value: 75, label: 'Q1' },
                  { id: 'q2', value: 82, label: 'Q2' },
                  { id: 'q3', value: 68, label: 'Q3' },
                  { id: 'q4', value: 91, label: 'Q4' },
                ]}
              />
            </div>

            {/* Right column - reports */}
            <div className="space-y-6">
              <ReportGeneratorPanel />
            </div>
          </div>
        </main>

        {/* Chat sidebar */}
        <E2IChatSidebar defaultOpen={true} />
      </div>
    </E2ICopilotProvider>
  );
}

export default FullDashboardExample;
