import { Page, Route } from '@playwright/test'

/**
 * API mock data and route interception utilities.
 * Use these to simulate backend responses in E2E tests.
 */

// ============================================================================
// Mock Data
// ============================================================================

export const mockKPIs = {
  trx_volume: 125000,
  nrx_volume: 45000,
  market_share: 23.5,
  conversion_rate: 18.2,
  hcp_reach: 8500,
  patient_starts: 3200,
}

export const mockAgents = [
  { id: 'orchestrator', name: 'Orchestrator', tier: 1, status: 'active', description: 'Query routing' },
  { id: 'causal_impact', name: 'Causal Impact', tier: 2, status: 'active', description: 'Causal analysis' },
  { id: 'gap_analyzer', name: 'Gap Analyzer', tier: 2, status: 'active', description: 'ROI detection' },
  { id: 'drift_monitor', name: 'Drift Monitor', tier: 3, status: 'idle', description: 'Drift detection' },
  { id: 'explainer', name: 'Explainer', tier: 5, status: 'active', description: 'Explanations' },
  { id: 'feedback_learner', name: 'Feedback Learner', tier: 5, status: 'idle', description: 'Learning' },
]

export const mockCausalGraph = {
  nodes: [
    { id: 'trx_volume', label: 'TRx Volume', type: 'outcome' },
    { id: 'hcp_visits', label: 'HCP Visits', type: 'treatment' },
    { id: 'market_share', label: 'Market Share', type: 'outcome' },
    { id: 'conversion_rate', label: 'Conversion Rate', type: 'mediator' },
  ],
  edges: [
    { source: 'hcp_visits', target: 'conversion_rate', weight: 0.65 },
    { source: 'conversion_rate', target: 'trx_volume', weight: 0.82 },
    { source: 'trx_volume', target: 'market_share', weight: 0.71 },
  ],
}

export const mockHealthStatus = {
  overall: 'healthy',
  components: {
    api: 'operational',
    database: 'operational',
    ml_models: 'operational',
    redis: 'operational',
    falkordb: 'operational',
  },
  timestamp: new Date().toISOString(),
}

export const mockKnowledgeGraph = {
  nodes: [
    { id: '1', label: 'Remibrutinib', type: 'brand' },
    { id: '2', label: 'TRx Volume', type: 'kpi' },
    { id: '3', label: 'HCP Engagement', type: 'factor' },
    { id: '4', label: 'Patient Starts', type: 'kpi' },
  ],
  edges: [
    { from: '1', to: '2', label: 'drives' },
    { from: '3', to: '2', label: 'influences' },
    { from: '2', to: '4', label: 'correlates' },
  ],
  stats: {
    total_nodes: 4,
    total_edges: 3,
    node_types: { brand: 1, kpi: 2, factor: 1 },
  },
}

export const mockFeatureImportance = [
  { feature: 'hcp_visits', importance: 0.35, shap_value: 0.42 },
  { feature: 'marketing_spend', importance: 0.28, shap_value: 0.31 },
  { feature: 'patient_demographics', importance: 0.18, shap_value: 0.22 },
  { feature: 'competition_activity', importance: 0.12, shap_value: 0.15 },
  { feature: 'seasonality', importance: 0.07, shap_value: 0.09 },
]

// ============================================================================
// Mock Route Handlers
// ============================================================================

export async function mockApiRoutes(page: Page): Promise<void> {
  // CopilotKit info endpoint - MUST be mocked first to prevent CopilotKit errors
  await page.route('**/api/copilotkit/info**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        agents: [
          { name: 'default', description: 'Default agent' },
          { name: 'orchestrator', description: 'Query routing' },
          { name: 'causal_impact', description: 'Causal analysis' },
        ],
        actions: [],
        copilotReadable: [],
      }),
    })
  })

  // CopilotKit main endpoint
  await page.route('**/api/copilotkit', async (route: Route) => {
    const request = route.request()
    if (request.method() === 'POST') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          threadId: 'test-thread-id',
          messages: [],
          agentState: {},
        }),
      })
    } else {
      await route.continue()
    }
  })

  // Health endpoints
  await page.route('**/health', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'healthy', version: '4.1.0' }),
    })
  })

  await page.route('**/ready', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'ready', checks: mockHealthStatus.components }),
    })
  })

  // KPI endpoints
  await page.route('**/api/copilotkit/kpis**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ kpis: mockKPIs, data_source: 'mock' }),
    })
  })

  // Agent status endpoints
  await page.route('**/api/copilotkit/agents**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ agents: mockAgents, data_source: 'mock' }),
    })
  })

  // Knowledge graph endpoints
  await page.route('**/api/graph/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockKnowledgeGraph),
    })
  })

  // Causal analysis endpoints
  await page.route('**/api/causal/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        causal_graph: mockCausalGraph,
        effect_estimate: 0.42,
        confidence_interval: [0.35, 0.49],
      }),
    })
  })

  // Feature importance endpoints
  await page.route('**/api/explain/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ features: mockFeatureImportance }),
    })
  })

  // Memory/RAG endpoints
  await page.route('**/api/memory/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ results: [], total: 0 }),
    })
  })

  await page.route('**/api/rag/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ results: [], metadata: {} }),
    })
  })
}

/**
 * Setup mock routes that simulate error responses.
 */
export async function mockApiErrors(page: Page): Promise<void> {
  await page.route('**/api/**', async (route: Route) => {
    await route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ error: 'Internal server error', message: 'Simulated error' }),
    })
  })
}

/**
 * Setup mock routes that simulate slow responses.
 */
export async function mockSlowResponses(page: Page, delayMs = 2000): Promise<void> {
  await page.route('**/api/**', async (route: Route) => {
    await new Promise(resolve => setTimeout(resolve, delayMs))
    await route.continue()
  })
}
