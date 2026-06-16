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

// Data Quality wiring mocks (issue #301): KPI list + workstreams + per-KPI
// metadata/value + drift status/history. These match the FastAPI surfaces at
// /api/kpis/* and /api/monitoring/drift/* via the api-client baseURL.
export const mockKPIList = {
  kpis: [
    {
      id: 'WS1-DQ-001',
      name: 'Source Coverage - Patients',
      definition: 'Percentage of eligible patients present in source vs reference universe',
      formula: 'covered_patients / reference_patients',
      calculation_type: 'direct',
      workstream: 'ws1_data_quality',
      tables: ['patient_journeys'],
      columns: ['patient_id', 'coverage_status'],
      threshold: { target: 85, warning: 70, critical: 50 },
      unit: '%',
      frequency: 'daily',
      primary_causal_library: 'none',
    },
    {
      id: 'WS1-DQ-002',
      name: 'Completeness - HCP Master',
      definition: 'HCP master record completeness',
      formula: 'non_null_hcp / total_hcp',
      calculation_type: 'direct',
      workstream: 'ws1_data_quality',
      tables: ['hcp_master'],
      columns: ['npi'],
      threshold: { target: 98, warning: 90, critical: 80 },
      unit: '%',
      frequency: 'daily',
      primary_causal_library: 'none',
    },
  ],
  total: 2,
  workstream: 'ws1_data_quality',
}

export const mockDriftLatest = {
  task_id: 'mock-task-1',
  model_id: 'data_quality_pipeline',
  status: 'completed',
  overall_drift_score: 0.08,
  features_checked: 8,
  features_with_drift: ['npi'],
  results: [],
  drift_summary: '1 feature shows drift',
  recommended_actions: [],
  detection_latency_ms: 100,
  timestamp: '2026-01-02T08:00:00Z',
}

export const mockDriftHistory = {
  model_id: 'data_quality_pipeline',
  total_records: 0,
  records: [],
}

// ============================================================================
// Mock Route Handlers
// ============================================================================

/**
 * Seed a fake authenticated Supabase session in localStorage BEFORE the SPA boots.
 *
 * Many pages are gated by <ProtectedRoute>, which redirects to /login when
 * useIsAuthenticated() returns false (== no session.access_token + no user).
 * Without seeding, every protected page e2e test redirects and the selectors
 * never match. This is the root cause behind issue #306's "DataProfilingTab not
 * visible" failures — the page never renders.
 *
 * The seeded session matches the Zustand store's `partialize` shape ({session,
 * redirectTo}) and the Supabase client's `Session` type minimally. Tests that
 * don't drive auth-flow (i.e. all DataQuality e2e tests) only need the session
 * to exist for ProtectedRoute to allow render-through.
 */
async function seedAuthSession(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const now = new Date().toISOString()
    const fakeUser = {
      id: 'e2e-mock-user-id',
      aud: 'authenticated',
      role: 'authenticated',
      email: 'e2e@test.local',
      email_confirmed_at: now,
      phone: '',
      confirmed_at: now,
      last_sign_in_at: now,
      app_metadata: { provider: 'email' },
      user_metadata: { full_name: 'E2E User' },
      identities: [],
      created_at: now,
      updated_at: now,
    }
    const fakeSession = {
      access_token: 'e2e-mock-access-token',
      refresh_token: 'e2e-mock-refresh-token',
      expires_in: 3600,
      expires_at: Math.floor(Date.now() / 1000) + 3600,
      token_type: 'bearer',
      user: fakeUser,
    }
    try {
      // (1) Supabase auth-js storage (drives `supabase.auth.getSession()` which
      // AuthProvider calls on mount). The configured `storageKey` is `e2i-auth-token`
      // (see src/lib/supabase.ts:83). Without this, getSession() returns null and
      // AuthProvider calls clearAuth(), wiping any seeded Zustand state.
      window.localStorage.setItem(
        'e2i-auth-token',
        JSON.stringify({
          ...fakeSession,
          // Supabase storage uses a flat session object (not wrapped in {state})
        })
      )

      // (2) Zustand auth-store. Include `user` so the persist-rehydrate merge
      // populates both `session` AND `user` — useIsAuthenticated() requires
      // both to be truthy. (partialize officially only persists `session`+
      // `redirectTo`, but Zustand persist's `setState(persistedState)` is a
      // plain merge, so extra keys hydrate into store state on boot.)
      window.localStorage.setItem(
        'e2i-auth-store',
        JSON.stringify({
          state: {
            session: fakeSession,
            user: fakeUser,
            isInitialized: true,
            isLoading: false,
            redirectTo: null,
          },
          version: 0,
        })
      )
    } catch {
      // localStorage may not be available in some contexts; ignore
    }
  })

  // Block real Supabase network calls (the SPA's AuthProvider calls
  // supabase.auth.getSession() on mount; without interception it may overwrite
  // the seeded session with a null/empty response from the configured Supabase
  // URL, returning the user to /login. We respond with our seeded session.
  await page.route('**/auth/v1/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        access_token: 'e2e-mock-access-token',
        refresh_token: 'e2e-mock-refresh-token',
        token_type: 'bearer',
        expires_in: 3600,
        user: {
          id: 'e2e-mock-user-id',
          email: 'e2e@test.local',
          aud: 'authenticated',
          role: 'authenticated',
        },
      }),
    })
  })
}

export async function mockApiRoutes(page: Page): Promise<void> {
  // Seed auth BEFORE any routes so the SPA bootstrap finds a session.
  await seedAuthSession(page)

  // CopilotKit info endpoint - MUST be mocked first to prevent CopilotKit errors.
  // The CopilotKit runtime returns `agents` as a DICT keyed by name (see
  // @copilotkitnext/runtime handleGetRuntimeInfo → `agentsDict`), NOT an array.
  // Using an array breaks `useAgent: Agent 'default' not found` and crashes every
  // page that mounts E2ICopilotProvider — which is the root cause of issue #306.
  await page.route('**/api/copilotkit/info**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        version: '1.0.0',
        agents: {
          default: { name: 'default', description: 'Default agent', className: 'MockAgent' },
          orchestrator: { name: 'orchestrator', description: 'Query routing', className: 'MockAgent' },
          causal_impact: { name: 'causal_impact', description: 'Causal analysis', className: 'MockAgent' },
        },
        actions: [],
        copilotReadable: [],
        audioFileTranscriptionEnabled: false,
      }),
    })
  })

  // CopilotKit main endpoint — JSON-RPC multiplexer keyed on `method` in body.
  // Modern CopilotKit (v1+) POSTs {"method":"info",...} here for agent discovery
  // rather than GET /info; the prior mock returned a thread-response shape for ALL
  // POSTs, which caused `useAgent: Agent 'default' not found after runtime sync`
  // on every page (root cause of issue #306's e2e shard failures).
  await page.route('**/api/copilotkit', async (route: Route) => {
    const request = route.request()
    if (request.method() === 'POST') {
      let method = ''
      try {
        const body = request.postDataJSON() as { method?: string } | null
        method = body?.method ?? ''
      } catch {
        method = ''
      }

      if (method === 'info') {
        // CopilotKitCoreRuntime.runtimeInfo shape: agents dict keyed by name,
        // matching @copilotkitnext/runtime handleGetRuntimeInfo (agentsDict).
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            version: '1.0.0',
            agents: {
              default: { name: 'default', description: 'Default agent', className: 'MockAgent' },
              orchestrator: { name: 'orchestrator', description: 'Query routing', className: 'MockAgent' },
              causal_impact: { name: 'causal_impact', description: 'Causal analysis', className: 'MockAgent' },
            },
            actions: [],
            copilotReadable: [],
            audioFileTranscriptionEnabled: false,
          }),
        })
        return
      }

      // Default: thread / run response (legacy shape, harmless for tests that
      // don't actually drive the agent loop).
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

  // Live KPI list endpoint (used by DataQuality page via useKPIList).
  // ORDER MATTERS: more specific routes registered before catch-alls. Playwright
  // dispatches routes in registration order, so we register /kpis and /monitoring
  // explicitly before any generic /api/** route would shadow them.
  await page.route('**/api/kpis**', async (route: Route) => {
    const url = route.request().url()
    // Per-KPI metadata or value
    const kpiMatch = url.match(/\/api\/kpis\/([^/?]+)(?:\/value)?/)
    if (kpiMatch && !url.endsWith('/workstreams') && !url.endsWith('/health')) {
      const kpiId = kpiMatch[1]
      const kpi = mockKPIList.kpis.find((k) => k.id === kpiId) ?? mockKPIList.kpis[0]
      if (url.includes('/value')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            kpi_id: kpiId,
            value: 92.5,
            status: 'good',
            calculated_at: new Date().toISOString(),
            cached: false,
            metadata: {},
          }),
        })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(kpi),
      })
      return
    }
    // Workstreams listing
    if (url.endsWith('/workstreams')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ workstreams: [{ id: 'ws1_data_quality', name: 'WS1: Data Quality', kpi_count: 2 }], total: 1 }),
      })
      return
    }
    // Default: list
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockKPIList),
    })
  })

  // Drift detection endpoints (used by DataQuality page via use-monitoring)
  await page.route('**/api/monitoring/drift/latest/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockDriftLatest),
    })
  })

  await page.route('**/api/monitoring/drift/history/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockDriftHistory),
    })
  })

  await page.route('**/api/monitoring/drift/detect', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockDriftLatest),
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

  // --------------------------------------------------------------------------
  // Home-page tile sources (populate fabricated Home features with real data).
  // ORDER MATTERS: Playwright matches the MOST-RECENTLY-registered route first,
  // so these specific routes are registered LAST (here, at the end of
  // mockApiRoutes) so they win over the broad catch-alls registered above:
  //   - /api/copilotkit/kpis/summary  must beat  /api/copilotkit/kpis**
  //   - /api/kpis/batch + /api/kpis/WS1-MP-001  must beat  /api/kpis**
  // --------------------------------------------------------------------------

  // QUICK_STATS rollup (Total TRx (MTD), HCPs Reached). Real shape from
  // GET /api/copilotkit/kpis/summary (data_source 'database' = real DB values).
  await page.route('**/api/copilotkit/kpis/summary**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        brand: 'All',
        period: 'Last 90 days',
        metrics: {
          trx_volume: 125000,
          nrx_volume: 45000,
          market_share: 23.5,
          conversion_rate: 18.2,
          hcp_reach: 8500,
          patient_starts: 3200,
        },
        data_source: 'database',
      }),
    })
  })

  // Active Campaigns = count of running experiments (GET /api/experiments/active-count).
  await page.route('**/api/experiments/active-count**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ active_count: 12 }),
    })
  })

  // System Health card + Model Health (GET /health-score/full).
  // The Home System Health card uses the FULL (all-dimension) check via
  // useFullHealthCheck() — it renders Components/Models/Pipelines/Agents rows
  // from the four per-dimension scores, omitting any that are null (unmeasured).
  // Per-dimension scores are 0-1 fractions; overall is 0-100; grade A is
  // self-consistent with these scores. Detail arrays mirror the schema-verified
  // shape used in ai-insights.spec.ts (HealthScoreResponseWireSchema).
  await page.route('**/health-score/full**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        check_id: 'e2e-health-1',
        check_scope: 'full',
        overall_health_score: 92,
        health_grade: 'A',
        component_health_score: 0.95,
        model_health_score: 0.88,
        pipeline_health_score: 0.82,
        agent_health_score: 0.92,
        component_statuses: [],
        model_metrics: [],
        pipeline_statuses: [],
        agent_statuses: [],
        critical_issues: [],
        warnings: [],
        recommendations: [],
        health_summary: 'All systems operational',
        check_latency_ms: 120,
        timestamp: new Date().toISOString(),
        data_provenance: 'measured',
      }),
    })
  })

  // Agent Status card (GET /api/agents/status). FE derives per-tier counts.
  await page.route('**/api/agents/status**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        agents: [
          { id: 'scope_definer', name: 'Scope Definer', tier: 0, status: 'idle', capabilities: ['scoping'] },
          { id: 'orchestrator', name: 'Orchestrator', tier: 1, status: 'active', capabilities: ['routing'] },
          { id: 'causal_impact', name: 'Causal Impact', tier: 2, status: 'active', capabilities: ['causal'] },
          { id: 'gap_analyzer', name: 'Gap Analyzer', tier: 2, status: 'active', capabilities: ['roi'] },
          { id: 'drift_monitor', name: 'Drift Monitor', tier: 3, status: 'idle', capabilities: ['drift'] },
        ],
        total: 5,
        timestamp: new Date().toISOString(),
      }),
    })
  })

  // KPI batch values (POST /api/kpis/batch) — drives the KPI Overview cards.
  await page.route('**/api/kpis/batch**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        workstream: null,
        total_kpis: 2,
        successful: 1,
        failed: 1,
        calculated_at: new Date().toISOString(),
        results: [
          { kpi_id: 'WS1-DQ-001', value: 92.5, status: 'good', error: null, calculated_at: new Date().toISOString(), cached: false, metadata: {} },
          // A view-less KPI returns null/error → honest "Not yet computed".
          { kpi_id: 'WS1-DQ-002', value: null, status: 'unknown', error: 'not_computed', calculated_at: new Date().toISOString(), cached: false, metadata: {} },
        ],
      }),
    })
  })

  // Model Accuracy ROC-AUC (GET /api/kpis/WS1-MP-001) — real value ~0.80.
  await page.route('**/api/kpis/WS1-MP-001**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        kpi_id: 'WS1-MP-001',
        value: 0.7998,
        status: 'good',
        calculated_at: new Date().toISOString(),
        cached: false,
        metadata: {},
      }),
    })
  })

  // AI Insights — Source A: executive insights (empty today is the honest state;
  // return one item so the populated path is exercised in e2e).
  await page.route('**/api/executive-insights**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          insight_id: 'i1',
          title: 'High-Value Territory Opportunity',
          narrative: 'Northeast shows untapped CSU potential.',
          brand: 'All',
          region: 'northeast',
          kpi: 'trx',
          crystallized_at: new Date().toISOString(),
          source_count: 3,
          effect_size: 0.23,
          effect_direction: 'positive',
          recommended_next_analysis: 'Increase HCP engagement',
        },
      ]),
    })
  })

  // AI Insights — Source B: gap opportunities (OpportunityListResponse).
  await page.route('**/api/gaps/opportunities**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        total_count: 0,
        quick_wins_count: 0,
        strategic_bets_count: 0,
        opportunities: [],
        total_addressable_value: 0,
      }),
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
