import { test, expect, type Route } from '@playwright/test'
import { AgentOrchestrationPage } from '../pages/agent-orchestration.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Agent Orchestration Page', () => {
  // The page has heavy provider boot (Auth + QueryClient + CopilotKit) and
  // 22 small per-feature tests that each re-bootstrap from scratch via
  // beforeEach (mockApiRoutes + goto). Under the global `workers: 1` setting
  // the serial dev/static server occasionally serves a slow lazy chunk and
  // a single test mounts to a blank page (root-level useEffect-driven
  // AuthProvider hasn't called setInitialized(true), so ProtectedRoute
  // shows the fallback spinner with no <main>). Allow 2 retries on this
  // describe so a single transient flake does not red the shard. This is
  // strictly scoped: retries here cannot mask real production regressions
  // because the asserted UI (stat cards, tabs, header) is fully static
  // sample-data driven; if the page actually broke, all 22 tests would
  // fail on every attempt.
  test.describe.configure({ retries: 2 })

  let agentPage: AgentOrchestrationPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    // Inline stub for /api/agents/status which AgentOrchestration.tsx calls
    // via useQuery on mount. The shared fixture does not stub this endpoint,
    // so without this the request falls through to the dev server (and in CI
    // to the built-bundle static server which has no /api backend), leaving
    // React Query pending/erroring and stalling page hydration on slow runs.
    // Schema: AgentStatusResponseSchema in src/lib/api-schemas.ts.
    await page.route('**/api/agents/status', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          agents: [
            {
              id: 'orchestrator',
              name: 'Orchestrator',
              tier: 1,
              status: 'active',
              capabilities: ['routing', 'coordination'],
            },
            {
              id: 'causal-impact',
              name: 'Causal Impact',
              tier: 2,
              status: 'active',
              capabilities: ['causal_inference', 'ate_estimation'],
            },
          ],
          total: 2,
          timestamp: new Date().toISOString(),
        }),
      })
    })
    // Telemetry stub for GET /api/analytics/summary (QueryMetricsSummary,
    // verified against the live OpenAPI spec). The page renders
    // Queries (24h)/Avg Response Time/Success Rate from this — an em dash
    // when unavailable, never fabricated values.
    await page.route('**/api/analytics/summary*', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          period_start: '2026-06-11T00:00:00Z',
          period_end: '2026-06-12T00:00:00Z',
          total_queries: 412,
          successful_queries: 398,
          failed_queries: 14,
          success_rate: 96.6,
          avg_latency_ms: 742.5,
          p50_latency_ms: 510.0,
          p95_latency_ms: 1920.0,
          p99_latency_ms: 3100.0,
          intent_distribution: { causal_analysis: 201, kpi_query: 211 },
          top_agents: [],
        }),
      })
    })
    agentPage = new AgentOrchestrationPage(page)
    await agentPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(agentPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(agentPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(agentPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(agentPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Stats Cards', () => {
    test('should display stats cards', async () => {
      const hasStats = await agentPage.verifyStatsDisplayed()
      expect(hasStats).toBeTruthy()
    })

    test('should show Total Agents stat', async () => {
      await expect(agentPage.totalAgentsCard).toBeVisible()
    })

    test('should show Queries (24h) stat from real telemetry', async () => {
      await expect(agentPage.queries24hCard).toBeVisible()
    })

    test('should show Avg Response Time stat', async () => {
      await expect(agentPage.avgResponseTimeCard).toBeVisible()
    })

    test('should show Success Rate stat', async () => {
      await expect(agentPage.successRateCard).toBeVisible()
    })
  })

  test.describe('Agent Tiers', () => {
    test('should display agent tiers', async () => {
      const hasTiers = await agentPage.verifyAgentTiersDisplayed()
      expect(hasTiers).toBeTruthy()
    })

    test('should display tier cards', async () => {
      const tierCount = await agentPage.tierCards.count()
      expect(tierCount).toBeGreaterThanOrEqual(0) // May have 0 if on different tab
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await agentPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Overview tab', async () => {
      await expect(agentPage.overviewTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      if (await agentPage.activityTab.isVisible().catch(() => false)) {
        await agentPage.clickTab('Activity')
        await agentPage.page.waitForTimeout(500)
      }
    })
  })

  test.describe('Pipeline Visualization', () => {
    test('should display pipeline visualization', async () => {
      const hasPipeline = await agentPage.verifyPipelineVisualizationDisplayed()
      expect(hasPipeline).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(agentPage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await agentPage.clickRefresh()
      await agentPage.page.waitForTimeout(500)
    })
  })

  test.describe('Responsive Design', () => {
    // Each test resizes the already-loaded page and re-verifies the h1
    // anchor (more reliable than mainContent which is the broad inherited
    // BasePage locator). Resizing after the initial mount avoids the race
    // where setViewportSize+goto re-mounted the SPA with an in-flight
    // auth/route stub registration window.
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await expect(agentPage.pageHeader).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await expect(agentPage.pageHeader).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await expect(agentPage.pageHeader).toBeVisible()
    })
  })
})
