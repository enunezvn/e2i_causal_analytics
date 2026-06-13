/**
 * Analytics Page E2E Tests (#19 coverage gap)
 * ===========================================
 *
 * `/analytics` was a routed data page with NO e2e coverage. It is the
 * agent-performance & query-analytics dashboard. These specs stub the REAL
 * endpoint the page calls (`GET /api/analytics/dashboard`) and assert HONEST
 * states:
 *   - real dashboard -> KPI cards reflect the live summary
 *   - endpoint error -> "Failed to load analytics" (labeled error, the page
 *     returns the error view WITHOUT the normal header — so we navigate plainly
 *     and wait for the error copy)
 *
 * The error stub uses a 400 (client error) so react-query's shouldRetry does
 * NOT retry it (4xx are not retried), giving a deterministic, fast error
 * state instead of a multi-second backoff on a 5xx.
 */

import { test, expect, type Page, type Route } from '@playwright/test'
import { AnalyticsPage } from '../pages/analytics.page'
import { harnessBase } from '../fixtures/page-harness'

const now = new Date().toISOString()

const DASHBOARD = {
  summary: {
    period_start: now,
    period_end: now,
    total_queries: 717,
    successful_queries: 700,
    failed_queries: 17,
    success_rate: 97.6,
    avg_latency_ms: 1840,
    p50_latency_ms: 1500,
    p95_latency_ms: 3200,
    p99_latency_ms: 5100,
    intent_distribution: { causal: 2400, gap: 1842 },
    top_agents: ['orchestrator', 'causal_impact'],
  },
  agent_metrics: [
    {
      agent_name: 'orchestrator',
      agent_tier: 1,
      total_invocations: 4242,
      successful_invocations: 4100,
      failed_invocations: 142,
      success_rate: 96.7,
      avg_latency_ms: 1840,
      p50_latency_ms: 1500,
      p95_latency_ms: 3200,
      p99_latency_ms: 5100,
      min_latency_ms: 320,
      max_latency_ms: 8200,
      avg_confidence: 0.87,
    },
  ],
  latency_trend: [{ timestamp: now, value: 1840 }],
  query_volume_trend: [{ timestamp: now, value: 4242 }],
  latency_breakdown: {
    classification_ms: 120,
    rag_retrieval_ms: 400,
    routing_ms: 80,
    agent_dispatch_ms: 900,
    synthesis_ms: 340,
    total_ms: 1840,
  },
  generated_at: now,
}

async function stubAnalyticsDashboard(
  page: Page,
  opts: { status?: number; body?: unknown } = {},
): Promise<void> {
  await page.route('**/api/analytics/dashboard**', async (route: Route) => {
    if (opts.status && opts.status >= 400) {
      await route.fulfill({
        status: opts.status,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'analytics service error' }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(opts.body ?? DASHBOARD),
    })
  })
}

test.describe('Analytics Page', () => {
  let analyticsPage: AnalyticsPage

  test.describe('Loaded state', () => {
    test.beforeEach(async ({ page }) => {
      await harnessBase(page)
      await stubAnalyticsDashboard(page)
      analyticsPage = new AnalyticsPage(page)
      await analyticsPage.goto()
    })

    test('loads at /analytics', async ({ page }) => {
      await expect(page).toHaveURL(/analytics/)
    })

    test('displays the page header', async () => {
      await expect(analyticsPage.pageHeader).toBeVisible()
    })

    test('renders the KPI cards', async () => {
      await expect(analyticsPage.totalQueriesCard).toBeVisible()
      await expect(analyticsPage.avgLatencyCard).toBeVisible()
    })

    test('reflects the real total_queries from the live dashboard (falsifiability)', async ({
      page,
    }) => {
      // formatNumber renders sub-1000 values verbatim, so total_queries=717
      // appears as "717". A unique value proves the KPI is driven by the live
      // dashboard, not a hard-coded constant.
      await expect(page.getByText('717', { exact: true }).first()).toBeVisible()
    })
  })

  test.describe('Error state', () => {
    test('shows a labeled error when the dashboard endpoint fails', async ({ page }) => {
      await harnessBase(page)
      // 400 is NOT retried by shouldRetry -> deterministic, fast error view.
      await stubAnalyticsDashboard(page, { status: 400 })
      analyticsPage = new AnalyticsPage(page)
      // The error view returns early WITHOUT the normal header, so navigate
      // plainly and wait for the labeled error copy instead of the heading.
      await page.goto('/analytics')
      await expect(analyticsPage.errorState).toBeVisible({ timeout: 15000 })
    })
  })
})
