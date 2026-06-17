/**
 * Experiments Page E2E Tests (#19 coverage gap)
 * =============================================
 *
 * `/experiments` was a routed data page with NO e2e coverage. It is the A/B
 * testing & experiment-monitoring dashboard. Experiments are derived from live
 * monitor data ONLY — no sample fallback. These specs assert HONEST states:
 *   - on load (no monitoring run) -> EmptyState "No experiments loaded yet"
 *   - after "Run Monitoring" -> experiment cards from the REAL monitor response
 *     (`POST /api/experiments/monitor`)
 *
 * We do NOT assert against fabricated experiment data — the empty state IS the
 * honest contract before the monitoring service is queried.
 */

import { test, expect, type Page, type Route } from '@playwright/test'
import { ExperimentsPage } from '../pages/experiments.page'
import { harnessBase } from '../fixtures/page-harness'

const LIVE_EXP_NAME = 'E2E-LIVE-EXP-7c1d4e'

const MONITOR_RESPONSE = {
  experiments_checked: 1,
  healthy_count: 1,
  warning_count: 0,
  critical_count: 0,
  experiments: [
    {
      experiment_id: 'exp-e2e-1',
      experiment_name: LIVE_EXP_NAME,
      health_status: 'healthy',
      total_enrolled: 1820,
      enrollment_rate_per_day: 65.0,
      current_information_fraction: 0.42,
      has_srm: false,
      active_alerts: 0,
      last_checked: new Date().toISOString(),
    },
  ],
  alerts: [],
  monitor_summary: '1 experiment checked, all healthy',
  errors: [],
  recommended_actions: [],
  check_latency_ms: 240,
  timestamp: new Date().toISOString(),
}

async function stubMonitorEndpoint(page: Page, body: unknown): Promise<void> {
  await page.route('**/api/experiments/monitor**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body),
    })
  })
}

test.describe('Experiments Page', () => {
  let expPage: ExperimentsPage

  test.beforeEach(async ({ page }) => {
    await harnessBase(page)
    expPage = new ExperimentsPage(page)
    await expPage.goto()
  })

  test.describe('Page Load', () => {
    test('loads at /experiments', async ({ page }) => {
      await expect(page).toHaveURL(/experiments/)
    })

    test('displays the page header', async () => {
      await expect(expPage.pageHeader).toBeVisible()
    })

    test('displays the page description', async () => {
      await expect(expPage.pageDescription).toBeVisible()
    })

    test('renders the KPI overview cards', async () => {
      await expect(expPage.activeExperimentsCard).toBeVisible()
    })
  })

  test.describe('Honest empty state', () => {
    test('shows "No experiments loaded yet" before a monitoring run', async () => {
      // Experiments derive from live monitor data only; with no run yet the
      // page renders an explicit empty state instead of fabricated experiments.
      await expect(expPage.emptyState).toBeVisible()
    })
  })

  test.describe('Loaded state (falsifiability)', () => {
    // Prove experiments are driven by the REAL monitor response: stub a
    // monitor result with a unique experiment name, run monitoring, and assert
    // that name renders (the empty state is replaced). If the page reverted to
    // a sample fallback this unique name would never appear.
    test('renders experiment cards from the live monitor response', async ({ page }) => {
      await stubMonitorEndpoint(page, MONITOR_RESPONSE)

      await expect(expPage.emptyState).toBeVisible()
      await expPage.clickRunMonitoring()

      await expect(page.getByText(LIVE_EXP_NAME)).toBeVisible({ timeout: 10000 })
      await expect(expPage.emptyState).toBeHidden()
    })
  })
})
