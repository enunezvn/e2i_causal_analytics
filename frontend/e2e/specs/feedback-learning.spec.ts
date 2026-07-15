/**
 * Feedback Learning Page E2E Tests (#19 coverage gap)
 * ===================================================
 *
 * `/feedback-learning` was a routed data page with NO e2e coverage. It is the
 * Tier-5 self-improvement dashboard. These specs stub the REAL endpoints the
 * page calls (`GET /api/feedback/health`, `GET /api/feedback/patterns`,
 * `GET /api/feedback/updates`) and assert HONEST states:
 *   - agent available -> "Online" status; unavailable -> "Offline"
 *   - empty updates -> "No knowledge updates proposed"
 *
 * These endpoints are plain (non-Zod) gets; the stubs mirror the real wire
 * shapes the page consumes.
 */

import { test, expect, type Page, type Route } from '@playwright/test'
import { FeedbackLearningPage } from '../pages/feedback-learning.page'
import { harnessBase } from '../fixtures/page-harness'

const HEALTH_ONLINE = {
  status: 'healthy',
  agent_available: true,
  last_learning_cycle: new Date().toISOString(),
  cycles_24h: 17,
  patterns_active: 0,
  pending_updates: 0,
}

const HEALTH_OFFLINE = {
  status: 'degraded',
  agent_available: false,
  last_learning_cycle: null,
  cycles_24h: 0,
  patterns_active: 0,
  pending_updates: 0,
}

const PATTERNS_EMPTY = { total_count: 0, critical_count: 0, high_count: 0, patterns: [] }
const UPDATES_EMPTY = { total_count: 0, proposed_count: 0, applied_count: 0, updates: [] }

async function stubFeedbackEndpoints(
  page: Page,
  opts: { offline?: boolean } = {},
): Promise<void> {
  await page.route('**/api/feedback/health**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(opts.offline ? HEALTH_OFFLINE : HEALTH_ONLINE),
    })
  })
  await page.route('**/api/feedback/patterns**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(PATTERNS_EMPTY),
    })
  })
  await page.route('**/api/feedback/updates**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(UPDATES_EMPTY),
    })
  })
}

test.describe('Feedback Learning Page', () => {
  let fbPage: FeedbackLearningPage

  test.describe('Agent online', () => {
    test.beforeEach(async ({ page }) => {
      await harnessBase(page)
      await stubFeedbackEndpoints(page)
      fbPage = new FeedbackLearningPage(page)
      await fbPage.goto()
    })

    test('loads at /feedback-learning', async ({ page }) => {
      await expect(page).toHaveURL(/feedback-learning/)
    })

    test('displays the page header', async () => {
      await expect(fbPage.pageHeader).toBeVisible()
    })

    test('displays the page description', async () => {
      await expect(fbPage.pageDescription).toBeVisible()
    })

    test('renders "Online" agent status from real health data', async () => {
      await expect(fbPage.onlineStatus).toBeVisible()
    })

    test('shows honest empty state on the Knowledge Updates tab', async () => {
      await fbPage.openUpdatesTab()
      await expect(fbPage.noUpdatesEmptyState).toBeVisible({ timeout: 10000 })
    })
  })

  test.describe('Agent offline (falsifiability)', () => {
    test('renders "Offline" status when the feedback agent is unavailable', async ({ page }) => {
      await harnessBase(page)
      await stubFeedbackEndpoints(page, { offline: true })
      fbPage = new FeedbackLearningPage(page)
      await fbPage.goto()

      await expect(fbPage.offlineStatus).toBeVisible()
    })
  })
})
