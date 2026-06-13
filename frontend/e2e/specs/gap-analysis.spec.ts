/**
 * Gap Analysis Page E2E Tests (#19 coverage gap)
 * ==============================================
 *
 * `/gap-analysis` was a routed data page with NO e2e coverage. It is the
 * Tier-2 Gap Analyzer ROI-opportunity dashboard. F-002 removed the
 * SAMPLE_OPPORTUNITIES fallback, so data comes strictly from the API. These
 * specs stub the REAL endpoints the page calls
 * (`GET /api/gaps/opportunities`, `GET /api/gaps/health`) and assert HONEST
 * states:
 *   - empty list -> EmptyState "No gap opportunities available"
 *   - KPI overview cards reflect the real `total_addressable_value`
 *
 * We do NOT assert against fabricated sample opportunities — the empty state
 * IS the honest contract when the API returns nothing.
 */

import { test, expect, type Page, type Route } from '@playwright/test'
import { GapAnalysisPage } from '../pages/gap-analysis.page'
import { harnessBase } from '../fixtures/page-harness'

const HEALTH_OK = {
  status: 'healthy',
  agent_available: true,
  last_analysis: new Date().toISOString(),
  analyses_24h: 4,
}

const OPPS_EMPTY = {
  total_count: 0,
  quick_wins_count: 0,
  strategic_bets_count: 0,
  opportunities: [],
  total_addressable_value: 0,
}

async function stubGapEndpoints(
  page: Page,
  opportunitiesBody: unknown = OPPS_EMPTY,
): Promise<void> {
  await page.route('**/api/gaps/health**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(HEALTH_OK),
    })
  })

  await page.route('**/api/gaps/opportunities**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(opportunitiesBody),
    })
  })
}

test.describe('Gap Analysis Page', () => {
  let gapPage: GapAnalysisPage

  test.describe('Empty (honest) state', () => {
    test.beforeEach(async ({ page }) => {
      await harnessBase(page)
      await stubGapEndpoints(page)
      gapPage = new GapAnalysisPage(page)
      await gapPage.goto()
    })

    test('loads at /gap-analysis', async ({ page }) => {
      await expect(page).toHaveURL(/gap-analysis/)
    })

    test('displays the page header', async () => {
      await expect(gapPage.pageHeader).toBeVisible()
    })

    test('displays the page description', async () => {
      await expect(gapPage.pageDescription).toBeVisible()
    })

    test('renders the KPI overview cards', async () => {
      await expect(gapPage.totalAddressableCard).toBeVisible()
      await expect(gapPage.opportunitiesCard).toBeVisible()
    })

    test('shows honest empty state when no opportunities are returned', async () => {
      await expect(gapPage.emptyState).toBeVisible()
    })
  })

  test.describe('Loaded KPI value (falsifiability)', () => {
    // Prove the "Total Addressable" KPI is driven by the REAL API payload, not
    // a hard-coded number: stub a unique non-zero total_addressable_value and
    // assert the formatted value ($2.3M) renders. If the page reverted to a
    // fabricated constant this would fail.
    test('reflects total_addressable_value from the live API', async ({ page }) => {
      await harnessBase(page)
      await stubGapEndpoints(page, {
        total_count: 0,
        quick_wins_count: 0,
        strategic_bets_count: 0,
        opportunities: [],
        total_addressable_value: 2_300_000,
      })
      gapPage = new GapAnalysisPage(page)
      await gapPage.goto()

      await expect(gapPage.totalAddressableCard).toBeVisible()
      await expect(page.getByText('$2.3M', { exact: true })).toBeVisible()
    })
  })
})
