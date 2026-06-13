/**
 * Segment Analysis Page E2E Tests (#19 coverage gap)
 * ==================================================
 *
 * `/segment-analysis` was a routed data page with NO e2e coverage. It is the
 * Tier-2 Heterogeneous Optimizer dashboard. The page was rewritten (F-002) to
 * render ONLY real API results — no fabricated sample fallback — so these
 * specs stub the REAL backend endpoints the page calls
 * (`GET /api/segments/health`, `GET /api/segments/policies`,
 * `POST /api/segments/analyze`) and assert HONEST states:
 *   - real health data -> "Agents Ready" badge + "N analyses today"
 *   - no analysis run yet -> EmptyState "No segment analysis available"
 *   - health endpoint 500 -> labeled QueryErrorState "Failed to load segment health"
 *   - real analyze result -> KPI summary cards appear (empty state gone)
 *
 * We do NOT assert against fabricated sample data — the empty/error states ARE
 * the honest contract.
 */

import { test, expect, type Page, type Route } from '@playwright/test'
import { SegmentAnalysisPage } from '../pages/segment-analysis.page'
import { harnessBase } from '../fixtures/page-harness'

const HEALTH_OK = {
  status: 'healthy',
  agent_available: true,
  econml_available: true,
  causalml_available: true,
  last_analysis: new Date().toISOString(),
  analyses_24h: 7,
}

// Faithful empty mirror of PolicyListResponseWireSchema (the api-client
// Zod-validates GET /segments/policies). Wrong keys here would silently put
// usePolicies into a validation-error path, contradicting the "empty" intent.
const POLICIES_EMPTY = {
  total_count: 0,
  recommendations: [],
  expected_total_lift: 0,
}

async function stubSegmentEndpoints(
  page: Page,
  opts: { healthStatus?: number } = {},
): Promise<void> {
  await page.route('**/api/segments/health**', async (route: Route) => {
    if (opts.healthStatus && opts.healthStatus >= 400) {
      await route.fulfill({
        status: opts.healthStatus,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'segment service unavailable' }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(HEALTH_OK),
    })
  })

  await page.route('**/api/segments/policies**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(POLICIES_EMPTY),
    })
  })
}

test.describe('Segment Analysis Page', () => {
  let segPage: SegmentAnalysisPage

  test.beforeEach(async ({ page }) => {
    await harnessBase(page)
    await stubSegmentEndpoints(page)
    segPage = new SegmentAnalysisPage(page)
    await segPage.goto()
  })

  test.describe('Page Load', () => {
    test('loads at /segment-analysis', async ({ page }) => {
      await expect(page).toHaveURL(/segment-analysis/)
    })

    test('displays the page header', async () => {
      await expect(segPage.pageHeader).toBeVisible()
    })

    test('displays the page description', async () => {
      await expect(segPage.pageDescription).toBeVisible()
    })

    test('shows the analysis configuration card', async () => {
      await expect(segPage.configurationCard).toBeVisible()
    })
  })

  test.describe('Honest states', () => {
    test('renders real health status as "Agents Ready"', async () => {
      // Driven by the stubbed GET /api/segments/health (agent + econml ready).
      await expect(segPage.agentsReadyBadge).toBeVisible()
    })

    test('renders the analyses-today count from real health data', async ({ page }) => {
      // 7 analyses_24h from the stub -> "7 analyses today".
      await expect(page.getByText(/7 analyses today/i)).toBeVisible()
    })

    test('shows honest empty state before any analysis is run', async () => {
      // No fabricated results — the page renders an explicit EmptyState.
      await expect(segPage.emptyState).toBeVisible()
      // And the result tabs are NOT shown until a real result exists.
      await expect(segPage.resultTabs).toBeHidden()
    })
  })

  test.describe('Error state', () => {
    test('shows a labeled error when health endpoint fails', async ({ page }) => {
      // Re-stub health to 500 and reload. The page must surface a labeled
      // error state — NOT silently fall back to plausible-looking values.
      await stubSegmentEndpoints(page, { healthStatus: 500 })
      await page.reload()
      await expect(segPage.pageHeader).toBeVisible()
      await expect(segPage.healthErrorState).toBeVisible({ timeout: 10000 })
    })
  })

  test.describe('Loaded state (falsifiability)', () => {
    // If the page regressed to rendering a fabricated sample result on load,
    // the empty state would never appear (covered above). Here we prove the
    // KPI summary is driven by a REAL analyze response: stub a non-empty
    // result, run the analysis, and assert the empty state is replaced by the
    // KPI cards.
    test('renders KPI summary from a real analyze result', async ({ page }) => {
      // Shape-correct mirror of SegmentAnalysisResponse (CATEResult uses
      // cate_ci_lower/cate_ci_upper/sample_size; SegmentProfile needs
      // responder_type + defining_features as object records). The POST
      // /segments/analyze response is NOT Zod-validated, but a faithful shape
      // keeps the stub honest. overall_ate=0.18 -> the KPI card renders the
      // unique value "0.180" (toFixed(3)), a falsifiability anchor that a
      // fabricated/hardcoded result would not reproduce.
      const analyzeResult = {
        analysis_id: 'e2e-seg-analysis-1',
        status: 'completed',
        cate_by_segment: {
          region: [
            {
              segment_name: 'region',
              segment_value: 'northeast',
              cate_estimate: 0.21,
              cate_ci_lower: 0.15,
              cate_ci_upper: 0.27,
              sample_size: 420,
              statistical_significance: true,
            },
          ],
        },
        overall_ate: 0.18,
        heterogeneity_score: 0.42,
        high_responders: [
          {
            segment_id: 'northeast',
            responder_type: 'high',
            cate_estimate: 0.21,
            defining_features: [{ feature: 'digital_engagement', value: 'high' }],
            size: 420,
            size_percentage: 18.5,
            confidence: 0.86,
            recommendation: 'Increase rep visits',
          },
        ],
        low_responders: [],
        policy_recommendations: [],
        expected_total_lift: 12.4,
        confidence: 0.86,
        key_insights: ['Northeast region shows strong response to rep visits'],
        warnings: [],
      }

      await page.route('**/api/segments/analyze**', async (route: Route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(analyzeResult),
        })
      })

      // Empty state present first; running the analysis must replace it.
      await expect(segPage.emptyState).toBeVisible()
      await segPage.clickRunAnalysis()

      // Assert the KPI card AND its unique value from the stubbed overall_ate
      // (0.18 -> "0.180"). The static-label-only assertion would pass against
      // any fabricated result; the unique value would not.
      await expect(segPage.overallAteCard).toBeVisible({ timeout: 10000 })
      await expect(page.getByText('0.180', { exact: true })).toBeVisible()
      await expect(segPage.emptyState).toBeHidden()
    })
  })

  test.describe('Responsive', () => {
    test('renders on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 812 })
      await expect(segPage.pageHeader).toBeVisible()
    })
  })
})
