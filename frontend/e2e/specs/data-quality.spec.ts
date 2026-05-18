import { test, expect } from '@playwright/test'
import { DataQualityPage } from '../pages/data-quality.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Data Quality Page', () => {
  let dataPage: DataQualityPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    dataPage = new DataQualityPage(page)
    await dataPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(dataPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(dataPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(dataPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(dataPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Overall Score', () => {
    test('should display overall quality score', async () => {
      const hasScore = await dataPage.verifyOverallScoreDisplayed()
      expect(hasScore).toBeTruthy()
    })

    test('should show overall quality card', async () => {
      await expect(dataPage.overallQualityCard).toBeVisible()
    })
  })

  test.describe('Dimension Cards', () => {
    test('should display dimension cards', async () => {
      const hasCards = await dataPage.verifyDimensionCardsDisplayed()
      expect(hasCards).toBeTruthy()
    })

    test('should show Completeness dimension', async () => {
      await expect(dataPage.completenessCard).toBeVisible()
    })

    test('should show Accuracy dimension', async () => {
      await expect(dataPage.accuracyCard).toBeVisible()
    })

    test('should show Consistency dimension', async () => {
      await expect(dataPage.consistencyCard).toBeVisible()
    })

    test('should show Timeliness dimension', async () => {
      await expect(dataPage.timelinessCard).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await dataPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Validation Rules tab', async () => {
      await expect(dataPage.validationRulesTab).toBeVisible()
    })

    test('should show Data Profiling tab', async () => {
      await expect(dataPage.dataProfilingTab).toBeVisible()
    })

    test('should show Quality Issues tab', async () => {
      await expect(dataPage.qualityIssuesTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await dataPage.clickTab('Data Profiling')
      await dataPage.page.waitForTimeout(500)
    })
  })

  test.describe('Quality Issues Tab', () => {
    test('should display issues when tab clicked', async () => {
      await dataPage.clickTab('Quality Issues')
      const hasIssues = await dataPage.verifyIssuesDisplayed()
      expect(hasIssues).toBeTruthy()
    })
  })

  test.describe('Data Profiling Tab', () => {
    test('should display profiling when tab clicked', async () => {
      await dataPage.clickTab('Data Profiling')
      const hasTrends = await dataPage.verifyTrendsDisplayed()
      expect(hasTrends).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(dataPage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await dataPage.clickRefresh()
      await dataPage.page.waitForTimeout(500)
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await dataPage.goto()
      await expect(dataPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await dataPage.goto()
      await expect(dataPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await dataPage.goto()
      await expect(dataPage.mainContent).toBeVisible()
    })
  })

  // ---------------------------------------------------------------------------
  // Post-#330 DOM contract — coverage for live-wired behaviors that #320 +
  // #330 introduced. #306 was closed by #320 but the spec was never updated
  // to assert against the new accessible-name + interaction surfaces:
  //   - #322 status filter Select wired to per-row computed status
  //   - #323 page-level driftHistoryError banner (default tab visible)
  //   - #325 Export button disabled while datasets are loading
  //   - #326 triggerDrift POST sends time_window=30d (aligned with display)
  //   - #328 accessible names on Search input + Status SelectTrigger
  // Refs #332.
  // ---------------------------------------------------------------------------
  test.describe('Post-#330 features', () => {
    test('#328 search input has accessible label', async () => {
      // The Input is associated with an sr-only <Label htmlFor="dq-search">.
      // getByLabel proves the association by accessible-name, not placeholder.
      await expect(dataPage.ruleSearchInputByLabel).toBeVisible()
    })

    test('#328 status filter SelectTrigger has accessible aria-label', async () => {
      // Addressed as a combobox by its aria-label rather than visible text
      // (the visible text is the SelectValue placeholder "Status").
      await expect(dataPage.statusFilter).toBeVisible()
    })

    test('#322 status filter exposes Pass / Warning / Fail / All options', async () => {
      await dataPage.statusFilter.click()
      // Radix SelectContent renders options as role=option once open.
      for (const label of [/^all$/i, /^pass$/i, /^warning$/i, /^fail$/i]) {
        await expect(dataPage.page.getByRole('option', { name: label })).toBeVisible()
      }
    })

    test('#322 search filter hides every row and triggers the empty-state', async () => {
      // The empty-state ("No data quality KPIs match your filters") is shared
      // between the status filter (#322) and the search filter. We assert via
      // search — it's deterministic regardless of per-row useKPIDetail timing
      // (the status filter path depends on every row's value query having
      // resolved, which the shared api-mocks don't currently surface as a
      // numeric value with the URL shape useKPIValue produces).
      await dataPage.searchRules('this-rule-id-does-not-exist-zzzzz')
      await expect(dataPage.noKpisMatchEmptyState).toBeVisible()
    })

    test('#322 clearing the search restores rows and hides the empty-state', async () => {
      // Round-trip: empty-state → cleared → table rows back. Guards the parent's
      // visibleKpiCount memo against filter-state staleness.
      await dataPage.searchRules('this-rule-id-does-not-exist-zzzzz')
      await expect(dataPage.noKpisMatchEmptyState).toBeVisible()
      await dataPage.searchRules('')
      await expect(dataPage.noKpisMatchEmptyState).not.toBeVisible()
    })

    test('#323 surfaces drift-history error banner on default (Validation Rules) tab', async ({ page }) => {
      // Override the shared drift-history mock to a 5xx BEFORE navigating so
      // the page mounts with an error state already populated. The banner is
      // hoisted page-level (#323) so it's visible from the default tab.
      //
      // Using 4xx (404) rather than 5xx so the shared QueryClient's shouldRetry
      // skips retries (4xx => no retry except 408/429); 5xx would trigger 3
      // exponential-backoff retries (~7s) before the error state lands.
      await page.route('**/api/monitoring/drift/history/**', async (route) => {
        await route.fulfill({
          status: 404,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'simulated drift-history failure' }),
        })
      })
      // Re-navigate so the new route takes effect on the underlying query.
      await dataPage.goto()
      await expect(dataPage.driftHistoryErrorBanner).toBeVisible({ timeout: 15000 })
      // And we are on the default (rules) tab, NOT Quality Issues.
      await expect(dataPage.validationRulesTab).toHaveAttribute('data-state', 'active')
    })

    test('#325 Export button becomes enabled once the datasets finish loading', async () => {
      // isAnyLoading = kpiLoading || driftLoading || driftHistoryLoading. Once
      // all three resolve via the shared mocks, the button must be enabled.
      // This guards the disabled-while-loading wiring against a regression
      // where isAnyLoading is left "true" forever.
      await expect(dataPage.exportButton).toBeEnabled()
    })

    test('#326 Refresh triggers drift detection with time_window=30d', async ({ page }) => {
      // Issue #326 aligned the trigger window with the 30-day display window.
      // Assert on the request payload to catch a silent revert to '7d'.
      const requestPromise = page.waitForRequest(
        (req) =>
          req.url().includes('/api/monitoring/drift/detect') && req.method() === 'POST'
      )
      await dataPage.clickRefresh()
      const request = await requestPromise
      const body = request.postDataJSON() as {
        model_id?: string
        time_window?: string
        check_data_drift?: boolean
      } | null
      expect(body?.time_window).toBe('30d')
      expect(body?.model_id).toBe('data_quality_pipeline')
      expect(body?.check_data_drift).toBe(true)
    })
  })
})
