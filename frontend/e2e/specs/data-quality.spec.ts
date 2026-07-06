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

    test('#325 Export button becomes enabled once the datasets finish loading', async () => {
      // isAnyLoading = kpiLoading || the five dimension-source KPI fetches.
      // Once they resolve via the shared mocks, the button must be enabled.
      // This guards the disabled-while-loading wiring against a regression
      // where isAnyLoading is left "true" forever.
      await expect(dataPage.exportButton).toBeEnabled()
    })

    test('drift consolidation: links to /monitoring instead of a per-page drift section', async () => {
      // The old drift section read a `data_quality_pipeline` model id that no
      // sweep monitors — it could never show data. The page must link to
      // /monitoring, where model & data drift actually live.
      await expect(dataPage.monitoringLink).toBeVisible()
      await expect(dataPage.monitoringLink).toHaveAttribute('href', '/monitoring')
    })

    test('drift consolidation: Refresh re-reads KPIs and does NOT POST drift detection', async ({ page }) => {
      // The Refresh button previously queued a drift-detection task for the
      // unmonitored `data_quality_pipeline` id — a CTA that could never light
      // the page up. It now refetches the KPI queries instead.
      let driftPosted = false
      page.on('request', (req) => {
        if (req.url().includes('/api/monitoring/drift/detect') && req.method() === 'POST') {
          driftPosted = true
        }
      })
      const kpiRefetch = page.waitForRequest(
        (req) => req.url().includes('/api/kpis') && req.method() === 'GET'
      )
      await dataPage.clickRefresh()
      await kpiRefetch
      expect(driftPosted).toBe(false)
    })
  })
})
