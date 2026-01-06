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
})
