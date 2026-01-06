import { test, expect } from '@playwright/test'
import { PredictiveAnalyticsPage } from '../pages/predictive-analytics.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Predictive Analytics Page', () => {
  let predictivePage: PredictiveAnalyticsPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    predictivePage = new PredictiveAnalyticsPage(page)
    await predictivePage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(predictivePage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(predictivePage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(predictivePage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(predictivePage.pageDescription).toBeVisible()
    })
  })

  test.describe('Model Selector', () => {
    test('should display model selector', async () => {
      await expect(predictivePage.modelSelector).toBeVisible()
    })

    test('should allow model selection', async () => {
      // Use a valid model name: 'Conversion Model'
      await predictivePage.selectModel('Conversion')
      await predictivePage.page.waitForTimeout(500)
    })
  })

  test.describe('KPI Cards', () => {
    test('should display KPI cards', async () => {
      const hasKpis = await predictivePage.verifyKPICardsDisplayed()
      expect(hasKpis).toBeTruthy()
    })

    test('should show Accuracy stat', async () => {
      await expect(predictivePage.accuracyCard).toBeVisible()
    })

    test('should show High Risk Entities stat', async () => {
      // Note: KPI cards are High Risk Entities, Avg Model Confidence, etc.
      await expect(predictivePage.highRiskEntitiesCard).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await predictivePage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Predictions tab', async () => {
      await expect(predictivePage.predictionsTab).toBeVisible()
    })

    test('should show Distribution tab', async () => {
      await expect(predictivePage.distributionTab).toBeVisible()
    })

    test('should show Uplift tab', async () => {
      // Note: Tab is called "Uplift" in UI
      await expect(predictivePage.upliftTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await predictivePage.clickTab('Distribution')
      await predictivePage.page.waitForTimeout(500)
    })
  })

  test.describe('Predictions Tab', () => {
    test('should display predictions', async () => {
      const hasPredictions = await predictivePage.verifyPredictionsDisplayed()
      expect(hasPredictions).toBeTruthy()
    })
  })

  test.describe('Distribution Tab', () => {
    test('should display distribution when tab clicked', async () => {
      await predictivePage.clickTab('Distribution')
      const hasDistribution = await predictivePage.verifyDistributionDisplayed()
      expect(hasDistribution).toBeTruthy()
    })
  })

  test.describe('Uplift Tab', () => {
    test('should display uplift segments when tab clicked', async () => {
      // Note: Tab is called "Uplift" in UI, not "Segments"
      await predictivePage.clickTab('Uplift')
      const hasSegments = await predictivePage.verifySegmentsDisplayed()
      expect(hasSegments).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(predictivePage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await predictivePage.clickRefresh()
      await predictivePage.page.waitForTimeout(500)
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await predictivePage.goto()
      await expect(predictivePage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await predictivePage.goto()
      await expect(predictivePage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await predictivePage.goto()
      await expect(predictivePage.mainContent).toBeVisible()
    })
  })
})
