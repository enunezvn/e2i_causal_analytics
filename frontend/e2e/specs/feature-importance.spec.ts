import { test, expect } from '@playwright/test'
import { FeatureImportancePage } from '../pages/feature-importance.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Feature Importance Page', () => {
  let featurePage: FeatureImportancePage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    featurePage = new FeatureImportancePage(page)
    await featurePage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(featurePage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(featurePage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(featurePage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(featurePage.pageDescription).toBeVisible()
    })
  })

  test.describe('Model Selector', () => {
    test('should display model selector', async () => {
      await expect(featurePage.modelSelector).toBeVisible()
    })

    test('should allow model selection', async () => {
      // Use a valid model name from SAMPLE_MODELS: 'HCP Tier Classifier'
      await featurePage.selectModel('HCP Tier')
      await featurePage.page.waitForTimeout(500)
    })
  })

  test.describe('Model Info', () => {
    test('should display model info', async () => {
      const hasModelInfo = await featurePage.verifyModelInfoDisplayed()
      expect(hasModelInfo).toBeTruthy()
    })

    test('should show Base Value stat', async () => {
      await expect(featurePage.baseValueDisplay).toBeVisible()
    })

    test('should show Top Feature stat', async () => {
      await expect(featurePage.topFeatureDisplay).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await featurePage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Bar Chart tab', async () => {
      await expect(featurePage.barChartTab).toBeVisible()
    })

    test('should show Beeswarm tab', async () => {
      await expect(featurePage.beeswarmTab).toBeVisible()
    })

    test('should show Waterfall tab', async () => {
      await expect(featurePage.waterfallTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await featurePage.clickTab('Beeswarm')
      await featurePage.page.waitForTimeout(500)
    })
  })

  test.describe('Bar Chart Tab', () => {
    test('should display bar chart by default', async () => {
      const hasBarChart = await featurePage.verifyBarChartDisplayed()
      expect(hasBarChart).toBeTruthy()
    })
  })

  test.describe('Beeswarm Tab', () => {
    test('should display beeswarm when tab clicked', async () => {
      await featurePage.clickTab('Beeswarm')
      const hasBeeswarm = await featurePage.verifyBeeswarmDisplayed()
      expect(hasBeeswarm).toBeTruthy()
    })
  })

  test.describe('Waterfall Tab', () => {
    test('should display waterfall when tab clicked', async () => {
      await featurePage.clickTab('Waterfall')
      const hasWaterfall = await featurePage.verifyWaterfallDisplayed()
      expect(hasWaterfall).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(featurePage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await featurePage.clickRefresh()
      await featurePage.page.waitForTimeout(500)
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await featurePage.goto()
      await expect(featurePage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await featurePage.goto()
      await expect(featurePage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await featurePage.goto()
      await expect(featurePage.mainContent).toBeVisible()
    })
  })
})
