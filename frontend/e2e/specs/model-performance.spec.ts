import { test, expect } from '@playwright/test'
import { ModelPerformancePage } from '../pages/model-performance.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Model Performance Page', () => {
  let modelPage: ModelPerformancePage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    modelPage = new ModelPerformancePage(page)
    await modelPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(modelPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(modelPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(modelPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(modelPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Model Selector', () => {
    test('should display model selector', async () => {
      await expect(modelPage.modelSelector).toBeVisible()
    })

    test('should allow model selection', async () => {
      // Use a valid model name from SAMPLE_MODELS: 'HCP Tier Classifier'
      await modelPage.selectModel('HCP Tier')
      await modelPage.page.waitForTimeout(500)
    })
  })

  test.describe('KPI Cards', () => {
    test('should display KPI cards', async () => {
      const hasKpis = await modelPage.verifyKPICardsDisplayed()
      expect(hasKpis).toBeTruthy()
    })

    test('should show Accuracy metric', async () => {
      await expect(modelPage.accuracyCard).toBeVisible()
    })

    test('should show Precision metric', async () => {
      await expect(modelPage.precisionCard).toBeVisible()
    })

    test('should show Recall metric', async () => {
      await expect(modelPage.recallCard).toBeVisible()
    })

    test('should show F1 Score metric', async () => {
      await expect(modelPage.f1ScoreCard).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await modelPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Confusion Matrix tab as first tab', async () => {
      // Note: There is no Overview tab - first tab is Confusion Matrix
      await expect(modelPage.confusionMatrixTab).toBeVisible()
    })

    test('should show ROC Curve tab', async () => {
      await expect(modelPage.rocCurveTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await modelPage.clickTab('ROC Curve')
      await modelPage.page.waitForTimeout(500)
    })
  })

  test.describe('Overview Tab', () => {
    test('should display performance metrics', async () => {
      const hasMetrics = await modelPage.verifyMetricsDisplayed()
      expect(hasMetrics).toBeTruthy()
    })
  })

  test.describe('Confusion Matrix Tab', () => {
    test('should display confusion matrix when tab clicked', async () => {
      await modelPage.clickTab('Confusion Matrix')
      const hasMatrix = await modelPage.verifyConfusionMatrixDisplayed()
      expect(hasMatrix).toBeTruthy()
    })
  })

  test.describe('ROC Curve Tab', () => {
    test('should display ROC curve when tab clicked', async () => {
      await modelPage.clickTab('ROC Curve')
      const hasRoc = await modelPage.verifyROCCurveDisplayed()
      expect(hasRoc).toBeTruthy()
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await modelPage.goto()
      await expect(modelPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await modelPage.goto()
      await expect(modelPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await modelPage.goto()
      await expect(modelPage.mainContent).toBeVisible()
    })
  })
})
