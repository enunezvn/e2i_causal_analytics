import { test, expect } from '@playwright/test'
import { InterventionImpactPage } from '../pages/intervention-impact.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Intervention Impact Page', () => {
  let impactPage: InterventionImpactPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    impactPage = new InterventionImpactPage(page)
    await impactPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(impactPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(impactPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(impactPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(impactPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Intervention Selector', () => {
    test('should display intervention selector', async () => {
      await expect(impactPage.interventionSelector).toBeVisible()
    })

    test('should allow intervention selection', async () => {
      await impactPage.selectIntervention('Campaign')
      await impactPage.page.waitForTimeout(500)
    })
  })

  test.describe('Intervention Summary', () => {
    test('should display intervention summary', async () => {
      const hasSummary = await impactPage.verifyInterventionSummaryDisplayed()
      expect(hasSummary).toBeTruthy()
    })
  })

  test.describe('KPI Summary Cards', () => {
    test('should display KPI summary', async () => {
      const hasKpis = await impactPage.verifyKPISummaryDisplayed()
      expect(hasKpis).toBeTruthy()
    })

    test('should show Average Treatment Effect card', async () => {
      await expect(impactPage.avgTreatmentEffectCard).toBeVisible()
    })

    test('should show Significant Effects card', async () => {
      await expect(impactPage.significantEffectsCard).toBeVisible()
    })

    test('should show Cumulative Impact card', async () => {
      await expect(impactPage.cumulativeImpactCard).toBeVisible()
    })

    test('should show ROI Estimate card', async () => {
      await expect(impactPage.roiEstimateCard).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await impactPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Causal Impact tab', async () => {
      await expect(impactPage.causalImpactTab).toBeVisible()
    })

    test('should show Before/After tab', async () => {
      await expect(impactPage.beforeAfterTab).toBeVisible()
    })

    test('should show Treatment Effects tab', async () => {
      await expect(impactPage.treatmentEffectsTab).toBeVisible()
    })

    test('should show Segment Analysis tab', async () => {
      await expect(impactPage.segmentAnalysisTab).toBeVisible()
    })

    test('should show Digital Twin tab', async () => {
      await expect(impactPage.digitalTwinTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await impactPage.clickTab('Before/After')
      await impactPage.page.waitForTimeout(500)
    })
  })

  test.describe('Causal Impact Tab', () => {
    test('should display causal impact analysis', async () => {
      const hasCausal = await impactPage.verifyCausalImpactDisplayed()
      expect(hasCausal).toBeTruthy()
    })
  })

  test.describe('Before/After Tab', () => {
    test('should display before/after comparison when tab clicked', async () => {
      await impactPage.clickTab('Before/After')
      const hasComparison = await impactPage.verifyBeforeAfterDisplayed()
      expect(hasComparison).toBeTruthy()
    })
  })

  test.describe('Treatment Effects Tab', () => {
    test('should display treatment effects when tab clicked', async () => {
      await impactPage.clickTab('Treatment Effects')
      const hasEffects = await impactPage.verifyTreatmentEffectsDisplayed()
      expect(hasEffects).toBeTruthy()
    })
  })

  test.describe('Segment Analysis Tab', () => {
    test('should display segment analysis when tab clicked', async () => {
      await impactPage.clickTab('Segment')
      const hasSegments = await impactPage.verifySegmentAnalysisDisplayed()
      expect(hasSegments).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(impactPage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await impactPage.clickRefresh()
      await impactPage.page.waitForTimeout(500)
    })

    test('should have download button', async () => {
      await expect(impactPage.downloadButton).toBeVisible()
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await impactPage.goto()
      await expect(impactPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await impactPage.goto()
      await expect(impactPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await impactPage.goto()
      await expect(impactPage.mainContent).toBeVisible()
    })
  })
})
