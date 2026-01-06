import { test, expect } from '@playwright/test'
import { DigitalTwinPage } from '../pages/digital-twin.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Digital Twin Page', () => {
  let twinPage: DigitalTwinPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    twinPage = new DigitalTwinPage(page)
    await twinPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(twinPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(twinPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(twinPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(twinPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Stats Cards', () => {
    test('should display stats cards', async () => {
      const hasStats = await twinPage.verifyStatsDisplayed()
      expect(hasStats).toBeTruthy()
    })

    test('should show Simulations Today stat', async () => {
      await expect(twinPage.simulationsTodayCard).toBeVisible()
    })

    test('should show Model Fidelity stat', async () => {
      await expect(twinPage.modelFidelityCard).toBeVisible()
    })
  })

  test.describe('Simulation Panel', () => {
    test('should display simulation panel', async () => {
      const hasPanel = await twinPage.verifySimulationPanelDisplayed()
      expect(hasPanel).toBeTruthy()
    })

    test('should show configure simulation section', async () => {
      await expect(twinPage.configureSimulationSection).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await twinPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Results tab button', async () => {
      await expect(twinPage.resultsTab).toBeVisible()
    })

    test('should show History tab button', async () => {
      await expect(twinPage.historyTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await twinPage.clickResultsTab()
      await twinPage.page.waitForTimeout(500)
    })
  })

  test.describe('Results Tab', () => {
    test('should display results when tab clicked', async () => {
      await twinPage.clickResultsTab()
      const hasResults = await twinPage.verifyResultsDisplayed()
      expect(hasResults).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have run simulation button', async () => {
      await expect(twinPage.runSimulationButton).toBeVisible()
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await twinPage.goto()
      await expect(twinPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await twinPage.goto()
      await expect(twinPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await twinPage.goto()
      await expect(twinPage.mainContent).toBeVisible()
    })
  })
})
