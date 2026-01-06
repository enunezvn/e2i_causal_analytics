import { test, expect } from '@playwright/test'
import { SystemHealthPage } from '../pages/system-health.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('System Health Page', () => {
  let healthPage: SystemHealthPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    healthPage = new SystemHealthPage(page)
    await healthPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(healthPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(healthPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(healthPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(healthPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Overview Stats', () => {
    test('should display overview stats', async () => {
      const hasStats = await healthPage.verifyOverviewStatsDisplayed()
      expect(hasStats).toBeTruthy()
    })

    test('should show Services stat', async () => {
      await expect(healthPage.servicesCard).toBeVisible()
    })

    test('should show Model Health stat', async () => {
      await expect(healthPage.modelHealthCard).toBeVisible()
    })

    test('should show Active Alerts stat', async () => {
      await expect(healthPage.activeAlertsCard).toBeVisible()
    })
  })

  test.describe('Service Status', () => {
    test('should display service status section', async () => {
      const hasStatus = await healthPage.verifyServiceStatusDisplayed()
      expect(hasStatus).toBeTruthy()
    })

    test('should show API Gateway service', async () => {
      await expect(healthPage.apiGatewayService).toBeVisible()
    })

    test('should show PostgreSQL service', async () => {
      await expect(healthPage.postgresService).toBeVisible()
    })

    test('should show Redis Cache service', async () => {
      await expect(healthPage.redisService).toBeVisible()
    })

    test('should show FalkorDB service', async () => {
      await expect(healthPage.falkordbService).toBeVisible()
    })
  })

  test.describe('Model Health', () => {
    test('should display model health section', async () => {
      const hasModels = await healthPage.verifyModelHealthDisplayed()
      expect(hasModels).toBeTruthy()
    })

    test('should show Propensity Model', async () => {
      await expect(healthPage.propensityModel).toBeVisible()
    })
  })

  test.describe('Active Alerts', () => {
    test('should display alerts section', async () => {
      const hasAlerts = await healthPage.verifyAlertsDisplayed()
      expect(hasAlerts).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(healthPage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await healthPage.clickRefresh()
      await healthPage.page.waitForTimeout(500)
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await healthPage.goto()
      await expect(healthPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await healthPage.goto()
      await expect(healthPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await healthPage.goto()
      await expect(healthPage.mainContent).toBeVisible()
    })
  })
})
