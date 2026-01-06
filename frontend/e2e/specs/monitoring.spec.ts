import { test, expect } from '@playwright/test'
import { MonitoringPage } from '../pages/monitoring.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Monitoring Page', () => {
  let monitoringPage: MonitoringPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    monitoringPage = new MonitoringPage(page)
    await monitoringPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(monitoringPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(monitoringPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(monitoringPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(monitoringPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Overview Metrics', () => {
    test('should display overview metrics', async () => {
      const hasMetrics = await monitoringPage.verifyOverviewMetricsDisplayed()
      expect(hasMetrics).toBeTruthy()
    })

    test('should show Total Requests metric', async () => {
      await expect(monitoringPage.totalRequestsCard).toBeVisible()
    })

    test('should show Error Rate metric', async () => {
      await expect(monitoringPage.errorRateCard).toBeVisible()
    })

    test('should show Avg Latency metric', async () => {
      await expect(monitoringPage.avgLatencyCard).toBeVisible()
    })

    test('should show Active Users metric', async () => {
      await expect(monitoringPage.activeUsersCard).toBeVisible()
    })
  })

  test.describe('Time Range Selector', () => {
    test('should display time range selector', async () => {
      await expect(monitoringPage.timeRangeSelector).toBeVisible()
    })

    test('should allow time range selection', async () => {
      await monitoringPage.selectTimeRange('24 hours')
      await monitoringPage.page.waitForTimeout(500)
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await monitoringPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show API Usage tab', async () => {
      await expect(monitoringPage.apiUsageTab).toBeVisible()
    })

    test('should show User Activity tab', async () => {
      await expect(monitoringPage.userActivityTab).toBeVisible()
    })

    test('should show Errors tab', async () => {
      await expect(monitoringPage.errorsTab).toBeVisible()
    })

    test('should show System tab', async () => {
      await expect(monitoringPage.systemTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await monitoringPage.clickTab('Errors')
      await monitoringPage.page.waitForTimeout(500)
    })
  })

  test.describe('API Usage Tab', () => {
    test('should display API usage', async () => {
      const hasUsage = await monitoringPage.verifyAPIUsageDisplayed()
      expect(hasUsage).toBeTruthy()
    })
  })

  test.describe('User Activity Tab', () => {
    test('should display user activity when tab clicked', async () => {
      await monitoringPage.clickTab('User Activity')
      const hasActivity = await monitoringPage.verifyUserActivityDisplayed()
      expect(hasActivity).toBeTruthy()
    })
  })

  test.describe('Errors Tab', () => {
    test('should display error logs when tab clicked', async () => {
      await monitoringPage.clickTab('Errors')
      const hasErrors = await monitoringPage.verifyErrorLogsDisplayed()
      expect(hasErrors).toBeTruthy()
    })
  })

  test.describe('System Tab', () => {
    test('should display system metrics when tab clicked', async () => {
      await monitoringPage.clickTab('System')
      const hasSystem = await monitoringPage.verifySystemMetricsDisplayed()
      expect(hasSystem).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(monitoringPage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await monitoringPage.clickRefresh()
      await monitoringPage.page.waitForTimeout(500)
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await monitoringPage.goto()
      await expect(monitoringPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await monitoringPage.goto()
      await expect(monitoringPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await monitoringPage.goto()
      await expect(monitoringPage.mainContent).toBeVisible()
    })
  })
})
