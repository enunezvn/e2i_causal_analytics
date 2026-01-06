import { test, expect } from '@playwright/test'
import { KPIDictionaryPage } from '../pages/kpi-dictionary.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('KPI Dictionary Page', () => {
  let kpiPage: KPIDictionaryPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    kpiPage = new KPIDictionaryPage(page)
    await kpiPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(kpiPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(kpiPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(kpiPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(kpiPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Stats Cards', () => {
    test('should display stats cards', async () => {
      const hasStats = await kpiPage.verifyStatsDisplayed()
      expect(hasStats).toBeTruthy()
    })

    test('should show Total KPIs stat', async () => {
      await expect(kpiPage.totalKPIsCard).toBeVisible()
    })

    test('should show Workstreams stat', async () => {
      await expect(kpiPage.workstreamsCard).toBeVisible()
    })

    test('should show Causal KPIs stat', async () => {
      await expect(kpiPage.causalKPIsCard).toBeVisible()
    })
  })

  test.describe('Search', () => {
    test('should display search input', async () => {
      await expect(kpiPage.searchInput).toBeVisible()
    })

    test('should allow KPI search', async () => {
      await kpiPage.searchKPIs('ROI')
      await kpiPage.page.waitForTimeout(500)
    })

    test('should filter results when searching', async () => {
      const searchWorks = await kpiPage.verifySearchWorks()
      expect(searchWorks).toBeTruthy()
    })

    test('should clear search', async () => {
      await kpiPage.searchKPIs('test')
      await kpiPage.clearSearch()
      await kpiPage.page.waitForTimeout(500)
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await kpiPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show All KPIs tab', async () => {
      await expect(kpiPage.allKPIsTab).toBeVisible()
    })

    test('should show Data Quality tab', async () => {
      await expect(kpiPage.dataQualityTab).toBeVisible()
    })

    test('should show Model Performance tab', async () => {
      await expect(kpiPage.modelPerformanceTab).toBeVisible()
    })

    test('should show Trigger Performance tab', async () => {
      await expect(kpiPage.triggerPerformanceTab).toBeVisible()
    })

    test('should show Business Impact tab', async () => {
      await expect(kpiPage.businessImpactTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await kpiPage.clickTab('Data Quality')
      await kpiPage.page.waitForTimeout(500)
    })
  })

  test.describe('KPI Cards', () => {
    test('should display KPI cards', async () => {
      const hasCards = await kpiPage.verifyKPICardsDisplayed()
      expect(hasCards).toBeTruthy()
    })
  })

  test.describe('Workstream Selection', () => {
    test('should filter by workstream', async () => {
      await kpiPage.selectWorkstream('Model Performance')
      await kpiPage.page.waitForTimeout(500)
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await kpiPage.goto()
      await expect(kpiPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await kpiPage.goto()
      await expect(kpiPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await kpiPage.goto()
      await expect(kpiPage.mainContent).toBeVisible()
    })
  })
})
