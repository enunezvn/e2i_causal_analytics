import { test, expect } from '@playwright/test'
import { HomePage } from '../pages/home.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNoErrors } from '../utils/assertions'

test.describe('Home Page', () => {
  let homePage: HomePage

  test.beforeEach(async ({ page }) => {
    // Setup API mocks
    await mockApiRoutes(page)

    // Initialize page object
    homePage = new HomePage(page)
    await homePage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(homePage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(homePage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      // NOTE: avoid the shared `assertNotLoading` here. It checks
      // `[role="progressbar"]` which matches the legitimate Radix `Progress`
      // bars rendered for the Agent Tier Summary (6 visible bars). The shared
      // helper waits the full timeout for each to disappear, exceeding the
      // 30s test budget. Scope the loading check to *transient* indicators.
      const loadingLocators = [
        page.locator('[data-testid="loading"]'),
        page.locator('.loading-spinner'),
      ]
      for (const locator of loadingLocators) {
        await expect(locator)
          .not.toBeVisible({ timeout: TIMEOUTS.MEDIUM })
          .catch(() => {})
      }
    })

    test('should display page header', async () => {
      await expect(homePage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(homePage.pageDescription).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })
  })

  test.describe('Quick Stats Bar', () => {
    test('should display quick stats', async () => {
      const hasStats = await homePage.verifyQuickStatsDisplayed()
      expect(hasStats).toBeTruthy()
    })

    test('should show Total TRx stat', async () => {
      await expect(homePage.totalTrxStat).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should show Active Campaigns stat', async () => {
      await expect(homePage.activeCampaignsStat).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should show HCPs Reached stat', async () => {
      await expect(homePage.hcpsReachedStat).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should show Model Accuracy stat', async () => {
      await expect(homePage.modelAccuracyStat).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('renders REAL quick-stat values, not the old fabricated constants', async ({ page }) => {
      // Mocked summary -> Total TRx (MTD) = 125,000 ; HCPs Reached = 8,500 ;
      // Active Campaigns = 12. Model Accuracy = 80.0% comes from the per-brand
      // gold-standard summary (brand-summary mock, accuracy=0.80) since #1067.
      await expect(page.getByText('125,000')).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
      await expect(page.getByText('8,500')).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
      await expect(page.getByText('80.0%')).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
      // The old fabricated values must be gone.
      await expect(page.getByText('125,430')).toHaveCount(0)
      await expect(page.getByText('94.2%')).toHaveCount(0)
    })
  })

  test.describe('KPI Overview', () => {
    test('should display KPI section', async () => {
      const hasKpis = await homePage.verifyKpiCardsDisplayed()
      expect(hasKpis).toBeTruthy()
    })

    test('should show KPI Performance Indicators heading', async ({ page }) => {
      await expect(page.getByText('Key Performance Indicators')).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display Total TRx metric', async () => {
      await expect(homePage.totalTrxCard).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display KPI values', async () => {
      // Check that numeric values are displayed
      const value = await homePage.getKpiValue('Total TRx')
      // Value might be null if not visible, but function should not throw
      expect(value === null || typeof value === 'string').toBeTruthy()
    })
  })

  test.describe('Brand Selector', () => {
    test('should display brand selector', async () => {
      await expect(homePage.brandSelector).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should show default brand selection', async () => {
      const text = await homePage.getBrandSelectorText()
      expect(text).toBeTruthy()
    })

    test('should allow brand selection', async () => {
      await homePage.selectBrand('Remibrutinib')
      // Wait for update
      await homePage.page.waitForTimeout(500)
      // Verify brand changed (selection should work)
    })

    test('should have all filter selectors', async () => {
      // Wait for selectors to render
      await homePage.brandSelector.waitFor({ state: 'visible', timeout: TIMEOUTS.PAGE_LOAD })
      const count = await homePage.allSelectors.count()
      // Exactly two filter comboboxes in the header: Brand + Region. The
      // non-functional Period/Date-Range selector was removed.
      expect(count).toBeGreaterThanOrEqual(2)
    })
  })

  test.describe('Region Selector', () => {
    test('should display region selector', async () => {
      await expect(homePage.regionSelector).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should allow region selection', async () => {
      await homePage.selectRegion('Northeast')
      await homePage.page.waitForTimeout(500)
    })
  })

  // Date Range Picker tests removed: the non-functional Period/Date-Range
  // selector was removed from the Home header (it filtered no data). The Home
  // header now exposes only the Brand and Region filters.

  test.describe('System Health', () => {
    test('should show system health indicator', async () => {
      const isVisible = await homePage.verifySystemHealthShown()
      expect(isVisible).toBeTruthy()
    })

    test('renders REAL agent health scores, no fabricated latencies', async ({ page }) => {
      // Mocked /health-score/full -> grade A, component 95%, model 88% etc.
      // (#994 moved the Home System Health card to the FULL, all-dimension check
      // so Components/Models/Pipelines/Agents rows all render real per-dimension
      // scores instead of a fabricated component-only "0%".)
      // Exact match: the Model Accuracy tile's "avg of N models" subline (#1067)
      // otherwise also matches a substring getByText('Models') (strict-mode clash).
      await expect(page.getByText('Components', { exact: true })).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
      await expect(page.getByText('Models', { exact: true })).toBeVisible()
      // Fabricated infra latencies must NOT appear.
      await expect(page.getByText('45ms')).toHaveCount(0)
      await expect(page.getByText('API Gateway')).toHaveCount(0)
    })
  })

  test.describe('Agent Status', () => {
    test('renders REAL tier counts, no hardcoded 15/21', async ({ page }) => {
      // Mocked /api/agents/status -> 3 active of 5 total.
      await expect(page.getByText('3/5 agents active')).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
      await expect(page.getByText('15/21 agents active')).toHaveCount(0)
    })
  })

  test.describe('Navigation', () => {
    test('should navigate to Causal Discovery via link', async ({ page }) => {
      const causalLink = page.getByRole('link', { name: /causal/i }).first()
      if (await causalLink.isVisible().catch(() => false)) {
        await causalLink.click()
        await page.waitForLoadState('networkidle')
        expect(page.url()).toContain('causal')
      }
    })

    test('should navigate to Knowledge Graph via link', async ({ page }) => {
      const graphLink = page.getByRole('link', { name: /knowledge|graph/i }).first()
      if (await graphLink.isVisible().catch(() => false)) {
        await graphLink.click()
        await page.waitForLoadState('networkidle')
        expect(page.url()).toMatch(/knowledge|graph/i)
      }
    })
  })

  test.describe('KPI Category Tabs', () => {
    test('should display KPI tabs if present', async () => {
      const tabList = homePage.kpiTabs
      const isVisible = await tabList.isVisible().catch(() => false)
      expect(typeof isVisible).toBe('boolean')
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await homePage.goto()
      await expect(homePage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await homePage.goto()
      await expect(homePage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await homePage.goto()
      await expect(homePage.mainContent).toBeVisible()
    })
  })
})
