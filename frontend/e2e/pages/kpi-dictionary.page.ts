import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for KPI Dictionary page.
 * Comprehensive reference for all defined KPIs with definitions, formulas, and thresholds.
 */
export class KPIDictionaryPage extends BasePage {
  readonly url = ROUTES.KPI_DICTIONARY
  readonly pageTitle = /KPI Dictionary|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  /**
   * Override base goto with an explicit wait for the KPI Dictionary heading.
   * Under load (Vite dev server + sequential beforeEach), the page can still
   * be white when assertions begin because BasePage.goto only waits on the
   * generic mainContent selector and swallows timeouts. Anchor on a
   * page-specific element to ensure the React tree mounted before each test.
   */
  async goto(): Promise<void> {
    await this.page.goto(this.url)
    await this.page.waitForLoadState('domcontentloaded')
    // Wait for the page-specific heading so subsequent locators find the
    // rendered content rather than racing the React mount.
    await this.pageHeader.waitFor({ state: 'visible', timeout: 15000 }).catch(() => {})
    // Allow React Query to populate data + interactive elements to settle.
    await this.page.waitForTimeout(300)
  }

  // Page Header
  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /KPI Dictionary/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/reference|KPIs|definitions|formulas|thresholds/i).first()
  }

  // Stats Cards
  get totalKPIsCard(): Locator {
    return this.page.getByText('Total KPIs').first()
  }

  get workstreamsCard(): Locator {
    return this.page.getByText('Workstreams').first()
  }

  get causalKPIsCard(): Locator {
    return this.page.getByText('Causal KPIs').first()
  }

  get systemStatusCard(): Locator {
    return this.page.getByText('System Status').first()
  }

  // Search
  get searchInput(): Locator {
    return this.page.getByPlaceholder(/search KPIs/i)
  }

  get filterCountText(): Locator {
    return this.page.getByText(/showing.*of.*KPIs/i).first()
  }

  // Tabs (Workstream Categories)
  get tabsList(): Locator {
    return this.page.getByRole('tablist')
  }

  get allKPIsTab(): Locator {
    return this.page.getByRole('tab', { name: /all kpis/i })
  }

  get dataQualityTab(): Locator {
    return this.page.getByRole('tab', { name: /data quality/i })
  }

  get modelPerformanceTab(): Locator {
    return this.page.getByRole('tab', { name: /model performance/i })
  }

  get triggerPerformanceTab(): Locator {
    return this.page.getByRole('tab', { name: /trigger performance/i })
  }

  get businessImpactTab(): Locator {
    return this.page.getByRole('tab', { name: /business impact/i })
  }

  get brandSpecificTab(): Locator {
    return this.page.getByRole('tab', { name: /brand-specific/i })
  }

  get causalMetricsTab(): Locator {
    return this.page.getByRole('tab', { name: /causal metrics/i })
  }

  // KPI Cards
  get kpiCards(): Locator {
    return this.page.locator('.rounded-lg.border').filter({ hasText: /Formula|Definition/i })
  }

  get kpiIdBadges(): Locator {
    return this.page.locator('.rounded').filter({ hasText: /WS\d+-|BR-|CM-/ })
  }

  // Threshold Indicators
  get targetIndicators(): Locator {
    return this.page.getByText(/Target:/i)
  }

  get warningIndicators(): Locator {
    return this.page.getByText(/Warning:/i)
  }

  get criticalIndicators(): Locator {
    return this.page.getByText(/Critical:/i)
  }

  // Footer Info
  get thresholdsInfoSection(): Locator {
    return this.page.getByText('About KPI Thresholds').first()
  }

  // Actions
  async searchKPIs(query: string): Promise<void> {
    await this.searchInput.fill(query)
  }

  async clearSearch(): Promise<void> {
    await this.searchInput.clear()
  }

  async clickTab(tabName: string): Promise<void> {
    await this.page.getByRole('tab', { name: new RegExp(tabName, 'i') }).click()
  }

  async selectWorkstream(workstream: string): Promise<void> {
    await this.page.getByRole('tab', { name: new RegExp(workstream, 'i') }).click()
  }

  // Verification methods
  async verifyStatsDisplayed(): Promise<boolean> {
    if (await this.page.getByText('Total KPIs').first().isVisible({ timeout: 10000 }).catch(() => false)) {
      return true
    }
    const stats = ['Workstreams', 'Causal KPIs', 'System Status']
    for (const stat of stats) {
      if (await this.page.getByText(stat).first().isVisible({ timeout: 2000 }).catch(() => false)) {
        return true
      }
    }
    return false
  }

  async verifyTabsDisplayed(): Promise<boolean> {
    // The page has multiple tablists - main sections and workstream tabs.
    // Wait for any tablist to be visible.
    const allTablists = this.page.getByRole('tablist')
    if (await allTablists.first().isVisible({ timeout: 10000 }).catch(() => false)) {
      return true
    }
    // Fallback: check for specific tab triggers
    const hasAllKPIs = await this.page.getByRole('tab', { name: /all kpis/i }).isVisible({ timeout: 2000 }).catch(() => false)
    const hasKPICards = await this.page.getByRole('tab', { name: /kpi cards/i }).isVisible({ timeout: 2000 }).catch(() => false)
    return hasAllKPIs || hasKPICards
  }

  async verifyKPICardsDisplayed(): Promise<boolean> {
    try {
      // Wait for at least one KPI card with formula
      await this.page.getByText('Formula').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  async verifySearchWorks(): Promise<boolean> {
    try {
      // Ensure the search input is ready before interacting; the filter count
      // text only renders once the page has populated KPI data.
      await this.searchInput.waitFor({ state: 'visible', timeout: 10000 })
      await this.searchInput.fill('ROI')
      // Wait for filter to apply and the count line to be visible.
      await this.filterCountText.waitFor({ state: 'visible', timeout: 10000 })
      const filterText = await this.filterCountText.textContent()
      // Accept either "Showing X of Y" or any non-empty match for the regex,
      // since the page renders "Showing {filteredKPIs.length} of {stats.total} KPIs".
      return !!filterText && /showing.*of.*kpis/i.test(filterText)
    } catch {
      return false
    }
  }
}
