import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for KPI Dictionary page.
 * Comprehensive reference for all 46 KPIs with definitions, formulas, and thresholds.
 */
export class KPIDictionaryPage extends BasePage {
  readonly url = ROUTES.KPI_DICTIONARY
  readonly pageTitle = /KPI Dictionary|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
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
    try {
      await this.page.getByText('Total KPIs').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      const stats = ['Workstreams', 'Causal KPIs', 'System Status']
      for (const stat of stats) {
        if (await this.page.getByText(stat).first().isVisible().catch(() => false)) {
          return true
        }
      }
      return false
    }
  }

  async verifyTabsDisplayed(): Promise<boolean> {
    try {
      // The page has multiple tablists - main sections and workstream tabs
      // First try to find any tablist
      const allTablists = this.page.getByRole('tablist')
      await allTablists.first().waitFor({ state: 'visible', timeout: 5000 })
      return await allTablists.first().isVisible()
    } catch {
      // Fallback: check for specific tab triggers
      try {
        const hasAllKPIs = await this.page.getByRole('tab', { name: /all kpis/i }).isVisible({ timeout: 2000 }).catch(() => false)
        const hasKPICards = await this.page.getByRole('tab', { name: /kpi cards/i }).isVisible({ timeout: 2000 }).catch(() => false)
        return hasAllKPIs || hasKPICards
      } catch {
        return false
      }
    }
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
      await this.searchInput.fill('ROI')
      // Wait for filter to apply
      await this.page.waitForTimeout(500)
      const filterText = await this.filterCountText.textContent()
      return filterText !== null && filterText.includes('Showing')
    } catch {
      return false
    }
  }
}
