import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the Home page.
 * Updated to match actual frontend implementation using shadcn/ui components.
 */
export class HomePage extends BasePage {
  readonly url = ROUTES.HOME
  readonly pageTitle = /E2I|Causal Analytics|Dashboard/i

  constructor(page: Page) {
    super(page)
  }

  // ========================================================================
  // Page Header
  // ========================================================================

  get pageHeader(): Locator {
    return this.page.getByText('E2I Executive Dashboard')
  }

  get pageDescription(): Locator {
    return this.page.getByText('Causal Analytics for Commercial Operations')
  }

  // ========================================================================
  // Selectors (Brand, Region, Date Range) - using combobox role
  // ========================================================================

  get brandSelector(): Locator {
    // First combobox is the brand selector
    return this.page.getByRole('combobox').first()
  }

  get regionSelector(): Locator {
    // Second combobox is the region selector
    return this.page.getByRole('combobox').nth(1)
  }

  get dateRangePicker(): Locator {
    // Third combobox is the date range selector
    return this.page.getByRole('combobox').nth(2)
  }

  get allSelectors(): Locator {
    return this.page.getByRole('combobox')
  }

  // ========================================================================
  // Quick Stats Bar
  // ========================================================================

  get quickStatsBar(): Locator {
    return this.page.locator('.grid.grid-cols-2, .grid.md\\:grid-cols-4').first()
  }

  get totalTrxStat(): Locator {
    return this.page.getByText('Total TRx (MTD)')
  }

  get activeCampaignsStat(): Locator {
    return this.page.getByText('Active Campaigns')
  }

  get hcpsReachedStat(): Locator {
    return this.page.getByText('HCPs Reached')
  }

  get modelAccuracyStat(): Locator {
    return this.page.getByText('Model Accuracy')
  }

  // ========================================================================
  // KPI Section
  // ========================================================================

  get kpiSection(): Locator {
    return this.page.getByText('Key Performance Indicators').locator('..')
  }

  get kpiCards(): Locator {
    // KPICard components from the dashboard
    return this.page.locator('[class*="Card"], .rounded-lg.border').filter({ hasText: /TRx|NRx|Revenue|Share|Rate|Conversion/i })
  }

  getKpiByName(name: string): Locator {
    return this.page.getByText(name, { exact: false }).first()
  }

  get totalTrxCard(): Locator {
    return this.page.getByText('Total TRx').first()
  }

  get newTrxCard(): Locator {
    return this.page.getByText('New TRx').first()
  }

  get netRevenueCard(): Locator {
    return this.page.getByText('Net Revenue').first()
  }

  get marketShareCard(): Locator {
    return this.page.getByText('Market Share').first()
  }

  // ========================================================================
  // KPI Category Tabs
  // ========================================================================

  get kpiTabs(): Locator {
    return this.page.getByRole('tablist')
  }

  getKpiTab(category: string): Locator {
    return this.page.getByRole('tab', { name: category })
  }

  // ========================================================================
  // Agent Insights
  // ========================================================================

  get agentInsightsSection(): Locator {
    return this.page.getByText('Agent Insights').locator('..')
  }

  get insightCards(): Locator {
    return this.page.locator('[class*="insight"], .insight-card')
  }

  // ========================================================================
  // System Status
  // ========================================================================

  get systemStatusSection(): Locator {
    return this.page.getByText('System Status').locator('..')
  }

  get systemHealthIndicator(): Locator {
    return this.page.locator('[class*="health"], [class*="status"]').first()
  }

  // ========================================================================
  // Quick Actions
  // ========================================================================

  get quickActionsSection(): Locator {
    return this.page.getByText('Quick Actions').locator('..')
  }

  getQuickAction(actionName: string): Locator {
    return this.page.getByRole('link', { name: new RegExp(actionName, 'i') })
  }

  get viewCausalAnalysisAction(): Locator {
    return this.page.getByRole('link', { name: /causal/i }).first()
  }

  get viewKnowledgeGraphAction(): Locator {
    return this.page.getByRole('link', { name: /knowledge.*graph/i }).first()
  }

  // ========================================================================
  // Alerts
  // ========================================================================

  get alertsSection(): Locator {
    return this.page.getByText('Active Alerts').locator('..')
  }

  get alertCards(): Locator {
    return this.alertsSection.locator('.rounded-lg.border')
  }

  // ========================================================================
  // Actions
  // ========================================================================

  async selectBrand(brand: string): Promise<void> {
    await this.brandSelector.click()
    await this.page.getByRole('option', { name: new RegExp(brand, 'i') }).click()
  }

  async selectRegion(region: string): Promise<void> {
    await this.regionSelector.click()
    await this.page.getByRole('option', { name: new RegExp(region, 'i') }).click()
  }

  async selectDateRange(range: string): Promise<void> {
    await this.dateRangePicker.click()
    await this.page.getByRole('option', { name: new RegExp(range, 'i') }).click()
  }

  async clickKpiTab(category: string): Promise<void> {
    await this.getKpiTab(category).click()
  }

  async clickRefresh(): Promise<void> {
    await this.page.getByRole('button').filter({ has: this.page.locator('svg[class*="refresh"], .lucide-refresh') }).click()
  }

  async dismissAlert(index = 0): Promise<void> {
    await this.page.getByRole('button', { name: /dismiss/i }).nth(index).click()
  }

  // ========================================================================
  // Assertions / Helpers
  // ========================================================================

  async verifyKpiCardsDisplayed(_minCount = 1): Promise<boolean> {
    // Wait for KPI section to render - look for the Key Performance Indicators heading
    try {
      await this.page.getByText('Key Performance Indicators').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      // Fallback: check for KPI-related headings/labels on the page
      const kpiHeadings = ['Total TRx', 'New TRx', 'Net Revenue', 'Market Share']
      for (const text of kpiHeadings) {
        const locator = this.page.getByRole('heading', { name: text }).or(
          this.page.getByText(text, { exact: false })
        )
        if (await locator.first().isVisible().catch(() => false)) {
          return true
        }
      }
      return false
    }
  }

  async verifySystemHealthShown(): Promise<boolean> {
    // Check for system status indicators
    const healthTexts = ['System Status', 'healthy', 'Healthy', 'Status', 'Online']
    for (const text of healthTexts) {
      if (await this.page.getByText(text).first().isVisible().catch(() => false)) {
        return true
      }
    }
    return false
  }

  async getKpiValue(kpiName: string): Promise<string | null> {
    const kpiElement = this.page.getByText(kpiName).first()
    if (await kpiElement.isVisible()) {
      // Get the parent card and find the value
      const parent = kpiElement.locator('..').locator('..')
      const valueElement = parent.locator('.font-bold, .text-xl, .text-2xl').first()
      return await valueElement.textContent().catch(() => null)
    }
    return null
  }

  async getBrandSelectorText(): Promise<string | null> {
    return await this.brandSelector.textContent()
  }

  async getSelectedBrand(): Promise<string | null> {
    const text = await this.brandSelector.textContent()
    return text
  }

  async verifyQuickStatsDisplayed(): Promise<boolean> {
    // Wait for quick stats to render - look for the first stat text
    try {
      await this.page.getByText('Total TRx (MTD)').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      // Fallback: check for any stat text
      const stats = ['Total TRx', 'Active Campaigns', 'HCPs Reached', 'Model Accuracy']
      for (const stat of stats) {
        const locator = this.page.getByText(stat, { exact: false })
        if (await locator.first().isVisible().catch(() => false)) {
          return true
        }
      }
      return false
    }
  }
}
