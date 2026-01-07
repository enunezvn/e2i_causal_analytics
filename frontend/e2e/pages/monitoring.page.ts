import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for Monitoring page.
 * Displays user activity logs, API usage statistics, error tracking, and performance metrics.
 */
export class MonitoringPage extends BasePage {
  readonly url = ROUTES.MONITORING
  readonly pageTitle = /Monitoring|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  // Page Header
  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Monitoring/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/user activity|API usage|error tracking|performance/i).first()
  }

  // Selectors
  get timeRangeSelector(): Locator {
    return this.page.getByRole('combobox').first()
  }

  // Action Buttons
  get refreshButton(): Locator {
    return this.page.getByRole('button', { name: /refresh/i })
  }

  get exportButton(): Locator {
    return this.page.getByRole('button', { name: /export/i })
  }

  // Overview Metrics
  get totalRequestsCard(): Locator {
    return this.page.getByText('Total Requests').first()
  }

  get errorRateCard(): Locator {
    return this.page.getByText('Error Rate').first()
  }

  get avgLatencyCard(): Locator {
    return this.page.getByText('Avg Latency').first()
  }

  get activeUsersCard(): Locator {
    return this.page.getByText('Active Users').first()
  }

  get totalErrorsCard(): Locator {
    return this.page.getByText('Total Errors').first()
  }

  get uptimeCard(): Locator {
    return this.page.getByText('Uptime').first()
  }

  // Tabs
  get tabsList(): Locator {
    return this.page.getByRole('tablist')
  }

  get apiUsageTab(): Locator {
    return this.page.getByRole('tab', { name: /api usage/i })
  }

  get userActivityTab(): Locator {
    return this.page.getByRole('tab', { name: /user activity/i })
  }

  get errorsTab(): Locator {
    return this.page.getByRole('tab', { name: /errors/i })
  }

  get systemTab(): Locator {
    return this.page.getByRole('tab', { name: /system/i })
  }

  // API Usage Tab Content
  get requestVolumeCard(): Locator {
    return this.page.getByText('Request Volume').first()
  }

  get responseLatencyCard(): Locator {
    return this.page.getByText('Response Latency').first()
  }

  get endpointStatisticsCard(): Locator {
    return this.page.getByText('Endpoint Statistics').first()
  }

  // User Activity Tab Content
  get userActivityLogCard(): Locator {
    return this.page.getByText('User Activity Log').first()
  }

  get activitySearchInput(): Locator {
    return this.page.getByPlaceholder(/search activities/i)
  }

  // Errors Tab Content
  get errorLogsCard(): Locator {
    return this.page.getByText('Error Logs').first()
  }

  get errorSearchInput(): Locator {
    return this.page.getByPlaceholder(/search errors/i)
  }

  get errorLevelFilter(): Locator {
    return this.page.getByRole('combobox').filter({ hasText: /level|all/i }).first()
  }

  // System Tab Content
  get systemResourcesCard(): Locator {
    return this.page.getByText('System Resources').first()
  }

  get serviceHealthCard(): Locator {
    return this.page.getByText('Service Health').first()
  }

  // Actions
  async selectTimeRange(range: string): Promise<void> {
    await this.timeRangeSelector.click()
    await this.page.getByRole('option', { name: new RegExp(range, 'i') }).click()
  }

  async clickTab(tabName: string): Promise<void> {
    await this.page.getByRole('tab', { name: new RegExp(tabName, 'i') }).click()
  }

  async clickRefresh(): Promise<void> {
    await this.refreshButton.click()
  }

  async clickExport(): Promise<void> {
    await this.exportButton.click()
  }

  async searchActivities(query: string): Promise<void> {
    await this.activitySearchInput.fill(query)
  }

  async searchErrors(query: string): Promise<void> {
    await this.errorSearchInput.fill(query)
  }

  // Verification methods
  async verifyOverviewMetricsDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Total Requests').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      const metrics = ['Error Rate', 'Avg Latency', 'Active Users', 'Uptime']
      for (const metric of metrics) {
        if (await this.page.getByText(metric).first().isVisible().catch(() => false)) {
          return true
        }
      }
      return false
    }
  }

  async verifyTabsDisplayed(): Promise<boolean> {
    try {
      // Wait for main content to load first
      await this.page.waitForTimeout(1000)
      await this.tabsList.waitFor({ state: 'visible', timeout: 5000 })
      return await this.tabsList.isVisible()
    } catch {
      // Fallback: check for specific tab triggers
      try {
        const hasApiTab = await this.page.getByRole('tab', { name: /api/i }).first().isVisible({ timeout: 2000 }).catch(() => false)
        const hasActivityTab = await this.page.getByRole('tab', { name: /activity/i }).first().isVisible({ timeout: 2000 }).catch(() => false)
        const hasErrorsTab = await this.page.getByRole('tab', { name: /errors/i }).first().isVisible({ timeout: 2000 }).catch(() => false)
        return hasApiTab || hasActivityTab || hasErrorsTab
      } catch {
        return false
      }
    }
  }

  async verifyAPIUsageDisplayed(): Promise<boolean> {
    try {
      // The CardTitle text is "Request Volume & Errors" - use regex for partial match
      await this.page.getByText(/Request Volume/i).first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      // Fallback: check for other API usage elements
      try {
        const hasLatency = await this.page.getByText('Response Latency').first().isVisible({ timeout: 2000 }).catch(() => false)
        const hasEndpoint = await this.page.getByText('Endpoint Statistics').first().isVisible({ timeout: 2000 }).catch(() => false)
        return hasLatency || hasEndpoint
      } catch {
        return false
      }
    }
  }

  async verifyUserActivityDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('User Activity Log').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  async verifyErrorLogsDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Error Logs').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  async verifySystemMetricsDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('System Resources').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }
}
