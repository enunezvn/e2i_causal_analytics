import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for Monitoring page.
 *
 * Backend: `src/api/routes/monitoring.py`
 *   - `/api/monitoring/runs`
 *   - `/api/monitoring/alerts`
 *   - `/api/monitoring/health/{model_id}`
 *
 * After PR #318 wired this page to live data, the rendered UI is no longer
 * the legacy "Total Requests / Error Rate / Avg Latency / Active Users" mock.
 * Selectors here reflect the live KPI cards + tabs (see Monitoring.tsx).
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

  // Selectors — the page has TWO comboboxes (Model + Time Range), so always
  // target by aria-label to avoid ordering accidents.
  get modelSelector(): Locator {
    return this.page.getByRole('combobox', { name: /Model/i })
  }

  get timeRangeSelector(): Locator {
    return this.page.getByRole('combobox', { name: /Time Range/i })
  }

  // Action Buttons
  get refreshButton(): Locator {
    return this.page.getByRole('button', { name: /refresh/i })
  }

  get exportButton(): Locator {
    return this.page.getByRole('button', { name: /export/i })
  }

  // Overview Metrics (live KPI cards from Monitoring.tsx).
  get totalRunsCard(): Locator {
    return this.page.getByText('Total Runs').first()
  }

  get driftRateCard(): Locator {
    return this.page.getByText('Drift Rate').first()
  }

  get avgRunDurationCard(): Locator {
    return this.page.getByText('Avg Run Duration').first()
  }

  get activeAlertsCard(): Locator {
    return this.page.getByText('Active Alerts').first()
  }

  get driftEventsCard(): Locator {
    return this.page.getByText('Drift Events').first()
  }

  get healthScoreCard(): Locator {
    return this.page.getByText('Health Score').first()
  }

  // Tabs
  get tabsList(): Locator {
    return this.page.getByRole('tablist')
  }

  get driftTrendTab(): Locator {
    // #996 relabeled this tab "API Usage" → "Drift Trend" (it charts drift
    // run-telemetry, not API usage). Locate by the live tab text.
    return this.page.getByRole('tab', { name: /drift trend/i })
  }

  get runsTab(): Locator {
    // The "Runs" tab is the live equivalent of the legacy "User Activity" tab
    // (TabsTrigger value="activity" but label "Runs"). Use exact match so we
    // don't accidentally hit the heading "Recent Runs" if it were ever a tab.
    return this.page.getByRole('tab', { name: /^Runs$/i })
  }

  get errorsTab(): Locator {
    return this.page.getByRole('tab', { name: /errors/i })
  }

  get systemTab(): Locator {
    return this.page.getByRole('tab', { name: /system/i })
  }

  // Drift Trend Tab Content
  get featuresCheckedCard(): Locator {
    // Live CardTitle: "Features Checked & Drift Detected"
    return this.page.getByText(/Features Checked/i).first()
  }

  get runDurationCard(): Locator {
    return this.page.getByText('Run Duration').first()
  }

  get recentRunsCard(): Locator {
    return this.page.getByText('Recent Runs').first()
  }

  // Runs Tab Content (live equivalent of "User Activity")
  get monitoringRunsCard(): Locator {
    return this.page.getByText('Monitoring Runs').first()
  }

  // Errors Tab Content (live equivalent of "Error Logs")
  get alertFeedCard(): Locator {
    return this.page.getByText('Alert Feed').first()
  }

  get errorSearchInput(): Locator {
    return this.page.getByPlaceholder(/search alerts/i)
  }

  get errorLevelFilter(): Locator {
    return this.page.getByRole('combobox', { name: /Severity Filter/i })
  }

  // System Tab Content (live equivalent of "System Resources")
  get modelHealthCard(): Locator {
    return this.page.getByText('Model Health').first()
  }

  get recommendationsCard(): Locator {
    return this.page.getByText('Recommendations').first()
  }

  // Actions
  async selectTimeRange(range: string): Promise<void> {
    await this.timeRangeSelector.click()
    await this.page.getByRole('option', { name: new RegExp(range, 'i') }).click()
  }

  async selectModel(modelLabel: string): Promise<void> {
    await this.modelSelector.click()
    await this.page.getByRole('option', { name: new RegExp(modelLabel, 'i') }).click()
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

  async searchAlerts(query: string): Promise<void> {
    await this.errorSearchInput.fill(query)
  }

  // Verification methods
  async verifyOverviewMetricsDisplayed(): Promise<boolean> {
    // At least one of the live KPI cards must be visible.
    //
    // NB: We use `waitFor({ state: 'visible' })` (not `.isVisible()`) because
    // `.isVisible()` is a single synchronous check that doesn't auto-retry on
    // visibility — it only retries the locator resolution. When the page is
    // still hydrating, `.isVisible()` can return `false` immediately while
    // `waitFor` polls until the element resolves visible OR the timeout fires.
    const cards = ['Total Runs', 'Drift Rate', 'Avg Run Duration', 'Active Alerts', 'Drift Events', 'Health Score']
    for (const c of cards) {
      try {
        await this.page.getByText(c).first().waitFor({ state: 'visible', timeout: 5000 })
        return true
      } catch {
        // try next card
      }
    }
    return false
  }

  async verifyTabsDisplayed(): Promise<boolean> {
    try {
      await this.page.waitForTimeout(500)
      await this.tabsList.waitFor({ state: 'visible', timeout: 5000 })
      return await this.tabsList.isVisible()
    } catch {
      const hasDriftTab = await this.page
        .getByRole('tab', { name: /drift/i })
        .first()
        .isVisible({ timeout: 2000 })
        .catch(() => false)
      const hasErrorsTab = await this.page
        .getByRole('tab', { name: /errors/i })
        .first()
        .isVisible({ timeout: 2000 })
        .catch(() => false)
      return hasDriftTab || hasErrorsTab
    }
  }

  async verifyDriftTrendDisplayed(): Promise<boolean> {
    try {
      // Live CardTitle: "Features Checked & Drift Detected"
      await this.page
        .getByText(/Features Checked/i)
        .first()
        .waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      const hasRunDuration = await this.page
        .getByText('Run Duration')
        .first()
        .isVisible({ timeout: 2000 })
        .catch(() => false)
      const hasRecentRuns = await this.page
        .getByText('Recent Runs')
        .first()
        .isVisible({ timeout: 2000 })
        .catch(() => false)
      return hasRunDuration || hasRecentRuns
    }
  }

  async verifyRunsDisplayed(): Promise<boolean> {
    try {
      await this.page
        .getByText('Monitoring Runs')
        .first()
        .waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  async verifyAlertFeedDisplayed(): Promise<boolean> {
    try {
      await this.page
        .getByText('Alert Feed')
        .first()
        .waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  async verifyModelHealthDisplayed(): Promise<boolean> {
    try {
      await this.page
        .getByText('Model Health')
        .first()
        .waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }
}
