import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the Time Series Analysis page.
 *
 * NOTE (post-PR #313, issue #302): The page was rewired onto the live
 * `/api/monitoring/performance/{model_id}/trend` + `/api/kpis/{id}` endpoints,
 * and the 38 `sample*` constants were retired. The current UI exposes two
 * mode tabs — "Model performance" and "KPI history" — instead of the
 * earlier "Trends & Forecast / Seasonality / Anomalies / Comparison" tabs.
 * This page object reflects that contract. See `time-series.spec.ts` for
 * inline performance-trend mocks; do not modify `e2e/fixtures/api-mocks.ts`
 * per the agent contract on #332.
 */
export class TimeSeriesPage extends BasePage {
  readonly url = ROUTES.TIME_SERIES
  readonly pageTitle = /Time Series|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  /**
   * Override the base `goto()` to wait for the React-lazy chunk to mount.
   * The TimeSeries page is `lazy()`-imported (see `src/router/routes.tsx`)
   * and the Suspense fallback renders a generic spinner inside `<main>`,
   * so the base-page heuristic returns prematurely. Wait for the `<h1>`
   * explicitly before letting assertions run.
   */
  async goto(): Promise<void> {
    await super.goto()
    // Allow the lazy chunk + initial render to settle. The page header
    // is unconditional inside TimeSeries.tsx — its visibility is the
    // canonical signal that the lazy module has mounted.
    await this.pageHeader.waitFor({ state: 'visible', timeout: 20000 }).catch(() => {})
  }

  // Page Header
  get pageHeader(): Locator {
    // <h1>Time Series Analysis</h1> in TimeSeries.tsx
    return this.page.getByRole('heading', { name: /Time Series/i }).first()
  }

  get pageDescription(): Locator {
    // "Time series trends, forecasting, seasonality decomposition, and anomaly detection."
    return this.page.getByText(/trends|forecasting|seasonality|anomaly/i).first()
  }

  // Selectors. The header renders the metric Select (or KPI Select in KPI mode)
  // first and the time-range Select second. shadcn SelectTrigger surfaces as
  // role="combobox".
  get metricSelector(): Locator {
    return this.page.getByRole('combobox').first()
  }

  get timeRangeSelector(): Locator {
    return this.page.getByRole('combobox').nth(1)
  }

  // Action Buttons. The refresh / export buttons have aria-labels in the
  // new TimeSeries.tsx (`Refresh` / `Export`).
  get refreshButton(): Locator {
    return this.page.getByRole('button', { name: /refresh/i }).first()
  }

  get exportButton(): Locator {
    return this.page.getByRole('button', { name: /export/i }).first()
  }

  // KPI Summary Cards (post-#302). The five cards are titled "Current Value",
  // "Average", "Maximum", "Minimum", "Data Points".
  get currentValueCard(): Locator {
    return this.page.getByText('Current Value').first()
  }

  get averageCard(): Locator {
    return this.page.getByText('Average').first()
  }

  get maximumCard(): Locator {
    return this.page.getByText('Maximum').first()
  }

  get minimumCard(): Locator {
    return this.page.getByText('Minimum').first()
  }

  get dataPointsCard(): Locator {
    return this.page.getByText('Data Points').first()
  }

  /**
   * "Trend stat" — the new UI surfaces a "Trend Summary" card whose body
   * contains a `<p>Trend</p>` label (alongside Current / Baseline / Change
   * labels). Match the card title to keep the assertion stable across
   * styling changes — the title is unconditional within the trend-summary
   * card and is rendered once `performanceTrend.data` resolves.
   */
  get trendCard(): Locator {
    return this.page.getByText('Trend Summary').first()
  }

  // Tabs (post-#302): "Model performance" / "KPI history"
  get tabsList(): Locator {
    return this.page.getByRole('tablist')
  }

  get modelPerformanceTab(): Locator {
    return this.page.getByRole('tab', { name: /model performance/i })
  }

  get kpiHistoryTab(): Locator {
    return this.page.getByRole('tab', { name: /kpi history/i })
  }

  /**
   * Legacy alias for "Trend tab" — the first/primary tab on the page.
   * The post-#302 UI calls this "Model performance".
   */
  get trendTab(): Locator {
    return this.modelPerformanceTab
  }

  /**
   * Legacy alias for "Seasonality tab" — the second tab on the page.
   * The post-#302 UI calls this "KPI history".
   */
  get seasonalityTab(): Locator {
    return this.kpiHistoryTab
  }

  /**
   * Legacy alias for "Anomalies tab" — the page no longer surfaces a
   * dedicated anomalies tab, so we fall back to the KPI history tab.
   */
  get anomaliesTab(): Locator {
    return this.kpiHistoryTab
  }

  // Performance mode content (active tab on load)
  get performanceTrendCard(): Locator {
    return this.page.getByText('Performance Trend').first()
  }

  get trendSummaryCard(): Locator {
    return this.page.getByText('Trend Summary').first()
  }

  // KPI mode content
  get kpiHistoryCard(): Locator {
    return this.page.getByText('KPI History').first()
  }

  // Recharts surface — present in both modes.
  get chartSurface(): Locator {
    return this.page.locator('svg.recharts-surface').first()
  }

  // Model selector (footer card, performance mode only).
  get modelIdInput(): Locator {
    return this.page.getByLabel(/model id/i).first()
  }

  // Actions
  async selectMetric(metricName: string): Promise<void> {
    await this.metricSelector.click()
    await this.page.getByRole('option', { name: new RegExp(metricName, 'i') }).click()
  }

  async selectTimeRange(range: string): Promise<void> {
    await this.timeRangeSelector.click()
    await this.page.getByRole('option', { name: new RegExp(range, 'i') }).click()
  }

  async clickTab(tabName: string): Promise<void> {
    // Map legacy tab names to the new UI labels so existing specs keep working.
    const legacyMap: Record<string, RegExp> = {
      Trend: /model performance/i,
      Trends: /model performance/i,
      Seasonality: /kpi history/i,
      Anomalies: /kpi history/i,
    }
    const matcher = legacyMap[tabName] ?? new RegExp(tabName, 'i')
    await this.page.getByRole('tab', { name: matcher }).first().click()
  }

  async clickRefresh(): Promise<void> {
    await this.refreshButton.click()
  }

  async clickExport(): Promise<void> {
    await this.exportButton.click()
  }

  /**
   * Enter a model ID into the free-text "Model ID" field (performance mode).
   * `usePerformanceTrend` is `enabled: !!model_id` and `DEFAULT_MODEL_ID` is
   * '', so the performance view — KPI summary stats, the chart line, and the
   * "Trend Summary" card — stays empty until a model is entered. Driving this
   * the way a real user does is what enables the trend fetch (and lets the
   * spec's inline `/trend` mock fulfil it).
   */
  async enterModelId(modelId: string): Promise<void> {
    await this.modelIdInput.fill(modelId)
  }

  // Verification helpers
  async verifyKPICardsDisplayed(): Promise<boolean> {
    try {
      // Use 10s timeout to match the default `toBeVisible` wait — the
      // large dist bundle can delay KPICard hydration past 5s in CI.
      await this.currentValueCard.waitFor({ state: 'visible', timeout: 10000 })
      return true
    } catch {
      // Fallback: any of the other four cards is sufficient.
      const fallbacks = [
        this.averageCard,
        this.maximumCard,
        this.minimumCard,
        this.dataPointsCard,
      ]
      for (const card of fallbacks) {
        if (await card.isVisible().catch(() => false)) return true
      }
      return false
    }
  }

  async verifyTabsDisplayed(): Promise<boolean> {
    try {
      await this.tabsList.waitFor({ state: 'visible', timeout: 10000 })
      return await this.tabsList.isVisible()
    } catch {
      return false
    }
  }

  /**
   * Validate that the active tab's chart is rendered. Requires BOTH the
   * "Performance Trend" card title AND the recharts SVG to be visible —
   * the card title is unconditional within `<TabsContent value="performance">`
   * but does not by itself prove the chart actually rendered. No page-header
   * fallback: that would let this method pass when the hook payload is
   * malformed and the chart never mounts.
   */
  async verifyTrendChartDisplayed(): Promise<boolean> {
    try {
      const cardVisible = await this.performanceTrendCard
        .isVisible({ timeout: 10000 })
        .catch(() => false)
      if (!cardVisible) return false
      return await this.chartSurface.isVisible({ timeout: 10000 }).catch(() => false)
    } catch {
      return false
    }
  }

  /**
   * Validate that the "KPI history" tab content renders after a switch.
   * Requires BOTH the "KPI History" card AND the recharts SVG to be
   * visible. No page-header fallback (see `verifyTrendChartDisplayed`).
   */
  async verifyDecompositionDisplayed(): Promise<boolean> {
    try {
      const cardVisible = await this.kpiHistoryCard
        .isVisible({ timeout: 10000 })
        .catch(() => false)
      if (!cardVisible) return false
      return await this.chartSurface.isVisible({ timeout: 10000 }).catch(() => false)
    } catch {
      return false
    }
  }
}
