import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the Time Series (KPI history) page.
 *
 * NOTE: The page is now the single-mode KPI-history home. Its former
 * "Model performance" mode (walk-forward backtest trends per cohort ×
 * brand, the `#ts-cohort` / `#ts-brand` selects and the mode tabs) moved
 * to the Model Performance page — see the sibling PR. The current UI is:
 * KPI select + time-range select in the header, five summary stat cards,
 * the "KPI History" chart (with an honest empty-state when a KPI has no
 * real history), and the "Current KPI Status" card. See
 * `time-series.spec.ts` for inline KPI-history mocks; do not modify
 * `e2e/fixtures/api-mocks.ts` per the agent contract on #332.
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
    // "KPI metric history over time."
    return this.page.getByText(/KPI metric history/i).first()
  }

  // Selectors. The header renders the KPI Select first and the time-range
  // Select second. shadcn SelectTrigger surfaces as role="combobox" and both
  // carry aria-labels ("kpi" / "time range").
  get kpiSelector(): Locator {
    return this.page.getByRole('combobox', { name: /^kpi$/i })
  }

  get timeRangeSelector(): Locator {
    return this.page.getByRole('combobox', { name: /time range/i })
  }

  // Action Buttons. The refresh / export buttons have aria-labels in
  // TimeSeries.tsx (`Refresh` / `Export`).
  get refreshButton(): Locator {
    return this.page.getByRole('button', { name: /refresh/i }).first()
  }

  get exportButton(): Locator {
    return this.page.getByRole('button', { name: /export/i }).first()
  }

  // Summary stat cards. The five cards are titled "Current Value",
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

  // KPI history content
  get kpiHistoryCard(): Locator {
    return this.page.getByText('KPI History').first()
  }

  get currentKpiStatusCard(): Locator {
    return this.page.getByText('Current KPI Status').first()
  }

  // Honest empty-state shown when a KPI has no real history rows.
  get kpiHistoryEmptyState(): Locator {
    return this.page.getByTestId('kpi-history-empty')
  }

  // Recharts surface for the history chart.
  get chartSurface(): Locator {
    return this.page.locator('svg.recharts-surface').first()
  }

  // Actions
  async selectKpi(kpiName: string): Promise<void> {
    await this.kpiSelector.click()
    await this.page.getByRole('option', { name: new RegExp(kpiName, 'i') }).click()
  }

  async selectTimeRange(range: string): Promise<void> {
    await this.timeRangeSelector.click()
    await this.page.getByRole('option', { name: new RegExp(range, 'i') }).click()
  }

  async clickRefresh(): Promise<void> {
    await this.refreshButton.click()
  }

  async clickExport(): Promise<void> {
    await this.exportButton.click()
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

  /**
   * Validate that the KPI history chart is rendered. Requires BOTH the
   * "KPI History" card title AND the recharts SVG to be visible — the card
   * title is unconditional but does not by itself prove the chart actually
   * rendered. No page-header fallback: that would let this method pass when
   * the hook payload is malformed and the chart never mounts.
   */
  async verifyHistoryChartDisplayed(): Promise<boolean> {
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
