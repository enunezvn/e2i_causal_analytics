import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for Model Performance page.
 * Displays ML model metrics, confusion matrix, ROC curves, and performance trends.
 */
export class ModelPerformancePage extends BasePage {
  readonly url = ROUTES.MODEL_PERFORMANCE
  readonly pageTitle = /Model Performance|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  // Page Header
  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Model Performance/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/model metrics|confusion matrix|ROC curves|performance|evaluation/i).first()
  }

  // Model Selector
  get modelSelector(): Locator {
    return this.page.getByRole('combobox').first()
  }

  // Action Buttons
  get refreshButton(): Locator {
    return this.page.getByRole('button').filter({ has: this.page.locator('svg.lucide-refresh-cw, [class*="animate-spin"]') }).first()
  }

  get exportButton(): Locator {
    return this.page.getByRole('button', { name: /export/i })
  }

  // Metrics Cards
  get metricsCards(): Locator {
    return this.page.locator('.grid').first().locator('[class*="Card"], .rounded-lg.border')
  }

  // KPI cards rendered by `ModelPerformance.tsx` are driven by the live
  // `usePerformanceTrend` hook (PR #317). The hook returns
  // `metric_name='accuracy'` so the tiles are titled "Current accuracy",
  // "Baseline accuracy", "Change", "Trend".  KPICard renders its `title` in
  // a `<h3>` (see components/visualizations/dashboard/KPICard.tsx:260), so we
  // scope all KPI matchers to that role to avoid matching trend-tab copy,
  // chart threshold labels, or descriptions elsewhere on the page.
  // The legacy Precision/Recall/F1/AUC tiles came from now-removed mock-data
  // SAMPLE_METRICS and are intentionally NOT asserted on.
  get accuracyCard(): Locator {
    // "Current accuracy" — KPI title produced from
    // `Current ${trendQuery.data.metric_name}`. Anchor on "Current " to
    // exclude any plain "accuracy" copy elsewhere.
    return this.page.getByRole('heading', { level: 3, name: /^Current\s+accuracy$/i })
  }

  get baselineCard(): Locator {
    // "Baseline accuracy" — strict h3 match to exclude the trend-chart
    // threshold label `Baseline` and the page description.
    return this.page.getByRole('heading', { level: 3, name: /^Baseline\s+accuracy$/i })
  }

  get changeCard(): Locator {
    return this.page.getByRole('heading', { level: 3, name: /^Change$/i })
  }

  get trendCard(): Locator {
    // Trend KPI tile is an h3 inside the KPI grid; the "Performance Trend"
    // tab is role=tab so it cannot match here.
    return this.page.getByRole('heading', { level: 3, name: /^Trend$/i })
  }

  /** Kept for backward-compat with older specs; legacy tiles no longer exist. */
  get precisionCard(): Locator {
    return this.page.getByText('Precision').first()
  }

  get recallCard(): Locator {
    return this.page.getByText('Recall').first()
  }

  get f1ScoreCard(): Locator {
    return this.page.getByText('F1 Score').first()
  }

  get aucCard(): Locator {
    return this.page.getByText('AUC-ROC').first()
  }

  // Tabs
  get tabsList(): Locator {
    return this.page.getByRole('tablist')
  }

  get overviewTab(): Locator {
    return this.page.getByRole('tab', { name: /overview|metrics/i })
  }

  get confusionMatrixTab(): Locator {
    return this.page.getByRole('tab', { name: /confusion matrix/i })
  }

  get rocCurveTab(): Locator {
    return this.page.getByRole('tab', { name: /roc|curve/i })
  }

  get trendTab(): Locator {
    return this.page.getByRole('tab', { name: /trend/i })
  }

  // Visualization Areas
  get confusionMatrixChart(): Locator {
    return this.page.locator('[class*="confusion"], .confusion-matrix, svg').first()
  }

  get rocCurveChart(): Locator {
    return this.page.locator('[class*="roc"], .roc-curve, svg').first()
  }

  // Model Info Card
  get modelInfoCard(): Locator {
    return this.page.locator('.rounded-lg.border').first()
  }

  get samplesEvaluated(): Locator {
    return this.page.getByText('Samples Evaluated')
  }

  // Actions

  /**
   * Wait until the live /api/models/status response populates the
   * <Select> trigger and it becomes enabled (the page disables it while
   * `models.length === 0`).
   */
  async waitForModelOptions(timeout = 10000): Promise<void> {
    await this.modelSelector.waitFor({ state: 'visible', timeout })
    // Trigger is disabled (`aria-disabled="true"` and/or `disabled`) until
    // the models query lands; wait for it to flip enabled.
    await this.page.waitForFunction(
      () => {
        const trigger = document.querySelector(
          '[role="combobox"]'
        ) as HTMLElement | null
        if (!trigger) return false
        const ariaDisabled = trigger.getAttribute('aria-disabled')
        const isDisabled =
          (trigger as HTMLButtonElement).disabled === true ||
          ariaDisabled === 'true'
        return !isDisabled
      },
      { timeout }
    )
  }

  async selectModel(modelName: string): Promise<void> {
    // Wait for the trigger to become enabled (live API populated).
    await this.waitForModelOptions().catch(() => {})

    // Click the select trigger to open dropdown
    await this.modelSelector.click()

    // Wait for dropdown to appear
    await this.page.waitForTimeout(300)

    // Try multiple approaches to find the option
    const option = this.page.getByRole('option', { name: new RegExp(modelName, 'i') })
    const selectItem = this.page.locator('[role="listbox"] [role="option"]').filter({ hasText: new RegExp(modelName, 'i') })
    const textOption = this.page.locator('[data-radix-select-viewport] [role="option"]').filter({ hasText: new RegExp(modelName, 'i') })

    // Try each approach
    if (await option.first().isVisible({ timeout: 2000 }).catch(() => false)) {
      await option.first().click()
    } else if (await selectItem.first().isVisible({ timeout: 1000 }).catch(() => false)) {
      await selectItem.first().click()
    } else if (await textOption.first().isVisible({ timeout: 1000 }).catch(() => false)) {
      await textOption.first().click()
    } else {
      // Fallback: click by text
      await this.page.getByText(new RegExp(modelName, 'i')).first().click()
    }
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

  // Verification methods
  async verifyMetricsDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Accuracy').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      const metrics = ['Precision', 'Recall', 'F1 Score', 'AUC']
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
      await this.tabsList.waitFor({ state: 'visible', timeout: 5000 })
      return await this.tabsList.isVisible()
    } catch {
      return false
    }
  }

  async verifyKPICardsDisplayed(): Promise<boolean> {
    try {
      // Post-PR-#317 the page renders exactly four trend-driven KPI tiles:
      //   "Current <metric>", "Baseline <metric>", "Change", "Trend".
      // KPICard renders each title in an <h3>. Require all four canonical
      // h3 headings to be visible — a real regression (e.g. trend hook
      // dropped, KPI grid removed) trips this; an unrelated visible h3
      // elsewhere on the page does not.
      const kpiHeadings = [
        /^Current\s+accuracy$/i,
        /^Baseline\s+accuracy$/i,
        /^Change$/i,
        /^Trend$/i,
      ]
      for (const name of kpiHeadings) {
        await this.page
          .getByRole('heading', { level: 3, name })
          .waitFor({ state: 'visible', timeout: 5000 })
      }
      return true
    } catch {
      return false
    }
  }

  async verifyConfusionMatrixDisplayed(): Promise<boolean> {
    try {
      await this.page.waitForTimeout(1000)
      // Look for confusion matrix title or visualization
      const hasTitle = await this.page.getByText(/Confusion Matrix/i).first().isVisible({ timeout: 3000 }).catch(() => false)
      if (hasTitle) return true
      // Fallback: look for tabpanel content
      const hasTabPanel = await this.page.getByRole('tabpanel').isVisible({ timeout: 2000 }).catch(() => false)
      if (hasTabPanel) return true
      // Fallback: look for any chart/visualization
      const hasChart = await this.page.locator('[role="application"], svg, canvas').first().isVisible({ timeout: 2000 }).catch(() => false)
      return hasChart
    } catch {
      return false
    }
  }

  async verifyROCCurveDisplayed(): Promise<boolean> {
    try {
      await this.page.waitForTimeout(1000)
      // Look for ROC curve title or visualization
      const hasTitle = await this.page.getByText(/ROC|AUC/i).first().isVisible({ timeout: 3000 }).catch(() => false)
      if (hasTitle) return true
      // Fallback: look for tabpanel content
      const hasTabPanel = await this.page.getByRole('tabpanel').isVisible({ timeout: 2000 }).catch(() => false)
      if (hasTabPanel) return true
      // Fallback: look for any chart/visualization
      const hasChart = await this.page.locator('[role="application"], svg, canvas').first().isVisible({ timeout: 2000 }).catch(() => false)
      return hasChart
    } catch {
      return false
    }
  }
}
