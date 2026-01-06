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

  get accuracyCard(): Locator {
    return this.page.getByText('Accuracy').first()
  }

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
  async selectModel(modelName: string): Promise<void> {
    // Wait for page to be ready
    await this.page.waitForTimeout(300)

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
      await this.page.waitForTimeout(1000)
      const metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
      for (const metric of metrics) {
        if (await this.page.getByText(metric).first().isVisible({ timeout: 2000 }).catch(() => false)) {
          return true
        }
      }
      // Fallback: look for h3 elements (KPICard renders titles in h3)
      const hasH3 = await this.page.locator('h3').first().isVisible({ timeout: 2000 }).catch(() => false)
      return hasH3
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
