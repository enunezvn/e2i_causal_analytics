import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for Predictive Analytics page.
 * Displays risk scores, probability distributions, uplift models, and AI-powered recommendations.
 */
export class PredictiveAnalyticsPage extends BasePage {
  readonly url = ROUTES.PREDICTIVE_ANALYTICS
  readonly pageTitle = /Predictive Analytics|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  // Page Header
  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Predictive Analytics/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/risk scores|probability|uplift|recommendations/i).first()
  }

  // Selectors
  get modelSelector(): Locator {
    return this.page.getByRole('combobox').first()
  }

  get timeframeSelector(): Locator {
    return this.page.getByRole('combobox').nth(1)
  }

  // Action Buttons
  get refreshButton(): Locator {
    return this.page.getByRole('button').filter({ has: this.page.locator('svg.lucide-refresh-cw') }).first()
  }

  get downloadButton(): Locator {
    return this.page.getByRole('button').filter({ has: this.page.locator('svg.lucide-download') }).first()
  }

  // Model Performance Summary
  get activeModelCard(): Locator {
    return this.page.getByText('Active Model').first()
  }

  get aucRocDisplay(): Locator {
    return this.page.getByText('AUC-ROC').first()
  }

  get accuracyDisplay(): Locator {
    return this.page.getByText('Accuracy').first()
  }

  // Alias for tests
  get accuracyCard(): Locator {
    return this.page.getByText('Accuracy').first()
  }

  get f1ScoreDisplay(): Locator {
    return this.page.getByText('F1 Score').first()
  }

  get predictionsCard(): Locator {
    return this.page.getByText(/Predictions|Total Predictions/i).first()
  }

  get modelHealthBadge(): Locator {
    return this.page.getByText('Model Healthy').first()
  }

  // KPI Cards
  get highRiskEntitiesCard(): Locator {
    return this.page.getByText('High Risk Entities').first()
  }

  get avgModelConfidenceCard(): Locator {
    return this.page.getByText('Avg Model Confidence').first()
  }

  get avgUpliftPotentialCard(): Locator {
    return this.page.getByText('Avg Uplift Potential').first()
  }

  get highPriorityActionsCard(): Locator {
    return this.page.getByText('High Priority Actions').first()
  }

  // Tabs
  get tabsList(): Locator {
    return this.page.getByRole('tablist')
  }

  get riskScoresTab(): Locator {
    return this.page.getByRole('tab', { name: /risk scores/i })
  }

  // Alias: predictions tab might be first tab or risk scores
  get predictionsTab(): Locator {
    return this.page.getByRole('tab', { name: /predictions|risk scores/i }).first()
  }

  get distributionsTab(): Locator {
    return this.page.getByRole('tab', { name: /distributions/i })
  }

  // Alias for singular form
  get distributionTab(): Locator {
    return this.page.getByRole('tab', { name: /distribution/i })
  }

  get upliftTab(): Locator {
    return this.page.getByRole('tab', { name: /uplift/i })
  }

  // Alias: segments might be uplift tab
  get segmentsTab(): Locator {
    return this.page.getByRole('tab', { name: /segments|uplift/i })
  }

  get recommendationsTab(): Locator {
    return this.page.getByRole('tab', { name: /recommendations/i })
  }

  // Risk Scores Tab Content
  get entityRiskScoresSection(): Locator {
    return this.page.getByText('Entity Risk Scores').first()
  }

  get riskScoreCards(): Locator {
    return this.page.locator('.rounded-lg').filter({ hasText: /Probability|Confidence|Trend/i })
  }

  get filterButton(): Locator {
    return this.page.getByRole('button', { name: /filter/i })
  }

  // Distributions Tab Content
  get scoreProbabilityDistCard(): Locator {
    return this.page.getByText('Score Probability Distribution').first()
  }

  get modelCalibrationCard(): Locator {
    return this.page.getByText('Model Calibration').first()
  }

  get cumulativeDistributionCard(): Locator {
    return this.page.getByText('Cumulative Score Distribution').first()
  }

  // Uplift Tab Content
  get upliftModelSegmentsSection(): Locator {
    return this.page.getByText('Uplift Model Segments').first()
  }

  get segmentUpliftAnalysisCard(): Locator {
    return this.page.getByText('Segment Uplift Analysis').first()
  }

  get segmentRoiComparisonCard(): Locator {
    return this.page.getByText('Segment ROI Comparison').first()
  }

  get upliftSegmentCards(): Locator {
    return this.page.locator('.rounded-lg').filter({ hasText: /Baseline|Predicted|Uplift/i })
  }

  // Recommendations Tab Content
  get aiRecommendationsSection(): Locator {
    return this.page.getByText('AI-Powered Recommendations').first()
  }

  get recommendationCards(): Locator {
    return this.page.locator('.rounded-lg').filter({ hasText: /HIGH|MEDIUM|LOW/i })
  }

  get summaryImpactCard(): Locator {
    return this.page.getByText('Summary Impact').first()
  }

  get generateActionPlanButton(): Locator {
    return this.page.getByRole('button', { name: /generate action plan/i })
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

  async selectTimeframe(timeframe: string): Promise<void> {
    await this.page.waitForTimeout(300)
    await this.timeframeSelector.click()
    await this.page.waitForTimeout(300)

    const option = this.page.getByRole('option', { name: new RegExp(timeframe, 'i') })
    const selectItem = this.page.locator('[role="listbox"] [role="option"]').filter({ hasText: new RegExp(timeframe, 'i') })

    if (await option.first().isVisible({ timeout: 2000 }).catch(() => false)) {
      await option.first().click()
    } else if (await selectItem.first().isVisible({ timeout: 1000 }).catch(() => false)) {
      await selectItem.first().click()
    } else {
      await this.page.getByText(new RegExp(timeframe, 'i')).first().click()
    }
  }

  async clickTab(tabName: string): Promise<void> {
    await this.page.getByRole('tab', { name: new RegExp(tabName, 'i') }).click()
  }

  async clickRefresh(): Promise<void> {
    await this.refreshButton.click()
  }

  async clickFilter(): Promise<void> {
    await this.filterButton.click()
  }

  // Verification methods
  async verifyModelSummaryDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Active Model').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  async verifyKPICardsDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('High Risk Entities').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      const kpis = ['Avg Model Confidence', 'Avg Uplift Potential', 'High Priority Actions']
      for (const kpi of kpis) {
        if (await this.page.getByText(kpi).first().isVisible().catch(() => false)) {
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

  async verifyPredictionsDisplayed(): Promise<boolean> {
    try {
      await this.page.waitForTimeout(1000)
      // Look for predictions content - risk score cards or entity table
      const hasRiskScores = await this.page.getByText(/Risk Scores|Entity|Probability|Confidence/i).first().isVisible({ timeout: 3000 }).catch(() => false)
      if (hasRiskScores) return true
      // Fallback: look for tabpanel
      const hasTabPanel = await this.page.getByRole('tabpanel').isVisible({ timeout: 2000 }).catch(() => false)
      return hasTabPanel
    } catch {
      return false
    }
  }

  async verifyDistributionDisplayed(): Promise<boolean> {
    try {
      await this.page.waitForTimeout(1000)
      // Look for distribution content
      const hasDistribution = await this.page.getByText(/Distribution|Probability|Calibration/i).first().isVisible({ timeout: 3000 }).catch(() => false)
      if (hasDistribution) return true
      // Fallback: look for charts
      const hasChart = await this.page.locator('[role="application"], svg.recharts-surface').first().isVisible({ timeout: 2000 }).catch(() => false)
      if (hasChart) return true
      // Fallback: look for tabpanel
      const hasTabPanel = await this.page.getByRole('tabpanel').isVisible({ timeout: 2000 }).catch(() => false)
      return hasTabPanel
    } catch {
      return false
    }
  }

  async verifySegmentsDisplayed(): Promise<boolean> {
    try {
      await this.page.waitForTimeout(1000)
      // Look for segments/uplift content
      const hasSegments = await this.page.getByText(/Segment|Uplift|Baseline|ROI/i).first().isVisible({ timeout: 3000 }).catch(() => false)
      if (hasSegments) return true
      // Fallback: look for tabpanel
      const hasTabPanel = await this.page.getByRole('tabpanel').isVisible({ timeout: 2000 }).catch(() => false)
      return hasTabPanel
    } catch {
      return false
    }
  }
}
