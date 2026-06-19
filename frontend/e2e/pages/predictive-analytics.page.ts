import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for Predictive Analytics page.
 *
 * The page is DATA-DRIVEN: a model selector + a "Score holdout cohort" action
 * that produces a Ranked Targets card (provenance + distribution + ranked
 * entities) and a Prediction Detail card (drill-down SHAP + Advanced what-if).
 * There are no tabs, KPI cards, or refresh button, and the old always-on
 * "Input Features" form is now behind an Advanced what-if toggle.
 *
 * Selectors below target the new UI shape. Tests mock
 * /api/models/status (drives the selector) and /api/models/{name}/info
 * (drives the what-if form) inline; see specs/predictive-analytics.spec.ts.
 */
export class PredictiveAnalyticsPage extends BasePage {
  readonly url = ROUTES.PREDICTIVE_ANALYTICS
  readonly pageTitle = /Predictive Analytics|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  // ============================================================================
  // Page Header
  // ============================================================================
  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Predictive Analytics/i }).first()
  }

  get pageDescription(): Locator {
    return this.page
      .getByText(/rank the top targets|holdout cohort|feature contributions/i)
      .first()
  }

  // ============================================================================
  // Model Selector (rendered only when models.length > 0)
  // ============================================================================
  get modelSelector(): Locator {
    // SelectTrigger renders an element with role="combobox" and the page sets
    // aria-label="Model" on it. Anchor on aria-label so we don't accidentally
    // pick up other comboboxes on the page (e.g. brand selector).
    return this.page.getByRole('combobox', { name: /Model/i }).first()
  }

  // ============================================================================
  // Active Model Card (rendered when selectedModel is set)
  // ============================================================================
  get activeModelLabel(): Locator {
    return this.page.getByText('Active Model').first()
  }

  // ============================================================================
  // Ranked Targets card (primary cohort-scoring surface)
  //
  // CardTitle in shadcn renders as a plain <div>, so getByRole('heading')
  // does NOT match. We anchor on the text content.
  // ============================================================================
  get rankedTargetsHeading(): Locator {
    return this.page.getByText(/Ranked Targets/i).first()
  }

  get scoreCohortButton(): Locator {
    return this.page.getByRole('button', { name: /Score holdout cohort|Scoring/i }).first()
  }

  // ============================================================================
  // Prediction Detail card (drill-down + Advanced what-if)
  // ============================================================================
  get predictionDetailHeading(): Locator {
    return this.page.getByText(/Prediction Detail/i).first()
  }

  get predictionDetailPlaceholder(): Locator {
    return this.page.getByText(/Click a ranked entity/i).first()
  }

  // ============================================================================
  // Actions
  // ============================================================================

  /**
   * Open the model selector and pick an option matching `modelName`.
   * The dropdown renders inside a Radix portal, so we try the standard
   * role-based queries first and fall back to the Radix viewport selector.
   */
  async selectModel(modelName: string): Promise<void> {
    await this.modelSelector.waitFor({ state: 'visible', timeout: 5000 })
    await this.modelSelector.click()
    await this.page.waitForTimeout(200)

    const byRole = this.page.getByRole('option', { name: new RegExp(modelName, 'i') })
    const byViewport = this.page
      .locator('[data-radix-select-viewport] [role="option"]')
      .filter({ hasText: new RegExp(modelName, 'i') })
    const byListbox = this.page
      .locator('[role="listbox"] [role="option"]')
      .filter({ hasText: new RegExp(modelName, 'i') })

    if (await byRole.first().isVisible({ timeout: 2000 }).catch(() => false)) {
      await byRole.first().click()
    } else if (
      await byViewport.first().isVisible({ timeout: 1000 }).catch(() => false)
    ) {
      await byViewport.first().click()
    } else if (
      await byListbox.first().isVisible({ timeout: 1000 }).catch(() => false)
    ) {
      await byListbox.first().click()
    } else {
      // Last-ditch: a div with the option text inside the open dropdown.
      await this.page.getByText(new RegExp(`^${modelName}$`, 'i')).first().click()
    }
  }

  // ============================================================================
  // Verification helpers
  // ============================================================================

  async verifyActiveModelCard(): Promise<boolean> {
    return this.activeModelLabel
      .waitFor({ state: 'visible', timeout: 5000 })
      .then(() => true)
      .catch(() => false)
  }

  async verifyRankedTargetsCard(): Promise<boolean> {
    return this.rankedTargetsHeading
      .waitFor({ state: 'visible', timeout: 5000 })
      .then(() => true)
      .catch(() => false)
  }

  async verifyPredictionDetailCard(): Promise<boolean> {
    return this.predictionDetailHeading
      .waitFor({ state: 'visible', timeout: 5000 })
      .then(() => true)
      .catch(() => false)
  }
}
