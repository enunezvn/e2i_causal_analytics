import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for Predictive Analytics page.
 *
 * Wave PR #319 (issue #300) rewired this page from a synthetic
 * risk-score / uplift / KPI dashboard to a live-data form backed by
 * /api/models/predict/{model_name}. The current UI is a model selector
 * + Input Features form + Prediction Result card — there are no longer
 * tabs, KPI cards, or refresh button.
 *
 * Selectors below target the new UI shape. Tests mock
 * /api/models/status (drives the selector) and /api/models/{name}/info
 * (drives the feature inputs) inline; see specs/predictive-analytics.spec.ts.
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
      .getByText(/Run live predictions|deployed models|feature contributions/i)
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
  // Input Features card
  //
  // CardTitle in shadcn renders as a plain <div>, so getByRole('heading')
  // does NOT match. We anchor on the exact text content.
  // ============================================================================
  get inputFeaturesHeading(): Locator {
    return this.page.getByText('Input Features', { exact: true }).first()
  }

  get runPredictionButton(): Locator {
    return this.page.getByRole('button', { name: /Run Prediction|Running/i }).first()
  }

  // ============================================================================
  // Prediction Result card
  // ============================================================================
  get predictionResultHeading(): Locator {
    return this.page.getByText('Prediction Result', { exact: true }).first()
  }

  get predictionResultPlaceholder(): Locator {
    return this.page.getByText(/Submit features above to run a prediction/i).first()
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

  async verifyInputFeaturesCard(): Promise<boolean> {
    return this.inputFeaturesHeading
      .waitFor({ state: 'visible', timeout: 5000 })
      .then(() => true)
      .catch(() => false)
  }

  async verifyPredictionResultCard(): Promise<boolean> {
    return this.predictionResultHeading
      .waitFor({ state: 'visible', timeout: 5000 })
      .then(() => true)
      .catch(() => false)
  }
}
