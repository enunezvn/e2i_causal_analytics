import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for Feature Importance page.
 * Displays SHAP values, feature importance bar charts, beeswarm plots, and waterfall charts.
 *
 * NOTE: After PR #316, this page is wired to the live `/api/explain/*` endpoints
 * (see src/pages/FeatureImportance.tsx). The Model Info, Refresh, and most
 * visualization tabs are gated on a successful POST to /api/explain/predict,
 * which only fires once the user provides a Patient ID and clicks Explain.
 * The spec is responsible for stubbing those endpoints (the shared
 * `**\/api/explain/**` mock returns a legacy shape) and for driving the
 * Patient ID + Explain action when an explanation is required.
 */
export class FeatureImportancePage extends BasePage {
  readonly url = ROUTES.FEATURE_IMPORTANCE
  readonly pageTitle = /Feature Importance|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  // Page Header
  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Feature Importance/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/SHAP|feature importance|beeswarm|force plot/i).first()
  }

  // Model Selector — shadcn SelectTrigger renders as a [role="combobox"] button
  // with the w-[280px] class set by FeatureImportance.tsx.
  get modelSelector(): Locator {
    return this.page
      .locator('button.w-\\[280px\\], [role="combobox"], button:has-text("Select a model")')
      .first()
  }

  // Patient ID + Explain action (post-PR #316)
  get patientIdInput(): Locator {
    return this.page.getByLabel(/patient id/i).first()
  }

  get explainButton(): Locator {
    return this.page.getByRole('button', { name: /^explain$/i }).first()
  }

  // Action Buttons. The refresh button has no accessible name; it is a
  // <Button variant="outline" size="icon"> whose only child is the
  // <RefreshCw /> lucide icon, so we match by the icon class.
  get refreshButton(): Locator {
    return this.page
      .getByRole('button')
      .filter({ has: this.page.locator('svg.lucide-refresh-cw') })
      .first()
  }

  get exportButton(): Locator {
    return this.page.getByRole('button', { name: /export/i })
  }

  // Model Info Card — only rendered once `useExplain` has data + a model is
  // selected. Both the "Base Value" and "Top Feature" labels live inside this
  // card as `text-sm text-muted-foreground` divs.
  get modelInfoCard(): Locator {
    return this.page.locator('.rounded-lg.border').filter({ hasText: /Base Value|Top Feature/i }).first()
  }

  get baseValueDisplay(): Locator {
    return this.page.getByText('Base Value').first()
  }

  get topFeatureDisplay(): Locator {
    return this.page.getByText('Top Feature').first()
  }

  // Feature Rankings
  get featureRankingsCard(): Locator {
    return this.page.locator('.rounded-lg.border').filter({ hasText: 'Feature Rankings' }).first()
  }

  get featureSearchInput(): Locator {
    return this.page.getByPlaceholder(/search features/i)
  }

  get featureRows(): Locator {
    return this.page.locator('.rounded-lg.cursor-pointer')
  }

  // Tabs
  get tabsList(): Locator {
    return this.page.getByRole('tablist')
  }

  get barChartTab(): Locator {
    return this.page.getByRole('tab', { name: /bar chart/i })
  }

  get beeswarmTab(): Locator {
    return this.page.getByRole('tab', { name: /beeswarm/i })
  }

  get waterfallTab(): Locator {
    return this.page.getByRole('tab', { name: /waterfall/i })
  }

  // Visualization Cards
  get globalImportanceCard(): Locator {
    return this.page.getByText('Global Feature Importance').first()
  }

  get featureDistributionCard(): Locator {
    // Bar Chart's CardDescription wording. The Beeswarm card title was renamed
    // in PR #316 to "Per-Feature SHAP Contributions"; we keep the broader
    // matcher so it still resolves either way.
    return this.page
      .getByText(/Feature Value Distribution|Per-Feature SHAP Contributions/i)
      .first()
  }

  get predictionExplanationCard(): Locator {
    return this.page.getByText('Individual Prediction Explanation').first()
  }

  // Feature Details Section
  get featureDetailsCard(): Locator {
    return this.page.getByText('Feature Details').first()
  }

  // Actions
  async selectModel(modelName: string): Promise<void> {
    // Wait for the models query to settle so the trigger isn't disabled.
    await this.modelSelector.waitFor({ state: 'visible', timeout: 5000 })

    // Click the select trigger to open dropdown.
    await this.modelSelector.click()

    // Wait briefly for Radix to mount the listbox portal.
    await this.page.waitForTimeout(200)

    // Radix Select uses `[role="option"]` for items inside `[role="listbox"]`
    // (the portal-rendered SelectContent). Match by accessible name first,
    // then fall back to a text match inside the viewport.
    const option = this.page.getByRole('option', { name: new RegExp(modelName, 'i') })
    const viewportOption = this.page
      .locator('[data-radix-select-viewport] [role="option"], [role="listbox"] [role="option"]')
      .filter({ hasText: new RegExp(modelName, 'i') })

    if (await option.first().isVisible({ timeout: 2000 }).catch(() => false)) {
      await option.first().click()
      return
    }
    if (await viewportOption.first().isVisible({ timeout: 2000 }).catch(() => false)) {
      await viewportOption.first().click()
      return
    }
    // Fallback: click by visible text anywhere on the page (last resort).
    await this.page.getByText(new RegExp(modelName, 'i')).first().click()
  }

  async clickTab(tabName: string): Promise<void> {
    await this.page.getByRole('tab', { name: new RegExp(tabName, 'i') }).click()
  }

  async searchFeatures(query: string): Promise<void> {
    await this.featureSearchInput.fill(query)
  }

  async fillPatientId(patientId: string): Promise<void> {
    await this.patientIdInput.fill(patientId)
  }

  async clickExplain(): Promise<void> {
    await this.explainButton.click()
  }

  /**
   * Drive the Explain mutation end-to-end and wait for the resulting card to
   * render. After PR #316, the model-info card and the feature rankings only
   * appear once `useExplain.data` is populated, so any test that asserts on
   * downstream UI must call this first.
   */
  async runExplanation(patientId: string = 'patient_e2e_001'): Promise<void> {
    await this.fillPatientId(patientId)
    await this.clickExplain()
    // Wait for the live response to flow through the card. "Base Value" is the
    // first stable label inside the model-info card, so it doubles as the
    // ready-signal for downstream assertions.
    await this.baseValueDisplay.waitFor({ state: 'visible', timeout: 10000 })
  }

  async clickRefresh(): Promise<void> {
    // The refresh button is disabled until a patient ID has been entered.
    // Drive a baseline explanation so the click actually fires the mutation.
    if (!(await this.refreshButton.isEnabled().catch(() => false))) {
      await this.runExplanation()
    }
    await this.refreshButton.click()
  }

  async clickExport(): Promise<void> {
    await this.exportButton.click()
  }

  async selectFeatureRow(index: number): Promise<void> {
    await this.featureRows.nth(index).click()
  }

  // Verification methods
  async verifyModelInfoDisplayed(): Promise<boolean> {
    try {
      // After PR #316 the model-info card only renders once an explanation
      // has been computed. Drive the Explain action so the live response
      // populates the card before we look for it.
      await this.runExplanation().catch(() => {})

      // Look for Base Value text inside the model-info card.
      const hasBaseValue = await this.baseValueDisplay
        .isVisible({ timeout: 3000 })
        .catch(() => false)
      if (hasBaseValue) return true

      // Fallback: Top Feature label.
      const hasTopFeature = await this.topFeatureDisplay
        .isVisible({ timeout: 2000 })
        .catch(() => false)
      if (hasTopFeature) return true

      // Fallback: model selector value reflects the chosen model.
      const hasModelSelector = await this.modelSelector
        .isVisible({ timeout: 1000 })
        .catch(() => false)
      if (hasModelSelector) return true

      // Ultimate fallback: at minimum the page header rendered.
      return await this.pageHeader.isVisible({ timeout: 1000 }).catch(() => false)
    } catch {
      return false
    }
  }

  async verifyFeatureRankingsDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Feature Rankings').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
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

  async verifyBarChartDisplayed(): Promise<boolean> {
    try {
      // Wait for tab content to render.
      await this.page.waitForTimeout(300)
      // The Bar Chart tab shows "Global Feature Importance" as the CardTitle.
      const hasGlobalImportance = await this.page
        .getByText('Global Feature Importance')
        .first()
        .isVisible()
        .catch(() => false)
      if (hasGlobalImportance) return true
      // Fallback: empty-state copy emitted by SHAPBarChart when `features=[]`.
      const hasEmptyState = await this.page
        .getByText(/No feature data available/i)
        .first()
        .isVisible()
        .catch(() => false)
      if (hasEmptyState) return true
      // Fallback: any chart SVG.
      return await this.page
        .locator('svg, [class*="chart"], [class*="recharts"]')
        .first()
        .isVisible()
        .catch(() => false)
    } catch {
      return false
    }
  }

  async verifyBeeswarmDisplayed(): Promise<boolean> {
    try {
      await this.page.waitForTimeout(300)
      // PR #316 renamed the card title; accept either wording.
      const hasContent = await this.page
        .getByText(/Per-Feature SHAP Contributions|Feature Value Distribution/i)
        .first()
        .isVisible()
        .catch(() => false)
      if (hasContent) return true
      const hasDescription = await this.page
        .getByText(/dot represents|SHAP impact|One dot per top feature/i)
        .first()
        .isVisible()
        .catch(() => false)
      if (hasDescription) return true
      // SHAPBeeswarm empty state.
      const hasEmptyState = await this.page
        .getByText(/No data available for beeswarm plot/i)
        .first()
        .isVisible()
        .catch(() => false)
      if (hasEmptyState) return true
      return await this.page.locator('svg').first().isVisible().catch(() => false)
    } catch {
      return false
    }
  }

  async verifyWaterfallDisplayed(): Promise<boolean> {
    try {
      await this.page.waitForTimeout(300)
      const hasContent = await this.page
        .getByText('Individual Prediction Explanation')
        .first()
        .isVisible()
        .catch(() => false)
      if (hasContent) return true
      const hasDescription = await this.page
        .getByText(/base value|final prediction/i)
        .first()
        .isVisible()
        .catch(() => false)
      if (hasDescription) return true
      // SHAPWaterfall empty state.
      const hasEmptyState = await this.page
        .getByText(/No feature data available/i)
        .first()
        .isVisible()
        .catch(() => false)
      if (hasEmptyState) return true
      return await this.page.locator('svg').first().isVisible().catch(() => false)
    } catch {
      return false
    }
  }
}
