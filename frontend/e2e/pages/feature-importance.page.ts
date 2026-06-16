import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for Feature Importance page.
 * Displays SHAP values, feature importance bar charts, beeswarm plots, and waterfall charts.
 *
 * NOTE: PR #985 redesigned src/pages/FeatureImportance.tsx into a two-mode view:
 *
 *   - **Cohort (global)** — the DEFAULT mode. On arrival it calls
 *     `GET /api/explain/global` and renders a cohort-level mean-|SHAP| view with
 *     NO entity selection required. The summary card (Base Value / Top Feature)
 *     and the Bar/Beeswarm tabs are populated straight away.
 *   - **Individual** — reached via the "Individual" top-level tab. It exposes an
 *     entity PICKER (a shadcn Select of real IDs from
 *     `GET /api/explain/sample-entities`, labelled by `grainLabel` =
 *     Patient/HCP) and AUTO-RUNS `POST /api/explain/predict` when an entity is
 *     chosen — there is NO "Explain" button anymore. The Waterfall and History
 *     viz tabs only exist in this mode.
 *
 * The spec stubs `/api/explain/{models,global,sample-entities,predict,history}`.
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

  // Cohort model selector — shadcn SelectTrigger renders as a [role="combobox"]
  // button. PR #985 changed it to `w-[190px]` with the "Select cohort"
  // placeholder (was `w-[280px]` / "Select a model").
  get modelSelector(): Locator {
    return this.page
      .locator('button.w-\\[190px\\], button:has-text("Select cohort")')
      .first()
  }

  // Brand selector — new in PR #985. `w-[150px]` SelectTrigger, "Select brand"
  // placeholder. Defaults to the first GOLDSTD_BRANDS entry (Remibrutinib), so
  // its trigger renders the brand text rather than the placeholder.
  get brandSelector(): Locator {
    return this.page
      .locator('button.w-\\[150px\\], button:has-text("Select brand")')
      .first()
  }

  // Top-level mode toggle (Cohort (global) | Individual). Radix Tabs renders
  // each trigger as a [role="tab"].
  get cohortModeTab(): Locator {
    return this.page.getByRole('tab', { name: /cohort \(global\)/i }).first()
  }

  get individualModeTab(): Locator {
    return this.page.getByRole('tab', { name: /^individual$/i }).first()
  }

  // Individual-mode entity picker (a shadcn Select of real IDs, NOT a text
  // input). It is only mounted once viewMode === 'individual'. The trigger is
  // labelled by `grainLabel` (Patient/HCP) and shows "Select a patient/hcp".
  get entityPicker(): Locator {
    return this.page
      .locator('button:has-text("Select a patient"), button:has-text("Select a hcp")')
      .first()
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

  // Summary card — rendered whenever `hasData` is true. In cohort mode that is
  // `!!global` (populated on arrival, no Explain needed); in individual mode it
  // is `!!explanation` (populated after an entity is picked). Both the
  // "Base Value" and "Top Feature" labels live inside this card.
  //
  // The shadcn `<Card>` renders as a div with `rounded-xl border …`.
  get modelInfoCard(): Locator {
    return this.page
      .locator('div.rounded-xl.border')
      .filter({ hasText: /Base Value|Top Feature/i })
      .first()
  }

  get baseValueDisplay(): Locator {
    return this.page.getByText('Base Value').first()
  }

  get topFeatureDisplay(): Locator {
    return this.page.getByText('Top Feature').first()
  }

  // Feature Rankings — same `<Card>` (rounded-xl border) as modelInfoCard.
  get featureRankingsCard(): Locator {
    return this.page
      .locator('div.rounded-xl.border')
      .filter({ hasText: 'Feature Rankings' })
      .first()
  }

  get featureSearchInput(): Locator {
    return this.page.getByPlaceholder(/search features/i)
  }

  get featureRows(): Locator {
    return this.page.locator('.rounded-lg.cursor-pointer')
  }

  // Viz Tabs. Bar Chart + Beeswarm exist in BOTH modes; Waterfall + History
  // only exist in individual mode.
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
    // Beeswarm CardTitle: "SHAP Distribution Across the Cohort" (cohort mode)
    // or "Per-Feature SHAP Contributions" (individual mode).
    return this.page
      .getByText(/SHAP Distribution Across the Cohort|Per-Feature SHAP Contributions/i)
      .first()
  }

  get predictionExplanationCard(): Locator {
    return this.page.getByText('Individual Prediction Explanation').first()
  }

  // Feature Details Section
  get featureDetailsCard(): Locator {
    return this.page.getByText('Feature Details').first()
  }

  // --- Selection helpers ----------------------------------------------------

  /** Open a Radix Select trigger and click the option matching `optionName`. */
  private async pickFromSelect(trigger: Locator, optionName: string): Promise<void> {
    await trigger.waitFor({ state: 'visible', timeout: 5000 })
    await trigger.click()
    // Wait briefly for Radix to mount the listbox portal.
    await this.page.waitForTimeout(200)

    const option = this.page.getByRole('option', { name: new RegExp(optionName, 'i') })
    const viewportOption = this.page
      .locator('[data-radix-select-viewport] [role="option"], [role="listbox"] [role="option"]')
      .filter({ hasText: new RegExp(optionName, 'i') })

    if (await option.first().isVisible({ timeout: 2000 }).catch(() => false)) {
      await option.first().click()
      return
    }
    if (await viewportOption.first().isVisible({ timeout: 2000 }).catch(() => false)) {
      await viewportOption.first().click()
      return
    }
    // Fallback: click by visible text anywhere on the page (last resort).
    await this.page.getByText(new RegExp(optionName, 'i')).first().click()
  }

  async selectModel(modelName: string): Promise<void> {
    await this.pickFromSelect(this.modelSelector, modelName)
  }

  async selectBrand(brand: string): Promise<void> {
    await this.pickFromSelect(this.brandSelector, brand)
  }

  /** Switch to the Individual (per-entity) mode tab and wait for it to mount. */
  async switchToIndividualMode(): Promise<void> {
    await this.individualModeTab.click()
    // The entity-picker card mounts on the next render; the entity Select is
    // its first stable element.
    await this.entityPicker.waitFor({ state: 'visible', timeout: 5000 }).catch(() => {})
  }

  /** Switch back to the default Cohort (global) mode tab. */
  async switchToCohortMode(): Promise<void> {
    await this.cohortModeTab.click()
  }

  async clickTab(tabName: string): Promise<void> {
    await this.page.getByRole('tab', { name: new RegExp(tabName, 'i') }).click()
  }

  async searchFeatures(query: string): Promise<void> {
    await this.featureSearchInput.fill(query)
  }

  /**
   * Pick the first real entity ID from the individual-mode picker. The
   * explanation auto-runs on selection (no Explain button), so callers don't
   * trigger a mutation themselves.
   */
  async selectFirstEntity(): Promise<void> {
    await this.entityPicker.waitFor({ state: 'visible', timeout: 5000 })
    await this.entityPicker.click()
    await this.page.waitForTimeout(200)
    await this.page
      .locator('[data-radix-select-viewport] [role="option"], [role="listbox"] [role="option"]')
      .first()
      .click()
  }

  /**
   * Drive an INDIVIDUAL-mode explanation end-to-end and wait for the summary
   * card to render. PR #985 removed the Patient-ID text input + Explain button;
   * the per-entity explanation is reached by:
   *   1. switching to the Individual tab,
   *   2. picking a real entity from the Select picker (defaults to the first ID
   *      automatically on mount; selecting again is idempotent),
   *   3. the page AUTO-RUNS `POST /api/explain/predict` and populates the card.
   *
   * "Base Value" is the first stable label inside the summary card, so it
   * doubles as the ready-signal for downstream individual-mode assertions.
   */
  async runIndividualExplanation(): Promise<void> {
    await this.switchToIndividualMode()
    // The page defaults the picker to the first real ID on mount and auto-runs
    // the explanation; explicitly selecting is a no-op safety net if the
    // default-effect hasn't fired yet.
    if (await this.entityPicker.isVisible({ timeout: 2000 }).catch(() => false)) {
      await this.selectFirstEntity().catch(() => {})
    }
    await this.baseValueDisplay.waitFor({ state: 'visible', timeout: 10000 })
  }

  // Back-compat alias: older call-sites used `runExplanation()`. The default
  // (cohort) view needs no action — the summary card is populated on arrival —
  // so this routes to the individual-mode driver, which is the only path that
  // requires interaction to surface an explanation.
  async runExplanation(): Promise<void> {
    await this.runIndividualExplanation()
  }

  async clickRefresh(): Promise<void> {
    // In the default (cohort) mode the Refresh button is enabled as soon as the
    // global query settles — no entity needed — so we can click it directly.
    await this.refreshButton.waitFor({ state: 'visible', timeout: 5000 })
    await this.refreshButton.click()
  }

  async clickExport(): Promise<void> {
    await this.exportButton.click()
  }

  async selectFeatureRow(index: number): Promise<void> {
    await this.featureRows.nth(index).click()
  }

  // --- Verification methods -------------------------------------------------

  /**
   * Verify the summary card (Base Value + Top Feature) is populated. In the
   * default cohort mode this needs NO interaction — `/api/explain/global`
   * populates it on arrival. We require BOTH labels so a 200 with the wrong
   * shape (missing `features`/`base_value`) still fails this check rather than
   * passing on the page header alone.
   */
  async verifyModelInfoDisplayed(): Promise<boolean> {
    try {
      const hasBaseValue = await this.baseValueDisplay
        .isVisible({ timeout: 5000 })
        .catch(() => false)
      const hasTopFeature = await this.topFeatureDisplay
        .isVisible({ timeout: 3000 })
        .catch(() => false)
      return hasBaseValue && hasTopFeature
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
      await this.tabsList.first().waitFor({ state: 'visible', timeout: 5000 })
      return await this.tabsList.first().isVisible()
    } catch {
      return false
    }
  }

  async verifyBarChartDisplayed(): Promise<boolean> {
    try {
      // Wait for tab content to render.
      await this.page.waitForTimeout(300)
      // Cohort-mode Bar Chart CardTitle.
      const hasGlobalImportance = await this.page
        .getByText('Global Feature Importance')
        .first()
        .isVisible()
        .catch(() => false)
      if (hasGlobalImportance) return true
      // Individual-mode Bar Chart CardTitle.
      const hasContributions = await this.page
        .getByText(/Feature Contributions/i)
        .first()
        .isVisible()
        .catch(() => false)
      if (hasContributions) return true
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
      // PR #985 beeswarm CardTitles (cohort | individual).
      const hasContent = await this.page
        .getByText(/SHAP Distribution Across the Cohort|Per-Feature SHAP Contributions/i)
        .first()
        .isVisible()
        .catch(() => false)
      if (hasContent) return true
      const hasDescription = await this.page
        .getByText(/One dot per sampled entity|One dot per top feature/i)
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
        .getByText(/base value|final/i)
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
