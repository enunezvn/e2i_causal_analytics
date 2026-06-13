import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the Experiments page (`/experiments`).
 *
 * A/B testing & experiment-monitoring dashboard. Experiments are derived from
 * live monitor data ONLY — no sample fallback. Honest states this POM exposes:
 *  - no monitor data yet -> EmptyState "No experiments to display"
 *  - after "Run Monitoring" -> experiment cards from the real monitor response
 */
export class ExperimentsPage extends BasePage {
  readonly url = '/experiments'
  readonly pageTitle = /Experiments|A\/B Testing|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /A\/B Testing & Experiments/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /A\/B Testing & Experiments/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/Monitor experiment health, enrollment/i).first()
  }

  get activeExperimentsCard(): Locator {
    return this.page.getByText('Active Experiments', { exact: true }).first()
  }

  // Honest empty state before any monitoring run.
  get emptyState(): Locator {
    return this.page.getByText('No experiments to display', { exact: true }).first()
  }

  get runMonitoringButton(): Locator {
    return this.page.getByRole('button', { name: /Run Monitoring/i }).first()
  }

  get experimentsTab(): Locator {
    return this.page.getByRole('tab', { name: /^Experiments$/i }).first()
  }

  async clickRunMonitoring(): Promise<void> {
    await this.runMonitoringButton.click()
  }
}
