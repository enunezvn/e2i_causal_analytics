import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the Gap Analysis page (`/gap-analysis`).
 *
 * ROI opportunity-detection dashboard (Tier-2 Gap Analyzer). F-002 removed the
 * SAMPLE_OPPORTUNITIES fallback — data comes strictly from the API. Honest
 * states this POM exposes:
 *  - KPI overview cards (Total Addressable, Opportunities, ...) from real data
 *  - empty: EmptyState "No gap opportunities available" when the list is empty
 */
export class GapAnalysisPage extends BasePage {
  readonly url = '/gap-analysis'
  readonly pageTitle = /Gap Analysis|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /Gap Analysis/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Gap Analysis/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/ROI opportunity detection/i).first()
  }

  get totalAddressableCard(): Locator {
    return this.page.getByText('Total Addressable', { exact: true }).first()
  }

  get opportunitiesCard(): Locator {
    return this.page.getByText('Opportunities', { exact: true }).first()
  }

  // Honest empty state when no opportunities are returned.
  get emptyState(): Locator {
    return this.page.getByText('No gap opportunities available', { exact: true }).first()
  }

  get runAnalysisButton(): Locator {
    return this.page.getByRole('button', { name: /Run Analysis/i }).first()
  }
}
