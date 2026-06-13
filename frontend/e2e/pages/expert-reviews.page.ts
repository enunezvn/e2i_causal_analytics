import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the Expert Reviews page (`/expert-reviews`).
 *
 * Human-in-the-loop review queue for REVIEW-band causal DAGs. The page
 * documents its honest states explicitly (loading spinner, error banner,
 * EmptyState — no hardcoded fallback). This POM exposes:
 *  - empty: EmptyState "No pending reviews" (GET /api/expert-reviews/pending []]
 *  - error: WarningBanner "Failed to load pending reviews"
 *  - loaded: a pending-review row (Brand/Treatment/Outcome table)
 */
export class ExpertReviewsPage extends BasePage {
  readonly url = '/expert-reviews'
  readonly pageTitle = /Expert Reviews|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /Expert Reviews/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Expert Reviews/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/Human-in-the-loop validation queue/i).first()
  }

  get emptyState(): Locator {
    return this.page.getByText('No pending reviews', { exact: true }).first()
  }

  get errorState(): Locator {
    return this.page.getByText('Failed to load pending reviews', { exact: true }).first()
  }
}
