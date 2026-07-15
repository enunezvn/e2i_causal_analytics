import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the Feedback Learning page (`/feedback-learning`).
 *
 * Tier-5 self-improvement dashboard (pattern detection + knowledge updates).
 * Honest states:
 *  - agent status "Online"/"Offline" from GET /api/feedback/health
 *  - "Knowledge Updates" tab -> "No knowledge updates proposed" when empty
 */
export class FeedbackLearningPage extends BasePage {
  readonly url = '/feedback-learning'
  readonly pageTitle = /Feedback Learning|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /Feedback Learning/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Feedback Learning/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/Tier 5 self-improvement system/i).first()
  }

  get onlineStatus(): Locator {
    return this.page.getByText('Online', { exact: true }).first()
  }

  get offlineStatus(): Locator {
    return this.page.getByText('Offline', { exact: true }).first()
  }

  get updatesTab(): Locator {
    return this.page.getByRole('tab', { name: /Knowledge Updates/i }).first()
  }

  get noUpdatesEmptyState(): Locator {
    return this.page.getByText('No knowledge updates proposed', { exact: true }).first()
  }

  async openUpdatesTab(): Promise<void> {
    await this.updatesTab.click()
  }
}
