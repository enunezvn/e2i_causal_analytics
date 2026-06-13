import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the Analytics page (`/analytics`).
 *
 * Agent-performance & query-analytics dashboard. Honest states:
 *  - loading spinner while the dashboard loads
 *  - error: "Failed to load analytics" with the error message
 *  - loaded: KPI cards (Total Queries, Avg Latency, ...) from
 *    GET /api/analytics/dashboard
 */
export class AnalyticsPage extends BasePage {
  readonly url = '/analytics'
  readonly pageTitle = /Analytics|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /Agent Analytics/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Agent Analytics/i }).first()
  }

  get totalQueriesCard(): Locator {
    return this.page.getByText('Total Queries', { exact: true }).first()
  }

  get avgLatencyCard(): Locator {
    return this.page.getByText('Avg Latency', { exact: true }).first()
  }

  get errorState(): Locator {
    return this.page.getByText('Failed to load analytics', { exact: true }).first()
  }
}
