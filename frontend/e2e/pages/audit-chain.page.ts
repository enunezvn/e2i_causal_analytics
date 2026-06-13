import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the Audit Chain page (`/audit-chain`).
 *
 * Cryptographic audit-trail / workflow-verification dashboard. Honest states:
 *  - loading spinner while workflows load
 *  - error: EmptyState "Failed to load workflows"
 *  - empty: EmptyState "No workflows found"
 *  - loaded: workflow rows from GET /api/audit/recent
 */
export class AuditChainPage extends BasePage {
  readonly url = '/audit-chain'
  readonly pageTitle = /Audit Chain|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /Audit Chain/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Audit Chain/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/Cryptographic audit trail/i).first()
  }

  get emptyState(): Locator {
    return this.page.getByText('No workflows found', { exact: true }).first()
  }

  get errorState(): Locator {
    return this.page.getByText('Failed to load workflows', { exact: true }).first()
  }
}
