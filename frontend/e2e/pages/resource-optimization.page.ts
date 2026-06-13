import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the Resource Optimization page (`/resource-optimization`).
 *
 * Mathematical-optimization dashboard (Tier-4 Resource Optimizer / scipy).
 * Honest states: "Solver Ready"/"Solver Unavailable" badge from
 * GET /api/resources/health, and EmptyState "Run an optimization to see
 * results" before any optimization is run.
 */
export class ResourceOptimizationPage extends BasePage {
  readonly url = '/resource-optimization'
  readonly pageTitle = /Resource Optimization|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /Resource Optimization/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Resource Optimization/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/Mathematical optimization for budget/i).first()
  }

  get solverReadyBadge(): Locator {
    return this.page.getByText('Solver Ready', { exact: true }).first()
  }

  get solverUnavailableBadge(): Locator {
    return this.page.getByText('Solver Unavailable', { exact: true }).first()
  }

  get emptyState(): Locator {
    return this.page.getByText('Run an optimization to see results', { exact: true }).first()
  }
}
