import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the RETIRED Intervention Impact route.
 *
 * `/intervention-impact` was retired (T10): its unique Treatment Effects view
 * moved to the agent-led `/causal-analysis` page; its other tabs duplicated
 * `/causal-analysis` (History), the Segment Analysis page, and `/digital-twin`.
 * The old route is now a client-side redirect. This POM only asserts the
 * redirect; Treatment Effects behavior is covered by causal-analysis.spec.ts.
 */
export class InterventionImpactPage extends BasePage {
  readonly url = ROUTES.INTERVENTION_IMPACT
  readonly pageTitle = /Causal Analysis|E2I/i

  constructor(page: Page) {
    super(page)
  }

  // After the redirect, the unified Causal Analysis header renders.
  get redirectedHeader(): Locator {
    return this.page.getByRole('heading', { name: /Causal Analysis/i }).first()
  }
}
